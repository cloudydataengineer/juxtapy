from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from juxtapy.exceptions import JoinKeyError

_LEFT_SUFFIX = "__juxtapy_left"
_RIGHT_SUFFIX = "__juxtapy_right"
_INDICATOR_COL = "__juxtapy_merge"


def _validate_keys(df: pd.DataFrame, keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise JoinKeyError(f"Join column(s) not found in dataframe: {missing}")


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _match_mask(
    left: pd.Series, right: pd.Series, abs_tol: float, rel_tol: float
) -> pd.Series:
    both_null = left.isna() & right.isna()
    match = (left == right) | both_null
    if (abs_tol or rel_tol) and _is_numeric(left) and _is_numeric(right):
        diff = (left - right).abs()
        bound = abs_tol + rel_tol * pd.concat([left.abs(), right.abs()], axis=1).max(axis=1)
        match = match | (diff <= bound)
    return match


class PandasAdapter:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    def row_count(self) -> int:
        return len(self._df)

    def duplicate_key_count(self, keys: Sequence[str]) -> int:
        keys = list(keys)
        _validate_keys(self._df, keys)
        return int(self._df.duplicated(subset=keys).sum())

    def dtype_of(self, column: str) -> str:
        return str(self._df[column].dtype)

    def full_outer_join(self, other: PandasAdapter, keys: Sequence[str]) -> PandasJoinedAdapter:
        keys = list(keys)
        _validate_keys(self._df, keys)
        _validate_keys(other._df, keys)
        merged = pd.merge(
            self._df,
            other._df,
            on=keys,
            how="outer",
            suffixes=(_LEFT_SUFFIX, _RIGHT_SUFFIX),
            indicator=_INDICATOR_COL,
        )
        return PandasJoinedAdapter(merged, keys)


class PandasJoinedAdapter:
    def __init__(self, merged: pd.DataFrame, keys: list[str]) -> None:
        self._merged = merged
        self._keys = keys

    def only_in_left_count(self) -> int:
        return int((self._merged[_INDICATOR_COL] == "left_only").sum())

    def only_in_right_count(self) -> int:
        return int((self._merged[_INDICATOR_COL] == "right_only").sum())

    def common_count(self) -> int:
        return int((self._merged[_INDICATOR_COL] == "both").sum())

    def _column_pair(self, column: str) -> tuple[str, str]:
        left = f"{column}{_LEFT_SUFFIX}"
        right = f"{column}{_RIGHT_SUFFIX}"
        left = left if left in self._merged.columns else column
        right = right if right in self._merged.columns else column
        return left, right

    def _both_rows(self) -> pd.DataFrame:
        return self._merged[self._merged[_INDICATOR_COL] == "both"]

    def compare_columns(
        self,
        columns: Sequence[str],
        tolerances: Mapping[str, tuple[float, float]] | None = None,
    ) -> dict[str, tuple[int, int, int, int]]:
        tolerances = tolerances or {}
        both = self._both_rows()
        results: dict[str, tuple[int, int, int, int]] = {}
        for column in columns:
            left, right = self._column_pair(column)
            left_values, right_values = both[left], both[right]
            abs_tol, rel_tol = tolerances.get(column, (0.0, 0.0))
            match = _match_mask(left_values, right_values, abs_tol, rel_tol)
            match_count = int(match.sum())
            mismatch_count = int((~match).sum())
            null_count_df1 = int(left_values.isna().sum())
            null_count_df2 = int(right_values.isna().sum())
            results[column] = (match_count, mismatch_count, null_count_df1, null_count_df2)
        return results

    def sample_mismatched_rows(
        self,
        column: str,
        keys: Sequence[str],
        n: int,
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
    ) -> pd.DataFrame:
        both = self._both_rows()
        left, right = self._column_pair(column)
        left_values, right_values = both[left], both[right]
        mismatch_mask = ~_match_mask(left_values, right_values, abs_tol, rel_tol)
        sample = both.loc[mismatch_mask, list(keys) + [left, right]].head(n).copy()
        sample = sample.rename(columns={left: f"{column}_df1", right: f"{column}_df2"})
        return sample.reset_index(drop=True)
