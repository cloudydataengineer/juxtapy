from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from juxtapy.exceptions import JoinKeyError

_LEFT_SUFFIX = "__juxtapy_left"
_RIGHT_SUFFIX = "__juxtapy_right"
_INDICATOR_COL = "__juxtapy_merge"


def _validate_keys(df: pd.DataFrame, keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise JoinKeyError(f"Join column(s) not found in dataframe: {missing}")


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

    def compare_columns(self, columns: Sequence[str]) -> dict[str, tuple[int, int]]:
        both = self._both_rows()
        results: dict[str, tuple[int, int]] = {}
        for column in columns:
            left, right = self._column_pair(column)
            left_values, right_values = both[left], both[right]
            both_null = left_values.isna() & right_values.isna()
            match = (left_values == right_values) | both_null
            match_count = int(match.sum())
            mismatch_count = int((~match).sum())
            results[column] = (match_count, mismatch_count)
        return results

    def sample_mismatched_rows(self, column: str, keys: Sequence[str], n: int) -> pd.DataFrame:
        both = self._both_rows()
        left, right = self._column_pair(column)
        left_values, right_values = both[left], both[right]
        both_null = left_values.isna() & right_values.isna()
        mismatch_mask = ~((left_values == right_values) | both_null)
        sample = both.loc[mismatch_mask, list(keys) + [left, right]].head(n).copy()
        sample = sample.rename(columns={left: f"{column}_df1", right: f"{column}_df2"})
        return sample.reset_index(drop=True)
