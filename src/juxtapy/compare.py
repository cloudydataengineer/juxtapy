from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import pandas as pd

from juxtapy.backends.detect import infer_backend, make_adapter
from juxtapy.exceptions import JoinKeyError, JuxtapyError, MismatchThresholdError
from juxtapy.results import ColumnSummary, CompareReport, RowSummary, SchemaDiff


def _lower_columns(df: Any, backend: str) -> Any:
    if backend == "pandas":
        return df.rename(columns={c: c.lower() for c in df.columns})
    for c in df.columns:
        if c != c.lower():
            df = df.withColumnRenamed(c, c.lower())
    return df


class Compare:
    """Join two tables on a key and compare them column by column.

    Works with either pandas DataFrames or PySpark DataFrames (both inputs
    must use the same backend).
    """

    def __init__(
        self,
        df1: Any,
        df2: Any,
        join_columns: str | Sequence[str],
        df1_name: str = "df1",
        df2_name: str = "df2",
        columns_to_compare: Sequence[str] | None = None,
        ignore_columns: Sequence[str] | None = None,
        cast_column_names_lower: bool = True,
    ) -> None:
        backend1 = infer_backend(df1)
        backend2 = infer_backend(df2)
        if backend1 != backend2:
            raise JuxtapyError(
                f"df1 is a {backend1} dataframe but df2 is a {backend2} dataframe — "
                "both tables must use the same backend."
            )

        if isinstance(join_columns, str):
            join_columns = [join_columns]
        self.join_columns: list[str] = list(join_columns)
        self.df1_name = df1_name
        self.df2_name = df2_name

        if cast_column_names_lower:
            df1 = _lower_columns(df1, backend1)
            df2 = _lower_columns(df2, backend1)
            self.join_columns = [c.lower() for c in self.join_columns]

        self._adapter1 = make_adapter(df1)
        self._adapter2 = make_adapter(df2)

        missing1 = [k for k in self.join_columns if k not in self._adapter1.columns]
        missing2 = [k for k in self.join_columns if k not in self._adapter2.columns]
        if missing1 or missing2:
            raise JoinKeyError(
                f"Join column(s) missing — df1 missing {missing1}, df2 missing {missing2}"
            )

        cols1, cols2 = set(self._adapter1.columns), set(self._adapter2.columns)
        join_set = set(self.join_columns)
        shared: list[str] = sorted((cols1 & cols2) - join_set)

        if columns_to_compare is not None:
            wanted = set(columns_to_compare)
            unknown = wanted - set(shared)
            if unknown:
                raise JuxtapyError(
                    f"columns_to_compare not found as shared, non-key columns: {sorted(unknown)}"
                )
            shared = [c for c in shared if c in wanted]
        if ignore_columns:
            ignore = set(ignore_columns)
            shared = [c for c in shared if c not in ignore]

        self._compared_columns = shared
        self._only_in_df1_cols = sorted(cols1 - cols2 - join_set)
        self._only_in_df2_cols = sorted(cols2 - cols1 - join_set)

        self._joined = self._adapter1.full_outer_join(self._adapter2, self.join_columns)

        self._row_summary: RowSummary | None = None
        self._column_summary: list[ColumnSummary] | None = None
        self._schema_diff: SchemaDiff | None = None

    def row_summary(self) -> RowSummary:
        if self._row_summary is None:
            dup1 = self._adapter1.duplicate_key_count(self.join_columns)
            dup2 = self._adapter2.duplicate_key_count(self.join_columns)
            if dup1 or dup2:
                warnings.warn(
                    f"Duplicate join keys found: {dup1} duplicate row(s) in {self.df1_name}, "
                    f"{dup2} duplicate row(s) in {self.df2_name}. Row-level comparison for "
                    "duplicated keys may be ambiguous.",
                    stacklevel=2,
                )
            self._row_summary = RowSummary(
                rows_df1=self._adapter1.row_count(),
                rows_df2=self._adapter2.row_count(),
                common_rows=self._joined.common_count(),
                only_in_df1=self._joined.only_in_left_count(),
                only_in_df2=self._joined.only_in_right_count(),
                duplicate_keys_df1=dup1,
                duplicate_keys_df2=dup2,
            )
        return self._row_summary

    def column_summary(self) -> list[ColumnSummary]:
        if self._column_summary is None:
            if not self._compared_columns:
                self._column_summary = []
            else:
                counts = self._joined.compare_columns(self._compared_columns)
                summaries = [
                    ColumnSummary(
                        column=col,
                        match_count=counts[col][0],
                        mismatch_count=counts[col][1],
                        dtype1=self._adapter1.dtype_of(col),
                        dtype2=self._adapter2.dtype_of(col),
                    )
                    for col in self._compared_columns
                ]
                summaries.sort(key=lambda s: (-s.mismatch_count, -s.mismatch_pct, s.column))
                self._column_summary = summaries
        return self._column_summary

    def schema_diff(self) -> SchemaDiff:
        if self._schema_diff is None:
            dtype_changes: dict[str, Any] = {}
            for col in self._compared_columns:
                d1, d2 = self._adapter1.dtype_of(col), self._adapter2.dtype_of(col)
                if d1 != d2:
                    dtype_changes[col] = (d1, d2)
            self._schema_diff = SchemaDiff(
                only_in_df1=self._only_in_df1_cols,
                only_in_df2=self._only_in_df2_cols,
                dtype_changes=dtype_changes,
            )
        return self._schema_diff

    def sample_mismatches(self, column: str, n: int = 5) -> pd.DataFrame:
        if column not in self._compared_columns:
            raise JuxtapyError(f"'{column}' is not a shared, comparable column.")
        return self._joined.sample_mismatched_rows(column, self.join_columns, n)

    def matches(self, ignore_extra_columns: bool = False) -> bool:
        rs = self.row_summary()
        if not ignore_extra_columns and not rs.all_rows_match:
            return False
        return all(cs.mismatch_count == 0 for cs in self.column_summary())

    def assert_match(self, threshold: float = 1.0, column: str | None = None) -> None:
        """Raise MismatchThresholdError if the match rate falls below ``threshold``.

        Checks a single column's match rate if ``column`` is given, otherwise the
        overall match rate across all compared columns.
        """
        if column is not None:
            cs = next((c for c in self.column_summary() if c.column == column), None)
            if cs is None:
                raise JuxtapyError(f"'{column}' is not a compared column.")
            rate = cs.match_pct / 100.0
            if rate < threshold:
                raise MismatchThresholdError(
                    f"Column '{column}' match rate {rate:.4f} is below threshold {threshold:.4f}",
                    match_rate=rate,
                    threshold=threshold,
                    column=column,
                )
            return

        summaries = self.column_summary()
        total_match = sum(c.match_count for c in summaries)
        total_compared = sum(c.total_compared for c in summaries)
        rate = 1.0 if total_compared == 0 else total_match / total_compared
        if rate < threshold:
            raise MismatchThresholdError(
                f"Overall match rate {rate:.4f} is below threshold {threshold:.4f}",
                match_rate=rate,
                threshold=threshold,
                column=None,
            )

    def report(self, top_n_columns: int = 5, sample_rows_per_column: int = 5) -> CompareReport:
        columns = self.column_summary()
        worst = [c for c in columns if c.mismatch_count > 0][:top_n_columns]
        samples = {
            c.column: self.sample_mismatches(c.column, n=sample_rows_per_column) for c in worst
        }
        return CompareReport(
            df1_name=self.df1_name,
            df2_name=self.df2_name,
            join_columns=self.join_columns,
            row_summary=self.row_summary(),
            column_summary=columns,
            schema_diff=self.schema_diff(),
            samples=samples,
        )


def compare(
    df1: Any,
    df2: Any,
    join_columns: str | Sequence[str],
    **kwargs: Any,
) -> CompareReport:
    """Functional shortcut: ``Compare(df1, df2, join_columns, **kwargs).report()``."""
    return Compare(df1, df2, join_columns, **kwargs).report()
