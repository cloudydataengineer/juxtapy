from __future__ import annotations

import statistics
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from juxtapy.backends.detect import infer_backend, make_adapter
from juxtapy.exceptions import JoinKeyError, JuxtapyError, MismatchThresholdError
from juxtapy.results import (
    ColumnSummary,
    CompareReport,
    RowSummary,
    SchemaDiff,
    ValidationFailure,
)


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

    By default, a "match" requires exact equality. ``abs_tol``/``rel_tol`` allow
    numeric columns (on both sides) to match within a bound of
    ``abs_tol + rel_tol * max(|df1_val|, |df2_val|)`` — symmetric, so swapping
    df1/df2 gives the same result. ``tolerances={"column": (abs_tol, rel_tol)}``
    overrides the global bar for specific columns. Non-numeric columns always
    require exact equality, regardless of tolerance settings.
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
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        tolerances: Mapping[str, tuple[float, float]] | None = None,
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

        if abs_tol < 0 or rel_tol < 0:
            raise JuxtapyError(f"abs_tol/rel_tol must be >= 0, got abs_tol={abs_tol}, rel_tol={rel_tol}")
        if tolerances:
            unknown_tol_cols = set(tolerances) - set(shared)
            if unknown_tol_cols:
                raise JuxtapyError(f"tolerances key(s) not a compared column: {sorted(unknown_tol_cols)}")
            for col, (a, r) in tolerances.items():
                if a < 0 or r < 0:
                    raise JuxtapyError(f"tolerances[{col!r}] must be >= 0, got abs_tol={a}, rel_tol={r}")
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.tolerances: dict[str, tuple[float, float]] = dict(tolerances) if tolerances else {}
        self._resolved_tolerances: dict[str, tuple[float, float]] = {
            col: self.tolerances.get(col, (abs_tol, rel_tol)) for col in shared
        }

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
                counts = self._joined.compare_columns(
                    self._compared_columns, self._resolved_tolerances
                )
                summaries = [
                    ColumnSummary(
                        column=col,
                        match_count=counts[col][0],
                        mismatch_count=counts[col][1],
                        dtype1=self._adapter1.dtype_of(col),
                        dtype2=self._adapter2.dtype_of(col),
                        null_count_df1=counts[col][2],
                        null_count_df2=counts[col][3],
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
        abs_tol, rel_tol = self._resolved_tolerances[column]
        return self._joined.sample_mismatched_rows(column, self.join_columns, n, abs_tol, rel_tol)

    def matches(self, ignore_extra_columns: bool = False) -> bool:
        rs = self.row_summary()
        if not ignore_extra_columns and not rs.all_rows_match:
            return False
        return all(cs.mismatch_count == 0 for cs in self.column_summary())

    def assert_match(
        self,
        threshold: float | None = 1.0,
        column: str | Sequence[str] | None = None,
    ) -> None:
        """Raise MismatchThresholdError if a match rate falls below ``threshold``.

        ``column`` may be a single column name, a list of column names, or ``None``.
        With a list, every named column is checked and *all* failing columns are
        reported in a single raised error (via ``exc.failures``), not just the first.

        ``threshold`` defaults to ``1.0`` (exact match required), matching ``column=None``
        for the pooled rate across all compared columns' cells, or a single column's
        own rate. Pass ``threshold=None`` explicitly to auto-derive the bar instead, as
        ``mean(rates) - 2 * pstdev(rates)`` across the checked columns' match rates
        (all compared columns, if ``column`` is also ``None``) — this flags columns whose
        match rate is a statistical outlier relative to the others being checked. With a
        single checked column, the population stdev is 0, so that column always passes.

        Raises JuxtapyError if any named column isn't a compared column.
        """
        if column is None and threshold is not None:
            summaries = self.column_summary()
            total_match = sum(c.match_count for c in summaries)
            total_compared = sum(c.total_compared for c in summaries)
            rate = 1.0 if total_compared == 0 else total_match / total_compared
            if rate < threshold:
                raise MismatchThresholdError(
                    f"Overall match rate {rate:.4f} is below threshold {threshold:.4f}",
                    threshold=threshold,
                    failures=[(None, rate)],
                )
            return

        columns = [column] if isinstance(column, str) else column
        by_name = {cs.column: cs for cs in self.column_summary()}
        if columns is None:
            columns = list(by_name)
        else:
            columns = list(columns)
            unknown = [c for c in columns if c not in by_name]
            if unknown:
                raise JuxtapyError(f"Not a compared column(s): {unknown}")

        if not columns:
            return

        rates = {c: by_name[c].match_pct / 100.0 for c in columns}

        if threshold is None:
            values = list(rates.values())
            mean = statistics.mean(values)
            stdev = statistics.pstdev(values)
            effective_threshold = mean - 2 * stdev
            threshold_note = f" (auto: mean={mean:.4f}, stdev={stdev:.4f})"
        else:
            effective_threshold = threshold
            threshold_note = ""

        failures = [(c, r) for c, r in rates.items() if r < effective_threshold]
        if failures:
            lines = "\n".join(f"  {c}: match_rate={r:.4f}" for c, r in failures)
            raise MismatchThresholdError(
                f"{len(failures)} column(s) below threshold "
                f"{effective_threshold:.4f}{threshold_note}:\n{lines}",
                threshold=effective_threshold,
                failures=failures,
            )

    def validate(
        self,
        schema_check: bool = True,
        row_check: bool = True,
        column: str | Sequence[str] | None = None,
        threshold: float | None = None,
        null_check: bool = True,
        null_tolerance: float = 0.0,
    ) -> list[ValidationFailure]:
        """Run a bundle of data-quality checks and return only the ones that failed.

        Unlike assert_match, this never raises for a failed check — it returns a
        (possibly empty) list of ValidationFailure, so a pipeline can log/branch on
        the result directly instead of needing try/except. It still raises JuxtapyError
        for a genuinely invalid call (e.g. an unknown column name in ``column``).

        - schema_check: flag columns only on one side, or with a changed dtype
          (schema_diff().has_drift).
        - row_check: flag duplicate join keys, or rows only on one side (row_summary()).
        - column / threshold: forwarded to assert_match() for the per-column match-rate
          check. threshold=None (the default) means auto-threshold across all compared
          columns, or across ``column`` if given. Pass column=[] to skip this check.
        - null_check / null_tolerance: flag columns (same scope as ``column``) whose null
          rate increased from df1 to df2 by more than ``null_tolerance`` percentage
          points (default 0.0 — any increase at all is flagged). Decreases in null rate
          are never flagged.
        """
        failures: list[ValidationFailure] = []

        if schema_check:
            sd = self.schema_diff()
            if sd.only_in_df1:
                failures.append(
                    ValidationFailure(
                        "schema_drift", f"columns only in {self.df1_name}: {sd.only_in_df1}"
                    )
                )
            if sd.only_in_df2:
                failures.append(
                    ValidationFailure(
                        "schema_drift", f"columns only in {self.df2_name}: {sd.only_in_df2}"
                    )
                )
            if sd.dtype_changes:
                failures.append(
                    ValidationFailure("schema_drift", f"dtype changed: {sd.dtype_changes}")
                )

        if row_check:
            rs = self.row_summary()
            if rs.duplicate_keys_df1:
                failures.append(
                    ValidationFailure(
                        "duplicate_keys",
                        f"{rs.duplicate_keys_df1} duplicate join key row(s) in {self.df1_name}",
                    )
                )
            if rs.duplicate_keys_df2:
                failures.append(
                    ValidationFailure(
                        "duplicate_keys",
                        f"{rs.duplicate_keys_df2} duplicate join key row(s) in {self.df2_name}",
                    )
                )
            if rs.only_in_df1:
                failures.append(
                    ValidationFailure(
                        "rows_only_in_df1", f"{rs.only_in_df1} row(s) only in {self.df1_name}"
                    )
                )
            if rs.only_in_df2:
                failures.append(
                    ValidationFailure(
                        "rows_only_in_df2", f"{rs.only_in_df2} row(s) only in {self.df2_name}"
                    )
                )

        try:
            self.assert_match(threshold=threshold, column=column)
        except MismatchThresholdError as e:
            for col, rate in e.failures:
                label = f"column:{col}" if col is not None else "overall"
                failures.append(
                    ValidationFailure(
                        label, f"match_rate={rate:.4f} below threshold {e.threshold:.4f}"
                    )
                )

        if null_check:
            if column is None:
                null_columns = list(self._compared_columns)
            elif isinstance(column, str):
                null_columns = [column]
            else:
                null_columns = list(column)
            by_name = {cs.column: cs for cs in self.column_summary()}
            for col in null_columns:
                cs = by_name.get(col)
                if cs is None:
                    continue
                delta = cs.null_pct_df2 - cs.null_pct_df1
                if delta > null_tolerance:
                    failures.append(
                        ValidationFailure(
                            f"null_rate:{col}",
                            f"null% increased from {cs.null_pct_df1:.2f} to {cs.null_pct_df2:.2f}",
                        )
                    )

        return failures

    def _tolerance_note(self) -> str | None:
        if not (self.abs_tol or self.rel_tol or self.tolerances):
            return None
        note = f"abs_tol={self.abs_tol}, rel_tol={self.rel_tol}"
        if self.tolerances:
            overrides = ", ".join(
                f"{col}=(abs_tol={a}, rel_tol={r})" for col, (a, r) in self.tolerances.items()
            )
            note += f" (overrides: {overrides})"
        return note

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
            tolerance_note=self._tolerance_note(),
        )


def compare(
    df1: Any,
    df2: Any,
    join_columns: str | Sequence[str],
    **kwargs: Any,
) -> CompareReport:
    """Functional shortcut: ``Compare(df1, df2, join_columns, **kwargs).report()``."""
    return Compare(df1, df2, join_columns, **kwargs).report()
