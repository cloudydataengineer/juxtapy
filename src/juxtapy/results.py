from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RowSummary:
    rows_df1: int
    rows_df2: int
    common_rows: int
    only_in_df1: int
    only_in_df2: int
    duplicate_keys_df1: int
    duplicate_keys_df2: int

    @property
    def all_rows_match(self) -> bool:
        return self.only_in_df1 == 0 and self.only_in_df2 == 0


@dataclass
class ColumnSummary:
    column: str
    match_count: int
    mismatch_count: int
    dtype1: str
    dtype2: str

    @property
    def total_compared(self) -> int:
        return self.match_count + self.mismatch_count

    @property
    def match_pct(self) -> float:
        if self.total_compared == 0:
            return 100.0
        return 100.0 * self.match_count / self.total_compared

    @property
    def mismatch_pct(self) -> float:
        return 100.0 - self.match_pct


@dataclass
class SchemaDiff:
    only_in_df1: list[str]
    only_in_df2: list[str]
    dtype_changes: dict[str, tuple[str, str]]

    @property
    def has_drift(self) -> bool:
        return bool(self.only_in_df1 or self.only_in_df2 or self.dtype_changes)


@dataclass
class CompareReport:
    df1_name: str
    df2_name: str
    join_columns: list[str]
    row_summary: RowSummary
    column_summary: list[ColumnSummary]
    schema_diff: SchemaDiff
    samples: dict[str, pd.DataFrame] = field(default_factory=dict)
    tolerance_note: str | None = None

    def __str__(self) -> str:
        from juxtapy.report.text import render_text

        return render_text(self)

    def to_html(self) -> str:
        from juxtapy.report.html import render_html

        return render_html(self)

    def _repr_html_(self) -> str:
        return self.to_html()

    def to_dict(self) -> dict:
        return {
            "df1_name": self.df1_name,
            "df2_name": self.df2_name,
            "join_columns": self.join_columns,
            "row_summary": vars(self.row_summary),
            "column_summary": [vars(cs) for cs in self.column_summary],
            "schema_diff": vars(self.schema_diff),
            "tolerance": self.tolerance_note,
        }
