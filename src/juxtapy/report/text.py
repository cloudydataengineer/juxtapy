from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from juxtapy.results import CompareReport


def render_text(report: CompareReport) -> str:
    lines = []
    lines.append(f"juxtapy comparison: {report.df1_name} vs {report.df2_name}")
    lines.append(f"Join column(s): {', '.join(report.join_columns)}")
    lines.append("")

    rs = report.row_summary
    lines.append("Row summary")
    lines.append("-----------")
    lines.append(f"  Rows in {report.df1_name}: {rs.rows_df1}")
    lines.append(f"  Rows in {report.df2_name}: {rs.rows_df2}")
    lines.append(f"  Common (joined) rows: {rs.common_rows}")
    lines.append(f"  Only in {report.df1_name}: {rs.only_in_df1}")
    lines.append(f"  Only in {report.df2_name}: {rs.only_in_df2}")
    if rs.duplicate_keys_df1 or rs.duplicate_keys_df2:
        lines.append(
            f"  WARNING: duplicate join keys — {rs.duplicate_keys_df1} in {report.df1_name}, "
            f"{rs.duplicate_keys_df2} in {report.df2_name}"
        )
    lines.append("")

    sd = report.schema_diff
    if sd.has_drift:
        lines.append("Schema drift")
        lines.append("------------")
        if sd.only_in_df1:
            lines.append(f"  Columns only in {report.df1_name}: {', '.join(sd.only_in_df1)}")
        if sd.only_in_df2:
            lines.append(f"  Columns only in {report.df2_name}: {', '.join(sd.only_in_df2)}")
        for col, (d1, d2) in sd.dtype_changes.items():
            lines.append(f"  Type changed for '{col}': {d1} -> {d2}")
        lines.append("")

    lines.append("Column summary (worst mismatch first)")
    lines.append("--------------------------------------")
    if not report.column_summary:
        lines.append("  (no shared columns to compare)")
    for cs in report.column_summary:
        lines.append(
            f"  {cs.column}: {cs.match_count} match / {cs.mismatch_count} mismatch "
            f"({cs.match_pct:.2f}% match, dtypes: {cs.dtype1} vs {cs.dtype2})"
        )
    lines.append("")

    if report.samples:
        lines.append("Sample mismatches")
        lines.append("------------------")
        for col, sample_df in report.samples.items():
            lines.append(f"  {col}:")
            for line in sample_df.to_string(index=False).splitlines():
                lines.append(f"    {line}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
