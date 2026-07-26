from __future__ import annotations

import html as _html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from juxtapy.results import CompareReport

_STYLE = """
<style>
  .juxtapy-report { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 13px; }
  .juxtapy-report h3, .juxtapy-report h4 { margin: 12px 0 4px; }
  .juxtapy-report table { border-collapse: collapse; margin-bottom: 8px; }
  .juxtapy-report th, .juxtapy-report td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
  .juxtapy-report .jx-green { background-color: #d9f2d9; }
  .juxtapy-report .jx-amber { background-color: #fff2cc; }
  .juxtapy-report .jx-red { background-color: #f8d7da; }
  .juxtapy-report .jx-warning { color: #a94442; font-weight: bold; }
</style>
""".strip()


def _severity_class(match_pct: float) -> str:
    if match_pct >= 99.0:
        return "jx-green"
    if match_pct >= 90.0:
        return "jx-amber"
    return "jx-red"


def _esc(value: object) -> str:
    return _html.escape(str(value))


def render_html(report: CompareReport) -> str:
    parts = [_STYLE, '<div class="juxtapy-report">']
    parts.append(f"<h3>juxtapy comparison: {_esc(report.df1_name)} vs {_esc(report.df2_name)}</h3>")
    parts.append(f"<p>Join column(s): {_esc(', '.join(report.join_columns))}</p>")
    if report.tolerance_note:
        parts.append(f"<p>Tolerance: {_esc(report.tolerance_note)}</p>")

    rs = report.row_summary
    parts.append("<h4>Row summary</h4>")
    parts.append("<table>")
    parts.append(f"<tr><th>Rows in {_esc(report.df1_name)}</th><td>{rs.rows_df1}</td></tr>")
    parts.append(f"<tr><th>Rows in {_esc(report.df2_name)}</th><td>{rs.rows_df2}</td></tr>")
    parts.append(f"<tr><th>Common rows</th><td>{rs.common_rows}</td></tr>")
    parts.append(f"<tr><th>Only in {_esc(report.df1_name)}</th><td>{rs.only_in_df1}</td></tr>")
    parts.append(f"<tr><th>Only in {_esc(report.df2_name)}</th><td>{rs.only_in_df2}</td></tr>")
    parts.append("</table>")
    if rs.duplicate_keys_df1 or rs.duplicate_keys_df2:
        parts.append(
            f'<p class="jx-warning">Duplicate join keys: {rs.duplicate_keys_df1} in '
            f"{_esc(report.df1_name)}, {rs.duplicate_keys_df2} in {_esc(report.df2_name)}</p>"
        )

    sd = report.schema_diff
    if sd.has_drift:
        parts.append("<h4>Schema drift</h4><ul>")
        if sd.only_in_df1:
            parts.append(
                f"<li>Only in {_esc(report.df1_name)}: {_esc(', '.join(sd.only_in_df1))}</li>"
            )
        if sd.only_in_df2:
            parts.append(
                f"<li>Only in {_esc(report.df2_name)}: {_esc(', '.join(sd.only_in_df2))}</li>"
            )
        for col, (d1, d2) in sd.dtype_changes.items():
            parts.append(f"<li>Type changed for '{_esc(col)}': {_esc(d1)} &rarr; {_esc(d2)}</li>")
        parts.append("</ul>")

    parts.append("<h4>Column summary (worst mismatch first)</h4>")
    if not report.column_summary:
        parts.append("<p>(no shared columns to compare)</p>")
    else:
        parts.append(
            "<table><tr><th>Column</th><th>Match</th><th>Mismatch</th>"
            "<th>Match %</th><th>dtypes</th></tr>"
        )
        for cs in report.column_summary:
            cls = _severity_class(cs.match_pct)
            parts.append(
                f'<tr class="{cls}"><td>{_esc(cs.column)}</td><td>{cs.match_count}</td>'
                f"<td>{cs.mismatch_count}</td><td>{cs.match_pct:.2f}%</td>"
                f"<td>{_esc(cs.dtype1)} vs {_esc(cs.dtype2)}</td></tr>"
            )
        parts.append("</table>")

    if report.samples:
        parts.append("<h4>Sample mismatches</h4>")
        for col, sample_df in report.samples.items():
            parts.append(f"<p><b>{_esc(col)}</b></p>")
            parts.append(sample_df.to_html(index=False, border=0))

    parts.append("</div>")
    return "".join(parts)
