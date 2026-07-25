from __future__ import annotations

from juxtapy import Compare


def _report(sample_pandas_pair):
    df1, df2 = sample_pandas_pair
    return Compare(df1, df2, join_columns="id", df1_name="prod", df2_name="staging").report()


def test_text_report_contains_key_sections(sample_pandas_pair):
    text = str(_report(sample_pandas_pair))
    assert "Row summary" in text
    assert "Column summary" in text
    assert "Sample mismatches" in text
    assert "amount" in text and "name" in text


def test_html_report_is_valid_and_has_severity_classes(sample_pandas_pair):
    html = _report(sample_pandas_pair).to_html()
    assert "<table>" in html
    assert 'class="jx-red"' in html or 'class="jx-amber"' in html
    assert "amount" in html


def test_repr_html_matches_to_html(sample_pandas_pair):
    report = _report(sample_pandas_pair)
    assert report._repr_html_() == report.to_html()


def test_html_escapes_names():
    import pandas as pd

    df1 = pd.DataFrame({"id": [1, 2], "v": [1, 2]})
    df2 = pd.DataFrame({"id": [1, 2], "v": [1, 3]})
    report = Compare(df1, df2, join_columns="id", df1_name="<prod>", df2_name="staging").report()
    html = report.to_html()
    assert "<prod>" not in html
    assert "&lt;prod&gt;" in html


def test_sample_mismatches_present_for_worst_columns(sample_pandas_pair):
    report = _report(sample_pandas_pair)
    assert "amount" in report.samples
    sample_html = report.to_html()
    # column-first indexing avoids pandas upcasting the id to float when read row-first
    first_id = report.samples["amount"]["id"].iloc[0]
    assert str(int(first_id)) in sample_html
