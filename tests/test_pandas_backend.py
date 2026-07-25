from __future__ import annotations

import pandas as pd

from juxtapy.backends.pandas_backend import PandasAdapter


def test_row_count():
    df = pd.DataFrame({"id": [1, 2, 3]})
    assert PandasAdapter(df).row_count() == 3


def test_duplicate_key_count():
    df = pd.DataFrame({"id": [1, 1, 2, 3, 3]})
    assert PandasAdapter(df).duplicate_key_count(["id"]) == 2


def test_full_outer_join_row_classification():
    df1 = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    df2 = pd.DataFrame({"id": [2, 3, 4], "v": [20, 31, 40]})
    joined = PandasAdapter(df1).full_outer_join(PandasAdapter(df2), ["id"])
    assert joined.only_in_left_count() == 1
    assert joined.only_in_right_count() == 1
    assert joined.common_count() == 2


def test_compare_columns_exact_counts():
    df1 = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    df2 = pd.DataFrame({"id": [1, 2, 3], "v": [10, 21, 30]})
    joined = PandasAdapter(df1).full_outer_join(PandasAdapter(df2), ["id"])
    counts = joined.compare_columns(["v"])
    assert counts["v"] == (2, 1)


def test_compare_columns_null_safe():
    df1 = pd.DataFrame({"id": [1, 2, 3], "v": [None, 5, None]})
    df2 = pd.DataFrame({"id": [1, 2, 3], "v": [None, 5, 9]})
    joined = PandasAdapter(df1).full_outer_join(PandasAdapter(df2), ["id"])
    match_count, mismatch_count = joined.compare_columns(["v"])["v"]
    assert match_count == 2  # both-null and equal-value rows match
    assert mismatch_count == 1


def test_sample_mismatched_rows_shape():
    df1 = pd.DataFrame({"id": range(10), "v": range(10)})
    df2 = pd.DataFrame({"id": range(10), "v": [x + 1 for x in range(10)]})
    joined = PandasAdapter(df1).full_outer_join(PandasAdapter(df2), ["id"])
    sample = joined.sample_mismatched_rows("v", ["id"], n=3)
    assert len(sample) == 3
    assert list(sample.columns) == ["id", "v_df1", "v_df2"]
