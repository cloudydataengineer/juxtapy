from __future__ import annotations

import pytest

pytestmark = pytest.mark.spark


def test_row_count(spark_session):
    from juxtapy.backends.spark_backend import SparkAdapter

    df = spark_session.createDataFrame([(1,), (2,), (3,)], ["id"])
    assert SparkAdapter(df).row_count() == 3


def test_duplicate_key_count(spark_session):
    from juxtapy.backends.spark_backend import SparkAdapter

    df = spark_session.createDataFrame([(1,), (1,), (2,), (3,), (3,)], ["id"])
    assert SparkAdapter(df).duplicate_key_count(["id"]) == 2


def test_full_outer_join_row_classification(spark_session):
    from juxtapy.backends.spark_backend import SparkAdapter

    df1 = spark_session.createDataFrame([(1, 10), (2, 20), (3, 30)], ["id", "v"])
    df2 = spark_session.createDataFrame([(2, 20), (3, 31), (4, 40)], ["id", "v"])
    joined = SparkAdapter(df1).full_outer_join(SparkAdapter(df2), ["id"])
    assert joined.only_in_left_count() == 1
    assert joined.only_in_right_count() == 1
    assert joined.common_count() == 2


def test_compare_columns_exact_counts(spark_session):
    from juxtapy.backends.spark_backend import SparkAdapter

    df1 = spark_session.createDataFrame([(1, 10), (2, 20), (3, 30)], ["id", "v"])
    df2 = spark_session.createDataFrame([(1, 10), (2, 21), (3, 30)], ["id", "v"])
    joined = SparkAdapter(df1).full_outer_join(SparkAdapter(df2), ["id"])
    counts = joined.compare_columns(["v"])
    assert counts["v"] == (2, 1)


def test_compare_columns_null_safe(spark_session):
    from juxtapy.backends.spark_backend import SparkAdapter

    df1 = spark_session.createDataFrame([(1, None), (2, 5), (3, None)], ["id", "v"])
    df2 = spark_session.createDataFrame([(1, None), (2, 5), (3, 9)], ["id", "v"])
    joined = SparkAdapter(df1).full_outer_join(SparkAdapter(df2), ["id"])
    match_count, mismatch_count = joined.compare_columns(["v"])["v"]
    assert match_count == 2
    assert mismatch_count == 1


def test_sample_mismatched_rows_shape(spark_session):
    from juxtapy.backends.spark_backend import SparkAdapter

    rows1 = [(i, i) for i in range(10)]
    rows2 = [(i, i + 1) for i in range(10)]
    df1 = spark_session.createDataFrame(rows1, ["id", "v"])
    df2 = spark_session.createDataFrame(rows2, ["id", "v"])
    joined = SparkAdapter(df1).full_outer_join(SparkAdapter(df2), ["id"])
    sample = joined.sample_mismatched_rows("v", ["id"], n=3)
    assert len(sample) == 3
    assert list(sample.columns) == ["id", "v_df1", "v_df2"]
