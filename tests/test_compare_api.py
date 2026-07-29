from __future__ import annotations

import pandas as pd
import pytest

from juxtapy import Compare, JoinKeyError, JuxtapyError, MismatchThresholdError, ValidationFailure


def test_perfect_match(backend, make_df):
    df1 = make_df([(1, 10), (2, 20), (3, 30)], ["id", "v"])
    df2 = make_df([(1, 10), (2, 20), (3, 30)], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id")
    assert cmp.matches() is True
    rs = cmp.row_summary()
    assert rs.only_in_df1 == 0 and rs.only_in_df2 == 0
    assert all(cs.mismatch_count == 0 for cs in cmp.column_summary())


def test_known_mismatch_counts(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    summary = {cs.column: cs for cs in cmp.column_summary()}
    assert summary["amount"].match_count == 2
    assert summary["amount"].mismatch_count == 2
    assert summary["name"].match_count == 3
    assert summary["name"].mismatch_count == 1


def test_rows_only_in_one_table_excluded_from_column_counts(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    rs = cmp.row_summary()
    assert rs.only_in_df1 == 1
    assert rs.only_in_df2 == 1
    total_compared = sum(cs.total_compared for cs in cmp.column_summary())
    # 4 common rows * 2 compared columns = 8, regardless of the 1 row unique to each side
    assert total_compared == 8


def test_null_handling_parity(backend, make_df):
    df1 = make_df([(1, None), (2, 5), (3, None)], ["id", "v"])
    df2 = make_df([(1, None), (2, 5), (3, 9)], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id")
    cs = cmp.column_summary()[0]
    assert (cs.match_count, cs.mismatch_count) == (2, 1)


def test_column_summary_null_counts(backend, make_df):
    df1 = make_df([(1, None), (2, 5), (3, None), (4, 7)], ["id", "v"])
    df2 = make_df([(1, None), (2, 5), (3, 9), (4, None)], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id")
    cs = cmp.column_summary()[0]
    assert (cs.null_count_df1, cs.null_count_df2) == (2, 2)
    assert (cs.null_pct_df1, cs.null_pct_df2) == (50.0, 50.0)


def test_dtype_mismatch_reported(backend, make_df):
    df1 = make_df([(1, 10), (2, 20)], ["id", "v"])
    df2 = make_df([(1, "10"), (2, "20")], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id")
    cs = cmp.column_summary()[0]
    assert cs.dtype1 != cs.dtype2


def test_duplicate_join_keys_warn_not_crash(backend, make_df):
    df1 = make_df([(1, 10), (1, 11), (2, 20)], ["id", "v"])
    df2 = make_df([(1, 10), (2, 20)], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id")
    with pytest.warns(UserWarning, match="Duplicate join keys"):
        rs = cmp.row_summary()
    assert rs.duplicate_keys_df1 == 1  # second id=1 row counts as the duplicate


def test_ranking_order_worst_first(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    summaries = cmp.column_summary()
    mismatch_counts = [cs.mismatch_count for cs in summaries]
    assert mismatch_counts == sorted(mismatch_counts, reverse=True)
    assert summaries[0].column == "amount"  # 2 mismatches, worse than name's 1


def test_sample_mismatches_always_pandas_and_capped(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    sample = cmp.sample_mismatches("amount", n=1)
    assert isinstance(sample, pd.DataFrame)
    assert len(sample) == 1
    assert list(sample.columns) == ["id", "amount_df1", "amount_df2"]


def test_empty_dataframes(backend, request):
    if backend == "pandas":
        df1 = pd.DataFrame({"id": pd.Series(dtype="int64"), "v": pd.Series(dtype="int64")})
        df2 = pd.DataFrame({"id": pd.Series(dtype="int64"), "v": pd.Series(dtype="int64")})
    else:
        from pyspark.sql.types import LongType, StructField, StructType

        spark = request.getfixturevalue("spark_session")
        schema = StructType([StructField("id", LongType()), StructField("v", LongType())])
        df1 = spark.createDataFrame([], schema)
        df2 = spark.createDataFrame([], schema)
    cmp = Compare(df1, df2, join_columns="id")
    rs = cmp.row_summary()
    assert (rs.rows_df1, rs.rows_df2, rs.common_rows) == (0, 0, 0)
    assert cmp.matches() is True


def test_composite_join_keys(backend, make_df):
    df1 = make_df([(1, "a", 100), (1, "b", 200)], ["id", "region", "v"])
    df2 = make_df([(1, "a", 100), (1, "b", 999)], ["id", "region", "v"])
    cmp = Compare(df1, df2, join_columns=["id", "region"])
    cs = cmp.column_summary()[0]
    assert (cs.match_count, cs.mismatch_count) == (1, 1)


def test_ignore_columns_filter(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id", ignore_columns=["name"])
    assert [cs.column for cs in cmp.column_summary()] == ["amount"]


def test_columns_to_compare_filter(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id", columns_to_compare=["name"])
    assert [cs.column for cs in cmp.column_summary()] == ["name"]


def test_columns_to_compare_unknown_raises(df_pair):
    df1, df2 = df_pair
    with pytest.raises(JuxtapyError):
        Compare(df1, df2, join_columns="id", columns_to_compare=["nope"])


def test_missing_join_column_raises(backend, make_df):
    df1 = make_df([(1, 10)], ["id", "v"])
    df2 = make_df([(1, 10)], ["other_id", "v"])
    with pytest.raises(JoinKeyError):
        Compare(df1, df2, join_columns="id")


@pytest.mark.spark
def test_mixed_backend_raises(sample_pandas_pair, spark_session):
    df1, df2 = sample_pandas_pair
    spark_df2 = spark_session.createDataFrame(df2)
    with pytest.raises(JuxtapyError):
        Compare(df1, spark_df2, join_columns="id")


def test_unsupported_type_raises():
    with pytest.raises(JuxtapyError):
        Compare({"id": [1]}, {"id": [1]}, join_columns="id")


def test_schema_diff_added_removed_and_type_changed(backend, make_df):
    df1 = make_df([(1, 10, "x")], ["id", "shared_int", "only_in_1"])
    df2 = make_df([(1, "10")], ["id", "shared_int"])
    cmp = Compare(df1, df2, join_columns="id")
    sd = cmp.schema_diff()
    assert sd.only_in_df1 == ["only_in_1"]
    assert sd.only_in_df2 == []
    assert "shared_int" in sd.dtype_changes
    assert sd.has_drift is True


def test_assert_match_overall_threshold(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    cmp.assert_match(threshold=0.0)  # never fails
    with pytest.raises(MismatchThresholdError):
        cmp.assert_match(threshold=1.0)  # there are mismatches, must fail


def test_assert_match_per_column_threshold(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    cmp.assert_match(threshold=0.7, column="name")  # 75% match, passes
    with pytest.raises(MismatchThresholdError):
        cmp.assert_match(threshold=0.9, column="name")


def test_assert_match_column_list_all_pass(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    cmp.assert_match(threshold=0.4, column=["amount", "name"])  # 50% and 75%, both pass


def test_assert_match_column_list_one_fails(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    with pytest.raises(MismatchThresholdError) as exc_info:
        # amount=50% fails, name=75% passes
        cmp.assert_match(threshold=0.6, column=["amount", "name"])
    assert exc_info.value.failures == [("amount", pytest.approx(0.5))]
    assert exc_info.value.column == "amount"  # convenience attr, single failure
    assert exc_info.value.match_rate == pytest.approx(0.5)


def test_assert_match_column_list_multiple_failures_collected(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    with pytest.raises(MismatchThresholdError) as exc_info:
        cmp.assert_match(threshold=0.99, column=["amount", "name"])  # both fail
    failures = exc_info.value.failures
    assert [c for c, _ in failures] == ["amount", "name"]
    assert exc_info.value.column is None  # ambiguous with >1 failure
    assert exc_info.value.match_rate is None


def test_assert_match_unknown_column_in_list_raises(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    with pytest.raises(JuxtapyError):
        cmp.assert_match(threshold=0.5, column=["amount", "nope"])


def _pandas_frames_with_rates(rates: dict, n: int = 10):
    """(df1, df2) with an 'id' key and one column per rate, each column matching
    in exactly round(rate * n) of n rows."""
    ids = list(range(n))
    df1 = pd.DataFrame({"id": ids})
    df2 = pd.DataFrame({"id": ids})
    for col, rate in rates.items():
        match_count = round(rate * n)
        df1[col] = ids
        df2[col] = [ids[i] if i < match_count else -1 for i in range(n)]
    return df1, df2


def test_assert_match_auto_threshold_flags_outlier():
    rates = {f"good{i}": 1.0 for i in range(9)}
    rates["bad"] = 0.0
    df1, df2 = _pandas_frames_with_rates(rates)
    cmp = Compare(df1, df2, join_columns="id")
    with pytest.raises(MismatchThresholdError) as exc_info:
        cmp.assert_match(threshold=None, column=list(rates))
    assert [c for c, _ in exc_info.value.failures] == ["bad"]


def test_assert_match_auto_threshold_all_similar_passes():
    rates = {"a": 1.0, "b": 0.9, "c": 1.0}
    df1, df2 = _pandas_frames_with_rates(rates)
    cmp = Compare(df1, df2, join_columns="id")
    cmp.assert_match(threshold=None, column=list(rates))  # no outlier -> no raise


def test_assert_match_auto_threshold_single_column_always_passes():
    df1, df2 = _pandas_frames_with_rates({"a": 0.5})
    cmp = Compare(df1, df2, join_columns="id")
    cmp.assert_match(threshold=None, column="a")  # pstdev of 1 value is 0 -> never fails


def test_validate_returns_list_of_validation_failure(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    failures = cmp.validate()
    assert isinstance(failures, list)
    assert all(isinstance(f, ValidationFailure) for f in failures)


def test_validate_clean_data_returns_empty_list():
    df = pd.DataFrame({"id": [1, 2, 3], "amount": [10, 20, 30], "name": ["a", "b", "c"]})
    cmp = Compare(df, df.copy(), join_columns="id")
    assert cmp.validate() == []


def test_validate_schema_drift_flagged():
    df1 = pd.DataFrame({"id": [1], "shared_int": [10], "only_in_1": ["x"]})
    df2 = pd.DataFrame({"id": [1], "shared_int": ["10"]})
    cmp = Compare(df1, df2, join_columns="id")
    failures = cmp.validate(row_check=False, null_check=False, column=[])
    assert {f.check for f in failures} == {"schema_drift"}
    assert len(failures) == 2  # only_in_df1 + dtype_changes (only_in_df2 is empty)

    assert cmp.validate(schema_check=False, row_check=False, null_check=False, column=[]) == []


def test_validate_row_check_flags_duplicates_and_only_in_one_side():
    df1 = pd.DataFrame({"id": [1, 1, 2], "v": [10, 11, 20]})
    df2 = pd.DataFrame({"id": [1, 3], "v": [10, 30]})
    cmp = Compare(df1, df2, join_columns="id")
    with pytest.warns(UserWarning, match="Duplicate join keys"):
        failures = cmp.validate(schema_check=False, null_check=False, column=[])
    assert {f.check for f in failures} == {"duplicate_keys", "rows_only_in_df1", "rows_only_in_df2"}


def test_validate_column_check_reports_failing_columns():
    rates = {f"good{i}": 1.0 for i in range(9)}
    rates["bad"] = 0.0
    df1, df2 = _pandas_frames_with_rates(rates)
    cmp = Compare(df1, df2, join_columns="id")
    failures = cmp.validate(schema_check=False, row_check=False, null_check=False, column=list(rates))
    assert len(failures) == 1
    assert failures[0].check == "column:bad"


def test_validate_unknown_column_raises(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    with pytest.raises(JuxtapyError):
        cmp.validate(column=["nope"])


def test_validate_null_check_flags_increase():
    df1 = pd.DataFrame({"id": [1, 2, 3, 4], "v": [1, 2, 3, 4]})
    df2 = pd.DataFrame({"id": [1, 2, 3, 4], "v": [1, 2, None, None]})
    cmp = Compare(df1, df2, join_columns="id")
    # threshold=0.0 neutralizes the column match-rate check so only null_check is observed
    failures = cmp.validate(schema_check=False, row_check=False, threshold=0.0)
    assert "null_rate:v" in {f.check for f in failures}


def test_validate_null_check_does_not_flag_decrease():
    df1 = pd.DataFrame({"id": [1, 2, 3, 4], "v": [1, 2, None, None]})
    df2 = pd.DataFrame({"id": [1, 2, 3, 4], "v": [1, 2, 3, 4]})
    cmp = Compare(df1, df2, join_columns="id")
    failures = cmp.validate(schema_check=False, row_check=False, threshold=0.0)
    assert "null_rate:v" not in {f.check for f in failures}


def test_validate_null_check_respects_tolerance():
    df1 = pd.DataFrame({"id": range(10), "v": [1] * 10})
    df2 = pd.DataFrame({"id": range(10), "v": [1] * 9 + [None]})
    cmp = Compare(df1, df2, join_columns="id")
    failures = cmp.validate(schema_check=False, row_check=False, threshold=0.0)
    assert "null_rate:v" in {f.check for f in failures}  # null% 0 -> 10, default tolerance 0.0

    lenient = cmp.validate(schema_check=False, row_check=False, threshold=0.0, null_tolerance=15.0)
    assert "null_rate:v" not in {f.check for f in lenient}


def test_validate_null_check_disabled():
    df1 = pd.DataFrame({"id": [1, 2, 3, 4], "v": [1, 2, 3, 4]})
    df2 = pd.DataFrame({"id": [1, 2, 3, 4], "v": [1, 2, None, None]})
    cmp = Compare(df1, df2, join_columns="id")
    failures = cmp.validate(schema_check=False, row_check=False, threshold=0.0, null_check=False)
    assert not any(f.check.startswith("null_rate") for f in failures)


def test_report_end_to_end(df_pair):
    df1, df2 = df_pair
    cmp = Compare(df1, df2, join_columns="id")
    report = cmp.report(top_n_columns=1, sample_rows_per_column=1)
    assert report.row_summary.common_rows == 4
    assert len(report.column_summary) == 2
    assert set(report.samples.keys()) == {"amount"}  # only the single worst column sampled
    text = str(report)
    assert "Row summary" in text
    assert "amount" in text
