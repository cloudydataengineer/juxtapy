from __future__ import annotations

import pandas as pd
import pytest

from juxtapy import Compare, JuxtapyError


def test_tolerance_default_is_exact_match(backend, make_df):
    df1 = make_df([(1, 10.0)], ["id", "v"])
    df2 = make_df([(1, 10.5)], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id")
    cs = cmp.column_summary()[0]
    assert (cs.match_count, cs.mismatch_count) == (0, 1)


def test_abs_tol_within_and_outside_bound(backend, make_df):
    df1 = make_df([(1, 10.0), (2, 10.0)], ["id", "v"])
    df2 = make_df([(1, 10.05), (2, 10.5)], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id", abs_tol=0.1)
    cs = cmp.column_summary()[0]
    assert (cs.match_count, cs.mismatch_count) == (1, 1)


def test_rel_tol_symmetric_boundary(backend, make_df):
    # id 1: diff=2, bound=0.01*102=1.02 -> mismatch
    # id 2: diff=0.5, bound=0.01*100.5=1.005 -> match
    # id 3: diff=1, bound=0.01*max(100,99)=1.0 -> match (boundary, inclusive <=)
    df1 = make_df([(1, 100.0), (2, 100.0), (3, 100.0)], ["id", "v"])
    df2 = make_df([(1, 102.0), (2, 100.5), (3, 99.0)], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id", rel_tol=0.01)
    cs = cmp.column_summary()[0]
    assert (cs.match_count, cs.mismatch_count) == (2, 1)


def test_tolerance_not_applied_to_non_numeric_column(backend, make_df):
    df1 = make_df([(1, "abc")], ["id", "name"])
    df2 = make_df([(1, "abd")], ["id", "name"])
    cmp = Compare(df1, df2, join_columns="id", abs_tol=1000, rel_tol=1000)
    cs = cmp.column_summary()[0]
    assert (cs.match_count, cs.mismatch_count) == (0, 1)


def test_tolerance_per_column_override(backend, make_df):
    df1 = make_df([(1, 10.0, 100.0)], ["id", "a", "b"])
    df2 = make_df([(1, 10.5, 105.0)], ["id", "a", "b"])
    cmp = Compare(df1, df2, join_columns="id", tolerances={"a": (1.0, 0.0)})
    summary = {cs.column: cs for cs in cmp.column_summary()}
    assert (summary["a"].match_count, summary["a"].mismatch_count) == (1, 0)
    assert (summary["b"].match_count, summary["b"].mismatch_count) == (0, 1)


def test_sample_mismatches_consistent_with_tolerance(backend, make_df):
    df1 = make_df([(1, 10.0), (2, 10.0)], ["id", "v"])
    df2 = make_df([(1, 10.05), (2, 20.0)], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id", abs_tol=0.1)
    cs = cmp.column_summary()[0]
    sample = cmp.sample_mismatches("v")
    assert cs.mismatch_count == len(sample) == 1
    assert sample.iloc[0]["id"] == 2


def test_both_null_still_matches_with_tolerance_configured(backend, make_df):
    df1 = make_df([(1, None), (2, 5.0)], ["id", "v"])
    df2 = make_df([(1, None), (2, 5.0)], ["id", "v"])
    cmp = Compare(df1, df2, join_columns="id", abs_tol=0.1, rel_tol=0.1)
    cs = cmp.column_summary()[0]
    assert (cs.match_count, cs.mismatch_count) == (2, 0)


def _simple_pair():
    df1 = pd.DataFrame({"id": [1], "v": [1.0]})
    df2 = pd.DataFrame({"id": [1], "v": [1.0]})
    return df1, df2


def test_negative_abs_tol_raises():
    df1, df2 = _simple_pair()
    with pytest.raises(JuxtapyError):
        Compare(df1, df2, join_columns="id", abs_tol=-1)


def test_negative_rel_tol_raises():
    df1, df2 = _simple_pair()
    with pytest.raises(JuxtapyError):
        Compare(df1, df2, join_columns="id", rel_tol=-1)


def test_negative_tolerances_override_raises():
    df1, df2 = _simple_pair()
    with pytest.raises(JuxtapyError):
        Compare(df1, df2, join_columns="id", tolerances={"v": (-1.0, 0.0)})


def test_unknown_tolerances_column_raises():
    df1, df2 = _simple_pair()
    with pytest.raises(JuxtapyError):
        Compare(df1, df2, join_columns="id", tolerances={"nope": (0.0, 0.0)})
