from __future__ import annotations

import pandas as pd
import pytest

_BACKEND_PARAMS = [
    pytest.param("pandas", id="pandas"),
    pytest.param("spark", id="spark", marks=pytest.mark.spark),
]


@pytest.fixture(scope="session")
def spark_session():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("juxtapy-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture(params=_BACKEND_PARAMS)
def backend(request):
    return request.param


@pytest.fixture
def make_df(backend, request):
    """Build a dataframe in whichever backend is active for this test instance.

    Lazily fetches the (session-scoped, JVM-starting) spark_session fixture only
    when backend == "spark", so pandas-only test runs never pay Spark startup cost.
    """

    def _make(rows, columns):
        if backend == "pandas":
            return pd.DataFrame(rows, columns=columns)
        spark = request.getfixturevalue("spark_session")
        return spark.createDataFrame(rows, columns)

    return _make


@pytest.fixture
def sample_pandas_pair():
    df1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "amount": [10, 20, 30, 40, 50],
            "name": ["a", "b", "c", "d", "e"],
        }
    )
    df2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 6],
            "amount": [10, 25, 30, 45, 60],
            "name": ["a", "B", "c", "d", "f"],
        }
    )
    return df1, df2


@pytest.fixture(params=_BACKEND_PARAMS)
def df_pair(request, sample_pandas_pair):
    """(df1, df2) with intentional mismatches, in whichever backend is active.

    Lazily resolves spark_session only for the "spark" param instance, so a
    "pandas"-only test run (`pytest -m "not spark"`) never starts a JVM.
    """
    if request.param == "pandas":
        return sample_pandas_pair
    df1, df2 = sample_pandas_pair
    spark = request.getfixturevalue("spark_session")
    return spark.createDataFrame(df1), spark.createDataFrame(df2)
