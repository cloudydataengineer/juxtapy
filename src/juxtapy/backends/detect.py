from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from juxtapy.exceptions import JuxtapyError


def infer_backend(df: Any) -> str:
    if isinstance(df, pd.DataFrame):
        return "pandas"

    if "pyspark" in sys.modules:
        from pyspark.sql import DataFrame as SparkDataFrame

        if isinstance(df, SparkDataFrame):
            return "spark"

    raise JuxtapyError(
        f"Unsupported dataframe type: {type(df)!r}. juxtapy supports pandas.DataFrame "
        "and pyspark.sql.DataFrame (install with `pip install juxtapy[spark]` for Spark support)."
    )


def make_adapter(df: Any):
    backend = infer_backend(df)
    if backend == "pandas":
        from juxtapy.backends.pandas_backend import PandasAdapter

        return PandasAdapter(df)

    from juxtapy.backends.spark_backend import SparkAdapter

    return SparkAdapter(df)
