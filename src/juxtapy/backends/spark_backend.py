from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

from juxtapy.exceptions import JoinKeyError

_LEFT_SUFFIX = "__juxtapy_left"
_RIGHT_SUFFIX = "__juxtapy_right"
_ROW_CLASS_COL = "__juxtapy_row_class"


def _validate_keys(df: SparkDataFrame, keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise JoinKeyError(f"Join column(s) not found in dataframe: {missing}")


class SparkAdapter:
    def __init__(self, df: SparkDataFrame) -> None:
        self._df = df

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    def row_count(self) -> int:
        return self._df.count()

    def duplicate_key_count(self, keys: Sequence[str]) -> int:
        keys = list(keys)
        _validate_keys(self._df, keys)
        # matches pandas' df.duplicated().sum() semantics: count every occurrence
        # past the first in each duplicated group, not the full group size.
        row = (
            self._df.groupBy(*keys)
            .count()
            .filter(F.col("count") > 1)
            .agg(F.sum(F.col("count") - 1).alias("total"))
            .collect()[0]
        )
        return int(row["total"]) if row["total"] is not None else 0

    def dtype_of(self, column: str) -> str:
        return dict(self._df.dtypes)[column]

    def full_outer_join(self, other: SparkAdapter, keys: Sequence[str]) -> SparkJoinedAdapter:
        keys = list(keys)
        _validate_keys(self._df, keys)
        _validate_keys(other._df, keys)

        left = self._df
        for c in left.columns:
            left = left.withColumnRenamed(c, f"{c}{_LEFT_SUFFIX}")
        right = other._df
        for c in right.columns:
            right = right.withColumnRenamed(c, f"{c}{_RIGHT_SUFFIX}")

        join_cond = [
            F.col(f"{k}{_LEFT_SUFFIX}") == F.col(f"{k}{_RIGHT_SUFFIX}") for k in keys
        ]
        joined = left.join(right, on=join_cond, how="full_outer")

        first_key = keys[0]
        row_class = (
            F.when(F.col(f"{first_key}{_RIGHT_SUFFIX}").isNull(), F.lit("only_in_left"))
            .when(F.col(f"{first_key}{_LEFT_SUFFIX}").isNull(), F.lit("only_in_right"))
            .otherwise(F.lit("both"))
        )
        joined = joined.withColumn(_ROW_CLASS_COL, row_class)
        return SparkJoinedAdapter(joined)


class SparkJoinedAdapter:
    def __init__(self, joined: SparkDataFrame) -> None:
        self._joined = joined
        self._row_class_counts_cache: dict[str, int] = None  # type: ignore[assignment]

    def _row_class_counts(self) -> dict[str, int]:
        if self._row_class_counts_cache is None:
            rows = self._joined.groupBy(_ROW_CLASS_COL).count().collect()
            self._row_class_counts_cache = {r[_ROW_CLASS_COL]: r["count"] for r in rows}
        return self._row_class_counts_cache

    def only_in_left_count(self) -> int:
        return self._row_class_counts().get("only_in_left", 0)

    def only_in_right_count(self) -> int:
        return self._row_class_counts().get("only_in_right", 0)

    def common_count(self) -> int:
        return self._row_class_counts().get("both", 0)

    def compare_columns(self, columns: Sequence[str]) -> dict[str, tuple[int, int]]:
        columns = list(columns)
        if not columns:
            return {}
        both = self._joined.filter(F.col(_ROW_CLASS_COL) == "both")
        agg_exprs = [F.count(F.lit(1)).alias("__total")]
        for column in columns:
            left_col = F.col(f"{column}{_LEFT_SUFFIX}")
            right_col = F.col(f"{column}{_RIGHT_SUFFIX}")
            agg_exprs.append(
                F.sum(left_col.eqNullSafe(right_col).cast("long")).alias(f"__match__{column}")
            )
        row = both.agg(*agg_exprs).collect()[0]
        total = row["__total"] or 0
        results: dict[str, tuple[int, int]] = {}
        for column in columns:
            match_count = int(row[f"__match__{column}"] or 0)
            results[column] = (match_count, total - match_count)
        return results

    def sample_mismatched_rows(self, column: str, keys: Sequence[str], n: int) -> pd.DataFrame:
        left_col = F.col(f"{column}{_LEFT_SUFFIX}")
        right_col = F.col(f"{column}{_RIGHT_SUFFIX}")
        key_cols = [F.col(f"{k}{_LEFT_SUFFIX}").alias(k) for k in keys]
        mismatched = (
            self._joined.filter(F.col(_ROW_CLASS_COL) == "both")
            .filter(~left_col.eqNullSafe(right_col))
            .select(*key_cols, left_col.alias(f"{column}_df1"), right_col.alias(f"{column}_df2"))
            .limit(n)
        )
        return mismatched.toPandas()
