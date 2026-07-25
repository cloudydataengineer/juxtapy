"""PySpark comparison, e.g. from a Databricks notebook.

On Databricks: `%pip install juxtapy` (no [spark] extra — the runtime already
bundles a pinned PySpark). Then paste the cells below.

Locally: `uv sync --extra spark` and run with `spark-submit examples/04_pyspark_databricks.py`
(requires Java).
"""

from pyspark.sql import SparkSession

from juxtapy import Compare

spark = SparkSession.builder.appName("juxtapy-example").getOrCreate()

# In Databricks these would typically be spark.table("prod.orders") etc.
prod_df = spark.createDataFrame(
    [(1, 10, "a"), (2, 20, "b"), (3, 30, "c"), (4, 40, "d")],
    ["id", "amount", "name"],
)
staging_df = spark.createDataFrame(
    [(1, 10, "a"), (2, 25, "b"), (3, 30, "C"), (6, 60, "f")],
    ["id", "amount", "name"],
)

cmp = Compare(prod_df, staging_df, join_columns="id", df1_name="prod", df2_name="staging")

print(cmp.row_summary())
for col_summary in cmp.column_summary():
    print(col_summary)

# sample_mismatches() always returns a pandas DataFrame (even for Spark inputs),
# so it's straightforward to `display()` in a Databricks notebook.
print(cmp.sample_mismatches("amount"))

# cmp.report() in a Databricks/Jupyter notebook cell auto-renders as an HTML table.
report = cmp.report()
print(report)
