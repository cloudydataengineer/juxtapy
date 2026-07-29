# Databricks notebook source
# MAGIC %md
# MAGIC # juxtapy — full API walkthrough
# MAGIC
# MAGIC This notebook demonstrates every public function/class in `juxtapy` against PySpark
# MAGIC DataFrames, the way you'd typically use it on Databricks (e.g. comparing two
# MAGIC `spark.table(...)` results). It is also a valid plain Python script — the `# MAGIC`
# MAGIC comment lines are inert outside Databricks, and it creates its own `SparkSession`
# MAGIC (plus `display`/`displayHTML` fallbacks) when those aren't already provided, so
# MAGIC `python examples/05_databricks_full_demo.py` runs it unchanged too.
# MAGIC
# MAGIC **To use on Databricks:** Workspace → Import → upload this file directly (Databricks
# MAGIC recognizes the `# Databricks notebook source` header and rebuilds the cells automatically).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install
# MAGIC
# MAGIC No `[spark]` extra — Databricks Runtime already bundles a pinned PySpark, and installing
# MAGIC a different version via the extra risks a conflict.

# COMMAND ----------

# MAGIC %pip install juxtapy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Basic comparison — single join key
# MAGIC
# MAGIC Covers: `Compare(...)`, `row_summary()`, `column_summary()`, `sample_mismatches()`, `matches()`.

# COMMAND ----------

from juxtapy import Compare, JoinKeyError, JuxtapyError, MismatchThresholdError, compare

# `spark`, `display`, and `displayHTML` are provided automatically in a Databricks
# notebook. This fills them in only when running as a plain script elsewhere, so the
# same file works in both places.
try:
    spark, display, displayHTML  # noqa: B018
except NameError:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("juxtapy-demo").getOrCreate()
    display = print
    displayHTML = print

prod_df = spark.createDataFrame(
    [(1, 10, "a"), (2, 20, "b"), (3, 30, "c"), (4, 40, "d")],
    ["id", "amount", "name"],
)
staging_df = spark.createDataFrame(
    [(1, 10, "a"), (2, 25, "b"), (3, 30, "C"), (6, 60, "f")],
    ["id", "amount", "name"],
)

# id 4 only in prod, id 6 only in staging, id 2's amount drifted, id 3's name case changed
cmp = Compare(prod_df, staging_df, join_columns="id", df1_name="prod", df2_name="staging")

# COMMAND ----------

# DBTITLE 1,row_summary()
cmp.row_summary()

# COMMAND ----------

# DBTITLE 1,column_summary() — worst mismatch first
cmp.column_summary()

# COMMAND ----------

# DBTITLE 1,sample_mismatches(column, n) — always a pandas DataFrame
display(cmp.sample_mismatches("amount"))

# COMMAND ----------

# DBTITLE 1,matches() — bool: do the tables fully agree?
print("matches():", cmp.matches())
print("matches(ignore_extra_columns=True):", cmp.matches(ignore_extra_columns=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. schema_diff() — columns only on one side, or with a changed dtype

# COMMAND ----------

prod_schema_df = spark.createDataFrame([(1, 10, "x")], ["id", "shared_int", "only_in_prod"])
staging_schema_df = spark.createDataFrame([(1, "10")], ["id", "shared_int"])

cmp_schema = Compare(prod_schema_df, staging_schema_df, join_columns="id")
sd = cmp_schema.schema_diff()
print(sd)
print("has_drift:", sd.has_drift)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. assert_match() — CI/pipeline gating
# MAGIC
# MAGIC Overall and single-column checks (the original API):

# COMMAND ----------

try:
    cmp.assert_match(threshold=0.99)  # overall match rate across all compared columns
except MismatchThresholdError as e:
    print(f"Overall gate failed: match_rate={e.match_rate:.4f} < threshold={e.threshold:.4f}")

try:
    cmp.assert_match(threshold=1.0, column="amount")  # single column
except MismatchThresholdError as e:
    print(f"Column gate failed on '{e.column}': match_rate={e.match_rate:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Checking a *list* of columns in one call
# MAGIC
# MAGIC Raises once, reporting every column that failed (via `exc.failures`) — not just the first.
# MAGIC Here `currency` broke entirely on the staging side while the numeric columns only drifted
# MAGIC on one row each (normal noise).

# COMMAND ----------

import pandas as pd

wide_pdf = pd.DataFrame(
    {
        "id": range(10),
        "amount": range(10),
        "qty": range(10),
        "tax": range(10),
        "discount": range(10),
        "total": range(10),
        "currency": ["USD"] * 10,
    }
)
wide_staging_pdf = wide_pdf.copy()
for col in ["amount", "qty", "tax", "discount", "total"]:
    wide_staging_pdf.loc[0, col] = -1  # one row of ordinary drift -> 90% match
wide_staging_pdf["currency"] = "usd"  # encoding broke entirely -> 0% match

wide_prod = spark.createDataFrame(wide_pdf)
wide_staging = spark.createDataFrame(wide_staging_pdf)
wide_cmp = Compare(wide_prod, wide_staging, join_columns="id")
checked_columns = ["amount", "qty", "tax", "discount", "total", "currency"]

try:
    wide_cmp.assert_match(threshold=0.95, column=checked_columns)
except MismatchThresholdError as e:
    print(f"Columns below {e.threshold:.2f}: {e.failures}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Auto-derived threshold (`threshold=None`)
# MAGIC
# MAGIC No fixed bar needed — flags columns whose match rate is a statistical outlier
# MAGIC (`mean - 2*stdev`) relative to the other columns being checked. Needs a handful of
# MAGIC columns to be meaningful: with only 2–3 columns, 2 standard deviations rarely
# MAGIC separates a single outlier from the rest.

# COMMAND ----------

try:
    wide_cmp.assert_match(threshold=None, column=checked_columns)
except MismatchThresholdError as e:
    print(f"Auto threshold={e.threshold:.4f}; outlier column(s): {[c for c, _ in e.failures]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. report() — everything bundled into one object
# MAGIC
# MAGIC `str(report)` is a plain-text report; a bare `report` in a Databricks/Jupyter cell
# MAGIC auto-renders as an HTML table via `report._repr_html_()`; `report.to_dict()` gives a
# MAGIC plain dict (samples excluded) for logging/JSON.

# COMMAND ----------

report = cmp.report(top_n_columns=5, sample_rows_per_column=5)
print(str(report))

# COMMAND ----------

# DBTITLE 1,Auto-rendered HTML table
report  # noqa: B018 -- bare trailing expression is Databricks/Jupyter's auto-render idiom

# COMMAND ----------

# DBTITLE 1,...or force HTML rendering explicitly
displayHTML(report.to_html())

# COMMAND ----------

report.to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Numeric tolerance — `abs_tol`/`rel_tol`, per-column `tolerances`
# MAGIC
# MAGIC By default a "match" requires exact equality, so floating-point noise (a cast, a
# MAGIC rounding difference) shows up as a false-positive mismatch. `abs_tol`/`rel_tol` let
# MAGIC numeric columns match within `abs_tol + rel_tol * max(|df1_val|, |df2_val|)` instead —
# MAGIC symmetric, so swapping df1/df2 gives the same result. `tolerances` overrides the
# MAGIC global bar for specific columns. Non-numeric columns always require exact equality.

# COMMAND ----------

tol_prod_pdf = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "revenue": [1000.00, 2500.00, 375.50, 9999.99],
        "price": [19.99, 49.50, 12.00, 100.00],
    }
)
tol_staging_pdf = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        # id 1/2/3: ordinary floating-point noise; id 4: a genuinely different value
        "revenue": [1000.0000004, 2500.01, 375.50, 10500.00],
        # id 3: a small drift that revenue-level tolerance would hide, but matters a lot
        # for a per-unit price (it multiplies at scale)
        "price": [19.99, 49.50, 12.015, 100.00],
    }
)
tol_prod = spark.createDataFrame(tol_prod_pdf)
tol_staging = spark.createDataFrame(tol_staging_pdf)

cmp_exact = Compare(tol_prod, tol_staging, join_columns="id")
print("No tolerance (exact equality):")
for cs in cmp_exact.column_summary():
    print(f"  {cs.column}: {cs.match_count} match / {cs.mismatch_count} mismatch")

# COMMAND ----------

# DBTITLE 1,abs_tol absorbs noise; tolerances tightens price beyond the global bar
cmp_tol = Compare(
    tol_prod,
    tol_staging,
    join_columns="id",
    abs_tol=0.02,
    tolerances={"price": (0.001, 0.0)},
)
for cs in cmp_tol.column_summary():
    print(f"  {cs.column}: {cs.match_count} match / {cs.mismatch_count} mismatch")

# COMMAND ----------

# DBTITLE 1,report() documents the applied tolerance
cmp_tol.report()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Null counts + `validate()` — a one-call pipeline gate
# MAGIC
# MAGIC `column_summary()` entries include `null_count_df1`/`null_count_df2` (and
# MAGIC `null_pct_df1`/`null_pct_df2`). `validate()` bundles schema drift, row-count
# MAGIC issues, the column match-rate check, and a null-rate regression check into one
# MAGIC call — unlike `assert_match`, it never raises; it just returns the (possibly
# MAGIC empty) list of what failed, so a pipeline can log/branch on it directly.

# COMMAND ----------

val_prod_pdf = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "amount": [10, 20, 30, 40, 50],
        "region": ["us", "us", "eu", "eu", "eu"],
        "notes": [None, "ok", None, "ok", "ok"],
    }
)
val_staging_pdf = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 6],  # id 5 only in prod, id 6 only in staging
        "amount": [10, 20, 30, 999, 50],  # id 4's amount broke
        "region": ["us", "us", "eu", "eu", "eu"],
        "notes": [None, None, None, "ok", "ok"],  # notes got MORE null on staging
    }
)
val_prod = spark.createDataFrame(val_prod_pdf)
val_staging = spark.createDataFrame(val_staging_pdf)

val_cmp = Compare(val_prod, val_staging, join_columns="id", df1_name="prod", df2_name="staging")
for cs in val_cmp.column_summary():
    print(f"  {cs.column}: null% {cs.null_pct_df1:.1f} -> {cs.null_pct_df2:.1f}")

# COMMAND ----------

# DBTITLE 1,validate() returns only the failing checks
failures = val_cmp.validate(threshold=0.9, column=["amount", "region", "notes"])
for f in failures:
    print(f"  [{f.check}] {f.detail}")

# COMMAND ----------

# DBTITLE 1,scoping and tuning: skip schema/row checks, allow a bigger null swing
lenient = val_cmp.validate(schema_check=False, row_check=False, column=[], null_tolerance=50.0)
lenient  # noqa: B018 -- bare trailing expression is Databricks/Jupyter's auto-render idiom

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Composite (multi-column) join keys, `columns_to_compare`, `ignore_columns`

# COMMAND ----------

composite_pdf1 = pd.DataFrame(
    {
        "region": ["us", "us", "eu", "eu"],
        "sku": ["A", "B", "A", "B"],
        "amount": [100, 200, 300, 400],
        "qty": [1, 2, 3, 4],
        "name": ["x", "y", "z", "w"],
    }
)
composite_pdf2 = pd.DataFrame(
    {
        "region": ["us", "us", "eu", "eu"],
        "sku": ["A", "B", "A", "B"],
        "amount": [100, 250, 300, 400],  # mismatch: us/B
        "qty": [1, 2, 3, 9],  # mismatch: eu/B
        "name": ["x", "y", "Z", "w"],  # mismatch: eu/A
    }
)
composite_df1 = spark.createDataFrame(composite_pdf1)
composite_df2 = spark.createDataFrame(composite_pdf2)

# join_columns takes a list for a composite key; every shared, non-key column is
# compared automatically (amount, qty, name here)
cmp_composite = Compare(composite_df1, composite_df2, join_columns=["region", "sku"])
cmp_composite.column_summary()

# COMMAND ----------

# DBTITLE 1,columns_to_compare — restrict to specific columns
cmp_subset = Compare(
    composite_df1,
    composite_df2,
    join_columns=["region", "sku"],
    columns_to_compare=["amount", "qty"],
)
[c.column for c in cmp_subset.column_summary()]

# COMMAND ----------

# DBTITLE 1,ignore_columns — compare everything except a few
cmp_ignore = Compare(
    composite_df1,
    composite_df2,
    join_columns=["region", "sku"],
    ignore_columns=["name"],
)
[c.column for c in cmp_ignore.column_summary()]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. `compare()` — functional shortcut
# MAGIC
# MAGIC Equivalent to `Compare(df1, df2, join_columns, **kwargs).report()` in one call.

# COMMAND ----------

shortcut_report = compare(prod_df, staging_df, join_columns="id")
shortcut_report.row_summary  # noqa: B018 -- bare trailing expression is Databricks/Jupyter's auto-render idiom

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Exceptions
# MAGIC
# MAGIC `JoinKeyError` and the base `JuxtapyError` (also raised for e.g. an unknown
# MAGIC `columns_to_compare` entry, or mixing a pandas and a PySpark DataFrame in one `Compare`).

# COMMAND ----------

try:
    Compare(prod_df, staging_df, join_columns="nonexistent_col")
except JoinKeyError as e:
    print("JoinKeyError:", e)

try:
    Compare(prod_df, staging_df, join_columns="id", columns_to_compare=["not_a_column"])
except JuxtapyError as e:
    print("JuxtapyError:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference
# MAGIC
# MAGIC - Source & full API docs: https://github.com/cloudydataengineer/juxtapy
# MAGIC - `pip install juxtapy` (pandas only) or `pip install juxtapy[spark]` (pandas + PySpark, off-Databricks)
