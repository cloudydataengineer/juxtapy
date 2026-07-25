# juxtapy

Row-by-row, column-by-column comparison of two tables — pandas or PySpark — ranked by mismatch, with drill-down into the rows actually causing the disagreement.

```bash
pip install juxtapy          # pandas only
pip install juxtapy[spark]   # pandas + PySpark
```

## Why juxtapy

Most table-diffing tools hand you either a full row-level diff or a flat report. `juxtapy` is built around a different workflow: join two tables on a key, see which **columns** disagree the most, and immediately look at **sample rows** for the worst offenders — without writing a separate query for each investigation.

- **Ranked, not just reported.** `column_summary()` sorts columns worst-mismatch-first, so the columns most likely to matter are first, not buried in an alphabetical or column-order list.
- **Drill-down is first-class.** `sample_mismatches("amount")` returns the key + both values for the actual disagreeing rows — the thing you need to root-cause *why* two tables disagree, not just *that* they do.
- **CI-ready.** `assert_match(threshold=0.99)` raises if the match rate drops below a bar, so a comparison can gate a pipeline or CI job directly.
- **Schema drift is explicit.** Columns only on one side, or with a changed dtype, show up in the report rather than being silently dropped.
- **Notebook-friendly.** `CompareReport` renders as a color-highlighted HTML table automatically in Jupyter/Databricks.
- **Same API for pandas and PySpark.** Pass either kind of dataframe; PySpark is an optional extra, not a hard dependency.

## Usage

```python
from juxtapy import Compare

cmp = Compare(prod_df, staging_df, join_columns="id", df1_name="prod", df2_name="staging")

cmp.row_summary()                  # only-in-1 / only-in-2 / common / duplicate-key counts
cmp.column_summary()               # ColumnSummary list, worst-mismatch-first
cmp.sample_mismatches("amount", n=5)   # key + both values, for the actual mismatched rows
cmp.schema_diff()                  # columns only on one side, or with a changed dtype
cmp.matches()                      # bool: do the tables fully agree?
cmp.assert_match(threshold=0.99)   # raise MismatchThresholdError below the bar — CI gate

print(cmp.report())                # everything above, as one readable report
cmp.report()                       # in Jupyter/Databricks: auto-renders as an HTML table
```

Functional shortcut for the common case:

```python
from juxtapy import compare

report = compare(prod_df, staging_df, join_columns="id")
```

### PySpark

Works the same way with `pyspark.sql.DataFrame` inputs — both tables must use the same backend:

```python
cmp = Compare(spark_df1, spark_df2, join_columns=["id"])
```

**On Databricks**, install with `%pip install juxtapy` (no `[spark]` extra) — Databricks Runtime already bundles a pinned PySpark, and installing a different version via the extra risks a conflict. `Compare` just takes DataFrames you already have (e.g. from `spark.table(...)`).

## Development

```bash
uv sync --extra spark          # install with dev tools + PySpark for the full test suite
uv run pytest -m "not spark"   # fast loop (pandas only)
uv run pytest                  # full suite, including PySpark (requires Java locally)
uv run ruff check src tests
```

## License

MIT
