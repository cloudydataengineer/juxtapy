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

## Examples

Runnable, self-contained scripts live in [`examples/`](examples/):

- [`01_basic_pandas.py`](examples/01_basic_pandas.py) — single join key, full walkthrough of the API
- [`02_composite_keys_and_column_selection.py`](examples/02_composite_keys_and_column_selection.py) — multi-column join keys, `columns_to_compare`/`ignore_columns`
- [`03_ci_threshold_gate.py`](examples/03_ci_threshold_gate.py) — `assert_match` as a pipeline/CI gate
- [`04_pyspark_databricks.py`](examples/04_pyspark_databricks.py) — the same API against PySpark DataFrames
- [`05_databricks_full_demo.py`](examples/05_databricks_full_demo.py) — every public function/class in one notebook, importable directly into Databricks

## API Reference

### `Compare(df1, df2, join_columns, df1_name="df1", df2_name="df2", columns_to_compare=None, ignore_columns=None, cast_column_names_lower=True)`

Joins `df1` and `df2` on `join_columns` and prepares the comparison. Both dataframes must be the same backend (pandas or PySpark). Raises `JuxtapyError` if backends differ, `JoinKeyError` if a join column is missing from either side.

- `join_columns` — a single column name or a list, for composite keys.
- `columns_to_compare` — restrict comparison to these shared, non-key columns (raises `JuxtapyError` if any aren't shared columns). Defaults to *all* shared, non-key columns.
- `ignore_columns` — compare every shared column except these.
- `cast_column_names_lower` — lowercases column names (and `join_columns`) on both sides before comparing, so case differences in headers don't cause spurious "only on one side" columns. Default `True`.

**Methods** (results are cached after first call):

| Method | Returns | Description |
| --- | --- | --- |
| `row_summary()` | `RowSummary` | Row counts: total per side, common, only-in-df1/df2, duplicate join keys. Warns if duplicate keys are found. |
| `column_summary()` | `list[ColumnSummary]` | Per-column match/mismatch counts and dtypes, sorted worst-mismatch-first. |
| `schema_diff()` | `SchemaDiff` | Columns only on one side, and columns present on both sides with a changed dtype. |
| `sample_mismatches(column, n=5)` | `pandas.DataFrame` | Join key(s) + both values, for up to `n` of the actual mismatched rows for `column`. Always a pandas DataFrame, even for PySpark inputs. Raises `JuxtapyError` if `column` wasn't compared. |
| `matches(ignore_extra_columns=False)` | `bool` | `True` if every compared column fully agrees (and, unless `ignore_extra_columns=True`, rows match on both sides too). |
| `assert_match(threshold=1.0, column=None)` | `None` | Raises `MismatchThresholdError` if a match rate falls below `threshold`. `column` may be a single column name, a **list** of column names, or `None` (overall/all-columns). With a list, every named column is checked and *all* failing columns are reported in one error, not just the first. Pass `threshold=None` explicitly to auto-derive the bar instead, as `mean(rates) - 2 * pstdev(rates)` across the checked columns — flags columns whose match rate is a statistical outlier relative to the others being checked (with a single checked column, this always passes, since stdev of one value is 0). Designed for CI gating. |
| `report(top_n_columns=5, sample_rows_per_column=5)` | `CompareReport` | Everything above bundled into one object — the worst `top_n_columns` mismatched columns each get up to `sample_rows_per_column` sample rows. |

### `compare(df1, df2, join_columns, **kwargs)`

Functional shortcut: `Compare(df1, df2, join_columns, **kwargs).report()`.

### Result types (`juxtapy.results`)

- **`RowSummary`** — `rows_df1`, `rows_df2`, `common_rows`, `only_in_df1`, `only_in_df2`, `duplicate_keys_df1`, `duplicate_keys_df2`, plus `.all_rows_match` (bool property).
- **`ColumnSummary`** — `column`, `match_count`, `mismatch_count`, `dtype1`, `dtype2`, plus `.total_compared`, `.match_pct`, `.mismatch_pct` (properties).
- **`SchemaDiff`** — `only_in_df1`, `only_in_df2`, `dtype_changes` (`{column: (dtype1, dtype2)}`), plus `.has_drift` (bool property).
- **`CompareReport`** — `df1_name`, `df2_name`, `join_columns`, `row_summary`, `column_summary`, `schema_diff`, `samples` (`{column: DataFrame}`). `str(report)` renders a plain-text report; `report.to_html()` / `report._repr_html_()` render an HTML table (auto-used by Jupyter/Databricks); `report.to_dict()` returns a plain dict (samples excluded) for logging/JSON.

### Exceptions (`juxtapy.exceptions`)

- **`JuxtapyError`** — base exception for all juxtapy errors.
- **`JoinKeyError`** — a join column is missing from one or both tables.
- **`SchemaError`** — the two tables' schemas are incompatible for comparison.
- **`MismatchThresholdError`** — raised by `assert_match`; carries `.threshold` (the applied bar, auto-derived or explicit) and `.failures` (a list of `(column, match_rate)` pairs for every check that fell short — `column` is `None` for the overall/pooled check). `.column` and `.match_rate` are populated as a convenience only when there was exactly one failure; for multiple failing columns, use `.failures`.

## Development

```bash
uv sync --extra spark          # install with dev tools + PySpark for the full test suite
uv run pytest -m "not spark"   # fast loop (pandas only)
uv run pytest                  # full suite, including PySpark (requires Java locally)
uv run ruff check src tests
```

## License

MIT
