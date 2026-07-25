# Examples

Each script is self-contained (builds its own sample data — no external files needed).

| Script | Demonstrates |
| --- | --- |
| [`01_basic_pandas.py`](01_basic_pandas.py) | Single join key, `row_summary`, `column_summary`, `sample_mismatches`, `report` |
| [`02_composite_keys_and_column_selection.py`](02_composite_keys_and_column_selection.py) | Multi-column join keys, `columns_to_compare`, `ignore_columns` |
| [`03_ci_threshold_gate.py`](03_ci_threshold_gate.py) | `assert_match` for gating a pipeline/CI job, `matches()` |
| [`04_pyspark_databricks.py`](04_pyspark_databricks.py) | Same API against PySpark DataFrames (Databricks-ready) |

Run the pandas-only ones directly:

```bash
uv run python examples/01_basic_pandas.py
uv run python examples/02_composite_keys_and_column_selection.py
uv run python examples/03_ci_threshold_gate.py
```

The PySpark example needs the `spark` extra and a local Java install:

```bash
uv run --extra spark python examples/04_pyspark_databricks.py
```

On Databricks, skip the extra (`%pip install juxtapy`) and paste the cells from that file directly into a notebook — `spark` is already provided.
