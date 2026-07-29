"""Using validate() as a one-call pipeline/backfill gate.

Unlike assert_match, validate() never raises for a failed check — it returns a
(possibly empty) list of ValidationFailure, so a pipeline can log/branch on the
result directly. It bundles: schema drift, duplicate/only-in-one-side rows, the
same per-column match-rate check as assert_match, and a null-rate regression
check (flags a column whose null rate got worse, not better).

Run: python examples/07_pipeline_validate.py
"""

import pandas as pd

from juxtapy import Compare

prod = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "amount": [10, 20, 30, 40, 50],
        "region": ["us", "us", "eu", "eu", "eu"],
        "notes": [None, "ok", None, "ok", "ok"],
    }
)
staging = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 6],  # id 5 only in prod, id 6 only in staging
        "amount": [10, 20, 30, 999, 50],  # id 4's amount broke
        "region": ["us", "us", "eu", "eu", "eu"],
        "notes": [None, None, None, "ok", "ok"],  # notes got MORE null on staging
    }
)

cmp = Compare(prod, staging, join_columns="id", df1_name="prod", df2_name="staging")

# column=None (the default) would apply threshold to the *overall pooled* rate across
# all columns' cells (one combined number), same as assert_match(threshold=...). Naming
# the columns explicitly instead checks each one's own rate — region matches perfectly,
# but amount and notes both drop to 75%, below 0.9:
failures = cmp.validate(threshold=0.9, column=["amount", "region", "notes"])
if not failures:
    print("All checks passed — safe to proceed.")
else:
    print(f"{len(failures)} check(s) failed:")
    for f in failures:
        print(f"  [{f.check}] {f.detail}")

# Each ValidationFailure is just (check, detail) — easy to post to Slack, a PR
# comment, or a monitoring system without dumping the whole report.

print()
print("--- Scoping and tuning the checks ---")

# Only run the column match-rate + null-rate checks, skip schema/row checks:
column_only = cmp.validate(schema_check=False, row_check=False, column=["amount", "notes"])
print(f"column+null checks only: {[f.check for f in column_only]}")

# Tolerate a bigger null-rate swing before flagging it:
lenient_nulls = cmp.validate(schema_check=False, row_check=False, column=[], null_tolerance=50.0)
print(f"with null_tolerance=50: {[f.check for f in lenient_nulls]}")
