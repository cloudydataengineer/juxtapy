"""Using assert_match to gate a CI job or pipeline on data quality.

Run: python examples/03_ci_threshold_gate.py
"""

import pandas as pd

from juxtapy import Compare, MismatchThresholdError

prod = pd.DataFrame({"id": [1, 2, 3, 4], "amount": [10, 20, 30, 40]})
staging = pd.DataFrame({"id": [1, 2, 3, 4], "amount": [10, 20, 30, 999]})

cmp = Compare(prod, staging, join_columns="id")

# Overall match rate across all compared columns must be >= threshold, or this raises.
try:
    cmp.assert_match(threshold=0.99)
except MismatchThresholdError as e:
    print(f"Gate failed: match_rate={e.match_rate:.4f} < threshold={e.threshold:.4f}")

# Gate a single column instead of the whole comparison:
try:
    cmp.assert_match(threshold=1.0, column="amount")
except MismatchThresholdError as e:
    print(f"Column gate failed on '{e.column}': match_rate={e.match_rate:.4f}")

# matches() is the non-raising equivalent, handy for an if-statement:
if not cmp.matches():
    print("Tables do not fully agree — see cmp.report() for details.")


# --- Gating a list of columns at once --------------------------------------

wide_prod = pd.DataFrame(
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
wide_staging = wide_prod.copy()
# each numeric column has one row of ordinary drift -> 90% match
for col in ["amount", "qty", "tax", "discount", "total"]:
    wide_staging.loc[0, col] = -1
# currency encoding broke entirely on the staging side -> 0% match
wide_staging["currency"] = "usd"

wide_cmp = Compare(wide_prod, wide_staging, join_columns="id")
checked_columns = ["amount", "qty", "tax", "discount", "total", "currency"]

# Check several columns in one call — raises once, listing every column that failed
# (not just the first), via e.failures:
try:
    wide_cmp.assert_match(threshold=0.95, column=checked_columns)
except MismatchThresholdError as e:
    print(f"Columns below {e.threshold:.2f}: {e.failures}")

# No threshold given -> auto-derive one as mean - 2*stdev across the checked columns'
# match rates. The 5 numeric columns cluster around ~90%; currency at 0% is a clear
# statistical outlier relative to them, so it's flagged even without a fixed bar.
# (This needs a handful of columns to be meaningful — with only 2-3 columns, 2 standard
# deviations rarely separates a single outlier from the rest.)
try:
    wide_cmp.assert_match(threshold=None, column=checked_columns)
except MismatchThresholdError as e:
    print(f"Auto threshold={e.threshold:.4f}; outlier column(s): {[c for c, _ in e.failures]}")
