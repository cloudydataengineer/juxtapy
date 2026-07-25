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
