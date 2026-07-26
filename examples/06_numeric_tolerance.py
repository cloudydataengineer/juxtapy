"""Numeric tolerance: abs_tol/rel_tol (global) and tolerances (per-column override).

Run: python examples/06_numeric_tolerance.py
"""

import pandas as pd

from juxtapy import Compare

prod = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "revenue": [1000.00, 2500.00, 375.50, 9999.99],
        "price": [19.99, 49.50, 12.00, 100.00],
        "region": ["us", "us", "eu", "eu"],
    }
)
staging = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        # id 1/2/3: ordinary floating-point noise from a downstream cast/rounding.
        # id 4: a genuinely different value — a real data issue, not noise.
        "revenue": [1000.0000004, 2500.01, 375.50, 10500.00],
        # id 3: a small drift that revenue-level tolerance would hide, but matters
        # a lot for a per-unit price (it multiplies at scale).
        "price": [19.99, 49.50, 12.015, 100.00],
        "region": ["us", "us", "eu", "eu"],
    }
)

# Without tolerance, every float noise row above shows up as a mismatch:
cmp_exact = Compare(prod, staging, join_columns="id")
print("No tolerance (exact equality):")
for cs in cmp_exact.column_summary():
    print(f"  {cs.column}: {cs.match_count} match / {cs.mismatch_count} mismatch")

# abs_tol=0.02 absorbs the revenue rounding noise but still catches id 4's real
# ~$500 discrepancy. tolerances overrides price to a much tighter bound, since a
# global bound loose enough for revenue would silently hide the price drift.
cmp_tol = Compare(
    prod,
    staging,
    join_columns="id",
    abs_tol=0.02,
    tolerances={"price": (0.001, 0.0)},
)
print("\nWith abs_tol=0.02, tolerances={'price': (0.001, 0.0)}:")
for cs in cmp_tol.column_summary():
    print(f"  {cs.column}: {cs.match_count} match / {cs.mismatch_count} mismatch")

print("\nrevenue mismatches (the real issue, not the noise):")
print(cmp_tol.sample_mismatches("revenue"))

print("\nprice mismatches (caught by the tighter per-column override):")
print(cmp_tol.sample_mismatches("price"))

# The report is self-documenting about the tolerance that was applied:
print()
print(cmp_tol.report())
