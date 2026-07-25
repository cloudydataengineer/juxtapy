"""Composite (multi-column) join keys, plus narrowing which columns get compared.

Run: python examples/02_composite_keys_and_column_selection.py
"""

import pandas as pd

from juxtapy import Compare

prod = pd.DataFrame(
    {
        "region": ["us", "us", "eu", "eu"],
        "sku": ["A", "B", "A", "B"],
        "amount": [100, 200, 300, 400],
        "qty": [1, 2, 3, 4],
        "name": ["x", "y", "z", "w"],
    }
)
staging = pd.DataFrame(
    {
        "region": ["us", "us", "eu", "eu"],
        "sku": ["A", "B", "A", "B"],
        "amount": [100, 250, 300, 400],  # mismatch: us/B
        "qty": [1, 2, 3, 9],  # mismatch: eu/B
        "name": ["x", "y", "Z", "w"],  # mismatch: eu/A
    }
)

# join_columns takes a list for a composite key; every shared, non-key column
# (amount, qty, name) is compared automatically.
cmp = Compare(prod, staging, join_columns=["region", "sku"], df1_name="prod", df2_name="staging")

for col_summary in cmp.column_summary():
    print(col_summary)

# Restrict comparison to specific columns:
cmp_amount_only = Compare(
    prod,
    staging,
    join_columns=["region", "sku"],
    columns_to_compare=["amount", "qty"],
)
print("compared columns:", [c.column for c in cmp_amount_only.column_summary()])

# ...or compare everything except a few:
cmp_ignore_name = Compare(
    prod,
    staging,
    join_columns=["region", "sku"],
    ignore_columns=["name"],
)
print("compared columns:", [c.column for c in cmp_ignore_name.column_summary()])

# sample_mismatches always includes the full composite key alongside both values.
print()
print(cmp.sample_mismatches("amount"))
