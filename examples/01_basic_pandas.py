"""Basic single-key comparison with pandas.

Run: python examples/01_basic_pandas.py
"""

import pandas as pd

from juxtapy import Compare

prod = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "amount": [10, 20, 30, 40],
        "name": ["a", "b", "c", "d"],
    }
)
staging = pd.DataFrame(
    {
        "id": [1, 2, 3, 6],
        "amount": [10, 25, 30, 60],
        "name": ["a", "b", "C", "f"],
    }
)

cmp = Compare(prod, staging, join_columns="id", df1_name="prod", df2_name="staging")

print(cmp.row_summary())
print()
for col_summary in cmp.column_summary():
    print(col_summary)
print()
print("amount mismatches:")
print(cmp.sample_mismatches("amount"))
print()
print(cmp.report())
