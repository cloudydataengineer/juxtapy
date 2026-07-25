from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataFrameAdapter(Protocol):
    """Backend-neutral wrapper around a single dataframe (pandas or Spark)."""

    @property
    def columns(self) -> list[str]: ...

    def row_count(self) -> int: ...

    def duplicate_key_count(self, keys: Sequence[str]) -> int: ...

    def dtype_of(self, column: str) -> str: ...

    def full_outer_join(self, other: DataFrameAdapter, keys: Sequence[str]) -> JoinedAdapter: ...


@runtime_checkable
class JoinedAdapter(Protocol):
    """Result of joining two DataFrameAdapters on their join keys."""

    def only_in_left_count(self) -> int: ...

    def only_in_right_count(self) -> int: ...

    def common_count(self) -> int: ...

    def compare_columns(self, columns: Sequence[str]) -> dict[str, tuple[int, int]]:
        """Return {column: (match_count, mismatch_count)} for all given columns in one pass."""
        ...

    def sample_mismatched_rows(self, column: str, keys: Sequence[str], n: int) -> pd.DataFrame: ...
