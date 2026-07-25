from __future__ import annotations


class JuxtapyError(Exception):
    """Base exception for all juxtapy errors."""


class JoinKeyError(JuxtapyError):
    """Raised when join columns are missing or invalid on one or both tables."""


class SchemaError(JuxtapyError):
    """Raised when the two tables' schemas are incompatible for comparison."""


class MismatchThresholdError(JuxtapyError):
    """Raised by Compare.assert_match when the match rate falls below the required threshold."""

    def __init__(
        self,
        message: str,
        match_rate: float,
        threshold: float,
        column: str | None = None,
    ) -> None:
        super().__init__(message)
        self.match_rate = match_rate
        self.threshold = threshold
        self.column = column
