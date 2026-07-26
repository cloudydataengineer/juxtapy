from __future__ import annotations

from collections.abc import Sequence


class JuxtapyError(Exception):
    """Base exception for all juxtapy errors."""


class JoinKeyError(JuxtapyError):
    """Raised when join columns are missing or invalid on one or both tables."""


class SchemaError(JuxtapyError):
    """Raised when the two tables' schemas are incompatible for comparison."""


class MismatchThresholdError(JuxtapyError):
    """Raised by Compare.assert_match when one or more checked match rates fall below threshold.

    ``failures`` holds every ``(column, match_rate)`` pair that fell short — ``column`` is
    ``None`` for the overall (pooled) check. ``.column`` and ``.match_rate`` are populated as a
    convenience only when exactly one check failed; for multiple failures, use ``.failures``.
    """

    def __init__(
        self,
        message: str,
        threshold: float,
        failures: Sequence[tuple[str | None, float]],
    ) -> None:
        super().__init__(message)
        self.threshold = threshold
        self.failures = list(failures)
        self.column = self.failures[0][0] if len(self.failures) == 1 else None
        self.match_rate = self.failures[0][1] if len(self.failures) == 1 else None
