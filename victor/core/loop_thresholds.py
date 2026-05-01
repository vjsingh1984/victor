"""Shared default thresholds for blocked-loop recovery behavior."""

DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD = 4
"""Force blocked-loop recovery after this many consecutive blocked batches."""

DEFAULT_BLOCKED_TOTAL_THRESHOLD = 6
"""Force blocked-loop recovery after this many total blocked batches."""


def derive_blocked_total_threshold(consecutive_threshold: int) -> int:
    """Scale the total blocked threshold from the consecutive threshold.

    The default relationship is 4 consecutive blocked batches -> 6 total
    blocked batches. Custom loop thresholds preserve that ratio.
    """
    return int(
        consecutive_threshold
        * DEFAULT_BLOCKED_TOTAL_THRESHOLD
        / DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
    )
