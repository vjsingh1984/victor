# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Error tracking and metrics collection for observability.

This module provides:
- Error occurrence tracking
- Error rate calculations
- Error summary statistics
- Metrics export functionality
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""

    timestamp: datetime
    error_type: str
    error_message: str
    correlation_id: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "correlation_id": self.correlation_id,
            "context": self.context,
        }


class ErrorTracker:
    """Track errors for observability and monitoring.

    Thread-safe error tracking with:
    - Error occurrence recording
    - Error rate calculation (errors per hour)
    - Summary statistics
    - Recent error history
    - Metrics export

    Example:
        ```python
        tracker = get_error_tracker()
        tracker.record_error(
            error_type="ProviderNotFoundError",
            error_message="Provider 'xyz' not found",
            correlation_id="abc12345",
            context={"provider": "xyz"}
        )

        summary = tracker.get_error_summary()
        print(f"Total errors: {summary['total_errors']}")
        ```
    """

    def __init__(self, max_history: int = 1000):
        """Initialize error tracker.

        Args:
            max_history: Maximum number of error records to keep in memory
        """
        self._errors: list[ErrorRecord] = []
        self._error_counts: dict[str, int] = defaultdict(int)
        self._error_rates: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._max_history = max_history

    def record_error(
        self,
        error_type: str,
        error_message: str,
        correlation_id: str,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record an error occurrence.

        Args:
            error_type: Type/class name of the error
            error_message: Error message
            correlation_id: Unique correlation ID for this error
            context: Additional context about the error
        """
        with self._lock:
            timestamp = datetime.now(timezone.utc)

            error_record = ErrorRecord(
                timestamp=timestamp,
                error_type=error_type,
                error_message=error_message,
                correlation_id=correlation_id,
                context=context or {},
            )

            self._errors.append(error_record)
            self._error_counts[error_type] += 1

            # Trim history if needed
            if len(self._errors) > self._max_history:
                # Remove oldest records
                excess = len(self._errors) - self._max_history
                self._errors = self._errors[excess:]

            # Update error rates
            self._update_error_rates(error_type, timestamp)

    def _update_error_rates(self, error_type: str, timestamp: datetime) -> None:
        """Update error rate calculations (errors per hour).

        Args:
            error_type: Type of error
            timestamp: Current timestamp
        """
        # Get errors from last hour
        cutoff = timestamp - timedelta(hours=1)
        recent_errors = [
            e for e in self._errors if e.error_type == error_type and e.timestamp > cutoff
        ]

        error_rate = len(recent_errors)  # Errors per hour
        self._error_rates[error_type].append(error_rate)

        # Keep only last 24 data points
        if len(self._error_rates[error_type]) > 24:
            self._error_rates[error_type].pop(0)

    def get_error_summary(self) -> dict[str, Any]:
        """Get error summary statistics.

        Returns:
            Dictionary with summary statistics:
            - total_errors: Total number of errors recorded
            - error_counts: Dictionary of error type counts
            - error_types: List of unique error types
            - most_common: List of (error_type, count) tuples sorted by count
            - recent_errors: List of recent error records (last 100)
        """
        with self._lock:
            return {
                "total_errors": len(self._errors),
                "error_counts": dict(self._error_counts),
                "error_types": list(self._error_counts.keys()),
                "most_common": sorted(self._error_counts.items(), key=lambda x: x[1], reverse=True),
                "recent_errors": [e.to_dict() for e in self._errors[-100:]],
            }

    def get_error_rate(self, error_type: str) -> float:
        """Get current error rate (errors per hour).

        Args:
            error_type: Type of error

        Returns:
            Error rate (errors per hour) or 0.0 if no data
        """
        with self._lock:
            if self._error_rates[error_type]:
                return self._error_rates[error_type][-1]
            return 0.0

    def get_errors_by_type(self, error_type: str) -> list[ErrorRecord]:
        """Get all errors of a specific type.

        Args:
            error_type: Type of error

        Returns:
            List of error records
        """
        with self._lock:
            return [e for e in self._errors if e.error_type == error_type]

    def get_errors_by_timeframe(self, hours: int = 24) -> list[ErrorRecord]:
        """Get errors from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of error records
        """
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [e for e in self._errors if e.timestamp > cutoff]

    def export_metrics(self, filepath: str) -> None:
        """Export error metrics to JSON file.

        Args:
            filepath: Path to output JSON file
        """
        summary = self.get_error_summary()

        # Add error rates
        error_rates = {
            error_type: self.get_error_rate(error_type) for error_type in summary["error_types"]
        }

        metrics = {
            "summary": summary,
            "error_rates": error_rates,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)

    def clear(self) -> None:
        """Clear all error records."""
        with self._lock:
            self._errors.clear()
            self._error_counts.clear()
            self._error_rates.clear()


# Global singleton
_error_tracker: Optional[ErrorTracker] = None
_lock = threading.Lock()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance.

    Returns:
        Global ErrorTracker instance
    """
    global _error_tracker
    with _lock:
        if _error_tracker is None:
            _error_tracker = ErrorTracker()
        return _error_tracker


def reset_error_tracker() -> None:
    """Reset the global error tracker (useful for testing)."""
    global _error_tracker
    with _lock:
        _error_tracker = None
