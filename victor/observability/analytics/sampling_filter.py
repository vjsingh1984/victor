"""Semantic event sampling filter for usage logging.

Reduces disk I/O by filtering noise events (streaming content chunks,
duplicate progress updates) while preserving high-value decision
boundary events (tool calls, errors, session lifecycle).

Inserted at ``UsageLogger.log_event()`` before disk writes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Set


@dataclass
class SamplingPolicy:
    """Per-event-type sampling rules.

    Attributes:
        always_pass: Event types that are never filtered.
        sample_rate: For sampled event types, emit 1 in N events.
        dedup_window_seconds: Time window for deduplication of noisy events.
        sampled_types: Event types subject to 1-in-N sampling.
        deduped_types: Event types subject to time-window deduplication.
    """

    always_pass: Set[str] = field(default_factory=lambda: {
        "tool_call",
        "tool_result",
        "error",
        "recovery",
        "session_start",
        "session_end",
        "user_prompt",
        "assistant_response",
    })
    sample_rate: int = 10
    dedup_window_seconds: float = 5.0
    sampled_types: Set[str] = field(default_factory=lambda: {"content"})
    deduped_types: Set[str] = field(default_factory=lambda: {"progress", "milestone"})


class SemanticSamplingFilter:
    """Filters events before they hit disk I/O.

    Thread-safe for use in synchronous ``UsageLogger.log_event()`` calls.

    Args:
        policy: Sampling rules to apply.
    """

    def __init__(self, policy: SamplingPolicy | None = None) -> None:
        self._policy = policy or SamplingPolicy()
        # Counters for sampled types (event_type -> count seen)
        self._sample_counters: Dict[str, int] = {}
        # Last-emit timestamps for deduped types (event_type -> timestamp)
        self._dedup_timestamps: Dict[str, float] = {}
        # Stats
        self._events_passed: int = 0
        self._events_dropped: int = 0

    def should_emit(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Decide whether an event should be persisted to disk.

        Args:
            event_type: The event type string (e.g. "content", "tool_call").
            data: The event data dict (unused for now but available for
                future content-aware filtering).

        Returns:
            True if the event should be written to disk.
        """
        # High-value events always pass
        if event_type in self._policy.always_pass:
            self._events_passed += 1
            return True

        # 1-in-N sampling for noisy streaming events
        if event_type in self._policy.sampled_types:
            count = self._sample_counters.get(event_type, 0) + 1
            self._sample_counters[event_type] = count
            if count % self._policy.sample_rate == 0:
                self._events_passed += 1
                return True
            self._events_dropped += 1
            return False

        # Time-window deduplication for progress-like events
        if event_type in self._policy.deduped_types:
            now = time.monotonic()
            last = self._dedup_timestamps.get(event_type, 0.0)
            if now - last >= self._policy.dedup_window_seconds:
                self._dedup_timestamps[event_type] = now
                self._events_passed += 1
                return True
            self._events_dropped += 1
            return False

        # Unknown event types pass by default (safe fallback)
        self._events_passed += 1
        return True

    def get_stats(self) -> Dict[str, int]:
        """Return filter statistics."""
        return {
            "events_passed": self._events_passed,
            "events_dropped": self._events_dropped,
            "total_events": self._events_passed + self._events_dropped,
        }

    def reset(self) -> None:
        """Reset all counters and timestamps."""
        self._sample_counters.clear()
        self._dedup_timestamps.clear()
        self._events_passed = 0
        self._events_dropped = 0
