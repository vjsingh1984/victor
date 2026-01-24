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

"""Centralized topic pattern matching for event backends.

This module provides a single, optimized implementation of topic pattern
matching used by all event backends (Redis, Kafka, RabbitMQ, SQS, InMemory).

Pattern Syntax:
    - "*" matches everything
    - "tool.*" matches "tool.call", "tool.result" (single segment wildcard)
    - "*.error" matches "tool.error", "agent.error"
    - "tool.*.sub" matches "tool.call.sub", "tool.result.sub"
    - Exact match: "tool.call" only matches "tool.call"

The pattern matching uses segment-based comparison (dot-separated) rather
than character-based fnmatch, providing more predictable behavior for
hierarchical topic structures.

Example:
    from victor.core.events.pattern_matcher import matches_topic_pattern

    # Single-segment wildcards
    assert matches_topic_pattern("tool.call", "tool.*") == True
    assert matches_topic_pattern("tool.call.sub", "tool.*") == False  # Different segment count

    # Universal wildcard
    assert matches_topic_pattern("anything.here", "*") == True

    # Trailing wildcard (matches additional segments)
    assert matches_topic_pattern("tool.call.sub", "tool.**") == True
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple


def matches_topic_pattern(topic: str, pattern: str) -> bool:
    """Check if a topic matches a subscription pattern.

    This is the canonical pattern matching implementation used across all
    event backends. It uses segment-based matching (dot-separated) for
    predictable hierarchical topic matching.

    Args:
        topic: Event topic string (e.g., "tool.call", "agent.message")
        pattern: Subscription pattern with optional wildcards

    Returns:
        True if topic matches pattern

    Pattern Rules:
        - "*" matches any single segment OR everything if alone
        - "**" matches zero or more segments (only at end of pattern)
        - Exact segments must match exactly
        - Segment count must match unless trailing "**" is used

    Examples:
        >>> matches_topic_pattern("tool.call", "tool.*")
        True
        >>> matches_topic_pattern("tool.call", "tool.call")
        True
        >>> matches_topic_pattern("tool.call.sub", "tool.*")
        False  # Different segment count
        >>> matches_topic_pattern("tool.call.sub", "tool.**")
        True  # ** matches remaining segments
        >>> matches_topic_pattern("any.topic.here", "*")
        True  # Lone * matches everything
    """
    # Fast path: universal wildcard matches everything
    if pattern == "*":
        return True

    # Fast path: exact match
    if pattern == topic:
        return True

    # Split into segments for comparison
    return _matches_segments(topic, pattern)


@lru_cache(maxsize=1024)
def _split_pattern(pattern: str) -> Tuple[str, ...]:
    """Cache pattern splitting for performance."""
    return tuple(pattern.split("."))


@lru_cache(maxsize=4096)
def _split_topic(topic: str) -> Tuple[str, ...]:
    """Cache topic splitting for performance."""
    return tuple(topic.split("."))


def _matches_segments(topic: str, pattern: str) -> bool:
    """Segment-based pattern matching with caching.

    Args:
        topic: Event topic string
        pattern: Subscription pattern

    Returns:
        True if segments match according to pattern rules
    """
    pattern_parts = _split_pattern(pattern)
    topic_parts = _split_topic(topic)

    # Handle trailing "**" (matches zero or more remaining segments)
    if pattern_parts and pattern_parts[-1] == "**":
        prefix_parts = pattern_parts[:-1]
        # Must have at least as many topic segments as prefix
        if len(topic_parts) < len(prefix_parts):
            return False
        # Check prefix segments match
        return all(
            p == "*" or p == t for p, t in zip(prefix_parts, topic_parts[: len(prefix_parts)])
        )

    # Handle trailing "*" (matches exactly one more segment)
    if pattern_parts and pattern_parts[-1] == "*":
        # Segment counts must match
        if len(pattern_parts) != len(topic_parts):
            return False
        # Check all segments except last (which is wildcard)
        return all(p == "*" or p == t for p, t in zip(pattern_parts[:-1], topic_parts[:-1]))

    # Standard case: segment counts must match
    if len(pattern_parts) != len(topic_parts):
        return False

    # Check each segment
    return all(p == "*" or p == t for p, t in zip(pattern_parts, topic_parts))


def clear_pattern_cache() -> None:
    """Clear the pattern matching caches.

    Call this if patterns or topics are changed dynamically and you want
    to free memory. Normally not needed as caches are bounded.
    """
    _split_pattern.cache_clear()
    _split_topic.cache_clear()


__all__ = [
    "matches_topic_pattern",
    "clear_pattern_cache",
]
