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

"""Tests for the centralized pattern matching utility."""

import pytest

from victor.core.events.pattern_matcher import (
    matches_topic_pattern,
    clear_pattern_cache,
)


class TestMatchesTopicPattern:
    """Tests for matches_topic_pattern function."""

    def test_exact_match(self):
        """Test exact topic matching."""
        assert matches_topic_pattern("tool.call", "tool.call") is True
        assert matches_topic_pattern("agent.message", "agent.message") is True
        assert matches_topic_pattern("tool.call", "tool.result") is False

    def test_universal_wildcard(self):
        """Test universal wildcard '*' matches everything."""
        assert matches_topic_pattern("anything", "*") is True
        assert matches_topic_pattern("tool.call", "*") is True
        assert matches_topic_pattern("agent.message.deep", "*") is True
        assert matches_topic_pattern("", "*") is True

    def test_single_segment_wildcard(self):
        """Test single-segment wildcard 'tool.*' patterns."""
        # Should match same segment count
        assert matches_topic_pattern("tool.call", "tool.*") is True
        assert matches_topic_pattern("tool.result", "tool.*") is True
        assert matches_topic_pattern("tool.error", "tool.*") is True

        # Should NOT match different segment count
        assert matches_topic_pattern("tool.call.sub", "tool.*") is False
        assert matches_topic_pattern("tool", "tool.*") is False

        # Should NOT match different prefix
        assert matches_topic_pattern("agent.message", "tool.*") is False

    def test_leading_wildcard(self):
        """Test leading wildcard patterns."""
        assert matches_topic_pattern("tool.error", "*.error") is True
        assert matches_topic_pattern("agent.error", "*.error") is True
        assert matches_topic_pattern("system.error", "*.error") is True

        # Should NOT match different segment count
        assert matches_topic_pattern("tool.call.error", "*.error") is False
        assert matches_topic_pattern("error", "*.error") is False

    def test_multiple_segments_with_wildcards(self):
        """Test patterns with multiple wildcards."""
        assert matches_topic_pattern("tool.call.success", "tool.*.*") is True
        assert matches_topic_pattern("tool.result.error", "tool.*.*") is True
        assert matches_topic_pattern("tool.call", "tool.*.*") is False
        assert matches_topic_pattern("tool.call.sub.extra", "tool.*.*") is False

    def test_multi_segment_trailing_wildcard(self):
        """Test trailing '**' wildcard matches zero or more segments."""
        # Matches with extra segments
        assert matches_topic_pattern("tool.call.sub", "tool.**") is True
        assert matches_topic_pattern("tool.call.sub.deep", "tool.**") is True
        assert matches_topic_pattern("tool.call.sub.deep.deeper", "tool.**") is True

        # Matches exact (zero extra segments)
        assert matches_topic_pattern("tool.call", "tool.**") is True
        assert matches_topic_pattern("tool", "tool.**") is True

        # Should NOT match different prefix
        assert matches_topic_pattern("agent.call.sub", "tool.**") is False
        assert matches_topic_pattern("sub.tool.call", "tool.**") is False

    def test_wildcard_in_middle(self):
        """Test wildcard in middle of pattern."""
        assert matches_topic_pattern("tool.call.success", "tool.*.success") is True
        assert matches_topic_pattern("tool.result.success", "tool.*.success") is True
        assert matches_topic_pattern("tool.call.error", "tool.*.success") is False

    def test_complex_patterns(self):
        """Test complex multi-level patterns."""
        # Multi-level with single wildcards
        assert matches_topic_pattern("agent.tool.call.success", "agent.*.*.success") is True
        assert matches_topic_pattern("agent.tool.call.error", "agent.*.*.error") is True

        # Mismatched segment counts
        assert matches_topic_pattern("agent.tool.call", "agent.*.*.*") is False
        assert matches_topic_pattern("agent.tool.call.extra", "agent.*.*") is False

    def test_empty_strings(self):
        """Test handling of empty strings."""
        # Empty pattern only matches empty topic
        assert matches_topic_pattern("", "") is True
        assert matches_topic_pattern("tool.call", "") is False
        assert matches_topic_pattern("", "tool.*") is False

    def test_dots_in_topics(self):
        """Test topics with multiple dots."""
        assert matches_topic_pattern("a.b.c.d.e", "a.*.c.*.e") is True
        assert matches_topic_pattern("a.b.c.d.e", "a.*.*.*.e") is True
        assert matches_topic_pattern("a.b.c.d.e", "a.**") is True
        assert matches_topic_pattern("a.b.c.d.e", "*.b.*.d.*") is True


class TestPatternMatcherCache:
    """Tests for pattern matching cache."""

    def test_clear_cache(self):
        """Test clearing the pattern cache."""
        from victor.core.events.pattern_matcher import _split_pattern, _split_topic

        # Generate some cache entries
        _split_pattern("tool.*")
        _split_topic("tool.call")

        # Clear cache
        clear_pattern_cache()

        # Verify cache is cleared (cache_info should show 0 hits)
        assert _split_pattern.cache_info().currsize == 0
        assert _split_topic.cache_info().currsize == 0

    def test_cache_performance(self):
        """Test that caching improves performance."""
        import time

        # First call (cache miss)
        start = time.perf_counter()
        matches_topic_pattern("tool.call.sub.deep.here", "tool.**")
        first_duration = time.perf_counter() - start

        # Second call (cache hit - should be faster)
        start = time.perf_counter()
        matches_topic_pattern("tool.call.sub.deep.here", "tool.**")
        second_duration = time.perf_counter() - start

        # Cached call should be faster (or at least not significantly slower)
        # Note: This is a weak test due to timing variability, but demonstrates the concept
        assert second_duration <= first_duration * 10  # Allow 10x tolerance


class TestMessagingEventIntegration:
    """Test integration with MessagingEvent.matches_pattern()."""

    def test_event_matches_pattern(self):
        """Test that MessagingEvent uses the shared pattern matcher."""
        from victor.core.events.protocols import MessagingEvent

        event = MessagingEvent(topic="tool.call", data={})

        assert event.matches_pattern("tool.*") is True
        assert event.matches_pattern("agent.*") is False
        assert event.matches_pattern("*") is True
        assert event.matches_pattern("tool.**") is True

    def test_event_multi_segment_topic(self):
        """Test event with multi-segment topic."""
        from victor.core.events.protocols import MessagingEvent

        event = MessagingEvent(topic="tool.call.sub", data={})

        assert event.matches_pattern("tool.*") is False  # Different segment count
        assert event.matches_pattern("tool.**") is True  # Multi-segment wildcard
        assert event.matches_pattern("tool.*.sub") is True
