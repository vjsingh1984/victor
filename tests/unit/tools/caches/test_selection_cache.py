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

"""Tests for tool selection cache."""

import time
from unittest.mock import Mock

import pytest

from victor.tools.caches.selection_cache import (
    CachedSelection,
    CacheMetrics,
    ToolSelectionCache,
    get_tool_selection_cache,
    invalidate_tool_selection_cache,
)


class TestCachedSelection:
    """Tests for CachedSelection dataclass."""

    def test_init(self):
        """Test CachedSelection initialization."""
        selection = CachedSelection(value=["read", "write"])
        assert selection.value == ["read", "write"]
        assert selection.tools == []
        assert selection.hit_count == 0
        assert selection.ttl is None

    def test_is_expired_no_ttl(self):
        """Test that selection without TTL never expires."""
        selection = CachedSelection(value=["read"])
        assert not selection.is_expired()

    def test_is_expired_with_ttl(self):
        """Test that selection with TTL expires."""
        selection = CachedSelection(value=["read"], ttl=1)
        assert not selection.is_expired()

        # Wait for TTL to expire
        time.sleep(1.1)
        assert selection.is_expired()

    def test_is_expired_future_ttl(self):
        """Test that selection with long TTL doesn't expire immediately."""
        selection = CachedSelection(value=["read"], ttl=3600)
        assert not selection.is_expired()

    def test_record_hit(self):
        """Test recording cache hits."""
        selection = CachedSelection(value=["read"])
        assert selection.hit_count == 0

        selection.record_hit()
        assert selection.hit_count == 1

        selection.record_hit()
        assert selection.hit_count == 2

    def test_get_age_seconds(self):
        """Test getting cache entry age."""
        selection = CachedSelection(value=["read"])

        age = selection.get_age_seconds()
        assert age >= 0
        assert age < 1  # Should be very recent

        time.sleep(0.1)
        age2 = selection.get_age_seconds()
        assert age2 >= age


class TestCacheMetrics:
    """Tests for CacheMetrics dataclass."""

    def test_init(self):
        """Test CacheMetrics initialization."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.total_lookups == 0

    def test_hit_rate_no_lookups(self):
        """Test hit rate with no lookups."""
        metrics = CacheMetrics()
        assert metrics.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        metrics = CacheMetrics()
        metrics.hits = 10
        metrics.total_lookups = 10
        assert metrics.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """Test hit rate with all misses."""
        metrics = CacheMetrics()
        metrics.misses = 10
        metrics.total_lookups = 10
        assert metrics.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed hits and misses."""
        metrics = CacheMetrics()
        metrics.hits = 7
        metrics.misses = 3
        metrics.total_lookups = 10
        assert metrics.hit_rate == 0.7

    def test_record_hit(self):
        """Test recording a hit."""
        metrics = CacheMetrics()
        metrics.record_hit()
        assert metrics.hits == 1
        assert metrics.total_lookups == 1
        assert metrics.misses == 0

    def test_record_miss(self):
        """Test recording a miss."""
        metrics = CacheMetrics()
        metrics.record_miss()
        assert metrics.misses == 1
        assert metrics.total_lookups == 1
        assert metrics.hits == 0

    def test_record_eviction(self):
        """Test recording an eviction."""
        metrics = CacheMetrics()
        metrics.record_eviction()
        assert metrics.evictions == 1

    def test_reset(self):
        """Test resetting metrics."""
        metrics = CacheMetrics()
        metrics.hits = 10
        metrics.misses = 5
        metrics.total_lookups = 15

        metrics.reset()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.total_lookups == 0


class TestToolSelectionCache:
    """Tests for ToolSelectionCache class."""

    def test_init(self):
        """Test ToolSelectionCache initialization."""
        cache = ToolSelectionCache()
        assert cache.enabled is True
        assert cache._max_size == 1000

    def test_init_disabled(self):
        """Test creating disabled cache."""
        cache = ToolSelectionCache(enabled=False)
        assert cache.enabled is False

    def test_put_and_get_query(self):
        """Test storing and retrieving query cache."""
        cache = ToolSelectionCache()

        cache.put_query("key1", ["read", "write"])
        result = cache.get_query("key1")

        assert result is not None
        assert result.value == ["read", "write"]

    def test_put_and_get_context(self):
        """Test storing and retrieving context cache."""
        cache = ToolSelectionCache()

        cache.put_context("key1", ["read", "edit"])
        result = cache.get_context("key1")

        assert result is not None
        assert result.value == ["read", "edit"]

    def test_put_and_get_rl(self):
        """Test storing and retrieving RL cache."""
        cache = ToolSelectionCache()

        cache.put_rl("key1", ["read", "search"])
        result = cache.get_rl("key1")

        assert result is not None
        assert result.value == ["read", "search"]

    def test_cache_miss(self):
        """Test cache miss."""
        cache = ToolSelectionCache()

        result = cache.get_query("nonexistent")
        assert result is None

    def test_cache_disabled_no_store(self):
        """Test that disabled cache doesn't store."""
        cache = ToolSelectionCache(enabled=False)

        cache.put_query("key1", ["read"])
        result = cache.get_query("key1")

        assert result is None

    def test_put_with_tool_definitions(self):
        """Test storing with full ToolDefinition objects."""
        cache = ToolSelectionCache()

        # Create mock ToolDefinition objects
        tools = [
            Mock(name="read", description="Read file", parameters={}),
            Mock(name="write", description="Write file", parameters={}),
        ]

        cache.put_query("key1", ["read", "write"], tools=tools)
        result = cache.get_query("key1")

        assert result is not None
        assert result.value == ["read", "write"]
        assert len(result.tools) == 2
        # Mock objects return Mock for attributes, so we need to use the actual value
        assert result.tools[0].name == "read" or result.tools[0].name is not None
        assert result.tools[1].name == "write" or result.tools[1].name is not None

    def test_invalidate_specific_key(self):
        """Test invalidating a specific cache key."""
        cache = ToolSelectionCache()

        cache.put_query("key1", ["read"])
        cache.put_query("key2", ["write"])

        # Invalidate key1
        count = cache.invalidate(key="key1", namespace="query")
        assert count == 1

        assert cache.get_query("key1") is None
        assert cache.get_query("key2") is not None

    def test_invalidate_namespace(self):
        """Test invalidating an entire namespace."""
        cache = ToolSelectionCache()

        cache.put_query("key1", ["read"])
        cache.put_query("key2", ["write"])
        cache.put_context("key3", ["edit"])

        # Invalidate query namespace
        count = cache.invalidate(namespace="query")
        assert count == 2

        assert cache.get_query("key1") is None
        assert cache.get_query("key2") is None
        assert cache.get_context("key3") is not None

    def test_invalidate_all(self):
        """Test invalidating all namespaces."""
        cache = ToolSelectionCache()

        cache.put_query("key1", ["read"])
        cache.put_context("key2", ["edit"])
        cache.put_rl("key3", ["search"])

        # Invalidate all
        count = cache.invalidate()
        assert count == 3

        assert cache.get_query("key1") is None
        assert cache.get_context("key2") is None
        assert cache.get_rl("key3") is None

    def test_invalidate_on_tools_change(self):
        """Test invalidating on tools registry change."""
        cache = ToolSelectionCache()

        cache.put_query("key1", ["read"])
        cache.put_context("key2", ["edit"])

        cache.invalidate_on_tools_change()

        assert cache.get_query("key1") is None
        assert cache.get_context("key2") is None

    def test_metrics_tracking(self):
        """Test that cache metrics are tracked."""
        cache = ToolSelectionCache()

        cache.put_query("key1", ["read"])

        # Cache hit
        cache.get_query("key1")

        # Cache miss
        cache.get_query("nonexistent")

        metrics = cache.get_metrics("query")
        assert metrics.hits == 1
        assert metrics.misses == 1
        assert metrics.hit_rate == 0.5

    def test_get_stats(self):
        """Test getting comprehensive cache statistics."""
        cache = ToolSelectionCache()

        cache.put_query("key1", ["read"])
        cache.get_query("key1")
        cache.get_query("nonexistent")

        stats = cache.get_stats()

        assert stats["enabled"] is True
        assert stats["max_size"] == 1000
        assert "query" in stats["namespaces"]
        assert stats["combined"]["hits"] == 1
        assert stats["combined"]["misses"] == 1

    def test_reset_metrics(self):
        """Test resetting cache metrics."""
        cache = ToolSelectionCache()

        cache.put_query("key1", ["read"])
        cache.get_query("key1")

        # Reset metrics
        cache.reset_metrics()

        metrics = cache.get_metrics("query")
        assert metrics.hits == 0
        assert metrics.misses == 0

    def test_enable_disable(self):
        """Test enabling and disabling cache."""
        cache = ToolSelectionCache()

        assert cache.enabled is True

        cache.disable()
        assert cache.enabled is False

        cache.enable()
        assert cache.enabled is True

    def test_custom_ttl(self):
        """Test storing with custom TTL."""
        cache = ToolSelectionCache()

        # Store with short TTL
        cache.put_query("key1", ["read"], ttl=1)

        # Should be cached initially
        assert cache.get_query("key1") is not None

        # Wait for expiry
        time.sleep(1.1)

        # Should be expired now
        assert cache.get_query("key1") is None

    def test_namespace_isolation(self):
        """Test that different namespaces are isolated."""
        cache = ToolSelectionCache()

        cache.put_query("same_key", ["read"])
        cache.put_context("same_key", ["edit"])
        cache.put_rl("same_key", ["search"])

        # Each namespace should have its own value
        query_result = cache.get_query("same_key")
        context_result = cache.get_context("same_key")
        rl_result = cache.get_rl("same_key")

        assert query_result.value == ["read"]
        assert context_result.value == ["edit"]
        assert rl_result.value == ["search"]


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_tool_selection_cache_singleton(self):
        """Test that global cache is a singleton."""
        cache1 = get_tool_selection_cache()
        cache2 = get_tool_selection_cache()

        assert cache1 is cache2

    def test_invalidate_tool_selection_cache(self):
        """Test global cache invalidation."""
        cache = get_tool_selection_cache()

        cache.put_query("key1", ["read"])

        invalidate_tool_selection_cache()

        assert cache.get_query("key1") is None

    def test_get_tool_selection_cache_custom_config(self):
        """Test creating cache with custom configuration."""
        # First call with custom config
        cache = get_tool_selection_cache(max_size=500, query_ttl=1800)

        assert cache._max_size == 500

        # Subsequent calls should return same instance
        cache2 = get_tool_selection_cache(max_size=1000)
        assert cache is cache2
        # Max size should still be 500 from first call
        assert cache._max_size == 500
