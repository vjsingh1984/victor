"""Tests for query result caching module."""

import time
from unittest.mock import patch

import pytest

from victor.tools.query_cache import (
    CacheEntry,
    QueryCache,
    cache_key_from_args,
    cached_query,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(value="test_result")
        assert entry.value == "test_result"
        assert entry.is_valid()
        assert entry.hit_count == 0

    def test_cache_entry_expiration(self):
        """Test cache entry expiration with TTL."""
        # Create entry with 1ms TTL
        entry = CacheEntry(
            value="test_result",
            ttl=__import__("datetime").timedelta(milliseconds=1)
        )
        assert entry.is_valid()

        # Wait for expiration
        time.sleep(0.01)
        assert not entry.is_valid()

    def test_cache_entry_hit_recording(self):
        """Test hit count tracking."""
        entry = CacheEntry(value="test_result")
        assert entry.hit_count == 0

        entry.record_hit()
        assert entry.hit_count == 1

        entry.record_hit()
        entry.record_hit()
        assert entry.hit_count == 3

    def test_cache_entry_tags(self):
        """Test tag-based invalidation."""
        entry = CacheEntry(
            value="test_result",
            tags={"tool:foo", "category:read"}
        )

        # Should invalidate with matching tags
        assert entry.should_invalidate({"tool:foo"})
        assert entry.should_invalidate({"category:read"})

        # Should not invalidate with non-matching tags
        assert not entry.should_invalidate({"tool:bar"})
        assert not entry.should_invalidate({"category:write"})

        # Should invalidate with any matching tag
        assert entry.should_invalidate({"tool:bar", "tool:foo"})


class TestQueryCache:
    """Tests for QueryCache class."""

    def test_cache_miss_then_hit(self):
        """Test cache miss on first call, hit on subsequent."""
        cache = QueryCache()
        compute_count = [0]

        def compute_fn():
            compute_count[0] += 1
            return "result"

        # First call - miss
        result1 = cache.get("test_key", compute_fn)
        assert result1 == "result"
        assert compute_count[0] == 1

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Second call - hit
        result2 = cache.get("test_key", compute_fn)
        assert result2 == "result"
        assert compute_count[0] == 1  # Not called again

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 1

    def test_cache_entry_expiration(self):
        """Test that expired entries trigger recomputation."""
        cache = QueryCache(default_ttl_seconds=0.001)  # 1ms TTL
        compute_count = [0]

        def compute_fn():
            compute_count[0] += 1
            return f"result_v{compute_count[0]}"

        # First call
        result1 = cache.get("test_key", compute_fn)
        assert result1 == "result_v1"
        assert compute_count[0] == 1

        # Wait for expiration
        time.sleep(0.01)

        # Second call should recompute
        result2 = cache.get("test_key", compute_fn)
        assert result2 == "result_v2"
        assert compute_count[0] == 2

    def test_custom_ttl(self):
        """Test custom TTL per entry."""
        cache = QueryCache()
        compute_count = [0]

        def compute_fn():
            compute_count[0] += 1
            return "result"

        # Cache with 1ms TTL
        from datetime import timedelta
        cache.get(
            "test_key",
            compute_fn,
            ttl=timedelta(milliseconds=1)
        )
        assert compute_count[0] == 1

        # Wait for expiration
        time.sleep(0.01)

        # Should recompute
        cache.get("test_key", compute_fn, ttl=timedelta(milliseconds=1))
        assert compute_count[0] == 2

    def test_invalidate_specific_key(self):
        """Test invalidating a specific cache entry."""
        cache = QueryCache()

        cache.get("key1", lambda: "value1")
        cache.get("key2", lambda: "value2")

        stats = cache.get_stats()
        assert stats["cache_size"] == 2

        # Invalidate specific key
        cache.invalidate("key1")

        stats = cache.get_stats()
        assert stats["cache_size"] == 1

    def test_invalidate_all_entries(self):
        """Test invalidating all cache entries."""
        cache = QueryCache()

        cache.get("key1", lambda: "value1")
        cache.get("key2", lambda: "value2")
        cache.get("key3", lambda: "value3")

        stats = cache.get_stats()
        assert stats["cache_size"] == 3

        # Invalidate all
        cache.invalidate()

        stats = cache.get_stats()
        assert stats["cache_size"] == 0

    def test_tag_based_invalidation(self):
        """Test tag-based selective invalidation."""
        cache = QueryCache()

        # Cache entries with different tags
        cache.get("key1", lambda: "value1", tags={"tool:foo"})
        cache.get("key2", lambda: "value2", tags={"tool:bar"})
        cache.get("key3", lambda: "value3", tags={"category:read"})

        stats = cache.get_stats()
        assert stats["cache_size"] == 3

        # Invalidate entries with specific tool tag
        count = cache.invalidate_tags({"tool:foo"})
        assert count == 1

        stats = cache.get_stats()
        assert stats["cache_size"] == 2

        # Invalidate entries with category tag
        count = cache.invalidate_tags({"category:read"})
        assert count == 1

        stats = cache.get_stats()
        assert stats["cache_size"] == 1

        # Invalidate with multiple tags
        count = cache.invalidate_tags({"tool:foo", "tool:bar"})
        assert count == 1  # Only tool:bar remains

        stats = cache.get_stats()
        assert stats["cache_size"] == 0

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = QueryCache()

        # Perform some operations
        for i in range(10):
            cache.get(f"key{i}", lambda: "value")

        # Hit from cache
        cache.get("key5", lambda: "value")
        cache.get("key5", lambda: "value")
        cache.get("key5", lambda: "value")

        stats = cache.get_stats()
        assert stats["cache_size"] == 10
        assert stats["misses"] == 10
        assert stats["hits"] == 3  # 3 hits after first miss
        assert abs(stats["hit_rate"] - 0.23) < 0.01  # 3/13 (approx)

    def test_cache_cleanup(self):
        """Test automatic cache cleanup."""
        cache = QueryCache(max_size=5, cleanup_threshold=10)

        # Add entries with short TTL
        from datetime import timedelta
        for i in range(15):
            cache.get(
                f"key{i}",
                lambda: "value",
                ttl=timedelta(milliseconds=1)
            )

        # Wait for expiration
        time.sleep(0.01)

        # Try to add more - should trigger cleanup
        cache.get("new_key", lambda: "value")

        stats = cache.get_stats()
        # Should be under max_size after cleanup
        assert stats["cache_size"] <= cache._max_size

    def test_clear_cache(self):
        """Test clearing cache and resetting stats."""
        cache = QueryCache()

        cache.get("key1", lambda: "value1")
        cache.get("key2", lambda: "value2")

        stats_before = cache.get_stats()
        assert stats_before["total_checks"] == 2

        cache.clear()

        stats_after = cache.get_stats()
        assert stats_after["cache_size"] == 0
        assert stats_after["total_checks"] == 0


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_simple_args(self):
        """Test key generation with simple arguments."""
        key = cache_key_from_args("get_by_tag", "testing")
        assert key == "get_by_tag:testing"

    def test_kwargs(self):
        """Test key generation with keyword arguments."""
        key = cache_key_from_args("get_by_category", "read", limit=10)
        assert "get_by_category" in key
        assert "read" in key
        assert "limit=10" in key

    def test_complex_args(self):
        """Test key generation with complex arguments."""
        key = cache_key_from_args("get_tool", None)
        assert "get_tool" in key
        assert "None" in key

    def test_key_hashing(self):
        """Test that long keys are hashed."""
        long_string = "x" * 200
        key = cache_key_from_args("method", long_string)
        assert key.startswith("hash:")
        assert len(key) < 100


class TestCachedQueryDecorator:
    """Tests for @cached_query decorator."""

    def test_decorator_caches_results(self):
        """Test that decorator caches method results."""
        cache = QueryCache()
        call_count = [0]

        class TestClass:
            @cached_query(cache=lambda self: cache)
            def get_value(self, key: str) -> str:
                call_count[0] += 1
                return f"value_{key}"

        obj = TestClass()

        # First call - miss
        result1 = obj.get_value("test")
        assert result1 == "value_test"
        assert call_count[0] == 1

        # Second call - hit
        result2 = obj.get_value("test")
        assert result2 == "value_test"
        assert call_count[0] == 1  # Not called again

    def test_decorator_with_custom_key_fn(self):
        """Test decorator with custom key function."""
        cache = QueryCache()
        call_count = [0]

        def custom_key_fn(category: str, limit: int) -> str:
            return f"{category}:limit_{limit}"

        class TestClass:
            @cached_query(
                cache=lambda self: cache,
                key_fn=custom_key_fn
            )
            def get_by_category(self, category: str, limit: int) -> list:
                call_count[0] += 1
                return list(range(limit))

        obj = TestClass()

        # First call
        result1 = obj.get_by_category("test", 10)
        assert len(result1) == 10
        assert call_count[0] == 1

        # Second call with same args - hit
        result2 = obj.get_by_category("test", 10)
        assert len(result2) == 10
        assert call_count[0] == 1

        # Different args - miss
        result3 = obj.get_by_category("test", 20)
        assert len(result3) == 20
        assert call_count[0] == 2

    def test_decorator_with_tag_fn(self):
        """Test decorator with tag function for selective invalidation."""
        cache = QueryCache()
        call_count = [0]

        def tag_fn(tag: str) -> set:
            return {f"tag:{tag}"}

        class TestClass:
            @cached_query(
                cache=lambda self: cache,
                tag_fn=tag_fn
            )
            def get_by_tag(self, tag: str) -> list:
                call_count[0] += 1
                return [f"tool_{i}" for i in range(5)]

        obj = TestClass()

        # Cache results
        obj.get_by_tag("testing")
        obj.get_by_tag("production")
        assert call_count[0] == 2

        # Invalidate testing tag
        cache.invalidate_tags({"tag:testing"})

        # Should recompute testing but not production
        obj.get_by_tag("testing")
        obj.get_by_tag("production")

        assert call_count[0] == 3  # Only testing recomputed
