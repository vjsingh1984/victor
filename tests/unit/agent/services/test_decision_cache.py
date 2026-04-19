"""Tests for multi-level decision cache.

Tests cover:
- L1 cache operations (get, put, invalidate, clear)
- L2 disk cache operations
- Cache promotion/demotion
- TTL-based expiration
- LRU eviction
- Cache statistics
"""

import tempfile
from pathlib import Path

import pytest

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.decision_cache import (
    CacheEntry,
    DecisionCache,
    DiskCache,
    LRUCache,
    create_decision_cache,
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            value="test_value",
            timestamp=123456.0,
            ttl=300.0,
        )

        assert entry.value == "test_value"
        assert entry.timestamp == 123456.0
        assert entry.access_count == 0
        assert entry.ttl == 300.0

    def test_is_expired_false(self):
        """Test expiration check for non-expired entry."""
        import time

        entry = CacheEntry(
            value="test",
            timestamp=time.monotonic(),
            ttl=300.0,
        )

        assert entry.is_expired() is False

    def test_is_expired_true(self):
        """Test expiration check for expired entry."""
        import time

        entry = CacheEntry(
            value="test",
            timestamp=time.monotonic() - 400.0,  # 400 seconds ago
            ttl=300.0,
        )

        assert entry.is_expired() is True

    def test_touch_updates_stats(self):
        """Test that touch updates access time and count."""
        entry = CacheEntry(
            value="test",
            timestamp=100.0,
            ttl=300.0,
        )

        initial_count = entry.access_count
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_access > 100.0


class TestLRUCache:
    """Test LRU cache operations."""

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_cache_get_nonexistent(self):
        """Test getting non-existent key."""
        cache = LRUCache(max_size=10)

        result = cache.get("nonexistent")

        assert result is None

    def test_cache_get_expired(self):
        """Test getting expired entry."""
        import time

        cache = LRUCache(max_size=10)

        cache.put("key1", "value1", ttl=1.0)
        time.sleep(1.1)  # Wait for expiration

        result = cache.get("key1")

        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Access key1 to make it more recently used
        cache.get("key1")

        # Add new entry, should evict key2 (oldest)
        cache.put("key4", "value4")

        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_invalidate(self):
        """Test cache invalidation."""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1")
        invalidated = cache.invalidate("key1")

        assert invalidated is True
        assert cache.get("key1") is None

    def test_cache_invalidate_nonexistent(self):
        """Test invalidating non-existent key."""
        cache = LRUCache(max_size=10)

        invalidated = cache.invalidate("nonexistent")

        assert invalidated is False

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache._cache) == 0

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        import time

        cache = LRUCache(max_size=10)

        cache.put("key1", "value1", ttl=1.0)
        cache.put("key2", "value2", ttl=10.0)

        time.sleep(1.1)

        removed = cache.cleanup_expired()

        assert removed == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 twice
        cache.get("key1")
        cache.get("key1")

        # Access key2 once
        cache.get("key2")

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["total_accesses"] == 3
        assert stats["avg_access_count"] == 1.5

    def test_custom_ttl(self):
        """Test custom TTL per entry."""
        import time

        cache = LRUCache(max_size=10, default_ttl=10.0)

        cache.put("key1", "value1", ttl=1.0)
        cache.put("key2", "value2")  # Uses default TTL

        time.sleep(1.1)

        assert cache.get("key1") is None  # Expired (custom TTL)
        assert cache.get("key2") == "value2"  # Not expired (default TTL)


class TestDiskCache:
    """Test disk cache operations."""

    def test_disk_cache_put_and_get(self):
        """Test basic put and get operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=Path(tmpdir), max_size=10)

            cache.put("key1", "value1")
            result = cache.get("key1")

            assert result == "value1"

    def test_disk_cache_get_nonexistent(self):
        """Test getting non-existent key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=Path(tmpdir), max_size=10)

            result = cache.get("nonexistent")

            assert result is None

    def test_disk_cache_expiration(self):
        """Test expiration of disk cache entries."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=Path(tmpdir), max_size=10)

            cache.put("key1", "value1", ttl=1.0)
            time.sleep(1.1)

            result = cache.get("key1")

            assert result is None

    def test_disk_cache_invalidate(self):
        """Test cache invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=Path(tmpdir), max_size=10)

            cache.put("key1", "value1")
            invalidated = cache.invalidate("key1")

            assert invalidated is True
            assert cache.get("key1") is None

    def test_disk_cache_clear(self):
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=Path(tmpdir), max_size=10)

            cache.put("key1", "value1")
            cache.put("key2", "value2")

            cache.clear()

            assert cache.get("key1") is None
            assert cache.get("key2") is None
            assert len(cache._index) == 0

    def test_disk_cache_persistence(self):
        """Test that cache persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create first cache and add entry
            cache1 = DiskCache(cache_dir=cache_dir, max_size=10)
            cache1.put("key1", "value1")

            # Create second cache and verify entry persists
            cache2 = DiskCache(cache_dir=cache_dir, max_size=10)
            result = cache2.get("key1")

            assert result == "value1"

    def test_disk_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(cache_dir=Path(tmpdir), max_size=10)

            cache.put("key1", "value1")
            cache.put("key2", "value2")

            # Access key1 twice
            cache.get("key1")
            cache.get("key1")

            stats = cache.get_stats()

            assert stats["size"] == 2
            assert stats["max_size"] == 10
            assert stats["total_accesses"] == 2


class TestDecisionCache:
    """Test multi-level decision cache."""

    def test_cache_initialization_l1_only(self):
        """Test cache initialization with L1 only."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        assert cache.l1 is not None
        assert cache.l2 is None
        assert cache.l2_enabled is False

    def test_cache_initialization_with_l2(self):
        """Test cache initialization with L2 enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DecisionCache(
                l1_size=100,
                l2_enabled=True,
                l2_cache_dir=Path(tmpdir),
            )

            assert cache.l1 is not None
            assert cache.l2 is not None
            assert cache.l2_enabled is True

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context = {"message": "test message"}
        cache.put(DecisionType.TASK_COMPLETION, context, "result")

        result = cache.get(DecisionType.TASK_COMPLETION, context)

        assert result == "result"

    def test_cache_miss(self):
        """Test cache miss."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context = {"message": "test message"}
        result = cache.get(DecisionType.TASK_COMPLETION, context)

        assert result is None

    def test_cache_hit_increments_l1_hits(self):
        """Test that L1 hit increments counter."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context = {"message": "test"}
        cache.put(DecisionType.TASK_COMPLETION, context, "result")

        cache.get(DecisionType.TASK_COMPLETION, context)

        assert cache._l1_hits == 1
        assert cache._misses == 0

    def test_cache_miss_increments_misses(self):
        """Test that cache miss increments counter."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context = {"message": "test"}
        cache.get(DecisionType.TASK_COMPLETION, context)

        assert cache._misses == 1
        assert cache._l1_hits == 0

    def test_cache_invalidate(self):
        """Test cache invalidation."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context = {"message": "test"}
        cache.put(DecisionType.TASK_COMPLETION, context, "result")

        invalidated = cache.invalidate(DecisionType.TASK_COMPLETION, context)

        assert invalidated is True
        assert cache.get(DecisionType.TASK_COMPLETION, context) is None

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context1 = {"message": "test1"}
        context2 = {"message": "test2"}

        cache.put(DecisionType.TASK_COMPLETION, context1, "result1")
        cache.put(DecisionType.INTENT_CLASSIFICATION, context2, "result2")

        cache.clear()

        assert cache.get(DecisionType.TASK_COMPLETION, context1) is None
        assert cache.get(DecisionType.INTENT_CLASSIFICATION, context2) is None

    def test_cache_cleanup(self):
        """Test cleanup of expired entries."""
        import time

        cache = DecisionCache(l1_size=100, l2_enabled=False)

        cache.put(
            DecisionType.TASK_COMPLETION,
            {"message": "test1"},
            "result1",
            ttl=1.0,
        )
        cache.put(
            DecisionType.TASK_COMPLETION,
            {"message": "test2"},
            "result2",
            ttl=10.0,
        )

        time.sleep(1.1)

        removed = cache.cleanup()

        assert removed == 1

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context = {"message": "test"}
        cache.put(DecisionType.TASK_COMPLETION, context, "result")

        # One hit
        cache.get(DecisionType.TASK_COMPLETION, context)

        # One miss
        cache.get(DecisionType.TASK_COMPLETION, {"message": "other"})

        stats = cache.get_stats()

        assert stats["l1_hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5

    def test_cache_key_generation(self):
        """Test that cache keys are generated consistently."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context = {"message": "test", "value": 42}

        key1 = cache._make_key(DecisionType.TASK_COMPLETION, context)
        key2 = cache._make_key(DecisionType.TASK_COMPLETION, context)

        assert key1 == key2

        # Different context should produce different key
        different_context = {"message": "test", "value": 43}
        key3 = cache._make_key(DecisionType.TASK_COMPLETION, different_context)

        assert key1 != key3


class TestCachePromotion:
    """Test cache promotion between L1 and L2."""

    def test_promotion_to_l2_after_threshold(self):
        """Test that entries are promoted to L2 after threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DecisionCache(
                l1_size=100,
                l2_enabled=True,
                l2_cache_dir=Path(tmpdir),
                promotion_threshold=3,
            )

            context = {"message": "test"}

            # Access entry 3 times to trigger promotion
            cache.put(DecisionType.TASK_COMPLETION, context, "result")
            cache.get(DecisionType.TASK_COMPLETION, context)
            cache.get(DecisionType.TASK_COMPLETION, context)
            cache.get(DecisionType.TASK_COMPLETION, context)

            # Should be promoted to L2
            l2_stats = cache.l2.get_stats()
            assert l2_stats["size"] == 1

    def test_no_promotion_below_threshold(self):
        """Test that entries are not promoted below threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DecisionCache(
                l1_size=100,
                l2_enabled=True,
                l2_cache_dir=Path(tmpdir),
                promotion_threshold=5,
            )

            context = {"message": "test"}

            # Access entry only 2 times (below threshold)
            cache.put(DecisionType.TASK_COMPLETION, context, "result")
            cache.get(DecisionType.TASK_COMPLETION, context)
            cache.get(DecisionType.TASK_COMPLETION, context)

            # Should NOT be promoted to L2
            l2_stats = cache.l2.get_stats()
            assert l2_stats["size"] == 0

    def test_l2_demotion_to_l1_on_access(self):
        """Test that L2 entries are demoted to L1 on access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DecisionCache(
                l1_size=100,
                l2_enabled=True,
                l2_cache_dir=Path(tmpdir),
                promotion_threshold=1,
            )

            context = {"message": "test"}

            # Promote to L2
            cache.put(DecisionType.TASK_COMPLETION, context, "result")
            cache.get(DecisionType.TASK_COMPLETION, context)

            # Clear L1 to force L2 access
            cache.l1.clear()

            # Access from L2 should demote to L1
            result = cache.get(DecisionType.TASK_COMPLETION, context)

            assert result == "result"
            assert cache._l2_hits == 1

            # Should now be in L1
            l1_stats = cache.l1.get_stats()
            assert l1_stats["size"] == 1


class TestFactoryFunction:
    """Test factory function."""

    def test_create_with_defaults(self):
        """Test creating cache with default parameters."""
        cache = create_decision_cache()

        assert cache.l1.max_size == 100
        assert cache.l1.default_ttl == 300.0
        assert cache.l2_enabled is False

    def test_create_with_custom_params(self):
        """Test creating cache with custom parameters."""
        cache = create_decision_cache(
            l1_size=200,
            l1_ttl=600,
            l2_enabled=True,
        )

        assert cache.l1.max_size == 200
        assert cache.l1.default_ttl == 600.0
        assert cache.l2_enabled is True


class TestIntegration:
    """Integration tests for decision cache."""

    def test_full_cache_workflow(self):
        """Test complete workflow with L1 and L2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DecisionCache(
                l1_size=10,
                l2_enabled=True,
                l2_cache_dir=Path(tmpdir),
                promotion_threshold=2,
            )

            context = {"message": "test"}

            # First access: miss
            result = cache.get(DecisionType.TASK_COMPLETION, context)
            assert result is None
            assert cache._misses == 1

            # Cache the result
            cache.put(DecisionType.TASK_COMPLETION, context, "cached_result")

            # Second access: L1 hit
            result = cache.get(DecisionType.TASK_COMPLETION, context)
            assert result == "cached_result"
            assert cache._l1_hits == 1

            # Third access: L1 hit, promote to L2
            result = cache.get(DecisionType.TASK_COMPLETION, context)
            assert result == "cached_result"

            l2_stats = cache.l2.get_stats()
            assert l2_stats["size"] == 1

    def test_multiple_decision_types(self):
        """Test caching multiple decision types."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context = {"message": "test"}

        cache.put(DecisionType.TASK_COMPLETION, context, "completion_result")
        cache.put(DecisionType.INTENT_CLASSIFICATION, context, "intent_result")
        cache.put(DecisionType.TASK_TYPE_CLASSIFICATION, context, "type_result")

        assert cache.get(DecisionType.TASK_COMPLETION, context) == "completion_result"
        assert cache.get(DecisionType.INTENT_CLASSIFICATION, context) == "intent_result"
        assert cache.get(DecisionType.TASK_TYPE_CLASSIFICATION, context) == "type_result"

    def test_cache_invalidation_by_type(self):
        """Test invalidating specific decision type."""
        cache = DecisionCache(l1_size=100, l2_enabled=False)

        context = {"message": "test"}

        cache.put(DecisionType.TASK_COMPLETION, context, "result1")
        cache.put(DecisionType.INTENT_CLASSIFICATION, context, "result2")

        # Invalidate only task completion
        cache.invalidate(DecisionType.TASK_COMPLETION, context)

        assert cache.get(DecisionType.TASK_COMPLETION, context) is None
        assert cache.get(DecisionType.INTENT_CLASSIFICATION, context) == "result2"
