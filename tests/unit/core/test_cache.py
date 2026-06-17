"""Tests for victor.core.cache module."""

import time
from victor.core.cache import (
    CacheEntry,
    CacheStats,
    LRUCache,
    TTLCache,
    NamespacedCache,
    create_hash_cache_key,
    cached,
)


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_initialization(self):
        entry = CacheEntry(key="test", value="data")
        assert entry.key == "test"
        assert entry.value == "data"
        assert entry.access_count == 0
        assert entry.ttl is None

    def test_expiration_check(self):
        entry = CacheEntry(key="test", value="data", ttl=1.0)
        assert not entry.is_expired()
        time.sleep(1.1)
        assert entry.is_expired()

    def test_touch_updates_metadata(self):
        entry = CacheEntry(key="test", value="data")
        initial_time = entry.last_accessed_at
        time.sleep(0.1)
        entry.touch()
        assert entry.last_accessed_at > initial_time
        assert entry.access_count == 1


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=70, misses=30)
        assert stats.total_requests == 100
        assert stats.hit_rate == 0.7

    def test_hit_rate_empty(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_stats_addition(self):
        stats1 = CacheStats(hits=10, misses=5, evictions=2)
        stats2 = CacheStats(hits=20, misses=10, evictions=1)
        combined = stats1 + stats2
        assert combined.hits == 30
        assert combined.misses == 15
        assert combined.evictions == 3


class TestLRUCache:
    """Tests for LRUCache."""

    def test_set_and_get(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_returns_default_for_missing(self):
        cache = LRUCache(max_size=10)
        assert cache.get("missing") is None
        assert cache.get("missing", "default") == "default"

    def test_delete(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.delete("key1") is False

    def test_clear(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert len(cache) == 0
        assert "key1" not in cache

    def test_contains(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert "key1" in cache
        assert "key2" not in cache

    def test_lru_eviction(self):
        cache = LRUCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        assert len(cache) == 3

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")
        assert len(cache) == 3
        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_stats_tracking(self):
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.sets == 1
        assert stats.current_size == 1


class TestTTLCache:
    """Tests for TTLCache."""

    def test_set_and_get(self):
        cache = TTLCache(ttl_seconds=60.0)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_expiration(self):
        cache = TTLCache(ttl_seconds=0.5)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.6)
        assert cache.get("key1") is None

    def test_custom_ttl_per_entry(self):
        cache = TTLCache(ttl_seconds=60.0)
        cache.set("key1", "value1", ttl=0.5)
        assert cache.get("key1") == "value1"
        time.sleep(0.6)
        assert cache.get("key1") is None

    def test_max_size_enforcement(self):
        cache = TTLCache(ttl_seconds=60.0, max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        assert len(cache) == 2

    def test_cleanup_expired(self):
        cache = TTLCache(ttl_seconds=0.5)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache) == 2
        time.sleep(0.6)
        removed = cache.cleanup_expired()
        assert removed == 2
        assert len(cache) == 0


class TestNamespacedCache:
    """Tests for NamespacedCache."""

    def test_default_namespace(self):
        cache = NamespacedCache(default_namespace="app")
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("key1", namespace="app") == "value1"

    def test_separate_namespaces(self):
        cache = NamespacedCache(default_namespace="app")
        cache.set("key", "value1", namespace="ns1")
        cache.set("key", "value2", namespace="ns2")

        assert cache.get("key", namespace="ns1") == "value1"
        assert cache.get("key", namespace="ns2") == "value2"
        assert cache.get("key", namespace="app") is None

    def test_clear_namespace(self):
        cache = NamespacedCache(default_namespace="app")
        cache.set("key1", "value1", namespace="ns1")
        cache.set("key2", "value2", namespace="ns2")

        count = cache.clear_namespace("ns1")
        assert count == 1
        assert cache.get("key1", namespace="ns1") is None
        assert cache.get("key2", namespace="ns2") == "value2"

    def test_clear_all_namespaces(self):
        cache = NamespacedCache(default_namespace="app")
        cache.set("key1", "value1", namespace="ns1")
        cache.set("key2", "value2", namespace="ns2")

        cache.clear()
        assert cache.get("key1", namespace="ns1") is None
        assert cache.get("key2", namespace="ns2") is None

    def test_combined_stats(self):
        cache = NamespacedCache(default_namespace="app")
        cache.set("key1", "value1", namespace="ns1")
        cache.set("key2", "value2", namespace="ns2")

        stats = cache.get_stats()
        assert stats.current_size == 2


class TestHashCacheKey:
    """Tests for create_hash_cache_key."""

    def test_stable_hash_same_inputs(self):
        key1 = create_hash_cache_key("test", 123, param="value")
        key2 = create_hash_cache_key("test", 123, param="value")
        assert key1 == key2

    def test_different_hash_different_inputs(self):
        key1 = create_hash_cache_key("test", 123, param="value")
        key2 = create_hash_cache_key("test", 123, param="different")
        assert key1 != key2

    def test_kwargs_order_doesnt_matter(self):
        key1 = create_hash_cache_key("test", a=1, b=2)
        key2 = create_hash_cache_key("test", b=2, a=1)
        assert key1 == key2


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_caches_function_result(self):
        call_count = [0]

        @cached(ttl=60.0)
        def expensive_function(x):
            call_count[0] += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count[0] == 1  # Only called once

    def test_different_args_different_cache_entries(self):
        call_count = [0]

        @cached(ttl=60.0)
        def expensive_function(x):
            call_count[0] += 1
            return x * 2

        expensive_function(5)
        expensive_function(10)

        assert call_count[0] == 2  # Called twice for different args

    def test_cache_attribute_accessible(self):
        @cached(ttl=60.0)
        def func(x):
            return x * 2

        assert hasattr(func, "cache")
        assert isinstance(func.cache, LRUCache)
