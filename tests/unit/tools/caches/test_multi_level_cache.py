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

"""Unit tests for multi-level caching."""

import pytest
import tempfile
from pathlib import Path
from victor.tools.caches import MultiLevelCache, CacheEntry, LevelMetrics


class TestMultiLevelCache:
    """Test suite for MultiLevelCache."""

    def test_initialization(self):
        """Test cache initialization with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l2_dir=Path(tmpdir))

            metrics = cache.get_metrics()
            assert metrics["l1"]["entry_count"] == 0
            assert metrics["l2"]["entry_count"] == 0
            assert metrics["l3"]["entry_count"] == 0

    def test_custom_initialization(self):
        """Test cache initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(
                l1_size=50,
                l2_size=500,
                l3_size=5000,
                l2_dir=Path(tmpdir),
                l1_ttl=600,
                l2_ttl=3600,
            )

            assert cache._l1_size == 50
            assert cache._l2_size == 500
            assert cache._l3_size == 5000
            assert cache._l1_ttl == 600
            assert cache._l2_ttl == 3600

    def test_basic_operations(self):
        """Test basic cache put and get operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l2_dir=Path(tmpdir))

            # Put and get from L1
            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"

            # Get non-existent key
            assert cache.get("key2") is None

            # Get with default
            assert cache.get("key2", "default") == "default"

    def test_l1_hit(self):
        """Test L1 cache hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l1_size=5, l2_dir=Path(tmpdir))

            cache.put("key1", "value1")
            result = cache.get("key1")

            assert result == "value1"

            metrics = cache.get_metrics()
            assert metrics["l1"]["hits"] == 1
            assert metrics["l1"]["misses"] == 0

    def test_l1_miss_l2_hit(self):
        """Test L2 cache hit after L1 miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l1_size=2, l2_dir=Path(tmpdir))

            # Fill L1
            cache.put("key1", "value1")
            cache.put("key2", "value2")

            # Add to L2 by filling L1
            cache.put("key3", "value3")  # key1 evicted to L2

            # Add one more to evict key2 to L2
            cache.put("key4", "value4")  # key2 evicted to L2

            # Now key1 and key2 should be in L2
            # Access key1 from L2 (should promote to L1)
            result = cache.get("key1")
            assert result == "value1"

            metrics = cache.get_metrics()
            # Should have at least one L2 hit or promotion
            assert metrics["l2"]["hits"] >= 1 or metrics["l2"]["promotions"] >= 1

    def test_invalidation(self):
        """Test cache invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l2_dir=Path(tmpdir))

            cache.put("key1", "value1")
            cache.put("key2", "value2")

            # Invalidate specific key
            cache.invalidate("key1")
            assert cache.get("key1") is None
            assert cache.get("key2") == "value2"

            # Invalidate all
            cache.invalidate()
            assert cache.get("key2") is None

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l1_ttl=0.1, l2_dir=Path(tmpdir))  # Very short TTL (100ms)

            cache.put("key1", "value1")
            result = cache.get("key1")

            # Should not be expired immediately
            assert result == "value1"

            # Wait for TTL to expire
            time.sleep(0.15)
            result = cache.get("key1")

            # Should be expired now
            assert result is None

    def test_custom_ttl(self):
        """Test custom TTL per entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l1_ttl=0, l2_dir=Path(tmpdir))

            # Custom TTL that overrides default
            cache.put("key1", "value1", ttl=10)
            result = cache.get("key1")

            # Should not be expired yet
            assert result == "value1"

    def test_metrics(self):
        """Test cache metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l2_dir=Path(tmpdir))

            # Generate some activity
            cache.put("key1", "value1")
            cache.put("key2", "value2")
            cache.get("key1")  # Hit
            cache.get("key3")  # Miss

            metrics = cache.get_metrics()

            # Check combined metrics
            assert metrics["combined"]["total_hits"] >= 1
            assert metrics["combined"]["total_misses"] >= 1
            assert 0.0 <= metrics["combined"]["hit_rate"] <= 1.0

            # Check level-specific metrics
            assert metrics["l1"]["level"] == 1
            assert metrics["l2"]["level"] == 2
            assert metrics["l3"]["level"] == 3

    def test_enable_disable(self):
        """Test enable/disable functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l2_dir=Path(tmpdir))

            assert cache._enabled is True

            cache.disable()
            assert cache._enabled is False

            # Cache should return default when disabled
            cache.put("key1", "value1")
            assert cache.get("key1") is None

            cache.enable()
            assert cache._enabled is True

            # Cache should work again
            cache.put("key2", "value2")
            assert cache.get("key2") == "value2"

    def test_size_estimation(self):
        """Test size estimation for cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l2_dir=Path(tmpdir))

            # Different value types
            cache.put("key1", "small")
            cache.put("key2", "x" * 1000)
            cache.put("key3", ["list", "of", "values"])

            # Metrics should show non-zero size
            metrics = cache.get_metrics()
            assert metrics["l1"]["total_size"] > 0

    def test_promotion_demotion(self):
        """Test automatic promotion and demotion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l1_size=2, l2_dir=Path(tmpdir))

            # Fill L1
            cache.put("key1", "value1")
            cache.put("key2", "value2")

            # Add to L2 (evicts key1)
            cache.put("key3", "value3")

            # Add another to evict key2
            cache.put("key4", "value4")

            # Access key1 multiple times from L2 to promote to L1
            for _ in range(5):
                cache.get("key1")

            metrics = cache.get_metrics()
            # Should have demotions from L1 evictions
            assert metrics["combined"]["total_demotions"] >= 1

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MultiLevelCache(l2_dir=Path(tmpdir))
            errors = []

            def worker(worker_id):
                try:
                    for i in range(20):
                        key = f"worker{worker_id}_key{i}"
                        cache.put(key, f"value_{i}")
                        value = cache.get(key)
                        assert value == f"value_{i}"
                except Exception as e:
                    errors.append(e)

            # Run concurrent workers
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Check no errors occurred
            assert len(errors) == 0


class TestCacheEntry:
    """Test suite for CacheEntry."""

    def test_initialization(self):
        """Test cache entry initialization."""
        entry = CacheEntry(key="test_key", value="test_value", level=1)

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.level == 1
        assert entry.access_count == 0
        assert entry.ttl is None

    def test_expiration_check(self):
        """Test TTL expiration checking."""
        import time

        # No TTL - not expired
        entry = CacheEntry(key="test", value="value", level=1, ttl=None)
        assert entry.is_expired() is False

        # Zero TTL - not expired (treated as no expiration)
        entry = CacheEntry(key="test", value="value", level=1, ttl=0)
        assert entry.is_expired() is False

        # Long TTL - not expired
        entry = CacheEntry(key="test", value="value", level=1, ttl=3600)
        assert entry.is_expired() is False

        # Short TTL - expired after delay
        entry = CacheEntry(key="test", value="value", level=1, ttl=0.1)
        time.sleep(0.15)  # Wait for TTL to pass
        assert entry.is_expired() is True

    def test_access_recording(self):
        """Test access recording."""
        entry = CacheEntry(key="test", value="value", level=1)

        initial_count = entry.access_count
        entry.record_access()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > 0


class TestLevelMetrics:
    """Test suite for LevelMetrics."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = LevelMetrics(level=1)

        assert metrics.level == 1
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.promotions == 0
        assert metrics.demotions == 0
        assert metrics.evictions == 0
        assert metrics.total_size == 0
        assert metrics.entry_count == 0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = LevelMetrics(level=1)

        # No activity
        assert metrics.hit_rate == 0.0

        # Some hits and misses
        metrics.hits = 60
        metrics.misses = 40
        assert metrics.hit_rate == 0.6

        # Only hits
        metrics.hits = 100
        metrics.misses = 0
        assert metrics.hit_rate == 1.0

        # Only misses
        metrics.hits = 0
        metrics.misses = 100
        assert metrics.hit_rate == 0.0
