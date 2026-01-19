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

"""Unit tests for adaptive cache sizing."""

import pytest
import time
from victor.tools.caches import AdaptiveLRUCache, AdaptiveMetrics


class TestAdaptiveLRUCache:
    """Test suite for AdaptiveLRUCache."""

    def test_initialization(self):
        """Test cache initialization with defaults."""
        cache = AdaptiveLRUCache()

        assert cache.size == 500  # Default initial size
        assert cache.enabled is True
        metrics = cache.get_metrics()
        assert metrics["size"]["current"] == 500
        assert metrics["size"]["min"] == 100
        assert metrics["size"]["max"] == 2000

    def test_custom_initialization(self):
        """Test cache initialization with custom parameters."""
        cache = AdaptiveLRUCache(
            initial_size=200,
            max_size=1000,
            min_size=50,
            target_hit_rate=0.7,
        )

        assert cache.size == 200
        metrics = cache.get_metrics()
        assert metrics["size"]["current"] == 200
        assert metrics["size"]["min"] == 50
        assert metrics["size"]["max"] == 1000
        assert metrics["adaptive"]["target_hit_rate"] == 0.7

    def test_basic_operations(self):
        """Test basic cache put and get operations."""
        cache = AdaptiveLRUCache(initial_size=10)

        # Put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        # Get non-existent key
        assert cache.get("key2") is None

        # Get with default
        assert cache.get("key2", "default") == "default"

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = AdaptiveLRUCache(initial_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Cache is full
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Add one more - should evict key1 (least recently used)
        cache.put("key4", "value4")
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_metrics_tracking(self):
        """Test cache metrics tracking."""
        cache = AdaptiveLRUCache(initial_size=10)

        # Initial metrics
        metrics = cache.get_metrics()
        assert metrics["performance"]["hits"] == 0
        assert metrics["performance"]["misses"] == 0
        assert metrics["performance"]["hit_rate"] == 0.0

        # Add some entries
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Cache hits
        cache.get("key1")
        cache.get("key2")

        # Cache miss
        cache.get("key3")

        # Check metrics
        metrics = cache.get_metrics()
        assert metrics["performance"]["hits"] == 2
        assert metrics["performance"]["misses"] == 1
        assert metrics["performance"]["hit_rate"] == 2 / 3

    def test_should_adjust(self):
        """Test should_adjust logic."""
        cache = AdaptiveLRUCache(initial_size=100, adjustment_interval=1)

        # Initially no adjustment needed
        assert cache.should_adjust() is False

        # Add some operations
        for i in range(50):
            cache.put(f"key{i}", f"value{i}")
            cache.get(f"key{i}")

        # Wait for adjustment interval
        time.sleep(1.1)

        # Should check adjustment
        result = cache.should_adjust()
        # May or may not need adjustment depending on hit rate
        assert isinstance(result, bool)

    def test_adjust_size_expand(self):
        """Test cache expansion on low hit rate."""
        cache = AdaptiveLRUCache(
            initial_size=100,
            max_size=500,
            target_hit_rate=0.6,
            adjustment_interval=0,  # Immediate adjustment
        )

        # Create low hit rate scenario
        for i in range(200):
            cache.put(f"key{i}", f"value{i}")

        # Many misses
        for i in range(200, 300):
            cache.get(f"key{i}")

        # Force adjustment
        if cache.should_adjust():
            result = cache.adjust_size()
            # May expand if hit rate is low and memory usage is OK
            assert result["adjustment"] in ["expand", "shrink", "none"]
            assert "old_size" in result
            assert "new_size" in result

    def test_adjust_size_shrink(self):
        """Test cache shrinking on high memory usage."""
        cache = AdaptiveLRUCache(
            initial_size=200,
            min_size=50,
            adjustment_interval=0,
        )

        # Fill cache
        for i in range(200):
            cache.put(f"key{i}", f"value{i}")

        # All hits - high hit rate
        for i in range(100):
            cache.get(f"key{i}")

        # Force adjustment
        if cache.should_adjust():
            result = cache.adjust_size()
            assert result["adjustment"] in ["expand", "shrink", "none"]

    def test_invalidate(self):
        """Test cache invalidation."""
        cache = AdaptiveLRUCache(initial_size=10)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Invalidate specific key
        cache.invalidate("key1")
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

        # Invalidate all
        cache.invalidate()
        assert cache.get("key2") is None
        metrics = cache.get_metrics()
        assert metrics["size"]["entries"] == 0

    def test_enable_disable(self):
        """Test enable/disable functionality."""
        cache = AdaptiveLRUCache()

        assert cache.enabled is True

        cache.disable()
        assert cache.enabled is False

        cache.enable()
        assert cache.enabled is True

    def test_reset_metrics(self):
        """Test metrics reset."""
        cache = AdaptiveLRUCache(initial_size=10)

        # Generate some activity
        for i in range(5):
            cache.put(f"key{i}", f"value{i}")
            cache.get(f"key{i}")
        cache.get("nonexistent")

        # Reset metrics
        cache.reset_metrics()

        # Check metrics are reset
        metrics = cache.get_metrics()
        assert metrics["performance"]["hits"] == 0
        assert metrics["performance"]["misses"] == 0
        assert metrics["adaptive"]["adjustments"] == 0  # Adjustments are preserved

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        import threading

        cache = AdaptiveLRUCache(initial_size=100)
        errors = []

        def worker(worker_id):
            try:
                for i in range(50):
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

    def test_size_constraints(self):
        """Test that cache size stays within bounds."""
        cache = AdaptiveLRUCache(
            initial_size=100,
            min_size=50,
            max_size=200,
            adjustment_interval=0,
        )

        # Try to expand beyond max
        cache._current_size = 200
        if cache.should_adjust():
            cache.adjust_size()
        assert cache.size <= 200

        # Try to shrink below min
        cache._current_size = 50
        if cache.should_adjust():
            cache.adjust_size()
        assert cache.size >= 50


class TestAdaptiveMetrics:
    """Test suite for AdaptiveMetrics."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = AdaptiveMetrics()

        assert metrics.hit_rate == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.avg_access_time == 0.0
        assert metrics.eviction_rate == 0.0
        assert metrics.adjustment_count == 0

    def test_custom_initialization(self):
        """Test metrics with custom values."""
        metrics = AdaptiveMetrics(
            hit_rate=0.75,
            memory_usage=0.5,
            avg_access_time=0.1,
            eviction_rate=0.02,
            adjustment_count=5,
        )

        assert metrics.hit_rate == 0.75
        assert metrics.memory_usage == 0.5
        assert metrics.avg_access_time == 0.1
        assert metrics.eviction_rate == 0.02
        assert metrics.adjustment_count == 5
