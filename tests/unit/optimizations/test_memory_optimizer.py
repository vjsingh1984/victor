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

"""Tests for memory optimization module."""

import asyncio
import gc
import pytest

from victor.optimization.runtime.memory import (
    MemoryOptimizer,
    ObjectPool,
    MemoryProfiler,
    MemoryStats,
    get_memory_optimizer,
)


class TestObjectPool:
    """Test ObjectPool functionality."""

    def test_pool_acquire_release(self):
        """Test acquiring and releasing objects."""
        pool = ObjectPool(
            factory=lambda: [],
            max_size=10,
        )

        obj1 = pool.acquire()
        obj2 = pool.acquire()

        assert obj1 is not obj2

        pool.release(obj1)
        obj3 = pool.acquire()

        # Should reuse released object
        assert obj3 is obj1

    def test_pool_max_size(self):
        """Test pool respects max size."""
        pool = ObjectPool(
            factory=lambda: [],
            max_size=2,
        )

        obj1 = pool.acquire()
        obj2 = pool.acquire()
        obj3 = pool.acquire()  # New object, pool at capacity

        pool.release(obj1)
        pool.release(obj2)
        pool.release(obj3)  # Should be discarded

        # Only 2 objects in pool
        stats = pool.get_stats()
        assert stats["pool_size"] == 2

    def test_pool_reset_function(self):
        """Test pool resets objects on reuse."""
        reset_called = []

        def reset_func(obj):
            reset_called.append(obj)
            obj.clear()

        pool = ObjectPool(
            factory=lambda: [1, 2, 3],
            reset=reset_func,
            max_size=10,
        )

        obj1 = pool.acquire()
        obj1.extend([4, 5])

        pool.release(obj1)
        obj2 = pool.acquire()

        # Object should be reset
        assert len(reset_called) == 1
        assert obj2 == []

    def test_pool_stats(self):
        """Test pool statistics."""
        pool = ObjectPool(
            factory=lambda: [],
            max_size=10,
        )

        obj1 = pool.acquire()
        obj2 = pool.acquire()
        pool.release(obj1)
        obj3 = pool.acquire()

        stats = pool.get_stats()

        assert stats["created"] == 2
        assert stats["acquired"] == 3
        assert stats["reused"] == 1

    def test_pool_clear(self):
        """Test clearing pool."""
        pool = ObjectPool(
            factory=lambda: [],
            max_size=10,
        )

        obj1 = pool.acquire()
        pool.release(obj1)

        assert pool.get_stats()["pool_size"] == 1

        pool.clear()

        assert pool.get_stats()["pool_size"] == 0


class TestMemoryProfiler:
    """Test MemoryProfiler functionality."""

    def test_profiler_start_stop(self):
        """Test starting and stopping profiler."""
        profiler = MemoryProfiler(sample_interval=0.1)

        assert not profiler._running

        profiler.start()
        assert profiler._running

        profiler.stop()
        assert not profiler._running

    def test_profiler_get_stats(self):
        """Test getting profiler stats."""
        profiler = MemoryProfiler()

        stats = profiler.get_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.total_objects >= 0

    def test_profiler_detect_leaks(self):
        """Test leak detection."""
        profiler = MemoryProfiler(sample_interval=0.1)

        # No leaks detected without data
        leaks = profiler.detect_leaks()

        assert isinstance(leaks, list)


class TestMemoryOptimizer:
    """Test MemoryOptimizer functionality."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = MemoryOptimizer()

        assert optimizer is not None
        assert optimizer._pools == {}

    def test_enable_gc_tuning(self):
        """Test GC tuning."""
        # Get original thresholds
        original = gc.get_threshold()

        try:
            MemoryOptimizer.enable_gc_tuning(aggressive=True)
            thresholds = gc.get_threshold()
            assert thresholds == MemoryOptimizer.DEFAULT_GC_THRESHOLDS

            MemoryOptimizer.enable_gc_tuning(aggressive=False)
            thresholds = gc.get_threshold()
            assert thresholds == MemoryOptimizer.CONSERVATIVE_GC_THRESHOLDS

        finally:
            # Restore original
            gc.set_threshold(*original)

    def test_disable_gc_tuning(self):
        """Test disabling GC tuning."""
        MemoryOptimizer.disable_gc_tuning()

        # Should not raise
        thresholds = gc.get_threshold()
        assert thresholds is not None

    def test_create_pool(self):
        """Test creating object pool."""
        optimizer = MemoryOptimizer()

        pool = optimizer.create_pool(
            "test_pool",
            factory=lambda: [],
            max_size=10,
        )

        assert "test_pool" in optimizer._pools
        assert optimizer._pools["test_pool"] is pool

    def test_get_pool(self):
        """Test getting existing pool."""
        optimizer = MemoryOptimizer()

        pool1 = optimizer.create_pool("test_pool", factory=lambda: [])
        pool2 = optimizer.get_pool("test_pool")

        assert pool1 is pool2

        pool3 = optimizer.get_pool("nonexistent")
        assert pool3 is None

    def test_get_stats(self):
        """Test getting memory stats."""
        optimizer = MemoryOptimizer()

        stats = optimizer.get_stats()

        assert isinstance(stats, MemoryStats)

    def test_detect_leaks(self):
        """Test leak detection."""
        optimizer = MemoryOptimizer()

        leaks = optimizer.detect_leaks()

        assert isinstance(leaks, list)

    def test_collect_garbage(self):
        """Test garbage collection."""
        optimizer = MemoryOptimizer()

        result = optimizer.collect_garbage()

        assert "collected" in result
        assert "objects_before" in result
        assert "objects_after" in result

    def test_get_memory_summary(self):
        """Test memory summary string."""
        optimizer = MemoryOptimizer()

        summary = optimizer.get_memory_summary()

        assert "Memory Usage Summary" in summary
        assert "Total Objects" in summary


class TestGlobalMemoryOptimizer:
    """Test global memory optimizer instance."""

    def test_get_global_optimizer(self):
        """Test getting global optimizer."""
        optimizer1 = get_memory_optimizer()
        optimizer2 = get_memory_optimizer()

        assert optimizer1 is optimizer2


@pytest.mark.asyncio
async def test_memory_optimization_integration():
    """Integration test for memory optimizations."""
    optimizer = MemoryOptimizer()

    # Create pool
    pool = optimizer.create_pool(
        "buffers",
        factory=lambda: bytearray(4096),
        max_size=10,
    )

    # Use pool
    buffers = [pool.acquire() for _ in range(5)]

    # Release buffers
    for buf in buffers:
        pool.release(buf)

    # Check stats
    stats = pool.get_stats()
    assert stats["pool_size"] == 5
    assert stats["reuse_rate"] == 0.0  # No reuse yet

    # Acquire again (should reuse)
    new_buffers = [pool.acquire() for _ in range(5)]

    stats = pool.get_stats()
    assert stats["reuse_rate"] > 0
