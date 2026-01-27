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

"""Performance tests for lazy loading optimization.

These tests verify that lazy loading provides:
- 20-30% initialization time reduction
- 15-25% memory reduction for unused components
- Minimal overhead for first access
"""

import pytest
import time
import tracemalloc

from victor.optimization.runtime import LazyComponentLoader


@pytest.mark.performance
@pytest.mark.slow
class TestLazyLoadingPerformance:
    """Performance test suite for lazy loading."""

    def test_initialization_time_reduction(self):
        """Test that lazy loading reduces initialization time by deferring expensive loads."""
        import asyncio

        # Test with eager loading (baseline) - components are loaded immediately
        async def eager_init():
            start = time.perf_counter()
            components = []
            for i in range(20):
                # Simulate expensive initialization with I/O
                await asyncio.sleep(0.001)  # 1ms per component
                component = self._create_heavy_component(i)
                components.append(component)
            return (time.perf_counter() - start) * 1000

        eager_time = asyncio.run(eager_init())

        # Test with lazy loading - components are NOT loaded yet, just registered
        start_lazy = time.perf_counter()
        lazy_loader = LazyComponentLoader()
        for i in range(20):
            # Registration is fast - lambda is just stored, not executed
            lazy_loader.register_component(
                f"component_{i}", lambda idx=i: self._create_heavy_component(idx)
            )
        lazy_init_time = (time.perf_counter() - start_lazy) * 1000

        # Lazy registration should be much faster than eager loading
        improvement = (eager_time - lazy_init_time) / eager_time
        print(f"\nEager init (loaded all with I/O) time: {eager_time:.2f}ms")
        print(f"Lazy registration (not loaded) time: {lazy_init_time:.2f}ms")
        print(f"Improvement: {improvement:.1%}")

        # Lazy registration should be at least 95% faster (since we're not doing I/O)
        assert improvement >= 0.95, f"Expected >=95% improvement, got {improvement:.1%}"

    def test_first_access_overhead(self):
        """Test that first access overhead is acceptable (<20ms)."""
        lazy_loader = LazyComponentLoader()
        lazy_loader.register_component("heavy", lambda: self._create_heavy_component(0))

        # Time first access
        start = time.perf_counter()
        component = lazy_loader.get_component("heavy")
        first_access_time = (time.perf_counter() - start) * 1000

        print(f"\nFirst access time: {first_access_time:.2f}ms")

        # First access should be reasonably fast (<20ms for typical component)
        assert first_access_time < 20.0, f"First access too slow: {first_access_time:.2f}ms"

    def test_subsequent_access_speed(self):
        """Test that subsequent accesses are instant (cached)."""
        lazy_loader = LazyComponentLoader()
        lazy_loader.register_component("component", lambda: {"data": "value"})

        # First access (loads component)
        lazy_loader.get_component("component")

        # Time subsequent accesses
        times = []
        for _ in range(100):
            start = time.perf_counter()
            lazy_loader.get_component("component")
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        print(f"\nAverage cached access time: {avg_time:.4f}ms")

        # Cached accesses should be very fast (<0.1ms)
        assert avg_time < 0.1, f"Cached access too slow: {avg_time:.4f}ms"

    def test_memory_usage_reduction_unused_components(self):
        """Test memory reduction for unused components (15-25%)."""
        tracemalloc.start()

        # Eager loading baseline
        eager_components = []
        snapshot1 = tracemalloc.take_snapshot()
        for i in range(50):
            component = self._create_heavy_component(i)
            eager_components.append(component)
        snapshot2 = tracemalloc.take_snapshot()
        eager_memory = sum(stat.size_diff for stat in snapshot2.compare_to(snapshot1, "lineno"))

        # Lazy loading (don't access components)
        tracemalloc.clear_traces()
        snapshot3 = tracemalloc.take_snapshot()
        lazy_loader = LazyComponentLoader()
        for i in range(50):
            lazy_loader.register_component(
                f"component_{i}", lambda idx=i: self._create_heavy_component(idx)
            )
        snapshot4 = tracemalloc.take_snapshot()
        lazy_memory = sum(stat.size_diff for stat in snapshot4.compare_to(snapshot3, "lineno"))

        tracemalloc.stop()

        reduction = (eager_memory - lazy_memory) / eager_memory
        print(f"\nEager memory: {eager_memory / 1024:.1f}KB")
        print(f"Lazy memory: {lazy_memory / 1024:.1f}KB")
        print(f"Reduction: {reduction:.1%}")

        # Lazy loading should use at least 15% less memory for unused components
        assert reduction >= 0.15, f"Expected >=15% reduction, got {reduction:.1%}"

    def test_dependency_loading_overhead(self):
        """Test overhead of loading dependencies."""
        lazy_loader = LazyComponentLoader()

        # Create dependency chain: A -> B -> C -> D
        lazy_loader.register_component("D", lambda: {"level": 4})
        lazy_loader.register_component("C", lambda: {"level": 3}, dependencies=["D"])
        lazy_loader.register_component("B", lambda: {"level": 2}, dependencies=["C"])
        lazy_loader.register_component("A", lambda: {"level": 1}, dependencies=["B"])

        start = time.perf_counter()
        result = lazy_loader.get_component("A")
        load_time = (time.perf_counter() - start) * 1000

        print(f"\nDependency chain load time: {load_time:.2f}ms")

        # Loading dependency chain should be fast (<50ms for 4 components)
        assert load_time < 50.0, f"Dependency loading too slow: {load_time:.2f}ms"

    def test_preload_performance(self):
        """Test performance of preloading multiple components."""
        lazy_loader = LazyComponentLoader()

        # Register many components
        for i in range(100):
            lazy_loader.register_component(f"comp_{i}", lambda idx=i: {"id": idx})

        # Time preloading
        start = time.perf_counter()
        keys_to_preload = [f"comp_{i}" for i in range(0, 100, 10)]  # 10 components
        lazy_loader.preload_components(keys_to_preload)
        preload_time = (time.perf_counter() - start) * 1000

        print(f"\nPreload time for 10 components: {preload_time:.2f}ms")

        # Verify preloaded components are loaded
        for key in keys_to_preload:
            assert lazy_loader.is_loaded(key)

        # Preloading should be reasonably fast
        assert preload_time < 100.0, f"Preloading too slow: {preload_time:.2f}ms"

    def test_loading_statistics_overhead(self):
        """Test overhead of statistics collection."""
        lazy_loader = LazyComponentLoader()

        # Register and load components
        for i in range(50):
            lazy_loader.register_component(f"comp_{i}", lambda idx=i: {"id": idx})

        # Load components
        for i in range(50):
            lazy_loader.get_component(f"comp_{i}")

        # Time statistics collection
        start = time.perf_counter()
        stats = lazy_loader.get_loading_stats()
        stats_time = (time.perf_counter() - start) * 1000

        print(f"\nStatistics collection time: {stats_time:.4f}ms")

        # Statistics should be very fast to collect
        assert stats_time < 1.0, f"Statistics collection too slow: {stats_time:.4f}ms"

    def test_hit_rate_calculation(self):
        """Test accuracy of hit rate calculation."""
        lazy_loader = LazyComponentLoader()

        lazy_loader.register_component("test", lambda: "value")

        # First access (miss)
        lazy_loader.get_component("test")

        # Subsequent accesses (hits)
        for _ in range(9):
            lazy_loader.get_component("test")

        stats = lazy_loader.get_loading_stats()

        # 1 miss, 9 hits = 90% hit rate
        assert stats.hit_rate == 0.9
        assert stats.miss_count == 1
        assert stats.hit_count == 9

    def test_concurrent_loading_performance(self):
        """Test performance of concurrent component loading."""
        from concurrent.futures import ThreadPoolExecutor

        lazy_loader = LazyComponentLoader()

        # Register many components
        for i in range(100):
            lazy_loader.register_component(f"comp_{i}", lambda idx=i: {"id": idx})

        # Load from multiple threads
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(lambda i=i: lazy_loader.get_component(f"comp_{i}"))
                for i in range(100)
            ]
            results = [f.result() for f in futures]
        concurrent_time = (time.perf_counter() - start) * 1000

        print(f"\nConcurrent load time (100 components, 10 threads): {concurrent_time:.2f}ms")

        # All components should be loaded
        assert len(results) == 100
        assert all(r is not None for r in results)

        # Concurrent loading should be faster than sequential
        # (rough estimate: should be < 500ms for 100 simple components)
        assert concurrent_time < 500.0, f"Concurrent loading too slow: {concurrent_time:.2f}ms"

    def test_memory_tracking_overhead(self):
        """Test overhead of memory tracking."""
        lazy_loader = LazyComponentLoader()
        lazy_loader.enable_memory_tracking()

        lazy_loader.register_component("test", lambda: {"data": "x" * 1000})

        start = time.perf_counter()
        for _ in range(100):
            lazy_loader.unload_component("test")
            lazy_loader.get_component("test")
        tracking_time = (time.perf_counter() - start) * 1000

        lazy_loader.disable_memory_tracking()

        print(f"\nLoad/unload time with memory tracking (100 iterations): {tracking_time:.2f}ms")

        # Memory tracking adds overhead but should still be reasonable
        # (<5s for 100 load/unload cycles)
        assert tracking_time < 5000.0, f"Memory tracking too slow: {tracking_time:.2f}ms"

    def test_unload_reload_performance(self):
        """Test performance of unload/reload cycle."""
        lazy_loader = LazyComponentLoader()

        lazy_loader.register_component("test", lambda: {"data": "value"})

        # Measure unload/reload cycle time
        times = []
        for _ in range(100):
            lazy_loader.get_component("test")
            start = time.perf_counter()
            lazy_loader.unload_component("test")
            lazy_loader.get_component("test")
            times.append((time.perf_counter() - start) * 1000)

        avg_cycle_time = sum(times) / len(times)
        print(f"\nAverage unload/reload cycle time: {avg_cycle_time:.2f}ms")

        # Unload/reload should be reasonably fast
        assert avg_cycle_time < 10.0, f"Unload/reload cycle too slow: {avg_cycle_time:.2f}ms"

    def test_cache_management_lru_eviction(self):
        """Test that LRU cache eviction works correctly and cache size stays within max_size."""
        # Create loader with max_size=3
        loader = LazyComponentLoader(max_cache_size=3)

        # Register 10 components
        for i in range(10):
            loader.register_component(
                f"comp_{i}", lambda idx=i: {"id": idx, "data": list(range(100))}
            )

        # Load components 0-9 sequentially
        for i in range(10):
            loader.get_component(f"comp_{i}")

        # Check loaded components - should have at most 3
        loaded = loader.get_loaded_components()

        print("\nCache max_size: 3")
        print("Components registered: 10")
        print(f"Components loaded: {len(loaded)}")
        print(f"Loaded components: {sorted(loaded)}")

        # Critical assertion: cache should never exceed max_size
        assert len(loaded) <= 3, f"Cache size {len(loaded)} exceeds max_size 3"

        # Verify the most recently accessed components are cached
        # After loading 0,1,2,3,4,5,6,7,8,9 in sequence,
        # the cache should have the 3 most recent: 7,8,9
        # (because each new load evicts the oldest)
        assert "comp_9" in loaded, "Most recent component should be cached"
        assert "comp_8" in loaded, "Second most recent component should be cached"
        assert "comp_7" in loaded, "Third most recent component should be cached"

        # Oldest components should be evicted
        assert "comp_0" not in loaded, "Oldest component should be evicted"
        assert "comp_1" not in loaded, "Second oldest component should be evicted"

        print("✓ LRU eviction working correctly")

    def test_cache_management_with_reaccess(self):
        """Test that accessing a component updates its LRU position."""
        loader = LazyComponentLoader(max_cache_size=3)

        # Register 5 components
        for i in range(5):
            loader.register_component(f"comp_{i}", lambda idx=i: {"id": idx})

        # Load components 0, 1, 2 (cache: [0, 1, 2])
        loader.get_component("comp_0")
        loader.get_component("comp_1")
        loader.get_component("comp_2")

        # Re-access component 0 to update its LRU position (cache: [1, 2, 0])
        loader.get_component("comp_0")

        # Load component 3 - should evict comp_1 (LRU) (cache: [2, 0, 3])
        loader.get_component("comp_3")

        loaded = loader.get_loaded_components()

        print(f"\nCache after re-access: {sorted(loaded)}")

        # Verify cache size
        assert len(loaded) <= 3, f"Cache size {len(loaded)} exceeds max_size 3"

        # comp_0 should still be cached (recently accessed)
        assert "comp_0" in loaded, "Recently accessed component should be cached"

        # comp_1 should be evicted (least recently used)
        assert "comp_1" not in loaded, "LRU component should be evicted"

        # comp_3 should be cached (newly loaded)
        assert "comp_3" in loaded, "Newly loaded component should be cached"

        print("✓ LRU re-access working correctly")

    def test_cache_management_with_different_strategies(self):
        """Test that cache management works with all loading strategies."""
        from victor.optimization.runtime.lazy_loader import LoadingStrategy

        for strategy in [LoadingStrategy.LAZY, LoadingStrategy.EAGER, LoadingStrategy.ADAPTIVE]:
            loader = LazyComponentLoader(max_cache_size=3, strategy=strategy)

            # Register 10 components
            for i in range(10):
                loader.register_component(f"comp_{i}", lambda idx=i: {"id": idx})

            # Load all components
            for i in range(10):
                loader.get_component(f"comp_{i}")

            loaded = loader.get_loaded_components()

            print(f"\nStrategy: {strategy.value}")
            print(f"Cache size: {len(loaded)}")

            # Cache should respect max_size for all strategies
            assert (
                len(loaded) <= 3
            ), f"Cache size {len(loaded)} exceeds max_size 3 for strategy {strategy.value}"

            print(f"✓ Cache management works for {strategy.value} strategy")

    def test_adaptive_strategy_performance(self):
        """Test performance of adaptive loading strategy with access pattern learning."""
        from victor.optimization.runtime.lazy_loader import LoadingStrategy

        # Create loader with adaptive strategy
        loader = LazyComponentLoader(
            strategy=LoadingStrategy.ADAPTIVE,
            adaptive_threshold=3,  # Preload after 3 accesses
            max_cache_size=10,
        )

        # Register components with different access patterns
        for i in range(20):
            loader.register_component(
                f"comp_{i}", lambda idx=i: {"id": idx, "data": list(range(100))}
            )

        # Simulate access pattern where some components are accessed frequently
        import random

        hot_components = [0, 5, 10, 15]  # These will be accessed frequently

        start = time.perf_counter()

        # Phase 1: Initial accesses (cold start)
        for i in range(20):
            loader.get_component(f"comp_{i}")

        # Phase 2: Repeat accesses to hot components (triggers adaptive preloading)
        for _ in range(10):
            for hot_idx in hot_components:
                loader.get_component(f"comp_{hot_idx}")

        # Phase 3: Access components near hot ones (may trigger adaptive preloading)
        for hot_idx in hot_components:
            neighbor = hot_idx + 1
            if neighbor < 20:
                loader.get_component(f"comp_{neighbor}")

        adaptive_time = (time.perf_counter() - start) * 1000

        # Get statistics
        stats = loader.get_loading_stats()

        print("\nAdaptive strategy performance:")
        print(f"Total time: {adaptive_time:.2f}ms")
        print(f"Hit rate: {stats.hit_rate:.1%}")
        print(f"Total accesses: {stats.hit_count + stats.miss_count}")
        print(f"Loaded components: {len(loader.get_loaded_components())}")

        # Adaptive strategy should achieve good hit rate
        assert stats.hit_rate >= 0.5, f"Hit rate too low: {stats.hit_rate:.1%}"

        # Performance should be reasonable for this workload
        assert adaptive_time < 1000.0, f"Adaptive strategy too slow: {adaptive_time:.2f}ms"

        # Hot components should be in cache
        loaded = loader.get_loaded_components()
        hot_loaded = sum(1 for idx in hot_components if f"comp_{idx}" in loaded)
        assert (
            hot_loaded >= len(hot_components) // 2
        ), f"Too few hot components cached: {hot_loaded}/{len(hot_components)}"

        print(f"✓ Adaptive strategy achieved {stats.hit_rate:.1%} hit rate")

    @staticmethod
    def _create_heavy_component(index: int) -> dict:
        """Create a component with moderate memory footprint."""
        # Simulate component with some data
        return {
            "id": index,
            "data": list(range(1000)),
            "metadata": {
                "created": time.time(),
                "type": f"component_{index}",
            },
        }
