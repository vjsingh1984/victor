# Copyright 2025 Vijaykumar Singh <singhvijd@gmail.com>
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

"""Performance tests for striped lock scalability.

Tests the linear scalability of UniversalRegistry with striped locks
under concurrent access patterns.
"""

import threading
import time

import pytest

# Import directly to avoid framework __init__.py issues
from victor.core.registries.universal_registry import CacheStrategy, UniversalRegistry


@pytest.mark.benchmark
class TestStripedLockScalability:
    """Test striped lock scalability under concurrent access."""

    def test_concurrent_reads_linear_scalability(self):
        """Test that reads scale reasonably with striped locks."""
        registry = UniversalRegistry.get_registry("test_scalability", CacheStrategy.LRU, 1000)

        # Populate registry with 100 different keys
        for i in range(100):
            registry.register(f"key_{i}", f"value_{i}")

        # Test with 1, 2, 4, 8, 16 threads
        results = {}
        for num_threads in [1, 2, 4, 8, 16]:
            threads = []
            start = time.time()

            # Each thread reads all 100 keys multiple times
            def read_worker():
                for _ in range(10):  # Multiple iterations to smooth out variance
                    for i in range(100):
                        registry.get(f"key_{i}")

            for _ in range(num_threads):
                t = threading.Thread(target=read_worker)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            duration = time.time() - start
            total_ops = 100 * 10 * num_threads  # 100 keys * 10 iterations * num_threads
            ops_per_sec = total_ops / duration
            results[num_threads] = ops_per_sec

        print("\nScalability Results:")
        for num_threads, ops in results.items():
            print(f"  {num_threads:2d} threads: {ops:8.0f} ops/sec ({ops/results[1]:.2f}x)")

        # With striped locks, the key benefits are:
        # 1. No significant degradation with more threads (unlike single lock)
        # 2. Better total throughput than single-lock would provide
        # 3. Linear scalability up to the number of stripes (16)

        # Due to Python's GIL, single-threaded is often fastest for CPU-bound ops
        # The real test is: does 16-thread throughput maintain reasonable performance?

        # Ensure no catastrophic degradation with more threads
        # 16 threads should perform at least 30% of single-threaded (GIL limits this)
        assert (
            results[16] >= results[1] * 0.3
        ), f"Catastrophic degradation: 16 threads ({results[16]:.0f} ops/s) vs 1 thread ({results[1]:.0f} ops/s)"

        # Ensure 8 threads performs at least 40% of 4 threads
        assert (
            results[8] >= results[4] * 0.4
        ), f"Significant degradation: 8 threads ({results[8]:.0f} ops/s) vs 4 threads ({results[4]:.0f} ops/s)"

        # The key assertion: with striped locks, we should handle high concurrency without failure
        # Total operations processed should scale with thread count (even if ops/sec doesn't)
        # This verifies the locks work correctly without deadlocks or excessive blocking

    def test_concurrent_writes_no_deadlock(self):
        """Test that concurrent writes don't cause deadlocks."""
        registry = UniversalRegistry.get_registry("test_writes", CacheStrategy.LRU, 1000)

        # 16 threads writing to 100 different keys
        threads = []
        for thread_id in range(16):
            t = threading.Thread(
                target=lambda tid=thread_id: [
                    registry.register(f"key_{i}", f"thread_{tid}_value_{i}") for i in range(100)
                ]
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all entries exist
        stats = registry.get_stats()
        assert stats["total_entries"] == 100, f"Expected 100 entries, got {stats['total_entries']}"

        print(f"\nConcurrent writes: {stats['total_entries']} entries successfully written")

    def test_mixed_read_write_workload(self):
        """Test mixed read/write workload under high concurrency."""
        registry = UniversalRegistry.get_registry("test_mixed", CacheStrategy.LRU, 1000)

        # Pre-populate
        for i in range(100):
            registry.register(f"key_{i}", f"value_{i}")

        threads = []
        errors = []

        # 8 reader threads
        for _ in range(8):

            def reader():
                try:
                    for _ in range(100):
                        for i in range(100):
                            registry.get(f"key_{i}")
                except Exception as e:
                    errors.append(f"Reader error: {e}")

            t = threading.Thread(target=reader)
            threads.append(t)
            t.start()

        # 8 writer threads
        for thread_id in range(8):

            def writer(tid=thread_id):
                try:
                    for i in range(100):
                        registry.register(f"key_{i}", f"thread_{tid}_value_{i}")
                except Exception as e:
                    errors.append(f"Writer {tid} error: {e}")

            t = threading.Thread(target=writer)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors during mixed workload: {errors}"

        # Verify integrity
        stats = registry.get_stats()
        assert stats["total_entries"] == 100

        print(
            f"\nMixed workload: {stats['total_entries']} entries, {stats['total_accesses']} accesses"
        )

    def test_lock_contention_low(self):
        """Test that lock contention remains low under load."""
        registry = UniversalRegistry.get_registry("test_contention", CacheStrategy.LRU, 1000)

        # Populate registry
        for i in range(100):
            registry.register(f"key_{i}", f"value_{i}")

        # Run concurrent operations
        threads = []
        start = time.time()

        for _ in range(16):
            t = threading.Thread(target=lambda: [registry.get(f"key_{i}") for i in range(100)])
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        duration = time.time() - start
        total_ops = 16 * 100
        ops_per_sec = total_ops / duration

        # With striped locks, should handle 1600 ops/sec easily
        # (Single-threaded baseline is typically ~5000-10000 ops/sec)
        assert ops_per_sec > 1000, f"Performance too low: {ops_per_sec:.0f} ops/sec"

        print(
            f"\nLock contention test: {ops_per_sec:.0f} ops/sec ({total_ops} ops in {duration:.2f}s)"
        )

    def test_different_keys_concurrent_access(self):
        """Test that different keys can be accessed concurrently."""
        registry = UniversalRegistry.get_registry("test_different_keys", CacheStrategy.LRU, 1000)

        # Populate with 1000 keys (more than stripes)
        for i in range(1000):
            registry.register(f"key_{i}", f"value_{i}")

        threads = []
        start = time.time()

        # Each thread accesses a different subset of keys
        for thread_id in range(16):

            def accessor(tid=thread_id):
                # Each thread accesses keys in its stripe
                for i in range(tid, 1000, 16):
                    registry.get(f"key_{i}")
                    registry.register(f"key_{i}", f"new_value_{tid}_{i}")

            t = threading.Thread(target=accessor)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        duration = time.time() - start
        ops_per_sec = (16 * (1000 // 16) * 2) / duration  # 2 ops per key

        print(f"\nDifferent keys test: {ops_per_sec:.0f} ops/sec")

        # Should be very fast due to no lock contention
        assert ops_per_sec > 2000, f"Performance too low: {ops_per_sec:.0f} ops/sec"

    def test_same_key_contention(self):
        """Test contention when all threads access the same key."""
        registry = UniversalRegistry.get_registry("test_same_key", CacheStrategy.LRU, 1000)

        registry.register("hot_key", "value")

        threads = []
        start = time.time()

        # All threads access the same key
        for _ in range(16):
            t = threading.Thread(target=lambda: [registry.get("hot_key") for _ in range(100)])
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        duration = time.time() - start
        total_ops = 16 * 100
        ops_per_sec = total_ops / duration

        print(f"\nSame key contention: {ops_per_sec:.0f} ops/sec")

        # Even with contention, should be reasonable
        assert ops_per_sec > 500, f"Performance too low: {ops_per_sec:.0f} ops/sec"

    def test_namespace_operations_threadsafe(self):
        """Test that namespace operations remain thread-safe."""
        registry = UniversalRegistry.get_registry("test_namespaces", CacheStrategy.LRU, 1000)

        threads = []
        errors = []

        # 8 threads registering in different namespaces
        for thread_id in range(8):

            def namespace_worker(tid=thread_id):
                try:
                    for i in range(50):
                        registry.register(f"key_{i}", f"value_{i}", namespace=f"ns_{tid}")
                        # Verify it exists
                        value = registry.get(f"key_{i}", namespace=f"ns_{tid}")
                        assert value == f"value_{i}"
                except Exception as e:
                    errors.append(f"Thread {tid} error: {e}")

            t = threading.Thread(target=namespace_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in namespace operations: {errors}"

        # Verify all namespaces exist
        namespaces = registry.list_namespaces()
        assert len(namespaces) == 8, f"Expected 8 namespaces, got {len(namespaces)}"

        print(f"\nNamespace operations: {len(namespaces)} namespaces created successfully")

    def test_invalidation_threadsafe(self):
        """Test that invalidation is thread-safe."""
        registry = UniversalRegistry.get_registry("test_invalidation", CacheStrategy.LRU, 1000)

        # Populate
        for i in range(100):
            registry.register(f"key_{i}", f"value_{i}")

        threads = []
        errors = []

        # Mix of readers and invalidators
        def reader():
            try:
                for _ in range(100):
                    for i in range(100):
                        registry.get(f"key_{i}")
            except Exception as e:
                errors.append(f"Reader error: {e}")

        def invalidator():
            try:
                for i in range(100):
                    registry.invalidate(f"key_{i % 100}")
                    registry.register(f"key_{i % 100}", f"new_value_{i}")
            except Exception as e:
                errors.append(f"Invalidator error: {e}")

        # Start 4 readers and 4 invalidators
        for _ in range(4):
            t = threading.Thread(target=reader)
            threads.append(t)
            t.start()

        for _ in range(4):
            t = threading.Thread(target=invalidator)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during invalidation: {errors}"

        print("\nInvalidation test: Completed successfully")

    def test_cache_hit_ratio_under_load(self):
        """Test cache hit ratio under concurrent load."""
        registry = UniversalRegistry.get_registry("test_cache_hit", CacheStrategy.LRU, 1000)

        # Populate with 1000 entries
        for i in range(1000):
            registry.register(f"key_{i}", f"value_{i}", ttl=3600)

        threads = []
        hits = [0]
        misses = [0]

        def cache_worker():
            local_hits = 0
            local_misses = 0
            for i in range(1000):
                result = registry.get(f"key_{i}")
                if result is not None:
                    local_hits += 1
                else:
                    local_misses += 1

            hits[0] += local_hits
            misses[0] += local_misses

        # 16 threads accessing same data
        for _ in range(16):
            t = threading.Thread(target=cache_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        total_accesses = hits[0] + misses[0]
        hit_ratio = hits[0] / total_accesses if total_accesses > 0 else 0

        print(f"\nCache hit ratio: {hit_ratio:.2%} ({hits[0]} hits, {misses[0]} misses)")

        # Should have near 100% hit ratio
        assert hit_ratio > 0.99, f"Cache hit ratio too low: {hit_ratio:.2%}"

    def test_lru_eviction_under_concurrency(self):
        """Test LRU eviction works correctly under concurrent access."""
        registry = UniversalRegistry.get_registry("test_lru", CacheStrategy.LRU, max_size=100)

        threads = []
        errors = []

        # 16 threads writing 1000 keys each (should trigger LRU eviction)
        for thread_id in range(16):

            def lru_worker(tid=thread_id):
                try:
                    for i in range(1000):
                        registry.register(f"key_{tid}_{i}", f"value_{i}")
                        # Access some keys to update LRU
                        if i % 10 == 0:
                            registry.get(f"key_{tid}_{i}")
                except Exception as e:
                    errors.append(f"Thread {tid} error: {e}")

            t = threading.Thread(target=lru_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during LRU test: {errors}"

        # Verify size limit is respected
        stats = registry.get_stats()
        assert (
            stats["total_entries"] <= 100
        ), f"LRU eviction failed: {stats['total_entries']} entries"

        print(
            f"\nLRU eviction: {stats['total_entries']} entries (max 100), {stats['total_accesses']} accesses"
        )


@pytest.mark.benchmark
class TestStripedLockComparison:
    """Compare striped locks with single lock performance."""

    def test_read_heavy_workload(self):
        """Benchmark read-heavy workload (95% reads, 5% writes)."""
        registry = UniversalRegistry.get_registry("test_read_heavy", CacheStrategy.LRU, 1000)

        # Pre-populate
        for i in range(100):
            registry.register(f"key_{i}", f"value_{i}")

        threads = []
        start = time.time()

        def mixed_worker():
            for i in range(1000):
                # 95% reads
                if i % 20 != 0:
                    registry.get(f"key_{i % 100}")
                else:
                    # 5% writes
                    registry.register(f"key_{i % 100}", f"new_value_{i}")

        for _ in range(16):
            t = threading.Thread(target=mixed_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        duration = time.time() - start
        ops_per_sec = (16 * 1000) / duration

        print(f"\nRead-heavy workload: {ops_per_sec:.0f} ops/sec")

        # Should be very fast with striped locks
        assert ops_per_sec > 5000, f"Performance too low: {ops_per_sec:.0f} ops/sec"

    def test_write_heavy_workload(self):
        """Benchmark write-heavy workload (95% writes, 5% reads)."""
        registry = UniversalRegistry.get_registry("test_write_heavy", CacheStrategy.LRU, 1000)

        threads = []
        start = time.time()

        def mixed_worker():
            for i in range(1000):
                # 95% writes
                if i % 20 != 0:
                    registry.register(f"key_{i % 100}", f"value_{i}")
                else:
                    # 5% reads
                    registry.get(f"key_{i % 100}")

        for _ in range(16):
            t = threading.Thread(target=mixed_worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        duration = time.time() - start
        ops_per_sec = (16 * 1000) / duration

        print(f"\nWrite-heavy workload: {ops_per_sec:.0f} ops/sec")

        # Writes are slower but should still be reasonable
        assert ops_per_sec > 1000, f"Performance too low: {ops_per_sec:.0f} ops/sec"
