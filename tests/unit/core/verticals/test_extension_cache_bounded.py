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

"""Tests for bounded extension cache with thread safety (Phase 1.1).

Tests that the extension cache:
1. Respects max size limits
2. Is thread-safe under concurrent access
3. Uses LRU eviction policy
4. Properly handles cache invalidation
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import pytest

from victor.core.verticals.extension_loader import (
    ExtensionCacheEntry,
    VerticalExtensionLoader,
)


class TestBoundedExtensionCache:
    """Test bounded extension cache functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clean up cache before and after each test."""
        VerticalExtensionLoader._extensions_cache.clear()
        VerticalExtensionLoader._extension_versions.clear()
        yield
        VerticalExtensionLoader._extensions_cache.clear()
        VerticalExtensionLoader._extension_versions.clear()

    def test_cache_respects_max_size_limit(self):
        """Cache should not exceed _cache_max_size entries."""
        # Get the max size (should be configurable)
        max_size = getattr(VerticalExtensionLoader, "_cache_max_size", 1000)

        # Add more entries than max_size
        for i in range(max_size + 100):
            cache_key = f"TestVertical{i}:middleware"
            entry = ExtensionCacheEntry(value=f"value_{i}", ttl=3600)
            VerticalExtensionLoader._extensions_cache[cache_key] = entry

            # Trigger eviction if needed (simulate what _get_cached_extension does)
            VerticalExtensionLoader._evict_lru_if_needed()

        # Cache should not exceed max_size
        assert len(VerticalExtensionLoader._extensions_cache) <= max_size

    def test_cache_eviction_uses_lru_policy(self):
        """Oldest accessed entries should be evicted first."""
        # Set a small max size for testing
        original_max = getattr(VerticalExtensionLoader, "_cache_max_size", 1000)
        VerticalExtensionLoader._cache_max_size = 5

        try:
            # Add entries
            for i in range(5):
                cache_key = f"TestVertical{i}:ext"
                entry = ExtensionCacheEntry(value=f"value_{i}", ttl=3600)
                VerticalExtensionLoader._extensions_cache[cache_key] = entry
                VerticalExtensionLoader._evict_lru_if_needed()

            # Access entry 0 and 1 to make them recently used
            if "TestVertical0:ext" in VerticalExtensionLoader._extensions_cache:
                VerticalExtensionLoader._extensions_cache.move_to_end("TestVertical0:ext")
            if "TestVertical1:ext" in VerticalExtensionLoader._extensions_cache:
                VerticalExtensionLoader._extensions_cache.move_to_end("TestVertical1:ext")

            # Add a new entry which should trigger eviction of oldest (entry 2)
            cache_key = "TestVertical5:ext"
            entry = ExtensionCacheEntry(value="value_5", ttl=3600)
            VerticalExtensionLoader._extensions_cache[cache_key] = entry
            VerticalExtensionLoader._evict_lru_if_needed()

            # Entry 2 should have been evicted (it was oldest unreferenced)
            assert "TestVertical2:ext" not in VerticalExtensionLoader._extensions_cache
            # Entry 0 and 1 should still exist (recently accessed)
            assert "TestVertical0:ext" in VerticalExtensionLoader._extensions_cache
            assert "TestVertical1:ext" in VerticalExtensionLoader._extensions_cache

        finally:
            VerticalExtensionLoader._cache_max_size = original_max

    def test_cache_thread_safety_under_concurrent_access(self):
        """Cache should be thread-safe under concurrent access."""
        errors = []
        success_count = [0]

        def worker(thread_id: int):
            """Worker that reads and writes to cache."""
            try:
                for i in range(50):
                    # Write operation
                    cache_key = f"TestVertical{thread_id}_{i}:ext"
                    entry = ExtensionCacheEntry(value=f"value_{thread_id}_{i}", ttl=3600)

                    # Use the lock for thread-safe access
                    with VerticalExtensionLoader._cache_lock:
                        VerticalExtensionLoader._extensions_cache[cache_key] = entry
                        VerticalExtensionLoader._evict_lru_if_needed()

                    # Read operation
                    with VerticalExtensionLoader._cache_lock:
                        _ = VerticalExtensionLoader._extensions_cache.get(cache_key)

                success_count[0] += 1
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()  # Raises exception if worker failed

        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert success_count[0] == 10, "Not all workers completed"

    def test_cache_invalidation_clears_entries(self):
        """Cache invalidation should properly clear entries."""
        # Add some entries
        for i in range(10):
            cache_key = f"TestVertical{i}:middleware"
            entry = ExtensionCacheEntry(value=f"value_{i}", ttl=3600)
            VerticalExtensionLoader._extensions_cache[cache_key] = entry

        initial_count = len(VerticalExtensionLoader._extensions_cache)
        assert initial_count == 10

        # Clear all
        VerticalExtensionLoader.clear_extension_cache(clear_all=True)

        # All should be cleared
        assert len(VerticalExtensionLoader._extensions_cache) == 0

    def test_cache_lock_exists(self):
        """Cache should have a lock for thread safety."""
        assert hasattr(VerticalExtensionLoader, "_cache_lock")
        assert isinstance(VerticalExtensionLoader._cache_lock, type(threading.RLock()))

    def test_cache_max_size_is_configurable(self):
        """Cache max size should be a class variable."""
        assert hasattr(VerticalExtensionLoader, "_cache_max_size")
        assert isinstance(VerticalExtensionLoader._cache_max_size, int)
        assert VerticalExtensionLoader._cache_max_size > 0

    def test_evict_lru_method_exists(self):
        """_evict_lru_if_needed method should exist."""
        assert hasattr(VerticalExtensionLoader, "_evict_lru_if_needed")
        assert callable(VerticalExtensionLoader._evict_lru_if_needed)


class TestExtensionCacheEntryOrdering:
    """Test that cache uses OrderedDict for LRU ordering."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clean up cache before and after each test."""
        VerticalExtensionLoader._extensions_cache.clear()
        yield
        VerticalExtensionLoader._extensions_cache.clear()

    def test_cache_is_ordered_dict(self):
        """Cache should be an OrderedDict for LRU support."""
        from collections import OrderedDict

        assert isinstance(VerticalExtensionLoader._extensions_cache, OrderedDict)

    def test_move_to_end_on_access(self):
        """Accessing an entry should move it to the end (most recent)."""
        # Add entries in order
        for i in range(3):
            cache_key = f"Test{i}:ext"
            entry = ExtensionCacheEntry(value=f"value_{i}", ttl=3600)
            VerticalExtensionLoader._extensions_cache[cache_key] = entry

        # Access first entry (should move to end)
        VerticalExtensionLoader._extensions_cache.move_to_end("Test0:ext")

        # First entry should now be Test1:ext (the oldest after move)
        first_key = next(iter(VerticalExtensionLoader._extensions_cache))
        assert first_key == "Test1:ext"

        # Last entry should be Test0:ext (just moved to end)
        last_key = list(VerticalExtensionLoader._extensions_cache.keys())[-1]
        assert last_key == "Test0:ext"


class TestConcurrentCacheOperations:
    """Test concurrent cache operations don't cause race conditions."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clean up cache before and after each test."""
        VerticalExtensionLoader._extensions_cache.clear()
        VerticalExtensionLoader._extension_versions.clear()
        yield
        VerticalExtensionLoader._extensions_cache.clear()
        VerticalExtensionLoader._extension_versions.clear()

    def test_concurrent_reads_and_writes(self):
        """Concurrent reads and writes should not corrupt the cache."""
        barrier = threading.Barrier(4)
        results = {"reads": 0, "writes": 0, "errors": []}

        def reader():
            barrier.wait()  # Synchronize start
            try:
                for _ in range(100):
                    with VerticalExtensionLoader._cache_lock:
                        # Read random key
                        for key in list(VerticalExtensionLoader._extensions_cache.keys())[:5]:
                            _ = VerticalExtensionLoader._extensions_cache.get(key)
                    results["reads"] += 1
            except Exception as e:
                results["errors"].append(f"Reader: {e}")

        def writer():
            barrier.wait()  # Synchronize start
            try:
                for i in range(100):
                    with VerticalExtensionLoader._cache_lock:
                        cache_key = f"Writer{threading.current_thread().name}_{i}:ext"
                        entry = ExtensionCacheEntry(value=f"value_{i}", ttl=3600)
                        VerticalExtensionLoader._extensions_cache[cache_key] = entry
                        VerticalExtensionLoader._evict_lru_if_needed()
                    results["writes"] += 1
            except Exception as e:
                results["errors"].append(f"Writer: {e}")

        # Run 2 readers and 2 writers concurrently
        threads = [
            threading.Thread(target=reader, name="reader1"),
            threading.Thread(target=reader, name="reader2"),
            threading.Thread(target=writer, name="writer1"),
            threading.Thread(target=writer, name="writer2"),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Should have no errors
        assert len(results["errors"]) == 0, f"Errors: {results['errors']}"

    def test_concurrent_invalidation(self):
        """Concurrent invalidation should not cause issues."""
        # Pre-populate cache
        for i in range(100):
            cache_key = f"Test{i}:ext"
            entry = ExtensionCacheEntry(value=f"value_{i}", ttl=3600)
            VerticalExtensionLoader._extensions_cache[cache_key] = entry

        errors = []

        def invalidator():
            try:
                VerticalExtensionLoader.clear_extension_cache(clear_all=True)
            except Exception as e:
                errors.append(str(e))

        def writer():
            try:
                for i in range(50):
                    cache_key = f"NewTest{i}:ext"
                    entry = ExtensionCacheEntry(value=f"new_value_{i}", ttl=3600)
                    with VerticalExtensionLoader._cache_lock:
                        VerticalExtensionLoader._extensions_cache[cache_key] = entry
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=invalidator),
            threading.Thread(target=writer),
            threading.Thread(target=invalidator),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Should have no errors
        assert len(errors) == 0, f"Errors: {errors}"
