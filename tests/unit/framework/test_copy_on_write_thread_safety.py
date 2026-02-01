"""Thread-safety tests for CopyOnWriteState.

Tests concurrent read/write operations to verify thread-safety guarantees.
"""

import asyncio
import concurrent.futures
import threading
import time
from typing import Any

import pytest

from victor.framework.graph import CopyOnWriteState


class TestCopyOnWriteStateThreadSafety:
    """Thread-safety tests for CopyOnWriteState."""

    def test_concurrent_reads_unmodified(self):
        """Test concurrent reads from unmodified state."""
        source = {"key1": "value1", "key2": "value2", "key3": "value3"}
        cow = CopyOnWriteState(source)
        results = []
        errors = []

        def read_worker(key: str):
            try:
                for _ in range(100):
                    value = cow[key]
                    results.append((key, value))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_worker, args=(f"key{i}",)) for i in range(1, 4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent reads: {errors}"
        assert len(results) == 300
        assert not cow.was_modified

    def test_concurrent_writes(self):
        """Test concurrent writes to state."""
        source: dict[str, Any] = {"initial": "value"}
        cow = CopyOnWriteState(source)
        errors = []

        def write_worker(thread_id: int):
            try:
                for i in range(50):
                    cow[f"thread_{thread_id}_key_{i}"] = f"value_{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_worker, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent writes: {errors}"
        assert cow.was_modified

        # Verify all writes succeeded
        final_state = cow.get_state()
        for thread_id in range(5):
            for i in range(50):
                key = f"thread_{thread_id}_key_{i}"
                assert key in final_state, f"Missing key: {key}"
                assert final_state[key] == f"value_{i}"

    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        source = {"counter": 0}
        cow = CopyOnWriteState(source)
        read_results = []
        errors = []

        def reader():
            try:
                for _ in range(100):
                    value = cow.get("counter", 0)
                    read_results.append(value)
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(("reader", e))

        def writer():
            try:
                for i in range(100):
                    cow["counter"] = i
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(("writer", e))

        reader_threads = [threading.Thread(target=reader) for _ in range(3)]
        writer_thread = threading.Thread(target=writer)

        for t in reader_threads:
            t.start()
        writer_thread.start()

        for t in reader_threads:
            t.join()
        writer_thread.join()

        assert not errors, f"Errors during concurrent read/write: {errors}"
        # All read values should be valid integers
        assert all(isinstance(v, int) and 0 <= v < 100 for v in read_results)

    def test_concurrent_iteration(self):
        """Test iterating while modifying."""
        source = {f"key_{i}": i for i in range(10)}
        cow = CopyOnWriteState(source)
        iteration_results = []
        errors = []

        def iterator():
            try:
                for _ in range(20):
                    keys = list(cow.keys())
                    iteration_results.append(len(keys))
            except Exception as e:
                errors.append(("iterator", e))

        def modifier():
            try:
                for i in range(10, 30):
                    cow[f"key_{i}"] = i
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("modifier", e))

        iter_threads = [threading.Thread(target=iterator) for _ in range(3)]
        mod_thread = threading.Thread(target=modifier)

        for t in iter_threads:
            t.start()
        mod_thread.start()

        for t in iter_threads:
            t.join()
        mod_thread.join()

        assert not errors, f"Errors during concurrent iteration: {errors}"
        # Final state should have all keys
        assert len(cow.keys()) == 30

    def test_concurrent_update(self):
        """Test concurrent update operations."""
        source: dict[str, Any] = {}
        cow = CopyOnWriteState(source)
        errors = []

        def updater(thread_id: int):
            try:
                for i in range(20):
                    cow.update({f"t{thread_id}_k{i}": f"v{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent update: {errors}"
        # All keys should be present
        final = cow.get_state()
        assert len(final) == 100  # 5 threads * 20 keys each

    def test_concurrent_pop(self):
        """Test concurrent pop operations."""
        source = {f"key_{i}": i for i in range(100)}
        cow = CopyOnWriteState(source)
        popped_values = []
        errors = []
        lock = threading.Lock()

        def popper(keys_to_pop):
            try:
                for key in keys_to_pop:
                    try:
                        value = cow.pop(key)
                        with lock:
                            popped_values.append((key, value))
                    except KeyError:
                        # Key already popped by another thread
                        pass
            except Exception as e:
                errors.append(e)

        # Split keys among threads
        all_keys = [f"key_{i}" for i in range(100)]
        chunks = [all_keys[i::5] for i in range(5)]

        threads = [threading.Thread(target=popper, args=(chunk,)) for chunk in chunks]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent pop: {errors}"
        # All keys should have been popped exactly once
        assert len(popped_values) == 100
        assert len(cow.get_state()) == 0

    def test_concurrent_setdefault(self):
        """Test concurrent setdefault operations."""
        source: dict[str, Any] = {}
        cow = CopyOnWriteState(source)
        results = []
        errors = []
        lock = threading.Lock()

        def setter(thread_id: int):
            try:
                for i in range(20):
                    key = f"shared_key_{i}"
                    value = cow.setdefault(key, f"thread_{thread_id}")
                    with lock:
                        results.append((key, value))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=setter, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent setdefault: {errors}"
        # Each key should have a value from one of the threads
        final = cow.get_state()
        assert len(final) == 20  # 20 unique keys

    def test_thread_pool_executor(self):
        """Test with ThreadPoolExecutor for realistic concurrent usage."""
        source = {"count": 0}
        cow = CopyOnWriteState(source)

        def increment():
            current = cow.get("count", 0)
            cow["count"] = current + 1
            return cow["count"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(increment) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Due to race conditions in read-then-write, final count may vary
        # but all operations should complete without errors
        assert len(results) == 100
        assert cow.was_modified

    def test_no_deadlock_nested_operations(self):
        """Test that nested operations don't cause deadlock (RLock)."""
        source = {"a": {"b": {"c": 1}}}
        cow = CopyOnWriteState(source)

        def nested_operation():
            # This should not deadlock due to RLock
            if "a" in cow:
                cow["a"] = {"b": {"c": cow["a"]["b"]["c"] + 1}}
            return cow.get_state()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(nested_operation) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20

    def test_copy_isolation(self):
        """Test that copy() returns isolated snapshot."""
        source = {"key": "original"}
        cow = CopyOnWriteState(source)
        errors = []

        def modifier():
            try:
                for i in range(50):
                    cow["key"] = f"modified_{i}"
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def copier():
            copies = []
            try:
                for _ in range(50):
                    snapshot = cow.copy()
                    copies.append(snapshot)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
            return copies

        mod_thread = threading.Thread(target=modifier)
        copy_thread = threading.Thread(target=copier)

        mod_thread.start()
        copy_thread.start()

        mod_thread.join()
        copy_thread.join()

        assert not errors

    def test_to_dict_thread_safe(self):
        """Test to_dict returns consistent snapshot."""
        source = {f"key_{i}": i for i in range(10)}
        cow = CopyOnWriteState(source)
        snapshots = []
        errors = []

        def modifier():
            try:
                for i in range(10, 50):
                    cow[f"key_{i}"] = i
            except Exception as e:
                errors.append(e)

        def snapshot_taker():
            try:
                for _ in range(20):
                    snap = cow.to_dict()
                    snapshots.append(len(snap))
            except Exception as e:
                errors.append(e)

        mod_thread = threading.Thread(target=modifier)
        snap_threads = [threading.Thread(target=snapshot_taker) for _ in range(3)]

        mod_thread.start()
        for t in snap_threads:
            t.start()

        mod_thread.join()
        for t in snap_threads:
            t.join()

        assert not errors
        # All snapshots should be valid lengths
        assert all(10 <= s <= 50 for s in snapshots)


class TestCopyOnWriteStateAsyncSafety:
    """Test async safety (asyncio tasks sharing state)."""

    @pytest.mark.asyncio
    async def test_async_concurrent_access(self):
        """Test concurrent access from asyncio tasks."""
        source = {"counter": 0}
        cow = CopyOnWriteState(source)

        async def incrementer(task_id: int):
            for i in range(10):
                current = cow.get("counter", 0)
                cow["counter"] = current + 1
                cow[f"task_{task_id}_iter_{i}"] = True
                await asyncio.sleep(0.001)

        tasks = [incrementer(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # All task keys should be present
        final = cow.get_state()
        for task_id in range(5):
            for i in range(10):
                assert f"task_{task_id}_iter_{i}" in final

    @pytest.mark.asyncio
    async def test_async_reader_writer(self):
        """Test async readers and writers."""
        source = {"value": "initial"}
        cow = CopyOnWriteState(source)
        read_values = []

        async def reader():
            for _ in range(20):
                value = cow.get("value")
                read_values.append(value)
                await asyncio.sleep(0.001)

        async def writer():
            for i in range(20):
                cow["value"] = f"updated_{i}"
                await asyncio.sleep(0.001)

        await asyncio.gather(
            reader(),
            reader(),
            writer(),
        )

        assert len(read_values) == 40
        # All reads should return valid strings
        assert all(isinstance(v, str) for v in read_values)
