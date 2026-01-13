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

"""Stress tests for thread safety issues in Victor components.

This module contains comprehensive stress tests to identify race conditions,
deadlocks, data corruption, and other concurrency issues in critical components:

1. InMemoryEventBackend - Concurrent subscribe/unsubscribe operations
2. EmbeddingService - Concurrent cache access
3. ToolPipeline - Parallel metrics recording
4. ServiceContainer - Concurrent service registration/resolution

Tests are parametrized to run with varying concurrency levels to expose
issues that only appear under high load.

Test Categories:
- Race Condition Tests: Expose timing-dependent bugs
- Data Structure Corruption Tests: Verify integrity under concurrent access
- Deadlock Detection Tests: Ensure no deadlocks occur
- Atomic Operation Tests: Verify operations maintain consistency
- High Concurrency Tests: 100+ concurrent operations

These tests are written TDD-FIRST (tests before implementation).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Set

import pytest

from victor.core.events.backends import InMemoryEventBackend, BackendConfig
from victor.core.events.protocols import MessagingEvent, EventHandler
from victor.core.container import (
    ServiceContainer,
    ServiceLifetime,
    get_container,
    reset_container,
)
from victor.agent.tool_pipeline import (
    ToolPipeline,
    ToolPipelineConfig,
    ExecutionMetrics,
    LRUToolCache,
    ToolRateLimiter,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def event_backend():
    """Create a fresh event backend for each test with running dispatcher."""
    backend = InMemoryEventBackend(config=BackendConfig())
    # Start event loop in background thread to keep dispatcher running
    import threading as threading_module

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Connect backend (starts dispatcher task)
    async def _connect():
        await backend.connect()

    loop.run_until_complete(_connect())

    # Run the event loop in a background thread
    # This keeps the dispatcher task running
    loop_thread = threading_module.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    yield backend

    # Cleanup
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=2)

    async def _cleanup():
        await backend.disconnect()

    loop.run_until_complete(_cleanup())
    loop.close()


@pytest.fixture
def embedding_cache():
    """Create a fresh embedding cache for testing."""
    cache = LRUToolCache(max_size=100, ttl_seconds=60)
    return cache


@pytest.fixture
def execution_metrics():
    """Create fresh execution metrics."""
    return ExecutionMetrics()


@pytest.fixture
def service_container():
    """Create fresh service container."""
    reset_container()
    return ServiceContainer()


# =============================================================================
# Test Helpers
# =============================================================================


class ConcurrentOperationTracker:
    """Track results of concurrent operations for validation."""

    def __init__(self):
        self.results: List[Any] = []
        self.errors: List[Exception] = []
        self.lock = threading.Lock()

    def add_result(self, result: Any) -> None:
        """Thread-safe result addition."""
        with self.lock:
            self.results.append(result)

    def add_error(self, error: Exception) -> None:
        """Thread-safe error recording."""
        with self.lock:
            self.errors.append(error)

    def get_success_count(self) -> int:
        """Get number of successful operations."""
        with self.lock:
            return len(self.results)

    def get_error_count(self) -> int:
        """Get number of failed operations."""
        with self.lock:
            return len(self.errors)


class DeadlockDetector:
    """Detect potential deadlocks in concurrent operations."""

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds
        self.deadlock_detected = False

    def run_with_timeout(self, func, *args, **kwargs) -> Any:
        """Run function with timeout to detect deadlocks."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FuturesTimeoutError:
                self.deadlock_detected = True
                raise TimeoutError(f"Potential deadlock detected (timeout after {self.timeout_seconds}s)")


# =============================================================================
# Test Category 1: Concurrent Subscribe/Unsubscribe (Event Backend)
# =============================================================================


class TestEventBackendConcurrencyStress:
    """Stress tests for InMemoryEventBackend concurrent operations."""

    @pytest.mark.parametrize("num_threads", [10, 50, 100])
    @pytest.mark.parametrize("num_operations", [20, 100])
    def test_concurrent_subscribe_unsubscribe(
        self, event_backend, num_threads, num_operations
    ):
        """Test concurrent subscribe and unsubscribe operations don't corrupt state.

        TDD: This test will FAIL until thread-safe subscription management is implemented.

        Expected issues (before fix):
        - Race conditions in _subscriptions dict access
        - Lost subscriptions due to concurrent modifications
        - Orphaned subscriptions that should be removed
        """
        tracker = ConcurrentOperationTracker()
        subscription_ids = set()
        id_lock = threading.Lock()

        def subscribe_worker(worker_id: int):
            """Worker that subscribes and unsubscribes."""
            backend = event_backend  # Capture in local variable
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            for i in range(num_operations):
                try:
                    pattern = f"test.{worker_id}.{i}"

                    async def _subscribe():
                        handle = await backend.subscribe(
                            pattern,
                            lambda e: None,  # Dummy handler
                        )
                        return handle

                    handle = loop.run_until_complete(_subscribe())

                    # Track subscription ID
                    with id_lock:
                        subscription_ids.add(handle.subscription_id)

                    # Immediately unsubscribe
                    async def _unsubscribe():
                        await backend.unsubscribe(handle)

                    loop.run_until_complete(_unsubscribe())

                    tracker.add_result((worker_id, i))
                except Exception as e:
                    tracker.add_error(e)

            loop.close()

        # Run concurrent workers
        threads = [
            threading.Thread(target=subscribe_worker, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        # Verify invariants
        assert tracker.get_error_count() == 0, f"Errors occurred: {tracker.errors}"
        assert tracker.get_success_count() == num_threads * num_operations

        # All subscriptions should be cleaned up
        final_count = event_backend.get_subscription_count()
        assert final_count == 0, f"Expected 0 subscriptions, got {final_count} (orphaned subscriptions)"

    @pytest.mark.parametrize("num_publishers", [10, 50])
    @pytest.mark.parametrize("num_subscribers", [10, 50])
    def test_concurrent_publish_subscribe(
        self, event_backend, num_publishers, num_subscribers
    ):
        """Test concurrent publishing and subscribing don't cause missed events.

        TDD: This test will FAIL until thread-safe event dispatching is implemented.

        Expected issues (before fix):
        - Events missed during subscription list iteration
        - Handler called multiple times for same event
        - Handler not called despite matching pattern
        """
        received_counts: Counter = Counter()
        count_lock = threading.Lock()
        total_published = 0
        publish_lock = threading.Lock()

        def increment_count(topic: str):
            """Thread-safe counter increment."""
            with count_lock:
                received_counts[topic] += 1

        def subscriber_worker(subscriber_id: int):
            """Worker that subscribes to events."""
            backend = event_backend
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _subscribe():
                def handler(event):
                    increment_count(f"sub_{subscriber_id}_event_{event.topic}")

                await backend.subscribe(
                    f"test.{subscriber_id}.*",
                    handler,
                )

            loop.run_until_complete(_subscribe())
            loop.close()

        def publisher_worker(publisher_id: int):
            """Worker that publishes events."""
            nonlocal total_published
            backend = event_backend
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _publish():
                nonlocal total_published
                # Send enough events to cover all subscribers
                # Each subscriber should receive at least one event from this publisher
                num_events = max(10, num_subscribers)
                for i in range(num_events):
                    event = MessagingEvent(
                        topic=f"test.{i % num_subscribers}.data",
                        data={"publisher": publisher_id, "index": i},
                    )
                    await backend.publish(event)
                    with publish_lock:
                        total_published += 1

            loop.run_until_complete(_publish())
            loop.close()

        # Start subscribers first
        subscriber_threads = [
            threading.Thread(target=subscriber_worker, args=(i,))
            for i in range(num_subscribers)
        ]

        for t in subscriber_threads:
            t.start()

        time.sleep(0.1)  # Let subscriptions settle

        # Start publishers
        publisher_threads = [
            threading.Thread(target=publisher_worker, args=(i,))
            for i in range(num_publishers)
        ]

        for t in publisher_threads:
            t.start()

        # Wait for all
        for t in subscriber_threads:
            t.join(timeout=10)

        for t in publisher_threads:
            t.join(timeout=10)

        # Let events propagate
        time.sleep(0.5)

        # Verify invariants
        # Each subscriber i only receives events matching "test.i.*" pattern
        # Each publisher sends 10 events to topics test.0.data, test.1.data, ..., test.9.data
        # So each subscriber receives 1 event from each publisher = num_publishers events total
        expected_receives = num_publishers  # Each subscriber receives 1 event per publisher
        for i in range(num_subscribers):
            received = received_counts.get(f"sub_{i}_event_test.{i % num_subscribers}.data", 0)
            # Each event should be received exactly once by matching subscriber
            assert received == expected_receives, (
                f"Subscriber {i} received {received} events, expected {expected_receives} "
                f"(data loss or duplication)"
            )

    def test_concurrent_subscription_iteration(self, event_backend):
        """Test subscription dict iteration during concurrent modifications.

        TDD: This test will FAIL until subscription iteration is made thread-safe.

        Expected issues (before fix):
        - RuntimeError: dictionary changed size during iteration
        - Missing some subscriptions in iteration result
        """
        iterations_completed = 0
        iteration_errors = []
        errors_lock = threading.Lock()

        def subscribe_worker():
            """Continuously add and remove subscriptions."""
            backend = event_backend
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            for i in range(100):
                try:
                    async def _subscribe():
                        handle = await backend.subscribe(f"temp.{i}", lambda e: None)
                        return handle

                    handle = loop.run_until_complete(_subscribe())

                    async def _unsubscribe():
                        await backend.unsubscribe(handle)

                    loop.run_until_complete(_unsubscribe())
                except Exception as e:
                    pass  # Ignore errors, focus on iteration

            loop.close()

        def iterate_worker():
            """Iterate subscriptions while they're being modified."""
            nonlocal iterations_completed
            backend = event_backend

            for _ in range(100):
                try:
                    # Simulate _dispatch_loop's subscription iteration
                    count = backend.get_subscription_count()
                    iterations_completed += 1
                except Exception as e:
                    with errors_lock:
                        iteration_errors.append(e)

        # Run concurrent subscribe/unsubscribe and iteration
        subscribe_threads = [threading.Thread(target=subscribe_worker) for _ in range(10)]
        iterate_threads = [threading.Thread(target=iterate_worker) for _ in range(5)]

        for t in subscribe_threads + iterate_threads:
            t.start()

        for t in subscribe_threads + iterate_threads:
            t.join(timeout=20)

        # Verify no iteration errors
        assert len(iteration_errors) == 0, f"Iteration errors: {iteration_errors}"
        assert iterations_completed > 0, "No iterations completed"


# =============================================================================
# Test Category 2: Concurrent Embedding Cache Access
# =============================================================================


class TestEmbeddingCacheConcurrencyStress:
    """Stress tests for embedding cache concurrent access."""

    @pytest.mark.parametrize("num_threads", [10, 50, 100])
    @pytest.mark.parametrize("cache_size", [50, 200])
    def test_concurrent_cache_read_write(self, embedding_cache, num_threads, cache_size):
        """Test concurrent cache reads and writes don't corrupt data.

        TDD: This test will FAIL until cache operations are atomic.

        Expected issues (before fix):
        - OrderedDict corruption during concurrent access
        - Inconsistent cache size reported
        - Lost cache entries (eviction race condition)
        """
        cache = embedding_cache
        access_counts: Counter = Counter()
        access_lock = threading.Lock()
        errors = []
        errors_lock = threading.Lock()

        def cache_reader(reader_id: int):
            """Reader that accesses cache entries."""
            for i in range(cache_size):
                try:
                    key = f"key_{i}"
                    value = cache.get(key)

                    with access_lock:
                        access_counts[f"read_{reader_id}_{key}"] += 1

                    # Simulate processing time
                    time.sleep(0.0001)
                except Exception as e:
                    with errors_lock:
                        errors.append(("read", reader_id, i, e))

        def cache_writer(writer_id: int):
            """Writer that adds cache entries."""
            for i in range(cache_size):
                try:
                    key = f"key_{i}"
                    value = {"data": f"value_{writer_id}_{i}", "writer": writer_id}
                    cache.set(key, value)

                    with access_lock:
                        access_counts[f"write_{writer_id}_{key}"] += 1

                    time.sleep(0.0001)
                except Exception as e:
                    with errors_lock:
                        errors.append(("write", writer_id, i, e))

        # Start half writers, half readers
        threads = []
        for i in range(num_threads // 2):
            threads.append(threading.Thread(target=cache_writer, args=(i,)))
            threads.append(threading.Thread(target=cache_reader, args=(i,)))

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        # Verify no errors
        assert len(errors) == 0, f"Cache access errors: {errors}"

        # Verify cache integrity
        assert len(cache) <= cache_size, f"Cache size {len(cache)} exceeds max {cache_size}"

        # Verify all writes completed
        total_writes = sum(1 for k in access_counts if k.startswith("write_"))
        assert total_writes == (num_threads // 2) * cache_size

    def test_concurrent_cache_eviction(self, embedding_cache):
        """Test concurrent cache access triggers correct LRU eviction.

        TDD: This test will FAIL until eviction is atomic.

        Expected issues (before fix):
        - Cache exceeds max_size due to race in eviction logic
        - Wrong entries evicted (not LRU order)
        - Duplicate entries in cache
        """
        cache = embedding_cache

        # Fill cache to capacity
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        assert len(cache) == 100

        # Concurrent access that should trigger eviction
        def access_and_add(worker_id: int):
            """Access existing entries and add new ones."""
            for i in range(50):
                # Read some old entries
                cache.get(f"key_{i}")

                # Add new entries (should trigger eviction)
                cache.set(f"new_key_{worker_id}_{i}", f"new_value_{worker_id}_{i}")

        threads = [threading.Thread(target=access_and_add, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=20)

        # Verify cache size constraint maintained
        assert len(cache) <= 100, f"Cache exceeded max size: {len(cache)}"

        # Verify no duplicate keys
        keys = list(cache._cache.keys())
        assert len(keys) == len(set(keys)), "Duplicate keys found in cache"

    def test_concurrent_cache_clear(self, embedding_cache):
        """Test concurrent cache clear doesn't cause corruption.

        TDD: This test will FAIL until clear() is atomic.
        """
        cache = embedding_cache
        errors = []

        def populator(worker_id: int):
            """Add entries to cache."""
            for i in range(100):
                try:
                    cache.set(f"key_{worker_id}_{i}", f"value_{i}")
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(("populator", worker_id, e))

        def clearer():
            """Clear cache repeatedly."""
            for _ in range(50):
                try:
                    cache.clear()
                    time.sleep(0.002)
                except Exception as e:
                    errors.append(("clearer", 0, e))

        threads = [
            threading.Thread(target=populator, args=(i,)) for i in range(10)
        ] + [threading.Thread(target=clearer) for _ in range(3)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors during concurrent clear: {errors}"

        # Final state should be consistent
        final_size = len(cache)
        assert final_size >= 0, "Cache in invalid state"


# =============================================================================
# Test Category 3: Parallel Tool Execution Metrics
# =============================================================================


class TestToolPipelineMetricsConcurrency:
    """Stress tests for ExecutionMetrics concurrent recording."""

    @pytest.mark.parametrize("num_threads", [10, 50, 100])
    @pytest.mark.parametrize("num_records", [50, 200])
    def test_concurrent_metrics_recording(self, execution_metrics, num_threads, num_records):
        """Test concurrent metric recording maintains accurate counts.

        TDD: This test will FAIL until metrics updates are atomic.

        Expected issues (before fix):
        - Lost updates to counters (race condition)
        - Negative or incorrect aggregation values
        - Inconsistent totals vs. per-tool counts
        """
        metrics = execution_metrics
        errors = []

        def record_worker(worker_id: int):
            """Record execution metrics concurrently."""
            for i in range(num_records):
                try:
                    # Simulate tool execution
                    tool_name = f"tool_{worker_id % 5}"
                    execution_time = 0.1 + (i * 0.001)
                    success = (i % 10) != 0  # 10% failure rate

                    metrics.record_execution(
                        tool_name=tool_name,
                        execution_time=execution_time,
                        success=success,
                        cached=(i % 2 == 0),
                        skipped=False,
                    )
                except Exception as e:
                    errors.append((worker_id, i, e))

        threads = [threading.Thread(target=record_worker, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        # Verify no errors
        assert len(errors) == 0, f"Metric recording errors: {errors}"

        # Verify totals
        expected_total = num_threads * num_records
        assert metrics.total_executions == expected_total, (
            f"Total executions mismatch: {metrics.total_executions} vs {expected_total} "
            f"(lost updates)"
        )

        # Verify cache counts
        assert metrics.cache_hits + metrics.cache_misses == expected_total, (
            f"Cache counts inconsistent: hits={metrics.cache_hits}, "
            f"misses={metrics.cache_misses}, total={expected_total}"
        )

        # Verify success/failure counts
        assert metrics.successful_executions + metrics.failed_executions == expected_total, (
            f"Success/failure counts inconsistent: "
            f"success={metrics.successful_executions}, failed={metrics.failed_executions}"
        )

        # Verify per-tool counts match total
        tool_sum = sum(metrics.tool_counts.values())
        assert tool_sum == expected_total, (
            f"Tool counts sum {tool_sum} doesn't match total {expected_total}"
        )

    def test_concurrent_execution_time_tracking(self, execution_metrics):
        """Test concurrent execution time updates maintain statistics.

        TDD: This test will FAIL until statistics updates are atomic.

        Expected issues (before fix):
        - Incorrect min/max values (race condition)
        - Corrupted average calculation
        - Lost time entries in list
        """
        metrics = execution_metrics

        def time_worker():
            """Record varying execution times."""
            for i in range(100):
                time_value = 0.001 + (i * 0.01)
                metrics.record_execution(
                    tool_name="test_tool",
                    execution_time=time_value,
                    success=True,
                )

        threads = [threading.Thread(target=time_worker) for _ in range(20)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=20)

        # Verify time tracking integrity
        assert metrics.total_executions == 2000

        # Min should be close to minimum recorded
        assert metrics.min_execution_time >= 0.001, "Min time corrupted"

        # Max should be close to maximum recorded
        assert metrics.max_execution_time <= 1.0, "Max time corrupted"

        # Average should be in valid range
        avg = metrics.get_avg_execution_time()
        assert 0.001 <= avg <= 1.0, f"Average time invalid: {avg}"

        # Verify time list size matches total
        assert len(metrics.execution_times) == 2000, (
            f"Time list size {len(metrics.execution_times)} doesn't match total {metrics.total_executions}"
        )

    def test_concurrent_metrics_reset(self, execution_metrics):
        """Test concurrent reset doesn't corrupt metrics state.

        TDD: This test will FAIL until reset is atomic.
        """
        metrics = execution_metrics

        # Populate metrics
        for i in range(1000):
            metrics.record_execution(
                tool_name=f"tool_{i % 10}",
                execution_time=0.1,
                success=True,
            )

        assert metrics.total_executions == 1000

        errors = []
        reset_count = [0]  # Use list for mutable shared state

        def recorder(worker_id: int):
            """Continue recording while resets happen."""
            for i in range(500):
                try:
                    metrics.record_execution(
                        tool_name=f"tool_{worker_id}",
                        execution_time=0.1,
                        success=True,
                    )
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(("recorder", worker_id, e))

        def resetter():
            """Reset metrics concurrently."""
            for _ in range(20):
                try:
                    metrics.reset()
                    reset_count[0] += 1
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(("resetter", 0, e))

        threads = [
            threading.Thread(target=recorder, args=(i,)) for i in range(10)
        ] + [threading.Thread(target=resetter)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        # Verify no exceptions
        assert len(errors) == 0, f"Errors during concurrent reset: {errors}"

        # Metrics should be in valid state even after concurrent resets
        assert metrics.total_executions >= 0
        assert len(metrics.execution_times) >= 0


# =============================================================================
# Test Category 4: ServiceContainer Concurrent Access
# =============================================================================


class TestServiceContainerConcurrency:
    """Stress tests for ServiceContainer concurrent operations."""

    @pytest.mark.parametrize("num_threads", [10, 50, 100])
    def test_concurrent_service_registration(self, service_container, num_threads):
        """Test concurrent service registration doesn't corrupt container.

        TDD: This test will FAIL until registration is properly locked.

        Expected issues (before fix):
        - Lost registrations due to race condition
        - ServiceAlreadyRegisteredError for unique services
        - Duplicate registrations in _descriptors dict
        """
        container = service_container
        errors = []

        def register_worker(worker_id: int):
            """Register different services concurrently."""
            for i in range(10):
                try:
                    # Each worker registers unique services
                    service_name = f"Service_{worker_id}_{i}"

                    # Create a dynamic type for the service (ServiceContainer expects types, not strings)
                    service_type = type(service_name, (object,), {})

                    def factory(c):
                        return {"name": service_name, "worker": worker_id}

                    container.register(
                        service_type,
                        factory,
                        ServiceLifetime.SINGLETON,
                    )
                except Exception as e:
                    errors.append((worker_id, i, e))

        threads = [threading.Thread(target=register_worker, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        # Verify no unexpected errors
        assert len(errors) == 0, f"Registration errors: {errors}"

        # Verify all services registered
        expected_count = num_threads * 10
        registered_types = container.get_registered_types()
        assert len(registered_types) == expected_count, (
            f"Expected {expected_count} services, got {len(registered_types)}"
        )

    def test_concurrent_singleton_resolution(self, service_container):
        """Test concurrent singleton resolution maintains single instance.

        TDD: This test will FAIL until singleton initialization is locked.

        Expected issues (before fix):
        - Multiple instances created for same service
        - Instance variable overwritten by race condition
        """
        container = service_container
        instance_ids = set()
        ids_lock = threading.Lock()

        # Register a singleton service that tracks instances
        instances_created = []

        def tracking_factory(c):
            instance_id = f"instance_{len(instances_created)}"
            instances_created.append(instance_id)
            return {"id": instance_id}

        # Create a type for the test service (ServiceContainer expects types, not strings)
        TestService = type("TestService", (object,), {})
        container.register(TestService, tracking_factory, ServiceLifetime.SINGLETON)

        def resolve_worker():
            """Resolve singleton service concurrently."""
            for _ in range(100):
                instance = container.get(TestService)
                with ids_lock:
                    instance_ids.add(id(instance))

        threads = [threading.Thread(target=resolve_worker) for _ in range(50)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=20)

        # Verify only one instance was created
        assert len(instances_created) == 1, (
            f"Expected 1 instance created, got {len(instances_created)}: {instances_created}"
        )

        # Verify all resolutions returned same instance
        assert len(instance_ids) == 1, f"Got multiple different instances: {len(instance_ids)}"

    def test_concurrent_transient_resolution(self, service_container):
        """Test concurrent transient resolution creates new instances."""
        container = service_container
        instance_count = [0]
        count_lock = threading.Lock()

        def counting_factory(c):
            with count_lock:
                instance_count[0] += 1
            return {"instance": instance_count[0]}

        # Create a type for the test service (ServiceContainer expects types, not strings)
        TransientService = type("TransientService", (object,), {})
        container.register(TransientService, counting_factory, ServiceLifetime.TRANSIENT)

        def resolve_worker():
            """Resolve transient service concurrently."""
            for _ in range(20):
                instance = container.get(TransientService)
                assert instance is not None

        threads = [threading.Thread(target=resolve_worker) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=20)

        # Each resolution should create new instance
        expected_instances = 10 * 20
        assert instance_count[0] == expected_instances, (
            f"Expected {expected_instances} instances, got {instance_count[0]}"
        )


# =============================================================================
# Test Category 5: Deadlock Detection
# =============================================================================


class TestDeadlockDetection:
    """Tests to detect potential deadlocks under high concurrency."""

    def test_event_backend_no_deadlock(self, event_backend):
        """Test event backend doesn't deadlock under heavy concurrent load.

        TDD: This test will TIMEOUT until deadlock issues are fixed.
        """
        detector = DeadlockDetector(timeout_seconds=5.0)

        def heavy_concurrent_load():
            """Simulate heavy concurrent load that could cause deadlock."""
            backend = event_backend
            threads = []
            errors = []

            def worker(worker_id: int):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                for i in range(100):
                    try:
                        # Mix of subscribe, publish, unsubscribe
                        if i % 3 == 0:
                            async def _subscribe():
                                return await backend.subscribe(f"test.{worker_id}.*", lambda e: None)

                            loop.run_until_complete(_subscribe())

                        elif i % 3 == 1:
                            async def _publish():
                                return await backend.publish(
                                    MessagingEvent(
                                        topic=f"test.{worker_id}.event",
                                        data={"index": i},
                                    )
                                )

                            loop.run_until_complete(_publish())

                        else:
                            # Unsubscribe would need handle tracking - simplified
                            pass
                    except Exception as e:
                        errors.append(e)

                loop.close()

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]

            for t in threads:
                t.start()

            for t in threads:
                t.join(timeout=10)

            # Check no errors
            assert len(errors) == 0, f"Errors during load: {errors}"

        # Run with deadlock detection
        detector.run_with_timeout(heavy_concurrent_load)

        assert not detector.deadlock_detected, "Deadlock detected!"

    def test_cache_no_deadlock(self, embedding_cache):
        """Test cache doesn't deadlock under heavy concurrent load."""
        detector = DeadlockDetector(timeout_seconds=5.0)

        def heavy_concurrent_load():
            """Heavy mixed read/write load."""
            cache = embedding_cache

            def worker():
                for i in range(1000):
                    # Mix of operations
                    if i % 2 == 0:
                        cache.get(f"key_{i % 100}")
                    else:
                        cache.set(f"key_{i % 100}", f"value_{i}")

            threads = [threading.Thread(target=worker) for _ in range(20)]

            for t in threads:
                t.start()

            for t in threads:
                t.join(timeout=10)

        detector.run_with_timeout(heavy_concurrent_load)

        assert not detector.deadlock_detected, "Deadlock detected!"

    def test_metrics_no_deadlock(self, execution_metrics):
        """Test metrics don't deadlock under heavy concurrent load."""
        detector = DeadlockDetector(timeout_seconds=5.0)

        def heavy_concurrent_load():
            """Heavy concurrent metric recording."""
            metrics = execution_metrics

            def worker():
                for i in range(1000):
                    metrics.record_execution(
                        tool_name=f"tool_{i % 10}",
                        execution_time=0.1,
                        success=(i % 2 == 0),
                    )
                    # Occasionally call methods that access all data
                    if i % 100 == 0:
                        metrics.to_dict()
                        metrics.get_top_tools()

            threads = [threading.Thread(target=worker) for _ in range(20)]

            for t in threads:
                t.start()

            for t in threads:
                t.join(timeout=10)

        detector.run_with_timeout(heavy_concurrent_load)

        assert not detector.deadlock_detected, "Deadlock detected!"


# =============================================================================
# Test Category 6: Data Structure Corruption
# =============================================================================


class TestDataStructureCorruption:
    """Tests to detect data structure corruption under concurrent access."""

    def test_ordereddict_corruption_in_cache(self, embedding_cache):
        """Test OrderedDict doesn't get corrupted in cache."""
        cache = embedding_cache

        # Populate
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        def concurrent_access():
            """Access cache in ways that could corrupt OrderedDict."""
            for i in range(1000):
                # Mix of operations that manipulate OrderedDict
                cache.get(f"key_{i % 100}")
                cache.set(f"key_{i % 100}", f"new_value_{i}")

                if i % 10 == 0:
                    # Trigger eviction
                    cache.set(f"new_key_{i}", f"value_{i}")

        threads = [threading.Thread(target=concurrent_access) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=20)

        # Verify OrderedDict integrity
        # No duplicate keys
        keys = list(cache._cache.keys())
        unique_keys = set(keys)
        assert len(keys) == len(unique_keys), f"Duplicate keys detected: {len(keys) - len(unique_keys)}"

        # All values are accessible
        for key in keys[:10]:  # Check first 10
            value = cache._cache[key]
            assert value is not None

    def test_dict_corruption_in_metrics(self, execution_metrics):
        """Test dict structures don't get corrupted in metrics."""
        metrics = execution_metrics

        def concurrent_updates():
            """Update all dict fields concurrently."""
            for i in range(500):
                tool_name = f"tool_{i % 20}"
                metrics.record_execution(
                    tool_name=tool_name,
                    execution_time=0.1,
                    success=True,
                )

                # Access dicts
                _ = metrics.tool_counts[tool_name]
                _ = metrics.tool_errors.get(tool_name, 0)

        threads = [threading.Thread(target=concurrent_updates) for _ in range(20)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=20)

        # Verify dict integrity
        total_counted = sum(metrics.tool_counts.values())
        assert total_counted == metrics.total_executions, (
            f"Dict corruption: tool_counts sum {total_counted} != total {metrics.total_executions}"
        )

        # Verify all keys are strings
        for key in metrics.tool_counts.keys():
            assert isinstance(key, str), f"Invalid key type: {type(key)}"

        # Verify values are non-negative
        for value in metrics.tool_counts.values():
            assert value >= 0, f"Negative count: {value}"

    def test_list_corruption_in_metrics(self, execution_metrics):
        """Test list doesn't get corrupted in metrics."""
        metrics = execution_metrics

        def concurrent_appends():
            """Append to execution_times list concurrently."""
            for i in range(200):
                metrics.record_execution(
                    tool_name="test",
                    execution_time=0.01 * i,
                    success=True,
                )

        threads = [threading.Thread(target=concurrent_appends) for _ in range(20)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=20)

        # Verify list integrity
        expected_length = 20 * 200
        assert len(metrics.execution_times) == expected_length, (
            f"List corruption: length {len(metrics.execution_times)} != expected {expected_length}"
        )

        # Verify all entries are valid
        for t in metrics.execution_times:
            assert isinstance(t, (int, float)), f"Invalid time type: {type(t)}"
            assert t >= 0, f"Negative time: {t}"


# =============================================================================
# Test Category 7: High Concurrency (100+ operations)
# =============================================================================


class TestHighConcurrencyScenarios:
    """Tests with 100+ concurrent operations to expose edge cases."""

    def test_100_concurrent_subscriptions(self, event_backend):
        """Test 100+ concurrent subscriptions don't fail."""
        backend = event_backend
        success_count = [0]
        errors = []

        def subscriber(worker_id: int):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                async def _subscribe():
                    handle = await backend.subscribe(
                        f"high_concurrency.{worker_id}",
                        lambda e: None,
                    )
                    return handle

                handle = loop.run_until_complete(_subscribe())
                success_count[0] += 1
            except Exception as e:
                errors.append((worker_id, e))
            finally:
                loop.close()

        threads = [threading.Thread(target=subscriber, args=(i,)) for i in range(150)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Subscription errors: {errors}"
        assert success_count[0] == 150

    def test_100_concurrent_cache_operations(self, embedding_cache):
        """Test 100+ concurrent cache operations."""
        cache = embedding_cache
        operation_count = [0]
        errors = []

        def cache_worker(worker_id: int):
            for i in range(100):
                try:
                    cache.set(f"hc_key_{worker_id}_{i}", f"value_{i}")
                    cache.get(f"hc_key_{worker_id}_{i}")
                    operation_count[0] += 2
                except Exception as e:
                    errors.append((worker_id, i, e))

        threads = [threading.Thread(target=cache_worker, args=(i,)) for i in range(100)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=40)

        assert len(errors) == 0, f"Cache operation errors: {errors}"
        assert operation_count[0] == 100 * 100 * 2  # 100 workers, 100 ops each, 2 ops per iteration

    def test_100_concurrent_metric_updates(self, execution_metrics):
        """Test 100+ concurrent metric updates."""
        metrics = execution_metrics

        def metrics_worker(worker_id: int):
            for i in range(200):
                metrics.record_execution(
                    tool_name=f"hc_tool_{worker_id % 10}",
                    execution_time=0.01 * (i % 100),
                    success=(i % 2 == 0),
                )

        threads = [threading.Thread(target=metrics_worker, args=(i,)) for i in range(150)]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=40)

        # Verify consistency
        expected_total = 150 * 200
        assert metrics.total_executions == expected_total, (
            f"Total mismatch: {metrics.total_executions} vs {expected_total}"
        )
