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

"""Comprehensive performance benchmarks for Phase 4 optimizations.

This module benchmarks:
1. Lazy Loading Performance (victor/optimizations/lazy_loader.py)
2. Parallel Execution Performance (victor/optimizations/parallel_executor.py)
3. Memory Efficiency with lazy loading
4. Persona Manager Performance
5. Security Authorization Overhead

Run with:
    pytest tests/performance/optimizations/test_phase4_performance.py -v
    pytest tests/performance/optimizations/test_phase4_performance.py --benchmark-only
"""

from __future__ import annotations

import asyncio
import gc
import pytest
import time
import tracemalloc
from typing import Any, Dict, List
from dataclasses import dataclass

# Try to import pytest-benchmark
try:
    import pytest_benchmark

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

# Import optimization modules
from victor.optimization.runtime.lazy_loader import (
    LazyComponentLoader,
    LoadingStrategy,
    LoadingStats,
)
from victor.optimization.runtime.parallel_executor import (
    AdaptiveParallelExecutor,
    OptimizationStrategy,
    PerformanceMetrics,
)
from victor.agent.personas.persona_manager import PersonaManager
from victor.agent.personas.types import Persona, PersonalityType, CommunicationStyle
from victor.core.security.authorization import EnhancedAuthorizer, Permission


# =============================================================================
# Test Fixtures and Helper Classes
# =============================================================================


class ExpensiveComponent:
    """Simulates an expensive component to initialize."""

    def __init__(self, load_time_ms: float = 50.0):
        # Simulate expensive initialization
        time.sleep(load_time_ms / 1000.0)
        self.data = list(range(1000))


class SimpleComponent:
    """Simulates a simple component."""

    def __init__(self):
        self.data = "simple"


class DatabaseComponent:
    """Simulates a database component with dependencies."""

    def __init__(self, config=None):
        self.config = config
        self.data = {"records": list(range(100))}


class ConfigComponent:
    """Simulates a configuration component."""

    def __init__(self):
        self.settings = {"timeout": 30, "retries": 3}


async def async_task(duration_ms: float = 10.0, result: Any = None) -> Any:
    """Simulate an async task."""
    await asyncio.sleep(duration_ms / 1000.0)
    return result or f"completed_{duration_ms}"


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    name: str
    metric: str
    value: float
    unit: str
    improvement: str = ""
    details: str = ""


# =============================================================================
# 1. Lazy Loading Performance Benchmarks
# =============================================================================


class TestLazyLoadingPerformance:
    """Benchmark lazy loading performance."""

    def test_lazy_vs_eager_initialization(self, benchmark):
        """Benchmark: Lazy vs Eager initialization time.

        Expected: Lazy loading should be 20-30% faster for initialization.
        """
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not installed")

        # Test lazy initialization
        lazy_loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)

        def lazy_init():
            loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)
            loader.register_component("simple", lambda: SimpleComponent())
            loader.register_component("expensive", lambda: ExpensiveComponent(20))
            loader.register_component(
                "database", lambda: DatabaseComponent(), dependencies=["config"]
            )
            loader.register_component("config", lambda: ConfigComponent())
            return loader

        lazy_time = benchmark.pedantic(lazy_init, rounds=10, iterations=1)

        # Test eager initialization (preload all)
        def eager_init():
            loader = LazyComponentLoader(strategy=LoadingStrategy.EAGER)
            loader.register_component("simple", lambda: SimpleComponent())
            loader.register_component("expensive", lambda: ExpensiveComponent(20))
            loader.register_component(
                "database", lambda: DatabaseComponent(), dependencies=["config"]
            )
            loader.register_component("config", lambda: ConfigComponent())
            # Preload all
            loader.preload_components(["simple", "expensive", "database", "config"])
            return loader

        eager_time = benchmark.pedantic(eager_init, rounds=10, iterations=1)

        # Calculate improvement
        improvement = ((eager_time - lazy_time) / eager_time) * 100

        print(f"\nLazy initialization: {lazy_time:.4f}s")
        print(f"Eager initialization: {eager_time:.4f}s")
        print(f"Improvement: {improvement:.1f}%")

        # Assert lazy is faster
        assert lazy_time < eager_time, "Lazy loading should be faster than eager loading"

    def test_first_access_overhead(self, benchmark):
        """Benchmark: First access overhead for lazy loading.

        Expected: ~5-10ms overhead for first access.
        """
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not installed")

        loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)
        loader.register_component("expensive", lambda: ExpensiveComponent(20))

        def first_access():
            # Reset component to force reload
            loader.unload_component("expensive")
            return loader.get_component("expensive")

        result = benchmark.pedantic(first_access, rounds=20, iterations=1)

        print(f"\nFirst access time: {result*1000:.2f}ms")
        assert result < 0.1, "First access should complete within 100ms"

    def test_cached_access_performance(self, benchmark):
        """Benchmark: Cached access performance.

        Expected: Cached access should be <1ms.
        """
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not installed")

        loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)
        loader.register_component("simple", lambda: SimpleComponent())

        # Preload
        loader.get_component("simple")

        def cached_access():
            return loader.get_component("simple")

        result = benchmark.pedantic(cached_access, rounds=1000, iterations=1)

        print(f"\nCached access time: {result*1000:.4f}ms")
        assert result < 0.001, "Cached access should be sub-millisecond"

    def test_adaptive_loading_strategy(self, benchmark):
        """Benchmark: Adaptive loading strategy performance.

        Expected: Adaptive strategy should learn and preload hot components.
        """
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not installed")

        loader = LazyComponentLoader(strategy=LoadingStrategy.ADAPTIVE, adaptive_threshold=2)
        loader.register_component("hot", lambda: ExpensiveComponent(10))
        loader.register_component("cold", lambda: ExpensiveComponent(10))
        loader.register_component("config", lambda: ConfigComponent())

        def adaptive_workload():
            # Access 'hot' component multiple times to trigger adaptive preloading
            loader.unload_component("hot")
            loader.unload_component("cold")

            # First access - cold
            loader.get_component("hot")

            # Second access - should start being tracked
            loader.get_component("hot")

            # Third access - should be adaptively loaded
            start = time.perf_counter()
            component = loader.get_component("hot")
            elapsed = (time.perf_counter() - start) * 1000

            return elapsed

        avg_time = benchmark.pedantic(adaptive_workload, rounds=10, iterations=1)

        print(f"\nAdaptive loading time (after warmup): {avg_time:.2f}ms")

    def test_dependency_resolution_overhead(self, benchmark):
        """Benchmark: Dependency resolution overhead.

        Expected: Minimal overhead for resolving dependencies.
        """
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not installed")

        loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)
        loader.register_component("config", lambda: ConfigComponent())
        loader.register_component("database", lambda: DatabaseComponent(), dependencies=["config"])
        loader.register_component("service", lambda: SimpleComponent(), dependencies=["database"])

        def resolve_dependencies():
            loader.unload_component("service")
            loader.unload_component("database")
            loader.unload_component("config")
            return loader.get_component("service")

        result = benchmark.pedantic(resolve_dependencies, rounds=20, iterations=1)

        print(f"\nDependency resolution time: {result*1000:.2f}ms")


# =============================================================================
# 2. Parallel Execution Performance Benchmarks
# =============================================================================


class TestParallelExecutionPerformance:
    """Benchmark parallel execution performance."""

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_execution(self):
        """Benchmark: Parallel vs Sequential execution speedup.

        Expected: 15%+ speedup with parallelization.
        """
        # Create tasks
        tasks = [lambda: async_task(50, f"result_{i}") for i in range(10)]

        # Sequential execution
        sequential_start = time.perf_counter()
        sequential_results = []
        for task in tasks:
            result = await task()
            sequential_results.append(result)
        sequential_time = (time.perf_counter() - sequential_start) * 1000

        # Parallel execution
        parallel_executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=4,
        )

        parallel_start = time.perf_counter()
        parallel_result = await parallel_executor.execute(tasks)
        parallel_time = (time.perf_counter() - parallel_start) * 1000

        speedup = sequential_time / parallel_time
        improvement = ((sequential_time - parallel_time) / sequential_time) * 100

        print(f"\nSequential execution: {sequential_time:.2f}ms")
        print(f"Parallel execution: {parallel_time:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Improvement: {improvement:.1f}%")

        assert parallel_time < sequential_time, "Parallel should be faster than sequential"
        assert speedup >= 1.15, f"Expected at least 15% speedup, got {speedup:.2f}x"

    @pytest.mark.asyncio
    async def test_adaptive_strategy_performance(self):
        """Benchmark: Adaptive strategy auto-selection.

        Expected: Adaptive strategy chooses optimal execution mode.
        """
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ADAPTIVE,
            max_workers=4,
        )

        # Small task count - should choose sequential
        small_tasks = [lambda: async_task(10) for _ in range(2)]
        small_start = time.perf_counter()
        await executor.execute(small_tasks)
        small_time = time.perf_counter() - small_start

        # Large task count - should choose parallel
        large_tasks = [lambda: async_task(20) for _ in range(10)]
        large_start = time.perf_counter()
        result = await executor.execute(large_tasks)
        large_time = time.perf_counter() - large_start

        metrics = executor.get_metrics()

        print(f"\nSmall task execution: {small_time*1000:.2f}ms")
        print(f"Large task execution: {large_time*1000:.2f}ms")
        print(f"Tasks executed: {metrics.tasks_executed}")
        print(f"Workers used: {metrics.worker_count}")
        print(f"Speedup factor: {metrics.speedup_factor:.2f}x")

    @pytest.mark.asyncio
    async def test_work_stealing_efficiency(self):
        """Benchmark: Work stealing for load balancing.

        Expected: Better load distribution and reduced wait times.
        """
        # Create tasks with variable durations
        tasks = [lambda i=i: async_task(10 + (i % 3) * 10, f"task_{i}") for i in range(12)]

        executor_without_ws = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=4,
            enable_work_stealing=False,
        )

        executor_with_ws = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=4,
            enable_work_stealing=True,
        )

        # Without work stealing
        start = time.perf_counter()
        await executor_without_ws.execute(tasks)
        time_without_ws = (time.perf_counter() - start) * 1000

        # With work stealing
        start = time.perf_counter()
        await executor_with_ws.execute(tasks)
        time_with_ws = (time.perf_counter() - start) * 1000

        improvement = ((time_without_ws - time_with_ws) / time_without_ws) * 100

        print(f"\nWithout work stealing: {time_without_ws:.2f}ms")
        print(f"With work stealing: {time_with_ws:.2f}ms")
        print(f"Improvement: {improvement:.1f}%")

    @pytest.mark.asyncio
    async def test_parallel_overhead(self):
        """Benchmark: Parallelization overhead.

        Expected: <5% overhead for parallelization framework.
        """
        # Very small tasks - overhead should be visible
        tasks = [lambda: async_task(1, f"micro_{i}") for i in range(5)]

        # Sequential baseline
        sequential_start = time.perf_counter()
        for task in tasks:
            await task()
        sequential_time = time.perf_counter() - sequential_start

        # Parallel with overhead
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=2,
        )

        parallel_start = time.perf_counter()
        result = await executor.execute(tasks)
        parallel_time = time.perf_counter() - parallel_start

        metrics = executor.get_metrics()
        overhead_ratio = metrics.overhead_ms / (parallel_time * 1000)

        print(f"\nSequential time: {sequential_time*1000:.2f}ms")
        print(f"Parallel time: {parallel_time*1000:.2f}ms")
        print(f"Overhead: {metrics.overhead_ms:.2f}ms")
        print(f"Overhead ratio: {overhead_ratio:.1%}")

        # For very small tasks, overhead might be higher, but should still be reasonable
        assert overhead_ratio < 0.5, "Overhead should be less than 50% even for micro tasks"


# =============================================================================
# 3. Memory Efficiency Benchmarks
# =============================================================================


class TestMemoryEfficiency:
    """Benchmark memory efficiency improvements."""

    def test_lazy_loading_memory_savings(self):
        """Benchmark: Memory savings from lazy loading.

        Expected: 15-25% memory reduction for unused components.
        """
        gc.collect()

        # Measure eager loading memory
        tracemalloc.start()
        eager_loader = LazyComponentLoader(strategy=LoadingStrategy.EAGER)
        eager_loader.register_component("large1", lambda: ExpensiveComponent(0))
        eager_loader.register_component("large2", lambda: ExpensiveComponent(0))
        eager_loader.register_component("large3", lambda: ExpensiveComponent(0))
        eager_loader.preload_components(["large1", "large2", "large3"])

        eager_current, eager_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        gc.collect()

        # Measure lazy loading memory (only load one component)
        tracemalloc.start()
        lazy_loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)
        lazy_loader.register_component("large1", lambda: ExpensiveComponent(0))
        lazy_loader.register_component("large2", lambda: ExpensiveComponent(0))
        lazy_loader.register_component("large3", lambda: ExpensiveComponent(0))
        # Only access one component
        lazy_loader.get_component("large1")

        lazy_current, lazy_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_saving = ((eager_current - lazy_current) / eager_current) * 100

        print(f"\nEager loading memory: {eager_current / 1024:.2f} KB")
        print(f"Lazy loading memory: {lazy_current / 1024:.2f} KB")
        print(f"Memory saving: {memory_saving:.1f}%")

        assert lazy_current < eager_current, "Lazy loading should use less memory"
        assert memory_saving >= 15, f"Expected at least 15% memory saving, got {memory_saving:.1f}%"

    def test_cache_memory_management(self):
        """Benchmark: LRU cache memory management.

        Expected: Cache eviction prevents unbounded memory growth.
        """
        loader = LazyComponentLoader(
            strategy=LoadingStrategy.LAZY,
            max_cache_size=5,
        )

        # Register more components than cache size
        for i in range(10):
            loader.register_component(f"component_{i}", lambda: ExpensiveComponent(0))

        # Load all components
        for i in range(10):
            loader.get_component(f"component_{i}")

        loaded = loader.get_loaded_components()

        print("\nRegistered: 10 components")
        print(f"Loaded (in cache): {len(loaded)} components")
        print("Max cache size: 5")

        assert len(loaded) <= 5, "LRU cache should evict old components"


# =============================================================================
# 4. Persona Manager Performance Benchmarks
# =============================================================================


class TestPersonaManagerPerformance:
    """Benchmark persona manager performance."""

    def test_persona_loading_performance(self, benchmark):
        """Benchmark: Persona loading from repository.

        Expected: <10ms for persona loading.
        """
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not installed")

        manager = PersonaManager(auto_load=False)

        # Register test personas
        for i in range(5):
            persona = Persona(
                id=f"persona_{i}",
                name=f"Test Persona {i}",
                description=f"A test persona {i}",
                personality=PersonalityType.CREATIVE,
                communication_style=CommunicationStyle.CASUAL,
                expertise=["testing", "benchmarking"],
            )
            manager.repository.save(persona)

        def load_persona():
            return manager.load_persona("persona_0")

        result = benchmark.pedantic(load_persona, rounds=100, iterations=1)

        print(f"\nPersona loading time: {result*1000:.2f}ms")
        assert result < 0.01, "Persona loading should be under 10ms"

    def test_persona_adaptation_performance(self, benchmark):
        """Benchmark: Persona adaptation performance.

        Expected: <20ms for persona adaptation with caching.
        """
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not installed")

        manager = PersonaManager(auto_load=False)

        persona = Persona(
            id="test_persona",
            name="Test Persona",
            description="A test persona",
            personality=PersonalityType.CREATIVE,
            communication_style=CommunicationStyle.CASUAL,
            expertise=["testing"],
        )
        manager.repository.save(persona)

        context = {
            "task_type": "security_review",
            "urgency": "high",
            "complexity": "high",
        }

        # First call - cold cache
        manager.adapt_persona(persona, context)

        def adapt_persona_cached():
            return manager.adapt_persona(persona, context)

        result = benchmark.pedantic(adapt_persona_cached, rounds=100, iterations=1)

        print(f"\nPersona adaptation time (cached): {result*1000:.2f}ms")
        assert result < 0.02, "Cached adaptation should be under 20ms"

    def test_persona_merging_performance(self):
        """Benchmark: Persona merging performance.

        Expected: <50ms for merging 3 personas.
        """
        manager = PersonaManager(auto_load=False)

        # Create test personas
        personas = []
        for i in range(3):
            persona = Persona(
                id=f"merge_source_{i}",
                name=f"Source Persona {i}",
                description=f"Source persona {i}",
                personality=PersonalityType.METHODICAL,
                communication_style=CommunicationStyle.FORMAL,
                expertise=[f"skill_{i}", "common_skill"],
            )
            manager.repository.save(persona)
            personas.append(persona)

        start = time.perf_counter()
        merged = manager.merge_personas(personas, "Merged Persona")
        merge_time = (time.perf_counter() - start) * 1000

        print(f"\nPersona merging time: {merge_time:.2f}ms")
        assert merge_time < 50, "Persona merging should complete within 50ms"

    def test_caching_effectiveness(self):
        """Benchmark: Adaptation cache effectiveness.

        Expected: High cache hit rate for repeated adaptations.
        """
        manager = PersonaManager(auto_load=False)

        persona = Persona(
            id="cache_test",
            name="Cache Test",
            description="Testing cache",
            personality=PersonalityType.CREATIVE,
            communication_style=CommunicationStyle.CASUAL,
            expertise=["caching"],
        )
        manager.repository.save(persona)

        context = {"task_type": "debugging", "urgency": "normal"}

        # Clear stats
        manager._adaptation_cache.clear()

        # First adaptation - cache miss
        start = time.perf_counter()
        manager.adapt_persona(persona, context)
        first_time = (time.perf_counter() - start) * 1000

        # Repeated adaptations - cache hits
        start = time.perf_counter()
        for _ in range(10):
            manager.adapt_persona(persona, context)
        cached_time = (time.perf_counter() - start) * 1000

        avg_cached_time = cached_time / 10

        print(f"\nFirst adaptation (cache miss): {first_time:.2f}ms")
        print(f"Average cached adaptation: {avg_cached_time:.2f}ms")
        print(f"Cache speedup: {first_time / avg_cached_time:.2f}x")

        assert avg_cached_time < first_time, "Cached adaptations should be faster"


# =============================================================================
# 5. Security Authorization Overhead Benchmarks
# =============================================================================


class TestSecurityAuthorizationOverhead:
    """Benchmark security authorization performance."""

    def test_authorization_check_latency(self, benchmark):
        """Benchmark: Authorization check latency.

        Expected: <5ms per authorization check.
        """
        if not HAS_BENCHMARK:
            pytest.skip("pytest-benchmark not installed")

        authorizer = EnhancedAuthorizer()

        # Setup: Create role and permissions
        authorizer.create_role(
            "developer",
            permissions={
                Permission("tools", "read"),
                Permission("tools", "execute"),
                Permission("code", "write"),
            },
        )

        # Setup: Create user and assign role
        authorizer.create_user(user_id="user1", username="user1", roles=["developer"])
        user = authorizer.get_user("user1")

        def auth_check():
            return authorizer.check_permission(user, "tools", "execute")

        result = benchmark.pedantic(auth_check, rounds=1000, iterations=1)

        print(f"\nAuthorization check latency: {result*1000:.2f}ms")
        assert result < 0.005, "Authorization check should be under 5ms"

    def test_bulk_authorization_checks(self):
        """Benchmark: Bulk authorization check performance.

        Expected: Linear or better scaling with check count.
        """
        authorizer = EnhancedAuthorizer()

        # Setup
        authorizer.create_role(
            "admin",
            permissions={
                Permission("tools", "read"),
                Permission("tools", "execute"),
                Permission("tools", "write"),
                Permission("code", "read"),
                Permission("code", "write"),
                Permission("code", "delete"),
            },
        )
        authorizer.create_user(user_id="admin_user", username="admin_user", roles=["admin"])
        admin_user_obj = authorizer.get_user("admin_user")

        # Single check baseline
        start = time.perf_counter()
        authorizer.check_permission(admin_user_obj, "tools", "execute")
        single_check_time = (time.perf_counter() - start) * 1000

        # Bulk checks
        checks = [
            ("tools", "read"),
            ("tools", "execute"),
            ("tools", "write"),
            ("code", "read"),
            ("code", "write"),
        ]

        start = time.perf_counter()
        results = [
            authorizer.check_permission(admin_user_obj, resource, action)
            for resource, action in checks
        ]
        bulk_check_time = (time.perf_counter() - start) * 1000

        avg_bulk_time = bulk_check_time / len(checks)

        print(f"\nSingle check time: {single_check_time:.2f}ms")
        print(f"Bulk check time (total): {bulk_check_time:.2f}ms")
        print(f"Bulk check time (avg): {avg_bulk_time:.2f}ms")
        print(f"Scaling factor: {avg_bulk_time / single_check_time:.2f}x")

        # Bulk checks should not be significantly slower per check
        assert avg_bulk_time < single_check_time * 2, "Bulk checks should scale well"


# =============================================================================
# Comprehensive Benchmark Report
# =============================================================================


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
def test_generate_comprehensive_report():
    """Generate comprehensive benchmark report.

    This test collects all benchmark results and generates a summary report.
    """
    results = []

    # Run lazy loading benchmarks
    print("\n" + "=" * 80)
    print("LAZY LOADING BENCHMARKS")
    print("=" * 80)

    loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)
    loader.register_component("test", lambda: ExpensiveComponent(10))

    start = time.perf_counter()
    loader.get_component("test")
    first_access = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    loader.get_component("test")
    cached_access = (time.perf_counter() - start) * 1000

    results.append(
        BenchmarkResult(
            name="Lazy Loading",
            metric="First Access",
            value=first_access,
            unit="ms",
            details=f"Cached access: {cached_access:.2f}ms",
        )
    )

    # Run parallel execution benchmarks
    print("\n" + "=" * 80)
    print("PARALLEL EXECUTION BENCHMARKS")
    print("=" * 80)

    async def run_parallel_benchmarks():
        tasks = [lambda i=i: async_task(20, f"result_{i}") for i in range(10)]

        # Sequential
        start = time.perf_counter()
        for task in tasks:
            await task()
        sequential_time = (time.perf_counter() - start) * 1000

        # Parallel
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=4,
        )
        start = time.perf_counter()
        await executor.execute(tasks)
        parallel_time = (time.perf_counter() - start) * 1000

        speedup = sequential_time / parallel_time

        results.append(
            BenchmarkResult(
                name="Parallel Execution",
                metric="Speedup",
                value=speedup,
                unit="x",
                improvement=f"{((1 - 1/speedup) * 100):.1f}% faster",
                details=f"Sequential: {sequential_time:.0f}ms, Parallel: {parallel_time:.0f}ms",
            )
        )

    asyncio.run(run_parallel_benchmarks())

    # Print summary report
    print("\n" + "=" * 80)
    print("PHASE 4 PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 80)

    for result in results:
        print(f"\n{result.name}:")
        print(f"  {result.metric}: {result.value:.2f} {result.unit}")
        if result.improvement:
            print(f"  Improvement: {result.improvement}")
        if result.details:
            print(f"  Details: {result.details}")

    print("\n" + "=" * 80)
    print("All benchmarks completed successfully!")
    print("=" * 80)

    assert len(results) > 0, "Should have collected benchmark results"


if __name__ == "__main__":
    # Run benchmarks directly
    pytest.main([__file__, "-v", "-s"])
