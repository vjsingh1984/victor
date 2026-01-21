#!/usr/bin/env python3
"""Comprehensive Phase 4 Performance Benchmark Runner.

This script runs all performance benchmarks and generates a detailed report.
Run: python scripts/benchmark_phase4.py
"""

import asyncio
import gc
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.optimizations.lazy_loader import (
    LazyComponentLoader,
    LoadingStrategy,
)
from victor.optimizations.parallel_executor import (
    AdaptiveParallelExecutor,
    OptimizationStrategy,
)
from victor.agent.personas.persona_manager import PersonaManager
from victor.agent.personas.types import Persona, PersonalityType, CommunicationStyle
from victor.security.authorization_enhanced import EnhancedAuthorizer, Permission


# =============================================================================
# Helper Classes
# =============================================================================


class ExpensiveComponent:
    """Simulates expensive component initialization."""
    def __init__(self, load_time_ms: float = 20.0):
        time.sleep(load_time_ms / 1000.0)
        self.data = list(range(1000))


class SimpleComponent:
    """Simulates simple component."""
    def __init__(self):
        self.data = "simple"


async def async_task(duration_ms: float, result: Any = None) -> Any:
    """Simulate async task."""
    await asyncio.sleep(duration_ms / 1000.0)
    return result or f"completed_{duration_ms}"


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    category: str
    name: str
    metric: str
    value: float
    unit: str
    target: float = 0.0
    passed: bool = False
    details: str = ""


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    def add_result(self, result: BenchmarkResult) -> None:
        """Add result to report."""
        self.results.append(result)

    def print_summary(self) -> None:
        """Print benchmark summary."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        print("\n" + "="*100)
        print("PHASE 4 PERFORMANCE BENCHMARK REPORT")
        print("="*100)
        print(f"Total Duration: {duration:.2f} seconds")
        print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)

        # Group results by category
        categories: Dict[str, List[BenchmarkResult]] = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        # Print each category
        for category, cat_results in categories.items():
            print(f"\n{'='*100}")
            print(f"{category}")
            print('='*100)

            # Table header
            print(f"{'Benchmark':<40} {'Metric':<20} {'Value':<15} {'Target':<10} {'Status':<10}")
            print('-'*100)

            for result in cat_results:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                value_str = f"{result.value:.2f} {result.unit}"
                target_str = f"{result.target:.2f} {result.unit}" if result.target > 0 else "N/A"

                print(f"{result.name:<40} {result.metric:<20} {value_str:<15} {target_str:<10} {status:<10}")

                if result.details:
                    print(f"  └─ {result.details}")

        # Summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"\n{'='*100}")
        print(f"SUMMARY")
        print('='*100)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print('='*100 + "\n")


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_lazy_loading(report: BenchmarkReport) -> None:
    """Benchmark lazy loading performance."""
    print("\n[1/5] Running Lazy Loading Benchmarks...")

    # 1. Initialization time comparison
    print("  Testing lazy vs eager initialization...")

    lazy_times = []
    eager_times = []

    for _ in range(10):
        # Lazy initialization
        start = time.perf_counter()
        lazy_loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)
        lazy_loader.register_component("simple", lambda: SimpleComponent())
        lazy_loader.register_component("expensive", lambda: ExpensiveComponent(10))
        lazy_loader.register_component("db", lambda: SimpleComponent())
        lazy_time = (time.perf_counter() - start) * 1000
        lazy_times.append(lazy_time)

        # Eager initialization
        start = time.perf_counter()
        eager_loader = LazyComponentLoader(strategy=LoadingStrategy.EAGER)
        eager_loader.register_component("simple", lambda: SimpleComponent())
        eager_loader.register_component("expensive", lambda: ExpensiveComponent(10))
        eager_loader.register_component("db", lambda: SimpleComponent())
        eager_loader.preload_components(["simple", "expensive", "db"])
        eager_time = (time.perf_counter() - start) * 1000
        eager_times.append(eager_time)

    avg_lazy = sum(lazy_times) / len(lazy_times)
    avg_eager = sum(eager_times) / len(eager_times)
    improvement = ((avg_eager - avg_lazy) / avg_eager) * 100

    report.add_result(BenchmarkResult(
        category="1. Lazy Loading Performance",
        name="Initialization Time",
        metric="Avg Initialization",
        value=avg_lazy,
        unit="ms",
        target=avg_eager * 0.8,  # Target: 20% faster
        passed=avg_lazy < avg_eager,
        details=f"Eager: {avg_eager:.2f}ms, Improvement: {improvement:.1f}%"
    ))

    # 2. First access overhead
    print("  Testing first access overhead...")

    loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)
    loader.register_component("component", lambda: ExpensiveComponent(20))

    first_access_times = []
    for _ in range(20):
        loader.unload_component("component")
        start = time.perf_counter()
        loader.get_component("component")
        first_access = (time.perf_counter() - start) * 1000
        first_access_times.append(first_access)

    avg_first_access = sum(first_access_times) / len(first_access_times)

    report.add_result(BenchmarkResult(
        category="1. Lazy Loading Performance",
        name="First Access Overhead",
        metric="Avg First Access",
        value=avg_first_access,
        unit="ms",
        target=30.0,  # Target: <30ms
        passed=avg_first_access < 30.0,
        details="Includes component initialization time"
    ))

    # 3. Cached access performance
    print("  Testing cached access performance...")

    # Preload
    loader.get_component("component")

    cached_times = []
    for _ in range(1000):
        start = time.perf_counter()
        loader.get_component("component")
        cached = (time.perf_counter() - start) * 1000
        cached_times.append(cached)

    avg_cached = sum(cached_times) / len(cached_times)

    report.add_result(BenchmarkResult(
        category="1. Lazy Loading Performance",
        name="Cached Access",
        metric="Avg Cached Access",
        value=avg_cached,
        unit="ms",
        target=1.0,  # Target: <1ms
        passed=avg_cached < 1.0,
        details=f"Speedup vs first access: {avg_first_access / avg_cached:.0f}x"
    ))


def benchmark_parallel_execution(report: BenchmarkReport) -> None:
    """Benchmark parallel execution performance."""
    print("\n[2/5] Running Parallel Execution Benchmarks...")

    async def run_benchmarks():
        # 1. Parallel vs Sequential speedup
        print("  Testing parallel vs sequential execution...")

        tasks = [lambda i=i: async_task(50, f"result_{i}") for i in range(10)]

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
        improvement = ((sequential_time - parallel_time) / sequential_time) * 100

        report.add_result(BenchmarkResult(
            category="2. Parallel Execution Performance",
            name="Parallel Speedup",
            metric="Speedup Factor",
            value=speedup,
            unit="x",
            target=1.15,  # Target: 15% speedup
            passed=speedup >= 1.15,
            details=f"Sequential: {sequential_time:.0f}ms, Parallel: {parallel_time:.0f}ms, Improvement: {improvement:.1f}%"
        ))

        # 2. Adaptive strategy
        print("  Testing adaptive strategy...")

        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ADAPTIVE,
            max_workers=4,
        )

        # Small workload
        small_tasks = [lambda: async_task(10) for _ in range(2)]
        start = time.perf_counter()
        await executor.execute(small_tasks)
        small_time = (time.perf_counter() - start) * 1000

        # Large workload
        large_tasks = [lambda: async_task(20) for _ in range(10)]
        start = time.perf_counter()
        await executor.execute(large_tasks)
        large_time = (time.perf_counter() - start) * 1000

        metrics = executor.get_metrics()

        report.add_result(BenchmarkResult(
            category="2. Parallel Execution Performance",
            name="Adaptive Strategy",
            metric="Large Task Execution",
            value=large_time,
            unit="ms",
            target=1000.0,  # Reasonable target
            passed=large_time < 1000.0,
            details=f"Small tasks: {small_time:.0f}ms, Workers: {metrics.worker_count}, Speedup: {metrics.speedup_factor:.2f}x"
        ))

        # 3. Overhead measurement
        print("  Testing parallelization overhead...")

        # Very small tasks
        micro_tasks = [lambda i=i: async_task(1, f"micro_{i}") for i in range(5)]

        # Sequential baseline
        start = time.perf_counter()
        for task in micro_tasks:
            await task()
        seq_time = time.perf_counter() - start

        # Parallel with overhead
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=2,
        )
        start = time.perf_counter()
        await executor.execute(micro_tasks)
        par_time = time.perf_counter() - start

        metrics = executor.get_metrics()
        overhead_ms = metrics.overhead_ms
        overhead_ratio = overhead_ms / (par_time * 1000)

        report.add_result(BenchmarkResult(
            category="2. Parallel Execution Performance",
            name="Parallelization Overhead",
            metric="Overhead Ratio",
            value=overhead_ratio * 100,
            unit="%",
            target=50.0,  # Target: <50% for micro tasks
            passed=overhead_ratio < 0.5,
            details=f"Overhead: {overhead_ms:.2f}ms"
        ))

    asyncio.run(run_benchmarks())


def benchmark_memory_efficiency(report: BenchmarkReport) -> None:
    """Benchmark memory efficiency."""
    print("\n[3/5] Running Memory Efficiency Benchmarks...")

    gc.collect()

    # 1. Memory savings from lazy loading
    print("  Testing lazy loading memory savings...")

    # Eager loading
    tracemalloc.start()
    eager_loader = LazyComponentLoader(strategy=LoadingStrategy.EAGER)
    for i in range(5):
        eager_loader.register_component(f"large_{i}", lambda: ExpensiveComponent(0))
    eager_loader.preload_components([f"large_{i}" for i in range(5)])
    eager_current, eager_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gc.collect()

    # Lazy loading (only load 1 component)
    tracemalloc.start()
    lazy_loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY)
    for i in range(5):
        lazy_loader.register_component(f"large_{i}", lambda: ExpensiveComponent(0))
    lazy_loader.get_component("large_0")  # Only load one
    lazy_current, lazy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_saving = ((eager_current - lazy_current) / eager_current) * 100

    report.add_result(BenchmarkResult(
        category="3. Memory Efficiency",
        name="Lazy Loading Memory Savings",
        metric="Memory Reduction",
        value=memory_saving,
        unit="%",
        target=15.0,  # Target: 15% reduction
        passed=memory_saving >= 15.0,
        details=f"Eager: {eager_current/1024:.1f}KB, Lazy: {lazy_current/1024:.1f}KB"
    ))

    # 2. Cache management
    print("  Testing LRU cache management...")

    loader = LazyComponentLoader(strategy=LoadingStrategy.LAZY, max_cache_size=3)

    # Register 10 components
    for i in range(10):
        loader.register_component(f"comp_{i}", lambda: ExpensiveComponent(0))

    # Access all components
    for i in range(10):
        loader.get_component(f"comp_{i}")

    loaded_count = len(loader.get_loaded_components())

    report.add_result(BenchmarkResult(
        category="3. Memory Efficiency",
        name="LRU Cache Management",
        metric="Loaded Components",
        value=loaded_count,
        unit="count",
        target=3.0,  # Should not exceed max_cache_size
        passed=loaded_count <= 3,
        details=f"Max cache size: 3, Registered: 10"
    ))


def benchmark_persona_manager(report: BenchmarkReport) -> None:
    """Benchmark persona manager performance."""
    print("\n[4/5] Running Persona Manager Benchmarks...")

    manager = PersonaManager(auto_load=False)

    # Create test personas
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

    # 1. Persona loading
    print("  Testing persona loading...")

    load_times = []
    for _ in range(100):
        start = time.perf_counter()
        manager.load_persona("persona_0")
        load_time = (time.perf_counter() - start) * 1000
        load_times.append(load_time)

    avg_load = sum(load_times) / len(load_times)

    report.add_result(BenchmarkResult(
        category="4. Persona Manager Performance",
        name="Persona Loading",
        metric="Avg Load Time",
        value=avg_load,
        unit="ms",
        target=10.0,  # Target: <10ms
        passed=avg_load < 10.0,
        details="From repository"
    ))

    # 2. Persona adaptation (with caching)
    print("  Testing persona adaptation...")

    persona = manager.load_persona("persona_0")
    context = {"task_type": "security_review", "urgency": "high"}

    # First call - cold cache
    manager.adapt_persona(persona, context)

    # Cached calls
    adapt_times = []
    for _ in range(100):
        start = time.perf_counter()
        manager.adapt_persona(persona, context)
        adapt_time = (time.perf_counter() - start) * 1000
        adapt_times.append(adapt_time)

    avg_adapt = sum(adapt_times) / len(adapt_times)

    report.add_result(BenchmarkResult(
        category="4. Persona Manager Performance",
        name="Persona Adaptation (Cached)",
        metric="Avg Adapt Time",
        value=avg_adapt,
        unit="ms",
        target=20.0,  # Target: <20ms
        passed=avg_adapt < 20.0,
        details="With caching enabled"
    ))

    # 3. Persona merging
    print("  Testing persona merging...")

    from victor.agent.personas.types import PersonaConstraints

    personas = []
    for i in range(3):
        persona = Persona(
            id=f"merge_src_{i}",
            name=f"Source {i}",
            description=f"Source persona {i}",
            personality=PersonalityType.METHODICAL,
            communication_style=CommunicationStyle.FORMAL,
            expertise=[f"skill_{i}", "common"],
            constraints=PersonaConstraints(
                max_tool_calls=10,
                preferred_tools={"tool1", "tool2"},
            ),
        )
        manager.repository.save(persona)
        personas.append(persona)

    start = time.perf_counter()
    manager.merge_personas(personas, "Merged Persona")
    merge_time = (time.perf_counter() - start) * 1000

    report.add_result(BenchmarkResult(
        category="4. Persona Manager Performance",
        name="Persona Merging",
        metric="Merge Time",
        value=merge_time,
        unit="ms",
        target=50.0,  # Target: <50ms
        passed=merge_time < 50.0,
        details="Merging 3 personas"
    ))


def benchmark_security_overhead(report: BenchmarkReport) -> None:
    """Benchmark security authorization overhead."""
    print("\n[5/5] Running Security Authorization Benchmarks...")

    authorizer = EnhancedAuthorizer()

    # Setup
    authorizer.create_role(
        "developer",
        permissions={
            Permission("tools", "read"),
            Permission("tools", "execute"),
            Permission("code", "write"),
        }
    )
    user = authorizer.create_user("user1", "user1", roles=["developer"])

    # 1. Single authorization check
    print("  Testing authorization check latency...")

    check_times = []
    for _ in range(1000):
        start = time.perf_counter()
        authorizer.check_permission(user, "tools", "execute")
        check_time = (time.perf_counter() - start) * 1000
        check_times.append(check_time)

    avg_check = sum(check_times) / len(check_times)

    report.add_result(BenchmarkResult(
        category="5. Security Authorization Overhead",
        name="Authorization Check",
        metric="Avg Check Time",
        value=avg_check,
        unit="ms",
        target=5.0,  # Target: <5ms
        passed=avg_check < 5.0,
        details="Single permission check"
    ))

    # 2. Bulk authorization checks
    print("  Testing bulk authorization checks...")

    authorizer.create_role(
        "admin",
        permissions={
            Permission("tools", "read"),
            Permission("tools", "execute"),
            Permission("tools", "write"),
            Permission("code", "read"),
            Permission("code", "write"),
            Permission("code", "delete"),
        }
    )
    admin_user = authorizer.create_user("admin_user", "admin_user", roles=["admin"])

    # Single baseline
    start = time.perf_counter()
    authorizer.check_permission(admin_user, "tools", "execute")
    single_time = (time.perf_counter() - start) * 1000

    # Bulk checks
    resource_actions = [
        ("tools", "read"),
        ("tools", "execute"),
        ("tools", "write"),
        ("code", "read"),
        ("code", "write"),
    ]

    start = time.perf_counter()
    for resource, action in resource_actions:
        authorizer.check_permission(admin_user, resource, action)
    bulk_time = (time.perf_counter() - start) * 1000

    avg_bulk = bulk_time / len(resource_actions)
    scaling = avg_bulk / single_time

    report.add_result(BenchmarkResult(
        category="5. Security Authorization Overhead",
        name="Bulk Authorization",
        metric="Scaling Factor",
        value=scaling,
        unit="x",
        target=2.0,  # Target: <2x scaling
        passed=scaling < 2.0,
        details=f"Single: {single_time:.2f}ms, Avg bulk: {avg_bulk:.2f}ms"
    ))


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Run all benchmarks and generate report."""
    print("="*100)
    print("PHASE 4 PERFORMANCE BENCHMARK SUITE")
    print("="*100)
    print("Running comprehensive performance benchmarks on Phase 4 optimizations...")
    print("This may take several minutes...\n")

    report = BenchmarkReport()

    try:
        # Run all benchmark categories
        benchmark_lazy_loading(report)
        benchmark_parallel_execution(report)
        benchmark_memory_efficiency(report)
        benchmark_persona_manager(report)
        benchmark_security_overhead(report)

        # Print summary report
        report.print_summary()

        # Return exit code based on pass rate
        passed = sum(1 for r in report.results if r.passed)
        total = len(report.results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        if pass_rate >= 80:
            print("✓ Benchmarks completed successfully!")
            return 0
        else:
            print(f"⚠ Warning: Pass rate is {pass_rate:.1f}% (target: ≥80%)")
            return 1

    except Exception as e:
        print(f"\n✗ Error running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
