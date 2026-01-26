#!/usr/bin/env python3
"""Benchmark comparing eager vs lazy loading for Victor startup.

This benchmark measures the improvement in startup time when using lazy loading
for vertical imports. It simulates common usage patterns.
"""

import sys
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def clear_imports():
    """Clear imported modules to allow fresh imports."""
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith("victor")]
    for module in modules_to_clear:
        del sys.modules[module]


def benchmark_eager_loading():
    """Benchmark eager loading (legacy behavior)."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Eager Loading (VICTOR_LAZY_LOADING=false)")
    print("=" * 70)

    import os

    os.environ["VICTOR_LAZY_LOADING"] = "false"

    clear_imports()

    start = time.perf_counter()

    # Simulate typical startup: import core verticals
    import victor.core.verticals  # noqa

    core_time = time.perf_counter() - start

    # Simulate accessing a vertical
    start_access = time.perf_counter()
    from victor.core.verticals import VerticalRegistry

    coding = VerticalRegistry.get("coding")
    access_time = time.perf_counter() - start_access

    total_time = core_time  # No additional cost for access (already loaded)

    print(f"\nResults:")
    print(f"  Import victor.core.verticals: {core_time*1000:.2f}ms")
    print(f"  Access 'coding' vertical:    {access_time*1000:.2f}ms (cached)")
    print(f"  Total startup time:          {total_time*1000:.2f}ms")

    return {
        "mode": "eager",
        "core_import_ms": core_time * 1000,
        "first_access_ms": access_time * 1000,
        "total_startup_ms": total_time * 1000,
    }


def benchmark_lazy_loading():
    """Benchmark lazy loading (new behavior)."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Lazy Loading (VICTOR_LAZY_LOADING=true)")
    print("=" * 70)

    import os

    os.environ["VICTOR_LAZY_LOADING"] = "true"

    clear_imports()

    start = time.perf_counter()

    # Simulate typical startup: import core verticals
    import victor.core.verticals  # noqa

    core_time = time.perf_counter() - start

    # Simulate accessing a vertical (triggers lazy import)
    start_access = time.perf_counter()
    from victor.core.verticals import VerticalRegistry

    coding = VerticalRegistry.get("coding")
    access_time = time.perf_counter() - start_access

    total_time = core_time + access_time

    print(f"\nResults:")
    print(f"  Import victor.core.verticals: {core_time*1000:.2f}ms")
    print(f"  Access 'coding' vertical:    {access_time*1000:.2f}ms (lazy load)")
    print(f"  Total startup time:          {total_time*1000:.2f}ms")

    return {
        "mode": "lazy",
        "core_import_ms": core_time * 1000,
        "first_access_ms": access_time * 1000,
        "total_startup_ms": total_time * 1000,
    }


def benchmark_scenario_use_only_one_vertical():
    """Scenario: User only uses one vertical (e.g., coding)."""
    print("\n" + "=" * 70)
    print("SCENARIO: Use only one vertical (coding)")
    print("=" * 70)

    # Lazy loading
    import os

    os.environ["VICTOR_LAZY_LOADING"] = "true"
    clear_imports()

    start = time.perf_counter()
    import victor.core.verticals

    import_time = time.perf_counter() - start

    start = time.perf_counter()
    from victor.core.verticals import VerticalRegistry

    coding = VerticalRegistry.get("coding")
    config = coding.get_config()
    access_time = time.perf_counter() - start

    lazy_total = import_time + access_time

    print(f"\nLazy Loading:")
    print(f"  Import:     {import_time*1000:.2f}ms")
    print(f"  First use:  {access_time*1000:.2f}ms")
    print(f"  Total:      {lazy_total*1000:.2f}ms")

    # Eager loading
    os.environ["VICTOR_LAZY_LOADING"] = "false"
    clear_imports()

    start = time.perf_counter()
    import victor.core.verticals

    import_time = time.perf_counter() - start

    start = time.perf_counter()
    from victor.core.verticals import VerticalRegistry

    coding = VerticalRegistry.get("coding")  # Already loaded
    config = coding.get_config()
    access_time = time.perf_counter() - start

    eager_total = import_time + access_time

    print(f"\nEager Loading:")
    print(f"  Import:     {import_time*1000:.2f}ms")
    print(f"  First use:  {access_time*1000:.2f}ms")
    print(f"  Total:      {eager_total*1000:.2f}ms")

    improvement = ((eager_total - lazy_total) / eager_total) * 100
    print(f"\nImprovement: {improvement:.1f}% faster ({(eager_total-lazy_total)*1000:.2f}ms saved)")

    return {
        "lazy_ms": lazy_total * 1000,
        "eager_ms": eager_total * 1000,
        "improvement_percent": improvement,
    }


def benchmark_scenario_list_verticals():
    """Scenario: User just lists available verticals (no access)."""
    print("\n" + "=" * 70)
    print("SCENARIO: List verticals only (no access)")
    print("=" * 70)

    # Lazy loading
    import os

    os.environ["VICTOR_LAZY_LOADING"] = "true"
    clear_imports()

    start = time.perf_counter()
    import victor.core.verticals
    from victor.core.verticals import VerticalRegistry

    names = VerticalRegistry.list_names()
    lazy_time = time.perf_counter() - start

    print(f"\nLazy Loading: {lazy_time*1000:.2f}ms")
    print(f"  Found {len(names)} verticals")

    # Eager loading
    os.environ["VICTOR_LAZY_LOADING"] = "false"
    clear_imports()

    start = time.perf_counter()
    import victor.core.verticals
    from victor.core.verticals import VerticalRegistry

    names = VerticalRegistry.list_names()
    eager_time = time.perf_counter() - start

    print(f"\nEager Loading: {eager_time*1000:.2f}ms")
    print(f"  Found {len(names)} verticals")

    improvement = ((eager_time - lazy_time) / eager_time) * 100
    print(f"\nImprovement: {improvement:.1f}% faster ({(eager_time-lazy_time)*1000:.2f}ms saved)")

    return {
        "lazy_ms": lazy_time * 1000,
        "eager_ms": eager_time * 1000,
        "improvement_percent": improvement,
    }


def print_comparison_summary(eager_results, lazy_results):
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print("\nStartup Time Breakdown:")
    print(f"{'Metric':<30} {'Eager':>15} {'Lazy':>15} {'Improvement':>15}")
    print("-" * 70)

    print(
        f"{'Core Import':<30} {eager_results['core_import_ms']:>14.1f}ms {lazy_results['core_import_ms']:>14.1f}ms "
        f"{((eager_results['core_import_ms']-lazy_results['core_import_ms'])/eager_results['core_import_ms']*100):>14.1f}%"
    )

    print(
        f"{'First Vertical Access':<30} {eager_results['first_access_ms']:>14.1f}ms {lazy_results['first_access_ms']:>14.1f}ms "
        f"N/A (lazy loads here)"
    )

    print(
        f"{'Total Startup':<30} {eager_results['total_startup_ms']:>14.1f}ms {lazy_results['total_startup_ms']:>14.1f}ms "
        f"{((eager_results['total_startup_ms']-lazy_results['total_startup_ms'])/eager_results['total_startup_ms']*100):>14.1f}%"
    )

    # Calculate startup improvement
    startup_improvement_ms = eager_results["core_import_ms"] - lazy_results["core_import_ms"]
    startup_improvement_pct = (
        (eager_results["core_import_ms"] - lazy_results["core_import_ms"])
        / eager_results["core_import_ms"]
        * 100
    )

    print("\n" + "=" * 70)
    print(f"KEY FINDING: Lazy loading improves startup time by")
    print(f"  {startup_improvement_pct:.1f}% ({startup_improvement_ms:.1f}ms faster)")
    print("=" * 70)

    print("\nNOTE: First access to a vertical is slower with lazy loading,")
    print("but this is a one-time cost amortized over the entire session.")
    print("For typical usage (1-2 verticals per session), lazy loading provides")
    print("significant startup time improvements.")


if __name__ == "__main__":
    print("=" * 70)
    print("Victor Lazy Loading Benchmark")
    print("=" * 70)
    print("\nThis benchmark compares eager vs lazy loading for vertical imports.")
    print("It simulates common usage patterns to measure real-world impact.")

    # Run benchmarks
    eager_results = benchmark_eager_loading()
    lazy_results = benchmark_lazy_loading()

    # Run scenario benchmarks
    scenario1 = benchmark_scenario_use_only_one_vertical()
    scenario2 = benchmark_scenario_list_verticals()

    # Print comparison
    print_comparison_summary(eager_results, lazy_results)

    # Save results
    results = {
        "eager": eager_results,
        "lazy": lazy_results,
        "scenarios": {
            "use_one_vertical": scenario1,
            "list_verticals_only": scenario2,
        },
    }

    output_file = Path(__file__).parent / "lazy_loading_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
