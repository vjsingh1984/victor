#!/usr/bin/env python3
"""Benchmark lazy extension loading performance."""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def benchmark_eager_loading():
    """Benchmark eager extension loading."""
    print("=" * 60)
    print("Benchmark: Eager Extension Loading")
    print("=" * 60)

    # Set environment variable for eager loading
    os.environ["VICTOR_LAZY_EXTENSIONS"] = "false"

    times = {}

    # Test 1: Import core verticals module
    start = time.time()
    from victor.core.verticals import VerticalRegistry, list_verticals
    times["Import victor.core.verticals"] = time.time() - start
    print(f"✓ Import victor.core.verticals: {times['Import victor.core.verticals']:.3f}s")

    # Test 2: Import coding vertical
    start = time.time()
    from victor.coding import CodingAssistant
    times["Import coding vertical"] = time.time() - start
    print(f"✓ Import coding vertical: {times['Import coding vertical']:.3f}s")

    # Test 3: Get vertical config
    start = time.time()
    config = CodingAssistant.get_config()
    times["Get coding config"] = time.time() - start
    print(f"✓ Get coding config: {times['Get coding config']:.3f}s")

    # Test 4: Get extensions (eager loading)
    start = time.time()
    extensions = CodingAssistant.get_extensions()
    times["Get extensions (eager)"] = time.time() - start
    print(f"✓ Get extensions (eager): {times['Get extensions (eager)']:.3f}s")

    # Test 5: Access middleware
    start = time.time()
    middleware = extensions.middleware
    times["Access middleware"] = time.time() - start
    print(f"✓ Access middleware: {times['Access middleware']:.3f}s")

    total_eager = sum(times.values())
    print(f"\n📊 Total Time (Eager): {total_eager:.3f}s")

    return times, total_eager


def benchmark_lazy_loading():
    """Benchmark lazy extension loading."""
    print("\n" + "=" * 60)
    print("Benchmark: Lazy Extension Loading")
    print("=" * 60)

    # Clear import cache
    for module in list(sys.modules.keys()):
        if "victor" in module and "coding" in module:
            del sys.modules[module]

    # Set environment variable for lazy loading
    os.environ["VICTOR_LAZY_EXTENSIONS"] = "true"

    times = {}

    # Test 1: Import core verticals module
    start = time.time()
    from victor.core.verticals import VerticalRegistry, list_verticals
    times["Import victor.core.verticals"] = time.time() - start
    print(f"✓ Import victor.core.verticals: {times['Import victor.core.verticals']:.3f}s")

    # Test 2: Import coding vertical
    start = time.time()
    from victor.coding import CodingAssistant
    times["Import coding vertical"] = time.time() - start
    print(f"✓ Import coding vertical: {times['Import coding vertical']:.3f}s")

    # Test 3: Get vertical config
    start = time.time()
    config = CodingAssistant.get_config()
    times["Get coding config"] = time.time() - start
    print(f"✓ Get coding config: {times['Get coding config']:.3f}s")

    # Test 4: Get extensions (lazy loading)
    start = time.time()
    extensions = CodingAssistant.get_extensions()
    times["Get extensions (lazy)"] = time.time() - start
    print(f"✓ Get extensions (lazy wrapper): {times['Get extensions (lazy)']:.3f}s")

    # Test 5: First access to middleware (triggers loading)
    start = time.time()
    middleware = extensions.middleware
    times["First middleware access"] = time.time() - start
    print(f"✓ First middleware access (triggers load): {times['First middleware access']:.3f}s")

    # Test 6: Second access to middleware (uses cache)
    start = time.time()
    middleware2 = extensions.middleware
    times["Second middleware access (cached)"] = time.time() - start
    print(f"✓ Second middleware access (cached): {times['Second middleware access (cached)']:.3f}s")

    total_lazy = sum(times.values())
    print(f"\n📊 Total Time (Lazy): {total_lazy:.3f}s")

    return times, total_lazy


def print_comparison(eager_times, lazy_times, total_eager, total_lazy):
    """Print comparison between eager and lazy loading."""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    print(f"\n{'Operation':<40} {'Eager':>12} {'Lazy':>12} {'Speedup':>12}")
    print("-" * 76)

    # Compare all operations
    operations = set(eager_times.keys()) | set(lazy_times.keys())
    for op in sorted(operations):
        eager_time = eager_times.get(op, 0)
        lazy_time = lazy_times.get(op, 0)

        if eager_time > 0:
            speedup = ((eager_time - lazy_time) / eager_time) * 100
            print(f"{op:<40} {eager_time:>10.3f}s {lazy_time:>10.3f}s {speedup:>+10.1f}%")

    print("-" * 76)
    print(f"{'TOTAL':<40} {total_eager:>10.3f}s {total_lazy:>10.3f}s {((total_eager - total_lazy) / total_eager * 100):>+10.1f}%")
    print("=" * 60)

    # Calculate improvement
    improvement = ((total_eager - total_lazy) / total_eager) * 100
    print(f"\n🎉 Overall Improvement: {improvement:.1f}% faster")
    print(f"   Time Saved: {total_eager - total_lazy:.3f}s")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("Lazy Extension Loading Benchmark")
    print("=" * 60)
    print("\nThis benchmark compares eager vs lazy extension loading.")
    print("Lazy loading defers extension loading until first access,")
    print("significantly improving startup time.\n")

    # Run benchmarks
    eager_times, total_eager = benchmark_eager_loading()
    lazy_times, total_lazy = benchmark_lazy_loading()

    # Print comparison
    print_comparison(eager_times, lazy_times, total_eager, total_lazy)

    # Check if targets met
    print("\n" + "=" * 60)
    print("Target Validation")
    print("=" * 60)

    target_improvement = 20  # 20% target
    actual_improvement = ((total_eager - total_lazy) / total_eager) * 100

    print(f"\nTarget: >{target_improvement}% startup time improvement")
    print(f"Actual: {actual_improvement:.1f}% improvement")

    if actual_improvement > target_improvement:
        print(f"✅ Target MET! Exceeded by {actual_improvement - target_improvement:.1f}%")
    else:
        print(f"❌ Target NOT MET. Short by {target_improvement - actual_improvement:.1f}%")

    print(f"\nStartup time target: <2s")
    print(f"Actual startup time: {total_lazy:.3f}s")

    if total_lazy < 2.0:
        print("✅ Startup time target MET!")
    else:
        print("❌ Startup time target NOT MET.")


if __name__ == "__main__":
    main()
