#!/usr/bin/env python3
"""Startup time benchmark for Victor.

Measures the time taken to import and initialize various components.
This helps identify bottlenecks in the startup process.
"""

import sys
import time
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def measure_import(module_path: str) -> Tuple[float, bool]:
    """Measure time to import a module.

    Args:
        module_path: Full module path to import

    Returns:
        Tuple of (time_taken, success)
    """
    start = time.perf_counter()
    try:
        # Clear from sys.modules if already imported
        if module_path in sys.modules:
            del sys.modules[module_path]

        importlib.import_module(module_path)
        elapsed = time.perf_counter() - start
        return elapsed, True
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  ERROR: {e}")
        return elapsed, False


def run_benchmark() -> Dict[str, float]:
    """Run startup benchmark.

    Returns:
        Dictionary of operation -> time_taken
    """
    results = {}

    print("=" * 70)
    print("Victor Startup Time Benchmark")
    print("=" * 70)

    # Test 1: Import core verticals module
    print("\n[1] Importing victor.core.verticals...")
    t, success = measure_import("victor.core.verticals")
    results["import_core_verticals"] = t
    if success:
        print(f"  ✓ Success in {t*1000:.2f}ms")
    else:
        print(f"  ✗ Failed in {t*1000:.2f}ms")

    # Test 2: Import individual verticals
    verticals = [
        "victor.coding",
        "victor.research",
        "victor.devops",
        "victor.dataanalysis",
        "victor.rag",
        "victor.benchmark",
    ]

    print("\n[2] Importing individual verticals...")
    vertical_times = []
    for vertical in verticals:
        print(f"  Importing {vertical}...")
        t, success = measure_import(vertical)
        if success:
            print(f"    ✓ Success in {t*1000:.2f}ms")
            vertical_times.append(t)
        else:
            print(f"    ✗ Failed in {t*1000:.2f}ms")
        results[f"import_{vertical.split('.')[-1]}"] = t

    results["avg_vertical_import"] = (
        sum(vertical_times) / len(vertical_times) if vertical_times else 0
    )
    results["total_vertical_import"] = sum(vertical_times)

    # Test 3: Access vertical metadata
    print("\n[3] Accessing vertical metadata...")
    try:
        from victor.core.verticals import VerticalRegistry

        start = time.perf_counter()
        names = VerticalRegistry.list_names()
        t = time.perf_counter() - start
        results["list_verticals"] = t
        print(f"  ✓ Listed {len(names)} verticals in {t*1000:.2f}ms")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 4: Get vertical configuration (triggers extension loading)
    print("\n[4] Getting vertical configurations (triggers extension loading)...")
    try:
        from victor.coding import CodingAssistant

        start = time.perf_counter()
        config = CodingAssistant.get_config()
        t = time.perf_counter() - start
        results["coding_get_config"] = t
        print(f"  ✓ CodingAssistant.get_config() in {t*1000:.2f}ms")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 5: Get vertical extensions
    print("\n[5] Getting vertical extensions...")
    try:
        from victor.coding import CodingAssistant

        start = time.perf_counter()
        extensions = CodingAssistant.get_extensions()
        t = time.perf_counter() - start
        results["coding_get_extensions"] = t
        print(f"  ✓ CodingAssistant.get_extensions() in {t*1000:.2f}ms")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Test 6: Import heavy modules
    print("\n[6] Importing heavy modules...")
    heavy_modules = [
        "victor.coding.assistant",
        "victor.coding.middleware",
        "victor.coding.safety",
        "victor.coding.prompts",
        "victor.coding.workflows",
    ]

    for module in heavy_modules:
        print(f"  Importing {module}...")
        t, success = measure_import(module)
        if success:
            print(f"    ✓ Success in {t*1000:.2f}ms")
        else:
            print(f"    ✗ Failed in {t*1000:.2f}ms")
        results[f"import_{module.replace('.', '_')}"] = t

    return results


def print_summary(results: Dict[str, float]) -> None:
    """Print benchmark summary.

    Args:
        results: Dictionary of operation -> time_taken
    """
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)

    # Sort by time taken (slowest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 Slowest Operations:")
    for op, t in sorted_results[:10]:
        print(f"  {op:40s}: {t*1000:8.2f}ms")

    # Calculate totals
    total_time = sum(results.values())

    print("\n" + "-" * 70)
    print(f"Total Time: {total_time*1000:.2f}ms")
    print("-" * 70)

    # Categorize operations
    categories = {
        "Core Infrastructure": [
            "import_core_verticals",
            "list_verticals",
        ],
        "Vertical Imports": [
            "import_coding",
            "import_research",
            "import_devops",
            "import_dataanalysis",
            "import_rag",
            "import_benchmark",
        ],
        "Extension Loading": [
            "coding_get_config",
            "coding_get_extensions",
        ],
    }

    print("\nBreakdown by Category:")
    for category, ops in categories.items():
        category_time = sum(results.get(op, 0) for op in ops)
        if category_time > 0:
            print(
                f"  {category:30s}: {category_time*1000:8.2f}ms ({category_time/total_time*100:5.1f}%)"
            )


if __name__ == "__main__":
    # Run benchmark
    results = run_benchmark()

    # Print summary
    print_summary(results)

    # Save results to file
    import json
    from pathlib import Path

    output_file = Path(__file__).parent / "startup_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump({k: v * 1000 for k, v in results.items()}, f, indent=2)

    print(f"\nResults saved to: {output_file}")
