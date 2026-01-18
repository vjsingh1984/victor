#!/usr/bin/env python3
"""Benchmark startup time for vertical loading."""

import argparse
import sys
import time
import importlib


def benchmark_import(module_name: str, iterations: int = 5) -> dict:
    """Benchmark import time for a module.

    Args:
        module_name: Module to import (e.g., "victor.coding")
        iterations: Number of iterations to run

    Returns:
        Dictionary with timing statistics
    """
    times = []

    for i in range(iterations):
        # Clear module cache to force fresh import
        if module_name in sys.modules:
            del sys.modules[module_name]

        start = time.perf_counter()
        try:
            importlib.import_module(module_name)
        except Exception as e:
            return {"error": str(e)}
        end = time.perf_counter()
        times.append(end - start)

    return {
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
        "median": sorted(times)[len(times) // 2],
        "iterations": iterations,
    }


def benchmark_all_verticals() -> dict:
    """Benchmark loading all verticals.

    Returns:
        Dictionary with timing statistics for each vertical
    """
    verticals = ["coding", "research", "devops", "dataanalysis", "benchmark"]
    results = {}

    for vertical in verticals:
        module_name = f"victor.{vertical}"
        print(f"Benchmarking {module_name}...")
        results[vertical] = benchmark_import(module_name)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark vertical loading startup time")
    parser.add_argument(
        "--vertical",
        type=str,
        help="Specific vertical to benchmark (e.g., coding, research)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all verticals",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations (default: 5)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    args = parser.parse_args()

    if args.all:
        results = benchmark_all_verticals()

        if args.format == "json":
            import json

            print(json.dumps(results, indent=2))
        else:
            print("\n=== Startup Time Benchmark Results ===\n")
            print(f"{'Vertical':<15} {'Min (s)':<10} {'Mean (s)':<10} {'Max (s)':<10}")
            print("-" * 45)
            for vertical, stats in results.items():
                if "error" in stats:
                    print(f"{vertical:<15} ERROR: {stats['error']}")
                else:
                    print(
                        f"{vertical:<15} {stats['min']:<10.4f} {stats['mean']:<10.4f} {stats['max']:<10.4f}"
                    )
    elif args.vertical:
        results = benchmark_import(f"victor.{args.vertical}", args.iterations)

        if args.format == "json":
            import json

            print(json.dumps(results, indent=2))
        else:
            print(f"\n=== {args.vertical.capitalize()} Vertical Startup Time ===\n")
            if "error" in results:
                print(f"ERROR: {results['error']}")
            else:
                print(f"Min:    {results['min']:.4f}s")
                print(f"Mean:   {results['mean']:.4f}s")
                print(f"Median: {results['median']:.4f}s")
                print(f"Max:    {results['max']:.4f}s")
                print(f"Iterations: {results['iterations']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
