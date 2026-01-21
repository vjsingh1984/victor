#!/usr/bin/env python3
"""Example: Using the Victor AI Comprehensive Benchmark Suite.

This example demonstrates how to use the benchmark suite to measure
and track performance improvements in Victor AI.
"""

from pathlib import Path
from scripts.benchmark_comprehensive import (
    BenchmarkRunner,
    BenchmarkReport,
    BenchmarkResult,
    MarkdownFormatter,
    JsonFormatter,
)


def example_run_benchmarks():
    """Example: Run all benchmarks and generate report."""
    print("Example 1: Running all benchmarks\n")

    runner = BenchmarkRunner()

    # Run all benchmarks
    report = runner.run(scenario="all")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total: {len(report.results)}")
    print(f"  Passed: {report.get_passed_count()}")
    print(f"  Failed: {report.get_failed_count()}")


def example_run_specific_scenario():
    """Example: Run specific benchmark scenario."""
    print("\nExample 2: Running startup benchmarks only\n")

    runner = BenchmarkRunner()

    # Run only startup benchmarks
    report = runner.run(scenario="startup")

    # Print results
    for result in report.results:
        print(f"\n{result.name}:")
        for metric in result.metrics:
            print(f"  {metric.name}: {metric.value:.2f} {metric.unit}")


def example_custom_benchmark():
    """Example: Create and run custom benchmark."""
    print("\nExample 3: Creating custom benchmark\n")

    from scripts.benchmark_comprehensive import BenchmarkResult

    # Create custom result
    result = BenchmarkResult(name="Custom Operation", category="custom")

    # Measure custom operation
    import time

    start = time.perf_counter()

    # Your custom operation here
    time.sleep(0.01)  # Simulate work

    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000

    # Add metric with threshold
    result.add_metric(
        name="operation_time",
        value=elapsed_ms,
        unit="ms",
        description="Time to complete custom operation",
        threshold=20.0,  # Fail if >20ms
    )

    print(f"Custom operation took {elapsed_ms:.2f}ms")
    print(f"Status: {'PASS' if result.passed else 'FAIL'}")


def example_generate_reports():
    """Example: Generate reports in different formats."""
    print("\nExample 4: Generating reports\n")

    runner = BenchmarkRunner()

    # Run benchmarks
    report = runner.run(scenario="startup", save=False)

    # Generate Markdown report
    markdown_formatter = MarkdownFormatter()
    markdown_report = markdown_formatter.format(report)

    print("Markdown Report:")
    print(markdown_report[:500] + "...\n")

    # Generate JSON report
    json_formatter = JsonFormatter()
    json_report = json_formatter.format(report)

    print("JSON Report:")
    print(json_report[:300] + "...\n")


def example_compare_results():
    """Example: Compare two benchmark runs."""
    print("\nExample 5: Comparing benchmark results\n")

    import json
    from scripts.compare_benchmarks import compare_benchmark_runs, format_markdown_report

    # Simulate baseline results
    baseline = {
        "start_time": "2025-01-18T10:00:00",
        "results": [
            {
                "name": "Cold Start",
                "category": "startup",
                "metrics": [
                    {"name": "startup_time", "value": 8000.0, "unit": "ms"}
                ],
            }
        ],
    }

    # Simulate current results (improved)
    current = {
        "start_time": "2025-01-18T12:00:00",
        "results": [
            {
                "name": "Cold Start",
                "category": "startup",
                "metrics": [
                    {"name": "startup_time", "value": 150.0, "unit": "ms"}
                ],
            }
        ],
    }

    # Compare
    comparison = compare_benchmark_runs(baseline, current)

    # Format and print
    report = format_markdown_report(comparison)
    print(report)


def example_performance_targets():
    """Example: Set and verify performance targets."""
    print("\nExample 6: Setting performance targets\n")

    from scripts.benchmark_comprehensive import BenchmarkResult

    # Define performance targets
    targets = {
        "startup_time_ms": 200,
        "memory_mb": 50,
        "throughput_ops_per_sec": 1000,
        "p95_latency_ms": 2,
    }

    print("Performance Targets:")
    for metric, target in targets.items():
        print(f"  {metric}: {target}")

    # Simulate measured values
    measured = {
        "startup_time_ms": 150,
        "memory_mb": 45,
        "throughput_ops_per_sec": 1200,
        "p95_latency_ms": 1.5,
    }

    print("\nMeasured Values:")
    for metric, value in measured.items():
        target = targets[metric]
        status = "✓ PASS" if value <= target else "✗ FAIL"
        print(f"  {metric}: {value} {status}")


def example_ci_cd_integration():
    """Example: CI/CD integration pattern."""
    print("\nExample 7: CI/CD integration pattern\n")

    from scripts.benchmark_comprehensive import BenchmarkRunner

    runner = BenchmarkRunner()

    # Run benchmarks
    report = runner.run(scenario="all", save=True)

    # Check for failures
    if report.get_failed_count() > 0:
        print("❌ Performance regression detected!")
        print(f"   {report.get_failed_count()} benchmark(s) failed")

        # In CI, fail the build
        # exit(1)
    else:
        print("✅ All benchmarks passed!")
        print(f"   {report.get_passed_count()} benchmark(s) passed")


def main():
    """Run all examples."""
    print("=" * 80)
    print("Victor AI Comprehensive Benchmark Suite - Usage Examples")
    print("=" * 80)

    # Run examples
    example_run_benchmarks()
    example_run_specific_scenario()
    example_custom_benchmark()
    example_generate_reports()
    example_compare_results()
    example_performance_targets()
    example_ci_cd_integration()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
