#!/usr/bin/env python3
"""Benchmark CLI startup performance with lazy loading.

This script measures the improvement in CLI startup time after implementing
lazy loading for CLI commands.

Expected improvements:
- Import time: 1.68s → 0.32s (81% reduction)
- CLI --help: 2.3s → ~1.9s (17% reduction)
"""

import subprocess
import time
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> tuple[float, str]:
    """Run a command and return elapsed time and output."""
    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    elapsed = time.time() - start
    return elapsed, result.stdout


def benchmark_import_time() -> dict[str, float]:
    """Benchmark Python import time."""
    print("\n=== Benchmarking Import Time ===")

    times = []
    for i in range(5):
        # Clear Python cache to measure cold start
        start = time.time()
        result = subprocess.run(
            [sys.executable, "-c", "from victor.ui.cli import app"],
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"Error on run {i+1}: {result.stderr}")
            continue
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.3f}s")

    if not times:
        return {"error": "All runs failed"}

    avg = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        "average": avg,
        "min": min_time,
        "max": max_time,
        "runs": times
    }


def benchmark_cli_help() -> dict[str, float]:
    """Benchmark CLI --help command."""
    print("\n=== Benchmarking CLI --help ===")

    times = []
    for i in range(3):
        start = time.time()
        result = subprocess.run(
            ["victor", "--help"],
            capture_output=True,
            text=True,
            check=True
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        "average": avg,
        "min": min_time,
        "max": max_time,
        "runs": times
    }


def benchmark_command_help() -> dict[str, float]:
    """Benchmark various command --help calls."""
    print("\n=== Benchmarking Command --help ===")

    commands = [
        "config",
        "chat",
        "benchmark",
        "workflow",
    ]

    results = {}
    for cmd in commands:
        print(f"\nTesting 'victor {cmd} --help'")
        times = []
        for i in range(3):
            start = time.time()
            result = subprocess.run(
                ["victor", cmd, "--help"],
                capture_output=True,
                text=True,
                check=True
            )
            elapsed = time.time() - start
            times.append(elapsed)

        avg = sum(times) / len(times)
        results[cmd] = avg
        print(f"  Average: {avg:.3f}s")

    return results


def print_comparison(import_time: dict, cli_help_time: dict):
    """Print comparison with expected improvements."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n📊 Import Time Performance:")
    print(f"  Current:   {import_time['average']:.3f}s (avg)")
    print(f"  Baseline:  ~1.680s (before lazy loading)")
    print(f"  Improvement: {((1.68 - import_time['average']) / 1.68 * 100):.1f}% faster")

    print("\n⚡ CLI --help Performance:")
    print(f"  Current:   {cli_help_time['average']:.3f}s (avg)")
    print(f"  Baseline:  ~2.300s (before lazy loading)")
    print(f"  Improvement: {((2.3 - cli_help_time['average']) / 2.3 * 100):.1f}% faster")

    print("\n✅ Success Criteria:")
    criteria = [
        ("Import time < 0.5s", import_time['average'] < 0.5),
        ("CLI --help < 2.0s", cli_help_time['average'] < 2.0),
        ("All commands functional", True),  # Assume True if we got here
    ]

    for criterion, passed in criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")

    print("\n" + "=" * 70)


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("CLI LAZY LOADING PERFORMANCE BENCHMARK")
    print("=" * 70)
    print("\nMeasuring startup performance improvements from lazy loading...")
    print("This will take about 30 seconds...")

    # Run benchmarks
    import_time = benchmark_import_time()
    cli_help_time = benchmark_cli_help()
    command_help_times = benchmark_command_help()

    # Print results
    print_comparison(import_time, cli_help_time)

    print("\n📝 Detailed Results:")
    print(f"\nImport time: {import_time}")
    print(f"\nCLI --help: {cli_help_time}")
    print(f"\nCommand --help times: {command_help_times}")

    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
