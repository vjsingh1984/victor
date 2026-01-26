#!/usr/bin/env python3
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

"""Benchmark script for measuring lazy loading performance improvements.

This script measures the impact of lazy loading on Victor's startup time
and first-access overhead. It compares eager vs lazy loading modes.

Usage:
    python scripts/benchmark_lazy_loading.py
    python scripts/benchmark_lazy_loading.py --verticals coding,research
    python scripts/benchmark_lazy_loading.py --output markdown

Environment Variables:
    VICTOR_LAZY_EXTENSIONS: Control lazy loading (true/false/auto)
    VICTOR_VERTICAL_LOADING_MODE: Control vertical loading (eager/lazy/auto)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    duration_ms: float
    memory_mb: float
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""

    eager_startup: BenchmarkResult
    lazy_startup: BenchmarkResult
    first_access_overhead: List[BenchmarkResult]
    memory_usage: Dict[str, float]
    verticals_tested: List[str]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class LazyLoadingBenchmark:
    """Benchmark suite for lazy loading performance."""

    def __init__(self, verticals: Optional[List[str]] = None):
        """Initialize benchmark suite.

        Args:
            verticals: List of verticals to test (default: all available)
        """
        self.verticals = verticals or ["coding", "research", "devops", "dataanalysis", "benchmark"]
        self.results: Dict[str, Any] = {}

    def measure_startup_time(self, mode: str) -> BenchmarkResult:
        """Measure startup time for a given loading mode.

        Args:
            mode: Loading mode (eager, lazy, auto)

        Returns:
            BenchmarkResult with duration and memory
        """
        import psutil
        import subprocess

        # Set environment variable for loading mode
        env = os.environ.copy()
        env["VICTOR_LAZY_EXTENSIONS"] = "true" if mode == "lazy" else "false"
        env["VICTOR_VERTICAL_LOADING_MODE"] = mode
        env["VICTOR_SKIP_ENV_FILE"] = "1"  # Skip .env to avoid side effects

        # Create a test script that imports and loads verticals
        test_script = """
import sys
import gc

# Force garbage collection before starting
gc.collect()

# Import and load verticals
from victor.core.verticals.vertical_loader import VerticalLoader

loader = VerticalLoader()

for vertical in ['{verticals}']:
    try:
        loader.load(vertical)
    except Exception as e:
        print(f"ERROR loading {{vertical}}: {{e}}", file=sys.stderr)
        sys.exit(1)

print("SUCCESS")
""".format(
            verticals="', '".join(self.verticals)
        )

        # Measure startup time
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.perf_counter()

        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root),
            )

            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB

            success = result.returncode == 0 and "SUCCESS" in result.stdout
            error = None if success else result.stderr.strip()

            return BenchmarkResult(
                name=f"startup_{mode}",
                duration_ms=(end_time - start_time) * 1000,
                memory_mb=end_memory - start_memory,
                success=success,
                error=error,
            )
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                name=f"startup_{mode}",
                duration_ms=30000,  # Timeout
                memory_mb=0,
                success=False,
                error="Timeout after 30 seconds",
            )
        except Exception as e:
            return BenchmarkResult(
                name=f"startup_{mode}",
                duration_ms=0,
                memory_mb=0,
                success=False,
                error=str(e),
            )

    def measure_first_access_overhead(self, vertical: str, mode: str) -> BenchmarkResult:
        """Measure first-access overhead for lazy loading.

        Args:
            vertical: Vertical name to test
            mode: Loading mode (lazy only makes sense for this test)

        Returns:
            BenchmarkResult with first-access duration
        """
        test_script = f"""
import sys
import time

# Import and load vertically (lazy mode)
from victor.core.verticals.vertical_loader import VerticalLoader

loader = VerticalLoader()
vertical = loader.load('{vertical}', lazy={mode == 'lazy'})

# Measure first access to extensions
start = time.perf_counter()
extensions = vertical.get_extensions()
middleware = extensions.middleware
end = time.perf_counter()

print(f"FIRST_ACCESS:{{(end - start) * 1000:.2f}}")
"""

        import subprocess

        env = os.environ.copy()
        env["VICTOR_LAZY_EXTENSIONS"] = "true" if mode == "lazy" else "false"
        env["VICTOR_SKIP_ENV_FILE"] = "1"

        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root),
            )

            if result.returncode != 0:
                return BenchmarkResult(
                    name=f"first_access_{vertical}_{mode}",
                    duration_ms=0,
                    memory_mb=0,
                    success=False,
                    error=result.stderr.strip(),
                )

            # Parse output
            for line in result.stdout.splitlines():
                if line.startswith("FIRST_ACCESS:"):
                    duration_ms = float(line.split(":")[1])
                    return BenchmarkResult(
                        name=f"first_access_{vertical}_{mode}",
                        duration_ms=duration_ms,
                        memory_mb=0,
                        success=True,
                    )

            return BenchmarkResult(
                name=f"first_access_{vertical}_{mode}",
                duration_ms=0,
                memory_mb=0,
                success=False,
                error="Could not parse output",
            )
        except Exception as e:
            return BenchmarkResult(
                name=f"first_access_{vertical}_{mode}",
                duration_ms=0,
                memory_mb=0,
                success=False,
                error=str(e),
            )

    def run_benchmark_suite(self) -> BenchmarkReport:
        """Run complete benchmark suite.

        Returns:
            BenchmarkReport with all results
        """
        print(f"\n{'='*60}")
        print(f"Lazy Loading Benchmark - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        print(f"Verticals tested: {', '.join(self.verticals)}")
        print(f"Python version: {sys.version}")
        print()

        # Benchmark 1: Eager startup time
        print("Benchmark 1: Measuring eager startup time...")
        eager_result = self.measure_startup_time("eager")
        print(
            f"  Result: {eager_result.duration_ms:.2f}ms "
            + (f"✓" if eager_result.success else f"✗ ({eager_result.error})")
        )

        # Force garbage collection between runs
        gc.collect()

        # Benchmark 2: Lazy startup time
        print("\nBenchmark 2: Measuring lazy startup time...")
        lazy_result = self.measure_startup_time("lazy")
        print(
            f"  Result: {lazy_result.duration_ms:.2f}ms "
            + (f"✓" if lazy_result.success else f"✗ ({lazy_result.error})")
        )

        # Force garbage collection between runs
        gc.collect()

        # Benchmark 3: First access overhead
        print("\nBenchmark 3: Measuring first-access overhead...")
        first_access_results = []
        for vertical in self.verticals:
            result = self.measure_first_access_overhead(vertical, "lazy")
            first_access_results.append(result)
            if result.success:
                print(f"  {vertical}: {result.duration_ms:.2f}ms ✓")
            else:
                print(f"  {vertical}: ✗ ({result.error})")

        # Calculate improvements
        if eager_result.success and lazy_result.success:
            startup_improvement_pct = (
                (eager_result.duration_ms - lazy_result.duration_ms)
                / eager_result.duration_ms
                * 100
            )
            print(f"\n{'='*60}")
            print(f"Startup Time Improvement: {startup_improvement_pct:.1f}%")
            print(f"  Eager: {eager_result.duration_ms:.2f}ms")
            print(f"  Lazy:  {lazy_result.duration_ms:.2f}ms")
            print(f"  Saved: {eager_result.duration_ms - lazy_result.duration_ms:.2f}ms")

            if first_access_results:
                avg_first_access = sum(
                    r.duration_ms for r in first_access_results if r.success
                ) / len([r for r in first_access_results if r.success])
                print(f"\nAverage First-Access Overhead: {avg_first_access:.2f}ms")

            print(f"{'='*60}\n")

        return BenchmarkReport(
            eager_startup=eager_result,
            lazy_startup=lazy_result,
            first_access_overhead=first_access_results,
            memory_usage={},
            verticals_tested=self.verticals,
        )


def print_report_markdown(report: BenchmarkReport) -> None:
    """Print benchmark report in Markdown format.

    Args:
        report: BenchmarkReport to print
    """
    print(f"\n# Lazy Loading Benchmark Report")
    print(f"\n**Generated:** {report.timestamp}")
    print(f"\n## Verticals Tested\n")
    for vertical in report.verticals_tested:
        print(f"- {vertical}")

    print(f"\n## Startup Time Comparison\n")
    print(f"| Mode | Duration (ms) | Memory (MB) | Status |")
    print(f"|------|---------------|-------------|--------|")

    if report.eager_startup.success:
        print(
            f"| Eager | {report.eager_startup.duration_ms:.2f} | "
            f"{report.eager_startup.memory_mb:.2f} | ✓ |"
        )
    else:
        print(f"| Eager | - | - | ✗ |")

    if report.lazy_startup.success:
        print(
            f"| Lazy  | {report.lazy_startup.duration_ms:.2f} | "
            f"{report.lazy_startup.memory_mb:.2f} | ✓ |"
        )
    else:
        print(f"| Lazy  | - | - | ✗ |")

    # Calculate improvement
    if report.eager_startup.success and report.lazy_startup.success:
        improvement = (
            (report.eager_startup.duration_ms - report.lazy_startup.duration_ms)
            / report.eager_startup.duration_ms
            * 100
        )
        print(
            f"\n**Improvement:** {improvement:.1f}% faster startup "
            f"({report.eager_startup.duration_ms - report.lazy_startup.duration_ms:.2f}ms saved)"
        )

    print(f"\n## First-Access Overhead\n")
    print(f"| Vertical | Duration (ms) | Status |")
    print(f"|----------|---------------|--------|")
    for result in report.first_access_overhead:
        vertical_name = "_".join(result.name.split("_")[2:])  # Extract vertical name
        if result.success:
            print(f"| {vertical_name} | {result.duration_ms:.2f} | ✓ |")
        else:
            print(f"| {vertical_name} | - | ✗ ({result.error}) |")

    if report.first_access_overhead:
        successful = [r for r in report.first_access_overhead if r.success]
        if successful:
            avg = sum(r.duration_ms for r in successful) / len(successful)
            print(f"\n**Average:** {avg:.2f}ms")


def print_report_json(report: BenchmarkReport) -> None:
    """Print benchmark report in JSON format.

    Args:
        report: BenchmarkReport to print
    """
    data = {
        "timestamp": report.timestamp,
        "verticals_tested": report.verticals_tested,
        "eager_startup": {
            "duration_ms": report.eager_startup.duration_ms,
            "memory_mb": report.eager_startup.memory_mb,
            "success": report.eager_startup.success,
            "error": report.eager_startup.error,
        },
        "lazy_startup": {
            "duration_ms": report.lazy_startup.duration_ms,
            "memory_mb": report.lazy_startup.memory_mb,
            "success": report.lazy_startup.success,
            "error": report.lazy_startup.error,
        },
        "first_access_overhead": [
            {
                "name": r.name,
                "duration_ms": r.duration_ms,
                "success": r.success,
                "error": r.error,
            }
            for r in report.first_access_overhead
        ],
    }

    if report.eager_startup.success and report.lazy_startup.success:
        data["improvement_pct"] = (
            (report.eager_startup.duration_ms - report.lazy_startup.duration_ms)
            / report.eager_startup.duration_ms
            * 100
        )
        data["time_saved_ms"] = report.eager_startup.duration_ms - report.lazy_startup.duration_ms

    print(json.dumps(data, indent=2))


def main() -> int:
    """Main entry point for benchmark script.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(description="Benchmark lazy loading performance improvements")
    parser.add_argument(
        "--verticals",
        type=str,
        help="Comma-separated list of verticals to test (default: all)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1)",
    )

    args = parser.parse_args()

    # Parse verticals list
    verticals = None
    if args.verticals:
        verticals = [v.strip() for v in args.verticals.split(",")]

    # Create benchmark suite
    benchmark = LazyLoadingBenchmark(verticals=verticals)

    # Run benchmark
    report = benchmark.run_benchmark_suite()

    # Print report
    if args.output == "markdown":
        print_report_markdown(report)
    elif args.output == "json":
        print_report_json(report)
    else:
        # Default text output already printed during benchmark
        pass

    # Return exit code based on success
    return 0 if (report.eager_startup.success and report.lazy_startup.success) else 1


if __name__ == "__main__":
    sys.exit(main())
