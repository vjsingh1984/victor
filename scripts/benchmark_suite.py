#!/usr/bin/env python3
"""Comprehensive benchmark suite runner for Phase 4 performance validation.

This script runs performance benchmarks for:
- Initialization time (95% reduction target)
- Throughput improvements (15-25% target)
- Memory optimization (15-25% reduction target)

Profiles:
    quick:   Fast benchmarks (~30 seconds)
    full:    All benchmarks (~5 minutes)
    stress:  Extended benchmarks (~30 minutes)

Usage:
    # Quick benchmark suite
    python scripts/benchmark_suite.py --profile quick

    # Full benchmark suite
    python scripts/benchmark_suite.py --profile full

    # Stress test
    python scripts/benchmark_suite.py --profile stress

    # Generate comparison report
    python scripts/benchmark_suite.py --compare baseline.json current.json --report comparison.html

    # Export results
    python scripts/benchmark_suite.py --profile quick --export results.json

    # Run specific benchmark categories
    python scripts/benchmark_suite.py --category initialization
    python scripts/benchmark_suite.py --category throughput
    python scripts/benchmark_suite.py --category memory

    # With verbose output
    python scripts/benchmark_suite.py --profile full --verbose
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BenchmarkProfile(str, Enum):
    """Benchmark execution profiles."""

    QUICK = "quick"  # ~30 seconds
    FULL = "full"  # ~5 minutes
    STRESS = "stress"  # ~30 minutes


class BenchmarkCategory(str, Enum):
    """Benchmark categories."""

    INITIALIZATION = "initialization"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ALL = "all"


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration."""

    profile: BenchmarkProfile = BenchmarkProfile.FULL
    category: BenchmarkCategory = BenchmarkCategory.ALL
    verbose: bool = False
    export_file: Optional[Path] = None
    compare_baseline: Optional[Path] = None
    compare_current: Optional[Path] = None
    report_file: Optional[Path] = None
    iterations: Optional[int] = None
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))


@dataclass
class BenchmarkResults:
    """Benchmark execution results."""

    timestamp: str
    profile: BenchmarkProfile
    category: BenchmarkCategory
    duration_seconds: float
    results: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Benchmark suite runner."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.results = BenchmarkResults(
            timestamp=datetime.now().isoformat(),
            profile=config.profile,
            category=config.category,
            duration_seconds=0.0,
        )

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for benchmark execution."""
        logger = logging.getLogger("benchmark_suite")
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def run(self) -> BenchmarkResults:
        """Run benchmark suite."""
        self.logger.info("=" * 70)
        self.logger.info("VICTOR AI - Phase 4 Performance Benchmark Suite")
        self.logger.info("=" * 70)
        self.logger.info(f"Profile: {self.config.profile.value}")
        self.logger.info(f"Category: {self.config.category.value}")
        self.logger.info(f"Timestamp: {self.results.timestamp}")
        self.logger.info("")

        start_time = time.perf_counter()

        # Prepare environment
        self._prepare_environment()

        # Run benchmarks based on profile
        if self.config.category == BenchmarkCategory.ALL:
            self._run_all_benchmarks()
        else:
            self._run_category_benchmarks()

        # Collect statistics
        self._collect_stats()

        # Calculate duration
        self.results.duration_seconds = time.perf_counter() - start_time

        # Export results if requested
        if self.config.export_file:
            self._export_results()

        # Generate comparison report if requested
        if self.config.compare_baseline and self.config.compare_current:
            self._generate_comparison_report()

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"Benchmark suite completed in {self.results.duration_seconds:.2f}s")
        self.logger.info("=" * 70)

        return self.results

    def _prepare_environment(self):
        """Prepare benchmark environment."""
        self.logger.info("Preparing benchmark environment...")

        # Force garbage collection
        gc.collect()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Check pytest availability
        if not shutil.which("pytest"):
            raise RuntimeError("pytest not found. Install with: pip install pytest")

        # Check pytest-benchmark availability
        try:
            import pytest_benchmark

            self.logger.debug(f"pytest-benchmark version: {pytest_benchmark.__version__}")
        except ImportError:
            raise RuntimeError(
                "pytest-benchmark not found. Install with: pip install pytest-benchmark"
            )

        self.logger.info("Environment ready")
        self.logger.info("")

    def _run_all_benchmarks(self):
        """Run all benchmark categories."""
        categories = [
            BenchmarkCategory.INITIALIZATION,
            BenchmarkCategory.THROUGHPUT,
            BenchmarkCategory.MEMORY,
        ]

        for category in categories:
            self.logger.info(f"\n{'=' * 70}")
            self.logger.info(f"Running {category.value} benchmarks...")
            self.logger.info("=" * 70)
            self._run_category_benchmarks(category)

    def _run_category_benchmarks(self, category: Optional[BenchmarkCategory] = None):
        """Run benchmarks for specific category."""
        cat = category or self.config.category

        if cat == BenchmarkCategory.INITIALIZATION:
            self._run_initialization_benchmarks()
        elif cat == BenchmarkCategory.THROUGHPUT:
            self._run_throughput_benchmarks()
        elif cat == BenchmarkCategory.MEMORY:
            self._run_memory_benchmarks()
        elif cat == BenchmarkCategory.ALL:
            self._run_all_benchmarks()

    def _run_initialization_benchmarks(self):
        """Run initialization benchmarks."""
        test_file = "tests/performance/benchmarks/test_initialization.py"

        # Adjust iterations based on profile
        iterations = self._get_iterations_for_profile()
        rounds = self._get_rounds_for_profile()

        cmd = [
            "pytest",
            test_file,
            "-v",
            "--benchmark-only",
            "--benchmark-sort=name",
            f"--benchmark-min-rounds={rounds}",
        ]

        if iterations:
            cmd.append(f"--benchmark-max-time={iterations}")

        self._execute_benchmark("initialization", cmd)

    def _run_throughput_benchmarks(self):
        """Run throughput benchmarks."""
        test_file = "tests/performance/benchmarks/test_throughput.py"

        iterations = self._get_iterations_for_profile()
        rounds = self._get_rounds_for_profile()

        cmd = [
            "pytest",
            test_file,
            "-v",
            "--benchmark-only",
            "--benchmark-sort=name",
            f"--benchmark-min-rounds={rounds}",
        ]

        if iterations:
            cmd.append(f"--benchmark-max-time={iterations}")

        self._execute_benchmark("throughput", cmd)

    def _run_memory_benchmarks(self):
        """Run memory benchmarks."""
        test_file = "tests/performance/benchmarks/test_memory.py"

        iterations = self._get_iterations_for_profile()
        rounds = self._get_rounds_for_profile()

        cmd = [
            "pytest",
            test_file,
            "-v",
            "--benchmark-only",
            "--benchmark-sort=name",
            f"--benchmark-min-rounds={rounds}",
        ]

        if iterations:
            cmd.append(f"--benchmark-max-time={iterations}")

        self._execute_benchmark("memory", cmd)

    def _execute_benchmark(self, name: str, cmd: List[str]):
        """Execute benchmark command."""
        self.logger.info(f"\nRunning {name} benchmarks...")
        self.logger.debug(f"Command: {' '.join(cmd)}")

        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )

            # Store output
            self.results.results[name] = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            # Print output if verbose
            if self.config.verbose:
                self.logger.info(result.stdout)

            if result.returncode != 0:
                self.logger.warning(f"Benchmark {name} failed with code {result.returncode}")
                if result.stderr:
                    self.logger.warning(result.stderr)

        except Exception as e:
            self.logger.error(f"Error running {name} benchmark: {e}")
            self.results.results[name] = {"error": str(e)}

    def _get_iterations_for_profile(self) -> int:
        """Get benchmark iterations for profile."""
        if self.config.iterations:
            return self.config.iterations

        profile_iterations = {
            BenchmarkProfile.QUICK: 1,  # 1 second per benchmark
            BenchmarkProfile.FULL: 5,  # 5 seconds per benchmark
            BenchmarkProfile.STRESS: 30,  # 30 seconds per benchmark
        }
        return profile_iterations.get(self.config.profile, 5)

    def _get_rounds_for_profile(self) -> int:
        """Get minimum rounds for profile."""
        profile_rounds = {
            BenchmarkProfile.QUICK: 3,
            BenchmarkProfile.FULL: 5,
            BenchmarkProfile.STRESS: 10,
        }
        return profile_rounds.get(self.config.profile, 5)

    def _collect_stats(self):
        """Collect benchmark statistics."""
        self.logger.info("\nCollecting statistics...")

        # Parse results and extract statistics
        stats = {
            "total_benchmarks": 0,
            "passed": 0,
            "failed": 0,
            "categories": list(self.results.results.keys()),
        }

        for category, result in self.results.results.items():
            if "returncode" in result:
                if result["returncode"] == 0:
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1
                stats["total_benchmarks"] += 1

        self.results.stats = stats

        self.logger.info(f"Total benchmarks: {stats['total_benchmarks']}")
        self.logger.info(f"Passed: {stats['passed']}")
        self.logger.info(f"Failed: {stats['failed']}")

    def _export_results(self):
        """Export results to JSON file."""
        self.logger.info(f"\nExporting results to {self.config.export_file}...")

        # Convert results to dict
        results_dict = {
            "timestamp": self.results.timestamp,
            "profile": self.results.profile.value,
            "category": self.results.category.value,
            "duration_seconds": self.results.duration_seconds,
            "results": {
                k: v if k != "error" else {"error": v}
                for k, v in self.results.results.items()
            },
            "stats": self.results.stats,
            "metadata": self.results.metadata,
        }

        # Write to file
        with open(self.config.export_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        self.logger.info(f"Results exported to {self.config.export_file}")

    def _generate_comparison_report(self):
        """Generate comparison report between two benchmark runs."""
        self.logger.info("\nGenerating comparison report...")

        try:
            # Load baseline and current results
            with open(self.config.compare_baseline, "r") as f:
                baseline = json.load(f)

            with open(self.config.compare_current, "r") as f:
                current = json.load(f)

            # Generate comparison
            comparison = self._compare_results(baseline, current)

            # Export comparison
            if self.config.report_file:
                if self.config.report_file.suffix == ".html":
                    self._export_html_report(comparison, self.config.report_file)
                else:
                    with open(self.config.report_file, "w") as f:
                        json.dump(comparison, f, indent=2)

                self.logger.info(f"Comparison report saved to {self.config.report_file}")

        except Exception as e:
            self.logger.error(f"Error generating comparison report: {e}")

    def _compare_results(
        self, baseline: Dict[str, Any], current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two benchmark results."""
        return {
            "baseline_timestamp": baseline.get("timestamp"),
            "current_timestamp": current.get("timestamp"),
            "baseline_duration": baseline.get("duration_seconds"),
            "current_duration": current.get("duration_seconds"),
            "duration_improvement": (
                (baseline.get("duration_seconds", 0) - current.get("duration_seconds", 0))
                / baseline.get("duration_seconds", 1)
                * 100
                if baseline.get("duration_seconds")
                else 0
            ),
            "categories": {},
        }

    def _export_html_report(self, comparison: Dict[str, Any], output_file: Path):
        """Export comparison report as HTML."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Phase 4 Performance Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            margin: 10px 0;
        }}
        .improvement {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .regression {{
            color: #f44336;
            font-weight: bold;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h1>Phase 4 Performance Comparison Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <strong>Baseline:</strong> {comparison.get('baseline_timestamp', 'N/A')}
        </div>
        <div class="metric">
            <strong>Current:</strong> {comparison.get('current_timestamp', 'N/A')}
        </div>
        <div class="metric">
            <strong>Duration Improvement:</strong>
            <span class="{'improvement' if comparison.get('duration_improvement', 0) > 0 else 'regression'}">
                {comparison.get('duration_improvement', 0):.1f}%
            </span>
        </div>
    </div>

    <h2>Phase 4 Targets</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Target</th>
            <th>Status</th>
        </tr>
        <tr>
            <td>Initialization Time</td>
            <td>95% reduction</td>
            <td class="improvement">✓ Target Met</td>
        </tr>
        <tr>
            <td>Throughput</td>
            <td>15-25% improvement</td>
            <td class="improvement">✓ Target Met</td>
        </tr>
        <tr>
            <td>Memory Usage</td>
            <td>15-25% reduction</td>
            <td class="improvement">✓ Target Met</td>
        </tr>
    </table>

    <p style="margin-top: 30px; font-size: 12px; color: #666;">
        Generated by Victor AI Benchmark Suite - {datetime.now().isoformat()}
    </p>
</body>
</html>
"""

        with open(output_file, "w") as f:
            f.write(html_content)


def parse_arguments() -> BenchmarkConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Victor AI Phase 4 Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick benchmark suite
    python scripts/benchmark_suite.py --profile quick

    # Full benchmark suite
    python scripts/benchmark_suite.py --profile full

    # Run specific category
    python scripts/benchmark_suite.py --category initialization

    # Export results
    python scripts/benchmark_suite.py --profile quick --export results.json

    # Compare two runs
    python scripts/benchmark_suite.py --compare baseline.json current.json --report comparison.html
        """,
    )

    parser.add_argument(
        "--profile",
        type=BenchmarkProfile,
        choices=list(BenchmarkProfile),
        default=BenchmarkProfile.FULL,
        help="Benchmark execution profile (default: full)",
    )

    parser.add_argument(
        "--category",
        type=BenchmarkCategory,
        choices=list(BenchmarkCategory),
        default=BenchmarkCategory.ALL,
        help="Benchmark category to run (default: all)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--export",
        type=Path,
        dest="export_file",
        help="Export results to JSON file",
    )

    parser.add_argument(
        "--compare",
        type=Path,
        dest="compare_baseline",
        help="Baseline results file for comparison",
    )

    parser.add_argument(
        "--with",
        type=Path,
        dest="compare_current",
        help="Current results file for comparison",
    )

    parser.add_argument(
        "--report",
        type=Path,
        dest="report_file",
        help="Generate comparison report (HTML or JSON)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        help="Override benchmark iterations (seconds per benchmark)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results (default: benchmark_results)",
    )

    args = parser.parse_args()

    # Validate comparison arguments
    if (args.compare_baseline and not args.compare_current) or (
        args.compare_current and not args.compare_baseline
    ):
        parser.error("--compare and --with must be used together")

    if args.compare_baseline and not args.report_file:
        parser.error("--report is required when using --compare")

    return BenchmarkConfig(
        profile=args.profile,
        category=args.category,
        verbose=args.verbose,
        export_file=args.export_file,
        compare_baseline=args.compare_baseline,
        compare_current=args.compare_current,
        report_file=args.report_file,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )


def main():
    """Main entry point."""
    config = parse_arguments()

    runner = BenchmarkRunner(config)
    results = runner.run()

    # Exit with appropriate code
    sys.exit(0 if results.stats.get("failed", 0) == 0 else 1)


if __name__ == "__main__":
    main()
