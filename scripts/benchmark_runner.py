#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for Victor Workflow System

This script provides a unified interface for running all performance benchmarks
across the Victor workflow system, including:

- Team node execution benchmarks
- Visual workflow editor benchmarks
- Workflow execution benchmarks
- Custom benchmark suites

Usage:
    # Run all benchmarks
    python scripts/benchmark_runner.py --all

    # Run specific benchmark suite
    python scripts/benchmark_runner.py --suite team_nodes
    python scripts/benchmark_runner.py --suite editor
    python scripts/benchmark_runner.py --suite workflow_execution

    # Run with specific filters
    python scripts/benchmark_runner.py --suite team_nodes --filter "formation"

    # Generate report after running
    python scripts/benchmark_runner.py --all --report

    # Run with custom iterations
    python scripts/benchmark_runner.py --all --iterations 5

    # Compare with baseline
    python scripts/benchmark_runner.py --all --compare baseline.json

    # Output formats
    python scripts/benchmark_runner.py --all --format json
    python scripts/benchmark_runner.py --all --format markdown
    python scripts/benchmark_runner.py --all --format html

Examples:
    # Quick smoke test (run key benchmarks only)
    python scripts/benchmark_runner.py --quick

    # Full benchmark suite with report
    python scripts/benchmark_runner.py --all --report --format markdown

    # CI/CD integration (JSON output, fail on regression)
    python scripts/benchmark_runner.py --all --format json --regression-check
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Configuration
# =============================================================================


@dataclass
class BenchmarkSuite:
    """Configuration for a benchmark suite."""

    name: str
    description: str
    file_path: str
    markers: List[str] = field(default_factory=list)
    performance_targets: Dict[str, float] = field(default_factory=dict)


# Define all benchmark suites
BENCHMARK_SUITES = {
    "team_nodes": BenchmarkSuite(
        name="Team Node Performance",
        description="Benchmarks for team node execution with different formations",
        file_path="tests/performance/team_node_benchmarks.py",
        markers=["benchmark"],
        performance_targets={
            "team_execution_3_members": 5000,  # 5 seconds
            "recursion_overhead_per_level": 1,  # 1ms
            "memory_10_members": 10485760,  # 10MB
        },
    ),
    "editor": BenchmarkSuite(
        name="Visual Workflow Editor",
        description="Benchmarks for the visual workflow editor UI",
        file_path="tests/performance/editor_benchmarks.py",
        markers=["benchmark"],
        performance_targets={
            "editor_load_100_nodes": 500,  # 500ms
            "node_rendering_per_node": 16,  # 16ms (60fps)
            "auto_layout_100_nodes": 1000,  # 1 second
            "yaml_import_100_nodes": 500,  # 500ms
            "yaml_export_100_nodes": 300,  # 300ms
            "memory_100_nodes": 104857600,  # 100MB
        },
    ),
    "workflow_execution": BenchmarkSuite(
        name="Workflow Execution",
        description="Benchmarks for workflow execution performance",
        file_path="tests/performance/workflow_execution_benchmarks.py",
        markers=["benchmark"],
        performance_targets={
            "simple_workflow_5_nodes": 100,  # 100ms
            "medium_workflow_20_nodes": 400,  # 400ms
            "complex_workflow_50_nodes": 1000,  # 1 second
            "throughput_nodes_per_second": 100,  # 100 nodes/s
            "recursion_overhead_per_level": 5,  # 5%
            "tool_execution_per_call": 5,  # 5ms
        },
    ),
}


QUICK_BENCHMARKS = {
    "team_nodes": ["test_formation_performance"],
    "editor": ["test_editor_load_time", "test_node_rendering_performance"],
    "workflow_execution": ["test_linear_workflow_execution", "test_node_throughput"],
}


# =============================================================================
# Benchmark Runner
# =============================================================================


class BenchmarkRunner:
    """Unified benchmark runner for Victor workflow system."""

    def __init__(
        self,
        output_dir: str = "/tmp/benchmark_results",
        verbose: bool = False,
        iterations: int = 1,
    ):
        """Initialize the benchmark runner.

        Args:
            output_dir: Directory to store benchmark results
            verbose: Enable verbose output
            iterations: Number of times to run each benchmark
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self.iterations = iterations
        self.results: Dict[str, Any] = {}

    def run_suite(
        self,
        suite_name: str,
        filter_expr: Optional[str] = None,
        quick: bool = False,
    ) -> Dict[str, Any]:
        """Run a specific benchmark suite.

        Args:
            suite_name: Name of the benchmark suite to run
            filter_expr: Optional pytest filter expression
            quick: Run only quick benchmarks

        Returns:
            Dictionary with benchmark results
        """
        if suite_name not in BENCHMARK_SUITES:
            logger.error(f"Unknown benchmark suite: {suite_name}")
            logger.info(f"Available suites: {', '.join(BENCHMARK_SUITES.keys())}")
            return {}

        suite = BENCHMARK_SUITES[suite_name]
        logger.info(f"Running benchmark suite: {suite.name}")
        logger.info(f"Description: {suite.description}")

        # Build pytest command
        pytest_args = [
            "pytest",
            suite.file_path,
            "-v",
            "--tb=short",
            "--benchmark-only",
            "--benchmark-sort=name",
        ]

        # Add iterations
        if self.iterations > 1:
            pytest_args.extend(["--benchmark-min-rounds", str(self.iterations)])

        # Add filter if specified
        if filter_expr:
            pytest_args.extend(["-k", filter_expr])
        elif quick and suite_name in QUICK_BENCHMARKS:
            # Run only quick benchmarks
            quick_filters = "|".join(QUICK_BENCHMARKS[suite_name])
            pytest_args.extend(["-k", quick_filters])

        # Add JSON output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f"{suite_name}_{timestamp}.json"
        pytest_args.extend(["--benchmark-json", str(json_file)])

        # Run pytest
        logger.info(f"Running: {' '.join(pytest_args)}")
        start_time = time.time()

        try:
            result = subprocess.run(
                pytest_args,
                capture_output=not self.verbose,
                text=True,
                check=False,
            )

            elapsed = time.time() - start_time
            logger.info(f"Completed in {elapsed:.2f} seconds")

            # Load JSON results
            if json_file.exists():
                with open(json_file) as f:
                    benchmark_data = json.load(f)

                self.results[suite_name] = {
                    "file": str(json_file),
                    "data": benchmark_data,
                    "exit_code": result.returncode,
                    "elapsed_seconds": elapsed,
                }

                return self.results[suite_name]
            else:
                logger.warning(f"Benchmark JSON file not found: {json_file}")
                return {"exit_code": result.returncode, "elapsed_seconds": elapsed}

        except Exception as e:
            logger.error(f"Error running benchmark suite: {e}")
            return {"error": str(e)}

    def run_all(
        self,
        filter_expr: Optional[str] = None,
        quick: bool = False,
    ) -> Dict[str, Any]:
        """Run all benchmark suites.

        Args:
            filter_expr: Optional pytest filter expression
            quick: Run only quick benchmarks

        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Running all benchmark suites")
        logger.info("=" * 80)

        all_results = {}
        for suite_name in BENCHMARK_SUITES.keys():
            logger.info(f"\n{'=' * 80}")
            suite_result = self.run_suite(suite_name, filter_expr, quick)
            all_results[suite_name] = suite_result

        return all_results

    def compare_with_baseline(
        self,
        baseline_file: str,
    ) -> Dict[str, Any]:
        """Compare current results with baseline.

        Args:
            baseline_file: Path to baseline JSON file

        Returns:
            Dictionary with comparison results
        """
        baseline_path = Path(baseline_file)
        if not baseline_path.exists():
            logger.error(f"Baseline file not found: {baseline_file}")
            return {}

        with open(baseline_path) as f:
            baseline = json.load(f)

        comparisons = []

        for suite_name, suite_result in self.results.items():
            if "data" not in suite_result:
                continue

            current_data = suite_result["data"]
            baseline_suite = baseline.get(suite_name, {})

            # Compare benchmarks
            comparison = self._compare_suite_results(
                suite_name,
                current_data,
                baseline_suite,
            )
            comparisons.append(comparison)

        return {
            "baseline_file": baseline_file,
            "comparisons": comparisons,
            "timestamp": datetime.now().isoformat(),
        }

    def _compare_suite_results(
        self,
        suite_name: str,
        current: Dict[str, Any],
        baseline: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare results for a single suite."""
        benchmarks = current.get("benchmarks", {})
        baseline_benchmarks = baseline.get("benchmarks", {})

        regressions = []
        improvements = []

        for bench_name, bench_data in benchmarks.items():
            if bench_name not in baseline_benchmarks:
                continue

            baseline_data = baseline_benchmarks[bench_name]
            current_median = bench_data.get("stats", {}).get("median", 0)
            baseline_median = baseline_data.get("stats", {}).get("median", 0)

            if baseline_median > 0:
                change_pct = ((current_median - baseline_median) / baseline_median) * 100

                if change_pct > 10:  # Regression threshold
                    regressions.append(
                        {
                            "name": bench_name,
                            "baseline_ms": baseline_median,
                            "current_ms": current_median,
                            "change_pct": change_pct,
                        }
                    )
                elif change_pct < -10:  # Improvement threshold
                    improvements.append(
                        {
                            "name": bench_name,
                            "baseline_ms": baseline_median,
                            "current_ms": current_median,
                            "change_pct": change_pct,
                        }
                    )

        return {
            "suite": suite_name,
            "regressions": regressions,
            "improvements": improvements,
        }

    def check_regressions(
        self,
        comparison: Dict[str, Any],
        threshold: float = 10.0,
    ) -> bool:
        """Check for performance regressions.

        Args:
            comparison: Comparison results from compare_with_baseline()
            threshold: Regression threshold in percentage

        Returns:
            True if regressions detected
        """
        has_regressions = False

        for suite_comparison in comparison.get("comparisons", []):
            for regression in suite_comparison.get("regressions", []):
                logger.error(
                    f"REGRESSION: {suite_comparison['suite']}/{regression['name']}: "
                    f"{regression['change_pct']:.1f}% slower "
                    f"({regression['baseline_ms']:.2f}ms -> {regression['current_ms']:.2f}ms)"
                )
                has_regressions = True

        return has_regressions

    def generate_summary(
        self,
        format: str = "markdown",
    ) -> str:
        """Generate a summary of benchmark results.

        Args:
            format: Output format (markdown, json, html)

        Returns:
            Formatted summary string
        """
        if format == "json":
            return json.dumps(self.results, indent=2)

        elif format == "markdown":
            return self._generate_markdown_summary()

        elif format == "html":
            return self._generate_html_summary()

        else:
            logger.error(f"Unknown format: {format}")
            return ""

    def _generate_markdown_summary(self) -> str:
        """Generate markdown summary of benchmark results."""
        lines = [
            "# Victor Workflow System - Benchmark Results",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
        ]

        for suite_name, suite_result in self.results.items():
            if "data" not in suite_result:
                continue

            suite = BENCHMARK_SUITES[suite_name]
            benchmarks = suite_result["data"].get("benchmarks", {})

            lines.append(f"### {suite.name}")
            lines.append("")
            lines.append(f"*{suite.description}*")
            lines.append("")
            lines.append(f"| Benchmark | Median (ms) | Min (ms) | Max (ms) | Iterations |")
            lines.append(f"|-----------|-------------|----------|----------|------------|")

            for bench_name, bench_data in benchmarks.items():
                stats = bench_data.get("stats", {})
                median = stats.get("median", 0)
                min_val = stats.get("min", 0)
                max_val = stats.get("max", 0)
                iterations = stats.get("iterations", 0)

                lines.append(
                    f"| {bench_name} | {median:.2f} | {min_val:.2f} | {max_val:.2f} | {iterations} |"
                )

            lines.append("")

        return "\n".join(lines)

    def _generate_html_summary(self) -> str:
        """Generate HTML summary of benchmark results."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Victor Workflow System - Benchmark Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .timestamp {{ color: #888; font-style: italic; }}
    </style>
</head>
<body>
    <h1>Victor Workflow System - Benchmark Results</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        for suite_name, suite_result in self.results.items():
            if "data" not in suite_result:
                continue

            suite = BENCHMARK_SUITES[suite_name]
            benchmarks = suite_result["data"].get("benchmarks", {})

            html += f"    <h2>{suite.name}</h2>\n"
            html += f"    <p>{suite.description}</p>\n"
            html += "    <table>\n"
            html += "        <tr><th>Benchmark</th><th>Median (ms)</th><th>Min (ms)</th><th>Max (ms)</th><th>Iterations</th></tr>\n"

            for bench_name, bench_data in benchmarks.items():
                stats = bench_data.get("stats", {})
                median = stats.get("median", 0)
                min_val = stats.get("min", 0)
                max_val = stats.get("max", 0)
                iterations = stats.get("iterations", 0)

                html += f"        <tr><td>{bench_name}</td><td>{median:.2f}</td><td>{min_val:.2f}</td><td>{max_val:.2f}</td><td>{iterations}</td></tr>\n"

            html += "    </table>\n"

        html += """
</body>
</html>"""

        return html


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Victor Workflow System - Comprehensive Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  %(prog)s --all

  # Run specific suite
  %(prog)s --suite team_nodes

  # Run with filter
  %(prog)s --suite editor --filter "rendering"

  # Quick smoke test
  %(prog)s --quick

  # Generate report
  %(prog)s --all --report --format markdown

  # Compare with baseline
  %(prog)s --all --compare baseline.json --regression-check
        """,
    )

    parser.add_argument(
        "--suite",
        choices=list(BENCHMARK_SUITES.keys()),
        help="Run specific benchmark suite",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark suites",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick benchmarks (smoke test)",
    )

    parser.add_argument(
        "--filter",
        help="Pytest filter expression",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations per benchmark (default: 1)",
    )

    parser.add_argument(
        "--output-dir",
        default="/tmp/benchmark_results",
        help="Output directory for results (default: /tmp/benchmark_results)",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate summary report after running",
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Report format (default: markdown)",
    )

    parser.add_argument(
        "--compare",
        help="Compare results with baseline file",
    )

    parser.add_argument(
        "--regression-check",
        action="store_true",
        help="Exit with error if performance regression detected",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.suite and not args.all and not args.quick:
        parser.error("Must specify --suite, --all, or --quick")

    # Create runner
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        verbose=args.verbose,
        iterations=args.iterations,
    )

    # Run benchmarks
    start_time = time.time()

    if args.all or args.quick:
        results = runner.run_all(
            filter_expr=args.filter,
            quick=args.quick,
        )
    else:
        results = runner.run_suite(
            suite_name=args.suite,
            filter_expr=args.filter,
            quick=args.quick,
        )

    elapsed = time.time() - start_time
    logger.info(f"\nTotal execution time: {elapsed:.2f} seconds")

    # Compare with baseline if specified
    if args.compare:
        logger.info(f"\nComparing with baseline: {args.compare}")
        comparison = runner.compare_with_baseline(args.compare)

        if args.regression_check:
            has_regressions = runner.check_regressions(comparison)
            if has_regressions:
                logger.error("Performance regressions detected!")
                sys.exit(1)

    # Generate report if requested
    if args.report:
        logger.info(f"\nGenerating {args.format} report...")
        summary = runner.generate_summary(format=args.format)

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(args.output_dir) / f"benchmark_report_{timestamp}.{args.format}"

        with open(report_file, "w") as f:
            f.write(summary)

        logger.info(f"Report saved to: {report_file}")

        # Print to stdout if markdown
        if args.format == "markdown":
            print("\n" + summary)

    logger.info("\nBenchmark run completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
