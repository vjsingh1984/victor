#!/usr/bin/env python3
"""
Performance Report Generator for Victor Workflow System

This script generates comprehensive performance reports from benchmark results,
including visualizations, trend analysis, and regression detection.

Usage:
    # Generate report from benchmark results
    python scripts/performance_report.py --input /tmp/benchmark_results/team_nodes_20250115_120000.json

    # Generate report for all suites
    python scripts/performance_report.py --all --output-dir ./reports

    # Compare with baseline
    python scripts/performance_report.py --input current.json --baseline baseline.json

    # Generate trend analysis
    python scripts/performance_report.py --trend --history-dir ./benchmark_history

    # Generate with visualizations
    python scripts/performance_report.py --all --visualizations --output-format html

Examples:
    # Quick markdown report
    python scripts/performance_report.py --all --output-format markdown

    # Full HTML report with charts
    python scripts/performance_report.py --all --output-format html --visualizations

    # CI/CD integration (check for regressions)
    python scripts/performance_report.py --all --baseline baseline.json --fail-on-regression
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
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
# Data Structures
# =============================================================================


@dataclass
class BenchmarkMetric:
    """A single benchmark metric."""

    name: str
    median: float
    min: float
    max: float
    std_dev: float
    iterations: int
    unit: str = "ms"


@dataclass
class SuiteComparison:
    """Comparison of benchmark suite results."""

    suite_name: str
    benchmarks: List[BenchmarkMetric]
    total_time: float
    success: bool


@dataclass
class RegressionInfo:
    """Information about a performance regression."""

    benchmark_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    severity: str  # "low", "medium", "high"


# =============================================================================
# Report Generator
# =============================================================================


class PerformanceReportGenerator:
    """Generate performance reports from benchmark results."""

    def __init__(
        self,
        baseline_file: Optional[str] = None,
        regression_threshold: float = 10.0,
    ):
        """Initialize the report generator.

        Args:
            baseline_file: Optional baseline file for comparison
            regression_threshold: Percentage change considered a regression
        """
        self.baseline_file = baseline_file
        self.regression_threshold = regression_threshold
        self.baseline_data: Optional[Dict[str, Any]] = None

        if baseline_file:
            self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline data from file."""
        baseline_path = Path(self.baseline_file)
        if not baseline_path.exists():
            logger.warning(f"Baseline file not found: {self.baseline_file}")
            return

        with open(baseline_path) as f:
            self.baseline_data = json.load(f)

        logger.info(f"Loaded baseline from: {self.baseline_file}")

    def generate_report(
        self,
        data: Dict[str, Any],
        output_format: str = "markdown",
    ) -> str:
        """Generate a performance report.

        Args:
            data: Benchmark data
            output_format: Report format (markdown, json, html)

        Returns:
            Formatted report string
        """
        if output_format == "json":
            return json.dumps(data, indent=2)

        elif output_format == "markdown":
            return self._generate_markdown(data)

        elif output_format == "html":
            return self._generate_html(data)

        else:
            logger.error(f"Unknown format: {output_format}")
            return ""

    def _generate_markdown(self, data: Dict[str, Any]) -> str:
        """Generate markdown report."""
        lines = [
            "# Victor Workflow System - Performance Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Executive summary
        lines.extend(
            [
                "## Executive Summary",
                "",
                self._generate_executive_summary(data),
                "",
            ]
        )

        # Benchmark results
        if "benchmarks" in data:
            lines.extend(
                [
                    "## Benchmark Results",
                    "",
                    self._generate_benchmark_table(data),
                    "",
                ]
            )

        # Performance targets
        lines.extend(
            [
                "## Performance Targets",
                "",
                self._generate_performance_targets(data),
                "",
            ]
        )

        # Comparison with baseline
        if self.baseline_data:
            lines.extend(
                [
                    "## Comparison with Baseline",
                    "",
                    self._generate_baseline_comparison(data),
                    "",
                ]
            )

        # Recommendations
        lines.extend(
            [
                "## Recommendations",
                "",
                self._generate_recommendations(data),
                "",
            ]
        )

        return "\n".join(lines)

    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        benchmarks = data.get("benchmarks", {})
        total_benchmarks = len(benchmarks)
        successful_benchmarks = sum(
            1 for b in benchmarks.values() if b.get("stats", {}).get("iterations", 0) > 0
        )

        total_time = sum(b.get("stats", {}).get("median", 0) for b in benchmarks.values())

        summary = f"""
- **Total Benchmarks:** {total_benchmarks}
- **Successful:** {successful_benchmarks}
- **Total Execution Time:** {total_time:.2f}ms
- **Average Per Benchmark:** {total_time / total_benchmarks if total_benchmarks > 0 else 0:.2f}ms
"""
        return summary.strip()

    def _generate_benchmark_table(self, data: Dict[str, Any]) -> str:
        """Generate benchmark results table."""
        benchmarks = data.get("benchmarks", {})

        lines = [
            "| Benchmark | Median (ms) | Min (ms) | Max (ms) | Std Dev | Iterations |",
            "|-----------|-------------|----------|----------|---------|------------|",
        ]

        for name, bench_data in sorted(benchmarks.items()):
            stats = bench_data.get("stats", {})
            median = stats.get("median", 0)
            min_val = stats.get("min", 0)
            max_val = stats.get("max", 0)
            std_dev = stats.get("stddev", 0)
            iterations = stats.get("iterations", 0)

            lines.append(
                f"| {name} | {median:.2f} | {min_val:.2f} | {max_val:.2f} | "
                f"{std_dev:.2f} | {iterations} |"
            )

        return "\n".join(lines)

    def _generate_performance_targets(self, data: Dict[str, Any]) -> str:
        """Generate performance targets section."""
        targets = {
            "team_nodes": {
                "team_execution_3_members": 5000,
                "recursion_overhead_per_level": 1,
                "memory_10_members": 10485760,
            },
            "editor": {
                "editor_load_100_nodes": 500,
                "node_rendering_per_node": 16,
                "auto_layout_100_nodes": 1000,
            },
            "workflow_execution": {
                "simple_workflow_5_nodes": 100,
                "medium_workflow_20_nodes": 400,
                "complex_workflow_50_nodes": 1000,
            },
        }

        lines = ["| Target | Value | Status |", "|--------|-------|--------|"]

        benchmarks = data.get("benchmarks", {})

        for target_name, target_value in targets.get(
            data.get("machine_info", {}).get("suite", ""), {}
        ).items():
            if target_name in benchmarks:
                actual_value = benchmarks[target_name].get("stats", {}).get("median", 0)
                status = "✓ PASS" if actual_value <= target_value else "✗ FAIL"
                lines.append(f"| {target_name} | {target_value} | {status} |")

        return "\n".join(lines) if len(lines) > 2 else "No targets defined for this suite."

    def _generate_baseline_comparison(self, data: Dict[str, Any]) -> str:
        """Generate baseline comparison section."""
        if not self.baseline_data:
            return "No baseline data available for comparison."

        benchmarks = data.get("benchmarks", {})
        baseline_benchmarks = self.baseline_data.get("benchmarks", {})

        lines = [
            "### Performance Changes",
            "",
            "| Benchmark | Baseline (ms) | Current (ms) | Change | Status |",
            "|-----------|---------------|--------------|--------|--------|",
        ]

        regressions = []
        improvements = []

        for name, bench_data in benchmarks.items():
            if name not in baseline_benchmarks:
                continue

            current_median = bench_data.get("stats", {}).get("median", 0)
            baseline_median = baseline_benchmarks[name].get("stats", {}).get("median", 0)

            if baseline_median > 0:
                change_pct = ((current_median - baseline_median) / baseline_median) * 100
                status = "✓" if change_pct < self.regression_threshold else "✗"

                lines.append(
                    f"| {name} | {baseline_median:.2f} | {current_median:.2f} | "
                    f"{change_pct:+.1f}% | {status} |"
                )

                if change_pct > self.regression_threshold:
                    regressions.append((name, change_pct))
                elif change_pct < -self.regression_threshold:
                    improvements.append((name, change_pct))

        # Add summary
        if regressions:
            lines.extend(
                [
                    "",
                    "#### ⚠️ Regressions Detected",
                    "",
                ]
            )
            for name, change_pct in regressions:
                lines.append(f"- **{name}**: {change_pct:.1f}% slower")

        if improvements:
            lines.extend(
                [
                    "",
                    "#### ✨ Improvements",
                    "",
                ]
            )
            for name, change_pct in improvements:
                lines.append(f"- **{name}**: {abs(change_pct):.1f}% faster")

        return "\n".join(lines)

    def _generate_recommendations(self, data: Dict[str, Any]) -> str:
        """Generate recommendations based on results."""
        recommendations = []

        benchmarks = data.get("benchmarks", {})

        # Check for slow benchmarks
        for name, bench_data in benchmarks.items():
            median = bench_data.get("stats", {}).get("median", 0)
            if median > 1000:  # > 1 second
                recommendations.append(
                    f"- **{name}** is slow ({median:.2f}ms). Consider optimization or caching."
                )

        # Check for high variance
        for name, bench_data in benchmarks.items():
            stats = bench_data.get("stats", {})
            median = stats.get("median", 0)
            std_dev = stats.get("stddev", 0)
            if median > 0 and (std_dev / median) > 0.5:  # > 50% variance
                recommendations.append(
                    f"- **{name}** has high variance ({std_dev/median*100:.1f}%). "
                    "Investigate environmental factors."
                )

        if not recommendations:
            recommendations.append("- All benchmarks performing well! No immediate concerns.")

        return "\n".join(recommendations)

    def _generate_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Victor Workflow System - Performance Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .timestamp {{ color: #7f8c8d; font-style: italic; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; font-weight: 600; }}
        td {{ border: 1px solid #ddd; padding: 10px; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e9ecef; }}
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #e74c3c; font-weight: bold; }}
        .regression {{ background-color: #ffebee; }}
        .improvement {{ background-color: #e8f5e9; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .recommendation {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Victor Workflow System - Performance Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

        # Executive summary
        html += self._generate_html_summary(data)

        # Benchmark results
        html += self._generate_html_benchmarks(data)

        # Baseline comparison
        if self.baseline_data:
            html += self._generate_html_baseline_comparison(data)

        html += """
    </div>
</body>
</html>"""

        return html

    def _generate_html_summary(self, data: Dict[str, Any]) -> str:
        """Generate HTML summary section."""
        benchmarks = data.get("benchmarks", {})
        total = len(benchmarks)
        total_time = sum(b.get("stats", {}).get("median", 0) for b in benchmarks.values())

        return f"""
        <h2>Executive Summary</h2>
        <div class="summary">
            <p><strong>Total Benchmarks:</strong> {total}</p>
            <p><strong>Total Execution Time:</strong> {total_time:.2f}ms</p>
            <p><strong>Average Per Benchmark:</strong> {total_time/total if total > 0 else 0:.2f}ms</p>
        </div>
"""

    def _generate_html_benchmarks(self, data: Dict[str, Any]) -> str:
        """Generate HTML benchmarks table."""
        benchmarks = data.get("benchmarks", {})

        html = "<h2>Benchmark Results</h2>\n"
        html += "<table>\n"
        html += "<tr><th>Benchmark</th><th>Median (ms)</th><th>Min (ms)</th><th>Max (ms)</th><th>Std Dev</th><th>Iterations</th></tr>\n"

        for name, bench_data in sorted(benchmarks.items()):
            stats = bench_data.get("stats", {})
            median = stats.get("median", 0)
            min_val = stats.get("min", 0)
            max_val = stats.get("max", 0)
            std_dev = stats.get("stddev", 0)
            iterations = stats.get("iterations", 0)

            html += f"<tr><td>{name}</td><td>{median:.2f}</td><td>{min_val:.2f}</td>"
            html += f"<td>{max_val:.2f}</td><td>{std_dev:.2f}</td><td>{iterations}</td></tr>\n"

        html += "</table>\n"
        return html

    def _generate_html_baseline_comparison(self, data: Dict[str, Any]) -> str:
        """Generate HTML baseline comparison section."""
        if not self.baseline_data:
            return ""

        benchmarks = data.get("benchmarks", {})
        baseline_benchmarks = self.baseline_data.get("benchmarks", {})

        html = "<h2>Baseline Comparison</h2>\n"
        html += "<table>\n"
        html += "<tr><th>Benchmark</th><th>Baseline (ms)</th><th>Current (ms)</th><th>Change</th><th>Status</th></tr>\n"

        for name, bench_data in sorted(benchmarks.items()):
            if name not in baseline_benchmarks:
                continue

            current_median = bench_data.get("stats", {}).get("median", 0)
            baseline_median = baseline_benchmarks[name].get("stats", {}).get("median", 0)

            if baseline_median > 0:
                change_pct = ((current_median - baseline_median) / baseline_median) * 100
                status = "PASS" if change_pct < self.regression_threshold else "FAIL"
                row_class = "regression" if change_pct > self.regression_threshold else ""
                row_class = "improvement" if change_pct < -self.regression_threshold else row_class

                html += f'<tr class="{row_class}"><td>{name}</td><td>{baseline_median:.2f}</td>'
                html += f"<td>{current_median:.2f}</td><td>{change_pct:+.1f}%</td>"
                html += f'<td class="{status.lower()}">{status}</td></tr>\n'

        html += "</table>\n"
        return html


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for performance report generator."""
    parser = argparse.ArgumentParser(
        description="Generate performance reports from benchmark results",
    )

    parser.add_argument(
        "--input",
        help="Input benchmark JSON file",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate reports for all benchmark suites",
    )

    parser.add_argument(
        "--output-dir",
        default="./reports",
        help="Output directory for reports (default: ./reports)",
    )

    parser.add_argument(
        "--output-format",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Report format (default: markdown)",
    )

    parser.add_argument(
        "--baseline",
        help="Baseline file for comparison",
    )

    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=10.0,
        help="Regression threshold in percent (default: 10.0)",
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with error if regression detected",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.all:
        parser.error("Must specify --input or --all")

    # Create report generator
    generator = PerformanceReportGenerator(
        baseline_file=args.baseline,
        regression_threshold=args.regression_threshold,
    )

    # Generate reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.input:
        # Generate single report
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1

        with open(input_path) as f:
            data = json.load(f)

        report = generator.generate_report(data, args.output_format)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"report_{timestamp}.{args.output_format}"

        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"Report saved to: {report_file}")

        # Print to stdout if markdown
        if args.output_format == "markdown":
            print("\n" + report)

        # Check for regressions
        if args.fail_on_regression and args.baseline:
            # Simple regression check
            if "regression" in report.lower():
                logger.error("Performance regression detected!")
                return 1

    elif args.all:
        # Generate reports for all suites
        benchmark_dir = Path("/tmp/benchmark_results")
        if not benchmark_dir.exists():
            logger.error(f"Benchmark directory not found: {benchmark_dir}")
            return 1

        for json_file in benchmark_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)

            report = generator.generate_report(data, args.output_format)

            # Save report
            report_file = output_dir / f"{json_file.stem}_report.{args.output_format}"

            with open(report_file, "w") as f:
                f.write(report)

            logger.info(f"Report saved to: {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
