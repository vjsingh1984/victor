#!/usr/bin/env python3
"""Team Node Performance Benchmark Runner.

This script runs comprehensive benchmarks for team node execution and generates
formatted reports with performance characteristics, scalability analysis, and
optimization recommendations.

Usage:
    python scripts/benchmark_team_nodes.py run --all
    python scripts/benchmark_team_nodes.py run --group formations
    python scripts/benchmark_team_nodes.py report --format markdown
    python scripts/benchmark_team_nodes.py compare run1.json run2.json

Benchmark Groups:
    formations: Compare all 5 formation types
    size: Team size scaling (2, 5, 10 members)
    budget: Tool budget impact (5, 25, 50, 100)
    recursion: Recursion depth overhead
    timeout: Timeout handling performance
    memory: Memory usage profiling
    consensus: Consensus formation variants
    scenarios: Real-world task scenarios
    all: Run all benchmarks

Report Formats:
    markdown: Human-readable markdown tables
    json: Machine-readable JSON
    console: Simple console output
    csv: CSV format for data analysis
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BenchmarkMetric:
    """A single benchmark metric.

    Attributes:
        name: Metric name
        value: Metric value
        unit: Unit of measurement
        description: Optional description
    """

    name: str
    value: float
    unit: str
    description: Optional[str] = None


@dataclass
class BenchmarkRun:
    """Results from a benchmark run.

    Attributes:
        name: Benchmark name
        metrics: List of metrics collected
        timestamp: When the benchmark was run
        metadata: Additional metadata
    """

    name: str
    metrics: List[BenchmarkMetric] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, name: str, value: float, unit: str, description: Optional[str] = None):
        """Add a metric to this run."""
        self.metrics.append(BenchmarkMetric(name, value, unit, description))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "description": m.description,
                }
                for m in self.metrics
            ],
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report.

    Attributes:
        runs: List of benchmark runs
        start_time: Report generation start time
        end_time: Report generation end time
        metadata: Report metadata
    """

    runs: List[BenchmarkRun] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_run(self, run: BenchmarkRun):
        """Add a benchmark run."""
        self.runs.append(run)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "runs": [r.to_dict() for r in self.runs],
        }


# =============================================================================
# Report Formatters
# =============================================================================


class ReportFormatter:
    """Base class for report formatters."""

    def format(self, report: BenchmarkReport) -> str:
        """Format the report.

        Args:
            report: BenchmarkReport to format

        Returns:
            Formatted report string
        """
        raise NotImplementedError


class MarkdownFormatter(ReportFormatter):
    """Format benchmark report as Markdown."""

    def format(self, report: BenchmarkReport) -> str:
        """Format report as Markdown."""
        lines = [
            "# Team Node Performance Benchmark Report",
            "",
            f"**Generated:** {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            f"This report contains {len(report.runs)} benchmark runs measuring the",
            f"performance of team node execution across different formations,",
            f"team sizes, and execution scenarios.",
            "",
            "## Performance Summary",
            "",
            "| Benchmark | Avg Latency (ms) | Min (ms) | Max (ms) | Throughput (teams/s) |",
            "|-----------|------------------|----------|----------|----------------------|",
        ]

        for run in report.runs:
            avg = self._get_metric(run, "avg_latency")
            min_val = self._get_metric(run, "min_latency")
            max_val = self._get_metric(run, "max_latency")
            throughput = self._get_metric(run, "throughput")

            avg_str = f"{avg.value:.2f}" if avg else "N/A"
            min_str = f"{min_val.value:.2f}" if min_val else "N/A"
            max_str = f"{max_val.value:.2f}" if max_val else "N/A"
            tp_str = f"{throughput.value:.2f}" if throughput else "N/A"

            lines.append(f"| {run.name} | {avg_str} | {min_str} | {max_str} | {tp_str} |")

        # Group by category
        categories = self._group_by_category(report)

        if "formations" in categories:
            lines.extend(self._format_formation_comparison(categories["formations"]))

        if "scaling" in categories:
            lines.extend(self._format_scaling_analysis(categories["scaling"]))

        if "scenarios" in categories:
            lines.extend(self._format_scenario_analysis(categories["scenarios"]))

        # Add recommendations
        lines.extend(self._format_recommendations(report))

        # Add detailed metrics
        lines.append("")
        lines.append("## Detailed Metrics")
        lines.append("")

        for run in report.runs:
            lines.append(f"### {run.name}")
            lines.append("")
            lines.append("| Metric | Value | Unit | Description |")
            lines.append("|--------|-------|------|-------------|")

            for m in run.metrics:
                desc = m.description or ""
                lines.append(f"| {m.name} | {m.value:.4f} | {m.unit} | {desc} |")
            lines.append("")

        return "\n".join(lines)

    def _get_metric(self, run: BenchmarkRun, name: str) -> Optional[BenchmarkMetric]:
        """Get a metric by name."""
        for m in run.metrics:
            if m.name == name:
                return m
        return None

    def _group_by_category(self, report: BenchmarkReport) -> Dict[str, List[BenchmarkRun]]:
        """Group runs by category."""
        categories: Dict[str, List[BenchmarkRun]] = {}

        for run in report.runs:
            if "formation" in run.name.lower():
                categories.setdefault("formations", []).append(run)
            elif any(keyword in run.name.lower() for keyword in ["size", "scaling", "team_size"]):
                categories.setdefault("scaling", []).append(run)
            elif "scenario" in run.name.lower():
                categories.setdefault("scenarios", []).append(run)

        return categories

    def _format_formation_comparison(self, runs: List[BenchmarkRun]) -> List[str]:
        """Format formation comparison section."""
        lines = [
            "",
            "## Formation Comparison",
            "",
            "Performance characteristics of each team formation:",
            "",
            "| Formation | Avg Latency (ms) | Relative Speed | Best For |",
            "|-----------|------------------|----------------|----------|",
        ]

        # Find baseline (sequential)
        baseline = next((r for r in runs if "sequential" in r.name.lower()), None)

        for run in runs:
            avg = self._get_metric(run, "avg_latency")
            if not avg:
                continue

            formation_name = run.name.replace("_", " ").title()
            speedup = ""
            best_for = ""

            if "sequential" in run.name.lower():
                best_for = "Ordered tasks, context chaining"
                speedup = "1.00x (baseline)"
            elif "parallel" in run.name.lower():
                best_for = "Independent tasks, speed"
                if baseline:
                    baseline_avg = self._get_metric(baseline, "avg_latency")
                    if baseline_avg:
                        speedup = f"{baseline_avg.value / avg.value:.2f}x"
            elif "pipeline" in run.name.lower():
                best_for = "Staged processing, refinement"
                if baseline:
                    baseline_avg = self._get_metric(baseline, "avg_latency")
                    if baseline_avg:
                        speedup = f"{baseline_avg.value / avg.value:.2f}x"
            elif "hierarchical" in run.name.lower():
                best_for = "Manager-worker, delegation"
                if baseline:
                    baseline_avg = self._get_metric(baseline, "avg_latency")
                    if baseline_avg:
                        speedup = f"{baseline_avg.value / avg.value:.2f}x"
            elif "consensus" in run.name.lower():
                best_for = "Agreement required, quality"
                if baseline:
                    baseline_avg = self._get_metric(baseline, "avg_latency")
                    if baseline_avg:
                        speedup = f"{baseline_avg.value / avg.value:.2f}x"

            lines.append(f"| {formation_name} | {avg.value:.2f} | {speedup} | {best_for} |")

        return lines

    def _format_scaling_analysis(self, runs: List[BenchmarkRun]) -> List[str]:
        """Format scaling analysis section."""
        lines = [
            "",
            "## Team Size Scaling",
            "",
            "Performance characteristics as team size grows:",
            "",
            "| Team Size | Avg Latency (ms) | Per-Member Overhead | Scaling Factor |",
            "|-----------|------------------|---------------------|-----------------|",
        ]

        # Sort by team size
        sorted_runs = sorted(runs, key=lambda r: self._extract_team_size(r.name))

        for run in sorted_runs:
            avg = self._get_metric(run, "avg_latency")
            size = self._extract_team_size(run.name)

            if avg and size:
                overhead = avg.value / size
                baseline = sorted_runs[0]
                baseline_avg = self._get_metric(baseline, "avg_latency")
                scaling_factor = avg.value / baseline_avg.value if baseline_avg else 1.0

                lines.append(
                    f"| {size} | {avg.value:.2f} | {overhead:.2f} | {scaling_factor:.2f}x |"
                )

        return lines

    def _format_scenario_analysis(self, runs: List[BenchmarkRun]) -> List[str]:
        """Format scenario analysis section."""
        lines = [
            "",
            "## Real-World Scenario Performance",
            "",
            "Performance on representative task scenarios:",
            "",
            "| Scenario | Avg Latency (ms) | Complexity | Recommended Formation |",
            "|----------|------------------|------------|----------------------|",
        ]

        scenario_recommendations = {
            "simple": "Sequential (2 members)",
            "complex": "Pipeline (4 members)",
            "large_context": "Parallel (2+ members)",
        }

        for run in runs:
            avg = self._get_metric(run, "avg_latency")
            if not avg:
                continue

            scenario = run.name.replace("_", " ").title()
            complexity = "Low" if "simple" in run.name.lower() else "High"

            # Find matching recommendation
            recommendation = "Sequential"
            for key, rec in scenario_recommendations.items():
                if key in run.name.lower():
                    recommendation = rec
                    break

            lines.append(f"| {scenario} | {avg.value:.2f} | {complexity} | {recommendation} |")

        return lines

    def _format_recommendations(self, report: BenchmarkReport) -> List[str]:
        """Format optimization recommendations."""
        lines = [
            "",
            "## Performance Recommendations",
            "",
            "### Formation Selection",
            "",
            "**Use Sequential when:**",
            "- Tasks depend on previous results",
            "- Context chaining is important",
            "- Team size is small (2-3 members)",
            "",
            "**Use Parallel when:**",
            "- Tasks are independent",
            "- Speed is critical",
            "- Members have similar workloads",
            "",
            "**Use Pipeline when:**",
            "- Work flows through stages",
            "- Each stage refines previous output",
            "- Quality increases through stages",
            "",
            "**Use Hierarchical when:**",
            "- Clear manager-worker relationship",
            "- Delegation is natural",
            "- Manager needs to synthesize results",
            "",
            "**Use Consensus when:**",
            "- Agreement is critical",
            "- Quality > speed",
            "- Small team (3-5 members)",
            "",
            "### Team Sizing",
            "",
            "- **2-3 members**: Best for simple tasks, low overhead",
            "- **4-6 members**: Balanced for medium complexity",
            "- **7-10 members**: Use for complex tasks, monitor overhead",
            "",
            "### Tool Budgets",
            "",
            "- **5-10**: Quick tasks, exploration",
            "- **15-25**: Standard development tasks",
            "- **30-50**: Complex features, comprehensive testing",
            "- **50+**: Large refactoring, architecture work",
            "",
        ]

        return lines

    def _extract_team_size(self, name: str) -> Optional[int]:
        """Extract team size from benchmark name."""
        import re

        match = re.search(r"size[_\s]*(\d+)", name.lower())
        if match:
            return int(match.group(1))
        return None


class JsonFormatter(ReportFormatter):
    """Format benchmark report as JSON."""

    def format(self, report: BenchmarkReport) -> str:
        """Format report as JSON."""
        report.end_time = datetime.now()
        return json.dumps(report.to_dict(), indent=2)


class ConsoleFormatter(ReportFormatter):
    """Format benchmark report for console output."""

    def format(self, report: BenchmarkReport) -> str:
        """Format report for console."""
        lines = [
            "=" * 80,
            "TEAM NODE PERFORMANCE BENCHMARK RESULTS",
            "=" * 80,
            "",
        ]

        for run in report.runs:
            lines.append(f"\n{run.name}")
            lines.append("-" * 40)

            for m in run.metrics:
                desc = f" - {m.description}" if m.description else ""
                lines.append(f"  {m.name}: {m.value:.4f} {m.unit}{desc}")

        return "\n".join(lines)


class CsvFormatter(ReportFormatter):
    """Format benchmark report as CSV."""

    def format(self, report: BenchmarkReport) -> str:
        """Format report as CSV."""
        lines = ["benchmark,metric,value,unit,description"]

        for run in report.runs:
            for m in run.metrics:
                desc = m.description or ""
                lines.append(f'"{run.name}","{m.name}",{m.value},"{m.unit}","{desc}"')

        return "\n".join(lines)


# =============================================================================
# Benchmark Runner
# =============================================================================


class BenchmarkRunner:
    """Run team node benchmarks."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize benchmark runner.

        Args:
            project_root: Project root directory (defaults to script dir parent)
        """
        if project_root is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent

        self.project_root = project_root
        self.benchmark_file = (
            project_root / "tests" / "performance" / "test_team_node_performance.py"
        )

    def run(
        self,
        group: Optional[str] = None,
        verbose: bool = False,
        iterations: int = 100,
        save: bool = True,
    ) -> BenchmarkReport:
        """Run benchmarks.

        Args:
            group: Benchmark group to run
            verbose: Enable verbose output
            iterations: Number of iterations (for pytest-benchmark)
            save: Save results to file

        Returns:
            BenchmarkReport with results
        """
        report = BenchmarkReport()
        report.metadata["iterations"] = iterations
        report.metadata["group"] = group or "all"

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.benchmark_file),
            "-v" if verbose else "",
            "-s" if verbose else "",
            "-m",
            "benchmark or summary",
            "--tb=short",
        ]

        # Add group filter
        if group and group != "all":
            cmd.extend(["-k", group])

        cmd = [c for c in cmd if c]

        print(f"\n{'=' * 80}")
        print("Running Team Node Performance Benchmarks")
        print(f"{'=' * 80}\n")
        print(f"Group: {group or 'all'}")
        print(f"Iterations: {iterations}")
        print(f"Working directory: {self.project_root}")
        print()

        # Run benchmarks
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=not verbose,
            text=True,
        )

        if result.returncode != 0:
            print("Benchmark execution failed!")
            if not verbose:
                print(result.stdout)
                print(result.stderr)
            return report

        # Parse output to extract metrics
        if result.stdout:
            parsed_runs = self._parse_output(result.stdout)
            for run in parsed_runs:
                report.add_run(run)

        report.end_time = datetime.now()

        # Save results if requested
        if save and report.runs:
            self._save_results(report)

        return report

    def _parse_output(self, output: str) -> List[BenchmarkRun]:
        """Parse benchmark output to extract metrics.

        Args:
            output: Pytest output string

        Returns:
            List of BenchmarkRun objects
        """
        runs = []

        # Look for result patterns from pytest-benchmark
        import re

        # Pattern: "test_name: 12.34ms"
        patterns = [
            (
                r"test_formation_performance.*?\[([^\]]+)\].*?:\s*([\d.]+)ms",
                "Formation Performance",
            ),
            (r"test_team_size_scaling.*?\[(\d+)\].*?:\s*([\d.]+)ms", "Team Size Scaling"),
            (r"test_tool_budget_impact.*?\[(\d+)\].*?:\s*([\d.]+)ms", "Tool Budget Impact"),
            (r"test_recursion_depth_overhead.*?\[(\d+)\].*?:\s*([\d.]+)ms", "Recursion Depth"),
            (r"test_timeout_performance.*?\[(\d+)\].*?:\s*([\d.]+)ms", "Timeout Handling"),
            (r"test_memory_per_member.*?\[(\d+)\].*?:\s*([\d.]+)ms", "Memory Per Member"),
            (
                r"test_consensus_formation_performance.*?\[(\d+),\s*(\d+)\].*?:\s*([\d.]+)ms",
                "Consensus Formation",
            ),
            (r"test_simple_task_scenario.*?:\s*([\d.]+)ms", "Simple Task Scenario"),
            (r"test_complex_task_scenario.*?:\s*([\d.]+)ms", "Complex Task Scenario"),
            (r"test_large_context_scenario.*?:\s*([\d.]+)ms", "Large Context Scenario"),
        ]

        for pattern, name_base in patterns:
            matches = re.findall(pattern, output, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        param, value = match
                        run_name = f"{name_base}_{param}"
                    elif len(match) == 3:
                        param1, param2, value = match
                        run_name = f"{name_base}_{param1}members_{param2}rounds"
                    else:
                        continue
                else:
                    value = match
                    run_name = name_base

                run = BenchmarkRun(name=run_name)
                run.add_metric("avg_latency", float(value), "ms", "Average execution latency")
                runs.append(run)

        return runs

    def _save_results(self, report: BenchmarkReport) -> None:
        """Save benchmark results to file.

        Args:
            report: BenchmarkReport to save
        """
        results_dir = self.project_root / ".benchmark_results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = results_dir / f"team_nodes_{timestamp}.json"

        with open(json_path, "w") as f:
            f.write(json.dumps(report.to_dict(), indent=2))

        print(f"\nResults saved to: {json_path}")


# =============================================================================
# Report Generator
# =============================================================================


def generate_report(
    format: str = "markdown",
    input_file: Optional[Path] = None,
) -> str:
    """Generate benchmark report.

    Args:
        format: Output format (markdown, json, console, csv)
        input_file: Optional input JSON file with results

    Returns:
        Formatted report string
    """
    # Load results from file if provided
    report = BenchmarkReport()

    if input_file and input_file.exists():
        with open(input_file) as f:
            data = json.load(f)

        for run_data in data.get("runs", []):
            run = BenchmarkRun(
                name=run_data["name"],
                timestamp=datetime.fromisoformat(run_data["timestamp"]),
            )

            for m in run_data.get("metrics", []):
                run.add_metric(
                    name=m["name"],
                    value=m["value"],
                    unit=m["unit"],
                    description=m.get("description"),
                )

            report.add_run(run)

    # Select formatter
    formatters = {
        "markdown": MarkdownFormatter(),
        "json": JsonFormatter(),
        "console": ConsoleFormatter(),
        "csv": CsvFormatter(),
    }

    formatter = formatters.get(format.lower(), MarkdownFormatter())
    return formatter.format(report)


# =============================================================================
# Comparison Tool
# =============================================================================


def compare_runs(file1: Path, file2: Optional[Path] = None) -> str:
    """Compare two benchmark runs.

    Args:
        file1: First benchmark file (or directory to find latest)
        file2: Second benchmark file (optional)

    Returns:
        Comparison report
    """
    results_dir = Path(".benchmark_results")

    # Find files
    if file1.is_dir():
        files = sorted(
            results_dir.glob("team_nodes_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if len(files) < 2:
            return "Need at least 2 benchmark runs to compare"
        file1 = files[1]  # Second most recent
        file2 = files[0]  # Most recent
    elif file2 is None:
        # Use the most recent file as comparison
        files = sorted(
            results_dir.glob("team_nodes_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if files and files[0] != file1:
            file2 = files[0]

    if not file2 or not file2.exists():
        return f"Cannot find comparison file for {file1}"

    # Load both runs
    with open(file1) as f:
        run1_data = json.load(f)
    with open(file2) as f:
        run2_data = json.load(f)

    # Generate comparison report
    lines = [
        "# Team Node Benchmark Comparison Report",
        "",
        f"**Base:** {file1.name}",
        f"**Current:** {file2.name}",
        "",
        "## Performance Change",
        "",
        "| Benchmark | Base (ms) | Current (ms) | Change | Speedup |",
        "|-----------|-----------|--------------|--------|---------|",
    ]

    # Extract metrics from both runs
    metrics1 = {m["name"]: m for run in run1_data.get("runs", []) for m in run.get("metrics", [])}
    metrics2 = {m["name"]: m for run in run2_data.get("runs", []) for m in run.get("metrics", [])}

    for name, m1 in metrics1.items():
        if name in metrics2:
            m2 = metrics2[name]
            if m1["unit"] == "ms" and "latency" in name.lower():
                change = m2["value"] - m1["value"]
                pct_change = (change / m1["value"]) * 100 if m1["value"] > 0 else 0
                speedup = m1["value"] / m2["value"] if m2["value"] > 0 else 1.0

                change_str = f"{change:+.2f} ({pct_change:+.1f}%)"
                speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower"

                lines.append(
                    f"| {name} | {m1['value']:.2f} | {m2['value']:.2f} | {change_str} | {speedup_str} |"
                )

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Team Node Performance Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  %(prog)s run --all

  # Run specific benchmark group
  %(prog)s run --group formations

  # Generate markdown report
  %(prog)s report --format markdown

  # Compare two benchmark runs
  %(prog)s compare .benchmark_results/team_nodes_20240101_120000.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--group",
        choices=[
            "formations",
            "size",
            "budget",
            "recursion",
            "timeout",
            "memory",
            "consensus",
            "scenarios",
            "all",
        ],
        default="all",
        help="Benchmark group to run",
    )
    run_parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    run_parser.add_argument("--no-save", action="store_true", help="Don't save results")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument(
        "--format",
        choices=["markdown", "json", "console", "csv"],
        default="markdown",
        help="Output format",
    )
    report_parser.add_argument(
        "--input", type=Path, help="Input JSON file (uses latest if not specified)"
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark runs")
    compare_parser.add_argument("file1", type=Path, help="First benchmark file or directory")
    compare_parser.add_argument("file2", type=Path, nargs="?", help="Second benchmark file")

    # List command
    subparsers.add_parser("list", help="List saved benchmark results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    runner = BenchmarkRunner()

    if args.command == "run":
        report = runner.run(
            group=args.group,
            verbose=args.verbose,
            iterations=args.iterations,
            save=not args.no_save,
        )

        # Print summary
        formatter = ConsoleFormatter()
        print("\n")
        print(formatter.format(report))

    elif args.command == "report":
        report_text = generate_report(format=args.format, input_file=args.input)
        print(report_text)

    elif args.command == "compare":
        comparison = compare_runs(args.file1, args.file2)
        print(comparison)

    elif args.command == "list":
        results_dir = Path(".benchmark_results")
        if not results_dir.exists():
            print("No benchmark results found")
            return 1

        files = sorted(
            results_dir.glob("team_nodes_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        print(f"\nFound {len(files)} benchmark result(s):\n")
        for i, f in enumerate(files[:10], 1):
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            size = f.stat().st_size / 1024
            print(f"  {i:2d}. {f.name:50s} ({size:6.1f} KB, {mtime.strftime('%Y-%m-%d %H:%M')})")

        if len(files) > 10:
            print(f"\n  ... and {len(files) - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
