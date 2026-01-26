#!/usr/bin/env python3
"""Tool Selection Cache Benchmark Runner.

This script runs comprehensive benchmarks for the tool selection caching system
and generates formatted reports.

Usage:
    python scripts/benchmark_tool_selection.py run --all
    python scripts/benchmark_tool_selection.py run --group cold
    python scripts/benchmark_tool_selection.py report --format markdown
    python scripts/benchmark_tool_selection.py compare run1.json run2.json
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
            "# Tool Selection Cache Benchmark Report",
            "",
            f"**Generated:** {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            f"This report contains {len(report.runs)} benchmark runs measuring the",
            f"performance of the tool selection caching system.",
            "",
            "## Performance Summary",
            "",
            "| Benchmark | Avg Latency (ms) | P95 Latency (ms) | Hit Rate | Throughput (ops/s) |",
            "|-----------|------------------|------------------|----------|-------------------|",
        ]

        for run in report.runs:
            avg = self._get_metric(run, "avg_latency")
            p95 = self._get_metric(run, "p95_latency")
            hit_rate = self._get_metric(run, "hit_rate")
            throughput = self._get_metric(run, "throughput")

            avg_val = f"{avg.value:.2f}" if avg else "N/A"
            p95_val = f"{p95.value:.2f}" if p95 else "N/A"
            hit_val = f"{hit_rate.value:.1%}" if hit_rate else "N/A"
            tp_val = f"{throughput.value:.0f}" if throughput else "N/A"

            lines.append(f"| {run.name} | {avg_val} | {p95_val} | {hit_val} | {tp_val} |")

        # Add speedup comparison
        if len(report.runs) >= 2:
            baseline = report.runs[0]
            baseline_avg = self._get_metric(baseline, "avg_latency")

            if baseline_avg:
                lines.append("")
                lines.append("## Speedup Comparison")
                lines.append("")
                lines.append("| Benchmark | Speedup vs Baseline | Latency Reduction |")
                lines.append("|-----------|---------------------|-------------------|")

                for run in report.runs[1:]:
                    avg = self._get_metric(run, "avg_latency")
                    if avg and baseline_avg:
                        speedup = baseline_avg.value / avg.value
                        reduction = (1 - 1 / speedup) * 100
                        lines.append(f"| {run.name} | {speedup:.2f}x | {reduction:.1f}% |")

        # Add detailed metrics
        lines.append("")
        lines.append("## Detailed Metrics")
        lines.append("")

        for run in report.runs:
            lines.append(f"### {run.name}")
            lines.append("")
            lines.append("| Metric | Value | Unit |")
            lines.append("|--------|-------|------|")

            for m in run.metrics:
                desc = f" ({m.description})" if m.description else ""
                lines.append(f"| {m.name}{desc} | {m.value:.4f} | {m.unit} |")
            lines.append("")

        return "\n".join(lines)

    def _get_metric(self, run: BenchmarkRun, name: str) -> Optional[BenchmarkMetric]:
        """Get a metric by name."""
        for m in run.metrics:
            if m.name == name:
                return m
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
            "TOOL SELECTION CACHE BENCHMARK RESULTS",
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
    """Run tool selection cache benchmarks."""

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
            project_root / "tests" / "benchmarks" / "test_tool_selection_benchmark.py"
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
            group: Benchmark group to run (cold, warm, mixed, context, rl, all)
            verbose: Enable verbose output
            iterations: Number of iterations
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
        print("Running Tool Selection Cache Benchmarks")
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

        # Look for result patterns like:
        # "Cold cache: 12.34ms"
        # "Warm cache: 2.34ms"
        import re

        patterns = [
            (r"Cold[ _]cache.*?:\s*([\d.]+)ms", "Cold Cache"),
            (r"Warm[ _]cache.*?:\s*([\d.]+)ms", "Warm Cache"),
            (r"Mixed[ _]cache.*?:\s*([\d.]+)ms", "Mixed Cache"),
            (r"Context[ _]cache.*?:\s*([\d.]+)ms", "Context Cache"),
            (r"RL[ _]cache.*?:\s*([\d.]+)ms", "RL Cache"),
        ]

        for pattern, name in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                run = BenchmarkRun(name=name)
                run.add_metric("avg_latency", float(matches[0]), "ms", "Average latency")
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
        json_path = results_dir / f"tool_selection_cache_{timestamp}.json"

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
        files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(files) < 2:
            return "Need at least 2 benchmark runs to compare"
        file1 = files[1]  # Second most recent
        file2 = files[0]  # Most recent
    elif file2 is None:
        # Use the most recent file as comparison
        files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
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
        "# Benchmark Comparison Report",
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
                pct_change = (change / m1["value"]) * 100
                speedup = m1["value"] / m2["value"]

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
        description="Tool Selection Cache Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  %(prog)s run --all

  # Run specific benchmark group
  %(prog)s run --group cold

  # Generate markdown report
  %(prog)s report --format markdown

  # Compare two benchmark runs
  %(prog)s compare .benchmark_results/tool_selection_cache_20240101_120000.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--group",
        choices=[
            "cold",
            "warm",
            "mixed",
            "context",
            "rl",
            "size",
            "ttl",
            "memory",
            "concurrent",
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
            results_dir.glob("tool_selection_cache_*.json"),
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
