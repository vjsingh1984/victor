#!/usr/bin/env python3
"""Performance Benchmark Comparison Tool for Victor AI.

This tool compares benchmark results between two runs and generates
diff reports highlighting regressions and improvements.

Usage:
    python scripts/compare_benchmarks.py run1.json run2.json

    # Or compare with latest
    python scripts/compare_benchmarks.py baseline.json

    # Generate HTML report
    python scripts/compare_benchmarks.py run1.json run2.json --format html

    # Save comparison
    python scripts/compare_benchmarks.py run1.json run2.json --output comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MetricComparison:
    """Comparison of a single metric between two runs.

    Attributes:
        name: Metric name
        baseline_value: Value from baseline run
        current_value: Value from current run
        unit: Unit of measurement
        change_pct: Percentage change
        is_regression: Whether this indicates a regression
        is_improvement: Whether this indicates an improvement
        is_significant: Whether the change is statistically significant
    """

    name: str
    baseline_value: float
    current_value: float
    unit: str
    change_pct: float
    is_regression: bool
    is_improvement: bool
    is_significant: bool = False

    @property
    def status_emoji(self) -> str:
        """Get status emoji."""
        if self.is_regression:
            return "🔴"
        if self.is_improvement:
            return "🟢"
        return "⚪"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "unit": self.unit,
            "change_pct": self.change_pct,
            "is_regression": self.is_regression,
            "is_improvement": self.is_improvement,
            "is_significant": self.is_significant,
        }


@dataclass
class BenchmarkComparison:
    """Comparison of a benchmark between two runs.

    Attributes:
        name: Benchmark name
        category: Benchmark category
        metrics: List of metric comparisons
        overall_status: Overall status (regression, improvement, neutral)
    """

    name: str
    category: str
    metrics: List[MetricComparison] = field(default_factory=list)
    overall_status: str = "neutral"

    def add_metric(self, comparison: MetricComparison):
        """Add a metric comparison."""
        self.metrics.append(comparison)
        self._update_status()

    def _update_status(self):
        """Update overall status based on metrics."""
        has_regression = any(m.is_regression for m in self.metrics)
        has_improvement = any(m.is_improvement for m in self.metrics)

        if has_regression:
            self.overall_status = "regression"
        elif has_improvement:
            self.overall_status = "improvement"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "overall_status": self.overall_status,
            "metrics": [m.to_dict() for m in self.metrics],
        }


@dataclass
class ComparisonReport:
    """Complete comparison report.

    Attributes:
        baseline_file: Baseline file path
        current_file: Current file path
        baseline_time: Baseline timestamp
        current_time: Current timestamp
        comparisons: List of benchmark comparisons
        summary: Summary statistics
    """

    baseline_file: str
    current_file: str
    baseline_time: Optional[datetime] = None
    current_time: Optional[datetime] = None
    comparisons: List[BenchmarkComparison] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_regressions(self) -> List[BenchmarkComparison]:
        """Get list of regressions."""
        return [c for c in self.comparisons if c.overall_status == "regression"]

    def get_improvements(self) -> List[BenchmarkComparison]:
        """Get list of improvements."""
        return [c for c in self.comparisons if c.overall_status == "improvement"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_file": self.baseline_file,
            "current_file": self.current_file,
            "baseline_time": self.baseline_time.isoformat() if self.baseline_time else None,
            "current_time": self.current_time.isoformat() if self.current_time else None,
            "summary": self.summary,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


# =============================================================================
# Comparison Logic
# =============================================================================


def determine_if_regression(
    metric_name: str,
    baseline_value: float,
    current_value: float,
    threshold_pct: float = 10.0,
) -> bool:
    """Determine if a metric change indicates a regression.

    For latency/memory metrics: higher is worse
    For throughput metrics: lower is worse
    For hit rate/success metrics: lower is worse

    Args:
        metric_name: Name of the metric
        baseline_value: Baseline value
        current_value: Current value
        threshold_pct: Percentage threshold for significance

    Returns:
        True if regression
    """
    metric_name_lower = metric_name.lower()

    # Latency and memory metrics: higher is worse
    if any(
        word in metric_name_lower
        for word in ["latency", "time", "memory", "bytes", "ms", "mb", "startup", "bootstrap"]
    ):
        change_pct = ((current_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0
        return change_pct > threshold_pct

    # Throughput and rate metrics: lower is worse
    if any(
        word in metric_name_lower
        for word in ["throughput", "ops", "rate", "speed", "requests", "operations"]
    ):
        change_pct = ((baseline_value - current_value) / baseline_value) * 100 if baseline_value > 0 else 0
        return change_pct > threshold_pct

    # Hit rate and success metrics: lower is worse
    if any(word in metric_name_lower for word in ["hit_rate", "hit rate", "success", "accuracy"]):
        change_pct = ((baseline_value - current_value) / baseline_value) * 100 if baseline_value > 0 else 0
        return change_pct > threshold_pct

    return False


def determine_if_improvement(
    metric_name: str,
    baseline_value: float,
    current_value: float,
    threshold_pct: float = 5.0,
) -> bool:
    """Determine if a metric change indicates an improvement.

    For latency/memory metrics: lower is better
    For throughput metrics: higher is better
    For hit rate/success metrics: higher is better

    Args:
        metric_name: Name of the metric
        baseline_value: Baseline value
        current_value: Current value
        threshold_pct: Percentage threshold for significance

    Returns:
        True if improvement
    """
    metric_name_lower = metric_name.lower()

    # Latency and memory metrics: lower is better
    if any(
        word in metric_name_lower
        for word in ["latency", "time", "memory", "bytes", "ms", "mb", "startup", "bootstrap"]
    ):
        change_pct = ((baseline_value - current_value) / baseline_value) * 100 if baseline_value > 0 else 0
        return change_pct > threshold_pct

    # Throughput and rate metrics: higher is better
    if any(
        word in metric_name_lower
        for word in ["throughput", "ops", "rate", "speed", "requests", "operations"]
    ):
        change_pct = ((current_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0
        return change_pct > threshold_pct

    # Hit rate and success metrics: higher is better
    if any(word in metric_name_lower for word in ["hit_rate", "hit rate", "success", "accuracy"]):
        change_pct = ((current_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0
        return change_pct > threshold_pct

    return False


def compare_metrics(
    baseline_metrics: List[Dict[str, Any]],
    current_metrics: List[Dict[str, Any]],
) -> List[MetricComparison]:
    """Compare metrics between two runs.

    Args:
        baseline_metrics: Metrics from baseline run
        current_metrics: Metrics from current run

    Returns:
        List of metric comparisons
    """
    comparisons = []

    # Create lookup
    baseline_lookup = {m["name"]: m for m in baseline_metrics}
    current_lookup = {m["name"]: m for m in current_metrics}

    # Get all unique metric names
    all_names = set(baseline_lookup.keys()) | set(current_lookup.keys())

    for name in all_names:
        if name not in baseline_lookup or name not in current_lookup:
            # Metric added or removed, skip
            continue

        baseline_metric = baseline_lookup[name]
        current_metric = current_lookup[name]

        baseline_value = baseline_metric["value"]
        current_value = current_metric["value"]
        unit = baseline_metric.get("unit", "")

        # Calculate percentage change
        if baseline_value > 0:
            change_pct = ((current_value - baseline_value) / baseline_value) * 100
        else:
            change_pct = 0.0

        # Determine if regression or improvement
        is_regression = determine_if_regression(name, baseline_value, current_value)
        is_improvement = determine_if_improvement(name, baseline_value, current_value)
        is_significant = abs(change_pct) >= 5.0  # 5% threshold

        comparison = MetricComparison(
            name=name,
            baseline_value=baseline_value,
            current_value=current_value,
            unit=unit,
            change_pct=change_pct,
            is_regression=is_regression,
            is_improvement=is_improvement,
            is_significant=is_significant,
        )

        comparisons.append(comparison)

    return comparisons


def compare_benchmark_runs(
    baseline_results: Dict[str, Any],
    current_results: Dict[str, Any],
) -> ComparisonReport:
    """Compare two benchmark runs.

    Args:
        baseline_results: Baseline benchmark results
        current_results: Current benchmark results

    Returns:
        Comparison report
    """
    report = ComparisonReport(
        baseline_file=baseline_results.get("metadata", {}).get("file", "baseline"),
        current_file=current_results.get("metadata", {}).get("file", "current"),
    )

    # Extract timestamps
    if "start_time" in baseline_results:
        report.baseline_time = datetime.fromisoformat(baseline_results["start_time"])
    if "start_time" in current_results:
        report.current_time = datetime.fromisoformat(current_results["start_time"])

    # Create lookup
    baseline_lookup = {r["name"]: r for r in baseline_results.get("results", [])}
    current_lookup = {r["name"]: r for r in current_results.get("results", [])}

    # Compare all benchmarks
    all_names = set(baseline_lookup.keys()) | set(current_lookup.keys())

    for name in all_names:
        if name not in baseline_lookup or name not in current_lookup:
            # Benchmark added or removed, skip
            continue

        baseline_result = baseline_lookup[name]
        current_result = current_lookup[name]

        comparison = BenchmarkComparison(
            name=name,
            category=baseline_result.get("category", "unknown"),
        )

        # Compare metrics
        metric_comparisons = compare_metrics(
            baseline_result.get("metrics", []),
            current_result.get("metrics", []),
        )

        for mc in metric_comparisons:
            comparison.add_metric(mc)

        report.comparisons.append(comparison)

    # Generate summary
    regressions = report.get_regressions()
    improvements = report.get_improvements()

    report.summary = {
        "total_benchmarks": len(report.comparisons),
        "regressions": len(regressions),
        "improvements": len(improvements),
        "neutral": len(report.comparisons) - len(regressions) - len(improvements),
    }

    return report


# =============================================================================
# Report Formatters
# =============================================================================


def format_markdown_report(report: ComparisonReport) -> str:
    """Format comparison report as Markdown."""
    lines = [
        "# Victor AI Benchmark Comparison Report",
        "",
        f"**Baseline:** {Path(report.baseline_file).name}",
        f"**Current:** {Path(report.current_file).name}",
        "",
        if report.baseline_time else "",
        f"**Baseline Time:** {report.baseline_time.strftime('%Y-%m-%d %H:%M:%S')}"
        if report.baseline_time
        else "",
        f"**Current Time:** {report.current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        if report.current_time
        else "",
        "",
        "## Summary",
        "",
        f"- **Total Benchmarks:** {report.summary['total_benchmarks']}",
        f"- **Regressions:** {report.summary['regressions']} 🔴",
        f"- **Improvements:** {report.summary['improvements']} 🟢",
        f"- **Neutral:** {report.summary['neutral']} ⚪",
        "",
    ]

    # Add regressions section
    regressions = report.get_regressions()
    if regressions:
        lines.extend([
            "## Regressions 🔴",
            "",
            "The following benchmarks show performance degradation:",
            "",
        ])

        for comp in regressions:
            lines.append(f"### {comp.name}")
            lines.append("")
            lines.append("| Metric | Baseline | Current | Change | Status |")
            lines.append("|--------|----------|---------|--------|--------|")

            for mc in comp.metrics:
                if mc.is_regression:
                    change_str = f"{mc.change_pct:+.1f}%"
                    status_str = "🔴 REGRESSION"
                    lines.append(f"| {mc.name} | {mc.baseline_value:.2f} {mc.unit} | {mc.current_value:.2f} {mc.unit} | {change_str} | {status_str} |")

            lines.append("")

    # Add improvements section
    improvements = report.get_improvements()
    if improvements:
        lines.extend([
            "## Improvements 🟢",
            "",
            "The following benchmarks show performance improvements:",
            "",
        ])

        for comp in improvements:
            lines.append(f"### {comp.name}")
            lines.append("")
            lines.append("| Metric | Baseline | Current | Change | Status |")
            lines.append("|--------|----------|---------|--------|--------|")

            for mc in comp.metrics:
                if mc.is_improvement:
                    change_str = f"{mc.change_pct:+.1f}%"
                    status_str = "🟢 IMPROVEMENT"
                    lines.append(f"| {mc.name} | {mc.baseline_value:.2f} {mc.unit} | {mc.current_value:.2f} {mc.unit} | {change_str} | {status_str} |")

            lines.append("")

    # Add detailed comparison table
    lines.extend([
        "## Detailed Comparison",
        "",
        "| Benchmark | Metric | Baseline | Current | Change | Status |",
        "|-----------|--------|----------|---------|--------|--------|",
    ])

    for comp in report.comparisons:
        for mc in comp.metrics:
            change_str = f"{mc.change_pct:+.1f}%" if mc.is_significant else "-"
            status_str = mc.status_emoji
            lines.append(f"| {comp.name} | {mc.name} | {mc.baseline_value:.2f} {mc.unit} | {mc.current_value:.2f} {mc.unit} | {change_str} | {status_str} |")

    lines.append("")

    return "\n".join(lines)


def format_html_report(report: ComparisonReport) -> str:
    """Format comparison report as HTML."""
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Victor AI Benchmark Comparison</title>",
        "    <style>",
        "        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }",
        "        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }",
        "        h2 { color: #555; margin-top: 30px; }",
        "        .summary { display: flex; gap: 20px; margin: 20px 0; }",
        "        .summary-card { flex: 1; background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #007bff; }",
        "        .regression { border-left-color: #dc3545; }",
        "        .improvement { border-left-color: #28a745; }",
        "        .neutral { border-left-color: #6c757d; }",
        "        .summary-card h4 { margin: 0 0 10px 0; color: #333; }",
        "        .summary-card .value { font-size: 2em; font-weight: bold; }",
        "        table { width: 100%; border-collapse: collapse; margin: 20px 0; }",
        "        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
        "        th { background: #007bff; color: white; font-weight: 600; }",
        "        tr:hover { background: #f8f9fa; }",
        "        .regression-row { background: #f8d7da; }",
        "        .improvement-row { background: #d4edda; }",
        "        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold; }",
        "        .badge-regression { background: #dc3545; color: white; }",
        "        .badge-improvement { background: #28a745; color: white; }",
        "        .badge-neutral { background: #6c757d; color: white; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        "        <h1>Victor AI Benchmark Comparison</h1>",
        f"        <p><strong>Baseline:</strong> {Path(report.baseline_file).name}</p>",
        f"        <p><strong>Current:</strong> {Path(report.current_file).name}</p>",
        "",
        "        <h2>Summary</h2>",
        "        <div class='summary'>",
        f"            <div class='summary-card regression'><h4>Regressions</h4><div class='value'>{report.summary['regressions']}</div></div>",
        f"            <div class='summary-card improvement'><h4>Improvements</h4><div class='value'>{report.summary['improvements']}</div></div>",
        f"            <div class='summary-card neutral'><h4>Neutral</h4><div class='value'>{report.summary['neutral']}</div></div>",
        "        </div>",
        "",
        "        <h2>Detailed Comparison</h2>",
        "        <table>",
        "            <tr><th>Benchmark</th><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th><th>Status</th></tr>",
    ]

    for comp in report.comparisons:
        for mc in comp.metrics:
            row_class = ""
            badge_class = "badge-neutral"

            if mc.is_regression:
                row_class = "regression-row"
                badge_class = "badge-regression"
            elif mc.is_improvement:
                row_class = "improvement-row"
                badge_class = "badge-improvement"

            change_str = f"{mc.change_pct:+.1f}%" if mc.is_significant else "-"
            status_str = "🔴 REGRESSION" if mc.is_regression else ("🟢 IMPROVEMENT" if mc.is_improvement else "⚪ NEUTRAL")

            lines.append(f"            <tr class='{row_class}'><td>{comp.name}</td><td>{mc.name}</td><td>{mc.baseline_value:.2f} {mc.unit}</td><td>{mc.current_value:.2f} {mc.unit}</td><td>{change_str}</td><td><span class='badge {badge_class}'>{status_str}</span></td></tr>")

    lines.extend([
        "        </table>",
        "    </div>",
        "</body>",
        "</html>",
    ])

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def load_benchmark_results(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load benchmark results from file.

    Args:
        file_path: Path to benchmark results file

    Returns:
        Benchmark results dictionary or None if error
    """
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None


def find_latest_benchmark(results_dir: Path) -> Optional[Path]:
    """Find the latest benchmark file.

    Args:
        results_dir: Directory containing benchmark results

    Returns:
        Path to latest benchmark file or None
    """
    if not results_dir.exists():
        return None

    files = sorted(
        results_dir.glob("comprehensive_benchmark_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return files[0] if files else None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Victor AI benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific runs
  %(prog)s baseline.json current.json

  # Compare with latest run
  %(prog)s baseline.json

  # Generate HTML report
  %(prog)s baseline.json current.json --format html

  # Save comparison
  %(prog)s baseline.json current.json --output comparison.md
        """,
    )

    parser.add_argument("baseline", type=Path, help="Baseline benchmark file")
    parser.add_argument("current", type=Path, nargs="?", help="Current benchmark file (uses latest if not specified)")
    parser.add_argument("--format", choices=["markdown", "html"], default="markdown", help="Output format")
    parser.add_argument("--output", type=Path, help="Save comparison to file")
    parser.add_argument("--results-dir", type=Path, default=Path(".benchmark_results"), help="Benchmark results directory")

    args = parser.parse_args()

    # Load baseline
    baseline_results = load_benchmark_results(args.baseline)
    if not baseline_results:
        print(f"Error: Could not load baseline file: {args.baseline}", file=sys.stderr)
        return 1

    # Load current
    if args.current:
        current_results = load_benchmark_results(args.current)
        if not current_results:
            print(f"Error: Could not load current file: {args.current}", file=sys.stderr)
            return 1
    else:
        # Find latest
        latest = find_latest_benchmark(args.results_dir)
        if not latest:
            print(f"Error: No benchmark files found in {args.results_dir}", file=sys.stderr)
            return 1

        if latest == args.baseline:
            print(f"Error: Latest benchmark is the same as baseline", file=sys.stderr)
            return 1

        print(f"Using latest benchmark: {latest}")
        current_results = load_benchmark_results(latest)

    # Compare
    report = compare_benchmark_runs(baseline_results, current_results)

    # Format
    if args.format == "html":
        report_text = format_html_report(report)
    else:
        report_text = format_markdown_report(report)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text)
        print(f"Comparison saved to: {args.output}")
    else:
        print("\n" + report_text)

    # Exit with error if regressions found
    if report.summary["regressions"] > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
