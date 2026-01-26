#!/usr/bin/env python3
"""Enhanced Test Coverage Reporter for Victor

This tool generates comprehensive coverage reports with trends, uncovered code
analysis, and coverage goals enforcement. It enhances pytest-cov with additional
features specific to Victor's architecture.

Usage:
    python scripts/coverage_report.py --format html
    python scripts/coverage_report.py --coordinators
    python scripts/coverage_report.py --vertical coding --output coverage_coding.html

Requirements:
    pip install pytest-cov
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class CoverageMetrics:
    """Coverage metrics for a component."""

    name: str
    total_statements: int
    covered_statements: int
    missing_statements: int
    coverage_percent: float
    missing_lines: List[int] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        """Check if coverage meets minimum threshold (80%)."""
        return self.coverage_percent >= 80.0


@dataclass
class CoverageReport:
    """Coverage report for a project or component."""

    target_name: str
    total_statements: int = 0
    covered_statements: int = 0
    missing_statements: int = 0
    coverage_percent: float = 0.0
    components: Dict[str, CoverageMetrics] = field(default_factory=dict)
    trends: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        """Check if overall coverage meets minimum threshold."""
        return self.coverage_percent >= 80.0


class CoverageReporter:
    """Generates enhanced coverage reports."""

    def __init__(self, coverage_db: Path = Path(".coverage")):
        """Initialize reporter.

        Args:
            coverage_db: Path to coverage database
        """
        self.coverage_db = coverage_db
        self.report: Optional[CoverageReport] = None

        # Victor-specific coverage goals
        self.coverage_goals = {
            "coordinators": 80.0,
            "protocols": 85.0,
            "framework": 85.0,
            "tools": 75.0,
            "providers": 70.0,
        }

    def run_coverage(
        self,
        target: str = "victor",
        output_format: str = "term",
    ) -> CoverageReport:
        """Run coverage analysis.

        Args:
            target: Target to analyze (e.g., "victor", "victor/coding")
            output_format: Output format (term, json, html)

        Returns:
            CoverageReport with results
        """
        print(f"Running coverage analysis for: {target}")
        print("=" * 80)

        # Run pytest with coverage
        cmd = [
            "python",
            "-m",
            "pytest",
            "--cov=" + target,
            "--cov-report=json",
            "--cov-report=term-missing",
            "--cov-report=html",
            "-v",
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running coverage: {e}")
            return CoverageReport(target_name=target)

        # Load coverage data
        coverage_json = Path("coverage.json")
        if not coverage_json.exists():
            print("Error: coverage.json not generated")
            return CoverageReport(target_name=target)

        with open(coverage_json, "r") as f:
            data = json.load(f)

        # Parse coverage data
        self.report = self._parse_coverage_data(target, data)
        return self.report

    def _parse_coverage_data(
        self,
        target: str,
        data: Dict[str, Any],
    ) -> CoverageReport:
        """Parse coverage data from JSON.

        Args:
            target: Target name
            data: Coverage JSON data

        Returns:
            CoverageReport
        """
        report = CoverageReport(target_name=target)

        files = data.get("files", {})

        total_statements = 0
        covered_statements = 0

        for file_path, file_data in files.items():
            # Skip non-Python files
            if not file_path.endswith(".py"):
                continue

            summary = file_data.get("summary", {})
            num_statements = summary.get("num_statements", 0)
            covered = summary.get("covered_lines", 0)

            total_statements += num_statements
            covered_statements += covered

            # Get missing lines
            missing_lines = file_data.get("missing_lines", [])

            # Calculate coverage percent
            coverage_percent = (covered / num_statements * 100) if num_statements > 0 else 0.0

            # Create component metrics
            component_name = file_path.replace("/", ".").replace(".py", "")
            metrics = CoverageMetrics(
                name=component_name,
                total_statements=num_statements,
                covered_statements=covered,
                missing_statements=num_statements - covered,
                coverage_percent=coverage_percent,
                missing_lines=missing_lines,
            )

            report.components[component_name] = metrics

        report.total_statements = total_statements
        report.covered_statements = covered_statements
        report.missing_statements = total_statements - covered_statements
        report.coverage_percent = (
            covered_statements / total_statements * 100 if total_statements > 0 else 0.0
        )

        return report

    def filter_by_coordinators(self) -> CoverageReport:
        """Filter coverage report to only coordinators.

        Returns:
            Filtered CoverageReport
        """
        if not self.report:
            raise ValueError("No report loaded. Run run_coverage() first.")

        filtered = CoverageReport(target_name="coordinators")

        for name, metrics in self.report.components.items():
            if "coordinator" in name.lower():
                filtered.components[name] = metrics
                filtered.total_statements += metrics.total_statements
                filtered.covered_statements += metrics.covered_statements

        filtered.missing_statements = filtered.total_statements - filtered.covered_statements
        filtered.coverage_percent = (
            filtered.covered_statements / filtered.total_statements * 100
            if filtered.total_statements > 0
            else 0.0
        )

        return filtered

    def filter_by_vertical(self, vertical_name: str) -> CoverageReport:
        """Filter coverage report to a specific vertical.

        Args:
            vertical_name: Name of vertical (e.g., "coding")

        Returns:
            Filtered CoverageReport
        """
        if not self.report:
            raise ValueError("No report loaded. Run run_coverage() first.")

        filtered = CoverageReport(target_name=vertical_name)

        for name, metrics in self.report.components.items():
            if f".{vertical_name}." in name or name.startswith(f"victor.{vertical_name}"):
                filtered.components[name] = metrics
                filtered.total_statements += metrics.total_statements
                filtered.covered_statements += metrics.covered_statements

        filtered.missing_statements = filtered.total_statements - filtered.covered_statements
        filtered.coverage_percent = (
            filtered.covered_statements / filtered.total_statements * 100
            if filtered.total_statements > 0
            else 0.0
        )

        return filtered

    def print_report(self, report: Optional[CoverageReport] = None) -> None:
        """Print coverage report.

        Args:
            report: Report to print (defaults to self.report)
        """
        if not report:
            report = self.report

        if not report:
            print("No report to display")
            return

        status = "✓ COMPLIANT" if report.is_compliant else "✗ NON-COMPLIANT"
        print(f"\n{status}: {report.target_name}")
        print("=" * 80)
        print(f"Total Statements: {report.total_statements}")
        print(f"Covered: {report.covered_statements}")
        print(f"Missing: {report.missing_statements}")
        print(f"Coverage: {report.coverage_percent:.1f}%")
        print()

        # Show components
        if report.components:
            print("Component Breakdown:")
            print("-" * 80)

            # Sort by coverage percent
            sorted_components = sorted(
                report.components.items(),
                key=lambda x: x[1].coverage_percent,
            )

            for name, metrics in sorted_components:
                status_icon = "✓" if metrics.is_compliant else "✗"
                print(
                    f"  {status_icon} {name}: {metrics.coverage_percent:.1f}% "
                    f"({metrics.covered_statements}/{metrics.total_statements})"
                )

                # Show missing lines if low coverage
                if metrics.coverage_percent < 50 and metrics.missing_lines:
                    print(f"      Missing lines: {metrics.missing_lines[:10]}")
                    if len(metrics.missing_lines) > 10:
                        print(f"      ... and {len(metrics.missing_lines) - 10} more")

    def generate_html_report(
        self,
        output_path: Path,
        report: Optional[CoverageReport] = None,
    ) -> None:
        """Generate HTML coverage report.

        Args:
            output_path: Path to save HTML report
            report: Report to generate (defaults to self.report)
        """
        if not report:
            report = self.report

        if not report:
            raise ValueError("No report to generate")

        # Use pytest-cov's HTML report
        html_dir = Path("htmlcov")
        if html_dir.exists():
            print(f"\nHTML report available at: {html_dir}/index.html")
            return

        print("Warning: HTML report not generated. Run with --cov-report=html")

    def check_coverage_goals(self) -> Dict[str, bool]:
        """Check if coverage meets defined goals.

        Returns:
            Dict mapping component types to compliance status
        """
        if not self.report:
            raise ValueError("No report loaded. Run run_coverage() first.")

        goals_met = {}

        # Check coordinator coverage
        coordinator_report = self.filter_by_coordinators()
        goals_met["coordinators"] = coordinator_report.is_compliant

        # Check protocol coverage
        protocol_coverage = 0.0
        protocol_count = 0
        for name, metrics in self.report.components.items():
            if ".protocols." in name:
                protocol_coverage += metrics.coverage_percent
                protocol_count += 1

        if protocol_count > 0:
            protocol_coverage /= protocol_count
            goals_met["protocols"] = protocol_coverage >= self.coverage_goals["protocols"]
        else:
            goals_met["protocols"] = True

        return goals_met

    def save_trends(self, output_path: Path) -> None:
        """Save coverage trends to file.

        Args:
            output_path: Path to save trends data
        """
        if not self.report:
            raise ValueError("No report loaded. Run run_coverage() first.")

        # Load existing trends
        trends_data = {}
        if output_path.exists():
            with open(output_path, "r") as f:
                trends_data = json.load(f)

        # Add current coverage
        timestamp = datetime.now().isoformat()
        if self.report.target_name not in trends_data:
            trends_data[self.report.target_name] = []

        trends_data[self.report.target_name].append(
            {
                "timestamp": timestamp,
                "coverage_percent": self.report.coverage_percent,
                "total_statements": self.report.total_statements,
                "covered_statements": self.report.covered_statements,
            }
        )

        # Keep only last 30 entries
        trends_data[self.report.target_name] = trends_data[self.report.target_name][-30:]

        # Save
        with open(output_path, "w") as f:
            json.dump(trends_data, f, indent=2)

        print(f"\nTrends saved to: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate enhanced coverage reports")
    parser.add_argument(
        "--target",
        type=str,
        default="victor",
        help="Target to analyze (default: victor)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["term", "json", "html"],
        default="term",
        help="Output format (default: term)",
    )
    parser.add_argument(
        "--coordinators",
        action="store_true",
        help="Filter to coordinators only",
    )
    parser.add_argument(
        "--vertical",
        type=str,
        help="Filter to specific vertical (e.g., coding)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for report",
    )
    parser.add_argument(
        "--check-goals",
        action="store_true",
        help="Check if coverage meets defined goals",
    )
    parser.add_argument(
        "--save-trends",
        type=Path,
        help="Save coverage trends to file",
    )

    args = parser.parse_args()

    reporter = CoverageReporter()

    # Run coverage
    report = reporter.run_coverage(args.target, args.format)

    # Apply filters
    if args.coordinators:
        report = reporter.filter_by_coordinators()
    elif args.vertical:
        report = reporter.filter_by_vertical(args.vertical)

    # Print report
    reporter.print_report(report)

    # Generate HTML if requested
    if args.format == "html":
        output_path = args.output or Path("coverage_report.html")
        reporter.generate_html_report(output_path, report)

    # Check goals if requested
    if args.check_goals:
        print("\n" + "=" * 80)
        print("COVERAGE GOALS")
        print("=" * 80)

        goals_met = reporter.check_coverage_goals()
        for component, met in goals_met.items():
            status = "✓ MET" if met else "✗ NOT MET"
            goal = reporter.coverage_goals.get(component, 80.0)
            print(f"{component}: {status} (goal: {goal}%)")

        if not all(goals_met.values()):
            return 1

    # Save trends if requested
    if args.save_trends:
        reporter.save_trends(args.save_trends)

    # Exit code based on compliance
    if not report.is_compliant:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
