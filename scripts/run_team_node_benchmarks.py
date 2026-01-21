#!/usr/bin/env python3
"""Team Node Benchmark Runner Script.

Quick script to run team node performance benchmarks and generate reports.

Usage:
    python scripts/run_team_node_benchmarks.py --all
    python scripts/run_team_node_benchmarks.py --group single_level
    python scripts/run_team_node_benchmarks.py --report

Options:
    --all              Run all benchmark groups
    --group GROUP      Run specific benchmark group
    --report           Generate summary report only
    --verbose          Enable verbose output
    --no-cov           Disable coverage reporting
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_benchmarks(group=None, verbose=False, no_cov=True):
    """Run team node performance benchmarks.

    Args:
        group: Benchmark group to run (None = all)
        verbose: Enable verbose output
        no_cov: Disable coverage reporting
    """
    benchmark_file = Path("tests/performance/test_team_node_performance_benchmark.py")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(benchmark_file),
        "-v",
    ]

    if verbose:
        cmd.append("-s")

    if no_cov:
        cmd.append("--no-cov")

    if group:
        cmd.extend(["-k", group])
    else:
        cmd.append("--benchmark-only")

    # Run benchmarks
    result = subprocess.run(cmd)
    return result.returncode


def generate_report():
    """Generate performance summary report."""
    benchmark_file = Path("tests/performance/test_team_node_performance_benchmark.py")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(benchmark_file),
        "-v",
        "-s",
        "--no-cov",
        "-k",
        "summary",
    ]

    print("\n" + "=" * 80)
    print("Generating Team Node Performance Summary Report")
    print("=" * 80 + "\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Team Node Performance Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python scripts/run_team_node_benchmarks.py --all

  # Run specific group
  python scripts/run_team_node_benchmarks.py --group single_level

  # Generate summary report only
  python scripts/run_team_node_benchmarks.py --report

Benchmark Groups:
  single_level        Single level team execution
  nested              Nested team execution overhead
  formation           Formation performance comparison
  memory              Memory usage benchmarks
  recursion_context   RecursionContext operations
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark groups",
    )

    parser.add_argument(
        "--group",
        type=str,
        help="Run specific benchmark group",
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate summary report only",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--cov",
        action="store_true",
        help="Enable coverage reporting",
    )

    args = parser.parse_args()

    # Generate report if requested
    if args.report:
        return generate_report()

    # Run benchmarks
    group = args.group if not args.all else None
    return run_benchmarks(group=group, verbose=args.verbose, no_cov=not args.cov)


if __name__ == "__main__":
    sys.exit(main())
