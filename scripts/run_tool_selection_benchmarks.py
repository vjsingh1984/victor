#!/usr/bin/env python3
"""Script to run tool selection benchmarks and generate reports."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_benchmarks(
    group: str = None,
    verbose: bool = True,
    save: bool = True,
    histogram: bool = False,
    compare_runs: bool = False,
):
    """Run pytest benchmarks with specified options.

    Args:
        group: Benchmark group to run (baseline, cache, batch, etc.)
        verbose: Enable verbose output
        save: Save benchmark results to file
        histogram: Generate benchmark histogram
        compare_runs: Compare with previous runs
    """
    cmd = [
        "pytest",
        "tests/benchmark/test_tool_selection_performance.py",
        "-v" if verbose else "",
        "--benchmark-only",
        "--sort=name",  # Sort by benchmark name for consistency
    ]

    if save:
        cmd.append("--benchmark-autosave")

    if histogram:
        cmd.append("--benchmark-histogram")

    if group:
        cmd.extend(["-k", group])

    # Remove empty strings
    cmd = [c for c in cmd if c]

    print(f"\n{'=' * 80}")
    print("Running tool selection performance benchmarks")
    print(f"{'=' * 80}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    return result.returncode


def generate_summary():
    """Generate benchmark summary report."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 80)

    # Run summary test to print targets
    subprocess.run(
        [
            "pytest",
            "tests/benchmark/test_tool_selection_performance.py::test_benchmark_summary",
            "-v",
            "-s",
        ]
    )


def compare_benchmarks(run1: str, run2: str = None):
    """Compare two benchmark runs.

    Args:
        run1: First benchmark run file
        run2: Second benchmark run file (optional, compares with latest if not specified)
    """
    cmd = ["pytest-benchmark", "compare"]

    if run2:
        cmd.extend([run1, run2])
    else:
        cmd.append(run1)

    cmd.extend(["--columns=min,max,mean,std,rounds"])

    print(f"\n{'=' * 80}")
    print("Comparing benchmark runs")
    print(f"{'=' * 80}\n")

    result = subprocess.run(cmd)
    return result.returncode


def list_benchmark_runs():
    """List all saved benchmark runs."""
    import glob

    benchmark_dir = Path(".benchmarks")

    if not benchmark_dir.exists():
        print("No benchmark runs found (.benchmarks/ directory doesn't exist)")
        return

    files = sorted(benchmark_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    print(f"\n{'=' * 80}")
    print("Saved benchmark runs (most recent first)")
    print(f"{'=' * 80}\n")

    for i, f in enumerate(files[:10], 1):  # Show last 10
        mtime = f.stat().st_mtime
        size = f.stat().st_size / 1024  # KB
        print(f"{i:2d}. {f.name:50s} ({size:6.1f} KB)")

    if len(files) > 10:
        print(f"\n... and {len(files) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Run and analyze tool selection performance benchmarks"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--group",
        choices=["baseline", "cache", "batch", "category", "memory", "bottlenecks"],
        help="Benchmark group to run",
    )
    run_parser.add_argument("--no-save", action="store_true", help="Don't save results")
    run_parser.add_argument("--histogram", action="store_true", help="Generate histogram")
    run_parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark runs")
    compare_parser.add_argument("run1", help="First benchmark run file")
    compare_parser.add_argument("run2", nargs="?", help="Second benchmark run file")

    # List command
    subparsers.add_parser("list", help="List saved benchmark runs")

    # Summary command
    subparsers.add_parser("summary", help="Print benchmark summary")

    args = parser.parse_args()

    if args.command == "run":
        returncode = run_benchmarks(
            group=args.group,
            verbose=not args.quiet,
            save=not args.no_save,
            histogram=args.histogram,
        )

        if returncode == 0:
            generate_summary()

        sys.exit(returncode)

    elif args.command == "compare":
        if not args.run1:
            print("Error: run1 is required for compare command")
            list_benchmark_runs()
            sys.exit(1)

        returncode = compare_benchmarks(args.run1, args.run2)
        sys.exit(returncode)

    elif args.command == "list":
        list_benchmark_runs()

    elif args.command == "summary":
        generate_summary()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
