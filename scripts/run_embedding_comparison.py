#!/usr/bin/env python
"""
Quick script to run embedding benchmarks and generate comparison reports.

This is a convenience wrapper around the benchmark suite for common workflows.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and print output."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}\n")

    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run embedding operation benchmarks and comparisons"
    )
    parser.add_argument(
        '--baseline',
        default='baseline',
        help="Name for baseline results"
    )
    parser.add_argument(
        '--compare',
        help="Compare against existing results"
    )
    parser.add_argument(
        '--category',
        choices=['all', 'similarity', 'topk', 'cache', 'matrix', 'pipeline'],
        default='all',
        help="Benchmark category to run"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help="Run quick version (fewer rounds)"
    )

    args = parser.parse_args()

    # Build pytest command
    pytest_cmd = [
        'pytest',
        'tests/benchmarks/test_embedding_operations_baseline.py',
        '-v',
        '--benchmark-only',
        '--benchmark-sort=name',
    ]

    # Add category filter
    if args.category != 'all':
        pytest_cmd.append(f'-k {args.category}')

    # Add quick mode
    if args.quick:
        pytest_cmd.extend(['--benchmark-min-rounds', '5'])

    # Step 1: Run benchmarks
    print("\n" + "=" * 60)
    print("STEP 1: Running Benchmarks")
    print("=" * 60)

    if args.compare:
        # Comparison mode
        pytest_cmd.extend([
            f'--benchmark-compare={args.compare}',
            f'--benchmark-save={args.baseline}'
        ])
    else:
        # Baseline mode
        pytest_cmd.append(f'--benchmark-save={args.baseline}')

    success = run_command(
        pytest_cmd,
        f"Running {args.category} benchmarks ({'quick' if args.quick else 'full'})"
    )

    if not success:
        print("\n❌ Benchmarks failed")
        sys.exit(1)

    # Step 2: Generate analysis report
    print("\n" + "=" * 60)
    print("STEP 2: Generating Analysis Report")
    print("=" * 60)

    results_file = f".benchmarks/{args.baseline}.json"
    report_file = f"benchmark_report_{args.baseline}.md"

    analyze_cmd = [
        'python',
        'scripts/analyze_embedding_benchmarks.py',
        'report',
        '--results', results_file,
        '--output', report_file
    ]

    success = run_command(
        analyze_cmd,
        "Generating performance analysis report"
    )

    if not success:
        print("\n⚠️  Report generation failed (but benchmarks succeeded)")
        sys.exit(0)

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n✅ Benchmarks completed successfully")
    print(f"📊 Results saved to: {results_file}")
    print(f"📝 Report saved to: {report_file}")

    if args.compare:
        print(f"\n🔍 Comparison with '{args.compare}':")
        print(f"   - Check the report for performance differences")
    else:
        print(f"\n💡 Next steps:")
        print(f"   - Run: python {sys.argv[0]} --compare {args.baseline}")
        print(f"   - This will generate a comparison report")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
