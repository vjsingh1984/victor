#!/usr/bin/env python3
"""
SOLID Remediation Metrics CLI Tool

This script collects and reports SOLID remediation metrics for monitoring
the performance and behavior of the new architecture in production.

Usage:
    python scripts/solid_metrics.py [command] [options]

Commands:
    collect    - Collect metrics and print summary
    export     - Export metrics to file (JSON or Prometheus)
    baseline   - Establish baseline metrics
    compare    - Compare current metrics against baseline

Examples:
    # Collect and print metrics
    python scripts/solid_metrics.py collect

    # Export to JSON
    python scripts/solid_metrics.py export --format json --output metrics.json

    # Export to Prometheus format
    python scripts/solid_metrics.py export --format prometheus --output metrics.txt

    # Establish baseline
    python scripts/solid_metrics.py baseline --output baseline.json

    # Compare against baseline
    python scripts/solid_metrics.py compare --baseline baseline.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.monitoring.solid_metrics import (
    get_metrics_collector,
    measure_startup_time,
    print_metrics_summary,
)


def command_collect(args) -> int:
    """Collect and print metrics.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success)
    """
    print("Collecting SOLID remediation metrics...\n")

    # Measure startup time
    print("Measuring startup time...")
    startup_time = measure_startup_time()
    print(f"Startup time: {startup_time:.3f}s\n")

    # Print summary
    print_metrics_summary()

    return 0


def command_export(args) -> int:
    """Export metrics to file.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    print("Collecting metrics...")

    # Measure startup time
    measure_startup_time()

    # Get collector
    collector = get_metrics_collector()

    # Export based on format
    if args.format == "json":
        metrics = collector.export_metrics()
        output = json.dumps(metrics, indent=2)
    elif args.format == "prometheus":
        output = collector.export_prometheus()
    else:
        print(f"Unknown format: {args.format}", file=sys.stderr)
        return 1

    # Write to file or stdout
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
        print(f"Metrics exported to: {args.output}")
    else:
        print("\n" + output)

    return 0


def command_baseline(args) -> int:
    """Establish baseline metrics.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success)
    """
    print("Establishing baseline metrics...\n")

    # Measure startup time
    measure_startup_time()

    # Get collector
    collector = get_metrics_collector()
    metrics = collector.export_metrics()

    # Add metadata
    metrics["_metadata"] = {
        "type": "baseline",
        "timestamp": metrics["collector"]["timestamp"],
        "version": "0.5.0",
    }

    # Save baseline
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))

    print(f"Baseline saved to: {args.output}\n")

    # Print summary
    print_metrics_summary()

    return 0


def command_compare(args) -> int:
    """Compare current metrics against baseline.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    # Load baseline
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {args.baseline}", file=sys.stderr)
        return 1

    baseline = json.loads(baseline_path.read_text())
    print(f"Loaded baseline from: {args.baseline}\n")

    # Collect current metrics
    print("Collecting current metrics...")
    measure_startup_time()

    collector = get_metrics_collector()
    current = collector.export_metrics()

    # Compare startup time
    baseline_startup = baseline["startup"]["total_time_seconds"]
    current_startup = current["startup"]["total_time_seconds"]
    startup_diff = baseline_startup - current_startup
    startup_improvement = (startup_diff / baseline_startup) * 100

    print("=" * 60)
    print("SOLID Remediation Metrics Comparison")
    print("=" * 60)

    print("\n📊 Startup Time:")
    print(f"  Baseline: {baseline_startup:.3f}s")
    print(f"  Current:  {current_startup:.3f}s")
    print(f"  Change:   {startup_diff:+.3f}s ({startup_improvement:+.1f}%)")

    if startup_diff > 0:
        print(f"  ✅ Improvement: {startup_improvement:.1f}% faster")
    elif startup_diff < 0:
        print(f"  ⚠️  Regression: {abs(startup_improvement):.1f}% slower")
    else:
        print(f"  ℹ️  No change")

    # Compare cache hit rates (if available)
    baseline_caches = baseline.get("caches", {})
    current_caches = current.get("caches", {})

    if baseline_caches or current_caches:
        print("\n💾 Cache Hit Rates:")
        all_caches = set(baseline_caches.keys()) | set(current_caches.keys())

        for cache_name in sorted(all_caches):
            baseline_rate = baseline_caches.get(cache_name, {}).get("hit_rate", 0)
            current_rate = current_caches.get(cache_name, {}).get("hit_rate", 0)

            print(f"  {cache_name}:")
            print(f"    Baseline: {baseline_rate:.1f}%")
            print(f"    Current:  {current_rate:.1f}%")

            if current_rate > baseline_rate:
                improvement = current_rate - baseline_rate
                print(f"    ✅ Improvement: +{improvement:.1f}%")
            elif current_rate < baseline_rate:
                regression = baseline_rate - current_rate
                print(f"    ⚠️  Regression: -{regression:.1f}%")

    # Compare error counts
    baseline_errors = baseline["errors"]["total_errors"]
    current_errors = current["errors"]["total_errors"]

    print("\n⚠️  Errors:")
    print(f"  Baseline: {baseline_errors}")
    print(f"  Current:  {current_errors}")

    if current_errors > baseline_errors:
        print(f"  ⚠️  Increase: {current_errors - baseline_errors} more errors")
    elif current_errors < baseline_errors:
        print(f"  ✅ Decrease: {baseline_errors - current_errors} fewer errors")

    print("\n" + "=" * 60 + "\n")

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SOLID Remediation Metrics CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Collect command
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect and print metrics"
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export metrics to file"
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "prometheus"],
        default="json",
        help="Export format (default: json)"
    )
    export_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
    )

    # Baseline command
    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Establish baseline metrics"
    )
    baseline_parser.add_argument(
        "--output", "-o",
        default="baseline_metrics.json",
        help="Baseline output file (default: baseline_metrics.json)"
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare against baseline"
    )
    compare_parser.add_argument(
        "--baseline", "-b",
        required=True,
        help="Baseline file to compare against"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == "collect":
        return command_collect(args)
    elif args.command == "export":
        return command_export(args)
    elif args.command == "baseline":
        return command_baseline(args)
    elif args.command == "compare":
        return command_compare(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
