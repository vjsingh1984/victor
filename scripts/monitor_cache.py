#!/usr/bin/env python3
"""
Cache monitoring script for Victor AI.

Tracks cache performance metrics and provides alerts when thresholds are exceeded.

Usage:
    python scripts/monitor_cache.py [--format json|text] [--interval 60]

Examples:
    # One-time check
    python scripts/monitor_cache.py

    # Continuous monitoring
    python scripts/monitor_cache.py --interval 60

    # JSON output for Prometheus
    python scripts/monitor_cache.py --format json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add victor to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.config import load_settings
from victor.tools.caches import AdvancedCacheManager


# Thresholds for alerts
THRESHOLDS = {
    "hit_rate_warning": 0.50,  # Alert if hit rate < 50%
    "hit_rate_critical": 0.40,  # Alert if hit rate < 40%
    "latency_warning_ms": 10,  # Alert if latency > 10ms
    "latency_critical_ms": 20,  # Alert if latency > 20ms
    "memory_warning_mb": 50,  # Alert if memory > 50MB
    "memory_critical_mb": 100,  # Alert if memory > 100MB
    "utilization_warning": 0.90,  # Alert if cache > 90% full
}


def get_cache_metrics() -> Dict[str, Any]:
    """Get cache metrics from AdvancedCacheManager."""
    settings = load_settings()
    cache = AdvancedCacheManager.from_settings(settings)

    metrics = cache.get_metrics()

    # Add derived metrics
    combined = metrics["combined"]
    combined["utilization"] = combined["total_entries"] / settings.tool_selection_cache_size

    return metrics, cache, settings


def format_text(metrics: Dict[str, Any]) -> str:
    """Format metrics as human-readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("Victor AI Cache Metrics")
    lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    # Overall performance
    combined = metrics["combined"]
    lines.append("📊 Overall Performance:")
    lines.append(f"  Hit Rate: {combined['hit_rate']:.1%}")
    lines.append(f"  Total Hits: {combined['total_hits']:,}")
    lines.append(f"  Total Misses: {combined['total_misses']:,}")
    lines.append(f"  Total Entries: {combined['total_entries']:,}")
    lines.append(f"  Cache Utilization: {combined['utilization']:.1%}")
    lines.append("")

    # Strategies enabled
    lines.append("🔧 Strategies Enabled:")
    for strategy, enabled in combined["strategies_enabled"].items():
        status = "✅" if enabled else "❌"
        lines.append(f"  {status} {strategy}")
    lines.append("")

    # Basic cache
    basic = metrics["basic_cache"]["combined"]
    lines.append("💾 Basic Cache (LRU):")
    lines.append(f"  Hit Rate: {basic['hit_rate']:.1%}")
    lines.append(f"  Hits: {basic['hits']:,}")
    lines.append(f"  Misses: {basic['misses']:,}")
    lines.append(f"  Entries: {basic['total_entries']:,}")
    lines.append("")

    # Persistent cache
    if "persistent_cache" in metrics and metrics["persistent_cache"]:
        persistent = metrics["persistent_cache"]
        lines.append("💿 Persistent Cache (SQLite):")
        lines.append(f"  Hits: {persistent.get('hits', 0):,}")
        lines.append(f"  Misses: {persistent.get('misses', 0):,}")
        lines.append(f"  Entries: {persistent.get('total_entries', 0):,}")
        lines.append("")

    # Adaptive TTL
    if "adaptive_ttl" in metrics and metrics["adaptive_ttl"]:
        adaptive = metrics["adaptive_ttl"]
        lines.append("🔄 Adaptive TTL:")
        lines.append(f"  Adjustments: {adaptive.get('ttl_adjustments', 0):,}")
        lines.append(f"  Avg TTL: {adaptive.get('avg_ttl', 0):.0f}s")
        lines.append(f"  Min TTL: {adaptive.get('min_ttl', 0):.0f}s")
        lines.append(f"  Max TTL: {adaptive.get('max_ttl', 0):.0f}s")
        lines.append("")

    # Multi-level cache
    if "multi_level" in metrics and metrics["multi_level"]:
        multi = metrics["multi_level"]
        lines.append("🎚️  Multi-Level Cache:")
        lines.append(f"  L1 Hit Rate: {multi.get('l1_hit_rate', 0):.1%}")
        lines.append(f"  L2 Hit Rate: {multi.get('l2_hit_rate', 0):.1%}")
        lines.append(f"  L3 Hit Rate: {multi.get('l3_hit_rate', 0):.1%}")
        lines.append("")

    # Predictive warming
    if "predictive_warming" in metrics and metrics["predictive_warming"]:
        predictive = metrics["predictive_warming"]
        lines.append("🔮 Predictive Warming:")
        lines.append(f"  Patterns Learned: {predictive.get('patterns_learned', 0):,}")
        lines.append(f"  Predictions Made: {predictive.get('predictions_made', 0):,}")
        lines.append(f"  Accuracy: {predictive.get('accuracy', 0):.1%}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def format_json(metrics: Dict[str, Any]) -> str:
    """Format metrics as JSON for Prometheus/Grafana."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "hit_rate": metrics["combined"]["hit_rate"],
            "total_hits": metrics["combined"]["total_hits"],
            "total_misses": metrics["combined"]["total_misses"],
            "total_entries": metrics["combined"]["total_entries"],
            "utilization": metrics["combined"]["utilization"],
            "strategies_enabled": metrics["combined"]["strategies_enabled"],
        },
    }

    # Add basic cache metrics
    if "basic_cache" in metrics:
        output["metrics"]["basic_cache"] = {
            "hit_rate": metrics["basic_cache"]["combined"]["hit_rate"],
            "hits": metrics["basic_cache"]["combined"]["hits"],
            "misses": metrics["basic_cache"]["combined"]["misses"],
            "entries": metrics["basic_cache"]["combined"]["total_entries"],
        }

    # Add persistent cache metrics
    if "persistent_cache" in metrics and metrics["persistent_cache"]:
        output["metrics"]["persistent_cache"] = {
            "hits": metrics["persistent_cache"].get("hits", 0),
            "misses": metrics["persistent_cache"].get("misses", 0),
            "entries": metrics["persistent_cache"].get("total_entries", 0),
        }

    # Add adaptive TTL metrics
    if "adaptive_ttl" in metrics and metrics["adaptive_ttl"]:
        output["metrics"]["adaptive_ttl"] = {
            "adjustments": metrics["adaptive_ttl"].get("ttl_adjustments", 0),
            "avg_ttl": metrics["adaptive_ttl"].get("avg_ttl", 0),
            "min_ttl": metrics["adaptive_ttl"].get("min_ttl", 0),
            "max_ttl": metrics["adaptive_ttl"].get("max_ttl", 0),
        }

    return json.dumps(output, indent=2)


def check_alerts(metrics: Dict[str, Any]) -> list:
    """Check metrics against thresholds and return alerts."""
    alerts = []

    combined = metrics["combined"]

    # Check hit rate
    hit_rate = combined["hit_rate"]
    if hit_rate < THRESHOLDS["hit_rate_critical"]:
        alerts.append(("CRITICAL", f"Hit rate is {hit_rate:.1%} (threshold: {THRESHOLDS['hit_rate_critical']:.1%})"))
    elif hit_rate < THRESHOLDS["hit_rate_warning"]:
        alerts.append(("WARNING", f"Hit rate is {hit_rate:.1%} (threshold: {THRESHOLDS['hit_rate_warning']:.1%})"))

    # Check utilization
    utilization = combined["utilization"]
    if utilization > THRESHOLDS["utilization_warning"]:
        alerts.append(("WARNING", f"Cache utilization is {utilization:.1%} (threshold: {THRESHOLDS['utilization_warning']:.1%})"))

    # Check memory (if available)
    if "memory_usage_mb" in metrics:
        memory_mb = metrics["memory_usage_mb"]
        if memory_mb > THRESHOLDS["memory_critical_mb"]:
            alerts.append(("CRITICAL", f"Memory usage is {memory_mb:.1f}MB (threshold: {THRESHOLDS['memory_critical_mb']}MB)"))
        elif memory_mb > THRESHOLDS["memory_warning_mb"]:
            alerts.append(("WARNING", f"Memory usage is {memory_mb:.1f}MB (threshold: {THRESHOLDS['memory_warning_mb']}MB)"))

    return alerts


def print_alerts(alerts: list) -> None:
    """Print alerts to console."""
    if not alerts:
        return

    print("\n" + "=" * 60)
    print("⚠️  ALERTS")
    print("=" * 60)

    for severity, message in alerts:
        if severity == "CRITICAL":
            print(f"🚨 {severity}: {message}")
        else:
            print(f"⚠️  {severity}: {message}")

    print("=" * 60 + "\n")


def monitor_once(format_type: str) -> None:
    """Run monitoring once and print results."""
    metrics, cache, settings = get_cache_metrics()

    if format_type == "json":
        output = format_json(metrics)
        print(output)
    else:
        output = format_text(metrics)
        print(output)

    # Check for alerts
    alerts = check_alerts(metrics)
    print_alerts(alerts)

    # Exit with error code if critical alerts
    if any(severity == "CRITICAL" for severity, _ in alerts):
        sys.exit(2)
    elif alerts:
        sys.exit(1)
    else:
        sys.exit(0)


def monitor_continuous(interval: int, format_type: str) -> None:
    """Run monitoring continuously."""
    print(f"Starting continuous monitoring (interval: {interval}s)")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            metrics, cache, settings = get_cache_metrics()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}]")

            if format_type == "json":
                output = format_json(metrics)
                print(output)
            else:
                # Brief output for continuous mode
                hit_rate = metrics["combined"]["hit_rate"]
                entries = metrics["combined"]["total_entries"]
                print(f"  Hit Rate: {hit_rate:.1%} | Entries: {entries:,}")

            # Check alerts
            alerts = check_alerts(metrics)
            if alerts:
                print_alerts(alerts)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor Victor AI cache performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-time check
  python scripts/monitor_cache.py

  # Continuous monitoring (every 60 seconds)
  python scripts/monitor_cache.py --interval 60

  # JSON output for Prometheus
  python scripts/monitor_cache.py --format json

  # Continuous JSON output
  python scripts/monitor_cache.py --interval 30 --format json
        """,
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Monitoring interval in seconds (default: 0, run once)",
    )

    args = parser.parse_args()

    if args.interval > 0:
        monitor_continuous(args.interval, args.format)
    else:
        monitor_once(args.format)


if __name__ == "__main__":
    main()
