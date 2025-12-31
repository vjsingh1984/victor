#!/usr/bin/env python3
"""P4 Multi-Provider Excellence - Monitoring Dashboard

This script provides a real-time dashboard for monitoring P4 feature performance:
- Hybrid search statistics
- RL threshold learning progress
- Tool deduplication metrics
- Query expansion effectiveness

Usage:
    python scripts/p4_dashboard.py
    python scripts/p4_dashboard.py --watch  # Auto-refresh every 5s
    python scripts/p4_dashboard.py --export report.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.agent.rl.coordinator import get_rl_coordinator
import sqlite3


def clear_screen():
    """Clear terminal screen."""
    print("\033[H\033[J", end="")


def format_percentage(value: float) -> str:
    """Format percentage with color."""
    if value > 0.7:
        return f"\033[91m{value:.1%}\033[0m"  # Red (bad)
    elif value > 0.3:
        return f"\033[93m{value:.1%}\033[0m"  # Yellow (ok)
    else:
        return f"\033[92m{value:.1%}\033[0m"  # Green (good)


def format_delta(value: float) -> str:
    """Format delta with color and sign."""
    if value > 0:
        return f"\033[92m+{value:.2f}\033[0m"  # Green (improvement)
    elif value < 0:
        return f"\033[91m{value:.2f}\033[0m"  # Red (degradation)
    else:
        return f"{value:.2f}"


def get_rl_learning_stats() -> Dict[str, Any]:
    """Get RL threshold learning statistics."""
    coordinator = get_rl_coordinator()
    learner = coordinator.get_learner("semantic_threshold")
    if not learner:
        return {
            "enabled": False,
            "total_outcomes": 0,
            "contexts": 0,
            "recommendations": 0,
        }

    # Query database for stats
    conn = sqlite3.connect(str(coordinator.db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM semantic_threshold_stats")
    total_contexts = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*) FROM semantic_threshold_stats
        WHERE recommended_threshold IS NOT NULL
    """
    )
    total_recommendations = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT context_key, embedding_model, task_type, tool_name,
               total_searches, zero_result_count, low_quality_count,
               avg_threshold, recommended_threshold
        FROM semantic_threshold_stats
    """
    )

    by_context = {}
    for row in cursor.fetchall():
        key, model, task_type, tool, total, zero_count, low_quality, avg_threshold, recommended = (
            row
        )
        zero_rate = (zero_count / total) if total > 0 else 0
        low_quality_rate = (low_quality / total) if total > 0 else 0

        by_context[key] = {
            "model": model,
            "task_type": task_type,
            "tool": tool,
            "total_searches": total,
            "zero_result_rate": zero_rate,
            "low_quality_rate": low_quality_rate,
            "avg_threshold": avg_threshold,
            "recommended_threshold": recommended,
            "has_recommendation": recommended is not None,
        }

    conn.close()

    return {
        "enabled": True,
        "total_outcomes": -1,  # Not tracked in new version
        "contexts": total_contexts,
        "recommendations": total_recommendations,
        "by_context": by_context,
    }


def get_recent_searches() -> List[Dict[str, Any]]:
    """Get recent search outcomes."""
    # Note: Recent search outcomes are no longer tracked in the framework version
    # The SQLite database only stores aggregate statistics, not individual outcomes
    return []


def print_dashboard(watch_mode: bool = False):
    """Print the P4 monitoring dashboard."""
    if watch_mode:
        clear_screen()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print(f"P4 MULTI-PROVIDER EXCELLENCE - MONITORING DASHBOARD")
    print(f"Updated: {timestamp}")
    print("=" * 80)
    print()

    # RL Threshold Learning Section
    print("📊 RL-BASED THRESHOLD LEARNING")
    print("-" * 80)

    rl_stats = get_rl_learning_stats()

    if not rl_stats["enabled"] or rl_stats["total_outcomes"] == 0:
        print("⚠️  No data collected yet. Enable with:")
        print("   enable_semantic_threshold_rl_learning: true")
        print()
    else:
        print(f"Total Searches: {rl_stats['total_outcomes']}")
        print(f"Active Contexts: {rl_stats['contexts']}")
        print(f"Ready Recommendations: {rl_stats['recommendations']}")
        print()

        if rl_stats["by_context"]:
            print("Context Details:")
            print()

            for key, ctx in sorted(
                rl_stats["by_context"].items(), key=lambda x: x[1]["total_searches"], reverse=True
            )[
                :10
            ]:  # Top 10 by activity
                print(f"  {ctx['model']}:{ctx['task_type']} → {ctx['tool']}")
                print(f"    Searches: {ctx['total_searches']:>4}  |  ", end="")
                print(f"Zero Results: {format_percentage(ctx['zero_result_rate'])}  |  ", end="")
                print(f"Low Quality: {format_percentage(ctx['low_quality_rate'])}")

                if ctx["has_recommendation"]:
                    delta = ctx["recommended_threshold"] - ctx["avg_threshold"]
                    print(f"    Current: {ctx['avg_threshold']:.2f}  →  ", end="")
                    print(f"✨ Recommended: {ctx['recommended_threshold']:.2f}  ", end="")
                    print(f"({format_delta(delta)})")
                else:
                    print(f"    Current: {ctx['avg_threshold']:.2f}  ", end="")
                    print("(need 5+ searches for recommendation)")

                print()

    # Recent Searches Section
    print()
    print("🔍 RECENT SEARCHES (Last 10)")
    print("-" * 80)

    recent = get_recent_searches()
    if not recent:
        print("No searches recorded yet")
    else:
        for i, search in enumerate(recent[:10], 1):
            timestamp = datetime.fromisoformat(search["timestamp"]).strftime("%H:%M:%S")
            flags = []
            if search["false_negatives"]:
                flags.append("\033[91mFALSE_NEG\033[0m")
            if search["false_positives"]:
                flags.append("\033[93mFALSE_POS\033[0m")
            flag_str = f" [{', '.join(flags)}]" if flags else ""

            print(f"{i:2}. [{timestamp}] {search['model']}:{search['task']}:{search['tool']}")
            print(f"    Query: '{search['query']}'")
            print(
                f"    Results: {search['results']:>3}  |  Threshold: {search['threshold']:.2f}{flag_str}"
            )
            print()

    # Configuration Recommendations
    print()
    print("💡 RECOMMENDATIONS")
    print("-" * 80)

    recommendations = []

    # Check for high false negative rates
    if rl_stats["enabled"] and rl_stats["by_context"]:
        for key, ctx in rl_stats["by_context"].items():
            if ctx["total_searches"] >= 5 and ctx["zero_result_rate"] > 0.3:
                recommendations.append(
                    f"⚠️  High false negatives ({ctx['zero_result_rate']:.1%}) for {key}"
                )
                if ctx["has_recommendation"]:
                    recommendations.append(
                        f"   → Lower threshold to {ctx['recommended_threshold']:.2f}"
                    )

    # Check for insufficient data
    if rl_stats["enabled"] and rl_stats["total_outcomes"] < 50:
        recommendations.append(
            f"ℹ️  Only {rl_stats['total_outcomes']} searches collected. "
            f"Need 50+ for reliable recommendations."
        )

    # Check if RL is disabled
    if not rl_stats["enabled"]:
        recommendations.append("ℹ️  RL threshold learning is disabled. Enable with:")
        recommendations.append("   enable_semantic_threshold_rl_learning: true")

    # Export suggestions
    if rl_stats["enabled"] and rl_stats["recommendations"] > 0:
        recommendations.append(
            f"✅ {rl_stats['recommendations']} recommendations ready. Export with:"
        )
        recommendations.append(
            "   python scripts/show_semantic_threshold_rl.py --export thresholds.yaml"
        )

    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✅ All systems optimal!")

    print()

    # Quick Stats Summary
    print()
    print("📈 QUICK STATS")
    print("-" * 80)

    if rl_stats["enabled"] and rl_stats["total_outcomes"] > 0:
        # Calculate overall metrics
        total_false_neg = (
            sum(
                ctx["zero_result_rate"] * ctx["total_searches"]
                for ctx in rl_stats["by_context"].values()
            )
            / rl_stats["total_outcomes"]
        )

        total_low_qual = (
            sum(
                ctx["low_quality_rate"] * ctx["total_searches"]
                for ctx in rl_stats["by_context"].values()
            )
            / rl_stats["total_outcomes"]
        )

        print(f"Overall False Negative Rate: {format_percentage(total_false_neg)}")
        print(f"Overall Low Quality Rate:    {format_percentage(total_low_qual)}")
        print(f"Search Success Rate:         {format_percentage(1.0 - total_false_neg)}")
    else:
        print("No data available")

    print()
    print("=" * 80)

    if watch_mode:
        print("Press Ctrl+C to exit watch mode")
        print("=" * 80)


def export_report(output_file: str):
    """Export dashboard data to JSON."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "rl_learning": get_rl_learning_stats(),
        "recent_searches": get_recent_searches(),
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Report exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="P4 Multi-Provider Excellence Monitoring Dashboard"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Auto-refresh dashboard every 5 seconds",
    )
    parser.add_argument(
        "--export",
        metavar="FILE",
        help="Export dashboard data to JSON file",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds for --watch mode (default: 5)",
    )

    args = parser.parse_args()

    if args.export:
        export_report(args.export)
        return

    if args.watch:
        try:
            while True:
                print_dashboard(watch_mode=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nExiting watch mode...")
    else:
        print_dashboard(watch_mode=False)


if __name__ == "__main__":
    main()
