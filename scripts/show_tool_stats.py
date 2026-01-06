#!/usr/bin/env python3
"""Display tool usage statistics from the learning cache."""

import pickle
from pathlib import Path
from datetime import datetime
import sys


def show_tool_stats():
    """Display tool usage statistics in a readable format."""
    cache_file = Path.home() / ".victor" / "embeddings" / "tool_usage_stats.pkl"

    if not cache_file.exists():
        print(f"❌ No usage statistics found at: {cache_file}")
        print("\nThe cache will be created after you use the enhanced tool selection.")
        print("Try running some commands with 'victor main' first.")
        return

    print("📊 Tool Usage Statistics")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Cache file: {cache_file}")
    print(f"File size: {cache_file.stat().st_size:,} bytes")
    print(
        f"Last modified: {datetime.fromtimestamp(cache_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print()

    try:
        with open(cache_file, "rb") as f:
            stats = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load cache: {e}")
        return

    if not stats:
        print("📭 No tool usage data yet. The cache is empty.")
        return

    print(f"🔧 Total tools tracked: {len(stats)}")
    print()

    # Sort by usage count (most used first)
    sorted_tools = sorted(stats.items(), key=lambda x: x[1]["usage_count"], reverse=True)

    print("📈 Most Used Tools")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'Tool Name':<30} {'Uses':<8} {'Success':<10} {'Rate':<10} {'Last Used'}")
    print("─" * 90)

    for tool_name, data in sorted_tools[:20]:  # Top 20
        usage_count = data["usage_count"]
        success_count = data["success_count"]
        success_rate = (success_count / usage_count * 100) if usage_count > 0 else 0
        last_used = datetime.fromtimestamp(data["last_used"]).strftime("%Y-%m-%d %H:%M")

        # Color code by success rate
        if success_rate >= 90:
            indicator = "🟢"
        elif success_rate >= 70:
            indicator = "🟡"
        else:
            indicator = "🔴"

        print(
            f"{indicator} {tool_name:<28} {usage_count:<8} {success_count:<10} {success_rate:>6.1f}%   {last_used}"
        )

    if len(sorted_tools) > 20:
        print(f"\n... and {len(sorted_tools) - 20} more tools")

    print()
    print("🎯 Learning Insights")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Calculate insights
    total_uses = sum(d["usage_count"] for d in stats.values())
    total_successes = sum(d["success_count"] for d in stats.values())
    overall_success_rate = (total_successes / total_uses * 100) if total_uses > 0 else 0

    print(f"Total tool invocations: {total_uses:,}")
    print(f"Total successful uses: {total_successes:,}")
    print(f"Overall success rate: {overall_success_rate:.1f}%")
    print()

    # Most successful tools
    high_success_tools = [
        (name, data)
        for name, data in stats.items()
        if data["usage_count"] >= 3 and (data["success_count"] / data["usage_count"]) >= 0.9
    ]

    if high_success_tools:
        print(f"🌟 High Success Tools (≥90%, ≥3 uses): {len(high_success_tools)}")
        for name, data in sorted(
            high_success_tools, key=lambda x: x[1]["usage_count"], reverse=True
        )[:5]:
            rate = data["success_count"] / data["usage_count"] * 100
            print(f"   • {name}: {data['usage_count']} uses, {rate:.0f}% success")

    print()

    # Recent context examples
    print("💭 Recent Query Contexts")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Show contexts for top 3 tools
    for tool_name, data in sorted_tools[:3]:
        contexts = data.get("recent_contexts", [])
        if contexts:
            print(f"\n{tool_name}:")
            for i, ctx in enumerate(contexts[-3:], 1):  # Last 3 contexts
                # Truncate long contexts
                ctx_display = ctx if len(ctx) <= 60 else ctx[:57] + "..."
                print(f"  {i}. {ctx_display}")

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✨ The system learns from these patterns to improve tool selection!")
    print()


if __name__ == "__main__":
    show_tool_stats()
