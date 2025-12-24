#!/usr/bin/env python3
"""Show semantic threshold RL learning status and recommendations.

This script displays:
1. Current RL learning statistics for all (model, task, tool) combinations
2. Recommended threshold adjustments
3. Recent search outcomes
4. Suggestions for manual configuration

Usage:
    python scripts/show_semantic_threshold_rl.py
    python scripts/show_semantic_threshold_rl.py --export thresholds.yaml
    python scripts/show_semantic_threshold_rl.py --model bge-small --task analysis
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.agent.rl.coordinator import get_rl_coordinator


def main():
    parser = argparse.ArgumentParser(
        description="Show semantic threshold RL learning status"
    )
    parser.add_argument(
        "--model",
        help="Filter by embedding model (e.g., 'bge-small', 'bge-large')",
    )
    parser.add_argument(
        "--task",
        help="Filter by task type (e.g., 'analysis', 'action', 'default')",
    )
    parser.add_argument(
        "--tool",
        help="Filter by tool name (e.g., 'code_search', 'semantic_code_search')",
    )
    parser.add_argument(
        "--export",
        metavar="FILE",
        help="Export recommendations to YAML file for configuration",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=10,
        help="Number of recent outcomes to show (default: 10)",
    )
    args = parser.parse_args()

    # Get coordinator and learner
    coordinator = get_rl_coordinator()
    learner = coordinator.get_learner("semantic_threshold")

    if not learner:
        print("ERROR: semantic_threshold learner not available")
        return 1

    # Display report (generate from database)
    print("=" * 80)
    print("Semantic Threshold RL Learning Report")
    print("=" * 80)
    print()

    # Query stats from database
    cursor = learner.db.cursor()
    cursor.execute("SELECT * FROM semantic_threshold_stats ORDER BY last_updated DESC")
    rows = cursor.fetchall()

    if not rows:
        print("No data collected yet.")
        return 0

    for row in rows:
        stats = dict(row)
        context_key = stats["context_key"]
        print(f"\n📊 {context_key}")
        print("-" * 80)
        print(f"  Searches: {stats['total_searches']}")
        print(f"  Zero Result Rate: {stats['zero_result_count'] / stats['total_searches']:.1%}")
        print(f"  Low Quality Rate: {stats['low_quality_count'] / stats['total_searches']:.1%}")
        print(f"  Avg Results: {stats['avg_results_count']:.1f}")
        print(f"  Current Avg Threshold: {stats['avg_threshold']:.2f}")

        if stats["recommended_threshold"] is not None:
            change = stats["recommended_threshold"] - stats["avg_threshold"]
            change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
            print(f"  ✨ Recommended: {stats['recommended_threshold']:.2f} ({change_str})")
        else:
            print("  Recommended: (need more data)")

    print()
    print("=" * 80)
    cursor.execute("SELECT COUNT(*) FROM rl_outcomes WHERE learner_name = 'semantic_threshold'")
    outcome_count = cursor.fetchone()[0]
    print(f"Total outcomes recorded: {outcome_count}")
    print("=" * 80)

    # Filter stats if requested
    if args.model or args.task or args.tool:
        print(f"\n{'='*80}")
        print("FILTERED RESULTS")
        print(f"{'='*80}\n")

        where_clauses = []
        params = []
        if args.model:
            where_clauses.append("embedding_model = ?")
            params.append(args.model)
        if args.task:
            where_clauses.append("task_type = ?")
            params.append(args.task)
        if args.tool:
            where_clauses.append("tool_name = ?")
            params.append(args.tool)

        where_sql = " AND ".join(where_clauses)
        cursor.execute(
            f"SELECT * FROM semantic_threshold_stats WHERE {where_sql}",
            params,
        )

        for row in cursor.fetchall():
            stats = dict(row)
            print(f"{stats['embedding_model']}:{stats['task_type']} - {stats['tool_name']}")
            print(f"  Searches: {stats['total_searches']}")
            print(f"  Zero Result Rate: {stats['zero_result_count'] / stats['total_searches']:.1%}")
            print(f"  Low Quality Rate: {stats['low_quality_count'] / stats['total_searches']:.1%}")
            print(f"  Avg Threshold: {stats['avg_threshold']:.2f}")
            if stats['recommended_threshold'] is not None:
                change = stats['recommended_threshold'] - stats['avg_threshold']
                print(f"  ✨ Recommended: {stats['recommended_threshold']:.2f} ({change:+.2f})")
            print()

    # Show recent outcomes
    print(f"\n{'='*80}")
    print(f"RECENT OUTCOMES (last {args.recent})")
    print(f"{'='*80}\n")

    cursor.execute(
        "SELECT * FROM rl_outcomes WHERE learner_name = 'semantic_threshold' ORDER BY timestamp DESC LIMIT ?",
        (args.recent,),
    )
    recent = cursor.fetchall()

    if recent:
        for i, row in enumerate(recent, 1):
            outcome = dict(row)
            import json
            metadata = json.loads(outcome.get("metadata", "{}"))

            context = f"{outcome['provider']}:{outcome['task_type']}:{outcome['model']}"
            fn = " [FALSE_NEG]" if metadata.get("false_negatives") else ""
            fp = " [FALSE_POS]" if metadata.get("false_positives") else ""
            query = metadata.get("query", "")[:40]
            results = metadata.get("results_count", 0)
            threshold = metadata.get("threshold_used", 0.0)

            print(
                f"{i:2d}. {context} - query='{query}...', "
                f"results={results}, threshold={threshold:.2f}{fn}{fp}"
            )
    else:
        print("No recent outcomes recorded yet.")

    # Export recommendations if requested
    if args.export:
        cursor.execute(
            "SELECT * FROM semantic_threshold_stats WHERE recommended_threshold IS NOT NULL"
        )
        recommendation_rows = cursor.fetchall()

        if not recommendation_rows:
            print(f"\nNo recommendations to export yet (need more data)")
            return

        # Build recommendations dict
        recommendations = {}
        for row in recommendation_rows:
            stats = dict(row)
            model_task = f"{stats['embedding_model']}:{stats['task_type']}"
            if model_task not in recommendations:
                recommendations[model_task] = {}
            recommendations[model_task][stats['tool_name']] = stats['recommended_threshold']

        import yaml

        output = {
            "semantic_threshold_overrides": recommendations,
            "enable_semantic_threshold_rl_learning": True,
        }

        filepath = Path(args.export)
        with open(filepath, "w") as f:
            f.write("# Auto-generated semantic threshold recommendations\n")
            f.write("# Generated by: python scripts/show_semantic_threshold_rl.py --export\n")
            f.write(f"# Based on {outcome_count} recorded outcomes\n\n")
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

        print(f"\n✅ Exported recommendations to: {filepath}")
        print(f"   Model:Task combinations: {len(recommendations)}")
        print(f"   Based on {outcome_count} outcomes")
        print(f"\nTo use these recommendations:")
        print(f"1. Merge {filepath} into your ~/.victor/profiles.yaml")
        print(f"2. Or copy the semantic_threshold_overrides section manually")


if __name__ == "__main__":
    main()
