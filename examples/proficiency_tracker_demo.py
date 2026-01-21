#!/usr/bin/env python
"""Comprehensive example demonstrating ProficiencyTracker features.

This script shows how to use the enhanced ProficiencyTracker with:
- Moving averages for performance metrics
- Improvement trajectory tracking
- Trend detection and analysis
- Data export for RL training
- Performance pattern analysis
"""

import logging
from victor.agent.improvement import (
    ProficiencyTracker,
    TaskOutcome,
    TrendDirection,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate ProficiencyTracker capabilities."""
    tracker = ProficiencyTracker(moving_avg_window=20)

    logger.info("=" * 70)
    logger.info("ProficiencyTracker Enhanced Features Demo")
    logger.info("=" * 70)

    # 1. Record outcomes for different tasks and tools
    logger.info("\n1. Recording task outcomes...")
    tasks = [
        ("code_review", "ast_analyzer"),
        ("code_review", "semantic_search"),
        ("test_generation", "test_generator"),
        ("refactoring", "code_transformer"),
    ]

    for i, (task, tool) in enumerate(tasks * 25):  # 100 outcomes per task-tool
        outcome = TaskOutcome(
            success=(i % 4 != 0),  # 75% success rate
            duration=1.0 + (i % 10) * 0.2,
            cost=0.001 * (1 + (i % 5) * 0.5),
            quality_score=0.7 + (i % 6) * 0.05,
        )
        tracker.record_outcome(task, tool, outcome)

    logger.info(f"✓ Recorded {100 * len(tasks)} outcomes")

    # 2. Get proficiency scores
    logger.info("\n2. Getting proficiency scores...")
    for task, tool in tasks:
        score = tracker.get_proficiency(tool)
        if score:
            logger.info(
                f"  {tool}: {score.success_rate:.1%} success, "
                f"{score.trend} trend, {score.total_executions} executions"
            )

    # 3. Calculate moving averages
    logger.info("\n3. Calculating moving averages...")
    for task, _ in tasks:
        ma = tracker.get_moving_average_metrics(task)
        if ma:
            logger.info(
                f"  {task}: {ma.success_rate_ma:.1%} MA, "
                f"σ={ma.std_dev:.3f}, window={ma.window_size}"
            )

    # 4. Record trajectory snapshots
    logger.info("\n4. Recording improvement trajectory...")
    for task, _ in tasks:
        tracker.record_trajectory_snapshot(task)

    trajectory = tracker.get_improvement_trajectory("code_review")
    logger.info(f"  Found {len(trajectory)} trajectory points for code_review")
    if trajectory:
        latest = trajectory[-1]
        logger.info(
            f"    Latest: {latest.success_rate:.1%} success, "
            f"{latest.trend} trend"
        )

    # 5. Get top proficiencies
    logger.info("\n5. Top 5 proficiencies...")
    top_tools = tracker.get_top_proficiencies(n=5)
    for tool, score in top_tools:
        logger.info(f"  {tool}: {score.success_rate:.1%} ({score.total_executions} uses)")

    # 6. Identify weaknesses
    logger.info("\n6. Identifying weaknesses (below 80% threshold)...")
    weaknesses = tracker.get_weaknesses(threshold=0.8)
    for tool in weaknesses:
        score = tracker.get_proficiency(tool)
        if score:
            logger.info(
                f"  {tool}: {score.success_rate:.1%} success rate (needs improvement)"
            )

    # 7. Manual proficiency update
    logger.info("\n7. Manually updating proficiency...")
    initial_rate = tracker.get_task_success_rate("code_review")
    tracker.update_proficiency("code_review", 0.05)
    new_rate = tracker.get_task_success_rate("code_review")
    logger.info(f"  code_review: {initial_rate:.1%} → {new_rate:.1%}")

    # 8. Trend detection
    logger.info("\n8. Detecting trends...")
    test_values = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    trend = tracker.detect_trend_direction(test_values)
    logger.info(f"  Values: {test_values}")
    logger.info(f"  Detected trend: {trend}")

    # 9. Statistics summary
    logger.info("\n9. Statistics summary...")
    stats = tracker.get_statistics_summary()
    logger.info(f"  Total tools: {stats['total_tools']}")
    logger.info(f"  Total tasks: {stats['total_tasks']}")
    logger.info(f"  Total outcomes: {stats['total_outcomes']}")
    logger.info(f"  Success rate: {stats['success_rate']['average']:.1%}")
    logger.info(f"  Avg duration: {stats['duration']['average']:.2f}s")
    logger.info(f"  Avg quality: {stats['quality_score']['average']:.2f}")

    # 10. Performance pattern analysis
    logger.info("\n10. Performance pattern analysis...")
    patterns = tracker.analyze_performance_patterns()
    logger.info(f"  Improving tools: {len(patterns['improving_tools'])}")
    logger.info(f"  Declining tools: {len(patterns['declining_tools'])}")
    logger.info(f"  Fastest tools: {len(patterns['fastest_tools'])}")
    logger.info(f"  Most reliable: {len(patterns['most_reliable_tools'])}")

    # 11. Export training data
    logger.info("\n11. Exporting training data...")
    try:
        df = tracker.export_training_data()
        logger.info(f"  Exported {len(df)} training samples")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Shape: {df.shape}")
    except ImportError:
        logger.warning("  ⚠ Pandas not available, skipping export")

    # 12. Export proficiency history
    logger.info("\n12. Exporting proficiency history...")
    try:
        history_df = tracker.export_proficiency_history("code_review")
        logger.info(f"  Exported {len(history_df)} trajectory points")
    except ImportError:
        logger.warning("  ⚠ Pandas not available, skipping export")

    logger.info("\n" + "=" * 70)
    logger.info("Demo completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
