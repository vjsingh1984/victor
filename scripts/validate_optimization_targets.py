#!/usr/bin/env python3
"""
Validation script for arXiv optimization targets.

Simulates realistic workload to validate that optimization targets are met:
- 40%+ token reduction on tool outputs
- 30%+ fast-path execution (skip planning)
- 40%+ small model usage

Run after deployment to staging to validate targets before production rollout.
"""

import sys
from pathlib import Path

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from victor.tools.output_pruner import get_output_pruner
from victor.framework.agentic_loop import PlanningGate
from victor.agent.paradigm_router import get_paradigm_router, ProcessingParadigm, ModelTier


def simulate_workload(num_tasks: int = 100):
    """Simulate realistic workload to validate optimization targets.

    Args:
        num_tasks: Number of tasks to simulate

    Returns:
        Dict with validation results
    """
    print(f"\n{'='*80}")
    print(f"ARXIV OPTIMIZATION TARGET VALIDATION")
    print(f"{'='*80}")
    print(f"Simulating {num_tasks} tasks...")
    print(f"")

    # Task mix reflecting real-world usage
    task_mix = [
        # 40% simple tasks (create_simple, action, search)
        *[(("create_simple", "create a new file", 2, 0.1, 10, "code_generation")) for _ in range(25)],
        *[(("action", "run tests", 1, 0.1, 5, "command")) for _ in range(10)],
        *[(("search", "find function", 3, 0.2, 15, "search")) for _ in range(5)],

        # 35% medium tasks (edit, debug, refactor)
        *[(("edit", "fix bug in function", 5, 0.4, 50, "edit")) for _ in range(20)],
        *[(("debug", "debug error", 8, 0.5, 70, "debug")) for _ in range(10)],
        *[(("refactor", "refactor module", 6, 0.5, 60, "refactor")) for _ in range(5)],

        # 25% complex tasks (design, analysis_deep, test)
        *[(("design", "design architecture", 15, 0.8, 150, "code_generation")) for _ in range(15)],
        *[(("analysis_deep", "analyze performance", 12, 0.7, 120, "analysis")) for _ in range(7)],
        *[(("test", "write comprehensive tests", 10, 0.6, 100, "test")) for _ in range(3)],
    ]

    # Take only num_tasks
    task_mix = task_mix[:num_tasks]

    # Initialize components
    pruner = get_output_pruner()
    gate = PlanningGate(enabled=True)
    router = get_paradigm_router()

    # Track metrics
    fast_path_count = 0
    small_model_count = 0
    direct_paradigm_count = 0

    total_original_lines = 0
    total_pruned_lines = 0

    pruning_count = 0

    # Simulate each task
    for i, (task_type, query, tool_budget, complexity, query_length, output_type) in enumerate(task_mix):
        # 1. Planning Gate
        use_planning = gate.should_use_llm_planning(
            task_type=task_type,
            tool_budget=tool_budget,
            query_complexity=complexity,
            query_length=query_length,
            context={},
        )
        if not use_planning:
            fast_path_count += 1

        # 2. Paradigm Router
        decision = router.route(
            task_type=task_type,
            query=query,
            history_length=0,
            query_complexity=complexity,
        )
        if decision.model_tier == ModelTier.SMALL:
            small_model_count += 1
        if decision.paradigm == ProcessingParadigm.DIRECT:
            direct_paradigm_count += 1

        # 3. Tool Output Pruning (simulate tool output)
        if output_type == "code_generation":
            # Simulate reading a file
            tool_output = "\n".join([
                f"# Line {i}",
                "import sys",
                "import os",
                "",
                "def function1():",
                "    pass",
                "# Comment line",
                "",
                "def function2():",
                "    pass",
            ] * 20)  # 200 lines

            pruned, info = pruner.prune(
                tool_output=tool_output,
                task_type=task_type,
                tool_name="read",
                context={"task_type": task_type},
            )

            total_original_lines += info.original_lines
            total_pruned_lines += info.pruned_lines
            if info.was_pruned:
                pruning_count += 1

    # Calculate metrics
    total_tasks = len(task_mix)
    fast_path_percentage = (fast_path_count / total_tasks) * 100
    small_model_percentage = (small_model_count / total_tasks) * 100
    direct_paradigm_percentage = (direct_paradigm_count / total_tasks) * 100

    if total_original_lines > 0:
        token_reduction = ((total_original_lines - total_pruned_lines) / total_original_lines) * 100
    else:
        token_reduction = 0.0

    # Collect component statistics
    gate_stats = gate.get_statistics()
    router_stats = router.get_statistics()

    # Print results
    print(f"VALIDATION RESULTS ({total_tasks} tasks simulated):")
    print(f"{'-'*80}")
    print(f"Token Reduction:")
    print(f"  Original lines: {total_original_lines}")
    print(f"  Pruned lines: {total_pruned_lines}")
    print(f"  Reduction: {token_reduction:.1f}%")
    print(f"  Target: ≥40% | Status: {'✅ PASS' if token_reduction >= 40 else '❌ FAIL'}")
    print(f"")

    print(f"Fast-Path Execution:")
    print(f"  Fast-path tasks: {fast_path_count}/{total_tasks}")
    print(f"  Fast-path rate: {fast_path_percentage:.1f}%")
    print(f"  Gate decisions: {gate_stats['total_decisions']}")
    print(f"  Target: ≥30% | Status: {'✅ PASS' if fast_path_percentage >= 30 else '❌ FAIL'}")
    print(f"")

    print(f"Small Model Usage:")
    print(f"  Small model tasks: {small_model_count}/{total_tasks}")
    print(f"  Small model rate: {small_model_percentage:.1f}%")
    print(f"  Direct paradigm: {direct_paradigm_percentage:.1f}%")
    print(f"  Total routings: {router_stats['total_routings']}")
    print(f"  Target: ≥40% | Status: {'✅ PASS' if small_model_percentage >= 40 else '❌ FAIL'}")
    print(f"")

    print(f"Component Statistics:")
    print(f"  Planning Gate: {gate_stats['fast_path_count']} fast-paths / {gate_stats['total_decisions']} decisions")
    print(f"  Paradigm Router: {router_stats.get('paradigm_counts', {})}")
    print(f"  Tool Pruner: {pruning_count} pruning operations")
    print(f"")

    # Overall validation
    all_pass = (
        token_reduction >= 40 and
        fast_path_percentage >= 30 and
        small_model_percentage >= 40
    )

    print(f"{'='*80}")
    if all_pass:
        print(f"✅ ALL TARGETS MET - Ready for production rollout")
    else:
        print(f"⚠️  SOME TARGETS NOT MET - Review and adjust before production")
    print(f"{'='*80}")
    print(f"")

    return {
        "token_reduction": token_reduction,
        "fast_path_percentage": fast_path_percentage,
        "small_model_percentage": small_model_percentage,
        "all_pass": all_pass,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate arXiv optimization targets"
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=100,
        help="Number of tasks to simulate (default: 100)",
    )

    args = parser.parse_args()

    results = simulate_workload(num_tasks=args.tasks)

    # Exit with appropriate code
    sys.exit(0 if results["all_pass"] else 1)


if __name__ == "__main__":
    main()
