#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Run SWE-bench evaluations with Victor.

This script runs real GitHub issue resolution tasks from the SWE-bench dataset
using Victor's agentic capabilities.

Usage:
    # Run with SWE-bench-lite from HuggingFace
    python scripts/run_swe_bench.py --dataset princeton-nlp/SWE-bench_Lite

    # Run with local JSONL file
    python scripts/run_swe_bench.py --tasks swe-bench.jsonl

    # Run with specific repos
    python scripts/run_swe_bench.py --repos django/django psf/requests --max-tasks 10

    # Run with specific model
    python scripts/run_swe_bench.py --model qwen2.5-coder:32b --base-url http://192.168.1.20:11434
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.evaluation.swe_bench_loader import (
    SWEBenchConfig,
    SWEBenchLoader,
    SWEBenchWorkspaceManager,
    get_swe_bench_repos,
)
from victor.evaluation.agent_adapter import (
    AdapterConfig,
    VictorAgentAdapter,
    create_victor_agent_callback,
)
from victor.evaluation.agentic_harness import (
    AgenticBenchmarkRunner,
    AgenticMetrics,
    AgenticTaskResult,
    FileEditValidator,
    PatchApplicationValidator,
    TestPassingValidator,
    ToolUsageValidator,
    generate_agentic_report,
)
from victor.evaluation.protocol import (
    BenchmarkTask,
    EvaluationConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_swe_bench(
    tasks_file: Optional[str] = None,
    dataset: Optional[str] = None,
    profile: str = "default",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    repos: Optional[list[str]] = None,
    instance_ids: Optional[list[str]] = None,
    max_tasks: Optional[int] = None,
    output_file: Optional[str] = None,
    timeout: int = 600,
    max_turns: int = 20,
    tool_budget: int = 50,
    cache_repos: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run SWE-bench evaluation.

    Args:
        tasks_file: Path to local JSONL file with tasks
        dataset: HuggingFace dataset name
        profile: Victor profile name
        base_url: Override base URL for provider
        model: Override model from profile
        repos: Filter by repository names
        instance_ids: Specific instance IDs to run
        max_tasks: Maximum number of tasks to run
        output_file: Output file for results (JSON)
        timeout: Timeout per task in seconds
        max_turns: Maximum conversation turns per task
        tool_budget: Maximum tool calls per task
        cache_repos: Whether to cache cloned repositories
        verbose: Enable verbose logging

    Returns:
        Dictionary with benchmark results
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load tasks
    loader_config = SWEBenchConfig(
        repos=repos,
        max_tasks=max_tasks,
        instance_ids=instance_ids,
    )
    loader = SWEBenchLoader(loader_config)

    if tasks_file:
        logger.info(f"Loading tasks from {tasks_file}")
        tasks = loader.load_from_file(tasks_file)
    elif dataset:
        logger.info(f"Loading tasks from HuggingFace: {dataset}")
        tasks = await loader.load_from_huggingface(dataset)
    else:
        logger.error("Must specify either --tasks or --dataset")
        return {"error": "No tasks specified"}

    if not tasks:
        logger.error("No tasks loaded")
        return {"error": "No tasks loaded"}

    logger.info(f"Loaded {len(tasks)} tasks")

    # Create workspace manager
    workspace_manager = SWEBenchWorkspaceManager(
        cache_dir=Path.home() / ".victor" / "swe_bench_cache" if cache_repos else None,
    )

    # Create adapter
    adapter_config = AdapterConfig(
        max_turns=max_turns,
        tool_budget=tool_budget,
        timeout_per_turn=timeout // max_turns,
        track_file_edits=True,
        track_diffs=True,
    )

    try:
        adapter = VictorAgentAdapter.from_profile(
            profile=profile,
            base_url=base_url,
            model_override=model,
            timeout=120,
            config=adapter_config,
        )
    except Exception as e:
        logger.error(f"Failed to create adapter: {e}")
        return {"error": str(e)}

    # Create benchmark runner with all validators
    runner = AgenticBenchmarkRunner(
        validators=[
            PatchApplicationValidator(),
            TestPassingValidator(),
            FileEditValidator(),
            ToolUsageValidator(min_tool_calls=1),
        ],
        timeout=timeout,
    )

    # Create callback that sets up workspace
    async def workspace_callback(task: BenchmarkTask, workspace_dir: Path):
        # Set up workspace with repo clone
        actual_workspace = await workspace_manager.setup_workspace(task, use_cache=cache_repos)

        # Execute task in workspace
        result = await adapter.execute_task(task, actual_workspace / "repo")

        return result

    # Run tasks
    results: list[AgenticTaskResult] = []
    metrics = AgenticMetrics()
    metrics.total_tasks = len(tasks)

    eval_config = EvaluationConfig(
        benchmark=tasks[0].benchmark,
        model=model or "default",
        timeout_per_task=timeout,
    )

    for i, task in enumerate(tasks, 1):
        logger.info(f"[{i}/{len(tasks)}] Running task: {task.task_id}")
        logger.info(f"  Repo: {task.repo}")
        logger.info(f"  Commit: {task.base_commit}")

        try:
            result = await runner.run_task(
                task,
                workspace_callback,
                eval_config,
            )
            results.append(result)

            # Update metrics
            if result.is_success:
                metrics.passed += 1
                logger.info(
                    f"  PASSED - {result.trace.turns} turns, {len(result.trace.tool_calls)} tool calls"
                )
            else:
                metrics.failed += 1
                logger.warning(f"  FAILED - {result.error_message or 'validation failed'}")

            metrics.total_turns += result.trace.turns
            metrics.total_tool_calls += len(result.trace.tool_calls)
            metrics.total_time_seconds += result.trace.duration_seconds

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            metrics.errors += 1

    # Calculate averages
    n = len(results)
    if n > 0:
        metrics.avg_patch_score = sum(r.patch_score for r in results) / n
        metrics.avg_test_score = sum(r.test_score for r in results) / n
        metrics.avg_edit_accuracy = sum(r.edit_accuracy for r in results) / n
        metrics.avg_tool_efficiency = sum(r.tool_efficiency for r in results) / n
        metrics.avg_overall_score = sum(r.overall_score for r in results) / n
    metrics.task_results = results

    # Generate report
    report = generate_agentic_report(metrics)

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "SWE-bench",
        "dataset": dataset or tasks_file,
        "profile": profile,
        "model": model,
        "config": {
            "max_turns": max_turns,
            "tool_budget": tool_budget,
            "timeout": timeout,
        },
        "metrics": metrics.to_dict(),
    }

    # Save output
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

        # Also save text report
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {txt_path}")

    # Print report
    print("\n" + report)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluations with Victor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Task source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--tasks",
        dest="tasks_file",
        help="Path to local JSONL file with SWE-bench tasks",
    )
    source_group.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset name (default: princeton-nlp/SWE-bench_Lite)",
    )

    # Filtering
    parser.add_argument(
        "--repos",
        nargs="+",
        help="Filter by repository names (e.g., django/django psf/requests)",
    )
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        help="Specific instance IDs to run",
    )
    parser.add_argument(
        "-n",
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to run",
    )

    # Victor configuration
    parser.add_argument(
        "--profile",
        default="default",
        help="Victor profile name (default: default)",
    )
    parser.add_argument(
        "--base-url",
        help="Override base URL for provider (e.g., http://192.168.1.20:11434)",
    )
    parser.add_argument(
        "--model",
        help="Override model from profile",
    )

    # Agent configuration
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per task in seconds (default: 600)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum conversation turns per task (default: 20)",
    )
    parser.add_argument(
        "--tool-budget",
        type=int,
        default=50,
        help="Maximum tool calls per task (default: 50)",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Output file for results (JSON)",
    )

    # Options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't cache cloned repositories",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-repos",
        action="store_true",
        help="List supported SWE-bench repositories and exit",
    )

    args = parser.parse_args()

    # Handle --list-repos
    if args.list_repos:
        print("Supported SWE-bench repositories:")
        for repo in get_swe_bench_repos():
            print(f"  - {repo}")
        sys.exit(0)

    # Set environment variable for Ollama if base_url provided
    if args.base_url:
        os.environ["OLLAMA_HOST"] = args.base_url

    # Determine task source
    tasks_file = args.tasks_file
    dataset = None if tasks_file else args.dataset

    result = asyncio.run(
        run_swe_bench(
            tasks_file=tasks_file,
            dataset=dataset,
            profile=args.profile,
            base_url=args.base_url,
            model=args.model,
            repos=args.repos,
            instance_ids=args.instance_ids,
            max_tasks=args.max_tasks,
            output_file=args.output_file,
            timeout=args.timeout,
            max_turns=args.max_turns,
            tool_budget=args.tool_budget,
            cache_repos=not args.no_cache,
            verbose=args.verbose,
        )
    )

    # Exit with error code if any failures
    if result.get("error"):
        sys.exit(1)
    metrics = result.get("metrics", {}).get("summary", {})
    if metrics.get("failed", 0) > 0 or metrics.get("errors", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
