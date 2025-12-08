#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Run agentic benchmarks (SWE-bench style) with Victor.

This script runs multi-turn agentic tasks that involve tool usage,
file editing, and multi-step problem solving.

Usage:
    # Run sample tasks with default profile
    python scripts/run_agentic_benchmark.py

    # Run with specific Ollama host
    python scripts/run_agentic_benchmark.py --base-url http://192.168.1.20:11434

    # Run with custom task file
    python scripts/run_agentic_benchmark.py --tasks tasks.json

    # Run with specific model
    python scripts/run_agentic_benchmark.py --model qwen2.5-coder:32b
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    TestPassingValidator,
    ToolUsageValidator,
)
from victor.evaluation.protocol import (
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_sample_tasks() -> List[BenchmarkTask]:
    """Get sample agentic tasks for testing."""
    return [
        BenchmarkTask(
            task_id="agentic/file_create",
            benchmark=BenchmarkType.CUSTOM,
            description="Create a Calculator class with basic arithmetic operations",
            prompt="""Create a Python file called 'calculator.py' that implements a Calculator class with:
            - add(a, b) method that returns a + b
            - subtract(a, b) method that returns a - b
            - multiply(a, b) method that returns a * b
            - divide(a, b) method that returns a / b (handle division by zero)

            Include docstrings and type hints.""",
            test_code="""
import calculator

def test_calculator():
    calc = calculator.Calculator()
    assert calc.add(2, 3) == 5
    assert calc.subtract(5, 3) == 2
    assert calc.multiply(3, 4) == 12
    assert calc.divide(10, 2) == 5.0
    try:
        calc.divide(1, 0)
        assert False, "Should raise exception"
    except (ValueError, ZeroDivisionError):
        pass
    print("All tests passed!")

test_calculator()
""",
            category="file_creation",
            difficulty="easy",
        ),
        BenchmarkTask(
            task_id="agentic/bug_fix",
            benchmark=BenchmarkType.CUSTOM,
            description="Fix off-by-one error in bubble sort implementation",
            prompt="""There's a bug in the 'buggy_sort.py' file. The bubble sort implementation
            has an off-by-one error that causes it to not fully sort the list.

            Find and fix the bug. Verify your fix works by testing with sample inputs.""",
            context_code="""# buggy_sort.py
def bubble_sort(arr):
    '''Sort an array using bubble sort.'''
    n = len(arr)
    for i in range(n - 1):  # Bug: should be range(n)
        for j in range(n - i - 2):  # Bug: should be n - i - 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""",
            test_code="""
import buggy_sort

def test_sort():
    assert buggy_sort.bubble_sort([3, 1, 2]) == [1, 2, 3]
    assert buggy_sort.bubble_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
    assert buggy_sort.bubble_sort([1]) == [1]
    assert buggy_sort.bubble_sort([]) == []
    print("All tests passed!")

test_sort()
""",
            category="bug_fix",
            difficulty="easy",
        ),
        BenchmarkTask(
            task_id="agentic/refactor",
            benchmark=BenchmarkType.CUSTOM,
            description="Refactor messy code with proper naming and type hints",
            prompt="""Refactor the 'messy_code.py' file to:
            1. Extract the repeated logic into a helper function
            2. Add proper error handling
            3. Add type hints
            4. Improve variable names

            The functionality should remain the same.""",
            context_code="""# messy_code.py
def p(d):
    r = []
    for x in d:
        if x > 0:
            t = x * 2
            if t < 100:
                r.append(t)
    for x in d:
        if x < 0:
            t = x * 2
            if t > -100:
                r.append(t)
    return r
""",
            test_code="""
import messy_code

def test_refactored():
    result = messy_code.p([1, -2, 50, -60, 3])
    # Positive doubled: 2, 100 (excluded), 6
    # Negative doubled: -4, -120 (excluded)
    assert sorted(result) == sorted([2, 6, -4])
    print("All tests passed!")

test_refactored()
""",
            category="refactoring",
            difficulty="medium",
        ),
    ]


def setup_workspace(task: BenchmarkTask, workspace_dir: Path) -> None:
    """Set up workspace with initial files for a task."""
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Write context code as file if provided
    if task.context_code:
        # Extract filename from context code comment
        lines = task.context_code.strip().split("\n")
        if lines and lines[0].startswith("#"):
            filename = lines[0].lstrip("# ").strip()
            code = "\n".join(lines[1:])
            (workspace_dir / filename).write_text(code)
        else:
            (workspace_dir / "context.py").write_text(task.context_code)


async def run_benchmark(
    profile: str = "default",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    tasks_file: Optional[str] = None,
    output_file: Optional[str] = None,
    max_tasks: Optional[int] = None,
    timeout: int = 300,
    max_turns: int = 15,
    tool_budget: int = 30,
    max_parallel: int = 1,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run agentic benchmark.

    Args:
        profile: Victor profile name
        base_url: Override base URL for provider
        model: Override model from profile
        tasks_file: JSON file with tasks (uses samples if None)
        output_file: Output file for results
        max_tasks: Maximum number of tasks to run
        timeout: Timeout per task in seconds
        max_turns: Maximum conversation turns per task
        tool_budget: Maximum tool calls per task
        max_parallel: Maximum number of tasks to run in parallel (default: 1)
        verbose: Enable verbose logging

    Returns:
        Dictionary with benchmark results
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load tasks
    if tasks_file:
        with open(tasks_file) as f:
            task_data = json.load(f)
        tasks = []
        for t in task_data:
            # Extract metadata fields if present
            meta = t.get("metadata", {})
            task = BenchmarkTask(
                task_id=t["task_id"],
                benchmark=BenchmarkType.CUSTOM,
                description=t.get("prompt", "")[:100] + "...",
                prompt=t["prompt"],
                context_code=t.get("context_code", ""),
                test_code=t.get("test_code", ""),
                difficulty=meta.get("difficulty", "medium"),
                category=meta.get("category", ""),
                tags=meta.get("expected_tools", []),
            )
            tasks.append(task)
    else:
        tasks = get_sample_tasks()

    if max_tasks:
        tasks = tasks[:max_tasks]

    logger.info(f"Running {len(tasks)} agentic tasks with profile '{profile}'")

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

    # Create benchmark runner with validators
    runner = AgenticBenchmarkRunner(
        validators=[
            FileEditValidator(),
            ToolUsageValidator(),
        ]
    )

    # Create callback
    callback = create_victor_agent_callback(adapter)

    # Run tasks
    results: List[AgenticTaskResult] = []
    metrics = AgenticMetrics()

    for i, task in enumerate(tasks, 1):
        logger.info(f"[{i}/{len(tasks)}] Running task: {task.task_id}")

        # Create temp workspace for task
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)
            setup_workspace(task, workspace_path)

            config = EvaluationConfig(
                benchmark=task.benchmark,
                model=model or "unknown",
                timeout_per_task=timeout,
            )

            try:
                result = await runner.run_task(task, callback, config)
                results.append(result)

                # Update metrics
                metrics.total_tasks += 1
                metrics.task_results.append(result)
                if result.is_success:
                    metrics.passed += 1
                    logger.info(f"  PASSED - {result.trace.turns} turns, {len(result.trace.tool_calls)} tool calls")
                else:
                    metrics.failed += 1
                    logger.warning(f"  FAILED - {result.trace.validations}")

                metrics.total_tool_calls += len(result.trace.tool_calls)
                metrics.total_turns += result.trace.turns
                metrics.total_time_seconds += result.trace.duration_seconds

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                metrics.total_tasks += 1
                metrics.errors += 1

    # Generate text report from metrics
    report = f"""
==========================================
AGENTIC BENCHMARK RESULTS
==========================================

Summary:
  Total tasks: {metrics.total_tasks}
  Passed: {metrics.passed}
  Failed: {metrics.failed}
  Errors: {metrics.errors}
  Pass rate: {metrics.pass_rate:.1%}

Efficiency Metrics:
  Total turns: {metrics.total_turns}
  Avg turns: {metrics.avg_turns:.1f}
  Total tool calls: {metrics.total_tool_calls}
  Avg tool calls: {metrics.avg_tool_calls:.1f}
  Total duration: {metrics.total_time_seconds:.1f}s

==========================================
"""

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile,
        "model": model,
        "metrics": {
            "total_tasks": metrics.total_tasks,
            "passed": metrics.passed,
            "failed": metrics.failed,
            "errors": metrics.errors,
            "pass_rate": metrics.pass_rate,
            "avg_tool_calls": metrics.avg_tool_calls,
            "avg_edit_accuracy": metrics.avg_edit_accuracy,
            "avg_turns": metrics.avg_turns,
            "avg_duration": metrics.total_time_seconds / metrics.total_tasks if metrics.total_tasks > 0 else 0,
            "total_duration": metrics.total_time_seconds,
        },
        "tasks": [
            {
                "task_id": r.task_id,
                "passed": r.is_success,
                "duration": r.trace.duration_seconds,
                "turns": r.trace.turns,
                "tool_calls": len(r.trace.tool_calls),
                "file_edits": len(r.trace.file_edits),
                "validation_results": r.trace.validations,
            }
            for r in results
        ],
    }

    # Save output
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
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
        description="Run agentic benchmarks with Victor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
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
    parser.add_argument(
        "--tasks",
        dest="tasks_file",
        help="JSON file with task definitions (uses built-in samples if not provided)",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "-n", "--max-tasks",
        type=int,
        help="Maximum number of tasks to run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per task in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum conversation turns per task (default: 15)",
    )
    parser.add_argument(
        "--tool-budget",
        type=int,
        default=30,
        help="Maximum tool calls per task (default: 30)",
    )
    parser.add_argument(
        "-p", "--parallel",
        type=int,
        default=1,
        help="Maximum number of tasks to run in parallel (default: 1 for sequential)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set environment variable for Ollama if base_url provided
    if args.base_url:
        os.environ["OLLAMA_HOST"] = args.base_url

    result = asyncio.run(
        run_benchmark(
            profile=args.profile,
            base_url=args.base_url,
            model=args.model,
            tasks_file=args.tasks_file,
            output_file=args.output_file,
            max_tasks=args.max_tasks,
            timeout=args.timeout,
            max_turns=args.max_turns,
            tool_budget=args.tool_budget,
            max_parallel=args.parallel,
            verbose=args.verbose,
        )
    )

    # Exit with error code if any failures
    if result.get("error") or result.get("metrics", {}).get("failed", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
