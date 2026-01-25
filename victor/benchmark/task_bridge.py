# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge between evaluation protocol types and framework types.

This module provides bidirectional conversion between:
- BenchmarkTask (evaluation protocol) <-> Task (framework)
- evaluation TaskResult <-> framework TaskResult

This enables the evaluation harness to use high-level framework APIs
while maintaining compatibility with existing benchmark infrastructure.
"""

from typing import Any, Dict, List, Optional

from victor.evaluation.protocol import (
    BenchmarkTask,
    TaskResult as EvalTaskResult,
    TaskStatus,
)
from victor.framework.task import FrameworkTaskType, Task, TaskResult


def benchmark_task_to_framework_task(
    benchmark_task: BenchmarkTask,
    include_context: bool = True,
) -> Task:
    """Convert a BenchmarkTask to a framework Task.

    This enables using Agent.run(task) with benchmark tasks,
    leveraging the framework's task handling, enrichment, and
    complexity classification.

    Args:
        benchmark_task: The benchmark task from evaluation protocol
        include_context: Whether to include context_code, hints in context dict

    Returns:
        Framework Task with appropriate type and context
    """
    # Determine task type from benchmark task characteristics
    task_type = _infer_task_type(benchmark_task)

    # Build context dictionary
    context: Dict[str, Any] = {
        "task_id": benchmark_task.task_id,
        "benchmark": benchmark_task.benchmark.value if hasattr(benchmark_task.benchmark, 'value') else str(benchmark_task.benchmark),
    }

    if include_context:
        if benchmark_task.context_code:
            context["context_code"] = benchmark_task.context_code
        if benchmark_task.hints:
            context["hints"] = benchmark_task.hints
        if benchmark_task.test_code:
            context["test_code"] = benchmark_task.test_code
        if benchmark_task.repo:
            context["repo"] = benchmark_task.repo
        if benchmark_task.base_commit:
            context["base_commit"] = benchmark_task.base_commit

    # Build list of relevant files if available
    # Note: context_files is not in BenchmarkTask, using hints as alternative
    files: List[str] = []
    # context_files not available in BenchmarkTask protocol
    # Files would need to be extracted from hints or other context

    # Determine tool budget based on complexity
    tool_budget = _estimate_tool_budget(benchmark_task)

    return Task(
        prompt=benchmark_task.prompt,
        type=task_type,
        files=files,
        context=context,
        tool_budget=tool_budget,
    )


def framework_result_to_benchmark_result(
    framework_result: TaskResult,
    task_id: str,
    tests_passed: Optional[int] = None,
    tests_total: Optional[int] = None,
) -> EvalTaskResult:
    """Convert a framework TaskResult to an evaluation TaskResult.

    This enables collecting metrics from Agent.run() and converting
    them to the format expected by the evaluation harness.

    Args:
        framework_result: Result from Agent.run()
        task_id: The benchmark task ID
        tests_passed: Number of tests passed (from test runner)
        tests_total: Total number of tests (from test runner)

    Returns:
        Evaluation TaskResult with metrics
    """
    # Determine status based on framework result
    if framework_result.error:
        status = TaskStatus.ERROR
    elif framework_result.success:
        if tests_passed is not None and tests_total is not None:
            status = TaskStatus.PASSED if tests_passed == tests_total else TaskStatus.FAILED
        else:
            status = TaskStatus.PASSED
    else:
        status = TaskStatus.FAILED

    # Extract metrics from framework result
    metadata = framework_result.metadata or {}

    return EvalTaskResult(
        task_id=task_id,
        status=status,
        generated_code=framework_result.content,
        tests_passed=tests_passed or 0,
        tests_total=tests_total or 0,
        tests_failed=(tests_total - tests_passed) if tests_passed and tests_total else 0,
        tokens_used=metadata.get("tokens_used", 0),
        tokens_input=metadata.get("tokens_input", 0),
        tokens_output=metadata.get("tokens_output", 0),
        tool_calls=len(framework_result.tool_calls) if framework_result.tool_calls else 0,
        turns=metadata.get("turns", 0),
        error_message=framework_result.error or "",
    )


def _infer_task_type(benchmark_task: BenchmarkTask) -> FrameworkTaskType:
    """Infer FrameworkTaskType from benchmark task characteristics."""
    prompt_lower = benchmark_task.prompt.lower()

    # Check for explicit task type indicators
    if any(word in prompt_lower for word in ["fix", "bug", "error", "issue", "patch"]):
        return FrameworkTaskType.EDIT
    elif any(word in prompt_lower for word in ["add", "implement", "create", "new"]):
        return FrameworkTaskType.CREATE
    elif any(word in prompt_lower for word in ["refactor", "improve", "optimize"]):
        return FrameworkTaskType.EDIT
    elif any(word in prompt_lower for word in ["find", "search", "locate", "where"]):
        return FrameworkTaskType.SEARCH
    elif any(word in prompt_lower for word in ["analyze", "review", "explain"]):
        return FrameworkTaskType.ANALYZE
    elif any(word in prompt_lower for word in ["run", "execute", "test"]):
        return FrameworkTaskType.EXECUTE
    else:
        # Default to EDIT for most benchmark tasks (SWE-bench, etc.)
        return FrameworkTaskType.EDIT


def _estimate_tool_budget(benchmark_task: BenchmarkTask) -> Optional[int]:
    """Estimate tool budget based on task complexity.

    Uses heuristics from task characteristics to determine
    appropriate tool budget for the agent.
    """
    # Use explicit complexity override if provided
    if benchmark_task.complexity_override:
        complexity_budgets = {
            "simple": 10,
            "medium": 20,
            "complex": 40,
            "action": 15,
            "research": 25,
            "coordination": 35,
        }
        return complexity_budgets.get(benchmark_task.complexity_override.lower(), 20)

    # Estimate based on task characteristics
    budget = 15  # Base budget

    # Larger context suggests more complex task
    if benchmark_task.context_code and len(benchmark_task.context_code) > 1000:
        budget += 5

    # Multiple hints suggest multi-step task
    if benchmark_task.hints and len(benchmark_task.hints) > 2:
        budget += 5

    # Repository-based tasks (SWE-bench) need more exploration
    if benchmark_task.repo:
        budget += 10

    return min(budget, 50)  # Cap at 50


def build_benchmark_prompt(
    benchmark_task: BenchmarkTask,
    workspace_path: Optional[str] = None,
) -> str:
    """Build an enriched prompt for benchmark task execution.

    Combines the base prompt with context, hints, and workspace info
    to create a comprehensive task description for the agent.

    Args:
        benchmark_task: The benchmark task
        workspace_path: Optional path to the workspace directory

    Returns:
        Enriched prompt string
    """
    sections = []

    # Main task description
    sections.append(f"## Task\n\n{benchmark_task.prompt}")

    # Add hints if available
    if benchmark_task.hints:
        hints_text = "\n".join(f"- {hint}" for hint in benchmark_task.hints)
        sections.append(f"## Hints\n\n{hints_text}")

    # Add context code if available
    if benchmark_task.context_code:
        sections.append(f"## Context Code\n\n```\n{benchmark_task.context_code}\n```")

    # Add workspace info
    if workspace_path:
        sections.append(f"## Workspace\n\nYou are working in: `{workspace_path}`")

    # Add test info if available
    if benchmark_task.test_code:
        sections.append("## Tests\n\nTest code is available. Run tests to verify your solution.")

    return "\n\n".join(sections)
