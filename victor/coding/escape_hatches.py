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

"""Escape hatches for Coding YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.

Example YAML usage:
    - id: check_tests
      type: condition
      condition: "tests_passing"  # References escape hatch
      branches:
        "passing": deploy
        "failing": fix_code
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================


def tests_passing(ctx: dict[str, Any]) -> str:
    """Check if tests are passing.

    Args:
        ctx: Workflow context with keys:
            - test_results (dict): Test execution results
            - min_coverage (float): Minimum coverage threshold

    Returns:
        "passing", "failing", or "no_tests"
    """
    test_results = ctx.get("test_results", {})
    min_coverage = ctx.get("min_coverage", 0.8)

    if not test_results:
        return "no_tests"

    passed = test_results.get("passed", 0)
    failed = test_results.get("failed", 0)
    coverage = test_results.get("coverage", 0)

    if failed > 0:
        return "failing"

    if coverage < min_coverage:
        return "failing"

    if passed > 0:
        return "passing"

    return "no_tests"


def code_quality_check(ctx: dict[str, Any]) -> str:
    """Assess code quality based on linting and static analysis.

    Args:
        ctx: Workflow context with keys:
            - lint_results (dict): Linter output
            - type_check_results (dict): Type checker output
            - quality_threshold (str): Minimum quality level

    Returns:
        "excellent", "good", "acceptable", or "needs_improvement"
    """
    lint_results = ctx.get("lint_results", {})
    type_check_results = ctx.get("type_check_results", {})

    lint_errors = lint_results.get("errors", 0)
    lint_warnings = lint_results.get("warnings", 0)
    type_errors = type_check_results.get("errors", 0)

    if lint_errors == 0 and type_errors == 0 and lint_warnings == 0:
        return "excellent"

    if lint_errors == 0 and type_errors == 0:
        return "good"

    if lint_errors <= 3 and type_errors <= 2:
        return "acceptable"

    return "needs_improvement"


def should_retry_implementation(ctx: dict[str, Any]) -> str:
    """Determine if implementation should be retried.

    Args:
        ctx: Workflow context with keys:
            - test_results (dict): Test execution results
            - iteration_count (int): Current iteration
            - max_iterations (int): Maximum allowed iterations
            - error (str): Error message if any

    Returns:
        "retry" or "give_up"
    """
    iteration = ctx.get("iteration_count", 0)
    max_iter = ctx.get("max_iterations", 3)
    test_results = ctx.get("test_results", {})
    error = ctx.get("error")

    if iteration >= max_iter:
        logger.info(f"Max iterations ({max_iter}) reached, giving up")
        return "give_up"

    if error and "fatal" in error.lower():
        return "give_up"

    failed = test_results.get("failed", 0)
    if failed > 0:
        return "retry"

    return "give_up"


def review_verdict(ctx: dict[str, Any]) -> str:
    """Determine code review verdict.

    Args:
        ctx: Workflow context with keys:
            - review_comments (list): Review comments
            - approval_status (str): Approval status
            - blocking_issues (int): Number of blocking issues

    Returns:
        "approved", "changes_requested", or "needs_discussion"
    """
    approval_status = ctx.get("approval_status", "pending")
    blocking_issues = ctx.get("blocking_issues", 0)
    review_comments = ctx.get("review_comments", [])

    if approval_status == "approved" and blocking_issues == 0:
        return "approved"

    if blocking_issues > 0:
        return "changes_requested"

    if len(review_comments) > 5:
        return "needs_discussion"

    if approval_status == "pending":
        return "needs_discussion"

    return "changes_requested"


def complexity_assessment(ctx: dict[str, Any]) -> str:
    """Assess task complexity for planning.

    Args:
        ctx: Workflow context with keys:
            - files_to_modify (int): Number of files to change
            - estimated_lines (int): Estimated lines of code
            - dependencies (list): External dependencies involved

    Returns:
        "simple", "moderate", "complex", or "major"
    """
    files = ctx.get("files_to_modify", 1)
    lines = ctx.get("estimated_lines", 0)
    dependencies = ctx.get("dependencies", [])

    dep_count = len(dependencies) if isinstance(dependencies, list) else 0

    if files <= 1 and lines <= 50 and dep_count == 0:
        return "simple"

    if files <= 3 and lines <= 200:
        return "moderate"

    if files <= 10 or lines <= 500:
        return "complex"

    return "major"


def complexity_check(ctx: dict[str, Any]) -> str:
    """Assess task complexity from task analysis for team routing.

    Used by team_node workflows to route tasks to appropriate team sizes.
    Evaluates task_analysis output to determine complexity level.

    Args:
        ctx: Workflow context with keys:
            - task_analysis (str|dict): Task analysis from planner agent
            - user_task (str): Original user task description

    Returns:
        "simple", "medium", or "complex"
    """
    task_analysis = ctx.get("task_analysis", "")
    user_task = ctx.get("user_task", "")

    # Handle string analysis (from agent output)
    if isinstance(task_analysis, str):
        analysis_lower = task_analysis.lower()

        # Check for explicit complexity mentions
        if any(kw in analysis_lower for kw in ["complex", "major", "significant", "large"]):
            return "complex"
        if any(kw in analysis_lower for kw in ["medium", "moderate", "several"]):
            return "medium"
        if any(kw in analysis_lower for kw in ["simple", "trivial", "straightforward", "minor"]):
            return "simple"

        # Estimate from team size mentions
        if "team size: 4" in analysis_lower or "team size: 3" in analysis_lower:
            return "complex"
        if "team size: 2" in analysis_lower:
            return "medium"
        if "team size: 1" in analysis_lower:
            return "simple"

    # Handle dict analysis
    elif isinstance(task_analysis, dict):
        complexity = task_analysis.get("complexity", "").lower()
        if complexity in ["complex", "major"]:
            return "complex"
        if complexity in ["medium", "moderate"]:
            return "medium"
        if complexity in ["simple", "trivial"]:
            return "simple"

        # Check team size from dict
        team_size = task_analysis.get("team_size", 1)
        if isinstance(team_size, int):
            if team_size >= 4:
                return "complex"
            if team_size >= 2:
                return "medium"
            return "simple"

    # Fallback: estimate from user task length/keywords
    task_lower = user_task.lower() if isinstance(user_task, str) else ""
    if len(task_lower) > 200 or any(
        kw in task_lower for kw in ["refactor", "redesign", "migrate", "overhaul"]
    ):
        return "complex"
    if len(task_lower) > 100 or any(
        kw in task_lower for kw in ["add feature", "implement", "create"]
    ):
        return "medium"

    return "simple"


def tdd_cycle_status(ctx: dict[str, Any]) -> str:
    """Determine TDD cycle status.

    Args:
        ctx: Workflow context with keys:
            - tests_written (bool): Whether tests are written
            - tests_passing (bool): Whether tests pass
            - implementation_complete (bool): Whether implementation is done

    Returns:
        "red", "green", or "refactor"
    """
    tests_written = ctx.get("tests_written", False)
    passing = ctx.get("tests_passing", False)
    impl_complete = ctx.get("implementation_complete", False)

    if not tests_written:
        return "red"

    if not passing:
        return "red"

    if passing and impl_complete:
        return "refactor"

    return "green"


def bugfix_priority(ctx: dict[str, Any]) -> str:
    """Determine bugfix priority level.

    Args:
        ctx: Workflow context with keys:
            - severity (str): Bug severity (critical, high, medium, low)
            - affected_users (int): Number of affected users
            - has_workaround (bool): Whether workaround exists

    Returns:
        "p0", "p1", "p2", or "p3"
    """
    severity = ctx.get("severity", "medium")
    affected_users = ctx.get("affected_users", 0)
    has_workaround = ctx.get("has_workaround", False)

    if severity == "critical":
        return "p0"

    if severity == "high" and affected_users > 100:
        return "p0"

    if severity == "high" or affected_users > 50:
        return "p1"

    if severity == "medium" and not has_workaround:
        return "p2"

    return "p3"


def should_continue_fixing(ctx: dict[str, Any]) -> str:
    """Determine if agent should continue attempting fixes.

    Multi-factor decision based on iteration count, error patterns, and progress.

    Args:
        ctx: Workflow context with keys:
            - fix_iterations (int): Number of fix attempts made
            - max_iterations (int): Maximum allowed iterations (default 5)
            - progress_made (bool): Whether last iteration made progress
            - test_results (dict): Current test results

    Returns:
        "continue_fixing", "escalate", or "submit_best_effort"
    """
    iterations = ctx.get("fix_iterations", 0)
    max_iter = ctx.get("max_iterations", 5)
    progress_made = ctx.get("progress_made", False)
    test_results = ctx.get("test_results", {})

    # Calculate pass rate
    passed = test_results.get("passed", 0)
    failed = test_results.get("failed", 0)
    total = passed + failed
    pass_rate = passed / total if total > 0 else 0

    # Max iterations reached
    if iterations >= max_iter:
        logger.info(f"Max fix iterations ({max_iter}) reached, submitting best effort")
        return "submit_best_effort"

    # High pass rate achieved
    if pass_rate >= 0.95:
        return "submit_best_effort"

    # Still making progress
    if progress_made and iterations < max_iter:
        return "continue_fixing"

    # No progress after several attempts
    if iterations >= 3 and not progress_made:
        return "escalate"

    return "continue_fixing"


# =============================================================================
# Transform Functions
# =============================================================================


def merge_code_analysis(ctx: dict[str, Any]) -> dict[str, Any]:
    """Merge results from parallel code analysis operations.

    Args:
        ctx: Workflow context with parallel analysis results

    Returns:
        Merged analysis results
    """
    ast_analysis = ctx.get("ast_analysis", {})
    semantic_analysis = ctx.get("semantic_analysis", {})
    dependency_analysis = ctx.get("dependency_analysis", {})

    issues = []
    issues.extend(ast_analysis.get("issues", []))
    issues.extend(semantic_analysis.get("issues", []))
    issues.extend(dependency_analysis.get("issues", []))

    return {
        "total_issues": len(issues),
        "issues": issues,
        "complexity_score": ast_analysis.get("complexity", 0),
        "dependencies": dependency_analysis.get("dependencies", []),
        "symbols": semantic_analysis.get("symbols", []),
    }


def format_implementation_plan(ctx: dict[str, Any]) -> dict[str, Any]:
    """Format implementation plan from analysis.

    Args:
        ctx: Workflow context with research_findings

    Returns:
        Formatted implementation plan
    """
    findings = ctx.get("research_findings", {})
    task = ctx.get("task", "")

    files_to_modify = findings.get("files_to_modify", [])
    approach = findings.get("approach", "")
    risks = findings.get("risks", [])

    steps = []
    for i, file in enumerate(files_to_modify, 1):
        steps.append(
            {
                "step": i,
                "file": file,
                "action": "modify",
            }
        )

    return {
        "task": task,
        "steps": steps,
        "approach": approach,
        "risks": risks,
        "estimated_files": len(files_to_modify),
    }


# =============================================================================
# Chat Workflow Escape Hatches (Phase 2)
# =============================================================================


def chat_task_complexity(ctx: dict[str, Any]) -> str:
    """Determine task complexity for chat workflow routing.

    Args:
        ctx: Workflow context with keys:
            - task_complexity (str): Complexity from extract_requirements
            - user_message (str): Original user message
            - required_files (list): Files required for the task

    Returns:
        "complex", "moderate", or "simple"
    """
    # First check if complexity was already determined
    complexity = ctx.get("task_complexity", "").lower()
    if complexity in ["complex", "moderate", "simple"]:
        return complexity

    # Fall back to analysis from user message
    user_message = ctx.get("user_message", "")
    required_files = ctx.get("required_files", [])

    # Check message length and keywords
    message_lower = user_message.lower()
    file_count = len(required_files) if isinstance(required_files, list) else 0

    # Complex task indicators
    complex_keywords = [
        "refactor",
        "redesign",
        "architecture",
        "migrate",
        "implement",
        "create",
        "build",
        "design",
        "system",
        "framework",
    ]
    has_complex_keyword = any(kw in message_lower for kw in complex_keywords)

    # Simple task indicators
    simple_keywords = [
        "what is",
        "how do",
        "explain",
        "show me",
        "quick",
        "simple",
        "trivial",
        "straightforward",
    ]
    has_simple_keyword = any(kw in message_lower for kw in simple_keywords)

    # Determine complexity
    if has_complex_keyword or file_count > 5 or len(message_lower) > 300:
        return "complex"
    elif has_simple_keyword and file_count <= 2 and len(message_lower) < 150:
        return "simple"
    else:
        return "moderate"


def has_pending_tool_calls(ctx: dict[str, Any]) -> str:
    """Check if there are pending tool calls to execute.

    Args:
        ctx: Workflow context with keys:
            - tool_calls (list): Tool calls from agent response
            - pending_tool_calls (list): Pending tool calls

    Returns:
        "has_tools" or "no_tools"
    """
    tool_calls = ctx.get("tool_calls", [])
    pending = ctx.get("pending_tool_calls", [])

    if tool_calls or pending:
        return "has_tools"
    return "no_tools"


def can_continue_iteration(ctx: dict[str, Any]) -> str:
    """Check if workflow can continue with another iteration.

    Args:
        ctx: Workflow context with keys:
            - iteration_count (int): Current iteration count
            - max_iterations (int): Maximum allowed iterations

    Returns:
        "continue" or "max_reached"
    """
    iteration = ctx.get("iteration_count", 0)
    max_iter = ctx.get("max_iterations", 50)

    if iteration < max_iter:
        return "continue"
    return "max_reached"


def update_conversation_with_tool_results(ctx: dict[str, Any]) -> dict[str, Any]:
    """Update conversation with tool results from execution.

    Args:
        ctx: Workflow context with tool execution results

    Returns:
        Updated conversation state
    """
    conversation_history = ctx.get("conversation_history", [])
    content = ctx.get("content", "")
    tool_calls = ctx.get("tool_calls", [])
    tool_results = ctx.get("tool_results", [])

    # Add assistant message with tool calls
    if tool_calls:
        conversation_history.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
        )

        # Add tool results
        for result in tool_results:
            conversation_history.append(
                {
                    "role": "tool",
                    "content": str(result),
                }
            )

    # Increment iteration count
    iteration_count = ctx.get("iteration_count", 0) + 1

    return {
        "conversation_history": conversation_history,
        "iteration_count": iteration_count,
        "last_content": content,
    }


def format_coding_response(ctx: dict[str, Any]) -> dict[str, Any]:
    """Format final coding response for user.

    Args:
        ctx: Workflow context with all conversation data

    Returns:
        Formatted final response
    """
    content = ctx.get("content", "")
    iteration_count = ctx.get("iteration_count", 0)
    conversation_history = ctx.get("conversation_history", [])

    # Extract files modified from tool calls
    files_modified = []
    for msg in conversation_history:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tool_call in msg.get("tool_calls", []):
                if tool_call.get("name") in ["write", "edit", "write_file", "edit_files"]:
                    args = tool_call.get("arguments", {})
                    if "path" in args:
                        files_modified.append(args["path"])

    final_response = content
    if not final_response:
        final_response = "Task completed."

    return {
        "final_response": final_response,
        "status": "completed",
        "iterations": iteration_count,
        "files_modified": files_modified,
        "metadata": {
            "total_iterations": iteration_count,
            "files_touched": len(files_modified),
        },
    }


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    # Original conditions
    "tests_passing": tests_passing,
    "code_quality_check": code_quality_check,
    "should_retry_implementation": should_retry_implementation,
    "review_verdict": review_verdict,
    "complexity_assessment": complexity_assessment,
    "complexity_check": complexity_check,
    "tdd_cycle_status": tdd_cycle_status,
    "bugfix_priority": bugfix_priority,
    "should_continue_fixing": should_continue_fixing,
    # Chat workflow conditions (Phase 2)
    "chat_task_complexity": chat_task_complexity,
    "has_pending_tool_calls": has_pending_tool_calls,
    "can_continue_iteration": can_continue_iteration,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    # Original transforms
    "merge_code_analysis": merge_code_analysis,
    "format_implementation_plan": format_implementation_plan,
    # Chat workflow transforms (Phase 2)
    "update_conversation_with_tool_results": update_conversation_with_tool_results,
    "format_coding_response": format_coding_response,
}

__all__ = [
    # Original Conditions
    "tests_passing",
    "code_quality_check",
    "should_retry_implementation",
    "review_verdict",
    "complexity_assessment",
    "complexity_check",
    "tdd_cycle_status",
    "bugfix_priority",
    "should_continue_fixing",
    # Chat workflow conditions (Phase 2)
    "chat_task_complexity",
    "has_pending_tool_calls",
    "can_continue_iteration",
    # Original Transforms
    "merge_code_analysis",
    "format_implementation_plan",
    # Chat workflow transforms (Phase 2)
    "update_conversation_with_tool_results",
    "format_coding_response",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
