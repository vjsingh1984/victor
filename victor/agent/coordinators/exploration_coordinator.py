# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Parallel exploration coordinator for the agentic loop.

Spawns concurrent RESEARCHER subagents during READING stage to explore
different aspects of a codebase in parallel. Uses the existing
SubAgentOrchestrator.fan_out() infrastructure — no new execution engine.

Usage:
    coordinator = ExplorationCoordinator(provider_context)
    findings = await coordinator.explore_parallel(
        "Fix the bug in separability_matrix",
        project_root=Path("/path/to/repo"),
    )
    # findings.summary contains aggregated file paths and context
"""

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    """Aggregated results from parallel exploration subagents."""

    file_paths: List[str] = field(default_factory=list)
    summary: str = ""
    duration_seconds: float = 0.0
    subagent_count: int = 0
    tool_calls_total: int = 0


class ExplorationCoordinator:
    """Coordinates parallel file exploration during READING stage.

    When a complex task is detected, this coordinator spawns 2-3 parallel
    RESEARCHER subagents to explore the codebase concurrently:
    - One finds source files related to the issue
    - One finds test files
    - One maps project structure (optional)

    Results are aggregated and injected into the main agent's conversation
    context, giving it a head start on understanding the codebase.
    """

    def __init__(self, provider_context: Any):
        """Initialize with the parent orchestrator's provider context.

        Args:
            provider_context: The provider context from ExecutionCoordinator
                             (provides settings, provider, model, tools)
        """
        self._context = provider_context

    async def explore_parallel(
        self,
        task_description: str,
        project_root: Path,
        max_agents: int = 3,
    ) -> ExplorationResult:
        """Run parallel exploration and return aggregated findings.

        Args:
            task_description: The user's task/issue description
            project_root: Root directory of the project to explore
            max_agents: Maximum concurrent subagents (default 3)

        Returns:
            ExplorationResult with file paths, summary, and metrics
        """
        start = time.time()

        try:
            from victor.agent.subagents.orchestrator import (
                SubAgentOrchestrator,
            )
            from victor.agent.subagents.base import SubAgentRole

            # Build subagent context from provider context
            from victor.agent.subagents.protocols import SubAgentContextAdapter

            sub_context = SubAgentContextAdapter(self._context)
            orchestrator = SubAgentOrchestrator(sub_context)

            # Decompose task into parallel exploration subtasks
            tasks = self._decompose_exploration(task_description)

            logger.info(
                "Starting parallel exploration: %d subagents for '%s'",
                min(len(tasks), max_agents),
                task_description[:80],
            )

            # Fan out using existing infrastructure
            fan_out_result = await orchestrator.fan_out(
                tasks[: max_agents],
                max_concurrent=max_agents,
            )

            result = self._aggregate(fan_out_result)
            result.duration_seconds = time.time() - start

            logger.info(
                "Parallel exploration complete: %d files found, %d tool calls, %.1fs",
                len(result.file_paths),
                result.tool_calls_total,
                result.duration_seconds,
            )

            return result

        except Exception as e:
            logger.warning("Parallel exploration failed: %s", e)
            return ExplorationResult(duration_seconds=time.time() - start)

    def _decompose_exploration(self, task: str) -> list:
        """Decompose task into parallel exploration subtasks.

        Generates 2-3 scoped research tasks from the user's task description.
        Each subtask gets a focused prompt and read-only tool access.
        """
        from victor.agent.subagents.orchestrator import SubAgentTask
        from victor.agent.subagents.base import SubAgentRole

        # Extract key terms from task for targeted search
        task_excerpt = task[:300]

        tasks = [
            SubAgentTask(
                role=SubAgentRole.RESEARCHER,
                task=(
                    f"Find the source code files most relevant to this issue. "
                    f"Use code_search to find key functions/classes, then read "
                    f"the most relevant file. Report file paths and key findings.\n\n"
                    f"Issue: {task_excerpt}"
                ),
                tool_budget=10,
                allowed_tools=["read", "ls", "code_search"],
            ),
            SubAgentTask(
                role=SubAgentRole.RESEARCHER,
                task=(
                    f"Find the test files related to this issue. Use ls to find "
                    f"test directories, then code_search to locate relevant tests. "
                    f"Report test file paths.\n\n"
                    f"Issue: {task_excerpt}"
                ),
                tool_budget=8,
                allowed_tools=["read", "ls", "code_search"],
            ),
        ]

        return tasks

    def _aggregate(self, fan_out_result: Any) -> ExplorationResult:
        """Extract file paths and summaries from subagent results."""
        paths: List[str] = []
        summaries: List[str] = []
        total_tools = 0

        for r in fan_out_result.results:
            total_tools += r.tool_calls_used
            if r.success and r.summary:
                summaries.append(r.summary)
                # Extract file paths from summary text
                for line in r.summary.split("\n"):
                    line = line.strip()
                    # Match paths like "astropy/modeling/separable.py"
                    path_match = re.search(
                        r"([a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,4})", line
                    )
                    if path_match:
                        path = path_match.group(1)
                        if "/" in path and path not in paths:
                            paths.append(path)

        # Deduplicate and limit
        paths = list(dict.fromkeys(paths))[:10]

        summary = "\n\n".join(summaries[:3]) if summaries else ""

        return ExplorationResult(
            file_paths=paths,
            summary=summary,
            subagent_count=len(fan_out_result.results),
            tool_calls_total=total_tools,
        )
