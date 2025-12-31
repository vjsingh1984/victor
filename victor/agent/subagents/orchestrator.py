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

"""Sub-agent orchestrator for spawning and managing sub-agents.

This module provides the SubAgentOrchestrator class that enables:
- Spawning sub-agents with role-specific configurations
- Parallel execution of multiple sub-agents (fan-out)
- Sequential task delegation
- Result aggregation

Example:
    orchestrator = SubAgentOrchestrator(parent_orchestrator)

    # Spawn a single sub-agent
    result = await orchestrator.spawn(
        SubAgentRole.RESEARCHER,
        "Find all API endpoints in the codebase",
        tool_budget=15,
    )

    # Fan out to multiple sub-agents in parallel
    results = await orchestrator.fan_out([
        SubAgentTask(SubAgentRole.RESEARCHER, "Research auth patterns"),
        SubAgentTask(SubAgentRole.RESEARCHER, "Research database patterns"),
        SubAgentTask(SubAgentRole.PLANNER, "Plan auth implementation"),
    ], max_concurrent=3)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Set

from victor.agent.subagents.base import (
    SubAgent,
    SubAgentConfig,
    SubAgentResult,
    SubAgentRole,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.providers.base import StreamChunk

logger = logging.getLogger(__name__)


# Default tool sets for each role
ROLE_DEFAULT_TOOLS: Dict[SubAgentRole, List[str]] = {
    SubAgentRole.RESEARCHER: [
        "read",
        "ls",
        "grep",
        "search",
        "code_search",
        "semantic_code_search",
        "web_search",
        "web_fetch",
    ],
    SubAgentRole.PLANNER: [
        "read",
        "ls",
        "grep",
        "search",
        "plan_files",
    ],
    SubAgentRole.EXECUTOR: [
        "read",
        "write",
        "edit",
        "ls",
        "grep",
        "search",
        "shell",
        "test",
        "git",
    ],
    SubAgentRole.REVIEWER: [
        "read",
        "ls",
        "grep",
        "search",
        "git",
        "test",
        "shell",
    ],
    SubAgentRole.TESTER: [
        "read",
        "write",
        "ls",
        "grep",
        "search",
        "test",
        "shell",
    ],
}

# Default budgets for each role
ROLE_DEFAULT_BUDGETS: Dict[SubAgentRole, int] = {
    SubAgentRole.RESEARCHER: 15,
    SubAgentRole.PLANNER: 10,
    SubAgentRole.EXECUTOR: 30,
    SubAgentRole.REVIEWER: 15,
    SubAgentRole.TESTER: 20,
}

# Default context limits for each role
ROLE_DEFAULT_CONTEXT: Dict[SubAgentRole, int] = {
    SubAgentRole.RESEARCHER: 50000,
    SubAgentRole.PLANNER: 30000,
    SubAgentRole.EXECUTOR: 80000,
    SubAgentRole.REVIEWER: 40000,
    SubAgentRole.TESTER: 50000,
}


@dataclass
class SubAgentTask:
    """Represents a task to be executed by a sub-agent.

    Used with fan_out() to specify multiple tasks for parallel execution.

    Attributes:
        role: Sub-agent role
        task: Task description
        tool_budget: Optional budget override
        allowed_tools: Optional tool list override
        context_limit: Optional context limit override
    """

    role: SubAgentRole
    task: str
    tool_budget: Optional[int] = None
    allowed_tools: Optional[List[str]] = None
    context_limit: Optional[int] = None


@dataclass
class FanOutResult:
    """Result from fan_out execution.

    Contains results from all sub-agents and aggregate statistics.

    Attributes:
        results: List of individual sub-agent results
        all_success: True if all sub-agents succeeded
        total_tool_calls: Sum of tool calls across all sub-agents
        total_duration: Total execution time (max of parallel tasks)
        errors: List of error messages from failed sub-agents
    """

    results: List[SubAgentResult]
    all_success: bool
    total_tool_calls: int
    total_duration: float
    errors: List[str] = field(default_factory=list)


class SubAgentOrchestrator:
    """Orchestrates spawning and execution of sub-agents.

    Provides high-level API for creating sub-agents with sensible defaults
    and managing parallel execution.

    Attributes:
        parent: Parent orchestrator that owns this sub-agent orchestrator
        active_subagents: Set of currently running sub-agents

    Example:
        orchestrator = SubAgentOrchestrator(parent)

        # Simple spawn with role defaults
        result = await orchestrator.spawn(
            SubAgentRole.RESEARCHER,
            "Find authentication patterns",
        )

        # Spawn with custom configuration
        result = await orchestrator.spawn(
            SubAgentRole.EXECUTOR,
            "Implement JWT authentication",
            tool_budget=40,
            allowed_tools=["read", "write", "edit", "shell"],
        )
    """

    def __init__(self, parent_orchestrator: "AgentOrchestrator"):
        """Initialize sub-agent orchestrator.

        Args:
            parent_orchestrator: Parent orchestrator that spawns sub-agents
        """
        self.parent = parent_orchestrator
        self.active_subagents: Set[SubAgent] = set()

        logger.info("SubAgentOrchestrator initialized")

    async def spawn(
        self,
        role: SubAgentRole,
        task: str,
        tool_budget: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        context_limit: Optional[int] = None,
        can_spawn_subagents: bool = False,
        timeout_seconds: int = 300,
    ) -> SubAgentResult:
        """Spawn a sub-agent to execute a task.

        Creates a sub-agent with the specified role and task, using sensible
        defaults for unspecified parameters based on the role.

        Args:
            role: Sub-agent role specialization
            task: Task description for the sub-agent
            tool_budget: Maximum tool calls (default: role-specific)
            allowed_tools: List of allowed tools (default: role-specific)
            context_limit: Maximum context size (default: role-specific)
            can_spawn_subagents: Whether sub-agent can spawn children
            timeout_seconds: Maximum execution time

        Returns:
            SubAgentResult with execution outcome

        Example:
            result = await orchestrator.spawn(
                SubAgentRole.RESEARCHER,
                "Find all database models in the codebase",
            )
            if result.success:
                print(result.summary)
        """
        # Apply role-specific defaults
        effective_budget = tool_budget or ROLE_DEFAULT_BUDGETS.get(role, 15)
        effective_tools = allowed_tools or ROLE_DEFAULT_TOOLS.get(role, ["read", "ls"])
        effective_context = context_limit or ROLE_DEFAULT_CONTEXT.get(role, 50000)

        # Create configuration
        config = SubAgentConfig(
            role=role,
            task=task,
            allowed_tools=effective_tools,
            tool_budget=effective_budget,
            context_limit=effective_context,
            can_spawn_subagents=can_spawn_subagents,
            timeout_seconds=timeout_seconds,
        )

        # Create and execute sub-agent
        subagent = SubAgent(config, self.parent)
        self.active_subagents.add(subagent)

        try:
            result = await asyncio.wait_for(
                subagent.execute(),
                timeout=timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"{role.value} sub-agent timed out after {timeout_seconds}s")
            return SubAgentResult(
                success=False,
                summary=f"Sub-agent timed out after {timeout_seconds} seconds",
                details={"role": role.value, "task": task[:200]},
                tool_calls_used=0,
                context_size=0,
                duration_seconds=float(timeout_seconds),
                error=f"Timeout after {timeout_seconds}s",
            )
        finally:
            self.active_subagents.discard(subagent)

    async def fan_out(
        self,
        tasks: List[SubAgentTask],
        max_concurrent: int = 3,
    ) -> FanOutResult:
        """Execute multiple sub-agent tasks in parallel.

        Spawns multiple sub-agents concurrently, respecting the concurrency
        limit. Results are returned in the same order as input tasks.

        Args:
            tasks: List of tasks to execute
            max_concurrent: Maximum concurrent sub-agents

        Returns:
            FanOutResult with all results and aggregate statistics

        Example:
            results = await orchestrator.fan_out([
                SubAgentTask(SubAgentRole.RESEARCHER, "Find auth code"),
                SubAgentTask(SubAgentRole.RESEARCHER, "Find database models"),
                SubAgentTask(SubAgentRole.PLANNER, "Plan refactoring"),
            ])

            for i, result in enumerate(results.results):
                print(f"Task {i}: {'Success' if result.success else 'Failed'}")
        """
        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task: SubAgentTask) -> SubAgentResult:
            async with semaphore:
                return await self.spawn(
                    role=task.role,
                    task=task.task,
                    tool_budget=task.tool_budget,
                    allowed_tools=task.allowed_tools,
                    context_limit=task.context_limit,
                )

        logger.info(
            f"Fan-out: spawning {len(tasks)} sub-agents " f"(max concurrent: {max_concurrent})"
        )

        # Execute all tasks
        results = await asyncio.gather(
            *[run_with_semaphore(task) for task in tasks],
            return_exceptions=True,
        )

        # Process results
        processed_results: List[SubAgentResult] = []
        errors: List[str] = []
        total_tool_calls = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed result
                error_msg = (
                    f"Task {i} ({tasks[i].role.value}): {type(result).__name__}: {str(result)}"
                )
                errors.append(error_msg)
                processed_results.append(
                    SubAgentResult(
                        success=False,
                        summary=f"Exception: {str(result)[:450]}",
                        details={"exception": str(result)},
                        tool_calls_used=0,
                        context_size=0,
                        duration_seconds=0,
                        error=error_msg,
                    )
                )
            else:
                processed_results.append(result)
                total_tool_calls += result.tool_calls_used
                if not result.success and result.error:
                    errors.append(f"Task {i} ({tasks[i].role.value}): {result.error}")

        total_duration = time.time() - start_time
        all_success = all(r.success for r in processed_results)

        logger.info(
            f"Fan-out complete: {sum(1 for r in processed_results if r.success)}/{len(tasks)} "
            f"succeeded, {total_tool_calls} total tool calls, {total_duration:.1f}s"
        )

        return FanOutResult(
            results=processed_results,
            all_success=all_success,
            total_tool_calls=total_tool_calls,
            total_duration=total_duration,
            errors=errors,
        )

    async def stream_spawn(
        self,
        role: SubAgentRole,
        task: str,
        *,
        tool_budget: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        context_limit: Optional[int] = None,
        can_spawn_subagents: bool = False,
        timeout_seconds: int = 300,
    ) -> AsyncIterator["StreamChunk"]:
        """Spawn a sub-agent with streaming output.

        Like spawn() but yields StreamChunk as agent generates content.
        Useful for real-time progress display during long-running tasks.

        Args:
            role: Sub-agent role specialization
            task: Task description for the sub-agent
            tool_budget: Maximum tool calls (default: role-specific)
            allowed_tools: List of allowed tools (default: role-specific)
            context_limit: Maximum context size (default: role-specific)
            can_spawn_subagents: Whether sub-agent can spawn children
            timeout_seconds: Maximum execution time

        Yields:
            StreamChunk with incremental content and tool calls

        Example:
            async for chunk in orchestrator.stream_spawn(
                SubAgentRole.RESEARCHER,
                "Find all API endpoints",
            ):
                print(chunk.content, end="", flush=True)
                if chunk.is_final:
                    print(f"\\nDone: {chunk.metadata}")
        """
        from victor.providers.base import StreamChunk

        # Apply role-specific defaults
        effective_budget = tool_budget or ROLE_DEFAULT_BUDGETS.get(role, 15)
        effective_tools = allowed_tools or ROLE_DEFAULT_TOOLS.get(role, ["read", "ls"])
        effective_context = context_limit or ROLE_DEFAULT_CONTEXT.get(role, 50000)

        # Create configuration
        config = SubAgentConfig(
            role=role,
            task=task,
            allowed_tools=effective_tools,
            tool_budget=effective_budget,
            context_limit=effective_context,
            can_spawn_subagents=can_spawn_subagents,
            timeout_seconds=timeout_seconds,
        )

        # Create sub-agent
        subagent = SubAgent(config, self.parent)
        self.active_subagents.add(subagent)

        start_time = time.time()

        try:
            # Stream with manual timeout checking per chunk
            async for chunk in subagent.stream_execute():
                # Check timeout before yielding each chunk
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    logger.warning(
                        f"{role.value} sub-agent stream timed out after {timeout_seconds}s"
                    )
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        metadata={
                            "error": f"Timeout after {timeout_seconds}s",
                            "timeout": True,
                            "role": role.value,
                            "task": task[:200],
                            "duration_seconds": float(timeout_seconds),
                            "success": False,
                        },
                    )
                    return

                yield chunk

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"{role.value} sub-agent stream spawn failed: {error_msg}", exc_info=True)
            yield StreamChunk(
                content="",
                is_final=True,
                metadata={
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "role": role.value,
                    "duration_seconds": time.time() - start_time,
                    "success": False,
                },
            )

        finally:
            self.active_subagents.discard(subagent)

    def get_active_count(self) -> int:
        """Get number of currently active sub-agents.

        Returns:
            Count of active sub-agents
        """
        return len(self.active_subagents)


__all__ = [
    "SubAgentOrchestrator",
    "SubAgentTask",
    "FanOutResult",
    "ROLE_DEFAULT_TOOLS",
    "ROLE_DEFAULT_BUDGETS",
    "ROLE_DEFAULT_CONTEXT",
]
