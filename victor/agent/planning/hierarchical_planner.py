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

"""Hierarchical task decomposition and planning engine.

This module provides the HierarchicalPlanner for advanced agentic behavior:
- Decomposes complex tasks into hierarchical subtasks
- Maintains dependency tracking between tasks
- Supports automatic re-planning based on execution feedback
- Estimates complexity for task prioritization
- Validates plans for correctness

The planner uses LLM-based decomposition to break down high-level goals
into executable steps with explicit dependencies.

Design Principles:
- SOLID compliance: Single responsibility for planning operations
- Protocol-based dependencies for testability
- Event-driven updates for observability
- Cache-friendly for performance

Example:
    from victor.agent.planning import HierarchicalPlanner

    planner = HierarchicalPlanner(orchestrator=orchestrator)

    # Decompose task
    graph = await planner.decompose_task("Implement user authentication")

    # Update plan after execution
    updated = await planner.update_plan(graph, completed_tasks=["task_1"])

    # Get next tasks
    next_tasks = await planner.suggest_next_tasks(graph)

    # Validate plan
    result = planner.validate_plan(graph)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.agent.planning.task_decomposition import (
    ComplexityScore,
    Task,
    TaskDecomposition,
    TaskGraph,
    UpdatedPlan,
    ValidationResult,
)

if TYPE_CHECKING:
    from victor.agent.protocols import ProviderManagerProtocol
    from victor.core.events import IEventBackend

logger = logging.getLogger(__name__)

# Default decomposition system prompt
DECOMPOSITION_SYSTEM_PROMPT = """You are an expert task planner. Your job is to break down complex tasks into clear, executable subtasks.

When given a complex task, analyze it and produce a hierarchical decomposition following these guidelines:

1. **Identify Major Phases**: Break the task into 3-7 major phases or milestones
2. **Define Dependencies**: Identify which phases must complete before others can start
3. **Estimate Complexity**: Rate each subtask on complexity (1-10 scale)
4. **Be Specific**: Each subtask should be clear and actionable

Output Format:
Return a JSON object with this structure:
{
  "root_task": "Brief description of the overall goal",
  "subtasks": [
    {
      "id": "task_1",
      "description": "Clear, actionable description",
      "depends_on": [],
      "estimated_complexity": 5,
      "context": {}
    },
    ...
  ]
}

Guidelines:
- Start with task_1, task_2, etc. for IDs
- For depends_on, use task IDs that must complete FIRST
- Complexity: 1-3 (simple), 4-6 (medium), 7-10 (complex)
- Keep descriptions concise but clear
- Include relevant context metadata when helpful

Example:
Task: "Implement user authentication"

{
  "root_task": "Implement secure user authentication system",
  "subtasks": [
    {
      "id": "task_1",
      "description": "Research existing authentication patterns in codebase",
      "depends_on": [],
      "estimated_complexity": 3,
      "context": {"type": "research", "tools": ["read_file", "search"]}
    },
    {
      "id": "task_2",
      "description": "Design authentication flow and data models",
      "depends_on": ["task_1"],
      "estimated_complexity": 6,
      "context": {"type": "design"}
    },
    {
      "id": "task_3",
      "description": "Implement authentication backend logic",
      "depends_on": ["task_2"],
      "estimated_complexity": 8,
      "context": {"type": "implementation"}
    },
    {
      "id": "task_4",
      "description": "Add login/logout endpoints",
      "depends_on": ["task_3"],
      "estimated_complexity": 5,
      "context": {"type": "implementation"}
    },
    {
      "id": "task_5",
      "description": "Write tests for authentication",
      "depends_on": ["task_3", "task_4"],
      "estimated_complexity": 6,
      "context": {"type": "testing"}
    }
  ]
}
"""


class HierarchicalPlanner:
    """Hierarchical task decomposition and planning engine.

    This planner breaks down complex tasks into executable subtasks with
    explicit dependencies, tracks execution progress, and supports
    dynamic re-planning.

    Key Features:
    - LLM-based task decomposition
    - Dependency-aware task ordering
    - Complexity estimation for prioritization
    - Automatic re-planning on failure
    - Plan validation

    Example:
        planner = HierarchicalPlanner(
            orchestrator=orchestrator,
            event_bus=event_bus
        )

        # Decompose complex task
        graph = await planner.decompose_task(
            "Implement user authentication with JWT"
        )

        # Get next tasks to execute
        ready = await planner.suggest_next_tasks(graph)

        # Update after execution
        updated = await planner.update_plan(
            graph,
            completed_tasks=["task_1", "task_2"]
        )
    """

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        provider_manager: Optional["ProviderManagerProtocol"] = None,
        event_bus: Optional["IEventBackend"] = None,
        decomposition_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the hierarchical planner.

        Args:
            orchestrator: Optional agent orchestrator for LLM calls
            provider_manager: Optional provider manager for LLM calls
            event_bus: Optional event backend for publishing events
            decomposition_prompt: Custom system prompt for decomposition
        """
        self._orchestrator = orchestrator
        self._provider_manager = provider_manager
        self._event_bus = event_bus
        self._decomposition_prompt = decomposition_prompt or DECOMPOSITION_SYSTEM_PROMPT

        # Internal decomposition service (no event_bus in new implementation)
        self._decomposer = TaskDecomposition()

        # Cache for decomposition results
        self._decomposition_cache: Dict[str, TaskGraph] = {}

    async def _emit_event(
        self, event_type: str, data: Dict[str, Any]
    ) -> None:
        """Emit a planning event.

        Args:
            event_type: Type of event (e.g., "decompose", "update_plan")
            data: Event data
        """
        if self._event_bus:
            try:
                from victor.core.events import MessagingEvent

                await self._event_bus.publish(
                    MessagingEvent(topic=f"planning.{event_type}", data=data)
                )
            except Exception as e:
                logger.debug(f"Failed to emit planning event: {e}")

    def _get_task_hash(self, task: str) -> str:
        """Get hash of task description for caching.

        Args:
            task: Task description

        Returns:
            SHA256 hash of task
        """
        return hashlib.sha256(task.encode()).hexdigest()

    async def _call_llm_for_decomposition(
        self, complex_task: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Call LLM to decompose a task.

        Args:
            complex_task: Complex task description
            context: Optional context for decomposition

        Returns:
            JSON string with task decomposition

        Raises:
            RuntimeError: If LLM call fails
        """
        # Build prompt
        user_message = f"Decompose this task into subtasks:\n\n{complex_task}"
        if context:
            user_message += f"\n\nContext:\n{json.dumps(context, indent=2)}"

        # Try to use orchestrator first
        if self._orchestrator:
            try:
                response = await self._orchestrator._provider_manager.chat(
                    messages=[
                        {"role": "system", "content": self._decomposition_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
                return response.content
            except Exception as e:
                logger.warning(f"Orchestrator LLM call failed: {e}")

        # Fall back to provider manager
        if self._provider_manager:
            try:
                response = await self._provider_manager.chat(
                    messages=[
                        {"role": "system", "content": self._decomposition_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
                return response.content
            except Exception as e:
                logger.error(f"Provider manager LLM call failed: {e}")
                raise RuntimeError(f"Failed to decompose task: {e}") from e

        raise RuntimeError("No LLM provider available for task decomposition")

    def _parse_decomposition_response(
        self, response: str, root_task: str
    ) -> List[Task]:
        """Parse LLM decomposition response into tasks.

        Args:
            response: LLM response string
            root_task: Original task description

        Returns:
            List of Task objects

        Raises:
            ValueError: If response is invalid JSON
        """
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            raise ValueError("No JSON found in LLM response")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}") from e

        # Validate structure
        if "subtasks" not in data:
            raise ValueError("Missing 'subtasks' field in decomposition")

        # Parse tasks
        tasks = []
        for i, task_data in enumerate(data["subtasks"], 1):
            task = Task(
                id=task_data.get("id", f"task_{i}"),
                description=task_data.get("description", ""),
                depends_on=task_data.get("depends_on", []),
                estimated_complexity=task_data.get("estimated_complexity", 5),
                context=task_data.get("context", {}),
            )
            tasks.append(task)

        return tasks

    async def decompose_task(
        self,
        complex_task: str,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> TaskGraph:
        """Decompose a complex task into a hierarchical execution graph.

        Uses LLM to break down the task into subtasks with dependencies.
        Results are cached by task hash for performance.

        Args:
            complex_task: Complex task description to decompose
            context: Optional context for decomposition (e.g., files, constraints)
            use_cache: Whether to use cached decomposition if available

        Returns:
            TaskGraph with decomposed tasks

        Raises:
            RuntimeError: If LLM decomposition fails
            ValueError: If LLM response is invalid
        """
        # Check cache
        task_hash = self._get_task_hash(complex_task)
        if use_cache and task_hash in self._decomposition_cache:
            logger.debug(f"Using cached decomposition for task: {complex_task[:50]}...")
            await self._emit_event(
                "decompose_cache_hit",
                {"task": complex_task, "hash": task_hash},
            )
            return self._decomposition_cache[task_hash]

        # Call LLM for decomposition
        await self._emit_event(
            "decompose_start",
            {"task": complex_task, "context": context},
        )

        try:
            response = await self._call_llm_for_decomposition(complex_task, context)
            tasks = self._parse_decomposition_response(response, complex_task)
            graph = self._decomposer.to_execution_graph(tasks)

            # Cache result
            self._decomposition_cache[task_hash] = graph

            await self._emit_event(
                "decompose_complete",
                {
                    "task": complex_task,
                    "hash": task_hash,
                    "num_tasks": len(tasks),
                },
            )

            return graph

        except Exception as e:
            await self._emit_event(
                "decompose_failed",
                {"task": complex_task, "error": str(e)},
            )
            raise

    async def update_plan(
        self,
        task_graph: TaskGraph,
        completed_tasks: List[str],
        failed_tasks: Optional[List[str]] = None,
    ) -> UpdatedPlan:
        """Update a task graph after execution progress.

        Marks tasks as completed/failed and identifies new ready tasks.

        Args:
            task_graph: Current task graph
            completed_tasks: List of task IDs that completed
            failed_tasks: Optional list of task IDs that failed

        Returns:
            UpdatedPlan with new ready tasks and status

        Raises:
            ValueError: If a task ID doesn't exist
        """
        failed_tasks = failed_tasks or []

        # Validate task IDs exist
        for task_id in completed_tasks + failed_tasks:
            if task_id not in task_graph.nodes:
                raise ValueError(f"Task '{task_id}' not found in graph")

        # Update task statuses
        for task_id in completed_tasks:
            task_graph.nodes[task_id].status = "completed"

        for task_id in failed_tasks:
            task_graph.nodes[task_id].status = "failed"

            # Mark dependent tasks as blocked
            for dependent_id in task_graph.get_dependents(task_id):
                if task_graph.nodes[dependent_id].status == "pending":
                    task_graph.nodes[dependent_id].status = "blocked"

        # Get newly ready tasks
        new_ready_tasks = self._decomposer.get_ready_tasks(task_graph)

        # Check if we can proceed
        can_proceed = len(new_ready_tasks) > 0 or any(
            t.status == "in_progress" for t in task_graph.nodes.values()
        )

        result = UpdatedPlan(
            graph=task_graph,
            new_ready_tasks=new_ready_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            can_proceed=can_proceed,
        )

        await self._emit_event(
            "update_plan",
            {
                "completed": completed_tasks,
                "failed": failed_tasks,
                "new_ready": [t.id for t in new_ready_tasks],
                "can_proceed": can_proceed,
            },
        )

        return result

    async def suggest_next_tasks(
        self,
        task_graph: TaskGraph,
    ) -> List[Task]:
        """Suggest next tasks to execute based on graph state.

        Returns ready tasks sorted by priority (complexity, dependencies).

        Args:
            task_graph: Current task graph

        Returns:
            List of tasks ready to execute, ordered by priority
        """
        ready_tasks = self._decomposer.get_ready_tasks(task_graph)

        await self._emit_event(
            "suggest_tasks",
            {
                "ready_count": len(ready_tasks),
                "task_ids": [t.id for t in ready_tasks],
            },
        )

        return ready_tasks

    def estimate_complexity(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComplexityScore:
        """Estimate the complexity of a task.

        Uses heuristics to estimate complexity based on:
        - Task description keywords
        - Context (e.g., number of files, codebase size)
        - Common patterns

        Args:
            task: Task description
            context: Optional context for estimation

        Returns:
            ComplexityScore with estimation
        """
        factors = []
        base_score = 5.0

        # Keyword-based complexity adjustment
        complex_keywords = [
            "refactor",
            "migrate",
            "restructure",
            "architecture",
            "implement",
            "design",
            "system",
        ]
        simple_keywords = [
            "list",
            "show",
            "display",
            "what",
            "where",
            "check",
        ]

        task_lower = task.lower()

        # Check for complex keywords
        complex_matches = sum(1 for kw in complex_keywords if kw in task_lower)
        if complex_matches > 0:
            adjustment = complex_matches * 0.5
            base_score += adjustment
            factors.append(f"Complex keywords: {complex_matches}")

        # Check for simple keywords
        simple_matches = sum(1 for kw in simple_keywords if kw in task_lower)
        if simple_matches > 0:
            adjustment = simple_matches * -0.3
            base_score += adjustment
            factors.append(f"Simple keywords: {simple_matches}")

        # Context-based adjustments
        if context:
            # File count
            file_count = context.get("file_count", 0)
            if file_count > 10:
                base_score += min(2.0, file_count / 20)
                factors.append(f"Many files: {file_count}")

            # Lines of code
            loc = context.get("lines_of_code", 0)
            if loc > 1000:
                base_score += min(1.5, loc / 5000)
                factors.append(f"Large codebase: {loc} LOC")

            # Multiple domains
            domains = context.get("domains", [])
            if len(domains) > 1:
                base_score += len(domains) * 0.3
                factors.append(f"Multiple domains: {domains}")

        # Clamp score to 1-10 range
        score = max(1.0, min(10.0, base_score))

        # Confidence based on factors
        confidence = 0.5 + (len(factors) * 0.1)
        confidence = min(0.9, confidence)

        # Estimate steps based on complexity
        estimated_steps = int(score * 1.5) + 2

        return ComplexityScore(
            score=score,
            confidence=confidence,
            factors=factors,
            estimated_steps=estimated_steps,
        )

    def validate_plan(self, task_graph: TaskGraph) -> ValidationResult:
        """Validate a task graph for correctness.

        Checks for:
        - Circular dependencies
        - Missing dependencies
        - Orphaned tasks
        - Invalid structure

        Args:
            task_graph: Task graph to validate

        Returns:
            ValidationResult with errors and warnings
        """
        result = self._decomposer.validate_plan(task_graph)

        # Emit validation event
        if self._event_bus:
            # Sync emit for validation (non-async method)
            try:
                import asyncio

                asyncio.create_task(
                    self._emit_event(
                        "validate_plan",
                        {
                            "is_valid": result.is_valid,
                            "errors": result.errors,
                            "warnings": result.warnings,
                            "has_cycles": result.has_cycles,
                        },
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to emit validation event: {e}")

        return result

    def clear_cache(self) -> None:
        """Clear the decomposition cache."""
        self._decomposition_cache.clear()
        logger.debug("Decomposition cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the decomposition cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_plans": len(self._decomposition_cache),
            "cache_keys": list(self._decomposition_cache.keys()),
        }


__all__ = [
    "HierarchicalPlanner",
    "DECOMPOSITION_SYSTEM_PROMPT",
]
