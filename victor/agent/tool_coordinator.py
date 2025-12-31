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

"""Tool Coordinator - Coordinates tool selection, budgeting, and execution.

This module extracts tool-related coordination logic from AgentOrchestrator,
providing a focused interface for:
- Tool selection (semantic + keyword-based)
- Tool budget management
- Tool caching coordination
- Tool execution coordination via ToolPipeline

Design Philosophy:
- Single Responsibility: Coordinates all tool-related operations
- Composable: Works with existing ToolPipeline, ToolSelector, etc.
- Observable: Provides budget/execution metrics
- Backward Compatible: Maintains API compatibility with orchestrator

Usage:
    coordinator = ToolCoordinator(
        tool_pipeline=pipeline,
        tool_selector=selector,
        budget_manager=budget_mgr,
    )

    # Select tools for current context
    tools = await coordinator.select_tools(context)

    # Check budget
    remaining = coordinator.get_remaining_budget()

    # Execute tool calls
    result = await coordinator.execute_tool_calls(tool_calls)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.tool_pipeline import ToolPipeline, PipelineExecutionResult
    from victor.agent.tool_selection import ToolSelector
    from victor.agent.budget_manager import BudgetManager
    from victor.storage.cache.tool_cache import ToolCache
    from victor.tools.base import BaseTool, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class TaskContext:
    """Context for tool selection and execution.

    Attributes:
        message: The user's current message/query
        task_type: Detected task type (e.g., "edit", "analyze", "debug")
        complexity: Task complexity level (e.g., "simple", "medium", "complex")
        goals: Detected goals or intents
        stage: Current conversation stage
        observed_files: Files already observed in this session
        executed_tools: Tools already executed in this session
    """

    message: str
    task_type: str = "unknown"
    complexity: str = "medium"
    goals: Optional[Any] = None
    stage: Optional[str] = None
    observed_files: Set[str] = field(default_factory=set)
    executed_tools: Set[str] = field(default_factory=set)
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCoordinatorConfig:
    """Configuration for ToolCoordinator.

    Attributes:
        default_budget: Default tool call budget
        budget_multiplier: Multiplier for budget based on complexity
        enable_caching: Whether to use tool result caching
        enable_sequence_tracking: Whether to track tool sequences
        max_tools_per_selection: Maximum tools to select per turn
        selection_threshold: Minimum similarity threshold for tool selection
    """

    default_budget: int = 25
    budget_multiplier: float = 1.0
    enable_caching: bool = True
    enable_sequence_tracking: bool = True
    max_tools_per_selection: int = 15
    selection_threshold: float = 0.3


@runtime_checkable
class IToolCoordinator(Protocol):
    """Protocol for tool coordination operations."""

    async def select_tools(self, context: TaskContext) -> List[Any]: ...
    def get_remaining_budget(self) -> int: ...
    def consume_budget(self, amount: int) -> None: ...
    async def execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], context: Optional[TaskContext] = None
    ) -> Any: ...


class ToolCoordinator:
    """Coordinates tool selection, budgeting, and execution.

    This class consolidates tool-related operations that were spread across
    the orchestrator, providing a unified interface for:

    1. Tool Selection: Combines semantic and keyword-based selection
    2. Budget Management: Tracks and enforces tool call budgets
    3. Caching: Coordinates tool result caching
    4. Execution: Delegates to ToolPipeline for actual execution

    Example:
        coordinator = ToolCoordinator(
            tool_pipeline=pipeline,
            tool_selector=selector,
            config=ToolCoordinatorConfig(default_budget=30),
        )

        # In orchestrator:
        context = TaskContext(message=user_query, task_type="edit")
        tools = await coordinator.select_tools(context)

        # Check if we can execute more tools
        if coordinator.get_remaining_budget() > 0:
            result = await coordinator.execute_tool_calls(tool_calls)
    """

    def __init__(
        self,
        tool_pipeline: "ToolPipeline",
        tool_selector: Optional["ToolSelector"] = None,
        budget_manager: Optional["BudgetManager"] = None,
        tool_cache: Optional["ToolCache"] = None,
        config: Optional[ToolCoordinatorConfig] = None,
        on_selection_complete: Optional[Callable[[str, int], None]] = None,  # method, count
        on_budget_warning: Optional[Callable[[int, int], None]] = None,  # remaining, total
    ) -> None:
        """Initialize the ToolCoordinator.

        Args:
            tool_pipeline: Pipeline for tool execution
            tool_selector: Optional selector for tool selection
            budget_manager: Optional manager for budget tracking
            tool_cache: Optional cache for tool results
            config: Configuration options
            on_selection_complete: Callback when tool selection completes
            on_budget_warning: Callback when budget is running low
        """
        self._pipeline = tool_pipeline
        self._selector = tool_selector
        self._budget_manager = budget_manager
        self._cache = tool_cache
        self._config = config or ToolCoordinatorConfig()

        # Callbacks
        self._on_selection_complete = on_selection_complete
        self._on_budget_warning = on_budget_warning

        # Internal state
        self._budget_used: int = 0
        self._total_budget: int = self._config.default_budget
        self._selection_history: List[Tuple[str, int]] = []  # (method, count)
        self._execution_count: int = 0

        logger.debug(
            f"ToolCoordinator initialized with budget={self._total_budget}, "
            f"caching={self._config.enable_caching}"
        )

    @property
    def budget(self) -> int:
        """Get the total tool budget."""
        return self._total_budget

    @budget.setter
    def budget(self, value: int) -> None:
        """Set the total tool budget."""
        self._total_budget = max(0, value)

    @property
    def budget_used(self) -> int:
        """Get the number of budget units used."""
        return self._budget_used

    @property
    def execution_count(self) -> int:
        """Get the total number of tool executions."""
        return self._execution_count

    def get_remaining_budget(self) -> int:
        """Get remaining tool call budget.

        Returns:
            Number of tool calls remaining
        """
        if self._budget_manager:
            return self._budget_manager.get_remaining_tool_calls()
        return max(0, self._total_budget - self._budget_used)

    def consume_budget(self, amount: int = 1) -> None:
        """Consume tool call budget.

        Args:
            amount: Number of budget units to consume
        """
        self._budget_used += amount

        if self._budget_manager:
            self._budget_manager.consume_tool_call(amount)

        remaining = self.get_remaining_budget()
        total = self._total_budget

        # Warn when budget is low (less than 20% remaining)
        if remaining < total * 0.2 and self._on_budget_warning:
            self._on_budget_warning(remaining, total)

        logger.debug(f"Budget consumed: {amount}, remaining: {remaining}/{total}")

    def reset_budget(self, new_budget: Optional[int] = None) -> None:
        """Reset the tool budget.

        Args:
            new_budget: New budget to set, or use default
        """
        self._budget_used = 0
        if new_budget is not None:
            self._total_budget = new_budget
        else:
            self._total_budget = self._config.default_budget

        if self._budget_manager:
            self._budget_manager.reset(new_budget or self._config.default_budget)

        logger.debug(f"Budget reset to {self._total_budget}")

    def set_budget_multiplier(self, multiplier: float) -> None:
        """Set budget multiplier for complexity adjustments.

        Args:
            multiplier: Multiplier to apply (e.g., 2.0 for complex tasks)
        """
        self._config.budget_multiplier = multiplier
        effective_budget = int(self._config.default_budget * multiplier)
        self._total_budget = effective_budget

        logger.debug(f"Budget multiplier set to {multiplier}, effective budget: {effective_budget}")

    async def select_tools(
        self,
        context: TaskContext,
        max_tools: Optional[int] = None,
    ) -> List[Any]:
        """Select appropriate tools for the current context.

        Uses the configured ToolSelector to determine which tools are
        relevant for the current task.

        Args:
            context: Task context for selection
            max_tools: Maximum tools to select (overrides config)

        Returns:
            List of selected tool definitions
        """
        if not self._selector:
            logger.warning("No tool selector configured, returning empty list")
            return []

        max_count = max_tools or self._config.max_tools_per_selection
        threshold = self._config.selection_threshold

        try:
            # Use selector's async select_tools method
            tools = await self._selector.select_tools(
                message=context.message,
                task_type=context.task_type,
                max_tools=max_count,
                threshold=threshold,
                context=context.additional_context,
            )

            # Track selection
            method = getattr(self._selector, "last_selection_method", "unknown")
            self._selection_history.append((method, len(tools)))

            if self._on_selection_complete:
                self._on_selection_complete(method, len(tools))

            logger.debug(
                f"Selected {len(tools)} tools via {method} for task_type={context.task_type}"
            )
            return tools

        except Exception as e:
            logger.warning(f"Tool selection failed: {e}, returning empty list")
            return []

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[TaskContext] = None,
    ) -> "PipelineExecutionResult":
        """Execute tool calls through the pipeline.

        Delegates to ToolPipeline for actual execution, handling budget
        tracking and caching coordination.

        Args:
            tool_calls: List of tool calls to execute
            context: Optional task context

        Returns:
            PipelineExecutionResult with execution details
        """
        # Check budget before execution
        remaining = self.get_remaining_budget()
        call_count = len(tool_calls)

        if remaining < call_count:
            logger.warning(f"Insufficient budget: {remaining} remaining, {call_count} requested")
            # Let pipeline handle partial execution based on budget

        # Build execution context
        execution_context = {}
        if context:
            execution_context = {
                "task_type": context.task_type,
                "complexity": context.complexity,
                "observed_files": context.observed_files,
                "executed_tools": context.executed_tools,
            }

        # Execute through pipeline
        result = await self._pipeline.execute_tool_calls(
            tool_calls=tool_calls,
            context=execution_context,
        )

        # Update tracking
        successful = result.successful_calls if hasattr(result, "successful_calls") else 0
        self._execution_count += successful
        self.consume_budget(successful)

        logger.debug(
            f"Executed {successful}/{call_count} tool calls, "
            f"budget remaining: {self.get_remaining_budget()}"
        )

        return result

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get tool selection statistics.

        Returns:
            Dict with selection method distribution and counts
        """
        method_counts: Dict[str, int] = {}
        total_selected = 0

        for method, count in self._selection_history:
            method_counts[method] = method_counts.get(method, 0) + 1
            total_selected += count

        return {
            "total_selections": len(self._selection_history),
            "total_tools_selected": total_selected,
            "method_distribution": method_counts,
            "avg_tools_per_selection": (
                total_selected / len(self._selection_history) if self._selection_history else 0
            ),
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics.

        Returns:
            Dict with execution counts and budget usage
        """
        return {
            "total_executions": self._execution_count,
            "budget_used": self._budget_used,
            "budget_total": self._total_budget,
            "budget_remaining": self.get_remaining_budget(),
            "budget_utilization": (
                self._budget_used / self._total_budget if self._total_budget > 0 else 0
            ),
        }

    def clear_selection_history(self) -> None:
        """Clear the selection history."""
        self._selection_history.clear()

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted.

        Returns:
            True if no budget remaining
        """
        return self.get_remaining_budget() <= 0


def create_tool_coordinator(
    tool_pipeline: "ToolPipeline",
    tool_selector: Optional["ToolSelector"] = None,
    budget_manager: Optional["BudgetManager"] = None,
    config: Optional[ToolCoordinatorConfig] = None,
) -> ToolCoordinator:
    """Factory function to create a ToolCoordinator.

    Args:
        tool_pipeline: Pipeline for tool execution
        tool_selector: Optional selector for tool selection
        budget_manager: Optional manager for budget tracking
        config: Configuration options

    Returns:
        Configured ToolCoordinator instance
    """
    return ToolCoordinator(
        tool_pipeline=tool_pipeline,
        tool_selector=tool_selector,
        budget_manager=budget_manager,
        config=config,
    )


__all__ = [
    "ToolCoordinator",
    "ToolCoordinatorConfig",
    "TaskContext",
    "IToolCoordinator",
    "create_tool_coordinator",
]
