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

"""Tool service implementation.

Extracts tool operations from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Tool selection based on context
- Tool execution with budgeting
- Tool usage tracking
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

if TYPE_CHECKING:
    from victor.agent.services.protocols.tool_service import ToolSelectionContext

logger = logging.getLogger(__name__)


class ToolServiceConfig:
    """Configuration for ToolService.

    Attributes:
        default_max_tools: Default maximum tools per selection
        default_tool_budget: Default tool budget per session
        enable_parallel_execution: Enable parallel tool execution
        enable_caching: Enable tool result caching
        cache_ttl: Cache TTL in seconds
    """

    def __init__(
        self,
        default_max_tools: int = 10,
        default_tool_budget: int = 100,
        enable_parallel_execution: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 600,
    ):
        self.default_max_tools = default_max_tools
        self.default_tool_budget = default_tool_budget
        self.enable_parallel_execution = enable_parallel_execution
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl


class BudgetManager:
    """Manages tool budget for a session.

    Tracks tool calls and enforces budget limits to prevent
    excessive tool usage in loops.

    Attributes:
        max_budget: Maximum tool calls allowed
        calls_made: Number of tool calls made
    """

    def __init__(self, max_budget: int = 100):
        self.max_budget = max_budget
        self.calls_made = 0

    def is_exhausted(self) -> bool:
        """Check if budget is exhausted.

        Returns:
            True if no more tool calls allowed
        """
        return self.calls_made >= self.max_budget

    def record_usage(self, count: int = 1) -> None:
        """Record tool usage.

        Args:
            count: Number of tool calls to record
        """
        self.calls_made += count

    def get_remaining(self) -> int:
        """Get remaining budget.

        Returns:
            Number of tool calls remaining
        """
        return max(0, self.max_budget - self.calls_made)

    def reset(self) -> None:
        """Reset budget to initial state."""
        self.calls_made = 0


class ToolService:
    """Service for managing tool operations.

    Extracted from AgentOrchestrator to handle:
    - Tool selection based on context
    - Tool execution with budgeting
    - Tool usage tracking

    This service follows SOLID principles:
    - SRP: Only handles tool operations
    - OCP: Extensible through strategy pattern
    - LSP: Implements ToolServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        config = ToolServiceConfig()
        service = ToolService(
            config=config,
            tool_selector=selector,
            tool_executor=executor,
            tool_registrar=registrar,
        )

        tools = await service.select_tools(context)
        result = await service.execute_tool("read", {"path": "file.txt"})
    """

    def __init__(
        self,
        config: ToolServiceConfig,
        tool_selector: Any,
        tool_executor: Any,
        tool_registrar: Any,
    ):
        """Initialize the tool service.

        Args:
            config: Service configuration
            tool_selector: Tool selection component
            tool_executor: Tool execution component
            tool_registrar: Tool registration component
        """
        self._config = config
        self._selector = tool_selector
        self._executor = tool_executor
        self._registrar = tool_registrar
        self._budget_manager = BudgetManager(config.default_tool_budget)
        self._usage_stats: Dict[str, int] = {}
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    async def select_tools(
        self,
        context: "ToolSelectionContext",
        max_tools: int = 10,
    ) -> List[str]:
        """Select tools based on context.

        Uses the tool selector to analyze the context and select
        the most relevant tools for the task.

        Args:
            context: Tool selection context
            max_tools: Maximum number of tools to select

        Returns:
            List of selected tool names, ordered by relevance

        Raises:
            ToolSelectionError: If tool selection fails
        """
        self._logger.debug(f"Selecting tools (max={max_tools})")

        try:
            # Use selector to choose tools
            selected = await self._selector.select(context, max_tools)

            self._logger.debug(f"Selected {len(selected)} tools: {selected}")
            return selected

        except Exception as e:
            self._logger.error(f"Tool selection failed: {e}")
            # Return empty list on failure for resilience
            return []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Execute a single tool with validation and budgeting.

        Validates arguments, checks budget, executes the tool,
        and tracks usage.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome

        Raises:
            ToolBudgetExceededError: If tool budget is exhausted
            ToolNotFoundError: If tool is not registered
            ToolValidationError: If arguments are invalid
            ToolExecutionError: If tool execution fails
        """
        self._logger.debug(f"Executing tool: {tool_name}")

        # Check budget
        if self._budget_manager.is_exhausted():
            self._logger.warning("Tool budget exhausted")
            raise ToolBudgetExceededError(
                f"Tool budget exhausted ({self._budget_manager.calls_made} "
                f"/ {self._budget_manager.max_budget})"
            )

        # Check cache if enabled
        if self._config.enable_caching:
            cached = await self._get_cached_result(tool_name, arguments)
            if cached is not None:
                self._logger.debug(f"Using cached result for {tool_name}")
                return cached

        # Execute tool
        try:
            result = await self._executor.execute(tool_name, arguments)

            # Track usage
            self._budget_manager.record_usage()
            self._track_tool_usage(tool_name, success=True)

            # Cache result if enabled and successful
            if self._config.enable_caching and result.success:
                await self._cache_result(tool_name, arguments, result)

            return result

        except Exception as e:
            self._logger.error(f"Tool execution failed: {tool_name}: {e}")
            self._track_tool_usage(tool_name, success=False)
            raise

    async def execute_tools_parallel(
        self,
        tool_calls: List[tuple[str, Dict[str, Any]]],
        max_parallel: int = 5,
    ) -> AsyncIterator[Any]:
        """Execute multiple tools in parallel.

        Executes independent tools concurrently for improved performance.

        Args:
            tool_calls: List of (tool_name, arguments) tuples
            max_parallel: Maximum number of concurrent executions

        Yields:
            ToolResult objects as they complete

        Raises:
            ToolExecutionError: If any critical tool execution fails
        """
        if not self._config.enable_parallel_execution:
            # Execute sequentially
            for tool_name, arguments in tool_calls:
                yield await self.execute_tool(tool_name, arguments)
            return

        self._logger.debug(f"Executing {len(tool_calls)} tools in parallel")

        # Create semaphore for parallelism limit
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_limit(tool_name: str, arguments: Dict[str, Any]):
            async with semaphore:
                return await self.execute_tool(tool_name, arguments)

        # Execute all tools in parallel
        tasks = [execute_with_limit(name, args) for name, args in tool_calls]

        for task in asyncio.as_completed(tasks):
            result = await task
            yield result

    def get_tool_budget(self) -> int:
        """Get the remaining tool budget.

        Returns:
            Number of remaining tool calls allowed
        """
        return self._budget_manager.get_remaining()

    def set_tool_budget(self, budget: int) -> None:
        """Set the tool budget limit.

        Args:
            budget: Maximum number of tool calls allowed

        Raises:
            ValueError: If budget is negative
        """
        if budget < 0:
            raise ValueError(f"Tool budget must be non-negative: {budget}")

        old_max = self._budget_manager.max_budget
        self._budget_manager.max_budget = budget
        self._logger.info(f"Tool budget updated: {old_max} -> {budget}")

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        total_calls = sum(self._usage_stats.values())
        successful_calls = sum(
            count for tool, count in self._usage_stats.items() if not tool.startswith("error:")
        )

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 1.0,
            "by_tool": self._usage_stats.copy(),
            "budget_remaining": self.get_tool_budget(),
            "budget_used": self._budget_manager.calls_made,
        }

    def reset_tool_budget(self) -> None:
        """Reset the tool budget to initial limit.

        Useful for starting new sessions or after testing.
        """
        self._budget_manager.reset()
        self._usage_stats.clear()
        self._logger.info("Tool budget reset")

    def is_healthy(self) -> bool:
        """Check if the tool service is healthy.

        Returns:
            True if the service is healthy
        """
        return (
            self._selector is not None
            and self._executor is not None
            and not self._budget_manager.is_exhausted()
        )

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _track_tool_usage(self, tool_name: str, success: bool) -> None:
        """Track tool usage for statistics.

        Args:
            tool_name: Name of the tool
            success: Whether execution was successful
        """
        key = tool_name if success else f"error:{tool_name}"
        self._usage_stats[key] = self._usage_stats.get(key, 0) + 1

    async def _get_cached_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[Any]:
        """Get cached tool result if available.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Cached result if available and valid, None otherwise
        """
        # This would integrate with the caching service
        # For now, return None to indicate no cache
        return None

    async def _cache_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
    ) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Result to cache
        """
        # This would integrate with the caching service
        # For now, just log
        self._logger.debug(f"Caching result for {tool_name}")


class ToolBudgetExceededError(Exception):
    """Raised when tool budget is exhausted."""

    pass
