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

"""Tool service protocol.

Defines the interface for tool operations, including selection,
execution, and budgeting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.tools.base import ToolResult


@runtime_checkable
class ToolSelectionContext(Protocol):
    """Context for tool selection decisions.

    Provides information about the current state to help
    make intelligent tool selection decisions.
    """

    @property
    def user_message(self) -> str:
        """The current user message."""
        ...

    @property
    def conversation_stage(self) -> str:
        """The current conversation stage."""
        ...

    @property
    def available_tools(self) -> List[str]:
        """List of available tool names."""
        ...

    @property
    def recent_tools(self) -> List[str]:
        """Recently used tools."""
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Additional metadata for selection."""
        ...


@runtime_checkable
class ToolServiceProtocol(Protocol):
    """Protocol for tool operations service.

    Handles:
    - Intelligent tool selection based on context
    - Tool execution with validation and error handling
    - Tool budget management and tracking
    - Tool usage analytics

    This protocol follows the Interface Segregation Principle (ISP)
    by focusing only on tool-related operations.

    Methods:
        select_tools: Select tools based on context and constraints
        execute_tool: Execute a single tool with validation
        execute_tools_parallel: Execute multiple tools in parallel
        get_tool_budget: Get remaining tool budget
        get_tool_usage_stats: Get tool usage statistics

    Example:
        class MyToolService(ToolServiceProtocol):
            def __init__(self, selector, executor, budget_manager):
                self._selector = selector
                self._executor = executor
                self._budget = budget_manager

            async def select_tools(self, context, max_tools=10):
                return await self._selector.select(context, max_tools)

            async def execute_tool(self, tool_name, arguments):
                if self._budget.is_exhausted():
                    raise ToolBudgetExceededError()
                result = await self._executor.execute(tool_name, arguments)
                self._budget.record_usage()
                return result
    """

    async def select_tools(
        self,
        context: "ToolSelectionContext",
        max_tools: int = 10,
    ) -> List[str]:
        """Select tools for execution based on context.

        Uses semantic analysis and heuristics to select the most
        relevant tools for the current task.

        Args:
            context: Tool selection context with message, stage, available tools
            max_tools: Maximum number of tools to select

        Returns:
            List of selected tool names, ordered by relevance

        Raises:
            ToolSelectionError: If tool selection fails
        """
        ...

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> "ToolResult":
        """Execute a single tool with validation and error handling.

        Validates arguments, checks budget, executes the tool,
        and tracks usage for analytics.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome

        Raises:
            ToolNotFoundError: If tool is not registered
            ToolValidationError: If arguments are invalid
            ToolBudgetExceededError: If tool budget is exhausted
            ToolExecutionError: If tool execution fails
        """
        ...

    async def execute_tools_parallel(
        self,
        tool_calls: List[tuple[str, Dict[str, Any]]],
        max_parallel: int = 5,
    ) -> AsyncIterator["ToolResult"]:
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
        ...

    def get_tool_budget(self) -> int:
        """Get the remaining tool budget.

        The tool budget limits the number of tool calls per session
        to prevent infinite loops and excessive API usage.

        Returns:
            Number of remaining tool calls allowed

        Example:
            budget = tool_service.get_tool_budget()
            if budget < 5:
                logger.warning(f"Low tool budget: {budget} remaining")
        """
        ...

    def set_tool_budget(self, budget: int) -> None:
        """Set the tool budget limit.

        Args:
            budget: Maximum number of tool calls allowed

        Raises:
            ValueError: If budget is negative
        """
        ...

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.

        Returns statistics about tool usage including:
        - Total tool calls
        - Calls by tool name
        - Success/failure rates
        - Average execution time

        Returns:
            Dictionary with usage statistics

        Example:
            stats = tool_service.get_tool_usage_stats()
            print(f"Total calls: {stats['total_calls']}")
            print(f"Success rate: {stats['success_rate']:.1%}")
        """
        ...

    def reset_tool_budget(self) -> None:
        """Reset the tool budget to initial limit.

        Useful for:
        - Starting new sessions
        - Testing and development
        - Recovery from budget exhaustion
        """
        ...

    def is_healthy(self) -> bool:
        """Check if the tool service is healthy.

        A healthy tool service should:
        - Have tools registered
        - Have budget available
        - Have selector and executor available

        Returns:
            True if the service is healthy, False otherwise
        """
        ...


@runtime_checkable
class ToolCachingProtocol(Protocol):
    """Protocol for tool result caching.

    Enables intelligent caching of tool results to avoid
    redundant executions and improve performance.
    """

    async def get_cached_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional["ToolResult"]:
        """Get cached tool result if available.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Cached ToolResult if available and valid, None otherwise
        """
        ...

    async def cache_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: "ToolResult",
        ttl: int = 600,
    ) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments (used as cache key)
            result: Result to cache
            ttl: Time to live in seconds (default: 10 minutes)
        """
        ...

    def invalidate_cache(self, tool_name: Optional[str] = None) -> None:
        """Invalidate cached results.

        Args:
            tool_name: Specific tool to invalidate, or None for all tools
        """
        ...
