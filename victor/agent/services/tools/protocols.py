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

"""Protocol definitions for decomposed tool services.

Following the Dependency Inversion Principle, these protocols define
the contracts that each decomposed service must implement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Protocol, Set

if TYPE_CHECKING:
    from victor.tools.base import BaseTool, ToolResult


class ToolSelectorServiceProtocol(Protocol):
    """Protocol for tool selection service.

    Responsible for tool selection logic, filtering, and validation.
    """

    async def select_tools(
        self,
        query: str,
        available_tools: Set[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Select tools based on query and context.

        Args:
            query: User query or task description
            available_tools: Set of available tool names
            context: Optional context for selection

        Returns:
            List of selected tool names (ordered by relevance)
        """
        ...

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is enabled, False otherwise
        """
        ...

    def get_enabled_tools(self) -> Set[str]:
        """Get set of enabled tools.

        Returns:
            Set of enabled tool names
        """
        ...

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set the enabled tools.

        Args:
            tools: Set of tool names to enable
        """
        ...

    def filter_hallucinated_tools(
        self, tool_calls: List[Dict[str, Any]], known_tools: Set[str]
    ) -> List[Dict[str, Any]]:
        """Filter out hallucinated tool calls.

        Args:
            tool_calls: List of tool calls to filter
            known_tools: Set of known tool names

        Returns:
            Filtered list of tool calls
        """
        ...


class ToolExecutorServiceProtocol(Protocol):
    """Protocol for tool execution service.

    Responsible for tool execution, retries, and error handling.
    """

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Execute a single tool.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            context: Optional execution context

        Returns:
            ToolResult with execution output
        """
        ...

    async def execute_tools_parallel(
        self, tool_calls: List[Dict[str, Any]], max_concurrency: int = 5
    ) -> List["ToolResult"]:
        """Execute multiple tools in parallel.

        Args:
            tool_calls: List of tool calls to execute
            max_concurrency: Maximum concurrent executions

        Returns:
            List of ToolResults (same order as tool_calls)
        """
        ...

    def validate_tool_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate a tool call before execution.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        ...

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the schema for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema or None if not found
        """
        ...


class ToolTrackerServiceProtocol(Protocol):
    """Protocol for tool tracking service.

    Responsible for budget tracking, usage metrics, and statistics.
    """

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted.

        Returns:
            True if budget exhausted, False otherwise
        """
        ...

    def get_remaining_budget(self) -> int:
        """Get remaining tool budget.

        Returns:
            Remaining budget count
        """
        ...

    def consume_budget(self, amount: int = 1) -> None:
        """Consume from the tool budget.

        Args:
            amount: Amount to consume (default: 1)
        """
        ...

    def reset_tool_budget(self) -> None:
        """Reset the tool budget to initial limit."""
        ...

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.

        Returns:
            Dictionary with usage metrics
        """
        ...

    def get_tool_call_count(self, tool_name: str) -> int:
        """Get call count for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of times tool was called
        """
        ...

    def get_tool_error_count(self, tool_name: str) -> int:
        """Get error count for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of times tool errored
        """
        ...

    def record_execution(self, tool_name: str, success: bool, duration_ms: float) -> None:
        """Record a tool execution for metrics.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            duration_ms: Execution duration in milliseconds
        """
        ...


class ToolPlannerServiceProtocol(Protocol):
    """Protocol for tool planning service.

    Responsible for execution planning and strategy.
    """

    async def plan_execution(
        self,
        task: str,
        available_tools: Set[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Plan tool execution for a task.

        Args:
            task: Task description
            available_tools: Set of available tool names
            context: Optional context for planning

        Returns:
            List of planned tool calls (ordered for execution)
        """
        ...

    def estimate_execution_cost(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate execution cost for tool calls.

        Args:
            tool_calls: List of tool calls to estimate

        Returns:
            Dictionary with cost estimates (time, tokens, etc.)
        """
        ...

    def optimize_execution_order(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize execution order for tool calls.

        Args:
            tool_calls: List of tool calls to optimize

        Returns:
            Optimized list of tool calls
        """
        ...


class ToolResultProcessorProtocol(Protocol):
    """Protocol for tool result processing service.

    Responsible for result processing, formatting, and aggregation.
    """

    def process_result(self, result: "ToolResult") -> Dict[str, Any]:
        """Process a single tool result.

        Args:
            result: ToolResult to process

        Returns:
            Processed result dictionary
        """
        ...

    def format_result_for_llm(self, result: "ToolResult") -> str:
        """Format a tool result for LLM consumption.

        Args:
            result: ToolResult to format

        Returns:
            Formatted result string
        """
        ...

    def aggregate_results(self, results: List["ToolResult"]) -> Dict[str, Any]:
        """Aggregate multiple tool results.

        Args:
            results: List of ToolResults to aggregate

        Returns:
            Aggregated results dictionary
        """
        ...

    def extract_insights(self, results: List["ToolResult"]) -> List[str]:
        """Extract insights from tool results.

        Args:
            results: List of ToolResults to analyze

        Returns:
            List of insight strings
        """
        ...
