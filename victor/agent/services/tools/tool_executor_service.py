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

"""Tool executor service implementation.

Handles tool execution, retries, and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List

if TYPE_CHECKING:
    from victor.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutorServiceConfig:
    """Configuration for ToolExecutorService.

    Attributes:
        max_retries: Maximum retry attempts for failed executions
        retry_delay: Delay between retries in seconds
        timeout: Execution timeout in seconds
        enable_parallel: Enable parallel execution
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
        enable_parallel: bool = True,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.enable_parallel = enable_parallel


class ToolExecutorService:
    """Service for tool execution.

    Responsible for:
    - Single tool execution
    - Parallel tool execution
    - Retry logic with exponential backoff
    - Timeout handling
    - Error recovery

    This service does NOT handle:
    - Tool selection (delegated to ToolSelectorService)
    - Budget tracking (delegated to ToolTrackerService)
    - Execution planning (delegated to ToolPlannerService)
    - Result processing (delegated to ToolResultProcessor)

    Example:
        config = ToolExecutorServiceConfig()
        executor = ToolExecutorService(
            config=config,
            tool_registry={"search": search_tool}
        )

        # Execute single tool
        result = await executor.execute_tool(
            "search",
            {"query": "test"}
        )

        # Execute multiple tools in parallel
        results = await executor.execute_tools_parallel([
            {"name": "search", "arguments": {"query": "test1"}},
            {"name": "search", "arguments": {"query": "test2"}},
        ])
    """

    def __init__(
        self,
        config: ToolExecutorServiceConfig,
        tool_registry: Dict[str, "BaseTool"],
    ):
        """Initialize ToolExecutorService.

        Args:
            config: Service configuration
            tool_registry: Dictionary of tool name -> tool instance
        """
        self.config = config
        self.tool_registry = tool_registry

        # Execution callbacks
        self._on_complete_callbacks: List[Callable] = []

        # Health tracking
        self._healthy = True

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Dict[str, Any] | None = None,
    ) -> "ToolResult":
        """Execute a single tool with retry logic.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            context: Optional execution context

        Returns:
            ToolResult with execution output

        Raises:
            ValueError: If tool not found or arguments invalid
            TimeoutError: If execution times out
        """
        if tool_name not in self.tool_registry:
            raise ValueError(f"Tool not found: {tool_name}")

        tool = self.tool_registry[tool_name]
        last_error = None

        # Retry loop
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()

                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_with_validation(tool, arguments, context),
                    timeout=self.config.timeout,
                )

                duration_ms = (time.time() - start_time) * 1000

                # Call completion callbacks
                for callback in self._on_complete_callbacks:
                    try:
                        callback(tool_name, result, duration_ms)
                    except Exception as e:
                        logger.error(f"Error in completion callback: {e}")

                return result

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Tool execution timed out: {tool_name}")
                logger.warning(f"Attempt {attempt + 1}: Tool execution timed out")

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}: Tool execution failed: {e}")

            # Retry delay with exponential backoff
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay * (2**attempt)
                logger.debug(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)

        # All retries exhausted
        from victor.tools.base import ToolResult

        return ToolResult(
            success=False,
            error=f"Execution failed after {self.config.max_retries} retries: {last_error}",
            output=None,
        )

    async def _execute_with_validation(
        self,
        tool: "BaseTool",
        arguments: Dict[str, Any],
        context: Dict[str, Any] | None,
    ) -> "ToolResult":
        """Execute tool with validation.

        Args:
            tool: Tool instance
            arguments: Tool arguments
            context: Optional execution context

        Returns:
            ToolResult

        Raises:
            ValueError: If arguments are invalid
        """
        # Validate arguments
        is_valid, error_message = self.validate_tool_call(tool.name, arguments)
        if not is_valid:
            raise ValueError(f"Invalid arguments: {error_message}")

        # Execute tool
        if asyncio.iscoroutinefunction(tool.execute):
            result = await tool.execute(**arguments)
        else:
            result = tool.execute(**arguments)

        return result

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
        if not self.config.enable_parallel:
            # Execute sequentially
            return [
                await self.execute_tool(
                    call["name"], call.get("arguments", {}), call.get("context")
                )
                for call in tool_calls
            ]

        # Create execution tasks
        async def execute_single(call: Dict[str, Any]) -> "ToolResult":
            return await self.execute_tool(
                call["name"], call.get("arguments", {}), call.get("context")
            )

        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrency)

        async def limited_execute(call: Dict[str, Any]) -> "ToolResult":
            async with semaphore:
                return await execute_single(call)

        # Run all executions
        results = await asyncio.gather(
            *[limited_execute(call) for call in tool_calls],
            return_exceptions=True,
        )

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                from victor.tools.base import ToolResult

                final_results.append(
                    ToolResult(
                        success=False,
                        error=str(result),
                        output=None,
                    )
                )
            else:
                final_results.append(result)

        return final_results

    def validate_tool_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate a tool call before execution.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if tool_name not in self.tool_registry:
            return False, f"Tool not found: {tool_name}"

        tool = self.tool_registry[tool_name]

        # Check if tool has schema
        if hasattr(tool, "get_schema") and callable(tool.get_schema):
            schema = tool.get_schema()

            # Validate against schema if available
            if schema and "parameters" in schema:
                parameters = schema["parameters"]
                required = parameters.get("required", [])

                # Check required parameters
                for param in required:
                    if param not in arguments:
                        return False, f"Missing required parameter: {param}"

        return True, None

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any] | None:
        """Get the schema for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema or None if not found
        """
        if tool_name not in self.tool_registry:
            return None

        tool = self.tool_registry[tool_name]

        if hasattr(tool, "get_schema") and callable(tool.get_schema):
            return tool.get_schema()

        return None

    def register_completion_callback(self, callback: Callable) -> None:
        """Register a callback to be called on tool completion.

        Args:
            callback: Function to call with (tool_name, result, duration_ms)
        """
        self._on_complete_callbacks.append(callback)

    def is_healthy(self) -> bool:
        """Check if service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return self._healthy
