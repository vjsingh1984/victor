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

"""Tool call coordination for validation, execution, and retry logic.

This coordinator provides a centralized interface for handling tool calls,
including validation, parsing, execution with retry logic, and result formatting.
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional

from victor.agent.coordinators.tool_call_protocol import (
    IToolCallCoordinator,
    ToolCallContext,
    ToolCallResult,
    ToolCallCoordinatorConfig,
)
from victor.agent.tool_calling.base import ToolCall

logger = logging.getLogger(__name__)


class ToolCallCoordinator(IToolCallCoordinator):
    """Coordinator for tool call operations.

    Handles tool call validation, parsing, execution with retry logic,
    and result formatting. Delegates to existing components for actual
    execution (ToolExecutor, ToolRetryCoordinator, etc.).

    Responsibilities:
    - Parse tool calls from LLM responses
    - Validate tool calls against budget and constraints
    - Execute tool calls with retry logic
    - Aggregate tool call results
    - Format tool outputs for LLM context

    Attributes:
        config: Coordinator configuration
        tool_executor: Tool executor for running tools
        tool_retry_coordinator: Retry coordinator (optional)
        tool_registry: Tool registry for validation
    """

    def __init__(
        self,
        config: ToolCallCoordinatorConfig,
        tool_executor: Any,
        tool_registry: Any,
        tool_retry_coordinator: Optional[Any] = None,
        sanitizer: Optional[Any] = None,
    ):
        """Initialize ToolCallCoordinator.

        Args:
            config: Coordinator configuration
            tool_executor: Tool executor instance
            tool_registry: Tool registry for validation
            tool_retry_coordinator: Optional retry coordinator
            sanitizer: Optional response sanitizer (for backward compatibility)
        """
        self._config = config
        self._tool_executor = tool_executor
        self._tool_registry = tool_registry
        self._tool_retry_coordinator = tool_retry_coordinator
        self._sanitizer = sanitizer  # Stored for backward compatibility

    async def handle_tool_calls(
        self,
        tool_calls: list[ToolCall],
        context: ToolCallContext,
    ) -> list[ToolCallResult]:
        """Handle multiple tool calls with validation, execution, and retry.

        Args:
            tool_calls: List of tool calls to execute
            context: Execution context with budget and constraints

        Returns:
            List of tool call results

        Raises:
            ToolCallValidationError: If tool calls fail validation
            ToolExecutionError: If tool execution fails critically
        """
        if not tool_calls:
            return []

        results: list[ToolCallResult] = []

        for tool_call in tool_calls:
            # Validate tool call
            validation_errors = self._validate_single_tool_call(tool_call, context)
            if validation_errors:
                results.append(
                    ToolCallResult(
                        tool_name=tool_call.name or "unknown",
                        arguments=tool_call.arguments or {},
                        output=None,
                        error="; ".join(validation_errors),
                        success=False,
                    )
                )
                continue

            # Execute tool with retry
            result = await self.execute_tool_with_retry(
                tool_name=tool_call.name,
                arguments=tool_call.arguments or {},
                context=context,
            )

            results.append(result)

        return results

    def _validate_single_tool_call(
        self,
        tool_call: ToolCall,
        context: ToolCallContext,
    ) -> list[str]:
        """Validate a single tool call.

        Args:
            tool_call: Tool call to validate
            context: Execution context

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check tool name
        if not tool_call.name:
            errors.append("Tool call missing name")
            return errors

        # Validate tool name format with sanitizer (backward compatibility)
        if self._sanitizer and hasattr(self._sanitizer, "is_valid_tool_name"):
            try:
                if not self._sanitizer.is_valid_tool_name(tool_call.name):
                    errors.append("Invalid tool name")
            except Exception:
                pass  # Fall through to registry check

        # Check tool availability
        if not self._is_tool_enabled(tool_call.name):
            errors.append(f"Tool '{tool_call.name}' is not available")

        # Check budget
        if context.tool_budget <= 0:
            errors.append("Tool budget exhausted")

        return errors

    def _is_tool_enabled(self, tool_name: str) -> bool:
        """Check if tool is enabled.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is enabled
        """
        # Use sanitizer if available (backward compatibility)
        if self._sanitizer and hasattr(self._sanitizer, "is_valid_tool_name"):
            try:
                if not self._sanitizer.is_valid_tool_name(tool_name):
                    return False
            except Exception:
                pass  # Fall through to registry check

        # Delegate to tool registry
        try:
            result = self._tool_registry.is_available(tool_name)
            return bool(result) if result is not None else False
        except Exception:
            return False

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolCallContext,
    ) -> ToolCallResult:
        """Execute a single tool with retry logic.

        Implements exponential backoff and intelligent retry strategies
        based on error type and tool semantics.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            context: Execution context

        Returns:
            Tool call result with output or error

        Raises:
            ToolExecutionError: If all retry attempts fail
        """
        start_time = time.monotonic()

        # Use retry coordinator if available
        if self._tool_retry_coordinator:
            try:
                result = await self._tool_retry_coordinator.execute_tool(
                    tool_name=tool_name,
                    tool_args=arguments,
                    context=context.__dict__,
                )

                duration_ms = (time.monotonic() - start_time) * 1000

                return ToolCallResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    output=result.result,
                    error=result.error_message if not result.success else None,
                    duration_ms=duration_ms,
                    success=result.success,
                )
            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                return ToolCallResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    output=None,
                    error=str(e),
                    duration_ms=duration_ms,
                    success=False,
                )

        # Fallback to inline retry logic
        last_error = None
        for attempt in range(self._config.max_retries):
            try:
                # Execute tool
                exec_result = await self._tool_executor.execute(
                    tool_name=tool_name,
                    arguments=arguments,
                    context=context.__dict__,
                )

                duration_ms = (time.monotonic() - start_time) * 1000

                if exec_result.success:
                    if attempt > 0:
                        logger.info(
                            f"Tool '{tool_name}' succeeded on retry attempt {attempt + 1}/{self._config.max_retries}"
                        )

                    return ToolCallResult(
                        tool_name=tool_name,
                        arguments=arguments,
                        output=exec_result.result,
                        duration_ms=duration_ms,
                        success=True,
                    )
                else:
                    # Check if error is retryable
                    error_msg = exec_result.error or "Unknown error"
                    non_retryable_errors = [
                        "Invalid",
                        "Missing required",
                        "Not found",
                        "disabled",
                    ]

                    if any(err in error_msg for err in non_retryable_errors):
                        return ToolCallResult(
                            tool_name=tool_name,
                            arguments=arguments,
                            output=exec_result.result,
                            error=error_msg,
                            duration_ms=duration_ms,
                            success=False,
                        )

                    last_error = error_msg
                    if attempt < self._config.max_retries - 1:
                        delay = min(
                            self._config.retry_delay
                            * (self._config.retry_backoff_multiplier**attempt),
                            self._config.retry_delay * 10,
                        )
                        logger.warning(
                            f"Tool '{tool_name}' failed (attempt {attempt + 1}/{self._config.max_retries}): {error_msg}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)

            except Exception as e:
                duration_ms = (time.monotonic() - start_time) * 1000
                last_error = str(e)

                if attempt < self._config.max_retries - 1:
                    delay = min(
                        self._config.retry_delay * (self._config.retry_backoff_multiplier**attempt),
                        self._config.retry_delay * 10,
                    )
                    logger.warning(
                        f"Tool '{tool_name}' exception (attempt {attempt + 1}/{self._config.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

        # All retries failed
        return ToolCallResult(
            tool_name=tool_name,
            arguments=arguments,
            output=None,
            error=last_error or "Unknown error",
            duration_ms=(time.monotonic() - start_time) * 1000,
            success=False,
        )

    def parse_tool_calls(
        self,
        raw_calls: list[dict[str, Any]],
    ) -> list[ToolCall]:
        """Parse raw tool calls from LLM response.

        Normalizes tool call format across different providers and handles
        argument normalization and validation.

        Args:
            raw_calls: Raw tool call data from LLM

        Returns:
            List of parsed ToolCall objects

        Raises:
            ToolCallParseError: If parsing fails
        """
        parsed_calls = []

        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                logger.warning(f"Skipping invalid tool call (not a dict): {raw_call}")
                continue

            tool_name = raw_call.get("name")
            if not tool_name:
                logger.warning("Skipping tool call without name")
                continue

            arguments = raw_call.get("arguments", {})

            # Handle JSON string arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"value": arguments}
            elif arguments is None:
                arguments = {}

            parsed_calls.append(
                ToolCall(
                    name=tool_name,
                    arguments=arguments,
                    id=raw_call.get("id"),
                )
            )

        return parsed_calls

    def validate_tool_calls(
        self,
        tool_calls: list[ToolCall],
        context: ToolCallContext,
    ) -> list[str]:
        """Validate tool calls against budget and constraints.

        Checks:
        - Tool availability and enablement
        - Budget constraints
        - Access permissions
        - Argument validity

        Args:
            tool_calls: Tool calls to validate
            context: Execution context

        Returns:
            List of validation error messages (empty if valid)

        Raises:
            ToolCallValidationError: If validation fails critically
        """
        all_errors = []

        for tool_call in tool_calls:
            errors = self._validate_single_tool_call(tool_call, context)
            all_errors.extend(errors)

        return all_errors

    def format_tool_output(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        output: Any,
    ) -> str:
        """Format tool output for inclusion in LLM context.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            output: Tool execution output

        Returns:
            Formatted output string
        """
        # Format with clear boundaries to prevent hallucination
        args_str = json.dumps(arguments, default=str, ensure_ascii=False)

        if isinstance(output, dict):
            output_str = json.dumps(output, default=str, ensure_ascii=False)
        else:
            output_str = str(output)

        # Truncate very long outputs
        max_output_length = 10000
        if len(output_str) > max_output_length:
            output_str = output_str[:max_output_length] + "... [truncated]"

        return f"""TOOL_OUTPUT: {tool_name}
ARGUMENTS: {args_str}
RESULT: {output_str}
END_TOOL_OUTPUT"""


def create_tool_call_coordinator(
    config: ToolCallCoordinatorConfig,
    tool_executor: Any,
    tool_registry: Any,
    tool_retry_coordinator: Optional[Any] = None,
    sanitizer: Optional[Any] = None,
) -> ToolCallCoordinator:
    """Factory function to create ToolCallCoordinator.

    Args:
        config: Coordinator configuration
        tool_executor: Tool executor instance
        tool_registry: Tool registry for validation
        tool_retry_coordinator: Optional retry coordinator
        sanitizer: Optional response sanitizer (for backward compatibility)

    Returns:
        Configured ToolCallCoordinator instance
    """
    return ToolCallCoordinator(
        config=config,
        tool_executor=tool_executor,
        tool_registry=tool_registry,
        tool_retry_coordinator=tool_retry_coordinator,
        sanitizer=sanitizer,
    )


__all__ = [
    "ToolCallCoordinator",
    "create_tool_call_coordinator",
]
