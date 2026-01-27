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

"""Tool retry coordinator for resilient tool execution.

This coordinator manages tool execution with retry logic, exponential backoff,
and cache integration. It provides a centralized location for retry strategy,
making it easier to test and maintain retry behavior.

Key Features:
- Exponential backoff retry logic
- Cache integration for performance
- Distinguish between retryable and non-retryable errors
- Task completion detection support
- Configurable retry parameters

Design Patterns:
- SRP: Single responsibility for tool retry logic
- Strategy Pattern: Pluggable retry strategies
- Dependency Inversion: Depends on protocols, not implementations
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from victor.tools.base import BaseTool
    from victor.agent.protocols import ToolExecutorProtocol


logger = logging.getLogger(__name__)


@dataclass
class ToolRetryConfig:
    """Configuration for tool retry behavior.

    Attributes:
        retry_enabled: Whether retry logic is enabled
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff (seconds)
        max_delay: Maximum delay between retries (seconds)
        non_retryable_errors: Error patterns that should not be retried
        cache_enabled: Whether tool result caching is enabled
        task_completion_enabled: Whether task completion detection is enabled
    """

    retry_enabled: bool = True
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    non_retryable_errors: List[str] = field(
        default_factory=lambda: ["Invalid", "Missing required", "Not found", "disabled"]
    )
    cache_enabled: bool = True
    task_completion_enabled: bool = False


@dataclass
class ToolExecutionResult:
    """Result of tool execution with metadata.

    Attributes:
        result: The tool execution result (or None if failed)
        success: Whether the tool execution succeeded
        error_message: Error message if execution failed
        attempts: Number of attempts made
        from_cache: Whether result came from cache
    """

    result: Optional[Any]
    success: bool
    error_message: Optional[str]
    attempts: int
    from_cache: bool = False


class ToolRetryCoordinator:
    """Coordinator for tool execution with retry logic.

    This coordinator handles tool execution with intelligent retry behavior,
    including exponential backoff, cache integration, and error classification.

    Example:
        ```python
        coordinator = ToolRetryCoordinator(
            tool_executor=tool_registry,
            config=ToolRetryConfig(max_attempts=3)
        )

        result = await coordinator.execute_tool(
            tool_name="read_file",
            tool_args={"path": "/tmp/file.txt"},
            context={}
        )

        if result.success:
            print(f"Tool result: {result.result}")
        else:
            print(f"Tool failed: {result.error_message}")
        ```
    """

    def __init__(
        self,
        tool_executor: "ToolExecutorProtocol",
        tool_cache: Optional[Any] = None,
        task_completion_detector: Optional[Any] = None,
        config: Optional[ToolRetryConfig] = None,
    ):
        """Initialize the tool retry coordinator.

        Args:
            tool_executor: Tool executor for running tools
            tool_cache: Optional tool cache for performance
            task_completion_detector: Optional task completion detector
            config: Retry configuration (uses defaults if None)
        """
        self._tool_executor = tool_executor
        self._tool_cache = tool_cache
        self._task_completion_detector = task_completion_detector
        self._config = config or ToolRetryConfig()

    async def execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolExecutionResult:
        """Execute a tool with retry logic and exponential backoff.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context

        Returns:
            ToolExecutionResult with execution outcome and metadata
        """
        # Try cache first if enabled
        if self._tool_cache and self._config.cache_enabled:
            cached = self._tool_cache.get(tool_name, tool_args)
            if cached is not None:
                logger.debug(f"Cache hit for tool '{tool_name}'")
                result, success, error = cached
                return ToolExecutionResult(
                    result=result, success=success, error_message=error, from_cache=True, attempts=1
                )

        # Calculate max attempts
        max_attempts = self._config.max_attempts if self._config.retry_enabled else 1

        last_error = None
        for attempt in range(max_attempts):
            try:
                # Execute tool
                tool_result = await self._tool_executor.execute(
                    tool_name, context=context, **tool_args
                )

                if tool_result.success:
                    # Handle success
                    return await self._handle_success(tool_name, tool_args, tool_result, attempt)
                else:
                    # Handle failure
                    failure_result = self._handle_failure(
                        tool_name, tool_result, attempt, max_attempts
                    )
                    if failure_result is not None:
                        return failure_result
                    last_error = tool_result.error or "Unknown error"

            except (TypeError, AttributeError, ValueError) as e:
                # Non-retryable errors - fail immediately
                logger.error(f"Tool '{tool_name}' permanent failure: {e}")
                return ToolExecutionResult(
                    result=None, success=False, error_message=str(e), attempts=attempt + 1
                )
            except (TimeoutError, ConnectionError, asyncio.TimeoutError) as e:
                # Retryable transient errors
                last_error = str(e)
                if attempt < max_attempts - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Tool '{tool_name}' transient error (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Tool '{tool_name}' failed after {max_attempts} attempts: {e}")
                    return ToolExecutionResult(
                        result=None, success=False, error_message=last_error, attempts=max_attempts
                    )
            except Exception as e:
                # Unknown errors - log and retry with caution
                last_error = str(e)
                if attempt < max_attempts - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Tool '{tool_name}' unexpected error (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Tool '{tool_name}' raised exception after {max_attempts} attempts: {e}"
                    )
                    return ToolExecutionResult(
                        result=None, success=False, error_message=last_error, attempts=max_attempts
                    )

        # Should not reach here, but handle it anyway
        return ToolExecutionResult(
            result=None,
            success=False,
            error_message=last_error or "Unknown error",
            attempts=max_attempts,
        )

    async def _handle_success(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
        attempt: int,
    ) -> ToolExecutionResult:
        """Handle successful tool execution.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            tool_result: Tool execution result
            attempt: Attempt number (0-indexed)

        Returns:
            ToolExecutionResult for successful execution
        """
        # Cache result if enabled
        if self._tool_cache and self._config.cache_enabled:
            self._tool_cache.set(tool_name, tool_args, tool_result)
            await self._invalidate_cache_entries(tool_name, tool_args)

        # Log retry success
        if attempt > 0:
            max_attempts = self._config.max_attempts if self._config.retry_enabled else 1
            logger.info(
                f"Tool '{tool_name}' succeeded on retry attempt {attempt + 1}/{max_attempts}"
            )

        # Record for task completion detection
        if self._task_completion_detector and self._config.task_completion_enabled:
            tool_result_dict = {"success": True}
            if "path" in tool_args:
                tool_result_dict["path"] = tool_args["path"]
            elif "file_path" in tool_args:
                tool_result_dict["file_path"] = tool_args["file_path"]
            self._task_completion_detector.record_tool_result(tool_name, tool_result_dict)

        return ToolExecutionResult(
            result=tool_result, success=True, error_message=None, attempts=attempt + 1
        )

    async def _invalidate_cache_entries(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """Invalidate cache entries based on tool execution.

        Args:
            tool_name: Name of the tool executed
            tool_args: Tool arguments
        """
        invalidating_tools = {
            "write_file",
            "edit_files",
            "execute_bash",
            "git",
            "docker",
        }

        if tool_name not in invalidating_tools:
            return

        touched_paths = []
        if "path" in tool_args:
            touched_paths.append(tool_args["path"])
        if "paths" in tool_args and isinstance(tool_args["paths"], list):
            touched_paths.extend(tool_args["paths"])

        if touched_paths and self._tool_cache is not None:
            self._tool_cache.invalidate_paths(touched_paths)
        elif self._tool_cache is not None:
            namespaces_to_clear = [
                "code_search",
                "semantic_code_search",
                "list_directory",
            ]
            self._tool_cache.clear_namespaces(namespaces_to_clear)

    def _handle_failure(
        self,
        tool_name: str,
        tool_result: Any,
        attempt: int,
        max_attempts: int,
    ) -> Optional[ToolExecutionResult]:
        """Handle failed tool execution.

        Args:
            tool_name: Name of the tool
            tool_result: Tool execution result
            attempt: Attempt number (0-indexed)
            max_attempts: Maximum number of attempts

        Returns:
            ToolExecutionResult if error is non-retryable, None if should retry
        """
        error_msg = tool_result.error or "Unknown error"

        # Check if error is non-retryable
        if any(err in error_msg for err in self._config.non_retryable_errors):
            logger.debug(f"Tool '{tool_name}' failed with non-retryable error: {error_msg}")
            return ToolExecutionResult(
                result=tool_result, success=False, error_message=error_msg, attempts=attempt + 1
            )

        # Retryable error - log and continue retry loop
        if attempt < max_attempts - 1:
            delay = self._calculate_backoff(attempt)
            logger.warning(
                f"Tool '{tool_name}' failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. "
                f"Retrying in {delay:.1f}s..."
            )
            return None  # Continue retry loop
        else:
            logger.error(f"Tool '{tool_name}' failed after {max_attempts} attempts: {error_msg}")
            return ToolExecutionResult(
                result=tool_result, success=False, error_message=error_msg, attempts=max_attempts
            )

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        base_delay: float = self._config.base_delay
        max_delay: float = self._config.max_delay
        calculated_delay: float = min(base_delay * (2**attempt), max_delay)
        return calculated_delay

    def get_config(self) -> ToolRetryConfig:
        """Get current retry configuration.

        Returns:
            Current ToolRetryConfig
        """
        return self._config

    def update_config(self, config: ToolRetryConfig) -> None:
        """Update retry configuration.

        Args:
            config: New configuration to use
        """
        self._config = config
        logger.info("Tool retry configuration updated")


def create_tool_retry_coordinator(
    tool_executor: "ToolExecutorProtocol",
    tool_cache: Optional[Any] = None,
    task_completion_detector: Optional[Any] = None,
    retry_enabled: bool = True,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> ToolRetryCoordinator:
    """Factory function to create a ToolRetryCoordinator with common defaults.

    Args:
        tool_executor: Tool executor for running tools
        tool_cache: Optional tool cache
        task_completion_detector: Optional task completion detector
        retry_enabled: Whether retry logic is enabled
        max_attempts: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay between retries

    Returns:
        Configured ToolRetryCoordinator instance
    """
    config = ToolRetryConfig(
        retry_enabled=retry_enabled,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    return ToolRetryCoordinator(
        tool_executor=tool_executor,
        tool_cache=tool_cache,
        task_completion_detector=task_completion_detector,
        config=config,
    )
