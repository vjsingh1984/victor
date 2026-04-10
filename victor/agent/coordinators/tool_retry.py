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

"""Tool Retry Executor - Extracted from ToolCoordinator.

Handles tool execution with retry logic, exponential backoff,
caching, and cache invalidation.

Extracted as part of E1 M3 (ToolCoordinator size reduction).
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from victor.agent.coordinators.tool_coordinator import ToolCoordinatorConfig
    from victor.agent.tool_pipeline import ToolPipeline
    from victor.storage.cache.tool_cache import ToolCache

logger = logging.getLogger(__name__)


class ToolRetryExecutor:
    """Executes tools with retry logic and exponential backoff.

    This class encapsulates retry/backoff logic that was previously
    embedded in ToolCoordinator, including:
    - Cache lookup before execution
    - Configurable retry with exponential backoff
    - Non-retryable error detection
    - Cache invalidation for mutating tools
    - Success callbacks

    Design: Takes config, pipeline, and cache at init time.
    """

    def __init__(
        self,
        config: "ToolCoordinatorConfig",
        pipeline: "ToolPipeline",
        cache: Optional["ToolCache"] = None,
    ) -> None:
        """Initialize with execution dependencies.

        Args:
            config: ToolCoordinatorConfig for retry settings
            pipeline: ToolPipeline for actual tool execution
            cache: Optional ToolCache for result caching
        """
        self._config = config
        self._pipeline = pipeline
        self._cache = cache

    async def execute_tool_with_retry(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any],
        tool_executor: Optional[Callable[..., Awaitable[Any]]] = None,
        cache: Optional[Any] = None,
        on_success: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Any], bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context
            tool_executor: Optional custom executor callable. If provided, called as
                ``await tool_executor(tool_name, tool_args, context)``.
                Defaults to ``self._pipeline._execute_single_tool``.
            cache: Optional cache override. Defaults to ``self._cache``.
            on_success: Optional callback invoked on success as
                ``on_success(tool_name, tool_args, result)``.
            retry_config: Optional dict with keys ``retry_enabled``, ``max_attempts``,
                ``base_delay``, ``max_delay`` to override coordinator config.

        Returns:
            Tuple of (result, success, error_message or None)
        """
        effective_cache = cache if cache is not None else self._cache

        # Try cache first for allowlisted tools
        if effective_cache:
            cached = effective_cache.get(tool_name, tool_args)
            if cached is not None:
                logger.debug(f"Cache hit for tool '{tool_name}'")
                return cached, True, None

        if retry_config:
            retry_enabled = retry_config.get("retry_enabled", self._config.retry_enabled)
            max_attempts = (
                retry_config.get("max_attempts", self._config.max_retry_attempts)
                if retry_enabled
                else 1
            )
            base_delay = retry_config.get("base_delay", self._config.retry_base_delay)
            max_delay = retry_config.get("max_delay", self._config.retry_max_delay)
        else:
            retry_enabled = self._config.retry_enabled
            max_attempts = self._config.max_retry_attempts if retry_enabled else 1
            base_delay = self._config.retry_base_delay
            max_delay = self._config.retry_max_delay

        last_error = None
        for attempt in range(max_attempts):
            try:
                if tool_executor:
                    result = await tool_executor(tool_name, tool_args, context)
                else:
                    result = await self._pipeline._execute_single_tool(
                        tool_name, tool_args, context
                    )

                if result.success:
                    # Cache successful result
                    if effective_cache:
                        effective_cache.set(tool_name, tool_args, result)
                        # Invalidate related cache entries
                        invalidating_tools = {
                            "write_file",
                            "edit_files",
                            "execute_bash",
                            "git",
                            "docker",
                        }
                        if tool_name in invalidating_tools:
                            touched_paths = []
                            if "path" in tool_args:
                                touched_paths.append(tool_args["path"])
                            if "paths" in tool_args and isinstance(tool_args["paths"], list):
                                touched_paths.extend(tool_args["paths"])
                            if touched_paths:
                                effective_cache.invalidate_paths(touched_paths)
                            else:
                                namespaces_to_clear = [
                                    "code_search",
                                    "semantic_code_search",
                                    "list_directory",
                                ]
                                effective_cache.clear_namespaces(namespaces_to_clear)

                    if attempt > 0:
                        logger.info(
                            f"Tool '{tool_name}' succeeded on retry attempt "
                            f"{attempt + 1}/{max_attempts}"
                        )

                    # Invoke success callback if provided
                    if on_success:
                        on_success(tool_name, tool_args, result)

                    return result, True, None
                else:
                    # Tool returned failure - check if retryable
                    error_msg = result.error or "Unknown error"

                    # Don't retry validation errors or permanent failures
                    non_retryable_errors = [
                        "Invalid",
                        "Missing required",
                        "Not found",
                        "disabled",
                    ]
                    if any(err in error_msg for err in non_retryable_errors):
                        logger.debug(
                            f"Tool '{tool_name}' failed with non-retryable error: {error_msg}"
                        )
                        return result, False, error_msg

                    last_error = error_msg
                    if attempt < max_attempts - 1:
                        # Calculate exponential backoff delay
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"Tool '{tool_name}' failed (attempt {attempt + 1}/{max_attempts}): "
                            f"{error_msg}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Tool '{tool_name}' failed after {max_attempts} attempts: {error_msg}"
                        )
                        return result, False, error_msg

            except Exception as e:
                # Check for non-retryable errors
                from victor.core.errors import ToolNotFoundError, ToolValidationError

                if isinstance(e, (ToolNotFoundError, ToolValidationError, PermissionError)):
                    logger.error(f"Tool '{tool_name}' permanent failure: {e}")
                    return None, False, str(e)

                # Retryable transient errors
                last_error = str(e)
                if attempt < max_attempts - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        f"Tool '{tool_name}' transient error "
                        f"(attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Tool '{tool_name}' failed after {max_attempts} attempts: {e}")
                    return None, False, last_error

        # Should not reach here, but handle it anyway
        return None, False, last_error or "Unknown error"


__all__ = [
    "ToolRetryExecutor",
]
