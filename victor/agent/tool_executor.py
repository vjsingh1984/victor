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

"""Tool execution with retry logic, caching, and metrics."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.safety import SafetyChecker, get_safety_checker
from victor.cache.tool_cache import ToolCache
from victor.core.retry import (
    RetryContext,
    RetryExecutor,
    RetryStrategy,
    tool_retry_strategy,
)
from victor.tools.base import BaseTool, Hook, HookError, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutionResult:
    """Result of a tool execution attempt."""

    def __init__(
        self,
        tool_name: str,
        success: bool,
        result: Any,
        error: Optional[str] = None,
        execution_time: float = 0.0,
        cached: bool = False,
        retries: int = 0,
        normalization_strategy: Optional[NormalizationStrategy] = None,
    ):
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.cached = cached
        self.retries = retries
        self.normalization_strategy = normalization_strategy


class ToolExecutor:
    """Executes tools with retry logic, caching, and metrics.

    Responsibilities:
    - Execute tools with proper error handling
    - Implement retry logic with exponential backoff
    - Cache idempotent tool results
    - Normalize malformed arguments
    - Track execution metrics
    """

    # Tools that are safe to cache (idempotent, read-only operations)
    DEFAULT_CACHEABLE_TOOLS = frozenset(
        [
            "code_search",
            "semantic_code_search",
            "list_directory",
            "read_file",
            "plan_files",
        ]
    )

    # Tools that modify state and should invalidate cache
    CACHE_INVALIDATING_TOOLS = frozenset(
        [
            "write_file",
            "edit_files",
            "execute_bash",
            "git",
            "docker",
        ]
    )

    def __init__(
        self,
        tool_registry: ToolRegistry,
        argument_normalizer: Optional[ArgumentNormalizer] = None,
        tool_cache: Optional[ToolCache] = None,
        safety_checker: Optional[SafetyChecker] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_strategy: Optional[RetryStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize tool executor.

        Args:
            tool_registry: Registry of available tools
            argument_normalizer: Normalizer for fixing malformed arguments
            tool_cache: Cache for idempotent tool results
            safety_checker: Checker for dangerous operations (uses global if None)
            max_retries: Maximum retry attempts for failed tools
            retry_delay: Initial delay between retries (exponential backoff)
            retry_strategy: Unified retry strategy (overrides max_retries/retry_delay)
            context: Shared context passed to all tools
        """
        self.tools = tool_registry
        self.normalizer = argument_normalizer or ArgumentNormalizer()
        self.cache = tool_cache
        self.safety_checker = safety_checker or get_safety_checker()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Use provided strategy or create one from legacy params
        self.retry_strategy = retry_strategy or tool_retry_strategy(
            max_retries=max_retries, base_delay=retry_delay
        )
        self.retry_executor = RetryExecutor(self.retry_strategy)
        self.context = context or {}

        # Execution statistics
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._failed_signatures: set[Tuple[str, str]] = set()

    def update_context(self, **kwargs: Any) -> None:
        """Update the shared context passed to tools."""
        self.context.update(kwargs)

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        skip_cache: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        """Execute a tool with retry logic and caching.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            skip_cache: Skip cache lookup even for cacheable tools
            context: Optional context to pass to the tool (merged with default context)

        Returns:
            ToolExecutionResult with execution outcome
        """
        # Merge default context with call-specific context
        exec_context = {**self.context}
        if context:
            exec_context.update(context)
        start_time = time.time()

        # Initialize stats for this tool
        if tool_name not in self._stats:
            self._stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "cache_hits": 0,
                "total_time": 0.0,
                "retries": 0,
            }

        self._stats[tool_name]["calls"] += 1

        # Normalize arguments
        normalized_args, strategy = self.normalizer.normalize_arguments(arguments, tool_name)

        # Check cache first
        if not skip_cache and self.cache:
            cached_result = self.cache.get(tool_name, normalized_args)
            if cached_result is not None:
                self._stats[tool_name]["cache_hits"] += 1
                logger.debug(f"Cache hit for {tool_name}")
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=True,
                    result=cached_result,
                    cached=True,
                    normalization_strategy=strategy,
                )

        # Get the tool
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
            )

        # Check if tool is enabled
        if not self.tools.is_tool_enabled(tool_name):
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' is disabled",
            )

        # Safety check for dangerous operations
        should_proceed, rejection_reason = await self.safety_checker.check_and_confirm(
            tool_name, normalized_args
        )
        if not should_proceed:
            logger.info(f"Tool execution blocked by safety check: {tool_name}")
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=rejection_reason or "Operation cancelled by safety check",
            )

        # Execute with retry
        result, success, error, retries = await self._execute_with_retry(
            tool, normalized_args, exec_context
        )

        execution_time = time.time() - start_time
        self._stats[tool_name]["total_time"] += execution_time
        self._stats[tool_name]["retries"] += retries

        if success:
            self._stats[tool_name]["successes"] += 1

            # Cache successful results for cacheable tools
            if self.cache and tool_name in self.DEFAULT_CACHEABLE_TOOLS:
                self.cache.set(tool_name, normalized_args, result)

            # Invalidate cache for tools that modify state
            if self.cache and tool_name in self.CACHE_INVALIDATING_TOOLS:
                self._invalidate_cache_for_write_tool(tool_name, normalized_args)
        else:
            self._stats[tool_name]["failures"] += 1
            # Track failed signature to avoid retrying same failure
            sig = (tool_name, str(sorted(normalized_args.items())))
            self._failed_signatures.add(sig)

        return ToolExecutionResult(
            tool_name=tool_name,
            success=success,
            result=result,
            error=error,
            execution_time=execution_time,
            retries=retries,
            normalization_strategy=strategy,
        )

    def _run_before_hooks(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Run before hooks for a tool execution.

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments

        Raises:
            HookError: If a critical hook fails
        """
        for before_hook in self.tools._before_hooks:
            hook_obj = before_hook if isinstance(before_hook, Hook) else None
            hook_name = hook_obj.name if hook_obj else getattr(before_hook, "__name__", "hook")
            is_critical = hook_obj.critical if hook_obj else False
            try:
                before_hook(tool_name, arguments)
            except Exception as e:
                if is_critical:
                    logger.error(f"Critical before hook '{hook_name}' failed: {e}")
                    raise HookError(hook_name, e, tool_name)
                else:
                    logger.warning(f"Before hook '{hook_name}' failed (non-critical): {e}")

    def _run_after_hooks(self, tool_name: str, result: Any) -> None:
        """Run after hooks for a tool execution.

        Args:
            tool_name: Name of the tool that was executed
            result: Tool execution result

        Raises:
            HookError: If a critical hook fails
        """
        for after_hook in self.tools._after_hooks:
            hook_obj = after_hook if isinstance(after_hook, Hook) else None
            hook_name = hook_obj.name if hook_obj else getattr(after_hook, "__name__", "hook")
            is_critical = hook_obj.critical if hook_obj else False
            try:
                after_hook(result)
            except Exception as e:
                if is_critical:
                    logger.error(f"Critical after hook '{hook_name}' failed: {e}")
                    raise HookError(hook_name, e, tool_name)
                else:
                    logger.warning(f"After hook '{hook_name}' failed (non-critical): {e}")

    async def _execute_with_retry(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Any, bool, Optional[str], int]:
        """Execute a tool with retry logic from unified RetryStrategy.

        Uses the configured retry strategy for exponential backoff and
        retry decisions. Hooks are run before/after each attempt.

        Args:
            tool: Tool to execute
            arguments: Tool arguments
            context: Context to pass to the tool

        Returns:
            Tuple of (result, success, error_message, retry_count)
        """
        retry_context = RetryContext(
            max_attempts=getattr(self.retry_strategy, "max_attempts", self.max_retries)
        )

        while True:
            retry_context.attempt += 1

            try:
                # Run before hooks - critical hooks can block execution
                self._run_before_hooks(tool.name, arguments)

                # Execute the tool
                result = await tool.execute(context, **arguments)

                # Run after hooks - critical hooks can raise errors
                self._run_after_hooks(tool.name, result)

                # Handle ToolResult
                if isinstance(result, ToolResult):
                    if result.success:
                        self.retry_strategy.on_success(retry_context)
                        return result.output, True, None, retry_context.attempt - 1
                    else:
                        # Don't retry if tool explicitly returned failure
                        error = result.error or "Tool returned failure"
                        return result.output, False, error, retry_context.attempt - 1
                else:
                    # Raw result (for tools that don't return ToolResult)
                    self.retry_strategy.on_success(retry_context)
                    return result, True, None, retry_context.attempt - 1

            except Exception as e:
                retry_context.record_exception(e)
                logger.warning(
                    f"Tool {tool.name} failed (attempt {retry_context.attempt}/"
                    f"{retry_context.max_attempts}): {e}"
                )

                if self.retry_strategy.should_retry(retry_context):
                    self.retry_strategy.on_retry(retry_context)
                    delay = self.retry_strategy.get_delay(retry_context)
                    retry_context.record_delay(delay)

                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    self.retry_strategy.on_failure(retry_context)
                    return None, False, str(e), retry_context.attempt - 1

    def has_failed_before(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Check if this exact tool call has failed before."""
        sig = (tool_name, str(sorted(arguments.items())))
        return sig in self._failed_signatures

    def clear_failed_signatures(self) -> None:
        """Clear the record of failed tool calls."""
        self._failed_signatures.clear()

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all tools."""
        return self._stats.copy()

    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """Get execution statistics for a specific tool."""
        return self._stats.get(tool_name, {}).copy()

    def reset_stats(self) -> None:
        """Reset all execution statistics."""
        self._stats.clear()

    def invalidate_cache_for_paths(self, paths: List[str]) -> None:
        """Invalidate cache entries for modified paths.

        Called when files are modified to ensure stale cache is cleared.

        Args:
            paths: List of file paths that were modified
        """
        if self.cache:
            self.cache.invalidate_paths(paths)

    def clear_cache(self) -> None:
        """Clear the entire tool cache."""
        if self.cache:
            self.cache.clear_all()

    def _invalidate_cache_for_write_tool(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Invalidate cache entries affected by write operations.

        Args:
            tool_name: Name of the write tool that was executed
            arguments: Arguments passed to the tool
        """
        if not self.cache:
            return

        # Extract paths from arguments based on tool type
        paths_to_invalidate: List[str] = []

        if tool_name == "write_file":
            if "path" in arguments:
                paths_to_invalidate.append(arguments["path"])
        elif tool_name == "edit_files":
            # edit_files can have multiple file edits
            if "edits" in arguments:
                for edit in arguments.get("edits", []):
                    if "path" in edit:
                        paths_to_invalidate.append(edit["path"])
            elif "path" in arguments:
                paths_to_invalidate.append(arguments["path"])
        elif tool_name == "execute_bash":
            # Bash commands can modify anything - invalidate all file-related caches
            self.cache.invalidate_by_tool("read_file")
            self.cache.invalidate_by_tool("list_directory")
            return
        elif tool_name in ("git", "docker"):
            # Git and docker operations can have wide-reaching effects
            self.cache.invalidate_by_tool("read_file")
            self.cache.invalidate_by_tool("list_directory")
            self.cache.invalidate_by_tool("code_search")
            return

        if paths_to_invalidate:
            self.cache.invalidate_paths(paths_to_invalidate)
