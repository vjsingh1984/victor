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

"""Tool Execution Coordinator - Orchestrates tool call execution flow.

This module extracts tool execution coordination logic from AgentOrchestrator,
providing a focused interface for:

- Tool call validation (structure, names, permissions)
- Argument normalization (JSON parsing, type coercion)
- Budget enforcement during execution
- Result formatting and error feedback
- Failed signature tracking to avoid tight loops

Design Philosophy:
- Single Responsibility: Coordinates tool call execution only
- Composable: Works with existing ToolExecutor, ToolPipeline, ToolRegistry
- Observable: Provides execution metrics and callbacks
- Backward Compatible: Maintains API compatibility with orchestrator

Usage:
    coordinator = ToolExecutionCoordinator(
        tool_executor=executor,
        tool_registry=registry,
        argument_normalizer=normalizer,
        tool_adapter=adapter,
        tool_cache=cache,
    )

    # Handle tool calls from model
    results = await coordinator.handle_tool_calls(
        tool_calls=[{"name": "read", "arguments": {"path": "file.py"}}],
        context={"provider": provider, "model": model},
    )

    # Check execution stats
    stats = coordinator.get_execution_stats()
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from victor.agent.tool_executor import ToolExecutor, ToolExecutionResult
    from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
    from victor.agent.tool_calling import BaseToolCallingAdapter
    from victor.storage.cache.tool_cache import ToolCache
    from victor.tools.registry import ToolRegistry
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.agent.tool_output_formatter import ToolOutputFormatter

    # Type alias for convenience
    ToolCallingAdapter = BaseToolCallingAdapter

from victor.agent.coordinators.base_config import BaseCoordinatorConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionConfig(BaseCoordinatorConfig):
    """Configuration for ToolExecutionCoordinator.

    Inherits common configuration from BaseCoordinatorConfig:
        enabled: Whether the coordinator is enabled
        timeout: Default timeout in seconds for operations
        max_retries: Maximum number of retry attempts for failed operations
        retry_enabled: Whether retry logic is enabled
        log_level: Logging level for coordinator messages
        enable_metrics: Whether to collect metrics

    Attributes:
        max_retry_attempts: Maximum retry attempts per tool call
        retry_base_delay: Base delay for exponential backoff (seconds)
        retry_max_delay: Maximum delay for retry (seconds)
        enable_failed_signature_tracking: Track failing calls to avoid loops
        enable_result_formatting: Format tool results with clear boundaries
        enable_error_feedback: Send error details back to model
        max_tool_budget: Maximum number of tool calls per session
    """

    max_retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0
    enable_failed_signature_tracking: bool = True
    enable_result_formatting: bool = True
    enable_error_feedback: bool = True
    max_tool_budget: int = 50


@dataclass
class ToolCallResult:
    """Result of a single tool call execution.

    Attributes:
        name: Tool name that was executed
        success: Whether execution was successful
        result: Tool output (if successful)
        error: Error message (if failed)
        elapsed: Execution time in seconds
        cached: Whether result came from cache
        skipped: Whether execution was skipped
        skip_reason: Reason for skipping (if skipped)
    """

    name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    elapsed: float = 0.0
    cached: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class ToolExecutionStats:
    """Statistics for tool execution.

    Attributes:
        total_calls: Total number of tool calls handled
        successful_calls: Number of successful executions
        failed_calls: Number of failed executions
        skipped_calls: Number of skipped calls
        budget_used: Number of budget units consumed
        budget_remaining: Number of budget units remaining
        failed_signatures: Set of failed call signatures tracked
        execution_times_ms: List of execution times in milliseconds
    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    skipped_calls: int = 0
    budget_used: int = 0
    budget_remaining: int = 50
    failed_signatures: Set[Tuple[str, str]] = field(default_factory=set)
    execution_times_ms: List[float] = field(default_factory=list)

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    def get_avg_execution_time_ms(self) -> float:
        """Calculate average execution time in milliseconds."""
        if not self.execution_times_ms:
            return 0.0
        return sum(self.execution_times_ms) / len(self.execution_times_ms)


@dataclass
class ExecutionContext:
    """Context for tool execution.

    Attributes:
        code_manager: Code manager for file operations
        provider: LLM provider instance
        model: Model name
        tool_registry: Tool registry for tool access
        workflow_registry: Workflow registry for tool access
        settings: Application settings
        session_state: Optional session state manager
        conversation_state: Optional conversation state machine
    """

    code_manager: Optional[Any] = None
    provider: Optional[Any] = None
    model: Optional[str] = None
    tool_registry: Optional["ToolRegistry"] = None
    workflow_registry: Optional[Any] = None
    settings: Optional[Any] = None
    session_state: Optional[Any] = None
    conversation_state: Optional[Any] = None


# Protocol for tool access checking
@dataclass
class ToolAccessDecision:
    """Result of a tool access check.

    Attributes:
        allowed: Whether tool is allowed
        reason: Reason for denial (if not allowed)
    """

    allowed: bool
    reason: Optional[str] = None


class ToolExecutionCoordinator:
    """Coordinates tool call execution flow.

    This class extracts tool execution logic from the orchestrator:
    - Validates tool call structure and permissions
    - Normalizes arguments (JSON parsing, type coercion)
    - Executes tools via ToolExecutor
    - Formats results with clear boundaries
    - Tracks failed signatures to avoid loops
    - Enforces budget constraints

    Example:
        coordinator = ToolExecutionCoordinator(
            tool_executor=executor,
            tool_registry=registry,
            argument_normalizer=normalizer,
        )

        # In orchestrator's _handle_tool_calls:
        results = await coordinator.handle_tool_calls(
            tool_calls=tool_calls,
            context=self._build_execution_context(),
        )
    """

    def __init__(
        self,
        tool_executor: "ToolExecutor",
        tool_registry: "ToolRegistry",
        argument_normalizer: "ArgumentNormalizer",
        tool_adapter: "ToolCallingAdapter",
        tool_cache: Optional["ToolCache"] = None,
        sanitizer: Optional["ResponseSanitizer"] = None,
        formatter: Optional["ToolOutputFormatter"] = None,
        config: Optional[ToolExecutionConfig] = None,
        # Callbacks
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_complete: Optional[Callable[[ToolCallResult], None]] = None,
        on_budget_warning: Optional[Callable[[int, int], None]] = None,
        # Tool access control
        tool_access_checker: Optional[Callable[[str], ToolAccessDecision]] = None,
    ) -> None:
        """Initialize the ToolExecutionCoordinator.

        Args:
            tool_executor: Executor for running tools
            tool_registry: Registry for tool access control
            argument_normalizer: Normalizer for tool arguments
            tool_adapter: Adapter for provider-specific argument handling
            tool_cache: Optional cache for tool results
            sanitizer: Optional sanitizer for validating tool names
            formatter: Optional formatter for tool output
            config: Configuration options
            on_tool_start: Callback when tool execution starts
            on_tool_complete: Callback when tool execution completes
            on_budget_warning: Callback when budget is running low
            tool_access_checker: Function to check if tool is accessible
        """
        self._executor = tool_executor
        self._registry = tool_registry
        self._argument_normalizer = argument_normalizer
        self._tool_adapter = tool_adapter
        self._cache = tool_cache
        self._sanitizer = sanitizer
        self._formatter = formatter
        self._config = config or ToolExecutionConfig()
        self._tool_access_checker = tool_access_checker

        # Callbacks
        self._on_tool_start = on_tool_start
        self._on_tool_complete = on_tool_complete
        self._on_budget_warning = on_budget_warning

        # Internal state
        self._stats = ToolExecutionStats(budget_remaining=self._config.max_tool_budget)

        logger.debug(
            f"ToolExecutionCoordinator initialized with budget={self._config.max_tool_budget}, "
            f"retry_enabled={self._config.retry_enabled}"
        )

    # =====================================================================
    # Main Entry Point
    # =====================================================================

    async def handle_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[ExecutionContext] = None,
    ) -> List[Dict[str, Any]]:
        """Handle tool calls from the model.

        This is the main entry point that orchestrates the complete flow:
        1. Validate tool calls (structure, names, permissions)
        2. Normalize arguments (JSON parsing, type coercion)
        3. Execute tools via ToolExecutor
        4. Format results with clear boundaries
        5. Track failed signatures to avoid loops

        Args:
            tool_calls: List of tool call requests from model
            context: Execution context for tools

        Returns:
            List of tool call results with success/error information
        """
        if not tool_calls:
            return []

        context = context or ExecutionContext()
        results: List[Dict[str, Any]] = []

        for tool_call in tool_calls:
            result = await self._handle_single_call(tool_call, context)
            if result:
                results.append(result)

            # Check budget
            if self._stats.budget_remaining <= 0:
                logger.warning("Tool budget exhausted, skipping remaining calls")
                break

        return results

    # =====================================================================
    # Single Call Handling
    # =====================================================================

    async def _handle_single_call(
        self,
        tool_call: Dict[str, Any],
        context: ExecutionContext,
    ) -> Optional[Dict[str, Any]]:
        """Handle a single tool call.

        Args:
            tool_call: Tool call dictionary with 'name' and 'arguments'
            context: Execution context

        Returns:
            Result dictionary or None if validation failed
        """
        # Validate structure
        if not isinstance(tool_call, dict):
            logger.warning(f"Skipping invalid tool call (not a dict): {tool_call}")
            return None  # type: ignore[unreachable]

        tool_name = tool_call.get("name")
        if not tool_name:
            return self._create_error_result(
                "",
                "Tool call missing name. Each tool call must include a 'name' field.",
            )

        # Validate tool name format
        if self._sanitizer and not self._sanitizer.is_valid_tool_name(tool_name):
            return self._create_error_result(
                tool_name,
                f"Invalid tool name '{tool_name}'. This tool does not exist. "
                "Use only tools from the provided tool list.",
            )

        # Resolve legacy/alias names to canonical form
        canonical_tool_name = self._resolve_tool_name(tool_name)

        # Check tool access
        access_decision = self._check_tool_access(canonical_tool_name)
        if not access_decision.allowed:
            return self._create_error_result(
                tool_name,
                f"Tool '{tool_name}' is not available. {access_decision.reason or ''}",
            )

        # Check budget
        if self._stats.budget_remaining <= 0:
            return self._create_error_result(
                tool_name,
                f"Tool budget reached ({self._config.max_tool_budget}). No more tool calls allowed.",
            )

        # Extract and normalize arguments
        tool_args = tool_call.get("arguments", {})
        normalized_args = self._normalize_arguments(tool_name, tool_args)

        # Check for repeated failures
        signature = self._get_call_signature(canonical_tool_name, normalized_args)
        if signature in self._stats.failed_signatures:
            logger.debug(f"Skipping repeated failing call: {canonical_tool_name}")
            return self._create_error_result(
                tool_name,
                f"Skipping repeated failing call to '{tool_name}' with same arguments",
            )

        # Execute the tool
        return await self._execute_tool(
            canonical_tool_name,
            normalized_args,
            context,
            signature,
        )

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: ExecutionContext,
        signature: Tuple[str, str],
    ) -> Dict[str, Any]:
        """Execute a tool with error handling.

        Args:
            tool_name: Canonical tool name
            tool_args: Normalized arguments
            context: Execution context
            signature: Call signature for failure tracking

        Returns:
            Result dictionary
        """
        start = time.monotonic()

        # Notify callback
        if self._on_tool_start:
            self._on_tool_start(tool_name, tool_args)

        # Build execution context dict for tool executor
        exec_context = self._build_exec_context(context)

        try:
            # Execute with retry logic
            result, success, error_msg = await self._execute_with_retry(
                tool_name, tool_args, exec_context
            )

            elapsed = time.monotonic() - start
            elapsed_ms = elapsed * 1000

            # Update stats
            self._stats.total_calls += 1
            self._stats.budget_used += 1
            self._stats.budget_remaining -= 1
            self._stats.execution_times_ms.append(elapsed_ms)

            if success:
                self._stats.successful_calls += 1
            else:
                self._stats.failed_calls += 1
                self._stats.failed_signatures.add(signature)

            # Build result
            tool_result = ToolCallResult(
                name=tool_name,
                success=success,
                result=result.result if success else None,
                error=error_msg,
                elapsed=elapsed,
            )

            # Notify callback
            if self._on_tool_complete:
                self._on_tool_complete(tool_result)

            # Return result dict
            return {
                "name": tool_name,
                "success": success,
                "result": tool_result.result,
                "error": error_msg,
                "elapsed": elapsed,
                "args": tool_args,
            }

        except Exception as e:
            elapsed = time.monotonic() - start
            elapsed_ms = elapsed * 1000

            logger.exception(f"Exception executing tool '{tool_name}': {e}")

            # Update stats
            self._stats.total_calls += 1
            self._stats.budget_used += 1
            self._stats.budget_remaining -= 1
            self._stats.failed_calls += 1
            self._stats.failed_signatures.add(signature)
            self._stats.execution_times_ms.append(elapsed_ms)

            return {
                "name": tool_name,
                "success": False,
                "error": str(e),
                "elapsed": elapsed,
                "args": tool_args,
            }

    # =====================================================================
    # Retry Logic
    # =====================================================================

    async def _execute_with_retry(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Any, bool, Optional[str]]:
        """Execute a tool with retry logic and exponential backoff.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            context: Execution context

        Returns:
            Tuple of (result, success, error_message or None)
        """
        # Try cache first
        if self._cache:
            cached = self._cache.get(tool_name, tool_args)
            if cached is not None:
                logger.debug(f"Cache hit for tool '{tool_name}'")
                return cached, True, None

        retry_enabled = self._config.retry_enabled
        max_attempts = self._config.max_retry_attempts if retry_enabled else 1
        base_delay = self._config.retry_base_delay
        max_delay = self._config.retry_max_delay

        last_error = None
        for attempt in range(max_attempts):
            try:
                result = await self._executor.execute(
                    tool_name=tool_name,
                    arguments=tool_args,
                    context=context,
                    skip_normalization=True,  # Already normalized
                )

                if result.success:
                    # Cache successful result
                    if self._cache:
                        self._cache.set(tool_name, tool_args, result)
                        self._invalidate_related_cache(tool_name, tool_args)

                    if attempt > 0:
                        logger.info(
                            f"Tool '{tool_name}' succeeded on retry attempt "
                            f"{attempt + 1}/{max_attempts}"
                        )

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
                        f"Tool '{tool_name}' transient error (attempt {attempt + 1}/{max_attempts}): "
                        f"{e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Tool '{tool_name}' failed after {max_attempts} attempts: {e}")
                    return None, False, last_error

        # Should not reach here
        return None, False, last_error or "Unknown error"

    # =====================================================================
    # Helper Methods
    # =====================================================================

    def _resolve_tool_name(self, tool_name: str) -> str:
        """Resolve tool alias to canonical name.

        Args:
            tool_name: Original tool name (may be alias)

        Returns:
            Canonical tool name
        """
        try:
            from victor.tools.decorators import resolve_tool_name

            return resolve_tool_name(tool_name)
        except Exception:
            return tool_name

    def _check_tool_access(self, tool_name: str) -> ToolAccessDecision:
        """Check if tool is accessible.

        Args:
            tool_name: Tool name to check

        Returns:
            Access decision with reason if denied
        """
        if self._tool_access_checker:
            return self._tool_access_checker(tool_name)

        if self._registry:
            if not self._registry.is_tool_enabled(tool_name):
                return ToolAccessDecision(
                    allowed=False,
                    reason=f"Tool '{tool_name}' is not enabled or not registered.",
                )

        return ToolAccessDecision(allowed=True)

    def _normalize_arguments(
        self,
        tool_name: str,
        tool_args: Any,
    ) -> Dict[str, Any]:
        """Normalize tool arguments.

        Args:
            tool_name: Tool name
            tool_args: Raw arguments (may be string or dict)

        Returns:
            Normalized arguments dict
        """
        # Handle string arguments
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                try:
                    tool_args = ast.literal_eval(tool_args)
                except Exception:
                    tool_args = {"value": tool_args}
        elif tool_args is None:
            tool_args = {}

        # Normalize via argument normalizer
        normalized_args, strategy = self._argument_normalizer.normalize_arguments(
            tool_args, tool_name
        )

        # Apply adapter-based normalization
        normalized_args = self._tool_adapter.normalize_arguments(normalized_args, tool_name)

        result: Dict[str, Any] = normalized_args
        return result

    def _build_exec_context(self, context: ExecutionContext) -> Dict[str, Any]:
        """Build execution context dict for tool executor.

        Args:
            context: ExecutionContext object

        Returns:
            Dict context for tool executor
        """
        return {
            "code_manager": context.code_manager,
            "provider": context.provider,
            "model": context.model,
            "tool_registry": context.tool_registry,
            "workflow_registry": context.workflow_registry,
            "settings": context.settings,
        }

    def _get_call_signature(self, tool_name: str, args: Dict[str, Any]) -> Tuple[str, str]:
        """Get call signature for failure tracking.

        Args:
            tool_name: Tool name
            args: Tool arguments

        Returns:
            Signature tuple
        """
        try:
            return (tool_name, json.dumps(args, sort_keys=True, default=str))
        except Exception:
            return (tool_name, str(args))

    def _invalidate_related_cache(self, tool_name: str, tool_args: Dict[str, Any]) -> None:
        """Invalidate related cache entries after a mutating tool.

        Args:
            tool_name: Tool that was executed
            tool_args: Arguments for the tool
        """
        if not self._cache:
            return

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
                self._cache.invalidate_paths(touched_paths)
            else:
                namespaces_to_clear = [
                    "code_search",
                    "semantic_code_search",
                    "list_directory",
                ]
                self._cache.clear_namespaces(namespaces_to_clear)

    def _create_error_result(self, tool_name: str, error: str) -> Dict[str, Any]:
        """Create an error result dictionary.

        Args:
            tool_name: Tool name
            error: Error message

        Returns:
            Error result dict
        """
        self._stats.total_calls += 1
        self._stats.failed_calls += 1

        return {
            "name": tool_name,
            "success": False,
            "result": None,
            "error": error,
            "elapsed": 0.0,
        }

    # =====================================================================
    # Result Formatting
    # =====================================================================

    def format_tool_output(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        output: Any,
    ) -> str:
        """Format tool output with clear boundaries.

        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool
            output: Tool output

        Returns:
            Formatted output string
        """
        if self._formatter:
            return self._formatter.format_tool_output(tool_name, tool_args, output)

        # Default formatting
        return f"TOOL_OUTPUT: {tool_name}\n{output}"

    # =====================================================================
    # Statistics and State Management
    # =====================================================================

    def get_execution_stats(self) -> ToolExecutionStats:
        """Get execution statistics.

        Returns:
            Current execution stats
        """
        return self._stats

    def reset_stats(self, new_budget: Optional[int] = None) -> None:
        """Reset execution statistics.

        Args:
            new_budget: New budget to set, or use default from config
        """
        self._stats = ToolExecutionStats(
            budget_remaining=new_budget or self._config.max_tool_budget
        )

    def clear_failed_signatures(self) -> None:
        """Clear the failed signatures cache."""
        self._stats.failed_signatures.clear()

    def consume_budget(self, amount: int = 1) -> None:
        """Manually consume budget.

        Args:
            amount: Amount to consume
        """
        self._stats.budget_used += amount
        self._stats.budget_remaining -= amount

        if (
            self._on_budget_warning
            and self._stats.budget_remaining < self._config.max_tool_budget * 0.2
        ):
            self._on_budget_warning(self._stats.budget_remaining, self._config.max_tool_budget)


def create_tool_execution_coordinator(
    tool_executor: "ToolExecutor",
    tool_registry: "ToolRegistry",
    argument_normalizer: "ArgumentNormalizer",
    tool_adapter: "ToolCallingAdapter",
    tool_cache: Optional["ToolCache"] = None,
    config: Optional[ToolExecutionConfig] = None,
) -> ToolExecutionCoordinator:
    """Factory function to create a ToolExecutionCoordinator.

    Args:
        tool_executor: Executor for running tools
        tool_registry: Registry for tool access control
        argument_normalizer: Normalizer for tool arguments
        tool_adapter: Adapter for provider-specific argument handling
        tool_cache: Optional cache for tool results
        config: Configuration options

    Returns:
        Configured ToolExecutionCoordinator instance
    """
    return ToolExecutionCoordinator(
        tool_executor=tool_executor,
        tool_registry=tool_registry,
        argument_normalizer=argument_normalizer,
        tool_adapter=tool_adapter,
        tool_cache=tool_cache,
        config=config,
    )


__all__ = [
    "ToolExecutionCoordinator",
    "ToolExecutionConfig",
    "ToolCallResult",
    "ToolExecutionStats",
    "ExecutionContext",
    "ToolAccessDecision",
    "create_tool_execution_coordinator",
]
