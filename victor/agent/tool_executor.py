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

"""Tool execution with retry logic, caching, metrics, and schema validation.

This module provides robust tool execution with:
- Pre-execution JSON schema validation (configurable strictness)
- Argument normalization and code correction
- Retry logic with exponential backoff
- Result caching for idempotent operations
- Execution metrics and statistics
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from typing import TYPE_CHECKING

from victor.agent.debug_logger import TRACE

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.safety import SafetyChecker, get_safety_checker
from victor.cache.tool_cache import ToolCache
from victor.core.errors import (
    ErrorCategory,
    ErrorHandler,
    ErrorInfo,
    ToolExecutionError,
    ToolTimeoutError,
    get_error_handler,
)
from victor.core.retry import (
    RetryContext,
    RetryExecutor,
    RetryStrategy,
    tool_retry_strategy,
)
from victor.tools.base import (
    AccessMode,
    BaseTool,
    Hook,
    HookError,
    ToolRegistry,
    ToolResult,
    ValidationResult,
)
from victor.tools.metadata_registry import (
    get_idempotent_tools as registry_get_idempotent_tools,
    get_cache_invalidating_tools as registry_get_cache_invalidating_tools,
)


class ValidationMode(Enum):
    """Mode for pre-execution argument validation.

    Controls how strictly the executor enforces JSON Schema validation
    before executing tools.

    Modes:
        STRICT: Validation errors block execution and return failure
        LENIENT: Validation errors are logged as warnings but execution proceeds
        OFF: No pre-execution validation (relies on tool's own validation)
    """

    STRICT = "strict"
    LENIENT = "lenient"
    OFF = "off"


if TYPE_CHECKING:
    from victor.agent.code_correction_middleware import CodeCorrectionMiddleware
    from victor.auth.rbac import RBACManager

logger = logging.getLogger(__name__)


class ToolExecutionResult:
    """Result of a tool execution attempt.

    Provides structured information about tool execution including success/failure,
    timing metrics, retry counts, and error details with correlation IDs for
    distributed tracing and debugging.
    """

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
        correlation_id: Optional[str] = None,
        error_info: Optional[ErrorInfo] = None,
    ):
        """Initialize tool execution result.

        Args:
            tool_name: Name of the executed tool
            success: Whether execution succeeded
            result: The tool's return value
            error: Error message if execution failed
            execution_time: Time taken in seconds
            cached: Whether result was from cache
            retries: Number of retry attempts made
            normalization_strategy: Strategy used for argument normalization
            correlation_id: Unique ID for tracking this execution across logs
            error_info: Structured error information if execution failed
        """
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.cached = cached
        self.retries = retries
        self.normalization_strategy = normalization_strategy
        self.correlation_id = correlation_id
        self.error_info = error_info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/logging.

        Returns:
            Dictionary with all execution result fields, including
            error details if present.
        """
        result = {
            "tool_name": self.tool_name,
            "success": self.success,
            "execution_time": self.execution_time,
            "cached": self.cached,
            "retries": self.retries,
            "correlation_id": self.correlation_id,
        }
        if self.error:
            result["error"] = self.error
        if self.error_info:
            result["error_details"] = self.error_info.to_dict()
        return result


class ToolExecutor:
    """Executes tools with retry logic, caching, and metrics.

    Responsibilities:
    - Execute tools with proper error handling
    - Implement retry logic with exponential backoff
    - Cache idempotent tool results
    - Normalize malformed arguments
    - Track execution metrics
    """

    @classmethod
    def is_cacheable_tool(cls, tool_name: str) -> bool:
        """Check if a tool is safe to cache (idempotent, read-only).

        Tools declare cacheability via @tool(access_mode=AccessMode.READONLY).

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool results can be cached
        """
        return tool_name in registry_get_idempotent_tools()

    @classmethod
    def is_cache_invalidating_tool(cls, tool_name: str) -> bool:
        """Check if a tool modifies state and should invalidate cache.

        Tools declare state modification via @tool(access_mode=AccessMode.WRITE/EXECUTE/MIXED).

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool modifies state
        """
        return tool_name in registry_get_cache_invalidating_tools()

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
        code_correction_middleware: Optional["CodeCorrectionMiddleware"] = None,
        enable_code_correction: bool = False,
        validation_mode: ValidationMode = ValidationMode.LENIENT,
        error_handler: Optional[ErrorHandler] = None,
        rbac_manager: Optional["RBACManager"] = None,
        current_user: Optional[str] = None,
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
            code_correction_middleware: Optional middleware for code validation/fixing
            enable_code_correction: Enable code correction for code-generating tools
            validation_mode: Pre-execution validation strictness (STRICT, LENIENT, OFF)
            error_handler: Centralized error handler for structured logging (uses global if None)
            rbac_manager: Optional RBAC manager for permission checks (disabled if None)
            current_user: Current user for RBAC checks (defaults to 'default_user')
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
        self.code_correction_middleware = code_correction_middleware
        self.enable_code_correction = enable_code_correction
        self.validation_mode = validation_mode
        self.error_handler = error_handler or get_error_handler()
        self.rbac_manager = rbac_manager
        self.current_user = current_user or "default_user"

        # Execution statistics
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._failed_signatures: set[Tuple[str, str]] = set()
        self._validation_failures: int = 0  # Track validation failures for metrics
        self._errors_by_category: Dict[str, int] = {}  # Track errors by category

    def update_context(self, **kwargs: Any) -> None:
        """Update the shared context passed to tools."""
        self.context.update(kwargs)

    def set_validation_mode(self, mode: ValidationMode) -> None:
        """Change the validation mode at runtime.

        Args:
            mode: New validation mode to use
        """
        self.validation_mode = mode
        logger.info("Validation mode changed to: %s", mode.value)

    def set_current_user(self, username: str) -> None:
        """Change the current user for RBAC checks.

        Args:
            username: New username for permission checks
        """
        self.current_user = username
        logger.info("Current user changed to: %s", username)

    def _check_rbac(
        self,
        tool: BaseTool,
        tool_name: str,
    ) -> Tuple[bool, Optional[str]]:
        """Check RBAC permissions for tool execution.

        Uses the tool's declared access_mode and category to determine
        if the current user has permission to execute it.

        Args:
            tool: The tool to check permissions for
            tool_name: Name of the tool

        Returns:
            Tuple of (allowed, denial_reason)
            - allowed: True if execution should proceed
            - denial_reason: Explanation if denied, None if allowed
        """
        # Skip RBAC if no manager configured (disabled by default)
        if self.rbac_manager is None:
            return True, None

        # Get tool metadata for permission check
        access_mode = getattr(tool, "access_mode", AccessMode.READONLY)
        category = getattr(tool, "category", "general")

        # Check access via RBAC manager
        if self.rbac_manager.check_tool_access(
            username=self.current_user,
            tool_name=tool_name,
            category=category,
            access_mode=access_mode,
        ):
            return True, None
        else:
            # Build informative denial message
            from victor.auth.rbac import Permission

            required_permission = Permission.from_access_mode(access_mode)
            return (
                False,
                f"RBAC denied: User '{self.current_user}' lacks '{required_permission.value}' "
                f"permission for tool '{tool_name}' (category: {category})",
            )

    def _validate_arguments(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Optional[ValidationResult]]:
        """Validate tool arguments against JSON Schema before execution.

        Performs pre-execution validation based on the configured validation_mode:
        - STRICT: Returns (False, result) on validation errors
        - LENIENT: Logs warnings and returns (True, result)
        - OFF: Returns (True, None) without validation

        Args:
            tool: The tool to validate arguments for
            arguments: Arguments to validate

        Returns:
            Tuple of (should_proceed, validation_result)
            - should_proceed: True if execution should continue
            - validation_result: ValidationResult or None if validation was skipped
        """
        if self.validation_mode == ValidationMode.OFF:
            return True, None

        try:
            validation = tool.validate_parameters_detailed(**arguments)

            if not validation.valid:
                self._validation_failures += 1
                error_summary = "; ".join(validation.errors[:3])  # Limit to first 3 errors
                if len(validation.errors) > 3:
                    error_summary += f" (+{len(validation.errors) - 3} more)"

                if self.validation_mode == ValidationMode.STRICT:
                    logger.error("STRICT validation failed for '%s': %s", tool.name, error_summary)
                    return False, validation
                else:  # LENIENT
                    logger.warning(
                        "Validation issues for '%s' (proceeding anyway): %s", tool.name, error_summary
                    )
                    return True, validation

            return True, validation

        except Exception as e:
            # Catch any exception during validation (RuntimeError, ValueError, etc)
            logger.warning("Validation error for '%s': %s", tool.name, str(e))
            # On validation system error, proceed in lenient mode, block in strict
            if self.validation_mode == ValidationMode.STRICT:
                return False, ValidationResult.failure([f"Validation system error: {e}"])
            return True, None

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        skip_cache: bool = False,
        context: Optional[Dict[str, Any]] = None,
        skip_normalization: bool = False,
    ) -> ToolExecutionResult:
        """Execute a tool with retry logic and caching.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            skip_cache: Skip cache lookup even for cacheable tools
            context: Optional context to pass to the tool (merged with default context)
            skip_normalization: Skip argument normalization (use when already normalized)

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

        # Normalize arguments (skip if already normalized by caller)
        if skip_normalization:
            normalized_args = arguments
            strategy = None
        else:
            normalized_args, strategy = self.normalizer.normalize_arguments(arguments, tool_name)

        # Code correction middleware - validate and fix code arguments
        if (
            self.enable_code_correction
            and self.code_correction_middleware is not None
            and self.code_correction_middleware.should_validate(tool_name)
        ):
            try:
                correction_result = self.code_correction_middleware.validate_and_fix(
                    tool_name, normalized_args
                )

                if correction_result.was_corrected:
                    # Apply the correction
                    normalized_args = self.code_correction_middleware.apply_correction(
                        normalized_args, correction_result
                    )
                    logger.info(
                        "Code auto-corrected for tool '%s': %d issues fixed",
                        tool_name,
                        len(correction_result.validation.errors)
                    )

                if not correction_result.validation.valid:
                    # Log validation errors but proceed - tool may still work
                    logger.warning(
                        "Code validation errors for tool '%s': %s",
                        tool_name,
                        list(correction_result.validation.errors)
                    )
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning("Code correction middleware failed: %s", str(e))

        # Check cache first
        if not skip_cache and self.cache:
            cached_result = self.cache.get(tool_name, normalized_args)
            if cached_result is not None:
                self._stats[tool_name]["cache_hits"] += 1
                logger.log(TRACE, "Cache hit for %s", tool_name)
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

        # Pre-execution schema validation
        should_proceed, validation_result = self._validate_arguments(tool, normalized_args)
        if not should_proceed:
            error_msg = "Argument validation failed"
            if validation_result and validation_result.errors:
                error_msg = f"Invalid arguments: {'; '.join(validation_result.errors[:3])}"
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=error_msg,
            )

        # Safety check for dangerous operations
        should_proceed, rejection_reason = await self.safety_checker.check_and_confirm(
            tool_name, normalized_args
        )
        if not should_proceed:
            logger.info("Tool execution blocked by safety check: %s", tool_name)
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=rejection_reason or "Operation cancelled by safety check",
            )

        # RBAC permission check (runs after safety check)
        rbac_allowed, rbac_denial = self._check_rbac(tool, tool_name)
        if not rbac_allowed:
            logger.warning(
                "Tool execution blocked by RBAC: %s for user %s", tool_name, self.current_user
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=rbac_denial or "Permission denied by RBAC",
            )

        # Execute with retry
        result, success, error, retries, error_info = await self._execute_with_retry(
            tool, normalized_args, exec_context
        )

        execution_time = time.time() - start_time
        self._stats[tool_name]["total_time"] += execution_time
        self._stats[tool_name]["retries"] += retries

        # Extract correlation ID from error_info if available
        correlation_id = error_info.correlation_id if error_info else None

        if success:
            self._stats[tool_name]["successes"] += 1

            # Cache successful results for cacheable tools (registry + fallback)
            if self.cache and self.is_cacheable_tool(tool_name):
                self.cache.set(tool_name, normalized_args, result)

            # Invalidate cache for tools that modify state (registry + fallback)
            if self.cache and self.is_cache_invalidating_tool(tool_name):
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
            correlation_id=correlation_id,
            error_info=error_info,
        )

    def _run_before_hooks(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Run before hooks for a tool execution.

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments

        Raises:
            HookError: If a critical hook fails
        """
        hooks = getattr(self.tools, "_before_hooks", [])
        for before_hook in hooks:
            hook_obj = before_hook if isinstance(before_hook, Hook) else None
            hook_name = hook_obj.name if hook_obj else getattr(before_hook, "__name__", "hook")
            is_critical = hook_obj.critical if hook_obj else False
            try:
                before_hook(tool_name, arguments)
            except Exception as e:
                if is_critical:
                    logger.error("Critical before hook '%s' failed: %s", hook_name, str(e))
                    raise HookError(hook_name, e, tool_name) from e
                else:
                    logger.warning("Before hook '%s' failed (non-critical): %s", hook_name, str(e))

    def _run_after_hooks(self, tool_name: str, result: Any) -> None:
        """Run after hooks for a tool execution.

        Args:
            tool_name: Name of the tool that was executed
            result: Tool execution result

        Raises:
            HookError: If a critical hook fails
        """
        hooks = getattr(self.tools, "_after_hooks", [])
        for after_hook in hooks:
            hook_obj = after_hook if isinstance(after_hook, Hook) else None
            hook_name = hook_obj.name if hook_obj else getattr(after_hook, "__name__", "hook")
            is_critical = hook_obj.critical if hook_obj else False
            try:
                after_hook(result)
            except Exception as e:
                if is_critical:
                    logger.error("Critical after hook '%s' failed: %s", hook_name, str(e))
                    raise HookError(hook_name, e, tool_name) from e
                else:
                    logger.warning("After hook '%s' failed (non-critical): %s", hook_name, str(e))

    async def _execute_with_retry(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Any, bool, Optional[str], int, Optional[ErrorInfo]]:
        """Execute a tool with retry logic from unified RetryStrategy.

        Uses the configured retry strategy for exponential backoff and
        retry decisions. Hooks are run before/after each attempt. Errors
        are tracked via the centralized ErrorHandler for structured logging.

        Args:
            tool: Tool to execute
            arguments: Tool arguments
            context: Context to pass to the tool

        Returns:
            Tuple of (result, success, error_message, retry_count, error_info)
            error_info is only populated on failure for structured error tracking
        """
        retry_context = RetryContext(
            max_attempts=getattr(self.retry_strategy, "max_attempts", self.max_retries)
        )
        last_error_info: Optional[ErrorInfo] = None

        while True:
            retry_context.attempt += 1

            try:
                # Run before hooks - critical hooks can block execution
                self._run_before_hooks(tool.name, arguments)

                # Execute the tool
                result = await tool.execute(_exec_ctx=context, **arguments)

                # Run after hooks - critical hooks can raise errors
                self._run_after_hooks(tool.name, result)

                # Handle ToolResult
                if isinstance(result, ToolResult):
                    if result.success:
                        self.retry_strategy.on_success(retry_context)
                        return result.output, True, None, retry_context.attempt - 1, None
                    else:
                        # Don't retry if tool explicitly returned failure
                        error = result.error or "Tool returned failure"
                        # Create structured error info for tool failures
                        error_info = self.error_handler.handle(
                            ToolExecutionError(
                                error,
                                tool_name=tool.name,
                                recovery_hint="Check tool arguments and try again.",
                            ),
                            context={"tool": tool.name, "arguments": arguments},
                        )
                        self._track_error_category(error_info.category)
                        return result.output, False, error, retry_context.attempt - 1, error_info
                else:
                    # Raw result (for tools that don't return ToolResult)
                    self.retry_strategy.on_success(retry_context)
                    return result, True, None, retry_context.attempt - 1, None

            except asyncio.TimeoutError as e:
                # Handle timeout specifically
                timeout_error = ToolTimeoutError(tool_name=tool.name)
                last_error_info = self.error_handler.handle(
                    timeout_error,
                    context={
                        "tool": tool.name,
                        "attempt": retry_context.attempt,
                        "arguments": arguments,
                    },
                )
                self._track_error_category(last_error_info.category)
                retry_context.record_exception(e)

                if self.retry_strategy.should_retry(retry_context):
                    self.retry_strategy.on_retry(retry_context)
                    delay = self.retry_strategy.get_delay(retry_context)
                    retry_context.record_delay(delay)
                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    self.retry_strategy.on_failure(retry_context)
                    return (
                        None,
                        False,
                        str(timeout_error),
                        retry_context.attempt - 1,
                        last_error_info,
                    )

            except (TimeoutError, asyncio.TimeoutError) as timeout_error:
                retry_context.record_exception(timeout_error)

                # Use centralized error handler for structured logging
                last_error_info = self.error_handler.handle(
                    timeout_error,
                    context={
                        "tool": tool.name,
                        "attempt": retry_context.attempt,
                        "max_attempts": retry_context.max_attempts,
                        "arguments": arguments,
                    },
                )
                self._track_error_category(last_error_info.category)

                if self.retry_strategy.should_retry(retry_context):
                    self.retry_strategy.on_retry(retry_context)
                    delay = self.retry_strategy.get_delay(retry_context)
                    logger.warning(
                        "[%s] Tool %s timeout - retrying in %.2fs "
                        "(attempt %d/%d): %s",
                        last_error_info.correlation_id,
                        tool.name,
                        delay,
                        retry_context.attempt,
                        retry_context.max_attempts,
                        str(timeout_error),
                    )
                    await asyncio.sleep(delay)
                    continue

                return (
                    None,
                    False,
                    str(timeout_error),
                    retry_context.attempt - 1,
                    last_error_info,
                )

            except (ToolExecutionError, ValueError, TypeError, KeyError, FileNotFoundError, OSError) as e:
                retry_context.record_exception(e)

                # Use centralized error handler for structured logging
                last_error_info = self.error_handler.handle(
                    e,
                    context={
                        "tool": tool.name,
                        "attempt": retry_context.attempt,
                        "max_attempts": retry_context.max_attempts,
                        "arguments": arguments,
                    },
                )
                self._track_error_category(last_error_info.category)

                logger.warning(
                    "[%s] Tool %s failed (attempt %d/%d): %s",
                    last_error_info.correlation_id,
                    tool.name,
                    retry_context.attempt,
                    retry_context.max_attempts,
                    str(e)
                )

                if self.retry_strategy.should_retry(retry_context):
                    self.retry_strategy.on_retry(retry_context)
                    delay = self.retry_strategy.get_delay(retry_context)
                    retry_context.record_delay(delay)

                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    self.retry_strategy.on_failure(retry_context)
                    # Include recovery hint in error message
                    error_msg = str(e)
                    if last_error_info.recovery_hint:
                        error_msg = f"{e}\nRecovery hint: {last_error_info.recovery_hint}"
                    return None, False, error_msg, retry_context.attempt - 1, last_error_info

    def _track_error_category(self, category: ErrorCategory) -> None:
        """Track error occurrences by category for metrics.

        Args:
            category: The error category to track
        """
        key = category.value
        self._errors_by_category[key] = self._errors_by_category.get(key, 0) + 1

    def has_failed_before(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Check if this exact tool call has failed before."""
        sig = (tool_name, str(sorted(arguments.items())))
        return sig in self._failed_signatures

    def clear_failed_signatures(self) -> None:
        """Clear the record of failed tool calls."""
        self._failed_signatures.clear()

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics for all tools.

        Returns:
            Dictionary with per-tool stats and global metrics including
            validation failures and error category breakdown.
        """
        stats = self._stats.copy()
        stats["_global"] = {
            "validation_failures": self._validation_failures,
            "validation_mode": self.validation_mode.value,
            "errors_by_category": self._errors_by_category.copy(),
            "recent_errors": [e.to_dict() for e in self.error_handler.get_recent_errors(5)],
        }
        return stats

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of recent errors for debugging.

        Returns:
            Dictionary with error counts by category and recent error details.
        """
        return {
            "errors_by_category": self._errors_by_category.copy(),
            "total_errors": sum(self._errors_by_category.values()),
            "recent_errors": [
                {
                    "correlation_id": e.correlation_id,
                    "category": e.category.value,
                    "message": e.message,
                    "recovery_hint": e.recovery_hint,
                }
                for e in self.error_handler.get_recent_errors(10)
            ],
        }

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
