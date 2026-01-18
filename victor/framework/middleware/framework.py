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

"""Framework-level middleware implementations.

This module provides common middleware that all verticals can use:

1. LoggingMiddleware - Log all tool calls for audit/debugging
2. SecretMaskingMiddleware - Mask secrets in tool results
3. MetricsMiddleware - Record tool execution metrics
4. GitSafetyMiddleware - Block dangerous git operations
5. OutputValidationMiddleware - Validate and optionally fix tool outputs
6. CacheMiddleware - Cache tool execution results
7. RateLimitMiddleware - Rate limit tool execution

Example usage:
    from victor.framework.middleware import (
        LoggingMiddleware,
        SecretMaskingMiddleware,
        MetricsMiddleware,
        GitSafetyMiddleware,
        OutputValidationMiddleware,
    )

    # Add to vertical's middleware list
    middleware = [
        LoggingMiddleware(log_level=logging.DEBUG),
        SecretMaskingMiddleware(replacement="[REDACTED]"),
        MetricsMiddleware(enable_timing=True),
        GitSafetyMiddleware(block_dangerous=True),
    ]

    # Create custom validation middleware
    class JsonValidator:
        def validate(self, content, context=None):
            try:
                json.loads(content)
                return ContentValidationResult(is_valid=True)
            except json.JSONDecodeError as e:
                return ContentValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(message=str(e))]
                )

    json_middleware = OutputValidationMiddleware(
        validator=JsonValidator(),
        applicable_tools={"generate_json", "create_config"},
        argument_names={"content", "json_data"},
    )
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable

from victor.core.vertical_types import MiddlewarePriority, MiddlewareResult
from victor.core.verticals.protocols import MiddlewareProtocol

logger = logging.getLogger(__name__)


# =============================================================================
# Output Validation Types
# =============================================================================


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # May need attention
    ERROR = "error"  # Must be fixed
    CRITICAL = "critical"  # Blocks execution


@dataclass
class ValidationIssue:
    """A single validation issue found during content validation.

    Attributes:
        message: Human-readable description of the issue
        severity: How serious the issue is
        location: Optional location info (line, column, path, etc.)
        suggestion: Optional suggested fix
        code: Optional error/warning code for programmatic handling
    """

    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    location: Optional[str] = None
    suggestion: Optional[str] = None
    code: Optional[str] = None


@dataclass
class ContentValidationResult:
    """Result of content validation.

    Renamed from ValidationResult to be semantically distinct:
    - ToolValidationResult (victor.tools.base): Tool parameter validation
    - ConfigValidationResult (victor.core.validation): Configuration validation
    - ContentValidationResult (here): Content validation with fixed_content
    - ParameterValidationResult (victor.agent.parameter_enforcer): Parameter enforcement
    - CodeValidationResult (victor.evaluation.correction.types): Code validation

    Attributes:
        is_valid: Whether the content passed validation
        issues: List of issues found (empty if valid)
        fixed_content: Optional auto-fixed content (if validator supports fixing)
        metadata: Additional validation metadata
    """

    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    fixed_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """Count of ERROR and CRITICAL severity issues."""
        return sum(
            1
            for issue in self.issues
            if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        """Count of WARNING severity issues."""
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)

    @property
    def has_fix(self) -> bool:
        """Whether a fix is available."""
        return self.fixed_content is not None


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for content validators.

    Validators check content (code, data, configuration) for issues
    and optionally provide fixes.

    Example implementations:
    - Python linter (using ast, pylint, ruff)
    - JSON schema validator
    - YAML syntax checker
    - SQL syntax validator
    - Pydantic model validator
    """

    def validate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ContentValidationResult:
        """Validate content and return results.

        Args:
            content: The content to validate
            context: Optional context (tool_name, file_path, etc.)

        Returns:
            ContentValidationResult with validation status and any issues
        """
        ...


class FixableValidatorProtocol(ValidatorProtocol, Protocol):
    """Extended protocol for validators that can auto-fix issues.

    Inherits from ValidatorProtocol and adds the ability to
    automatically fix detected issues.
    """

    def fix(
        self,
        content: str,
        issues: List[ValidationIssue],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Attempt to fix issues in content.

        Args:
            content: Original content with issues
            issues: Issues to fix
            context: Optional context

        Returns:
            Fixed content (or original if unfixable)
        """
        ...


# =============================================================================
# Logging Middleware
# =============================================================================


class LoggingMiddleware(MiddlewareProtocol):
    """Log all tool calls for audit/debugging.

    This middleware provides comprehensive logging of tool calls including:
    - Tool name and arguments (before execution)
    - Execution result and timing (after execution)
    - Optional argument sanitization for sensitive values

    Example:
        middleware = LoggingMiddleware(
            log_level=logging.DEBUG,
            include_arguments=True,
            exclude_tools={"read_file"},  # Skip verbose tools
        )
    """

    def __init__(
        self,
        log_level: int = logging.DEBUG,
        include_arguments: bool = True,
        include_results: bool = False,
        sanitize_arguments: bool = True,
        exclude_tools: Optional[Set[str]] = None,
        logger_name: Optional[str] = None,
    ):
        """Initialize the logging middleware.

        Args:
            log_level: Log level to use (default: DEBUG)
            include_arguments: Include arguments in log messages
            include_results: Include results in log messages (can be verbose)
            sanitize_arguments: Sanitize sensitive argument values
            exclude_tools: Tools to exclude from logging
            logger_name: Custom logger name (default: module logger)
        """
        self._log_level = log_level
        self._include_arguments = include_arguments
        self._include_results = include_results
        self._sanitize_arguments = sanitize_arguments
        self._exclude_tools = exclude_tools or set()
        self._logger = logging.getLogger(logger_name) if logger_name else logger
        self._start_times: Dict[str, float] = {}

    def _sanitize_value(self, key: str, value: Any) -> Any:
        """Sanitize sensitive argument values.

        Args:
            key: Argument key name
            value: Argument value

        Returns:
            Sanitized value (or original if not sensitive)
        """
        if not self._sanitize_arguments:
            return value

        # List of sensitive keys to redact
        sensitive_keys = {
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "key",
            "credential",
            "auth",
        }

        key_lower = key.lower()
        for sensitive in sensitive_keys:
            if sensitive in key_lower:
                return "[REDACTED]"

        # Truncate very long string values
        if isinstance(value, str) and len(value) > 500:
            return f"{value[:200]}... (truncated, {len(value)} chars total)"

        return value

    def _format_arguments(self, arguments: Dict[str, Any]) -> str:
        """Format arguments for logging.

        Args:
            arguments: Tool arguments

        Returns:
            Formatted string representation
        """
        if not self._include_arguments:
            return ""

        sanitized = {k: self._sanitize_value(k, v) for k, v in arguments.items()}
        return f" args={sanitized}"

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Log tool call before execution.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult (always proceeds)
        """
        if tool_name in self._exclude_tools:
            return MiddlewareResult()

        # Track start time for timing
        call_id = f"{tool_name}:{id(arguments)}"
        self._start_times[call_id] = time.time()

        args_str = self._format_arguments(arguments)
        self._logger.log(self._log_level, f"Tool call: {tool_name}{args_str}")

        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Log tool result after execution.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            None (no modification)
        """
        if tool_name in self._exclude_tools:
            return None

        # Calculate duration
        call_id = f"{tool_name}:{id(arguments)}"
        start_time = self._start_times.pop(call_id, None)
        duration_ms = (time.time() - start_time) * 1000 if start_time else 0

        status = "success" if success else "failed"
        result_str = ""
        if self._include_results and result:
            result_preview = str(result)[:200]
            if len(str(result)) > 200:
                result_preview += "..."
            result_str = f" result={result_preview}"

        self._logger.log(
            self._log_level,
            f"Tool {status}: {tool_name} ({duration_ms:.1f}ms){result_str}",
        )

        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Logging runs last (after other processing).

        Returns:
            DEFERRED priority
        """
        return MiddlewarePriority.DEFERRED

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            None (applies to all tools)
        """
        return None


# =============================================================================
# Secret Masking Middleware
# =============================================================================


class SecretMaskingMiddleware(MiddlewareProtocol):
    """Mask secrets in tool results.

    This middleware automatically detects and redacts secrets, credentials,
    and sensitive data in tool output using the core safety/secrets module.

    Example:
        middleware = SecretMaskingMiddleware(
            replacement="[REDACTED]",
            mask_in_arguments=True,  # Also mask secrets in inputs
        )
    """

    def __init__(
        self,
        replacement: str = "[REDACTED]",
        mask_in_arguments: bool = False,
        include_low_severity: bool = False,
    ):
        """Initialize the secret masking middleware.

        Args:
            replacement: Text to replace secrets with
            mask_in_arguments: Also mask secrets in input arguments
            include_low_severity: Include low-severity matches
        """
        self._replacement = replacement
        self._mask_in_arguments = mask_in_arguments
        self._include_low_severity = include_low_severity

    def _mask_content(self, content: str) -> str:
        """Mask secrets in content.

        Args:
            content: Text content to scan and mask

        Returns:
            Content with secrets replaced
        """
        from victor.security.safety.secrets import mask_secrets

        return mask_secrets(content, replacement=self._replacement)

    def _mask_dict_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively mask secrets in dictionary values.

        Args:
            data: Dictionary to process

        Returns:
            Dictionary with masked values
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._mask_content(value)
            elif isinstance(value, dict):
                result[key] = self._mask_dict_values(value)
            elif isinstance(value, list):
                result[key] = [self._mask_content(v) if isinstance(v, str) else v for v in value]
            else:
                result[key] = value
        return result

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Optionally mask secrets in input arguments.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult with potentially masked arguments
        """
        if not self._mask_in_arguments:
            return MiddlewareResult()

        # Mask secrets in string arguments
        masked_args = self._mask_dict_values(arguments)

        # Only return modified if there were changes
        if masked_args != arguments:
            return MiddlewareResult(
                proceed=True,
                modified_arguments=masked_args,
                metadata={"secrets_masked_in_input": True},
            )

        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Mask secrets in tool results.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            Masked result (or None if no secrets found)
        """
        if result is None:
            return None

        # Handle string results
        if isinstance(result, str):
            masked = self._mask_content(result)
            if masked != result:
                return masked
            return None

        # Handle dict results
        if isinstance(result, dict):
            masked = self._mask_dict_values(result)
            if masked != result:
                return masked
            return None

        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Secret masking runs with HIGH priority to mask secrets early.

        Returns:
            HIGH priority
        """
        return MiddlewarePriority.HIGH

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            None (applies to all tools)
        """
        return None


# =============================================================================
# Metrics Middleware
# =============================================================================


@dataclass
class ToolMetrics:
    """Metrics for a tool execution.

    Attributes:
        tool_name: Name of the tool
        call_count: Number of calls
        success_count: Number of successful calls
        failure_count: Number of failed calls
        total_duration_ms: Total execution time in milliseconds
        min_duration_ms: Minimum execution time
        max_duration_ms: Maximum execution time
    """

    tool_name: str
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count


class MetricsMiddleware(MiddlewareProtocol):
    """Record tool execution metrics.

    This middleware tracks:
    - Call counts (total, success, failure)
    - Execution timing (min, max, avg)
    - Success rates

    Metrics can be exported to various backends (Prometheus, JSON, etc.)

    Example:
        middleware = MetricsMiddleware(enable_timing=True)

        # ... execute tools ...

        # Get metrics summary
        summary = middleware.get_summary()
        for tool, metrics in summary.items():
            print(f"{tool}: {metrics.call_count} calls, "
                  f"{metrics.avg_duration_ms:.1f}ms avg")
    """

    def __init__(
        self,
        enable_timing: bool = True,
        callback: Optional[Callable[[str, ToolMetrics], None]] = None,
    ):
        """Initialize the metrics middleware.

        Args:
            enable_timing: Track execution timing
            callback: Optional callback for each recorded metric
        """
        self._enable_timing = enable_timing
        self._callback = callback
        self._metrics: Dict[str, ToolMetrics] = {}
        self._start_times: Dict[str, float] = {}

    def _get_or_create_metrics(self, tool_name: str) -> ToolMetrics:
        """Get or create metrics for a tool.

        Args:
            tool_name: Tool name

        Returns:
            ToolMetrics instance
        """
        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolMetrics(tool_name=tool_name)
        return self._metrics[tool_name]

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Record call start time.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult (always proceeds)
        """
        if self._enable_timing:
            call_id = f"{tool_name}:{id(arguments)}"
            self._start_times[call_id] = time.time()

        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Record execution metrics.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            None (no modification)
        """
        metrics = self._get_or_create_metrics(tool_name)
        metrics.call_count += 1

        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1

        # Calculate duration if timing is enabled
        if self._enable_timing:
            call_id = f"{tool_name}:{id(arguments)}"
            start_time = self._start_times.pop(call_id, None)
            if start_time:
                duration_ms = (time.time() - start_time) * 1000
                metrics.total_duration_ms += duration_ms
                metrics.min_duration_ms = min(metrics.min_duration_ms, duration_ms)
                metrics.max_duration_ms = max(metrics.max_duration_ms, duration_ms)

        # Invoke callback if provided
        if self._callback:
            try:
                self._callback(tool_name, metrics)
            except Exception:
                pass  # Don't fail on callback errors

        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Metrics runs with LOW priority (after main processing).

        Returns:
            LOW priority
        """
        return MiddlewarePriority.LOW

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            None (applies to all tools)
        """
        return None

    def get_metrics(self, tool_name: str) -> Optional[ToolMetrics]:
        """Get metrics for a specific tool.

        Args:
            tool_name: Tool name

        Returns:
            ToolMetrics or None if not recorded
        """
        return self._metrics.get(tool_name)

    def get_summary(self) -> Dict[str, ToolMetrics]:
        """Get summary of all metrics.

        Returns:
            Dict mapping tool names to their metrics
        """
        return dict(self._metrics)

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._start_times.clear()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-compatible metrics text
        """
        lines = []
        lines.append("# HELP tool_calls_total Total number of tool calls")
        lines.append("# TYPE tool_calls_total counter")
        for name, m in self._metrics.items():
            lines.append(f'tool_calls_total{{tool="{name}"}} {m.call_count}')

        lines.append("# HELP tool_calls_success_total Successful tool calls")
        lines.append("# TYPE tool_calls_success_total counter")
        for name, m in self._metrics.items():
            lines.append(f'tool_calls_success_total{{tool="{name}"}} {m.success_count}')

        lines.append("# HELP tool_duration_ms_avg Average tool duration in ms")
        lines.append("# TYPE tool_duration_ms_avg gauge")
        for name, m in self._metrics.items():
            lines.append(f'tool_duration_ms_avg{{tool="{name}"}} {m.avg_duration_ms:.2f}')

        return "\n".join(lines)


# =============================================================================
# Git Safety Middleware
# =============================================================================


class GitSafetyMiddleware(MiddlewareProtocol):
    """Block dangerous git operations.

    This middleware validates git operations before execution and can block
    dangerous operations like force push, hard reset, or cleaning untracked files.

    Designed to be used by any vertical that works with git (Coding, DevOps, etc.)

    Example:
        middleware = GitSafetyMiddleware(
            block_dangerous=True,   # Block force push, hard reset
            warn_on_risky=True,     # Add warnings for risky operations
            allowed_force_branches={"feature/*"},  # Allow force push on feature branches
        )
    """

    # Dangerous operations that should be blocked by default
    BLOCKED_OPERATIONS: frozenset = frozenset(
        {
            "push --force",
            "push -f",
            "push --force-with-lease",  # Can still lose commits
            "reset --hard HEAD~",
            "reset --hard origin",
            "clean -fd",
            "clean -fdx",
            "reflog expire --expire=now --all",
            "gc --prune=now",
        }
    )

    # Operations that should generate warnings
    WARNED_OPERATIONS: frozenset = frozenset(
        {
            "reset --hard",
            "checkout --",
            "stash drop",
            "stash clear",
            "branch -D",
            "branch --delete --force",
            "rebase",
            "cherry-pick",
            "filter-branch",
        }
    )

    # Protected branches that should never be force-pushed to
    PROTECTED_BRANCHES: frozenset = frozenset(
        {
            "main",
            "master",
            "develop",
            "release",
            "production",
            "staging",
        }
    )

    def __init__(
        self,
        block_dangerous: bool = True,
        warn_on_risky: bool = True,
        protected_branches: Optional[Set[str]] = None,
        allowed_force_branches: Optional[Set[str]] = None,
        custom_blocked: Optional[Set[str]] = None,
        custom_warned: Optional[Set[str]] = None,
    ):
        """Initialize git safety middleware.

        Args:
            block_dangerous: Whether to block dangerous operations
            warn_on_risky: Whether to add warnings for risky operations
            protected_branches: Additional branches to protect
            allowed_force_branches: Branch patterns where force push is allowed
            custom_blocked: Additional operations to block
            custom_warned: Additional operations to warn about
        """
        self._block_dangerous = block_dangerous
        self._warn_on_risky = warn_on_risky
        self._protected_branches = self.PROTECTED_BRANCHES | (protected_branches or set())
        self._allowed_force_branches = allowed_force_branches or set()
        self._blocked = self.BLOCKED_OPERATIONS | (custom_blocked or set())
        self._warned = self.WARNED_OPERATIONS | (custom_warned or set())

    def _is_protected_branch_operation(self, command: str) -> Optional[str]:
        """Check if command targets a protected branch.

        Args:
            command: Git command string

        Returns:
            Protected branch name if found, None otherwise
        """
        for branch in self._protected_branches:
            # Check for patterns like "push origin main --force"
            if branch in command and any(force in command for force in ["--force", "-f"]):
                return branch
        return None

    def _is_force_allowed(self, command: str) -> bool:
        """Check if force push is allowed for this command.

        Args:
            command: Git command string

        Returns:
            True if force is allowed on this branch
        """
        import fnmatch

        for pattern in self._allowed_force_branches:
            # Extract branch name from command
            parts = command.split()
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Check git operations for safety.

        Args:
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            MiddlewareResult with safety check results
        """
        # Only check git-related tools
        if tool_name not in {"git", "execute_bash", "bash", "shell", "run_command"}:
            return MiddlewareResult()

        command = arguments.get("command", "") or arguments.get("args", "")
        if not command:
            return MiddlewareResult()

        # Skip if not a git command
        if "git " not in command and not command.startswith("git"):
            return MiddlewareResult()

        # Check for blocked operations
        if self._block_dangerous:
            for blocked in self._blocked:
                if blocked in command:
                    # Check if force is allowed on this branch
                    if "force" in blocked and self._is_force_allowed(command):
                        continue
                    return MiddlewareResult(
                        proceed=False,
                        error_message=f"Blocked dangerous git operation: {blocked}. "
                        "This operation can cause data loss.",
                    )

            # Check for protected branch force push
            protected = self._is_protected_branch_operation(command)
            if protected:
                return MiddlewareResult(
                    proceed=False,
                    error_message=f"Blocked force push to protected branch: {protected}. "
                    "Force pushing to main/master/develop branches is not allowed.",
                )

        # Add warnings for risky operations
        if self._warn_on_risky:
            for warned in self._warned:
                if warned in command:
                    return MiddlewareResult(
                        proceed=True,
                        metadata={
                            "git_warning": f"Risky operation: {warned}. "
                            "Ensure you have a backup or know what you're doing.",
                        },
                    )

        return MiddlewareResult()

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """No-op after tool call.

        Returns:
            None (no modification)
        """
        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get priority - CRITICAL for safety checks.

        Returns:
            CRITICAL priority (runs first)
        """
        return MiddlewarePriority.CRITICAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get applicable tools.

        Returns:
            Set of git-related tools
        """
        return {"git", "execute_bash", "bash", "shell", "run_command"}


# =============================================================================
# Output Validation Middleware
# =============================================================================


class OutputValidationMiddleware(MiddlewareProtocol):
    """Generic middleware for validating and optionally fixing tool outputs.

    This middleware provides a configurable validation framework that can:
    - Validate tool arguments before execution using custom validators
    - Optionally auto-fix detected issues
    - Block execution on critical validation failures
    - Report validation metadata for downstream processing

    The middleware is domain-agnostic - validators can be implemented for:
    - Code linting (Python, JavaScript, SQL)
    - Schema validation (JSON, YAML, XML)
    - Data validation (Pydantic models, JSON Schema)
    - Configuration validation (Docker, Kubernetes, Terraform)

    Example:
        # Create a JSON validator
        class JsonValidator:
            def validate(self, content, context=None):
                import json
                try:
                    json.loads(content)
                    return ContentValidationResult(is_valid=True)
                except json.JSONDecodeError as e:
                    return ContentValidationResult(
                        is_valid=False,
                        issues=[ValidationIssue(
                            message=f"Invalid JSON: {e.msg}",
                            location=f"line {e.lineno}, column {e.colno}",
                        )]
                    )

        # Use in middleware
        middleware = OutputValidationMiddleware(
            validator=JsonValidator(),
            applicable_tools={"write_config", "create_json"},
            argument_names={"content", "data"},
            auto_fix=True,
            block_on_error=False,
        )
    """

    def __init__(
        self,
        validator: ValidatorProtocol,
        applicable_tools: Optional[Set[str]] = None,
        argument_names: Optional[Set[str]] = None,
        auto_fix: bool = True,
        block_on_error: bool = False,
        max_fix_iterations: int = 3,
        priority: MiddlewarePriority = MiddlewarePriority.HIGH,
    ):
        """Initialize the validation middleware.

        Args:
            validator: Validator instance implementing ValidatorProtocol
            applicable_tools: Set of tool names this middleware applies to.
                             None means apply to all tools.
            argument_names: Set of argument names to validate.
                           Default: {"content", "code", "data", "text", "body"}
            auto_fix: Whether to automatically fix issues if validator supports it
            block_on_error: Whether to block execution on validation errors
            max_fix_iterations: Maximum fix attempts before giving up
            priority: Middleware execution priority
        """
        self.validator = validator
        self._applicable_tools = applicable_tools
        self._argument_names = argument_names or {"content", "code", "data", "text", "body"}
        self.auto_fix = auto_fix
        self.block_on_error = block_on_error
        self.max_fix_iterations = max_fix_iterations
        self._priority = priority

    async def before_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> MiddlewareResult:
        """Validate arguments before tool execution.

        Validates content in specified arguments and optionally applies fixes.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments

        Returns:
            MiddlewareResult with validation status and optionally modified arguments
        """
        # Find arguments to validate
        content_to_validate: Dict[str, str] = {}
        for arg_name in self._argument_names:
            if arg_name in arguments and isinstance(arguments[arg_name], str):
                content_to_validate[arg_name] = arguments[arg_name]

        if not content_to_validate:
            return MiddlewareResult()

        # Build context for validator
        context = {
            "tool_name": tool_name,
            "argument_names": list(content_to_validate.keys()),
        }

        # Validate each content field
        all_issues: List[ValidationIssue] = []
        modified_arguments: Dict[str, Any] = {}
        any_fixed = False

        for arg_name, content in content_to_validate.items():
            result = self._validate_content(content, context)

            if result.issues:
                all_issues.extend(result.issues)

            # Attempt fix if enabled and validator supports it
            if not result.is_valid and self.auto_fix and result.has_fix:
                modified_arguments[arg_name] = result.fixed_content
                any_fixed = True
                logger.debug(
                    f"OutputValidationMiddleware: Fixed issues in '{arg_name}' for {tool_name}"
                )
            elif not result.is_valid and self.auto_fix:
                # Try to fix using the FixableValidatorProtocol
                fixed = self._try_fix(content, result.issues, context)
                if fixed != content:
                    modified_arguments[arg_name] = fixed
                    any_fixed = True
                    logger.debug(
                        f"OutputValidationMiddleware: Applied fixes to '{arg_name}' for {tool_name}"
                    )

        # Build metadata
        metadata = {
            "validation_performed": True,
            "validation_passed": len(all_issues) == 0,
            "issue_count": len(all_issues),
            "error_count": sum(
                1
                for i in all_issues
                if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            ),
            "warning_count": sum(1 for i in all_issues if i.severity == ValidationSeverity.WARNING),
            "content_fixed": any_fixed,
        }

        if all_issues:
            metadata["validation_issues"] = [
                {"message": i.message, "severity": i.severity.value, "location": i.location}
                for i in all_issues[:10]  # Limit to first 10 issues
            ]

        # Determine if we should block
        has_critical = any(i.severity == ValidationSeverity.CRITICAL for i in all_issues)
        should_block = self.block_on_error and (has_critical or metadata["error_count"] > 0)

        if should_block and not any_fixed:
            error_msg = f"Validation failed: {metadata['error_count']} errors"
            if all_issues:
                error_msg += f". First issue: {all_issues[0].message}"
            return MiddlewareResult(
                proceed=False,
                error_message=error_msg,
                metadata=metadata,
            )

        # Merge modified arguments with original
        final_arguments = None
        if modified_arguments:
            final_arguments = {**arguments, **modified_arguments}

        return MiddlewareResult(
            proceed=True,
            modified_arguments=final_arguments,
            metadata=metadata,
        )

    def _validate_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ContentValidationResult:
        """Validate content using the configured validator.

        Args:
            content: Content to validate
            context: Validation context

        Returns:
            ContentValidationResult from the validator
        """
        try:
            return self.validator.validate(content, context)
        except Exception as e:
            logger.warning(f"OutputValidationMiddleware: Validator error: {e}")
            return ContentValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        message=f"Validation error: {e}",
                        severity=ValidationSeverity.ERROR,
                    )
                ],
            )

    def _try_fix(
        self,
        content: str,
        issues: List[ValidationIssue],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Attempt to fix content if validator supports fixing.

        Args:
            content: Original content
            issues: Issues to fix
            context: Fix context

        Returns:
            Fixed content (or original if unfixable)
        """
        # Check if validator implements FixableValidatorProtocol
        if not hasattr(self.validator, "fix"):
            return content

        try:
            fixed = content
            for iteration in range(self.max_fix_iterations):
                fixed = self.validator.fix(fixed, issues, context)  # type: ignore
                # Re-validate to check if fix worked
                result = self._validate_content(fixed, context)
                if result.is_valid:
                    return fixed
                issues = result.issues
            return fixed
        except Exception as e:
            logger.warning(f"OutputValidationMiddleware: Fix error: {e}")
            return content

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Post-execution hook (no-op for validation middleware).

        Returns:
            None (no modification)
        """
        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get middleware priority.

        Returns:
            Configured priority (default HIGH)
        """
        return self._priority

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get applicable tools.

        Returns:
            Set of tool names or None for all tools
        """
        return self._applicable_tools


# =============================================================================
# Cache Middleware
# =============================================================================


class CacheResult:
    """Cache middleware result that supports attribute access.

    Provides both attribute access and dict-like interface for
    backward compatibility with MiddlewareResult.
    """

    def __init__(
        self,
        proceed: bool = True,
        cached_result: Optional[Any] = None,
        modified_arguments: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        **metadata,
    ):
        self.proceed = proceed
        self.cached_result = cached_result
        self.modified_arguments = modified_arguments
        self.error_message = error_message
        self._metadata = metadata

    def __getattr__(self, name):
        # Provide access to metadata keys as attributes
        if name in self._metadata:
            return self._metadata[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class CacheMiddleware(MiddlewareProtocol):
    """Cache tool execution results.

    This middleware caches results from idempotent tools to avoid
    redundant executions. Only successful results are cached.

    Consolidated from framework.py and common.py:
    - Supports both in-memory cache (default) and pluggable ICacheBackend
    - Auto-detects idempotent tools by name pattern
    - Returns CacheResult for cache hits (framework.py approach)
    - Optional cache namespace for isolation (common.py approach)

    Example:
        # Simple in-memory cache
        middleware = CacheMiddleware(
            ttl_seconds=300,  # 5 minutes
            cacheable_tools={"read", "ls", "grep"},
        )

        # With custom cache backend
        middleware = CacheMiddleware(
            cache_backend=redis_backend,
            ttl_seconds=3600,
            cache_namespace="tool_cache",
        )

        # First call - cache miss
        result1 = await middleware.before_tool_call("read", {"path": "file.txt"})
        assert result1.cached_result is None

        # After successful execution
        await middleware.after_tool_call("read", {"path": "file.txt"}, "content", True)

        # Second call - cache hit
        result2 = await middleware.before_tool_call("read", {"path": "file.txt"})
        assert result2.cached_result == "content"
        assert result2.proceed is False
    """

    # Idempotent tool patterns for auto-detection (from common.py)
    IDEMPOTENT_TOOL_PATTERNS = {
        "read_file",
        "grep",
        "search",
        "find",
        "list",
        "get_",
        "fetch_",
        "ast_parse",
        "code_search",
    }

    def __init__(
        self,
        ttl_seconds: int = 300,
        cacheable_tools: Optional[Set[str]] = None,
        key_components: Optional[List[str]] = None,
        cache_backend: Optional[Any] = None,  # ICacheBackend from common.py
        cache_namespace: str = "tool_cache",
        auto_detect_idempotent: bool = True,
    ):
        """Initialize the cache middleware.

        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            cacheable_tools: Set of tool names that can be cached (None = auto-detect)
            key_components: Components to include in cache key
                (["tool_name", "args"] includes both tool name and arguments)
            cache_backend: Optional ICacheBackend for pluggable storage (None = in-memory)
            cache_namespace: Namespace for cache isolation (for ICacheBackend)
            auto_detect_idempotent: Auto-detect idempotent tools by name pattern
        """
        import hashlib
        import json

        self._ttl_seconds = ttl_seconds
        self._cacheable_tools = cacheable_tools or set()
        self._key_components = key_components or ["tool_name", "args"]
        self._hashlib = hashlib
        self._json = json

        # Cache backend support (from common.py)
        self._cache_backend = cache_backend
        self._cache_namespace = cache_namespace
        self._auto_detect_idempotent = auto_detect_idempotent

        # In-memory cache storage (fallback if no backend)
        if cache_backend is None:
            self._cache: Dict[str, tuple[Any, float]] = {}
        else:
            self._cache = None  # Not used when backend is provided

        # Statistics
        self._hits = 0
        self._misses = 0

    def _build_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Build cache key from tool name and arguments.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Cache key as a hash string
        """
        key_parts = []

        if "tool_name" in self._key_components:
            key_parts.append(tool_name)

        if "args" in self._key_components:
            # Sort arguments for consistent hashing
            sorted_args = self._json.dumps(arguments, sort_keys=True)
            key_parts.append(sorted_args)

        key_string = ":".join(key_parts)
        return self._hashlib.sha256(key_string.encode()).hexdigest()

    def _is_cacheable(self, tool_name: str) -> bool:
        """Check if a tool is cacheable.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool should be cached
        """
        # Explicit cacheable tools
        if tool_name in self._cacheable_tools:
            return True

        # Auto-detect idempotent tools (from common.py)
        if self._auto_detect_idempotent:
            return self._is_tool_idempotent(tool_name)

        return False

    def _is_tool_idempotent(self, tool_name: str) -> bool:
        """Check if tool is idempotent by name pattern (from common.py).

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool appears to be idempotent
        """
        tool_lower = tool_name.lower()
        for pattern in self.IDEMPOTENT_TOOL_PATTERNS:
            if pattern.lower() in tool_lower:
                return True
        return False

    def _is_expired(self, cache_key: str) -> bool:
        """Check if a cache entry has expired (in-memory only).

        Args:
            cache_key: Cache entry key

        Returns:
            True if entry has expired
        """
        if self._cache is None or cache_key not in self._cache:
            return True

        _, expiry_time = self._cache[cache_key]
        import time

        return time.time() > expiry_time

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> CacheResult:
        """Check cache before tool execution.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            CacheResult with cached result if available
        """
        if not self._is_cacheable(tool_name):
            return CacheResult(proceed=True, cached_result=None)

        cache_key = self._build_cache_key(tool_name, arguments)

        # Try cache backend first (from common.py)
        if self._cache_backend is not None:
            try:
                cached_result = await self._cache_backend.get(cache_key, self._cache_namespace)
                if cached_result is not None:
                    self._hits += 1
                    return CacheResult(proceed=False, cached_result=cached_result)
            except Exception:
                # Backend error, fall through to in-memory
                pass

        # Check in-memory cache
        if self._cache is not None and cache_key in self._cache and not self._is_expired(cache_key):
            cached_result, _ = self._cache[cache_key]
            self._hits += 1
            return CacheResult(proceed=False, cached_result=cached_result)

        self._misses += 1
        return CacheResult(proceed=True, cached_result=None)

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Cache successful tool results.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            None (no modification)
        """
        # Only cache successful results from cacheable tools
        if not success or not self._is_cacheable(tool_name):
            return None

        cache_key = self._build_cache_key(tool_name, arguments)
        import time

        expiry_time = time.time() + self._ttl_seconds

        # Store in cache backend if available (from common.py)
        if self._cache_backend is not None:
            try:
                await self._cache_backend.set(
                    cache_key, result, self._cache_namespace, self._ttl_seconds
                )
            except Exception:
                # Backend error, try in-memory fallback
                if self._cache is not None:
                    self._cache[cache_key] = (result, expiry_time)
        elif self._cache is not None:
            # In-memory cache
            self._cache[cache_key] = (result, expiry_time)

        return None

    def invalidate(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Invalidate a specific cache entry.

        Args:
            tool_name: Name of the tool
            arguments: Arguments that were used
        """
        cache_key = self._build_cache_key(tool_name, arguments)

        # Invalidate in both backends
        if self._cache is not None:
            self._cache.pop(cache_key, None)

        if self._cache_backend is not None:
            # Try async invalidate (fire and forget)
            try:
                import asyncio

                asyncio.create_task(self._cache_backend.delete(cache_key, self._cache_namespace))
            except Exception:
                pass

    def clear(self) -> None:
        """Clear all cached entries."""
        if self._cache is not None:
            self._cache.clear()

        if self._cache_backend is not None:
            # Try async clear (fire and forget)
            try:
                import asyncio

                asyncio.create_task(self._cache_backend.clear(self._cache_namespace))
            except Exception:
                pass

        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with total_entries, hits, misses
        """
        # Clean up expired entries
        import time

        current_time = time.time()
        self._cache = {k: v for k, v in self._cache.items() if current_time < v[1]}

        total_entries = len(self._cache)
        hit_rate = (
            self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0
        )

        return {
            "total_entries": total_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Cache runs with HIGH priority to avoid unnecessary executions.

        Returns:
            HIGH priority
        """
        return MiddlewarePriority.HIGH

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            Set of cacheable tools
        """
        return self._cacheable_tools


# =============================================================================
# Rate Limit Middleware
# =============================================================================


class RateLimitResult:
    """Rate limit middleware result that supports attribute access.

    Provides both attribute access and dict-like interface for
    backward compatibility with MiddlewareResult.
    """

    def __init__(
        self,
        proceed: bool = True,
        blocked: bool = False,
        retry_after_seconds: Optional[float] = None,
        modified_arguments: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        **metadata,
    ):
        self.proceed = proceed
        self.blocked = blocked
        self.retry_after_seconds = retry_after_seconds
        self.modified_arguments = modified_arguments
        self.error_message = error_message
        self._metadata = metadata

    def __getattr__(self, name):
        # Provide access to metadata keys as attributes
        if name in self._metadata:
            return self._metadata[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class RateLimitMiddleware(MiddlewareProtocol):
    """Rate limit tool execution.

    This middleware limits the rate at which specific tools can be called.
    Useful for preventing abuse of expensive or sensitive operations.

    Example:
        middleware = RateLimitMiddleware(
            max_calls=5,
            time_window_seconds=60,  # 5 calls per minute
            blocked_tools={"write", "edit"},
        )

        # Make 5 calls (at limit)
        for i in range(5):
            result = await middleware.before_tool_call("write", {"path": f"test{i}.txt"})
            assert result.proceed is True

        # 6th call should be blocked
        result = await middleware.before_tool_call("write", {"path": "test6.txt"})
        assert result.proceed is False
        assert result.blocked is True
        assert result.retry_after_seconds is not None
    """

    def __init__(
        self,
        max_calls: int = 10,
        time_window_seconds: int = 60,
        blocked_tools: Optional[Set[str]] = None,
    ):
        """Initialize the rate limit middleware.

        Args:
            max_calls: Maximum number of allowed calls per time window
            time_window_seconds: Time window in seconds
            blocked_tools: Set of tool names to rate limit
        """
        self._max_calls = max_calls
        self._time_window_seconds = time_window_seconds
        self._blocked_tools = blocked_tools or set()

        # Track calls per tool: {tool_name: [(timestamp, count), ...]}
        self._call_history: Dict[str, List[float]] = {}

    def _is_limited(self, tool_name: str) -> bool:
        """Check if a tool is rate-limited.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool should be rate-limited
        """
        return tool_name in self._blocked_tools

    def _clean_old_calls(self, tool_name: str) -> None:
        """Remove calls outside the time window.

        Args:
            tool_name: Name of the tool
        """
        if tool_name not in self._call_history:
            return

        import time

        current_time = time.time()
        cutoff_time = current_time - self._time_window_seconds

        # Keep only calls within the time window
        self._call_history[tool_name] = [
            ts for ts in self._call_history[tool_name] if ts > cutoff_time
        ]

    def _get_call_count(self, tool_name: str) -> int:
        """Get the number of calls within the time window.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of calls in the current time window
        """
        self._clean_old_calls(tool_name)
        return len(self._call_history.get(tool_name, []))

    def _get_retry_after(self, tool_name: str) -> float:
        """Calculate seconds until next allowed call.

        Args:
            tool_name: Name of the tool

        Returns:
            Seconds to wait before retry
        """
        if tool_name not in self._call_history or not self._call_history[tool_name]:
            return 0.0

        # Get the oldest call in the window
        oldest_call = min(self._call_history[tool_name])
        import time

        current_time = time.time()
        # When the oldest call will exit the window
        window_end = oldest_call + self._time_window_seconds

        return max(0.0, window_end - current_time)

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> RateLimitResult:
        """Check rate limit before tool execution.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            RateLimitResult with rate limit status
        """
        if not self._is_limited(tool_name):
            return RateLimitResult(proceed=True, blocked=False, retry_after_seconds=None)

        # Clean old calls
        self._clean_old_calls(tool_name)

        # Get current call count
        call_count = self._get_call_count(tool_name)

        if call_count >= self._max_calls:
            # Rate limit exceeded
            retry_after = self._get_retry_after(tool_name)

            return RateLimitResult(
                proceed=False,
                blocked=True,
                retry_after_seconds=retry_after,
                error_message=f"Rate limit exceeded for {tool_name}. "
                f"Maximum {self._max_calls} calls per {self._time_window_seconds} seconds. "
                f"Retry after {retry_after:.1f} seconds.",
                call_count=call_count,
                limit=self._max_calls,
            )

        # Track this call
        import time

        current_time = time.time()
        if tool_name not in self._call_history:
            self._call_history[tool_name] = []
        self._call_history[tool_name].append(current_time)

        return RateLimitResult(
            proceed=True,
            blocked=False,
            retry_after_seconds=None,
            call_count=call_count + 1,
            limit=self._max_calls,
        )

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """No-op after tool call.

        Returns:
            None (no modification)
        """
        return None

    def reset(self, tool_name: str) -> None:
        """Reset rate limit for a specific tool.

        Args:
            tool_name: Name of the tool to reset
        """
        self._call_history.pop(tool_name, None)

    def reset_all(self) -> None:
        """Reset all rate limits."""
        self._call_history.clear()

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get rate limit statistics.

        Returns:
            Dict mapping tool names to their stats
        """
        stats = {}

        for tool_name in self._blocked_tools:
            call_count = self._get_call_count(tool_name)
            is_blocked = call_count >= self._max_calls

            stats[tool_name] = {
                "calls_made": call_count,
                "limit": self._max_calls,
                "blocked": is_blocked,
                "time_window_seconds": self._time_window_seconds,
            }

        return stats

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Rate limit runs with CRITICAL priority to block calls early.

        Returns:
            CRITICAL priority
        """
        return MiddlewarePriority.CRITICAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            Set of rate-limited tools
        """
        return self._blocked_tools


# =============================================================================
# ValidationMiddleware (from common.py)
# =============================================================================


class ValidationMiddleware(MiddlewareProtocol):
    """Validates tool arguments against JSON schemas.

    This middleware validates tool arguments before execution using JSON Schema
    validation. It helps catch errors early and provides clear error messages.

    Example:
        schemas = {
            "write_file": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            }
        }
        middleware = ValidationMiddleware(schemas=schemas)
    """

    # Dangerous system commands to block (shared with SafetyCheckMiddleware)
    DANGEROUS_COMMANDS = {
        "rm -rf /",
        "rm -rf /*",
        "mkfs",
        "dd if=/dev/zero",
        "dd if=/dev/random",
        "chmod 000 /",
        "chown -R root",
        ":(){:|:&};:",  # Fork bomb
        "mv / /dev/null",
    }

    # Write operations that should be checked
    WRITE_OPERATIONS = {
        "write_file",
        "edit_file",
        "create_file",
        "save_file",
    }

    def __init__(
        self,
        schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        enabled: bool = True,
        applicable_tools: Optional[Set[str]] = None,
    ):
        """Initialize ValidationMiddleware.

        Args:
            schemas: Dictionary mapping tool names to JSON schemas
            enabled: Whether middleware is enabled
            applicable_tools: Set of tool names this applies to (None = all)
        """
        self._schemas = schemas or {}
        self._enabled = enabled
        self._applicable_tools = applicable_tools

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Validate tool arguments against schema.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult indicating validation success/failure
        """
        if not self._enabled:
            return MiddlewareResult()

        if tool_name not in self._schemas:
            # No schema defined for this tool, allow it
            return MiddlewareResult(proceed=True)

        schema = self._schemas[tool_name]

        try:
            # Validate arguments against schema
            self._validate(schema, arguments)
            return MiddlewareResult(proceed=True)
        except ValueError as e:
            return MiddlewareResult(proceed=False, error_message=f"Validation failed: {str(e)}")

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """No-op after tool call.

        Returns:
            None (no modification)
        """
        return None

    def _validate(self, schema: Dict[str, Any], arguments: Dict[str, Any]) -> None:
        """Validate arguments against JSON schema.

        Args:
            schema: JSON schema
            arguments: Arguments to validate

        Raises:
            ValueError: If validation fails
        """
        # Simple JSON schema validation implementation
        # Check type
        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object":
                if not isinstance(arguments, dict):
                    raise ValueError(f"Expected object, got {type(arguments).__name__}")

        # Check required properties
        if "required" in schema:
            for required_prop in schema["required"]:
                if required_prop not in arguments:
                    raise ValueError(f"Missing required property: '{required_prop}'")

        # Check property types
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in arguments:
                    prop_value = arguments[prop_name]
                    self._validate_property(prop_name, prop_value, prop_schema)

    def _validate_property(self, prop_name: str, value: Any, prop_schema: Dict[str, Any]) -> None:
        """Validate a single property against its schema.

        Args:
            prop_name: Name of the property
            value: Value to validate
            prop_schema: Schema for the property

        Raises:
            ValueError: If validation fails
        """
        if "type" in prop_schema:
            expected_type = prop_schema["type"]

            type_mapping = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict,
            }

            expected_python_type = type_mapping.get(expected_type)
            if expected_python_type and not isinstance(value, expected_python_type):
                raise ValueError(
                    f"Property '{prop_name}': expected {expected_type}, "
                    f"got {type(value).__name__}"
                )

        # Validate array items
        if "type" in prop_schema and prop_schema["type"] == "array":
            if isinstance(value, list) and "items" in prop_schema:
                for i, item in enumerate(value):
                    self._validate_property(f"{prop_name}[{i}]", item, prop_schema["items"])

        # Nested object validation
        if "properties" in prop_schema and isinstance(value, dict):
            self._validate(prop_schema, value)

    def get_priority(self) -> MiddlewarePriority:
        """Get middleware priority.

        Returns:
            HIGH priority (validation should happen early)
        """
        return MiddlewarePriority.HIGH

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get applicable tools.

        Returns:
            Set of tool names or None for all tools
        """
        return self._applicable_tools


# =============================================================================
# SafetyCheckMiddleware (from common.py)
# =============================================================================


class SafetyCheckMiddleware(MiddlewareProtocol):
    """Checks for dangerous operations and blocks them.

    This middleware provides security checks to prevent dangerous operations
    like file deletions, system commands, and writes to sensitive paths.

    Example:
        middleware = SafetyCheckMiddleware(
            blocked_tools={"rm", "format_disk"},
            allowed_paths={"/tmp", "/home/user/workspace"}
        )
    """

    def __init__(
        self,
        blocked_tools: Optional[Set[str]] = None,
        allowed_paths: Optional[Set[str]] = None,
        enabled: bool = True,
    ):
        """Initialize SafetyCheckMiddleware.

        Args:
            blocked_tools: Set of tool names that are completely blocked
            allowed_paths: Set of allowed paths for write operations (None = all paths allowed)
            enabled: Whether middleware is enabled
        """
        self._blocked_tools = blocked_tools or set()
        self._allowed_paths = allowed_paths
        self._enabled = enabled

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Perform safety checks before tool execution.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult indicating if tool call should proceed
        """
        if not self._enabled:
            return MiddlewareResult()

        # Check if tool is completely blocked
        if tool_name in self._blocked_tools:
            return MiddlewareResult(
                proceed=False,
                error_message=f"Tool '{tool_name}' is blocked for safety reasons",
            )

        # Check write operations against allowed paths
        if tool_name in ValidationMiddleware.WRITE_OPERATIONS and self._allowed_paths:
            path = arguments.get("path", "")
            if not self._is_path_allowed(path):
                return MiddlewareResult(
                    proceed=False,
                    error_message=(
                        f"Path '{path}' is not in allowed paths. "
                        f"Allowed paths: {self._allowed_paths}"
                    ),
                )

        # Check for dangerous shell commands
        if tool_name in {"shell_execute", "execute_bash", "bash"}:
            command = arguments.get("command", "")
            if self._is_dangerous_command(command):
                return MiddlewareResult(
                    proceed=False,
                    error_message=(f"Command '{command}' is blocked for safety reasons"),
                )

        return MiddlewareResult(proceed=True)

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """No-op after tool call.

        Returns:
            None (no modification)
        """
        return None

    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is in allowed paths.

        Args:
            path: Path to check

        Returns:
            True if path is allowed, False otherwise
        """
        if not self._allowed_paths:
            return True

        # Normalize path
        import os

        path = os.path.normpath(path)

        # Check if path starts with any allowed path
        for allowed_path in self._allowed_paths:
            if path.startswith(allowed_path):
                return True

        return False

    def _is_dangerous_command(self, command: str) -> bool:
        """Check if command is dangerous.

        Args:
            command: Command string to check

        Returns:
            True if command is dangerous, False otherwise
        """
        command_lower = command.lower().strip()

        # Check against known dangerous commands
        for dangerous in ValidationMiddleware.DANGEROUS_COMMANDS:
            if dangerous.lower() in command_lower:
                return True

        # Check for suspicious patterns
        suspicious_patterns = ["rm -rf", "rm -fr", "dd if=/dev", "mkfs", "chmod 000 /"]
        for pattern in suspicious_patterns:
            if pattern in command_lower:
                return True

        return False

    def get_priority(self) -> MiddlewarePriority:
        """Get middleware priority.

        Returns:
            CRITICAL priority (safety checks must run first)
        """
        return MiddlewarePriority.CRITICAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get applicable tools.

        Returns:
            None (applies to all tools)
        """
        return None


__all__ = [
    # Middleware
    "LoggingMiddleware",
    "SecretMaskingMiddleware",
    "MetricsMiddleware",
    "GitSafetyMiddleware",
    "OutputValidationMiddleware",
    "CacheMiddleware",
    "RateLimitMiddleware",
    # Additional middleware from common.py
    "ValidationMiddleware",
    "SafetyCheckMiddleware",
    # Validation Types
    "ValidationSeverity",
    "ValidationIssue",
    "ContentValidationResult",
    "ValidatorProtocol",
    "FixableValidatorProtocol",
    # Types
    "ToolMetrics",
    # Cache Types
    "CacheResult",
    # Rate Limit Types
    "RateLimitResult",
]
