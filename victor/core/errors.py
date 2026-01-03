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

"""Centralized error handling for Victor.

This module provides:
- Custom exception types for different error categories
- Error handler utility with structured logging
- User-friendly error messages with recovery suggestions
- Correlation IDs for distributed tracing
"""

from __future__ import annotations

import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# Error Categories
# =============================================================================


class ErrorCategory(Enum):
    """Categories of errors for classification and handling."""

    # Provider errors
    PROVIDER_CONNECTION = "provider_connection"
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_INVALID_RESPONSE = "provider_invalid_response"

    # Tool errors
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_EXECUTION = "tool_execution"
    TOOL_VALIDATION = "tool_validation"
    TOOL_TIMEOUT = "tool_timeout"

    # Configuration errors
    CONFIG_INVALID = "config_invalid"
    CONFIG_MISSING = "config_missing"

    # Resource errors
    FILE_NOT_FOUND = "file_not_found"
    FILE_PERMISSION = "file_permission"
    NETWORK_ERROR = "network_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"

    # User errors
    VALIDATION_ERROR = "validation_error"
    INVALID_INPUT = "invalid_input"

    # System errors
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Custom Exception Types
# =============================================================================


class VictorError(Exception):
    """Base exception for all Victor errors.

    Provides structured error information including:
    - Error category and severity
    - Correlation ID for tracking
    - User-friendly message
    - Recovery suggestions
    - Original exception chain
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recovery_hint = recovery_hint
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "correlation_id": self.correlation_id,
            "recovery_hint": self.recovery_hint,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        result = f"[{self.correlation_id}] {self.message}"
        if self.recovery_hint:
            result += f"\nRecovery hint: {self.recovery_hint}"
        return result


class ProviderError(VictorError):
    """Errors related to LLM providers."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        raw_error: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.model = model
        self.details["provider"] = provider
        self.details["model"] = model
        # Backward compatibility with providers/base.py signature
        self.status_code = status_code
        self.raw_error = raw_error
        if status_code is not None:
            self.details["status_code"] = status_code


class ProviderConnectionError(ProviderError):
    """Provider connection failures."""

    def __init__(self, message: str, provider: Optional[str] = None, **kwargs: Any):
        super().__init__(
            message,
            provider=provider,
            category=ErrorCategory.PROVIDER_CONNECTION,
            recovery_hint="Check network connection and provider URL. Verify the provider service is running.",
            **kwargs,
        )


class ProviderAuthError(ProviderError):
    """Provider authentication failures."""

    def __init__(self, message: str, provider: Optional[str] = None, **kwargs: Any):
        super().__init__(
            message,
            provider=provider,
            category=ErrorCategory.PROVIDER_AUTH,
            recovery_hint="Check your API key or credentials. Ensure they are correctly set in environment variables or configuration.",
            **kwargs,
        )


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            provider=provider,
            category=ErrorCategory.PROVIDER_RATE_LIMIT,
            recovery_hint=(
                f"Wait {retry_after} seconds before retrying."
                if retry_after
                else "Wait and retry later. Consider using a different model or provider."
            ),
            **kwargs,
        )
        self.retry_after = retry_after
        self.details["retry_after"] = retry_after


class ProviderTimeoutError(ProviderError):
    """Provider request timeout."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        timeout: Optional[int] = None,
        status_code: Optional[int] = None,
        raw_error: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            provider=provider,
            category=ErrorCategory.PROVIDER_CONNECTION,
            recovery_hint=(
                f"Request timed out after {timeout} seconds. Try increasing timeout or check provider status."
                if timeout
                else "Request timed out. Check provider status and network connection."
            ),
            **kwargs,
        )
        self.timeout = timeout
        self.details["timeout"] = timeout
        # Backward compatibility with base.py signature
        self.status_code = status_code
        self.raw_error = raw_error


class ProviderNotFoundError(ProviderError):
    """Provider not found in registry."""

    def __init__(
        self,
        message: Optional[str] = None,
        provider: Optional[str] = None,
        available_providers: Optional[List[str]] = None,
        status_code: Optional[int] = None,
        raw_error: Optional[Any] = None,
        **kwargs: Any,
    ):
        # Generate message if not provided
        if message is None:
            message = f"Provider not found: {provider}"
            if available_providers:
                message += f". Available: {', '.join(available_providers[:5])}"
        super().__init__(
            message,
            provider=provider,
            category=ErrorCategory.CONFIG_INVALID,
            recovery_hint="Check provider name spelling. Use 'victor providers' to list available providers.",
            **kwargs,
        )
        self.available_providers = available_providers or []
        self.details["available_providers"] = self.available_providers
        # Backward compatibility with base.py signature
        self.status_code = status_code
        self.raw_error = raw_error


class ProviderInvalidResponseError(ProviderError):
    """Provider returned invalid or unexpected response."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            provider=provider,
            category=ErrorCategory.PROVIDER_INVALID_RESPONSE,
            recovery_hint="The provider returned an unexpected response format. Try again or use a different provider.",
            **kwargs,
        )
        self.response_data = response_data
        if response_data:
            self.details["response_keys"] = list(response_data.keys())[:10]


class ToolError(VictorError):
    """Errors related to tool execution."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.details["tool_name"] = tool_name


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""

    def __init__(self, tool_name: str, **kwargs: Any):
        super().__init__(
            f"Tool not found: {tool_name}",
            tool_name=tool_name,
            category=ErrorCategory.TOOL_NOT_FOUND,
            recovery_hint="Check tool name spelling. Use list_tools() to see available tools.",
            **kwargs,
        )


class ToolExecutionError(ToolError):
    """Tool execution failures."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            tool_name=tool_name,
            category=ErrorCategory.TOOL_EXECUTION,
            **kwargs,
        )


class ToolValidationError(ToolError):
    """Tool argument validation failures."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        invalid_args: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            tool_name=tool_name,
            category=ErrorCategory.TOOL_VALIDATION,
            recovery_hint="Check the required arguments for this tool.",
            **kwargs,
        )
        self.invalid_args = invalid_args or []
        self.details["invalid_args"] = self.invalid_args


class ToolTimeoutError(ToolError):
    """Tool execution timeout."""

    def __init__(
        self,
        tool_name: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        message = (
            f"Tool '{tool_name}' timed out after {timeout} seconds"
            if timeout
            else f"Tool '{tool_name}' timed out"
        )
        super().__init__(
            message,
            tool_name=tool_name,
            category=ErrorCategory.TOOL_TIMEOUT,
            recovery_hint="Try with a longer timeout or simplify the operation.",
            **kwargs,
        )
        self.timeout = timeout
        self.details["timeout"] = timeout


class ConfigurationError(VictorError):
    """Configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            category=ErrorCategory.CONFIG_INVALID,
            **kwargs,
        )
        self.config_key = config_key
        self.details["config_key"] = config_key


class ValidationError(VictorError):
    """Input validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION_ERROR,
            **kwargs,
        )
        self.field = field
        self.value = value
        self.details["field"] = field
        self.details["value"] = str(value) if value is not None else None


class FileError(VictorError):
    """File operation errors."""

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        self.path = path
        self.details["path"] = path


class FileNotFoundError(FileError):
    """File not found errors."""

    def __init__(self, path: str, **kwargs: Any):
        super().__init__(
            f"File not found: {path}",
            path=path,
            category=ErrorCategory.FILE_NOT_FOUND,
            recovery_hint="Check the file path. The file may have been moved or deleted.",
            **kwargs,
        )


class NetworkError(VictorError):
    """Network-related errors."""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK_ERROR,
            recovery_hint="Check your network connection. The service may be temporarily unavailable.",
            **kwargs,
        )
        self.url = url
        self.details["url"] = url


class ExtensionLoadError(VictorError):
    """Raised when a vertical extension fails to load.

    This exception is used by VerticalBase.get_extensions() to report
    extension loading failures. In strict mode, this exception is raised
    for critical failures. In non-strict mode, errors are collected and
    reported without halting execution.

    Attributes:
        extension_type: The type of extension that failed (e.g., 'safety', 'middleware')
        vertical_name: Name of the vertical where the failure occurred
        original_error: The underlying exception that caused the failure
        is_required: Whether this extension is required for the vertical to function

    Example:
        try:
            safety_ext = vertical.get_safety_extension()
        except Exception as e:
            raise ExtensionLoadError(
                message=f"Failed to load safety extension: {e}",
                extension_type="safety",
                vertical_name=vertical.name,
                original_error=e,
                is_required=True,
            )
    """

    def __init__(
        self,
        message: str,
        extension_type: str,
        vertical_name: str,
        original_error: Optional[Exception] = None,
        is_required: bool = False,
        **kwargs: Any,
    ):
        # Set recovery hint based on whether the extension is required
        if is_required:
            recovery_hint = (
                f"The '{extension_type}' extension is required for vertical '{vertical_name}'. "
                f"Fix the underlying error or mark the extension as optional."
            )
        else:
            recovery_hint = (
                f"The '{extension_type}' extension failed to load for vertical '{vertical_name}'. "
                f"The vertical will function with reduced capabilities."
            )

        super().__init__(
            message,
            category=ErrorCategory.CONFIG_INVALID,
            severity=ErrorSeverity.CRITICAL if is_required else ErrorSeverity.WARNING,
            recovery_hint=recovery_hint,
            cause=original_error,
            **kwargs,
        )
        self.extension_type = extension_type
        self.vertical_name = vertical_name
        self.original_error = original_error
        self.is_required = is_required
        self.details["extension_type"] = extension_type
        self.details["vertical_name"] = vertical_name
        self.details["is_required"] = is_required
        if original_error:
            self.details["original_error_type"] = type(original_error).__name__
            self.details["original_error_message"] = str(original_error)


# =============================================================================
# Error Information
# =============================================================================


@dataclass
class ErrorInfo:
    """Structured error information for logging and display."""

    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    correlation_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_hint: Optional[str] = None
    traceback: Optional[str] = None
    original_exception: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "recovery_hint": self.recovery_hint,
            "traceback": self.traceback,
            "original_exception": self.original_exception,
        }

    def to_user_message(self) -> str:
        """Get user-friendly error message."""
        msg = self.message
        if self.recovery_hint:
            msg += f"\n\nSuggestion: {self.recovery_hint}"
        return msg


# =============================================================================
# Error Handler
# =============================================================================


class ErrorHandler:
    """Centralized error handler with logging and reporting.

    Usage:
        handler = ErrorHandler()

        try:
            risky_operation()
        except Exception as e:
            error_info = handler.handle(e, context={"operation": "risky"})
            return error_info.to_user_message()
    """

    def __init__(
        self,
        logger_name: str = "victor",
        include_traceback: bool = True,
    ):
        self.logger = logging.getLogger(logger_name)
        self.include_traceback = include_traceback
        self._error_history: List[ErrorInfo] = []
        self._max_history = 100

    def handle(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        log_level: Optional[int] = None,
    ) -> ErrorInfo:
        """Handle an exception and return structured error info.

        Args:
            exception: The exception to handle.
            context: Additional context about the operation.
            log_level: Override the default log level.

        Returns:
            ErrorInfo with structured error details.
        """
        error_info = self._create_error_info(exception, context)

        # Log the error
        self._log_error(error_info, log_level)

        # Store in history
        self._add_to_history(error_info)

        return error_info

    def _create_error_info(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorInfo:
        """Create ErrorInfo from an exception."""
        # Handle VictorError specially
        if isinstance(exception, VictorError):
            return ErrorInfo(
                message=exception.message,
                category=exception.category,
                severity=exception.severity,
                correlation_id=exception.correlation_id,
                timestamp=exception.timestamp,
                details={**exception.details, **(context or {})},
                recovery_hint=exception.recovery_hint,
                traceback=traceback.format_exc() if self.include_traceback else None,
                original_exception=str(exception.cause) if exception.cause else None,
            )

        # Handle standard exceptions
        category, recovery_hint = self._categorize_exception(exception)

        return ErrorInfo(
            message=str(exception),
            category=category,
            severity=ErrorSeverity.ERROR,
            correlation_id=str(uuid.uuid4())[:8],
            details=context or {},
            recovery_hint=recovery_hint,
            traceback=traceback.format_exc() if self.include_traceback else None,
            original_exception=type(exception).__name__,
        )

    def _categorize_exception(self, exception: Exception) -> tuple[ErrorCategory, Optional[str]]:
        """Categorize a standard exception and provide recovery hint."""
        # File errors
        if isinstance(exception, (FileNotFoundError, builtins_FileNotFoundError)):
            return ErrorCategory.FILE_NOT_FOUND, "Check if the file exists and path is correct."

        if isinstance(exception, PermissionError):
            return ErrorCategory.FILE_PERMISSION, "Check file permissions."

        # Network errors
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK_ERROR, "Check network connection."

        # Validation errors
        if isinstance(exception, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION_ERROR, "Check input values and types."

        if isinstance(exception, KeyError):
            return ErrorCategory.CONFIG_MISSING, "Check that required configuration is set."

        # Default
        return ErrorCategory.UNKNOWN, None

    def _log_error(
        self,
        error_info: ErrorInfo,
        log_level: Optional[int] = None,
    ) -> None:
        """Log the error with appropriate level."""
        # Determine log level
        if log_level is None:
            level_map = {
                ErrorSeverity.DEBUG: logging.DEBUG,
                ErrorSeverity.INFO: logging.INFO,
                ErrorSeverity.WARNING: logging.WARNING,
                ErrorSeverity.ERROR: logging.ERROR,
                ErrorSeverity.CRITICAL: logging.CRITICAL,
            }
            log_level = level_map.get(error_info.severity, logging.ERROR)

        # Format message
        msg = f"[{error_info.correlation_id}] {error_info.category.value}: {error_info.message}"
        if error_info.details:
            msg += f" | details: {error_info.details}"

        self.logger.log(log_level, msg)

        # Log traceback at debug level
        if error_info.traceback:
            self.logger.debug(
                "[%s] Traceback:\n%s", error_info.correlation_id, error_info.traceback
            )

    def _add_to_history(self, error_info: ErrorInfo) -> None:
        """Add error to history (for debugging/reporting)."""
        self._error_history.append(error_info)
        if len(self._error_history) > self._max_history:
            self._error_history = self._error_history[-self._max_history :]

    def get_recent_errors(self, count: int = 10) -> List[ErrorInfo]:
        """Get recent errors from history."""
        return self._error_history[-count:]

    def clear_history(self) -> None:
        """Clear error history."""
        self._error_history = []


# Alias for builtin FileNotFoundError
builtins_FileNotFoundError = (
    __builtins__["FileNotFoundError"]
    if isinstance(__builtins__, dict)
    else __builtins__.FileNotFoundError
)


# =============================================================================
# Decorators
# =============================================================================


T = TypeVar("T")


def handle_errors(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    recovery_hint: Optional[str] = None,
    reraise: bool = False,
    default_return: Optional[Any] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to handle errors consistently.

    Args:
        category: Error category for logging.
        recovery_hint: Recovery hint to include in error.
        reraise: Whether to reraise the exception after logging.
        default_return: Value to return on error (if not reraising).

    Usage:
        @handle_errors(category=ErrorCategory.TOOL_EXECUTION)
        def risky_function():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            handler = ErrorHandler()
            try:
                return func(*args, **kwargs)
            except VictorError:
                raise  # Don't wrap VictorError
            except Exception as e:
                error_info = handler.handle(
                    e, context={"function": func.__name__, "args_count": len(args)}
                )
                if reraise:
                    raise VictorError(
                        str(e),
                        category=category,
                        recovery_hint=recovery_hint or error_info.recovery_hint,
                        cause=e,
                    ) from e
                return default_return  # type: ignore

        return wrapper

    return decorator


def handle_errors_async(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    recovery_hint: Optional[str] = None,
    reraise: bool = False,
    default_return: Optional[Any] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Async version of handle_errors decorator."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            handler = ErrorHandler()
            try:
                return await func(*args, **kwargs)
            except VictorError:
                raise
            except Exception as e:
                error_info = handler.handle(
                    e, context={"function": func.__name__, "args_count": len(args)}
                )
                if reraise:
                    raise VictorError(
                        str(e),
                        category=category,
                        recovery_hint=recovery_hint or error_info.recovery_hint,
                        cause=e,
                    ) from e
                return default_return  # type: ignore

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# Singleton Error Handler
# =============================================================================


_global_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = ErrorHandler()
    return _global_handler


def handle_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> ErrorInfo:
    """Handle an exception using the global handler.

    Convenience function for quick error handling.

    Args:
        exception: The exception to handle.
        context: Additional context.

    Returns:
        ErrorInfo with structured error details.
    """
    return get_error_handler().handle(exception, context)
