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
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

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

        # Track error in global error tracker (lazy import to avoid circular dependency)
        try:
            from victor.observability.error_tracker import get_error_tracker

            tracker = get_error_tracker()
            # Include details directly in context for easier access
            context = {
                "category": category.value,
                "severity": severity.value,
            }
            # Add all details to context for direct access
            if details:
                context.update(details)

            tracker.record_error(
                error_type=self.__class__.__name__,
                error_message=str(message),
                correlation_id=self.correlation_id,
                context=context,
            )
        except Exception:
            # Silently fail if error tracker is not available
            # (e.g., during early initialization or testing)
            pass

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
        # Add provider_name alias for backward compatibility
        self.provider_name = provider
        if provider:
            self.details["provider_name"] = provider


class ProviderInitializationError(ProviderError):
    """Provider failed to initialize.

    This error is raised when a provider exists in the registry but fails
    during initialization, typically due to missing configuration or invalid
    credentials.

    Attributes:
        provider: The provider name that failed to initialize
        provider_name: Alias for provider (backward compatibility)
        config_key: The configuration key that is missing or invalid (if known)
        original_error: The underlying exception that caused the failure
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        provider_name: Optional[str] = None,  # Backward compatibility
        config_key: Optional[str] = None,
        recovery_hint: Optional[str] = None,
        **kwargs: Any,
    ):
        # Support both provider and provider_name for backward compatibility
        if provider_name is not None and provider is None:
            provider = provider_name

        # Generate recovery hint if not provided
        if recovery_hint is None:
            if config_key:
                recovery_hint = (
                    f"Set {config_key} environment variable or check your configuration."
                )
            else:
                recovery_hint = "Check your API credentials and configuration."

        super().__init__(
            message,
            provider=provider,
            category=ErrorCategory.CONFIG_INVALID,
            recovery_hint=recovery_hint,
            **kwargs,
        )
        self.config_key = config_key
        self.details["config_key"] = config_key
        # Add provider_name alias for backward compatibility
        self.provider_name = provider or provider_name


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
        # Add tool_name to details before calling super().__init__
        # so it's available in error tracking
        if "details" not in kwargs:
            kwargs["details"] = {}
        kwargs["details"]["tool_name"] = tool_name

        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""

    def __init__(
        self, message: Optional[str] = None, tool_name: Optional[str] = None, **kwargs: Any
    ):
        # Support both positional and keyword arguments
        # If called with positional arg, it's the tool_name
        # If called with keyword arg, it's tool_name
        # If message is provided, use it instead of default
        if message is not None and tool_name is None and isinstance(message, str):
            # First positional arg is tool_name (backward compatibility)
            tool_name = message
            message = None

        if message is None:
            message = f"Tool not found: {tool_name}"

        super().__init__(
            message,
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
        arguments: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # Add recovery hint if not provided
        if kwargs.get("recovery_hint") is None:
            kwargs["recovery_hint"] = (
                "Check tool arguments and permissions. "
                "Verify the tool is compatible with your current environment."
            )

        # Initialize details if not present, then add arguments
        if "details" not in kwargs:
            kwargs["details"] = {}
        kwargs["details"]["arguments"] = arguments or {}

        super().__init__(
            message,
            tool_name=tool_name,
            category=ErrorCategory.TOOL_EXECUTION,
            **kwargs,
        )
        self.arguments = arguments or {}
        # Note: arguments already in details from super().__init__


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
        message: Optional[str] = None,
        tool_name: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        # If message is not provided, generate one
        if message is None:
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
        invalid_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        # Add recovery hint if not provided
        if kwargs.get("recovery_hint") is None:
            if config_key:
                kwargs["recovery_hint"] = (
                    f"Check configuration for '{config_key}'. Set the correct value in config or environment variables."
                )
            else:
                kwargs["recovery_hint"] = "Check your configuration file and environment variables."

        super().__init__(
            message,
            category=ErrorCategory.CONFIG_INVALID,
            **kwargs,
        )
        self.config_key = config_key
        self.details["config_key"] = config_key
        self.invalid_fields = invalid_fields or []
        self.details["invalid_fields"] = self.invalid_fields


class ConfigurationValidationError(ConfigurationError):
    """Configuration validation errors with detailed field information.

    This exception is raised when configuration validation fails,
    providing structured information about which fields are invalid
    and how to fix them.

    Attributes:
        config_key: The configuration key or file path that failed validation
        invalid_fields: List of field names that failed validation
        field_errors: Dictionary mapping field names to their error messages
        line_numbers: Dictionary mapping field names to line numbers (if available)
        validation_errors: List of detailed validation error messages

    Example:
        raise ConfigurationValidationError(
            message="Workflow validation failed with 3 errors",
            config_key="/path/to/workflow.yaml",
            invalid_fields=["start_node", "tool_budget"],
            field_errors={
                "start_node": "Start node 'init' not found in workflow",
                "tool_budget": "Tool budget must be positive integer"
            },
            line_numbers={"start_node": 15, "tool_budget": 23},
            recovery_hint="Fix validation errors in YAML file. Use 'victor workflow validate <path>' to check."
        )
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        invalid_fields: Optional[List[str]] = None,
        field_errors: Optional[Dict[str, str]] = None,
        line_numbers: Optional[Dict[str, int]] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        # Build detailed error message if not provided
        if validation_errors and len(message) < 200:
            # message is short, let's enhance it
            error_list = "\n  - ".join(validation_errors[:5])
            if len(validation_errors) > 5:
                error_list += f"\n  - ... and {len(validation_errors) - 5} more"
            message = f"Configuration validation failed with {len(validation_errors)} error(s):\n  - {error_list}"

        # Build recovery hint if not provided
        if kwargs.get("recovery_hint") is None:
            if config_key and config_key.endswith(".yaml"):
                kwargs["recovery_hint"] = (
                    f"Fix validation errors in '{config_key}'. Use 'victor workflow validate <path>' to check."
                )
            elif config_key:
                kwargs["recovery_hint"] = (
                    f"Fix validation errors in '{config_key}'. Check configuration format and required fields."
                )
            else:
                kwargs["recovery_hint"] = (
                    "Fix validation errors. Check configuration format and required fields."
                )

        # Initialize parent ConfigurationError
        super().__init__(
            message,
            config_key=config_key,
            invalid_fields=invalid_fields,
            **kwargs,
        )

        # Add additional structured error information
        self.field_errors = field_errors or {}
        self.line_numbers = line_numbers or {}
        self.validation_errors = validation_errors or []

        # Add to details for logging/tracking
        if self.field_errors:
            self.details["field_errors"] = self.field_errors
        if self.line_numbers:
            self.details["line_numbers"] = self.line_numbers
        if self.validation_errors:
            self.details["validation_errors"] = self.validation_errors

    def get_field_error(self, field_name: str) -> Optional[str]:
        """Get error message for a specific field.

        Args:
            field_name: Name of the field to get error for

        Returns:
            Error message for the field, or None if no error
        """
        return self.field_errors.get(field_name)

    def get_line_number(self, field_name: str) -> Optional[int]:
        """Get line number for a specific field error.

        Args:
            field_name: Name of the field

        Returns:
            Line number, or None if not available
        """
        return self.line_numbers.get(field_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with full validation details."""
        result = super().to_dict()
        result["field_errors"] = self.field_errors
        result["line_numbers"] = self.line_numbers
        result["validation_errors"] = self.validation_errors
        return result


class CapabilityRegistryRequiredError(ConfigurationError):
    """Raised when capability registry is required but not available.

    This exception is raised when a component requires a capability registry
    to be present on the orchestrator, but the orchestrator doesn't have one.
    This enforces dependency inversion and prevents fallback to private field writes.

    Attributes:
        component: The name of the component that requires the capability registry
        capability_name: The name of the capability being accessed
        required_methods: List of required methods that were expected

    Example:
        raise CapabilityRegistryRequiredError(
            component="VerticalIntegrationAdapter",
            capability_name="vertical_middleware",
            required_methods=["has_capability", "get_capability", "set_capability"]
        )
    """

    def __init__(
        self,
        message: Optional[str] = None,
        component: Optional[str] = None,
        capability_name: Optional[str] = None,
        required_methods: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        # Build message if not provided
        if message is None:
            if component and capability_name:
                message = (
                    f"{component} requires capability registry for '{capability_name}', "
                    f"but orchestrator does not support capability operations. "
                    f"Required methods: {', '.join(required_methods or [])}"
                )
            elif component:
                message = (
                    f"{component} requires capability registry, "
                    f"but orchestrator does not support capability operations."
                )
            else:
                message = (
                    "Capability registry is required but not available on orchestrator. "
                    "Ensure orchestrator supports capability operations."
                )

        # Build recovery hint if not provided
        if kwargs.get("recovery_hint") is None:
            if component:
                kwargs["recovery_hint"] = (
                    f"Update {component} to use capability registry methods instead of "
                    f"direct attribute access. Ensure orchestrator is properly initialized "
                    f"with capability support."
                )
            else:
                kwargs["recovery_hint"] = (
                    "Use capability registry methods (has_capability, get_capability, "
                    "set_capability) instead of direct attribute access."
                )

        super().__init__(message, **kwargs)

        self.component = component or "Unknown"
        self.capability_name = capability_name
        self.required_methods = required_methods or []

        # Add to details for logging/tracking
        if self.component:
            self.details["component"] = self.component
        if self.capability_name:
            self.details["capability_name"] = self.capability_name
        if self.required_methods:
            self.details["required_methods"] = self.required_methods


class ValidationError(VictorError):
    """Input validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ):
        # Add recovery hint if not provided
        if kwargs.get("recovery_hint") is None:
            if field:
                kwargs["recovery_hint"] = (
                    f"Check the value for '{field}'. Ensure it matches the expected format and type."
                )
            else:
                kwargs["recovery_hint"] = (
                    "Check your input values and ensure they match the expected format."
                )

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

    def __init__(self, message: Optional[str] = None, path: Optional[str] = None, **kwargs: Any):
        # Support both positional and keyword arguments
        # If called with positional arg, it's the path
        if message is not None and path is None and isinstance(message, str):
            path = message
            message = None

        if message is None:
            message = f"File not found: {path}"

        super().__init__(
            message,
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


class SearchError(VictorError):
    """Errors related to search backend operations.

    This exception is raised when search backends fail or encounter errors
    during query execution. It provides detailed information about which
    backends failed and why.

    Attributes:
        search_type: The type of search that failed (semantic, keyword, hybrid)
        failed_backends: List of backend names that failed
        failure_details: Dictionary mapping backend names to their error messages
        query: The search query that failed (optional)

    Example:
        raise SearchError(
            message="All 2 search backends failed for 'semantic'",
            search_type="semantic",
            failed_backends=["SemanticSearchBackend", "VectorSearchBackend"],
            failure_details={
                "SemanticSearchBackend": "Connection timeout",
                "VectorSearchBackend": "Index not found"
            },
            query="authentication logic"
        )
    """

    def __init__(
        self,
        message: str,
        search_type: Optional[str] = None,
        failed_backends: Optional[List[str]] = None,
        failure_details: Optional[Dict[str, str]] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ):
        # Generate recovery hint if not provided
        if kwargs.get("recovery_hint") is None:
            if failed_backends:
                backend_list = ", ".join(failed_backends[:3])
                if len(failed_backends) > 3:
                    backend_list += f" (and {len(failed_backends) - 3} more)"

                kwargs["recovery_hint"] = (
                    f"Check backend configuration and connectivity for: {backend_list}. "
                    f"Try: 1) Check network connection, 2) Verify API keys, "
                    f"3) Try alternative search type with 'victor config search.type=<type>'"
                )
            else:
                kwargs["recovery_hint"] = (
                    "Check search backend configuration. Try alternative search type."
                )

        super().__init__(
            message,
            category=ErrorCategory.NETWORK_ERROR,
            **kwargs,
        )
        self.search_type = search_type
        self.failed_backends = failed_backends or []
        self.failure_details = failure_details or {}
        self.query = query

        # Add to details
        if search_type:
            self.details["search_type"] = search_type
        if failed_backends:
            self.details["failed_backends"] = failed_backends
        if failure_details:
            self.details["failure_details"] = failure_details
        if query:
            self.details["query"] = query


class WorkflowExecutionError(VictorError):
    """Errors related to workflow execution failures.

    This exception is raised when a workflow fails during execution,
    providing detailed information about where the failure occurred
    and how to recover.

    Attributes:
        workflow_id: Identifier of the workflow that failed
        node_id: Identifier of the node where failure occurred
        node_type: Type of node (agent, compute, condition, etc.)
        checkpoint_id: Checkpoint ID for resuming (if available)
        execution_context: Additional context about the execution

    Example:
        raise WorkflowExecutionError(
            message="Workflow execution failed at node 'data_processor'",
            workflow_id="deep_research",
            node_id="data_processor",
            node_type="compute",
            checkpoint_id="chk_abc123",
            execution_context={"iteration": 3, "input_size": 1000}
        )
    """

    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        node_id: Optional[str] = None,
        node_type: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        execution_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # Generate recovery hint if not provided
        if kwargs.get("recovery_hint") is None:
            if checkpoint_id:
                if node_id:
                    kwargs["recovery_hint"] = (
                        f"Check workflow logs with correlation ID. "
                        f"Use checkpoint '{checkpoint_id}' to resume from node '{node_id}'. "
                        f"Fix the node and retry workflow."
                    )
                else:
                    kwargs["recovery_hint"] = (
                        f"Use checkpoint '{checkpoint_id}' to resume workflow execution. "
                        f"Fix the error and retry."
                    )
            elif node_id:
                kwargs["recovery_hint"] = (
                    f"Fix node '{node_id}' and retry workflow execution. "
                    f"Check logs for detailed error information."
                )
            else:
                kwargs["recovery_hint"] = (
                    "Check workflow logs for detailed error information. "
                    "Fix the error and retry workflow execution."
                )

        super().__init__(
            message,
            category=ErrorCategory.INTERNAL_ERROR,
            **kwargs,
        )
        self.workflow_id = workflow_id
        self.node_id = node_id
        self.node_type = node_type
        self.checkpoint_id = checkpoint_id
        self.execution_context = execution_context or {}

        # Add to details
        if workflow_id:
            self.details["workflow_id"] = workflow_id
        if node_id:
            self.details["node_id"] = node_id
        if node_type:
            self.details["node_type"] = node_type
        if checkpoint_id:
            self.details["checkpoint_id"] = checkpoint_id
        if execution_context:
            self.details["execution_context"] = execution_context


class RecursionDepthError(WorkflowExecutionError):
    """Raised when maximum recursion depth is exceeded.

    This exception is raised when nested workflow or team execution
    exceeds the maximum allowed recursion depth, preventing infinite
    nesting and stack overflow.

    Attributes:
        current_depth: Current recursion level when limit was exceeded
        max_depth: Maximum allowed recursion depth
        execution_stack: Stack trace of execution entries leading to the error

    Example:
        >>> raise RecursionDepthError(
        ...     message="Maximum recursion depth exceeded",
        ...     current_depth=4,
        ...     max_depth=3,
        ...     execution_stack=["workflow:main", "team:outer", "team:middle", "team:inner"]
        ... )
    """

    def __init__(
        self,
        message: str,
        current_depth: int,
        max_depth: int,
        execution_stack: List[str],
        **kwargs: Any,
    ):
        # Generate stack trace for error message
        stack_str = " → ".join(execution_stack) if execution_stack else "empty"

        # Generate recovery hint if not provided
        # Note: Don't set recovery_hint here as parent class will generate one
        # We'll override it after parent construction if needed

        # Call parent constructor
        # Note: parent class sets category=INTERNAL_ERROR, so we pass severity separately
        # and will override category after construction
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            **kwargs,
        )

        # Override category to RESOURCE_EXHAUSTED (more specific than parent's INTERNAL_ERROR)
        self.category = ErrorCategory.RESOURCE_EXHAUSTED

        # Override recovery hint if not provided by parent
        if kwargs.get("recovery_hint") is None:
            self.recovery_hint = (
                f"Reduce nesting depth in your workflow or team configuration. "
                f"Current depth: {current_depth}/{max_depth}. "
                f"Execution path: {stack_str}. "
                f"Consider restructuring to reduce nesting or increase max_recursion_depth."
            )

        self.current_depth = current_depth
        self.max_depth = max_depth
        self.execution_stack = execution_stack

        # Add to details
        self.details["current_depth"] = current_depth
        self.details["max_depth"] = max_depth
        self.details["execution_stack"] = execution_stack
        self.details["stack_trace"] = stack_str

    def __str__(self) -> str:
        stack_str = " → ".join(self.execution_stack) if self.execution_stack else "empty"
        return (
            f"[{self.correlation_id}] Recursion depth limit exceeded: "
            f"{self.current_depth}/{self.max_depth}\n"
            f"Execution stack: {stack_str}\n"
            f"{self.message}"
        )


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
                result = await func(*args, **kwargs)
                return cast(T, result)
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
                return cast(T, default_return)

        return wrapper

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
