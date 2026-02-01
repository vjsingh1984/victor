"""User-friendly error types for the Victor framework API.

These exceptions provide clear, actionable error messages for common
failure scenarios when using the simplified framework API.

Note: ProviderError, ToolError, and ConfigurationError are imported from
victor.core.errors and wrapped with AgentError for backward compatibility.
This eliminates duplication while maintaining the framework's user-friendly API.
"""

from __future__ import annotations

from typing import Any, Optional

# Import core error types (single source of truth)
from victor.core.errors import (
    ProviderError as CoreProviderError,
    ToolError as CoreToolError,
    ConfigurationError as CoreConfigurationError,
)


class AgentError(Exception):
    """Base exception for agent errors.

    All framework-specific exceptions inherit from this class,
    making it easy to catch all agent-related errors.

    Attributes:
        message: Human-readable error description
        recoverable: Whether the operation can be retried
        details: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        *,
        recoverable: bool = True,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.recoverable = recoverable
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


class ProviderError(CoreProviderError, AgentError):
    """Error from LLM provider communication.

    Raised when there's an issue communicating with the LLM provider,
    such as authentication failures, rate limits, or network errors.

    This class wraps victor.core.errors.ProviderError with AgentError
    for backward compatibility with the framework API.

    Attributes:
        provider: Name of the provider that failed
        status_code: HTTP status code if applicable
        message: Human-readable error description
        recoverable: Whether the operation can be retried
        details: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        recoverable: bool = True,
        details: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize core ProviderError with all its expected parameters
        CoreProviderError.__init__(
            self, message=message, provider=provider, status_code=status_code, **kwargs
        )
        # Also set AgentError attributes for framework compatibility
        self.recoverable = recoverable
        self.details = details or {}

    def __str__(self) -> str:
        # Framework-friendly formatting
        if hasattr(self, "provider") and self.provider:
            if hasattr(self, "status_code") and self.status_code:
                return f"[{self.provider}] {self.message} (status: {self.status_code})"
            return f"[{self.provider}] {self.message}"
        return self.message


class ToolError(CoreToolError, AgentError):
    """Error from tool execution.

    Raised when a tool fails to execute properly, such as
    file not found, permission denied, or invalid arguments.

    This class wraps victor.core.errors.ToolError with AgentError
    for backward compatibility with the framework API.

    Attributes:
        tool_name: Name of the tool that failed
        arguments: Arguments that were passed to the tool
        message: Human-readable error description
        recoverable: Whether the operation can be retried
        details: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: Optional[str] = None,
        arguments: Optional[dict[str, Any]] = None,
        recoverable: bool = True,
        details: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize core ToolError
        CoreToolError.__init__(self, message=message, tool_name=tool_name, **kwargs)
        # Set framework-specific attributes
        self.arguments = arguments or {}
        self.recoverable = recoverable
        self.details = details or {}

    def __str__(self) -> str:
        # Framework-friendly formatting
        if hasattr(self, "tool_name") and self.tool_name:
            return f"[{self.tool_name}] {self.message}"
        return self.message


class ConfigurationError(CoreConfigurationError, AgentError):
    """Error in agent configuration.

    Raised when the agent is configured with invalid or
    incompatible settings.

    This class wraps victor.core.errors.ConfigurationError with AgentError
    for backward compatibility with the framework API.

    Attributes:
        invalid_fields: List of field names that are invalid
        config_key: Specific config key that failed (from core)
        message: Human-readable error description
        recoverable: Whether the operation can be retried (always False)
        details: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        *,
        invalid_fields: Optional[list[str]] = None,
        config_key: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize core ConfigurationError
        CoreConfigurationError.__init__(self, message=message, config_key=config_key, **kwargs)
        # Set framework-specific attributes
        self.invalid_fields = invalid_fields or []
        self.recoverable = False  # Configuration errors are not recoverable
        self.details = details or {}

    def __str__(self) -> str:
        # Framework-friendly formatting
        if self.invalid_fields:
            fields = ", ".join(self.invalid_fields)
            return f"{self.message} (invalid fields: {fields})"
        return self.message


class BudgetExhaustedError(AgentError):
    """Tool budget has been exhausted.

    Raised when the agent has used all available tool calls
    and cannot continue processing.

    Attributes:
        budget: Total tool call budget
        used: Number of tool calls made
    """

    def __init__(
        self,
        message: str = "Tool call budget exhausted",
        *,
        budget: int,
        used: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, recoverable=False, **kwargs)
        self.budget = budget
        self.used = used

    def __str__(self) -> str:
        return f"{self.message} ({self.used}/{self.budget} calls used)"


class CancellationError(AgentError):
    """Operation was cancelled.

    Raised when an operation is cancelled by the user or system,
    such as pressing Ctrl+C or timeout.
    """

    def __init__(
        self,
        message: str = "Operation cancelled",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, recoverable=False, **kwargs)


class StateTransitionError(AgentError):
    """Invalid state transition attempted.

    Raised when attempting to transition to an invalid state
    in the conversation state machine.

    Attributes:
        from_state: Current state
        to_state: Attempted target state
    """

    def __init__(
        self,
        message: str,
        *,
        from_state: str,
        to_state: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, recoverable=False, **kwargs)
        self.from_state = from_state
        self.to_state = to_state

    def __str__(self) -> str:
        return f"{self.message} ({self.from_state} -> {self.to_state})"
