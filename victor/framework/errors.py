"""User-friendly error types for the Victor framework API.

These exceptions provide clear, actionable error messages for common
failure scenarios when using the simplified framework API.

ProviderError, ToolError, and ConfigurationError are thin wrappers over
their core counterparts. AgentError is re-exported from victor.core.errors.
No diamond inheritance — core classes already inherit AgentError via
victor.core.errors, so framework wrappers only need single inheritance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# AgentError is now canonical in core — re-export for backward compatibility.
from victor.core.errors import AgentError  # noqa: F401 (re-export)

# Import core classes to wrap (no alias rename needed here)
from victor.core.errors import (
    ProviderError as CoreProviderError,
    ToolError as CoreToolError,
    ConfigurationError as CoreConfigurationError,
)

__all__ = [
    "AgentError",
    "ProviderError",
    "ToolError",
    "ConfigurationError",
    "BudgetExhaustedError",
    "CancellationError",
    "StateTransitionError",
    "EdgeResolutionError",
]


class ProviderError(CoreProviderError):
    """Error from LLM provider communication.

    Single-inheritance wrapper (no diamond) — CoreProviderError already
    inherits AgentError since Phase 3 consolidation.

    Attributes:
        provider: Name of the provider that failed
        status_code: HTTP status code if applicable
        recoverable: Whether the operation can be retried
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = None,
        status_code: Optional[int] = None,
        recoverable: bool = True,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        CoreProviderError.__init__(
            self, message=message, provider=provider, status_code=status_code, **kwargs
        )
        self.recoverable = recoverable
        self.details = details or {}

    def __str__(self) -> str:
        if hasattr(self, "provider") and self.provider:
            if hasattr(self, "status_code") and self.status_code:
                return f"[{self.provider}] {self.message} (status: {self.status_code})"
            return f"[{self.provider}] {self.message}"
        return self.message


class ToolError(CoreToolError):
    """Error from tool execution.

    Single-inheritance wrapper — CoreToolError already inherits AgentError.

    Attributes:
        tool_name: Name of the tool that failed
        arguments: Arguments that were passed to the tool
        recoverable: Whether the operation can be retried
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = None,
        arguments: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        CoreToolError.__init__(self, message=message, tool_name=tool_name, **kwargs)
        self.arguments = arguments or {}
        self.recoverable = recoverable
        self.details = details or {}

    def __str__(self) -> str:
        if hasattr(self, "tool_name") and self.tool_name:
            return f"[{self.tool_name}] {self.message}"
        return self.message


class ConfigurationError(CoreConfigurationError):
    """Error in agent configuration.

    Single-inheritance wrapper — CoreConfigurationError already inherits AgentError.

    Attributes:
        invalid_fields: List of field names that are invalid
        recoverable: Always False — config errors cannot be retried
    """

    def __init__(
        self,
        message: str,
        *,
        invalid_fields: Optional[List[str]] = None,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        CoreConfigurationError.__init__(self, message=message, config_key=config_key, **kwargs)
        self.invalid_fields = invalid_fields or []
        self.recoverable = False
        self.details = details or {}

    def __str__(self) -> str:
        if self.invalid_fields:
            fields = ", ".join(self.invalid_fields)
            return f"{self.message} (invalid fields: {fields})"
        return self.message


class BudgetExhaustedError(AgentError):
    """Tool budget has been exhausted."""

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
    """Operation was cancelled."""

    def __init__(
        self,
        message: str = "Operation cancelled",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, recoverable=False, **kwargs)


class StateTransitionError(AgentError):
    """Invalid state transition attempted.

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


class EdgeResolutionError(AgentError):
    """No conditional edge matched the current state."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, recoverable=False, **kwargs)
