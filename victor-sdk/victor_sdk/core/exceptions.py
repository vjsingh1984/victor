"""Exception definitions for Victor SDK.

These exceptions provide typed error handling for vertical-specific errors
without depending on any runtime implementations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class VerticalException(Exception):
    """Base exception for all vertical-related errors.

    This exception provides structured error information for vertical
    configuration and protocol errors.
    """

    def __init__(
        self,
        message: str,
        vertical_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the exception.

        Args:
            message: Human-readable error message
            vertical_name: Name of the vertical that caused the error
            details: Additional error details as key-value pairs
        """
        self.message = message
        self.vertical_name = vertical_name
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context."""
        parts = [self.message]

        if self.vertical_name:
            parts.append(f"Vertical: {self.vertical_name}")

        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {detail_str}")

        return " | ".join(parts)


class VerticalConfigurationError(VerticalException):
    """Exception raised when vertical configuration is invalid.

    This includes:
    - Missing required configuration
    - Invalid configuration values
    - Conflicting configuration settings
    """

    pass


class VerticalProtocolError(VerticalException):
    """Exception raised when vertical protocol implementation is invalid.

    This includes:
    - Missing required protocol methods
    - Invalid protocol method signatures
    - Protocol contract violations
    """

    pass
