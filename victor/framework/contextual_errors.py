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

"""Contextual error handling with actionable suggestions.

This module provides error classes and helper functions that create
error messages with:
- What operation was being attempted
- Why it failed (technical reason)
- How to fix it (actionable suggestions)
- Related documentation or commands

Example:
    try:
        provider.call_api()
    except APIError as e:
        raise create_provider_error(
            operation="generate completion",
            provider="anthropic",
            error=e,
            suggestion="Check your API key: export ANTHROPIC_API_KEY=your_key",
        )
"""

from __future__ import annotations

import os
import sys
from typing import Any

# =============================================================================
# Contextual Error Classes
# =============================================================================


class ContextualError(Exception):
    """Base error class with context and suggestions."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        suggestion: str | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.operation = operation
        self.suggestion = suggestion
        self.error_code = error_code
        self.details = details or {}

        # Build full error message
        full_message = self._build_message()
        super().__init__(full_message)

    def _build_message(self) -> str:
        """Build the full error message with context."""
        parts = []

        if self.operation:
            parts.append(f"[{self.operation}]")

        parts.append(self.message)

        if self.suggestion:
            parts.append(f"\n\n💡 Suggestion: {self.suggestion}")

        if self.error_code:
            parts.append(f"\n\nError Code: {self.error_code}")

        if self.details:
            parts.append("\n\nDetails:")
            for key, value in self.details.items():
                parts.append(f"  • {key}: {value}")

        return "\n".join(parts)


class ProviderConnectionError(ContextualError):
    """Error when provider connection fails."""

    def __init__(
        self,
        provider: str,
        error: Exception | None = None,
        suggestion: str | None = None,
    ):
        self.provider = provider
        self.original_error = error

        # Build default suggestion based on provider
        if suggestion is None:
            suggestion = self._get_default_suggestion()

        message = f"Failed to connect to {provider} provider"
        if error:
            message += f": {error}"

        super().__init__(
            message=message,
            operation=f"Provider Initialization: {provider}",
            suggestion=suggestion,
            error_code="PROVIDER_CONNECTION_FAILED",
            details={"provider": provider},
        )

    def _get_default_suggestion(self) -> str:
        """Get default suggestion for this provider."""
        local_providers = {
            "ollama": "Start Ollama: 'ollama serve' or 'brew services start ollama'",
            "lmstudio": "Start LM Studio and ensure the API server is running",
            "vllm": "Start vLLM server: 'python -m vllm.entrypoints.api_server'",
            "llama.cpp": "Start llama.cpp server",
        }

        cloud_providers = {
            "anthropic": "Set ANTHROPIC_API_KEY environment variable",
            "openai": "Set OPENAI_API_KEY environment variable",
            "google": "Set GOOGLE_API_KEY environment variable",
            "azure": "Set AZURE_API_KEY and AZURE_API_BASE environment variables",
            "xai": "Set XAI_API_KEY environment variable",
            "cohere": "Set COHERE_API_KEY environment variable",
            "bedrock": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables",
        }

        if self.provider.lower() in local_providers:
            return local_providers[self.provider.lower()]
        elif self.provider.lower() in cloud_providers:
            env_var = self._get_api_key_var()
            suggestion = f"Set {env_var} environment variable"
            if self.provider.lower() == "azure":
                suggestion += " and AZURE_API_BASE"
            return suggestion

        return f"Check {self.provider} provider configuration and credentials"

    def _get_api_key_var(self) -> str:
        """Get the API key environment variable for this provider."""
        return {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_API_KEY",
            "xai": "XAI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID",
        }.get(self.provider.lower(), f"{self.provider.upper()}_API_KEY")


class ToolExecutionError(ContextualError):
    """Error when tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        operation: str,
        error: Exception | None = None,
        suggestion: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.tool_name = tool_name
        self.original_error = error

        message = f"Tool '{tool_name}' failed during {operation}"
        if error:
            message += f": {error}"

        all_details = {"tool": tool_name, "operation": operation}
        if details:
            all_details.update(details)

        super().__init__(
            message=message,
            operation=f"Tool Execution: {tool_name}",
            suggestion=suggestion or self._get_default_suggestion(operation, error),
            error_code="TOOL_EXECUTION_FAILED",
            details=all_details,
        )

    def _get_default_suggestion(self, operation: str, error: Exception | None) -> str:
        """Get default suggestion based on operation and error."""
        error_str = str(error).lower() if error else ""

        if "permission" in error_str or "access" in error_str:
            return "Check file/directory permissions. Try: ls -la <path>"
        elif "not found" in error_str or "no such file" in error_str:
            return "Verify the file/directory exists. Try: ls <path>"
        elif "docker" in error_str:
            return "Ensure Docker is running: docker ps"
        elif "network" in error_str or "connection" in error_str:
            return "Check network connectivity. Try: ping example.com"
        elif "timeout" in error_str:
            return "Operation timed out. Try again with a longer timeout or smaller input"

        return f"Check tool configuration and input parameters for '{operation}'"


class FileOperationError(ContextualError):
    """Error when file operations fail."""

    def __init__(
        self,
        operation: str,
        path: str,
        error: Exception | None = None,
        suggestion: str | None = None,
    ):
        self.path = path
        self.original_error = error

        message = f"Failed to {operation} '{path}'"
        if error:
            message += f": {error}"

        super().__init__(
            message=message,
            operation=f"File Operation: {operation}",
            suggestion=suggestion or self._get_default_suggestion(operation, error),
            error_code="FILE_OPERATION_FAILED",
            details={"path": path, "operation": operation},
        )

    def _get_default_suggestion(self, operation: str, error: Exception | None) -> str:
        """Get default suggestion for file operation."""
        error_str = str(error).lower() if error else ""

        if operation == "read" or operation == "open":
            if "not found" in error_str or "no such file" in error_str:
                return f"Verify the file exists: ls '{self.path}'"
            elif "permission" in error_str:
                return f"Check file permissions: ls -la '{self.path}'"

        elif operation == "write" or operation == "create":
            if "permission" in error_str:
                return f"Check directory permissions: ls -la $(dirname '{self.path}')"
            elif "no space" in error_str:
                return "Check disk space: df -h"

        return f"Check path and permissions: ls -la '{os.path.dirname(self.path) or '.'}'"


class ConfigurationError(ContextualError):
    """Error when configuration is invalid."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        suggestion: str | None = None,
        validation_errors: list[str] | None = None,
    ):
        self.field = field
        self.validation_errors = validation_errors or []

        full_suggestion = suggestion
        if not full_suggestion:
            full_suggestion = "Run 'victor config validate' for detailed diagnostics"

            if self.field:
                full_suggestion += f"\nCheck {self.field} in your profile YAML"

        all_details = {}
        if field:
            all_details["field"] = field
        if validation_errors:
            all_details["errors"] = validation_errors

        super().__init__(
            message=message,
            operation="Configuration Validation",
            suggestion=full_suggestion,
            error_code="CONFIGURATION_INVALID",
            details=all_details,
        )


class ResourceError(ContextualError):
    """Error when system resources are insufficient."""

    def __init__(
        self,
        resource_type: str,
        error: Exception | None = None,
        suggestion: str | None = None,
    ):
        self.resource_type = resource_type
        self.original_error = error

        message = f"Insufficient {resource_type}"
        if error:
            message += f": {error}"

        if not suggestion:
            suggestion = self._get_default_suggestion(resource_type)

        super().__init__(
            message=message,
            operation=f"Resource Check: {resource_type}",
            suggestion=suggestion,
            error_code="RESOURCE_INSUFFICIENT",
            details={"resource_type": resource_type},
        )

    def _get_default_suggestion(self, resource_type: str) -> str:
        """Get default suggestion for resource type."""
        if resource_type == "memory":
            return "Free up memory or increase available RAM. Check: free -h"
        elif resource_type == "disk":
            return "Free up disk space. Check: df -h"
        elif resource_type == "docker":
            return "Ensure Docker is running and has resources. Check: docker ps"
        else:
            return f"Check {resource_type} availability and system resources"


# =============================================================================
# Helper Functions
# =============================================================================


def create_provider_error(
    provider: str,
    operation: str,
    error: Exception,
    suggestion: str | None = None,
) -> ProviderConnectionError:
    """Create a contextual provider error.

    Args:
        provider: Provider name (e.g., "anthropic", "ollama")
        operation: What operation was being attempted
        error: The original exception
        suggestion: Optional custom suggestion

    Returns:
        ProviderConnectionError with context

    Example:
        try:
            response = await client.generate()
        except APIError as e:
            raise create_provider_error(
                provider="anthropic",
                operation="generate completion",
                error=e,
            )
    """
    return ProviderConnectionError(
        provider=provider,
        error=error,
        suggestion=suggestion,
    )


def create_tool_error(
    tool_name: str,
    operation: str,
    error: Exception,
    suggestion: str | None = None,
    details: dict[str, Any] | None = None,
) -> ToolExecutionError:
    """Create a contextual tool execution error.

    Args:
        tool_name: Name of the tool that failed
        operation: What operation was being attempted
        error: The original exception
        suggestion: Optional custom suggestion
        details: Optional additional details

    Returns:
        ToolExecutionError with context
    """
    return ToolExecutionError(
        tool_name=tool_name,
        operation=operation,
        error=error,
        suggestion=suggestion,
        details=details,
    )


def create_file_error(
    operation: str,
    path: str,
    error: Exception,
    suggestion: str | None = None,
) -> FileOperationError:
    """Create a contextual file operation error.

    Args:
        operation: What operation (read, write, delete, etc.)
        path: File path that failed
        error: The original exception
        suggestion: Optional custom suggestion

    Returns:
        FileOperationError with context
    """
    return FileOperationError(
        operation=operation,
        path=path,
        error=error,
        suggestion=suggestion,
    )


def wrap_error(
    error: Exception,
    context: str,
    suggestion: str | None = None,
) -> ContextualError:
    """Wrap any exception with contextual information.

    Args:
        error: The original exception
        context: Description of what was being attempted
        suggestion: Optional custom suggestion

    Returns:
        ContextualError wrapping the original error

    Example:
        try:
            parse_config(file_path)
        except ValueError as e:
            raise wrap_error(e, "parse configuration file", suggestion="Check YAML syntax")
    """
    return ContextualError(
        message=str(error),
        operation=context,
        suggestion=suggestion,
        details={"original_type": type(error).__name__},
    )


def format_exception_for_user(error: Exception) -> str:
    """Format an exception for user-friendly display.

    Args:
        error: The exception to format

    Returns:
        User-friendly error message with suggestions
    """
    # Already a contextual error
    if isinstance(error, ContextualError):
        return str(error)

    # Provider errors
    if "api key" in str(error).lower() or "unauthorized" in str(error).lower():
        return (
            f"{error}\n\n"
            "💡 Suggestion: Check your API key configuration.\n"
            "Run 'victor doctor' to verify provider setup."
        )

    # Network errors
    if "connection" in str(error).lower() or "network" in str(error).lower():
        return (
            f"{error}\n\n"
            "💡 Suggestion: Check your internet connection.\n"
            "For local providers (Ollama, LM Studio), ensure the service is running."
        )

    # Permission errors
    if "permission" in str(error).lower():
        return (
            f"{error}\n\n" "💡 Suggestion: Check file/directory permissions.\n" "Try: ls -la <path>"
        )

    # File not found
    if "not found" in str(error).lower():
        return f"{error}\n\n" "💡 Suggestion: Verify the file/path exists.\n" "Try: ls <path>"

    # Generic error with system info
    return (
        f"{error}\n\n"
        "💡 Run 'victor doctor' for diagnostics\n"
        f"System: Python {sys.version_info.major}.{sys.version_info.minor} | "
        f"{sys.platform}"
    )
