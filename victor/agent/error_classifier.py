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

"""Error classification for intelligent tool retry decisions.

Classifies errors as permanent, transient, or retryable to prevent
wasting agent turns on operations that will never succeed.

SOLID Principles:
- SRP: Single responsibility of error classification
- OCP: New error patterns can be added without modifying classification logic
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Set


class ErrorType(Enum):
    """Classification of tool execution errors."""

    PERMANENT = "permanent"
    """Errors that will never succeed on retry (e.g., file not found)."""

    TRANSIENT = "transient"
    """Errors that might succeed later (e.g., network timeout, rate limit)."""

    RETRYABLE = "retryable"
    """Errors where fixing the input might help (e.g., syntax error)."""


@dataclass
class ToolCallSignature:
    """Unique signature for a tool call to detect duplicates."""

    tool_name: str
    arguments_hash: int

    @classmethod
    def from_call(cls, tool_name: str, arguments: dict[str, Any]) -> "ToolCallSignature":
        """Create signature from tool call parameters."""
        # Create a hashable representation of arguments
        # Sort keys for consistent hashing
        try:
            args_tuple = tuple(sorted((k, str(v)) for k, v in arguments.items()))
            args_hash = hash(args_tuple)
        except (TypeError, ValueError):
            # Fallback for unhashable values
            args_hash = hash(str(arguments))

        return cls(tool_name=tool_name, arguments_hash=args_hash)

    def __hash__(self) -> int:
        return hash((self.tool_name, self.arguments_hash))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCallSignature):
            return False
        return (
            self.tool_name == other.tool_name
            and self.arguments_hash == other.arguments_hash
        )


class ToolErrorClassifier:
    """Classifies tool execution errors for retry decisions.

    Tracks permanently failed tool calls to prevent retrying
    operations that will never succeed.
    """

    # Patterns that indicate permanent failure
    PERMANENT_PATTERNS: list[str] = [
        "No such file or directory",
        "Permission denied",
        "ModuleNotFoundError",
        "ImportError: No module named",
        "command not found",
        "FileNotFoundError",
        "IsADirectoryError",
        "NotADirectoryError",
        "PermissionError",
        "directory not empty",
        "File exists",
        "read-only file system",
    ]

    # Patterns that indicate transient failure (might succeed later)
    TRANSIENT_PATTERNS: list[str] = [
        "Connection refused",
        "Connection timed out",
        "Network is unreachable",
        "rate limit",
        "too many requests",
        "Service Unavailable",
        "Gateway Timeout",
        "temporary",
        "try again",
    ]

    def __init__(self) -> None:
        """Initialize the classifier with empty failure tracking."""
        self._failed_calls: Set[ToolCallSignature] = set()

    def classify(self, error_message: str) -> ErrorType:
        """Classify an error message.

        Args:
            error_message: The error message from tool execution

        Returns:
            ErrorType indicating whether the error is permanent, transient, or retryable
        """
        error_lower = error_message.lower()

        # Check for permanent patterns first
        for pattern in self.PERMANENT_PATTERNS:
            if pattern.lower() in error_lower:
                return ErrorType.PERMANENT

        # Check for transient patterns
        for pattern in self.TRANSIENT_PATTERNS:
            if pattern.lower() in error_lower:
                return ErrorType.TRANSIENT

        # Default to retryable (e.g., syntax errors can be fixed)
        return ErrorType.RETRYABLE

    def record_failure(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        error_message: str,
    ) -> ErrorType:
        """Record a tool failure and return its classification.

        If the error is permanent, the call signature is stored to
        prevent future retry attempts.

        Args:
            tool_name: Name of the failed tool
            arguments: Arguments passed to the tool
            error_message: The error message

        Returns:
            ErrorType classification of the error
        """
        error_type = self.classify(error_message)

        if error_type == ErrorType.PERMANENT:
            signature = ToolCallSignature.from_call(tool_name, arguments)
            self._failed_calls.add(signature)

        return error_type

    def should_skip(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        """Check if a tool call should be skipped due to prior permanent failure.

        Args:
            tool_name: Name of the tool
            arguments: Arguments for the tool call

        Returns:
            True if this exact call previously failed permanently
        """
        signature = ToolCallSignature.from_call(tool_name, arguments)
        return signature in self._failed_calls

    def get_skip_reason(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Get reason for skipping a tool call.

        Args:
            tool_name: Name of the tool
            arguments: Arguments for the tool call

        Returns:
            Skip reason message if call should be skipped, None otherwise
        """
        if self.should_skip(tool_name, arguments):
            return (
                f"Skipping {tool_name}: This exact call previously failed with a "
                "permanent error and will not be retried."
            )
        return None

    def reset(self) -> None:
        """Clear all recorded failures."""
        self._failed_calls.clear()

    @property
    def failed_call_count(self) -> int:
        """Number of unique permanently failed calls recorded."""
        return len(self._failed_calls)


# Singleton instance for global error tracking across orchestrator runs
_global_classifier: ToolErrorClassifier | None = None


def get_error_classifier() -> ToolErrorClassifier:
    """Get or create the global error classifier instance."""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = ToolErrorClassifier()
    return _global_classifier


def reset_error_classifier() -> None:
    """Reset the global error classifier."""
    global _global_classifier
    if _global_classifier is not None:
        _global_classifier.reset()
