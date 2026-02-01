"""
Error Recovery Chain of Responsibility for Tool Execution Errors.

This module handles tool-level execution errors such as:
- Missing required parameters
- Type conversion errors
- File not found errors
- Permission errors
- Network/timeout errors

NOTE: This module is COMPLEMENTARY to victor/agent/recovery/, not a duplicate:
- THIS module: Handles TOOL execution errors (parameter issues, file errors)
- recovery/ module: Handles LLM response failures (empty responses, stuck loops)

Both modules implement Chain of Responsibility but for different error domains.

SOLID Principles Applied:
- Single Responsibility: Each handler handles one type of error
- Open/Closed: New handlers can be added without modifying existing ones
- Liskov Substitution: All handlers are interchangeable
- Interface Segregation: ErrorRecoveryHandler defines minimal interface
- Dependency Inversion: Handlers depend on abstractions, not concrete classes

Implements GAP-10 from Grok/DeepSeek provider testing.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class ErrorRecoveryAction(Enum):
    """Actions that can be taken to recover from a tool error.

    Renamed from RecoveryAction to be semantically distinct:
    - ErrorRecoveryAction (here): Tool error recovery actions (string values)
    - StrategyRecoveryAction (victor.agent.recovery.protocols): Recovery strategy actions (auto)
    - OrchestratorRecoveryAction (victor.agent.orchestrator_recovery): Orchestrator recovery dataclass
    """

    RETRY = "retry"
    RETRY_WITH_DEFAULTS = "retry_with_defaults"
    RETRY_WITH_INFERRED = "retry_with_inferred"
    SKIP = "skip"
    FALLBACK_TOOL = "fallback_tool"
    ASK_USER = "ask_user"
    ABORT = "abort"


@dataclass
class RecoveryResult:
    """Result of attempting to recover from an error."""

    action: ErrorRecoveryAction
    modified_args: Optional[Dict[str, Any]] = None
    fallback_tool: Optional[str] = None
    user_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_retry(self) -> bool:
        """Check if action is a retry variant."""
        return self.action in (
            ErrorRecoveryAction.RETRY,
            ErrorRecoveryAction.RETRY_WITH_DEFAULTS,
            ErrorRecoveryAction.RETRY_WITH_INFERRED,
        )

    @property
    def can_retry(self) -> bool:
        """Check if more retries are allowed."""
        return self.retry_count < self.max_retries


class ErrorRecoveryHandler(ABC):
    """Abstract handler in the chain of responsibility."""

    def __init__(self) -> None:
        self._next_handler: Optional["ErrorRecoveryHandler"] = None
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def set_next(self, handler: "ErrorRecoveryHandler") -> "ErrorRecoveryHandler":
        """Set the next handler in the chain."""
        self._next_handler = handler
        return handler

    @abstractmethod
    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if this handler can handle the error."""
        pass

    @abstractmethod
    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        """Handle the error and return recovery result."""
        pass

    def process(
        self,
        error: Exception,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> RecoveryResult:
        """Process the error through the chain."""
        if self.can_handle(error, tool_name, args):
            self._logger.info(
                f"Handler {self.__class__.__name__} handling error for {tool_name}: {error}"
            )
            result = self.handle(error, tool_name, args)
            result.metadata["handler"] = self.__class__.__name__
            return result
        elif self._next_handler:
            return self._next_handler.process(error, tool_name, args, context)
        else:
            self._logger.warning(f"No handler could process error for {tool_name}: {error}")
            return RecoveryResult(
                action=ErrorRecoveryAction.ABORT,
                user_message=f"Unrecoverable error in {tool_name}: {error}",
            )


class MissingParameterHandler(ErrorRecoveryHandler):
    """Handle missing required parameter errors."""

    # Default values for common parameters
    DEFAULTS: Dict[str, Any] = {
        "file_path": ".",
        "path": ".",
        "directory": ".",
        "limit": 100,
        "offset": 0,
        "start_line": 1,
        "end_line": -1,  # -1 means end of file
        "recursive": False,
        "max_depth": 3,
    }

    # Patterns to detect missing parameter errors
    PATTERNS = [
        r"missing \d+ required positional argument[s]?: '(\w+)'",
        r"required parameter '(\w+)' was not provided",
        r"'(\w+)' is a required property",
        r"missing required argument[s]?: (\w+)",
    ]

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        error_str = str(error).lower()
        # Check for various missing parameter/argument patterns
        has_missing = "missing" in error_str
        has_required = "required" in error_str
        has_argument = "argument" in error_str
        has_parameter = "parameter" in error_str
        has_property = "property" in error_str

        # Missing argument pattern: "missing 1 required positional argument"
        if has_missing and has_argument:
            return True
        # Required parameter pattern: "required parameter 'x' was not provided"
        if has_required and has_parameter:
            return True
        # Required property pattern: "'x' is a required property"
        if has_required and has_property:
            return True
        # Missing parameter pattern: "missing required argument"
        if has_missing and has_required:
            return True

        return False

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        # Extract missing parameter from error message
        error_str = str(error)
        param_name = None

        for pattern in self.PATTERNS:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                param_name = match.group(1)
                break

        # Fallback: look for quoted parameter names
        if not param_name:
            quoted_match = re.search(r"'(\w+)'", error_str)
            if quoted_match:
                param_name = quoted_match.group(1)

        if param_name and param_name in self.DEFAULTS:
            self._logger.info(
                f"Providing default value for missing param '{param_name}': {self.DEFAULTS[param_name]}"
            )
            return RecoveryResult(
                action=ErrorRecoveryAction.RETRY_WITH_DEFAULTS,
                modified_args={**args, param_name: self.DEFAULTS[param_name]},
                user_message=f"Using default value for '{param_name}'",
            )

        return RecoveryResult(
            action=ErrorRecoveryAction.SKIP,
            user_message=f"Cannot infer value for required parameter '{param_name}'",
        )


class ToolNotFoundHandler(ErrorRecoveryHandler):
    """Handle tool not found errors with fallback alternatives."""

    # Mapping from tools to their fallbacks
    FALLBACKS: Dict[str, str] = {
        "symbol": "grep",  # Fallback from symbol lookup to grep
        "get_symbol": "grep",
        "semantic_search": "grep",
        "semantic_code_search": "code_search",
        "tree": "ls",
        "find_files": "glob",
        "analyze_dependencies": "grep",
    }

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        error_str = str(error).lower()
        return (
            "not found" in error_str or "unknown tool" in error_str or "unregistered" in error_str
        )

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        if tool_name in self.FALLBACKS:
            fallback = self.FALLBACKS[tool_name]
            self._logger.info(f"Falling back from {tool_name} to {fallback}")
            return RecoveryResult(
                action=ErrorRecoveryAction.FALLBACK_TOOL,
                fallback_tool=fallback,
                user_message=f"Tool '{tool_name}' not available, using '{fallback}' instead",
            )

        return RecoveryResult(
            action=ErrorRecoveryAction.SKIP,
            user_message=f"Tool '{tool_name}' not found and no fallback available",
        )


class NetworkErrorHandler(ErrorRecoveryHandler):
    """Handle network-related errors with retry logic."""

    NETWORK_ERROR_PATTERNS = [
        "timeout",
        "connection",
        "network",
        "refused",
        "unreachable",
        "dns",
        "socket",
        "ssl",
        "certificate",
    ]

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self.NETWORK_ERROR_PATTERNS)

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        return RecoveryResult(
            action=ErrorRecoveryAction.RETRY,
            max_retries=3,
            user_message="Network error, retrying...",
            metadata={"retry_delay_seconds": 1.0},
        )


class FileNotFoundHandler(ErrorRecoveryHandler):
    """Handle file not found errors by attempting path variations."""

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        # Check error class first
        if isinstance(error, FileNotFoundError):
            return True
        error_str = str(error).lower()
        return (
            "file not found" in error_str
            or "no such file" in error_str
            or "does not exist" in error_str
        )

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        # Try common path variations
        path = args.get("path") or args.get("file_path") or args.get("file")

        if path:
            variations = self._get_path_variations(path)
            if variations:
                self._logger.info(f"Trying path variation: {variations[0]}")
                path_key = "path" if "path" in args else "file_path"
                return RecoveryResult(
                    action=ErrorRecoveryAction.RETRY_WITH_INFERRED,
                    modified_args={**args, path_key: variations[0]},
                    user_message=f"Trying alternate path: {variations[0]}",
                    metadata={"tried_variations": variations},
                )

        return RecoveryResult(
            action=ErrorRecoveryAction.SKIP,
            user_message="File not found and no alternatives discovered",
        )

    def _get_path_variations(self, path: str) -> List[str]:
        """Generate path variations to try."""
        import os

        variations = []

        # Remove leading ./ or ./
        if path.startswith("./"):
            variations.append(path[2:])
        elif not path.startswith("/"):
            variations.append(f"./{path}")

        # Try with/without extension
        base, ext = os.path.splitext(path)
        if ext == ".py":
            variations.append(f"{base}/__init__.py")
        elif not ext:
            variations.append(f"{path}.py")
            variations.append(f"{path}/index.py")
            variations.append(f"{path}/__init__.py")

        return variations


class RateLimitHandler(ErrorRecoveryHandler):
    """Handle rate limit errors with exponential backoff."""

    RATE_LIMIT_PATTERNS = [
        "rate limit",
        "too many requests",
        "429",
        "throttle",
        "quota exceeded",
    ]

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self.RATE_LIMIT_PATTERNS)

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        # Extract retry-after if present
        retry_after = 5.0  # default
        match = re.search(r"retry.after[:\s]*(\d+)", str(error), re.IGNORECASE)
        if match:
            retry_after = float(match.group(1))

        return RecoveryResult(
            action=ErrorRecoveryAction.RETRY,
            max_retries=3,
            user_message=f"Rate limited, waiting {retry_after}s...",
            metadata={"retry_delay_seconds": retry_after, "exponential_backoff": True},
        )


class PermissionErrorHandler(ErrorRecoveryHandler):
    """Handle permission errors."""

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        # Check error class first
        if isinstance(error, PermissionError):
            return True
        error_str = str(error).lower()
        return (
            "permission denied" in error_str
            or "access denied" in error_str
            or "forbidden" in error_str
        )

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        # For permission errors, we typically can't recover automatically
        return RecoveryResult(
            action=ErrorRecoveryAction.ASK_USER,
            user_message="Permission denied. Please check file permissions or run with elevated privileges.",
        )


class TypeErrorHandler(ErrorRecoveryHandler):
    """Handle type errors in tool arguments."""

    def can_handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> bool:
        return isinstance(error, TypeError) or "type" in str(error).lower()

    def handle(self, error: Exception, tool_name: str, args: Dict[str, Any]) -> RecoveryResult:
        # Try to fix common type issues
        modified_args = dict(args)
        fixed = False

        for key, value in args.items():
            # Convert string booleans
            if isinstance(value, str) and value.lower() in ("true", "false"):
                modified_args[key] = value.lower() == "true"
                fixed = True
            # Convert string numbers
            elif isinstance(value, str):
                try:
                    if "." in value:
                        modified_args[key] = float(value)
                    else:
                        modified_args[key] = int(value)
                    fixed = True
                except ValueError:
                    pass

        if fixed:
            return RecoveryResult(
                action=ErrorRecoveryAction.RETRY_WITH_INFERRED,
                modified_args=modified_args,
                user_message="Converted argument types",
            )

        return RecoveryResult(
            action=ErrorRecoveryAction.SKIP,
            user_message=f"Type error in arguments: {error}",
        )


def build_recovery_chain() -> ErrorRecoveryHandler:
    """Build the default error recovery chain.

    Chain order matters - more specific handlers should come first.
    """
    chain = MissingParameterHandler()
    (
        chain.set_next(TypeErrorHandler())
        .set_next(FileNotFoundHandler())
        .set_next(ToolNotFoundHandler())
        .set_next(RateLimitHandler())
        .set_next(NetworkErrorHandler())
        .set_next(PermissionErrorHandler())
    )
    return chain


# Singleton instance for convenience
_default_chain: Optional[ErrorRecoveryHandler] = None


def get_recovery_chain() -> ErrorRecoveryHandler:
    """Get the default recovery chain (singleton)."""
    global _default_chain
    if _default_chain is None:
        _default_chain = build_recovery_chain()
    return _default_chain


def recover_from_error(
    error: Exception,
    tool_name: str,
    args: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> RecoveryResult:
    """Convenience function to recover from an error using the default chain."""
    chain = get_recovery_chain()
    return chain.process(error, tool_name, args, context)
