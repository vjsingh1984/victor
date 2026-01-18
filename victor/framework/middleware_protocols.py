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

"""Generic Middleware Library - Protocol Definitions.

This module provides protocol definitions for tool execution middleware.
These protocols enable interception and modification of tool calls before
and after execution, supporting use cases like validation, transformation,
logging, and domain-specific processing.

The middleware system follows the Chain of Responsibility pattern with
support for:
- Multiple execution phases (PRE, POST, AROUND, ERROR)
- Priority-based ordering
- Tool-specific filtering
- Enable/disable functionality

Usage:
    from victor.framework.middleware_protocols import IMiddleware, MiddlewarePhase

    class MyMiddleware(IMiddleware):
        async def before_tool_call(
            self, tool_name: str, arguments: Dict[str, Any]
        ) -> MiddlewareResult:
            # Process tool call
            return MiddlewareResult()

        def get_priority(self) -> MiddlewarePriority:
            return MiddlewarePriority.HIGH
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Protocol, Set, runtime_checkable

# Re-export existing types from victor.core.vertical_types for backward compatibility
from victor.core.vertical_types import MiddlewarePriority, MiddlewareResult

# Re-export IIdempotentTool from cache protocols for convenience
from victor.protocols import IIdempotentTool


# =============================================================================
# Middleware Execution Phases
# =============================================================================


class MiddlewarePhase(Enum):
    """Middleware execution phases.

    Middleware can execute at different phases of the tool call lifecycle:
    - PRE: Before tool execution (for validation, transformation)
    - POST: After tool execution (for logging, result processing)
    - AROUND: Wraps tool execution (for timing, error handling)
    - ERROR: On error only (for error recovery, logging)

    Example:
        class LoggingMiddleware(IMiddleware):
            def get_phase(self) -> MiddlewarePhase:
                return MiddlewarePhase.AROUND
    """

    PRE = "pre"  # Before tool call
    POST = "post"  # After tool call
    AROUND = "around"  # Wraps tool call
    ERROR = "error"  # On error only


# =============================================================================
# Core Middleware Protocol
# =============================================================================


@runtime_checkable
class IMiddleware(Protocol):
    """Core middleware protocol for tool execution.

    This protocol defines the interface that all middleware must implement.
    Middleware can intercept and modify tool calls at different phases
    of execution, enabling layered processing of tool operations.

    Key Features:
    - Phase-based execution (PRE, POST, AROUND, ERROR)
    - Priority-based ordering (CRITICAL, HIGH, NORMAL, LOW, DEFERRED)
    - Tool-specific filtering (apply to specific tools or all)
    - Enable/disable functionality

    Implementation Requirements:
    - Subclasses must implement before_tool_call()
    - Subclasses may override after_tool_call() for post-processing
    - Subclasses may override get_priority() for execution ordering
    - Subclasses may override get_applicable_tools() for filtering
    - Subclasses may override get_phase() for execution timing

    Example:
        class CodeValidationMiddleware(IMiddleware):
            def __init__(self):
                self._enabled = True
                self._applicable_tools = {"write_file", "edit_file"}
                self._priority = MiddlewarePriority.HIGH
                self._phase = MiddlewarePhase.PRE

            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                if tool_name == "write_file" and "content" in arguments:
                    # Validate code syntax before writing
                    is_valid = self._validate_syntax(arguments["content"])
                    if not is_valid:
                        return MiddlewareResult(
                            proceed=False,
                            error_message="Syntax error detected"
                        )
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return self._priority

            def get_applicable_tools(self) -> Optional[Set[str]]:
                return self._applicable_tools

            def get_phase(self) -> MiddlewarePhase:
                return self._phase
    """

    @abstractmethod
    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Called before a tool is executed.

        This is the primary interception point for middleware. Use it to:
        - Validate arguments
        - Transform arguments
        - Block execution
        - Add metadata

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult indicating whether to proceed and optionally
            modified arguments or error messages
        """
        ...

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Called after a tool is executed.

        This is the post-processing interception point. Use it to:
        - Log results
        - Transform results
        - Extract metadata
        - Trigger side effects

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed to the tool
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            Modified result (or None to keep original)
        """
        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Middleware executes in priority order:
        - PRE phase: Lower values execute first
        - POST phase: Higher values execute first
        - ERROR phase: Higher values execute first

        Returns:
            Priority level for execution ordering
        """
        return MiddlewarePriority.NORMAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            Set of tool names, or None for all tools
        """
        return None

    def get_phase(self) -> MiddlewarePhase:
        """Get the execution phase for this middleware.

        Returns:
            Phase at which this middleware executes
        """
        return MiddlewarePhase.PRE


__all__ = [
    "IMiddleware",
    "MiddlewarePhase",
    "MiddlewarePriority",
    "MiddlewareResult",
    "IIdempotentTool",
]
