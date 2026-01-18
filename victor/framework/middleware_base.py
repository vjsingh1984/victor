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

"""Generic Middleware Library - Base Classes.

This module provides base classes for implementing middleware.
The BaseMiddleware class implements the IMiddleware protocol and
provides common functionality to reduce boilerplate in subclasses.

Features:
- Enable/disable functionality
- Configurable priority, phase, and tool filtering
- Built-in logging support
- Sensible default implementations

Usage:
    from victor.framework.middleware_base import BaseMiddleware

    class MyMiddleware(BaseMiddleware):
        async def before_tool_call(
            self, tool_name: str, arguments: Dict[str, Any]
        ) -> MiddlewareResult:
            self._logger.info(f"Processing {tool_name}")
            return MiddlewareResult()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set

from victor.framework.middleware_protocols import (
    IMiddleware,
    MiddlewarePhase,
    MiddlewarePriority,
    MiddlewareResult,
)


# =============================================================================
# Base Middleware Class
# =============================================================================


class BaseMiddleware(ABC, IMiddleware):
    """Abstract base class for middleware implementations.

    This class provides a foundation for creating middleware with:
    - Enable/disable functionality
    - Configurable priority, phase, and tool filtering
    - Built-in logging support
    - Sensible default implementations

    Subclasses must implement:
    - before_tool_call(): Process tool calls before execution

    Subclasses may override:
    - after_tool_call(): Process tool results after execution
    - get_priority(): Set custom priority (default: NORMAL)
    - get_applicable_tools(): Filter tools (default: all tools)
    - get_phase(): Set execution phase (default: PRE)

    Example:
        class CodeValidationMiddleware(BaseMiddleware):
            def __init__(self):
                super().__init__(
                    enabled=True,
                    applicable_tools={"write_file", "edit_file"},
                    priority=MiddlewarePriority.HIGH,
                    phase=MiddlewarePhase.PRE,
                    logger_name="victor.middleware.code_validation"
                )

            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                self._logger.info(f"Validating {tool_name}")

                if tool_name == "write_file" and "content" in arguments:
                    if not self._is_valid_code(arguments["content"]):
                        return MiddlewareResult(
                            proceed=False,
                            error_message="Invalid code syntax"
                        )

                return MiddlewareResult()

            def _is_valid_code(self, code: str) -> bool:
                # Custom validation logic
                return True
    """

    def __init__(
        self,
        enabled: bool = True,
        applicable_tools: Optional[Set[str]] = None,
        priority: MiddlewarePriority = MiddlewarePriority.NORMAL,
        phase: MiddlewarePhase = MiddlewarePhase.PRE,
        logger_name: Optional[str] = None,
    ):
        """Initialize base middleware.

        Args:
            enabled: Whether middleware is initially enabled
            applicable_tools: Set of tool names this applies to (None = all tools)
            priority: Execution priority for ordering
            phase: Execution phase (PRE, POST, AROUND, ERROR)
            logger_name: Name for logger (None = use module name)
        """
        self._enabled = enabled
        self._applicable_tools = applicable_tools
        self._priority = priority
        self._phase = phase
        self._logger = (
            logging.getLogger(logger_name) if logger_name else logging.getLogger(__name__)
        )

    @property
    def enabled(self) -> bool:
        """Check if middleware is enabled.

        Returns:
            True if middleware is enabled, False otherwise
        """
        return self._enabled

    def enable(self) -> None:
        """Enable the middleware."""
        self._enabled = True
        self._logger.debug(f"{self.__class__.__name__} enabled")

    def disable(self) -> None:
        """Disable the middleware."""
        self._enabled = False
        self._logger.debug(f"{self.__class__.__name__} disabled")

    @abstractmethod
    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Process tool call before execution.

        Subclasses must implement this method to define their
        pre-processing logic.

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
        """Process tool result after execution.

        Default implementation does nothing. Subclasses can override
        to provide post-processing logic.

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
        """Get middleware priority.

        Returns:
            Priority level for execution ordering
        """
        return self._priority

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get applicable tools.

        Returns:
            Set of tool names, or None for all tools
        """
        return self._applicable_tools

    def get_phase(self) -> MiddlewarePhase:
        """Get execution phase.

        Returns:
            Phase at which this middleware executes
        """
        return self._phase

    def applies_to_tool(self, tool_name: str) -> bool:
        """Check if middleware applies to a specific tool.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if middleware applies to the tool, False otherwise
        """
        if self._applicable_tools is None:
            return True
        return tool_name in self._applicable_tools

    def __repr__(self) -> str:
        """String representation of middleware.

        Returns:
            String representation showing class name, enabled status,
            priority, phase, and applicable tools
        """
        return (
            f"{self.__class__.__name__}("
            f"enabled={self._enabled}, "
            f"priority={self._priority.name}, "
            f"phase={self._phase.value}, "
            f"tools={self._applicable_tools})"
        )


__all__ = [
    "BaseMiddleware",
]
