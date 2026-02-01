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

"""Security analysis middleware.

This module provides middleware for security analysis operations,
including secret detection and security validation.
"""

from __future__ import annotations

import logging
from typing import Any

from victor.core.verticals.protocols import MiddlewareProtocol
from victor.core.vertical_types import MiddlewareResult

logger = logging.getLogger(__name__)


class SecurityAnalysisMiddleware(MiddlewareProtocol):
    """Middleware for security analysis operations.

    Provides:
    - Secret detection in outputs
    - Security validation
    - Audit logging integration
    """

    def __init__(self, enable_secret_detection: bool = True):
        """Initialize the middleware.

        Args:
            enable_secret_detection: Whether to scan for secrets in outputs
        """
        self._enable_secret_detection = enable_secret_detection

    def process_input(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Process input before tool execution.

        Args:
            input_data: Input data to process

        Returns:
            Processed input data
        """
        # For security analysis, we might want to track what's being analyzed
        if "file_path" in input_data:
            logger.debug(f"Security analysis on: {input_data['file_path']}")
        return input_data

    def process_output(self, output_data: dict[str, Any]) -> dict[str, Any]:
        """Process output after tool execution.

        Args:
            output_data: Output data to process

        Returns:
            Processed output data, potentially with secrets masked
        """
        if not self._enable_secret_detection:
            return output_data

        # Check for potential secrets in output
        if "content" in output_data:
            content = output_data["content"]
            if self._contains_potential_secrets(content):
                logger.warning("Potential secrets detected in output")
                output_data["_security_warning"] = "Potential secrets detected"

        return output_data

    def _contains_potential_secrets(self, content: str) -> bool:
        """Check if content might contain secrets.

        Args:
            content: Content to check

        Returns:
            True if potential secrets detected
        """
        import re

        # Simple patterns for common secrets
        secret_patterns = [
            r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?[\w-]{20,}",
            r"(?i)(secret|password|token)\s*[:=]\s*['\"]?[\w-]{8,}",
            r"(?i)bearer\s+[\w-]{20,}",
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub PAT
            r"sk-[a-zA-Z0-9]{48}",  # OpenAI key
        ]

        for pattern in secret_patterns:
            if re.search(pattern, content):
                return True
        return False

    async def before_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> "MiddlewareResult":
        """Called before a tool is executed.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments

        Returns:
            Middleware result with proceed/abort decision
        """
        from victor.core.vertical_types import MiddlewareResult

        # Basic security checks for dangerous tools
        dangerous_tools = {"write_file", "delete_file", "execute", "shell"}
        if tool_name in dangerous_tools:
            if "content" in arguments and self._contains_potential_secrets(arguments["content"]):
                logger.warning(f"Blocked potentially dangerous tool call: {tool_name}")
                return MiddlewareResult(
                    proceed=False,
                    error_message="Potential security risk detected in tool arguments",
                )

        return MiddlewareResult()

    @property
    def name(self) -> str:
        """Get middleware name."""
        return "security_analysis"

    @property
    def priority(self) -> int:
        """Get middleware priority (higher = runs first)."""
        return 100  # High priority for security checks


__all__ = ["SecurityAnalysisMiddleware"]
