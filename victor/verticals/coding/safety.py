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

"""Coding-specific safety patterns.

This module defines dangerous operation patterns specific to
software development, particularly git operations and code refactoring.

This module now delegates to the core safety infrastructure at
victor.safety.code_patterns for pattern scanning, while maintaining
backward compatibility for existing imports.
"""

from __future__ import annotations

from typing import Dict, List

from victor.safety.code_patterns import (
    CodePatternScanner,
    GIT_PATTERNS,
    REFACTORING_PATTERNS,
    PACKAGE_MANAGER_PATTERNS,
    BUILD_DEPLOY_PATTERNS,
    SENSITIVE_FILE_PATTERNS,
)
from victor.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


# Re-export core patterns with legacy names for backward compatibility
GIT_DANGEROUS_PATTERNS: List[SafetyPattern] = GIT_PATTERNS
CODING_FILE_PATTERNS: List[SafetyPattern] = SENSITIVE_FILE_PATTERNS


class CodingSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for coding vertical.

    Provides coding-specific dangerous operation patterns including
    git operations, refactoring, and package management.

    This class delegates to the core CodePatternScanner for pattern
    matching while providing the SafetyExtensionProtocol interface.
    """

    def __init__(
        self,
        include_git: bool = True,
        include_refactoring: bool = True,
        include_packages: bool = True,
        include_build: bool = True,
    ):
        """Initialize the safety extension.

        Args:
            include_git: Include git-specific patterns
            include_refactoring: Include refactoring patterns
            include_packages: Include package manager patterns
            include_build: Include build/deployment patterns
        """
        self._include_git = include_git
        self._include_refactoring = include_refactoring
        self._include_packages = include_packages
        self._include_build = include_build

        # Create a CodePatternScanner with matching configuration
        self._scanner = CodePatternScanner(
            include_git=include_git,
            include_refactoring=include_refactoring,
            include_packages=include_packages,
            include_build=include_build,
            include_files=True,
        )

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get coding-specific bash command patterns.

        Returns:
            List of safety patterns for dangerous bash commands
        """
        return self._scanner.all_patterns

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get coding-specific file operation patterns.

        Returns:
            List of safety patterns for file operations
        """
        return self._scanner.file_patterns

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Dict mapping tool names to restricted argument patterns
        """
        return {
            "write_file": [
                r"\.env$",  # Don't overwrite .env files
                r"\.git/",  # Don't write to .git directory
            ],
            "execute_bash": [
                r"rm\s+-rf\s+\.",  # Don't delete current directory recursively
            ],
        }

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier
        """
        return "coding"

    def scan_command(self, command: str) -> List[SafetyPattern]:
        """Scan a command for dangerous patterns.

        Args:
            command: The command to scan

        Returns:
            List of matched safety patterns
        """
        return self._scanner.scan_command(command).matches

    def is_sensitive_file(self, path: str) -> bool:
        """Check if a file path is sensitive.

        Args:
            path: File path to check

        Returns:
            True if the file is sensitive
        """
        return self._scanner.is_sensitive_file(path)


__all__ = [
    "CodingSafetyExtension",
    # Re-exported from core for backward compatibility
    "GIT_DANGEROUS_PATTERNS",
    "REFACTORING_PATTERNS",
    "PACKAGE_MANAGER_PATTERNS",
    "BUILD_DEPLOY_PATTERNS",
    "CODING_FILE_PATTERNS",
]
