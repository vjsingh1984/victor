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
"""

from __future__ import annotations

from typing import Dict, List

from victor.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


# Git-specific dangerous patterns
GIT_DANGEROUS_PATTERNS: List[SafetyPattern] = [
    # HIGH risk git operations
    SafetyPattern(
        pattern=r"git\s+reset\s+--hard",
        description="Discard all uncommitted changes",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+clean\s+-fd",
        description="Delete untracked files",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+push\s+.*--force",
        description="Force push (may lose commits)",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+push\s+-f",
        description="Force push (may lose commits)",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+rebase\s+.*--force",
        description="Force rebase",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+branch\s+-D",
        description="Force delete branch",
        risk_level="HIGH",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+push\s+.*:\s*\S+",
        description="Delete remote branch",
        risk_level="HIGH",
        category="git",
    ),
    # MEDIUM risk git operations
    SafetyPattern(
        pattern=r"git\s+checkout\s+--\s+",
        description="Discard changes to file",
        risk_level="MEDIUM",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+stash\s+drop",
        description="Discard stashed changes",
        risk_level="MEDIUM",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+stash\s+clear",
        description="Clear all stashes",
        risk_level="MEDIUM",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+rebase\s+-i",
        description="Interactive rebase (rewrites history)",
        risk_level="MEDIUM",
        category="git",
    ),
    SafetyPattern(
        pattern=r"git\s+commit\s+--amend",
        description="Amend last commit",
        risk_level="MEDIUM",
        category="git",
    ),
]


# Refactoring-specific patterns
REFACTORING_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"refactor.*--all",
        description="Refactor across entire codebase",
        risk_level="HIGH",
        category="refactoring",
    ),
    SafetyPattern(
        pattern=r"rename.*--recursive",
        description="Recursive symbol rename",
        risk_level="MEDIUM",
        category="refactoring",
    ),
]


# Package manager patterns
PACKAGE_MANAGER_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"pip\s+uninstall",
        description="Uninstall Python package",
        risk_level="MEDIUM",
        category="package",
    ),
    SafetyPattern(
        pattern=r"npm\s+uninstall",
        description="Uninstall npm package",
        risk_level="MEDIUM",
        category="package",
    ),
    SafetyPattern(
        pattern=r"yarn\s+remove",
        description="Remove yarn package",
        risk_level="MEDIUM",
        category="package",
    ),
    SafetyPattern(
        pattern=r"cargo\s+remove",
        description="Remove Rust crate",
        risk_level="MEDIUM",
        category="package",
    ),
    SafetyPattern(
        pattern=r"pip\s+install\s+--force-reinstall",
        description="Force reinstall package",
        risk_level="MEDIUM",
        category="package",
    ),
]


# Build/deployment patterns
BUILD_DEPLOY_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"docker\s+system\s+prune",
        description="Remove unused Docker data",
        risk_level="MEDIUM",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+image\s+prune\s+-a",
        description="Remove all unused images",
        risk_level="MEDIUM",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"kubectl\s+delete",
        description="Delete Kubernetes resources",
        risk_level="HIGH",
        category="kubernetes",
    ),
    SafetyPattern(
        pattern=r"terraform\s+destroy",
        description="Destroy infrastructure",
        risk_level="CRITICAL",
        category="infrastructure",
    ),
]


# File operation patterns specific to coding
CODING_FILE_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"\.env$",
        description="Environment file with secrets",
        risk_level="HIGH",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"\.pem$|\.key$",
        description="Private key file",
        risk_level="CRITICAL",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"credentials\.json",
        description="Credentials file",
        risk_level="HIGH",
        category="secrets",
    ),
    SafetyPattern(
        pattern=r"\.git/",
        description="Git internal directory",
        risk_level="HIGH",
        category="git",
    ),
]


class CodingSafetyExtension(SafetyExtensionProtocol):
    """Safety extension for coding vertical.

    Provides coding-specific dangerous operation patterns including
    git operations, refactoring, and package management.
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

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get coding-specific bash command patterns.

        Returns:
            List of safety patterns for dangerous bash commands
        """
        patterns = []

        if self._include_git:
            patterns.extend(GIT_DANGEROUS_PATTERNS)

        if self._include_refactoring:
            patterns.extend(REFACTORING_PATTERNS)

        if self._include_packages:
            patterns.extend(PACKAGE_MANAGER_PATTERNS)

        if self._include_build:
            patterns.extend(BUILD_DEPLOY_PATTERNS)

        return patterns

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get coding-specific file operation patterns.

        Returns:
            List of safety patterns for file operations
        """
        return CODING_FILE_PATTERNS.copy()

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


__all__ = [
    "CodingSafetyExtension",
    "GIT_DANGEROUS_PATTERNS",
    "REFACTORING_PATTERNS",
    "PACKAGE_MANAGER_PATTERNS",
    "BUILD_DEPLOY_PATTERNS",
    "CODING_FILE_PATTERNS",
]
