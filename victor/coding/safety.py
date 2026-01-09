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
victor.security.safety.code_patterns for pattern scanning, while maintaining
backward compatibility for existing imports.
"""

from __future__ import annotations

from typing import Dict, List

from victor.security.safety.code_patterns import (
    CodePatternScanner,
    GIT_PATTERNS,
    REFACTORING_PATTERNS,
    PACKAGE_MANAGER_PATTERNS,
    BUILD_DEPLOY_PATTERNS,
    SENSITIVE_FILE_PATTERNS,
)
from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


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
    # New framework-based safety rules
    "create_git_safety_rules",
    "create_file_safety_rules",
    "create_test_safety_rules",
    "create_all_coding_safety_rules",
]


# =============================================================================
# Framework-Based Safety Rules (New)
# =============================================================================

"""Framework-based safety rules for coding operations.

This section provides factory functions that register safety rules
with the framework-level SafetyEnforcer. This is the new recommended
approach for safety enforcement in coding workflows.

Example:
    from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
    from victor.coding.safety import create_all_coding_safety_rules

    enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
    create_all_coding_safety_rules(enforcer)

    # Check operations
    allowed, reason = enforcer.check_operation("git push --force origin main")
    if not allowed:
        print(f"Blocked: {reason}")
"""

from victor.framework.config import SafetyEnforcer, SafetyRule, SafetyLevel


def create_git_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_force_push: bool = True,
    block_main_push: bool = True,
    require_tests_before_commit: bool = False,
    protected_branches: list[str] | None = None,
) -> None:
    """Register git-specific safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_force_push: Block force push to protected branches
        block_main_push: Block direct push to main/master
        require_tests_before_commit: Require tests to pass before commit
        protected_branches: List of protected branch names (default: ["main", "master", "develop"])

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_git_safety_rules(
            enforcer,
            block_force_push=True,
            block_main_push=True,
            protected_branches=["main", "master", "production"]
        )
    """
    protected = protected_branches or ["main", "master", "develop"]

    if block_force_push:
        enforcer.add_rule(
            SafetyRule(
                name="git_block_force_push",
                description="Block force push to protected branches",
                check_fn=lambda op: "git push" in op and "--force" in op
                and any(branch in op for branch in protected),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

    if block_main_push:
        enforcer.add_rule(
            SafetyRule(
                name="git_block_main_push",
                description="Block direct push to main/master branches (use PRs)",
                check_fn=lambda op: "git push" in op
                and any(
                    f" origin {branch}" in op or f" upstream {branch}" in op
                    for branch in protected[:2]
                ),  # main, master only
                level=SafetyLevel.MEDIUM,
                allow_override=True,
            )
        )

    if require_tests_before_commit:
        enforcer.add_rule(
            SafetyRule(
                name="git_require_tests_before_commit",
                description="Require tests to pass before committing",
                check_fn=lambda op: "git commit" in op and "--no-verify" not in op,
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_file_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_destructive_commands: bool = True,
    protected_patterns: list[str] | None = None,
) -> None:
    """Register file operation safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_destructive_commands: Block commands like rm -rf, git clean -fdx
        protected_patterns: Glob patterns for protected files (default: ["**/.env", "**/secrets*"])

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_file_safety_rules(
            enforcer,
            block_destructive_commands=True,
            protected_patterns=["**/.env", "**/secrets*", "**/config/prod/*"]
        )
    """
    protected = protected_patterns or [".env", "secrets", "credential", "password"]

    if block_destructive_commands:
        enforcer.add_rule(
            SafetyRule(
                name="file_block_destructive",
                description="Block destructive file commands (rm -rf, git clean -fdx)",
                check_fn=lambda op: any(
                    cmd in op
                    for cmd in [
                        "rm -rf /",
                        "rm -rf /*",
                        "git clean -fdx",
                        "del /q /s",
                    ]
                ),
                level=SafetyLevel.HIGH,
                allow_override=False,
            )
        )

        enforcer.add_rule(
            SafetyRule(
                name="file_block_protected_modification",
                description="Block modification of protected files (.env, secrets, credentials)",
                check_fn=lambda op: any(
                    pattern in op.lower() for pattern in [p.lower() for p in protected]
                )
                and any(cmd in op for cmd in ["write(", "write_file(", "edit("]),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )


def create_test_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    block_destructive_tests: bool = True,
    require_test_isolation: bool = False,
) -> None:
    """Register test execution safety rules.

    Args:
        enforcer: SafetyEnforcer to register rules with
        block_destructive_tests: Block tests that modify real data
        require_test_isolation: Require tests to use mock/sandbox

    Example:
        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.MEDIUM))
        create_test_safety_rules(
            enforcer,
            block_destructive_tests=True,
            require_test_isolation=True
        )
    """
    if block_destructive_tests:
        enforcer.add_rule(
            SafetyRule(
                name="test_block_destructive",
                description="Block tests that modify production data or databases",
                check_fn=lambda op: "pytest" in op or "test" in op
                and any(db in op for db in ["prod", "production", "--database"]),
                level=SafetyLevel.HIGH,
                allow_override=True,
            )
        )

    if require_test_isolation:
        enforcer.add_rule(
            SafetyRule(
                name="test_require_isolation",
                description="Warn if tests don't use mocks or sandbox",
                check_fn=lambda op: ("pytest" in op or "unittest" in op)
                and "--mock" not in op
                and "mock" not in op.lower(),
                level=SafetyLevel.LOW,  # Warn only
                allow_override=True,
            )
        )


def create_all_coding_safety_rules(
    enforcer: SafetyEnforcer,
    *,
    git_protected_branches: list[str] | None = None,
    file_protected_patterns: list[str] | None = None,
) -> None:
    """Register all coding safety rules at once.

    This is a convenience function that registers all coding-specific
    safety rules with appropriate defaults.

    Args:
        enforcer: SafetyEnforcer to register rules with
        git_protected_branches: Protected git branches (default: ["main", "master", "develop"])
        file_protected_patterns: Protected file patterns (default: [".env", "secrets*"])

    Example:
        from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_all_coding_safety_rules(enforcer)

        # Now all operations are checked
        allowed, reason = enforcer.check_operation("git push --force origin main")
    """
    create_git_safety_rules(enforcer, protected_branches=git_protected_branches)
    create_file_safety_rules(enforcer, protected_patterns=file_protected_patterns)
    create_test_safety_rules(enforcer)
