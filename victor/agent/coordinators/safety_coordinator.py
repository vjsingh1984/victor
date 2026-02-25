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

"""Safety coordinator for agent orchestration.

This module provides the SafetyCoordinator which extracts
safety-related responsibilities from the orchestrator.

Design Pattern: Coordinator (SRP Compliance)
- Pattern matching for dangerous operations
- Safety rule enforcement
- Warning generation
- Operation blocking

Extracted from AgentOrchestrator to improve modularity and testability
as part of the SOLID refactoring initiative (Phase 2).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SafetyAction(Enum):
    """Actions to take when safety rule is triggered."""

    ALLOW = "allow"  # Allow the operation
    WARN = "warn"  # Allow with warning
    BLOCK = "block"  # Block the operation
    REQUIRE_CONFIRMATION = "require_confirmation"  # Require user confirmation


class SafetyCategory(Enum):
    """Categories of safety rules."""

    GIT = "git"  # Git operation safety
    FILE = "file"  # File operation safety
    SHELL = "shell"  # Shell command safety
    DOCKER = "docker"  # Docker operation safety
    NETWORK = "network"  # Network operation safety
    SYSTEM = "system"  # System-level operations


@dataclass
class SafetyRule:
    """A safety rule definition.

    Attributes:
        rule_id: Unique identifier for the rule
        category: Category this rule belongs to
        pattern: Pattern to match (regex for commands, tool name for tools)
        description: Human-readable description
        action: Action to take when rule is triggered
        severity: Rule severity (1-10, higher = more severe)
        tool_names: Tool names this rule applies to
        confirmation_prompt: Prompt for user confirmation
    """

    rule_id: str
    category: SafetyCategory
    pattern: str
    description: str
    action: SafetyAction
    severity: int = 5
    tool_names: List[str] = field(default_factory=list)
    confirmation_prompt: str = ""

    def matches(self, tool_name: str, args: List[str]) -> bool:
        """Check if this rule matches the given tool call.

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool

        Returns:
            True if the rule matches, False otherwise
        """
        # Check if tool name matches
        if self.tool_names and tool_name not in self.tool_names:
            return False

        # Check if pattern matches
        if self.pattern:
            # Check tool name pattern
            if re.search(self.pattern, tool_name, re.IGNORECASE):
                return True

            # Check args pattern
            args_str = " ".join(args)
            if re.search(self.pattern, args_str, re.IGNORECASE):
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "pattern": self.pattern,
            "description": self.description,
            "action": self.action.value,
            "severity": self.severity,
            "tool_names": self.tool_names,
            "confirmation_prompt": self.confirmation_prompt,
        }


@dataclass
class SafetyCheckResult:
    """Result of a safety check.

    Attributes:
        is_safe: Whether the operation is considered safe
        action: Action to take
        matched_rules: Rules that matched
        warnings: Warning messages
        block_reason: Reason for blocking (if blocked)
        confirmation_prompt: Prompt for user confirmation (if needed)
    """

    is_safe: bool
    action: SafetyAction
    matched_rules: List[SafetyRule] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    block_reason: str = ""
    confirmation_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "action": self.action.value,
            "matched_rules": [r.to_dict() for r in self.matched_rules],
            "warnings": self.warnings,
            "block_reason": self.block_reason,
            "confirmation_prompt": self.confirmation_prompt,
        }


@dataclass
class SafetyStats:
    """Statistics about safety checks.

    Attributes:
        total_checks: Total number of safety checks
        blocked_operations: Number of blocked operations
        warned_operations: Number of operations with warnings
        confirmed_operations: Number of operations requiring confirmation
        rule_hits: Number of times each rule was hit
    """

    total_checks: int = 0
    blocked_operations: int = 0
    warned_operations: int = 0
    confirmed_operations: int = 0
    rule_hits: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_checks": self.total_checks,
            "blocked_operations": self.blocked_operations,
            "warned_operations": self.warned_operations,
            "confirmed_operations": self.confirmed_operations,
            "rule_hits": self.rule_hits.copy(),
        }


class SafetyCoordinator:
    """Coordinates safety checks for the orchestrator.

    This class extracts safety-related responsibilities from the orchestrator,
    providing a focused interface for:

    1. Pattern Matching: Match dangerous operation patterns
    2. Rule Enforcement: Apply safety rules
    3. Warning Generation: Generate warnings for risky operations
    4. Operation Blocking: Block operations that are too dangerous

    Example:
        coordinator = SafetyCoordinator()

        # Register a safety rule
        coordinator.register_rule(SafetyRule(
            rule_id="no_force_push",
            category=SafetyCategory.GIT,
            pattern=r"push.*--force",
            description="Force push to main branch",
            action=SafetyAction.BLOCK,
            severity=8,
            tool_names=["git"],
        ))

        # Check if an operation is safe
        result = coordinator.check_safety("git", ["push", "--force", "origin", "main"])
        if not result.is_safe:
            print(f"Blocked: {result.block_reason}")
    """

    # Default safety rules
    DEFAULT_RULES: List[SafetyRule] = [
        # Git safety rules
        SafetyRule(
            rule_id="git_force_push_main",
            category=SafetyCategory.GIT,
            pattern=r"push.*--force.*main|push.*--force.*master",
            description="Force push to main/master branch",
            action=SafetyAction.BLOCK,
            severity=9,
            tool_names=["git"],
            confirmation_prompt="Force pushing to main branch can rewrite history. Are you sure?",
        ),
        SafetyRule(
            rule_id="git_force_push",
            category=SafetyCategory.GIT,
            pattern=r"push.*--force",
            description="Force push",
            action=SafetyAction.REQUIRE_CONFIRMATION,
            severity=7,
            tool_names=["git"],
            confirmation_prompt="Force push can rewrite history. Continue?",
        ),
        SafetyRule(
            rule_id="git_delete_branch",
            category=SafetyCategory.GIT,
            pattern=r"branch.*-D|branch.*--delete.*--force",
            description="Force delete branch",
            action=SafetyAction.WARN,
            severity=5,
            tool_names=["git"],
        ),
        # File operation safety rules
        SafetyRule(
            rule_id="file_delete_recursive",
            category=SafetyCategory.FILE,
            pattern=r"rm.*-rf|delete.*recursive",
            description="Recursive file deletion",
            action=SafetyAction.REQUIRE_CONFIRMATION,
            severity=8,
            tool_names=["shell", "execute_bash"],
            confirmation_prompt="This will recursively delete files. Continue?",
        ),
        SafetyRule(
            rule_id="file_overwrite_system",
            category=SafetyCategory.FILE,
            pattern=r"write.*/(etc|usr|bin|sbin)/",
            description="Write to system directory",
            action=SafetyAction.BLOCK,
            severity=10,
            tool_names=["write_file", "edit_files"],
        ),
        # Shell command safety rules
        SafetyRule(
            rule_id="shell_format_disk",
            category=SafetyCategory.SHELL,
            pattern=r"mkfs|format|fdisk",
            description="Disk formatting operation",
            action=SafetyAction.BLOCK,
            severity=10,
            tool_names=["shell", "execute_bash"],
        ),
        SafetyRule(
            rule_id="shell_remove_user",
            category=SafetyCategory.SHELL,
            pattern=r"userdel.*-r|user.*remove",
            description="Remove user account",
            action=SafetyAction.BLOCK,
            severity=9,
            tool_names=["shell", "execute_bash"],
        ),
        # Docker safety rules
        SafetyRule(
            rule_id="docker_rm_container",
            category=SafetyCategory.DOCKER,
            pattern=r"rm.*-f|container.*rm.*--force",
            description="Force remove Docker container",
            action=SafetyAction.WARN,
            severity=6,
            tool_names=["docker"],
        ),
        SafetyRule(
            rule_id="docker_rmi_image",
            category=SafetyCategory.DOCKER,
            pattern=r"rmi|image.*rm",
            description="Remove Docker image",
            action=SafetyAction.WARN,
            severity=5,
            tool_names=["docker"],
        ),
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        enable_default_rules: bool = True,
        max_severity_to_allow: int = 10,
    ):
        """Initialize the safety coordinator.

        Args:
            strict_mode: If True, treat warnings as blocks
            enable_default_rules: Whether to load default safety rules
            max_severity_to_allow: Maximum severity to allow without confirmation
        """
        self._strict_mode = strict_mode
        self._max_severity_to_allow = max_severity_to_allow
        self._rules: List[SafetyRule] = []
        self._rule_index: Dict[str, SafetyRule] = {}

        # Statistics
        self._stats: SafetyStats = SafetyStats()

        # Load default rules
        if enable_default_rules:
            for rule in self.DEFAULT_RULES:
                self.register_rule(rule)

        logger.debug(
            f"SafetyCoordinator initialized with {len(self._rules)} rules, "
            f"strict_mode={strict_mode}"
        )

    # ========================================================================
    # Rule Management
    # ========================================================================

    def register_rule(self, rule: SafetyRule) -> None:
        """Register a safety rule.

        Args:
            rule: Safety rule to register
        """
        self._rules.append(rule)
        self._rule_index[rule.rule_id] = rule
        logger.debug(f"Registered safety rule: {rule.rule_id}")

    def unregister_rule(self, rule_id: str) -> bool:
        """Unregister a safety rule.

        Args:
            rule_id: ID of the rule to unregister

        Returns:
            True if rule was removed, False if not found
        """
        if rule_id not in self._rule_index:
            return False

        rule = self._rule_index[rule_id]
        self._rules.remove(rule)
        del self._rule_index[rule_id]

        logger.debug(f"Unregistered safety rule: {rule_id}")
        return True

    def get_rule(self, rule_id: str) -> Optional[SafetyRule]:
        """Get a safety rule by ID.

        Args:
            rule_id: ID of the rule

        Returns:
            Safety rule or None if not found
        """
        return self._rule_index.get(rule_id)

    def list_rules(
        self, category: Optional[SafetyCategory] = None
    ) -> List[SafetyRule]:
        """List registered safety rules.

        Args:
            category: Optional category filter

        Returns:
            List of safety rules
        """
        if category:
            return [r for r in self._rules if r.category == category]
        return self._rules.copy()

    def clear_rules(self) -> None:
        """Clear all safety rules."""
        self._rules.clear()
        self._rule_index.clear()
        logger.info("Cleared all safety rules")

    # ========================================================================
    # Safety Checking
    # ========================================================================

    def check_safety(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyCheckResult:
        """Check if a tool operation is safe.

        Args:
            tool_name: Name of the tool being called
            args: Arguments to the tool
            context: Optional context for the check

        Returns:
            SafetyCheckResult with safety assessment
        """
        self._stats.total_checks += 1

        matched_rules = []
        warnings = []
        block_reason = ""
        confirmation_prompt = ""

        # Check all rules
        for rule in self._rules:
            if rule.matches(tool_name, args):
                matched_rules.append(rule)

                # Update stats
                self._stats.rule_hits[rule.rule_id] = (
                    self._stats.rule_hits.get(rule.rule_id, 0) + 1
                )

                # Collect warnings
                if rule.action == SafetyAction.WARN:
                    warnings.append(f"{rule.description} (severity: {rule.severity})")
                    self._stats.warned_operations += 1

        if not matched_rules:
            # No rules matched, operation is safe
            return SafetyCheckResult(is_safe=True, action=SafetyAction.ALLOW)

        # Sort rules by severity (highest first)
        matched_rules.sort(key=lambda r: r.severity, reverse=True)

        # Get the most severe action
        most_severe_rule = matched_rules[0]
        action = most_severe_rule.action

        # In strict mode, promote warnings to blocks
        if self._strict_mode and action == SafetyAction.WARN:
            action = SafetyAction.BLOCK

        # Check if we should block based on severity
        if action != SafetyAction.BLOCK and most_severe_rule.severity > self._max_severity_to_allow:
            action = SafetyAction.BLOCK

        # Generate result
        if action == SafetyAction.BLOCK:
            self._stats.blocked_operations += 1
            block_reason = most_severe_rule.description
            return SafetyCheckResult(
                is_safe=False,
                action=action,
                matched_rules=matched_rules,
                block_reason=block_reason,
            )

        elif action == SafetyAction.REQUIRE_CONFIRMATION:
            self._stats.confirmed_operations += 1
            confirmation_prompt = most_severe_rule.confirmation_prompt
            return SafetyCheckResult(
                is_safe=False,
                action=action,
                matched_rules=matched_rules,
                warnings=warnings,
                confirmation_prompt=confirmation_prompt,
            )

        else:  # WARN or ALLOW
            return SafetyCheckResult(
                is_safe=True,
                action=action,
                matched_rules=matched_rules,
                warnings=warnings,
            )

    def is_operation_safe(
        self,
        tool_name: str,
        args: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Quick check if an operation is safe.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            context: Optional context

        Returns:
            True if operation is safe, False otherwise
        """
        result = self.check_safety(tool_name, args, context)
        return result.is_safe

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> SafetyStats:
        """Get safety statistics.

        Returns:
            SafetyStats object
        """
        return self._stats

    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary.

        Returns:
            Dictionary of statistics
        """
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = SafetyStats()
        logger.info("Safety statistics reset")

    # ========================================================================
    # Configuration
    # ========================================================================

    def set_strict_mode(self, strict: bool) -> None:
        """Set strict mode.

        Args:
            strict: Whether to enable strict mode
        """
        self._strict_mode = strict
        logger.info(f"Strict mode set to {strict}")

    def set_max_severity_to_allow(self, severity: int) -> None:
        """Set maximum severity to allow without confirmation.

        Args:
            severity: Maximum severity level
        """
        self._max_severity_to_allow = severity
        logger.info(f"Max severity to allow set to {severity}")

    # ========================================================================
    # Observability
    # ========================================================================

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data for dashboard integration.

        Returns:
            Dictionary with observability data
        """
        return {
            "source_id": f"coordinator:safety:{id(self)}",
            "source_type": "coordinator",
            "coordinator_type": "safety",
            "stats": self._stats.to_dict(),
            "config": {
                "strict_mode": self._strict_mode,
                "max_severity_to_allow": self._max_severity_to_allow,
                "total_rules": len(self._rules),
            },
        }


__all__ = [
    "SafetyCoordinator",
    "SafetyRule",
    "SafetyCheckResult",
    "SafetyStats",
    "SafetyAction",
    "SafetyCategory",
]
