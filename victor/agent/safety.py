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

"""Safety module for dangerous operation detection and confirmation.

This module provides:
- Detection of dangerous bash commands
- Detection of destructive file operations
- Risk level categorization
- Confirmation request structures
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable
import re
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level categorization for operations."""

    SAFE = "safe"  # No risk - read operations
    LOW = "low"  # Minor changes - single file edits
    MEDIUM = "medium"  # Moderate changes - multi-file operations
    HIGH = "high"  # Significant changes - deletions, overwrites
    CRITICAL = "critical"  # System-level destructive operations


class ApprovalMode(Enum):
    """Mode for write operation approval.

    Controls when user confirmation is required for file modifications.
    """

    OFF = "off"  # No approval required - auto-approve all operations
    RISKY_ONLY = "risky_only"  # Only require approval for HIGH/CRITICAL risk
    ALL_WRITES = "all_writes"  # Require approval for ALL write operations


# Tools that perform write/modify operations and should be gated
# This is the single source of truth for write tool identification
# IMPORTANT: Only include actual @tool decorated functions from victor/tools/
WRITE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # Direct file modifications
        "write_file",  # filesystem.py - writes/overwrites files
        "edit_files",  # file_editor_tool.py - edits files with transactions
        # Patch/diff application
        "apply_patch",  # patch_tool.py - applies unified diffs
        # Bash execution (can run any command including destructive ones)
        "execute_bash",  # bash.py - executes shell commands
        # Git write operations
        "git",  # git_tool.py - commit, push, reset, etc. (read ops are fine)
        # Refactoring (modifies files in place)
        "refactor_rename_symbol",  # refactor_tool.py
        "refactor_extract_function",  # refactor_tool.py
        "refactor_inline_variable",  # refactor_tool.py
        "refactor_organize_imports",  # refactor_tool.py
        "rename_symbol",  # code_intelligence_tool.py
        # Scaffolding (creates new files/directories)
        "scaffold",  # scaffold_tool.py
        # Batch operations (may include writes)
        "batch",  # batch_processor_tool.py
    }
)


# Numeric ordering for risk level comparisons
_RISK_ORDER = {
    RiskLevel.SAFE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


@dataclass
class ConfirmationRequest:
    """Request for user confirmation before executing a dangerous operation."""

    tool_name: str
    risk_level: RiskLevel
    description: str
    details: List[str]
    arguments: Dict[str, Any]

    def format_message(self) -> str:
        """Format confirmation request as a user-friendly message."""
        icon = {
            RiskLevel.SAFE: "✅",
            RiskLevel.LOW: "📝",
            RiskLevel.MEDIUM: "⚠️",
            RiskLevel.HIGH: "🔴",
            RiskLevel.CRITICAL: "⛔",
        }.get(self.risk_level, "❓")

        lines = [
            f"{icon} **{self.risk_level.value.upper()} RISK OPERATION**",
            f"Tool: {self.tool_name}",
            f"Action: {self.description}",
        ]

        if self.details:
            lines.append("Details:")
            for detail in self.details:
                lines.append(f"  - {detail}")

        lines.append("\nProceed with this operation?")
        return "\n".join(lines)


# Confirmation callback type: async function that takes ConfirmationRequest
# and returns True if operation should proceed, False to cancel
ConfirmationCallback = Callable[[ConfirmationRequest], Awaitable[bool]]


class SafetyChecker:
    """Checks operations for dangerous patterns and requests confirmation.

    This class detects potentially dangerous operations and categorizes them
    by risk level. The CLI or UI layer can register a confirmation callback
    to prompt the user before execution.
    """

    # Bash patterns by risk level
    BASH_CRITICAL_PATTERNS = [
        # System destruction
        (r"rm\s+-rf\s+/(?:$|\s)", "Delete entire filesystem"),
        (r"rm\s+-rf\s+/\*", "Delete all root-level directories"),
        (r"dd\s+if=.*\s+of=/dev/sd", "Write directly to disk device"),
        (r"mkfs\.", "Format filesystem"),
        (r":()\{\s*:\|:&\s*\};:", "Fork bomb"),
        (r">\s*/dev/sd[a-z]", "Overwrite disk device"),
        (r"chmod\s+-R\s+777\s+/", "Set world-writable permissions on root"),
    ]

    BASH_HIGH_PATTERNS = [
        # Destructive file operations
        (r"rm\s+-rf\s+", "Recursively delete files/directories"),
        (r"rm\s+-r\s+", "Recursively delete files/directories"),
        (r"rm\s+.*\*", "Delete files with wildcard"),
        (r"rmdir\s+", "Remove directory"),
        # Git destructive operations
        (r"git\s+reset\s+--hard", "Discard all uncommitted changes"),
        (r"git\s+clean\s+-fd", "Delete untracked files"),
        (r"git\s+push\s+.*--force", "Force push (may lose commits)"),
        (r"git\s+push\s+-f", "Force push (may lose commits)"),
        (r"git\s+rebase\s+.*--force", "Force rebase"),
        (r"git\s+branch\s+-D", "Force delete branch"),
        # System changes
        (r"sudo\s+", "Execute with elevated privileges"),
        (r"chmod\s+-R", "Recursively change permissions"),
        (r"chown\s+-R", "Recursively change ownership"),
    ]

    BASH_MEDIUM_PATTERNS = [
        (r"rm\s+", "Delete file(s)"),
        (r"mv\s+.*\s+/dev/null", "Discard file to /dev/null"),
        (r"truncate\s+", "Truncate file"),
        (r">\s+\S+", "Overwrite file with redirection"),
        (r"git\s+checkout\s+--\s+", "Discard changes to file"),
        (r"git\s+stash\s+drop", "Discard stashed changes"),
        (r"pip\s+uninstall", "Uninstall Python package"),
        (r"npm\s+uninstall", "Uninstall npm package"),
    ]

    # File operation patterns
    DANGEROUS_FILE_EXTENSIONS = {
        ".env",
        ".pem",
        ".key",
        ".crt",
        ".p12",
        ".pfx",  # Secrets
        ".db",
        ".sqlite",
        ".sqlite3",  # Databases
        "/etc/passwd",
        "/etc/shadow",  # System files
    }

    def __init__(
        self,
        confirmation_callback: Optional[ConfirmationCallback] = None,
        auto_confirm_low_risk: bool = True,
        require_confirmation_threshold: RiskLevel = RiskLevel.HIGH,
        approval_mode: ApprovalMode = ApprovalMode.RISKY_ONLY,
    ):
        """Initialize safety checker.

        Args:
            confirmation_callback: Async callback for user confirmation
            auto_confirm_low_risk: Auto-approve LOW risk operations
            require_confirmation_threshold: Minimum risk level requiring confirmation
            approval_mode: When to require approval for write operations
                - OFF: Never require approval (dangerous, use for testing only)
                - RISKY_ONLY: Only require for HIGH/CRITICAL risk (default)
                - ALL_WRITES: Require for ALL file modifications (recommended for task mode)
        """
        self.confirmation_callback = confirmation_callback
        self.auto_confirm_low_risk = auto_confirm_low_risk
        self.require_confirmation_threshold = require_confirmation_threshold
        self.approval_mode = approval_mode

        # Compile regex patterns for efficiency
        self._critical_patterns = [
            (re.compile(p, re.IGNORECASE), desc) for p, desc in self.BASH_CRITICAL_PATTERNS
        ]
        self._high_patterns = [
            (re.compile(p, re.IGNORECASE), desc) for p, desc in self.BASH_HIGH_PATTERNS
        ]
        self._medium_patterns = [
            (re.compile(p, re.IGNORECASE), desc) for p, desc in self.BASH_MEDIUM_PATTERNS
        ]

    def check_bash_command(self, command: str) -> tuple[RiskLevel, List[str]]:
        """Check a bash command for dangerous patterns.

        Args:
            command: Bash command to check

        Returns:
            Tuple of (risk_level, list of matched pattern descriptions)
        """
        matched: List[str] = []
        max_risk = RiskLevel.SAFE

        # Check critical patterns first
        for pattern, desc in self._critical_patterns:
            if pattern.search(command):
                matched.append(desc)
                max_risk = RiskLevel.CRITICAL

        if max_risk == RiskLevel.CRITICAL:
            return max_risk, matched

        # Check high risk patterns
        for pattern, desc in self._high_patterns:
            if pattern.search(command):
                matched.append(desc)
                if _RISK_ORDER[max_risk] < _RISK_ORDER[RiskLevel.HIGH]:
                    max_risk = RiskLevel.HIGH

        if max_risk == RiskLevel.HIGH:
            return max_risk, matched

        # Check medium risk patterns
        for pattern, desc in self._medium_patterns:
            if pattern.search(command):
                matched.append(desc)
                if _RISK_ORDER[max_risk] < _RISK_ORDER[RiskLevel.MEDIUM]:
                    max_risk = RiskLevel.MEDIUM

        return max_risk, matched

    def check_file_operation(
        self,
        operation: str,
        file_path: str,
        overwrite: bool = False,
    ) -> tuple[RiskLevel, List[str]]:
        """Check a file operation for risk.

        Args:
            operation: Type of operation (write, edit, delete)
            file_path: Path to the file
            overwrite: Whether operation would overwrite existing content

        Returns:
            Tuple of (risk_level, list of risk descriptions)
        """
        risks: List[str] = []
        max_risk = RiskLevel.SAFE

        path = Path(file_path)

        # Check for sensitive file types
        for ext in self.DANGEROUS_FILE_EXTENSIONS:
            if file_path.endswith(ext) or ext in file_path:
                risks.append(f"Modifying sensitive file: {ext}")
                max_risk = RiskLevel.HIGH
                break

        # Check for destructive operations
        if operation == "delete":
            risks.append(f"Deleting file: {file_path}")
            if _RISK_ORDER[max_risk] < _RISK_ORDER[RiskLevel.HIGH]:
                max_risk = RiskLevel.HIGH
        elif operation == "write" and overwrite:
            # Check if file exists (would be overwritten)
            if path.exists():
                risks.append(f"Overwriting existing file: {file_path}")
                if _RISK_ORDER[max_risk] < _RISK_ORDER[RiskLevel.MEDIUM]:
                    max_risk = RiskLevel.MEDIUM

        # Check for system paths
        if file_path.startswith("/etc/") or file_path.startswith("/usr/"):
            risks.append("Modifying system file")
            max_risk = RiskLevel.HIGH

        return max_risk, risks

    def is_write_tool(self, tool_name: str) -> bool:
        """Check if a tool is a write/modify operation.

        Args:
            tool_name: Name of the tool

        Returns:
            True if the tool can modify files
        """
        return tool_name in WRITE_TOOL_NAMES

    async def check_and_confirm(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Check operation safety and request confirmation if needed.

        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments

        Returns:
            Tuple of (should_proceed, optional_rejection_reason)
        """
        # Fast path: approval mode OFF - auto-approve everything
        if self.approval_mode == ApprovalMode.OFF:
            return True, None

        risk_level = RiskLevel.SAFE
        descriptions: List[str] = []
        details: List[str] = []
        is_write_operation = self.is_write_tool(tool_name)

        # Check bash commands
        if tool_name == "execute_bash":
            command = arguments.get("command", "")
            risk_level, details = self.check_bash_command(command)
            if details:
                descriptions.append(f"Execute: {command[:100]}...")

        # Check file write operations
        elif tool_name == "write_file":
            file_path = arguments.get("path", arguments.get("file_path", ""))
            risk_level, details = self.check_file_operation("write", file_path, overwrite=True)
            if details:
                descriptions.append(f"Write to: {file_path}")

        # Check file edit operations
        elif tool_name == "edit_files":
            edits = arguments.get("edits", [])
            for edit in edits:
                file_path = edit.get("path", "")
                edit_risk, edit_details = self.check_file_operation("edit", file_path)
                if _RISK_ORDER[edit_risk] > _RISK_ORDER[risk_level]:
                    risk_level = edit_risk
                details.extend(edit_details)
            if edits:
                descriptions.append(f"Edit {len(edits)} file(s)")

        # Check git operations
        elif tool_name == "git":
            subcommand = arguments.get("subcommand", "")
            args = arguments.get("args", "")
            full_cmd = f"git {subcommand} {args}"
            risk_level, details = self.check_bash_command(full_cmd)
            if details:
                descriptions.append(f"Git: {subcommand}")

        # Determine if confirmation is needed
        requires_confirmation = False

        # ALL_WRITES mode: require confirmation for any write operation
        if self.approval_mode == ApprovalMode.ALL_WRITES and is_write_operation:
            requires_confirmation = True
            # If no descriptions yet, add a generic one
            if not descriptions:
                descriptions.append(f"Execute write operation: {tool_name}")
            # Set minimum risk to LOW for display purposes
            if risk_level == RiskLevel.SAFE:
                risk_level = RiskLevel.LOW
                details.append("Write operation requiring approval")

        # RISKY_ONLY mode: only require confirmation for high-risk operations
        elif self.approval_mode == ApprovalMode.RISKY_ONLY:
            if _RISK_ORDER[risk_level] >= _RISK_ORDER[self.require_confirmation_threshold]:
                requires_confirmation = True

        # Auto-approve if confirmation not required
        if not requires_confirmation:
            return True, None

        # No callback registered - log warning but proceed
        if not self.confirmation_callback:
            if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                logger.warning(
                    f"High-risk operation without confirmation callback: "
                    f"{tool_name} - {', '.join(details)}"
                )
            elif self.approval_mode == ApprovalMode.ALL_WRITES:
                logger.warning(
                    f"Write operation without confirmation callback in ALL_WRITES mode: "
                    f"{tool_name}"
                )
            return True, None

        # Create confirmation request
        request = ConfirmationRequest(
            tool_name=tool_name,
            risk_level=risk_level,
            description="; ".join(descriptions) if descriptions else f"Execute {tool_name}",
            details=details,
            arguments=arguments,
        )

        # Request confirmation
        try:
            confirmed = await self.confirmation_callback(request)
            if confirmed:
                return True, None
            else:
                return False, f"Operation cancelled by user: {request.description}"
        except Exception as e:
            logger.error(f"Confirmation callback failed: {e}")
            # If callback fails, block high-risk operations
            if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                return False, f"Confirmation failed: {e}"
            return True, None


# Default singleton instance
_default_checker: Optional[SafetyChecker] = None


def _resolve_approval_mode(mode_str: str) -> ApprovalMode:
    """Resolve approval mode from string setting."""
    mode_map = {
        "off": ApprovalMode.OFF,
        "risky_only": ApprovalMode.RISKY_ONLY,
        "all_writes": ApprovalMode.ALL_WRITES,
    }
    return mode_map.get(mode_str.lower(), ApprovalMode.RISKY_ONLY)


def get_safety_checker() -> SafetyChecker:
    """Get the default safety checker instance.

    The approval mode is configured via settings.write_approval_mode:
    - "off": No approval required (dangerous)
    - "risky_only": Only HIGH/CRITICAL risk (default)
    - "all_writes": All write operations require approval
    """
    global _default_checker
    if _default_checker is None:
        # Try to get approval mode from settings
        try:
            from victor.config.settings import load_settings

            settings = load_settings()
            approval_mode = _resolve_approval_mode(settings.write_approval_mode)
        except Exception:
            # Default to RISKY_ONLY if settings unavailable
            approval_mode = ApprovalMode.RISKY_ONLY

        _default_checker = SafetyChecker(approval_mode=approval_mode)
    return _default_checker


def set_confirmation_callback(callback: ConfirmationCallback) -> None:
    """Set the global confirmation callback for dangerous operations.

    Args:
        callback: Async function that takes ConfirmationRequest and returns bool
    """
    checker = get_safety_checker()
    checker.confirmation_callback = callback
