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

"""Shared turn policy for batch and streaming execution paths.

This module provides unified decision logic for both
TurnExecutor (batch) and StreamingChatPipeline (streaming):

- SpinDetector: detects blocked/stuck agent loops
- NudgePolicy: determines when/what nudge messages to inject
- FulfillmentCriteriaBuilder: auto-derives file-level criteria from tool results

Both execution paths import from this module to ensure consistent
behavior. No path-specific logic belongs here — only shared decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# Shared Constants
# ============================================================================

MAX_NO_TOOL_TURNS = 3
"""Maximum consecutive turns without tool calls before termination."""

MAX_ALL_BLOCKED = 3
"""Maximum consecutive turns where all tool calls are dedup-blocked."""

NUDGE_THRESHOLD = 2
"""Inject nudge message after this many no-tool turns."""

READ_ONLY_TOOLS = frozenset({"read", "ls", "list_directory", "grep"})
"""Tools considered read-only for code_search escalation nudge."""

READ_ONLY_ESCALATION_THRESHOLD = 5
"""Consecutive read-only turns before suggesting code_search."""


# ============================================================================
# SpinDetector
# ============================================================================


class SpinState(Enum):
    """Current spin detection state."""

    NORMAL = "normal"
    WARNING = "warning"  # Approaching limit
    BLOCKED = "blocked"  # All tools blocked by dedup
    STUCK = "stuck"  # No tools used for too long
    TERMINATED = "terminated"  # Limit exceeded


@dataclass
class SpinDetector:
    """Detects stuck/blocked agent loops.

    Tracks consecutive turns without tool calls and consecutive turns
    where all tool calls are blocked by dedup. Used by both batch
    (AgenticLoop) and streaming (StreamingChatPipeline) paths.

    Example:
        detector = SpinDetector()

        for each turn:
            detector.record_turn(has_tools=True, all_blocked=False)
            state = detector.state
            if state == SpinState.TERMINATED:
                break
    """

    consecutive_no_tool_turns: int = 0
    consecutive_all_blocked: int = 0
    consecutive_read_only_turns: int = 0
    total_tool_calls: int = 0
    has_used_code_search: bool = False

    @property
    def state(self) -> SpinState:
        """Current spin state based on tracking counters."""
        if self.consecutive_all_blocked >= MAX_ALL_BLOCKED:
            return SpinState.TERMINATED
        if self.consecutive_no_tool_turns >= MAX_NO_TOOL_TURNS:
            return SpinState.TERMINATED
        if self.consecutive_all_blocked >= 2:
            return SpinState.BLOCKED
        if self.consecutive_no_tool_turns >= NUDGE_THRESHOLD:
            return SpinState.WARNING
        return SpinState.NORMAL

    def record_turn(
        self,
        has_tool_calls: bool,
        all_blocked: bool = False,
        tool_names: Optional[Set[str]] = None,
        tool_count: int = 0,
    ) -> SpinState:
        """Record a turn and return updated state.

        Args:
            has_tool_calls: Whether model requested tool calls
            all_blocked: Whether all tool calls were blocked by dedup
            tool_names: Set of tool names used (for read-only tracking)
            tool_count: Number of tool calls in this turn

        Returns:
            Updated SpinState
        """
        if has_tool_calls:
            self.consecutive_no_tool_turns = 0
            self.total_tool_calls += tool_count

            if all_blocked:
                self.consecutive_all_blocked += 1
            else:
                self.consecutive_all_blocked = 0

            # Track read-only turns for code_search escalation
            if tool_names:
                if "code_search" in tool_names:
                    self.has_used_code_search = True
                if tool_names.issubset(READ_ONLY_TOOLS):
                    self.consecutive_read_only_turns += 1
                else:
                    self.consecutive_read_only_turns = 0
        else:
            self.consecutive_no_tool_turns += 1

        return self.state

    def reset(self) -> None:
        """Reset all counters for a new conversation."""
        self.consecutive_no_tool_turns = 0
        self.consecutive_all_blocked = 0
        self.consecutive_read_only_turns = 0
        self.total_tool_calls = 0
        self.has_used_code_search = False


# ============================================================================
# NudgePolicy
# ============================================================================


class NudgeType(Enum):
    """Types of nudge messages."""

    NONE = "none"
    USE_TOOLS = "use_tools"  # Agent not using tools
    DIFFERENT_TOOLS = "different_tools"  # All tools blocked by dedup
    CODE_SEARCH = "code_search"  # Too many read-only turns
    BUDGET_WARNING = "budget_warning"  # Past halfway on iteration budget


@dataclass
class NudgeDecision:
    """Decision about whether and what to nudge.

    Attributes:
        nudge_type: Type of nudge needed
        message: Nudge message text (user or system role)
        role: Message role ("user" or "system")
        should_inject: Whether to inject the nudge
    """

    nudge_type: NudgeType = NudgeType.NONE
    message: str = ""
    role: str = "user"

    @property
    def should_inject(self) -> bool:
        return self.nudge_type != NudgeType.NONE


class NudgePolicy:
    """Determines when and what nudge messages to inject.

    Uses SpinDetector state to decide. Both batch and streaming paths
    call this to get consistent nudge behavior.

    Example:
        policy = NudgePolicy()
        decision = policy.evaluate(detector, iteration=5, max_iterations=10)
        if decision.should_inject:
            chat_context.add_message(decision.role, decision.message)
    """

    def evaluate(
        self,
        detector: SpinDetector,
        iteration: int = 0,
        max_iterations: int = 10,
    ) -> NudgeDecision:
        """Evaluate whether a nudge is needed.

        Args:
            detector: Current spin detector state
            iteration: Current iteration number
            max_iterations: Maximum iterations

        Returns:
            NudgeDecision with nudge type and message
        """
        state = detector.state

        # All tools blocked by dedup
        if state == SpinState.BLOCKED:
            return NudgeDecision(
                nudge_type=NudgeType.DIFFERENT_TOOLS,
                message=(
                    "Your last tool calls were blocked because you already "
                    "called them with the same arguments. Try a DIFFERENT "
                    "tool or different arguments. If you've made your fix, "
                    "provide your final answer."
                ),
                role="user",
            )

        # Agent not using tools
        if state == SpinState.WARNING:
            nudge = NudgeDecision(
                nudge_type=NudgeType.USE_TOOLS,
                message=(
                    f"You have not called any tools in the last "
                    f"{detector.consecutive_no_tool_turns} turns. You MUST "
                    f"use a tool now (read, edit, write, shell) to make "
                    f"progress on the task. Do not respond with text only."
                ),
                role="user",
            )
            return nudge

        # Too many read-only turns without code_search
        if (
            detector.consecutive_read_only_turns >= READ_ONLY_ESCALATION_THRESHOLD
            and not detector.has_used_code_search
        ):
            return NudgeDecision(
                nudge_type=NudgeType.CODE_SEARCH,
                message=(
                    "You have been browsing files for several turns. "
                    "Consider using code_search(query='...') to find "
                    "relevant code more efficiently."
                ),
                role="user",
            )

        return NudgeDecision()

    def budget_warning(
        self,
        iteration: int,
        max_iterations: int,
    ) -> NudgeDecision:
        """Check if a budget warning should be issued.

        Args:
            iteration: Current iteration
            max_iterations: Maximum iterations

        Returns:
            NudgeDecision with budget warning if past halfway
        """
        if iteration > max_iterations // 2:
            remaining = max_iterations - iteration
            return NudgeDecision(
                nudge_type=NudgeType.BUDGET_WARNING,
                message=(
                    f"WARNING: {remaining} turns remaining out of "
                    f"{max_iterations}. Make your edits NOW."
                ),
                role="user",
            )
        return NudgeDecision()


# ============================================================================
# FulfillmentCriteriaBuilder
# ============================================================================


@dataclass
class FulfillmentCriteria:
    """Auto-derived fulfillment criteria from tool execution results.

    Built by analyzing tool calls to determine what files were created/modified,
    what tests were run, etc. Used by FulfillmentDetector for completion checking.
    """

    file_paths: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    original_error: Optional[str] = None
    required_patterns: List[str] = field(default_factory=list)
    doc_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to criteria dict for FulfillmentDetector."""
        criteria: Dict[str, Any] = {}
        if self.file_paths:
            criteria["file_path"] = self.file_paths[0]  # Primary file
        if self.test_files:
            criteria["test_files"] = self.test_files
        if self.original_error:
            criteria["original_error"] = self.original_error
        if self.required_patterns:
            criteria["required_patterns"] = self.required_patterns
        if self.doc_files:
            criteria["doc_files"] = self.doc_files
        return criteria


class FulfillmentCriteriaBuilder:
    """Builds fulfillment criteria from tool execution results.

    Analyzes tool calls to determine what files were created, modified,
    or tested. This enables auto-derived fulfillment checking without
    requiring explicit criteria from the user.

    Example:
        builder = FulfillmentCriteriaBuilder()
        for tool_result in turn.tool_results:
            builder.record_tool_result(tool_result)
        criteria = builder.build()
    """

    def __init__(self) -> None:
        self._written_files: List[str] = []
        self._edited_files: List[str] = []
        self._test_files: List[str] = []
        self._doc_files: List[str] = []
        self._errors: List[str] = []

    def record_tool_result(self, result: Dict[str, Any]) -> None:
        """Record a tool execution result.

        Args:
            result: Tool result dict with tool_name, args, success, etc.
        """
        tool_name = result.get("tool_name", "")
        args = result.get("args", {})
        success = result.get("success", False)

        if not success:
            error = result.get("error", "")
            if error:
                self._errors.append(error)
            return

        # Track file operations
        file_path = args.get("file_path", "") or args.get("path", "")

        if tool_name in ("write", "write_file", "create_file"):
            if file_path:
                self._written_files.append(file_path)
                if file_path.endswith(".md") or file_path.endswith(".rst"):
                    self._doc_files.append(file_path)

        elif tool_name in ("edit", "edit_file", "replace_in_file"):
            if file_path:
                self._edited_files.append(file_path)

        elif tool_name in ("shell", "bash", "run_command"):
            command = args.get("command", "")
            if "pytest" in command or "test" in command:
                # Extract test file if present
                parts = command.split()
                for part in parts:
                    if part.endswith(".py") and "test" in part:
                        self._test_files.append(part)

    def build(self) -> FulfillmentCriteria:
        """Build fulfillment criteria from recorded tool results.

        Returns:
            FulfillmentCriteria with auto-derived fields
        """
        all_files = self._written_files + self._edited_files
        return FulfillmentCriteria(
            file_paths=all_files,
            test_files=self._test_files,
            original_error=self._errors[0] if self._errors else None,
            doc_files=self._doc_files,
        )

    def reset(self) -> None:
        """Reset builder for next conversation."""
        self._written_files.clear()
        self._edited_files.clear()
        self._test_files.clear()
        self._doc_files.clear()
        self._errors.clear()
