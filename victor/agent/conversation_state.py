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

"""Conversation state machine for intelligent stage detection.

This module provides automatic detection of conversation stages to improve
tool selection accuracy. Instead of manual stage management, it infers the
current stage from:
- Tool execution history
- Message content patterns
- Observed file/resource access
- Conversation depth

Stages:
- INITIAL: First interaction, exploring the request
- PLANNING: Understanding scope, searching for files
- READING: Examining files, gathering context
- ANALYSIS: Reviewing code, analyzing structure
- EXECUTION: Making changes, running commands
- VERIFICATION: Testing, validating changes
- COMPLETION: Summarizing, wrapping up
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ConversationStage(Enum):
    """Stages in a typical coding assistant conversation."""

    INITIAL = auto()  # First interaction
    PLANNING = auto()  # Understanding scope, planning approach
    READING = auto()  # Reading files, gathering context
    ANALYSIS = auto()  # Analyzing code, understanding structure
    EXECUTION = auto()  # Making changes, running commands
    VERIFICATION = auto()  # Testing, validating
    COMPLETION = auto()  # Summarizing, done


# Tools associated with each stage
STAGE_TOOL_MAPPING: Dict[ConversationStage, Set[str]] = {
    ConversationStage.INITIAL: {
        "code_search",
        "semantic_code_search",
        "plan_files",
        "list_directory",
    },
    ConversationStage.PLANNING: {
        "code_search",
        "semantic_code_search",
        "plan_files",
        "list_directory",
        "analyze_docs",
    },
    ConversationStage.READING: {
        "read_file",
        "list_directory",
        "code_search",
        "semantic_code_search",
    },
    ConversationStage.ANALYSIS: {
        "code_review",
        "analyze_docs",
        "analyze_metrics",
        "security_scan",
        "code_intelligence",
    },
    ConversationStage.EXECUTION: {
        "write_file",
        "edit_file",
        "execute_bash",
        "git",
        "refactor_rename_symbol",
        "refactor_extract_function",
    },
    ConversationStage.VERIFICATION: {
        "testing",
        "execute_bash",
        "read_file",
        "code_review",
    },
    ConversationStage.COMPLETION: {
        "generate_docs",
        "git",
    },
}

# Keywords that suggest specific stages
STAGE_KEYWORDS: Dict[ConversationStage, List[str]] = {
    ConversationStage.INITIAL: ["what", "how", "where", "explain", "help", "can you"],
    ConversationStage.PLANNING: ["plan", "approach", "strategy", "design", "architect"],
    ConversationStage.READING: ["show", "read", "look at", "check", "find"],
    ConversationStage.ANALYSIS: ["analyze", "review", "examine", "understand", "why"],
    ConversationStage.EXECUTION: [
        "change",
        "modify",
        "create",
        "add",
        "remove",
        "fix",
        "implement",
        "write",
        "update",
    ],
    ConversationStage.VERIFICATION: ["test", "verify", "check", "validate", "run"],
    ConversationStage.COMPLETION: ["done", "finish", "complete", "summarize", "commit"],
}


@dataclass
class ConversationState:
    """Tracks the current state of a conversation.

    Attributes:
        stage: Current conversation stage
        tool_history: List of tools executed in order
        observed_files: Files that have been read
        modified_files: Files that have been modified
        message_count: Number of messages in conversation
        last_tools: Last N tools executed (for pattern detection)
    """

    stage: ConversationStage = ConversationStage.INITIAL
    tool_history: List[str] = field(default_factory=list)
    observed_files: Set[str] = field(default_factory=set)
    modified_files: Set[str] = field(default_factory=set)
    message_count: int = 0
    last_tools: List[str] = field(default_factory=list)
    _stage_confidence: float = 0.5

    def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool execution and update state.

        Args:
            tool_name: Name of the executed tool
            args: Tool arguments
        """
        self.tool_history.append(tool_name)
        self.last_tools.append(tool_name)
        if len(self.last_tools) > 5:
            self.last_tools.pop(0)

        # Track file access
        file_arg = args.get("file") or args.get("path") or args.get("file_path")
        if file_arg:
            if tool_name in {"read_file", "list_directory", "code_search"}:
                self.observed_files.add(str(file_arg))
            elif tool_name in {"write_file", "edit_file"}:
                self.modified_files.add(str(file_arg))

    def record_message(self) -> None:
        """Record a new message in the conversation."""
        self.message_count += 1


class ConversationStateMachine:
    """State machine for detecting and managing conversation stages.

    Automatically detects the current stage based on:
    - Tool execution patterns
    - Message content analysis
    - Conversation progression
    """

    def __init__(self) -> None:
        """Initialize the state machine."""
        self.state = ConversationState()

    def reset(self) -> None:
        """Reset state for a new conversation."""
        self.state = ConversationState()

    def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record tool execution and potentially transition stage.

        Args:
            tool_name: Name of the executed tool
            args: Tool arguments
        """
        self.state.record_tool_execution(tool_name, args)
        self._maybe_transition()

    def record_message(self, content: str, is_user: bool = True) -> None:
        """Record a message and potentially transition stage.

        Args:
            content: Message content
            is_user: Whether the message is from the user
        """
        self.state.record_message()

        if is_user:
            # Detect stage from user message content
            detected_stage = self._detect_stage_from_content(content)
            if detected_stage:
                self._transition_to(detected_stage, confidence=0.7)

    def get_stage(self) -> ConversationStage:
        """Get the current conversation stage.

        Returns:
            Current ConversationStage
        """
        return self.state.stage

    def get_stage_tools(self) -> Set[str]:
        """Get tools relevant to the current stage.

        Returns:
            Set of tool names relevant to current stage
        """
        return STAGE_TOOL_MAPPING.get(self.state.stage, set())

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state.

        Returns:
            Dictionary with state information
        """
        return {
            "stage": self.state.stage.name,
            "confidence": self.state._stage_confidence,
            "message_count": self.state.message_count,
            "tools_executed": len(self.state.tool_history),
            "files_observed": len(self.state.observed_files),
            "files_modified": len(self.state.modified_files),
            "last_tools": self.state.last_tools,
            "recommended_tools": list(self.get_stage_tools()),
        }

    def _detect_stage_from_content(self, content: str) -> Optional[ConversationStage]:
        """Detect stage from message content using keyword matching.

        Args:
            content: Message content to analyze

        Returns:
            Detected stage or None
        """
        content_lower = content.lower()

        # Score each stage based on keyword matches
        scores: Dict[ConversationStage, int] = {}

        for stage, keywords in STAGE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scores[stage] = score

        if scores:
            # Return stage with highest score
            best_stage = max(scores, key=scores.get)  # type: ignore
            if scores[best_stage] >= 2:  # Require at least 2 keyword matches
                return best_stage

        return None

    def _detect_stage_from_tools(self) -> Optional[ConversationStage]:
        """Detect stage from recent tool execution patterns.

        Returns:
            Detected stage or None
        """
        if not self.state.last_tools:
            return None

        # Score stages based on tool overlap
        scores: Dict[ConversationStage, int] = {}

        for stage, stage_tools in STAGE_TOOL_MAPPING.items():
            overlap = len(set(self.state.last_tools) & stage_tools)
            if overlap > 0:
                scores[stage] = overlap

        if scores:
            return max(scores, key=scores.get)  # type: ignore

        return None

    def _maybe_transition(self) -> None:
        """Check if we should transition to a new stage."""
        detected = self._detect_stage_from_tools()
        if detected and detected != self.state.stage:
            # Only transition if we have strong evidence
            stage_tools = STAGE_TOOL_MAPPING.get(detected, set())
            recent_overlap = len(set(self.state.last_tools) & stage_tools)

            if recent_overlap >= 2:  # At least 2 matching tools
                self._transition_to(detected, confidence=0.6 + (recent_overlap * 0.1))

    def _transition_to(self, new_stage: ConversationStage, confidence: float = 0.5) -> None:
        """Transition to a new stage.

        Args:
            new_stage: Stage to transition to
            confidence: Confidence in this transition
        """
        old_stage = self.state.stage

        # Don't transition backwards unless confidence is high
        if new_stage.value < old_stage.value and confidence < 0.8:
            return

        if new_stage != old_stage:
            logger.info(
                f"Stage transition: {old_stage.name} -> {new_stage.name} "
                f"(confidence: {confidence:.2f})"
            )
            self.state.stage = new_stage
            self.state._stage_confidence = confidence

    def should_include_tool(self, tool_name: str) -> bool:
        """Check if a tool should be included based on current stage.

        This provides a soft recommendation - tools outside the current
        stage are not excluded, just deprioritized.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is recommended for current stage
        """
        stage_tools = self.get_stage_tools()

        # Always include if in current stage's tools
        if tool_name in stage_tools:
            return True

        # Also include if in adjacent stages (flexible)
        current_idx = self.state.stage.value
        for stage in ConversationStage:
            if abs(stage.value - current_idx) <= 1:
                if tool_name in STAGE_TOOL_MAPPING.get(stage, set()):
                    return True

        return False

    def get_tool_priority_boost(self, tool_name: str) -> float:
        """Get priority boost for a tool based on current stage.

        Args:
            tool_name: Name of the tool

        Returns:
            Boost value (0.0 to 0.2) to add to similarity score
        """
        if tool_name in self.get_stage_tools():
            return 0.15  # High boost for stage-relevant tools

        # Check adjacent stages
        current_idx = self.state.stage.value
        for stage in ConversationStage:
            if abs(stage.value - current_idx) == 1:
                if tool_name in STAGE_TOOL_MAPPING.get(stage, set()):
                    return 0.08  # Medium boost for adjacent stage tools

        return 0.0  # No boost
