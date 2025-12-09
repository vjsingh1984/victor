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

from victor.tools.metadata_registry import get_tools_by_stage as registry_get_tools_by_stage

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


# Stage-to-tool mapping is now fully decorator-driven.
# Tools define their stages via @tool(stages=["reading", "execution"]) decorator.
# Use registry_get_tools_by_stage() to get tools for a stage.

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

        # Track file access (use canonical tool names)
        file_arg = args.get("file") or args.get("path") or args.get("file_path")
        if file_arg:
            if tool_name in {"read", "ls", "search", "overview"}:
                self.observed_files.add(str(file_arg))
            elif tool_name in {"write", "edit"}:
                self.modified_files.add(str(file_arg))

    def record_message(self) -> None:
        """Record a new message in the conversation."""
        self.message_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the conversation state to a dictionary.

        Returns:
            Dictionary representation of the state.
        """
        return {
            "stage": self.stage.name,
            "tool_history": self.tool_history,
            "observed_files": list(self.observed_files),
            "modified_files": list(self.modified_files),
            "message_count": self.message_count,
            "last_tools": self.last_tools,
            "stage_confidence": self._stage_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Restore conversation state from a dictionary.

        Args:
            data: Dictionary containing serialized state.

        Returns:
            Restored ConversationState instance.
        """
        state = cls()
        state.stage = ConversationStage[data.get("stage", "INITIAL")]
        state.tool_history = data.get("tool_history", [])
        state.observed_files = set(data.get("observed_files", []))
        state.modified_files = set(data.get("modified_files", []))
        state.message_count = data.get("message_count", 0)
        state.last_tools = data.get("last_tools", [])
        state._stage_confidence = data.get("stage_confidence", 0.5)
        return state


class ConversationStateMachine:
    """State machine for detecting and managing conversation stages.

    Automatically detects the current stage based on:
    - Tool execution patterns
    - Message content analysis
    - Conversation progression

    Includes cooldown mechanism to prevent rapid stage thrashing.
    """

    # Minimum seconds between stage transitions (prevents thrashing)
    # Increased from 3.0 to 5.0 to reduce stage thrashing observed in Ollama models
    TRANSITION_COOLDOWN_SECONDS: float = 5.0

    # Minimum tools required to trigger stage transition
    MIN_TOOLS_FOR_TRANSITION: int = 3

    # Confidence threshold for backward stage transitions
    BACKWARD_TRANSITION_THRESHOLD: float = 0.85

    def __init__(self) -> None:
        """Initialize the state machine."""
        self.state = ConversationState()
        self._last_transition_time: float = 0.0
        self._transition_count: int = 0

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

        Tools define their stages via @tool(stages=["reading", "execution"]) decorator.
        The metadata registry indexes tools by stage for efficient lookup.

        Returns:
            Set of tool names relevant to current stage
        """
        return self._get_tools_for_stage(self.state.stage)

    def _get_tools_for_stage(self, stage: ConversationStage) -> Set[str]:
        """Get tools for a specific stage from the metadata registry.

        Tools define their stages via @tool(stages=["reading", "execution"]) decorator.
        The registry indexes tools by stage for efficient lookup.

        Args:
            stage: The conversation stage to get tools for

        Returns:
            Set of tool names relevant to the stage
        """
        stage_name = stage.name.lower()
        return registry_get_tools_by_stage(stage_name)

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

        # Score stages based on tool overlap (uses registry + static fallback)
        scores: Dict[ConversationStage, int] = {}

        for stage in ConversationStage:
            stage_tools = self._get_tools_for_stage(stage)
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
            stage_tools = self._get_tools_for_stage(detected)
            recent_overlap = len(set(self.state.last_tools) & stage_tools)

            # Use class constant for minimum tools threshold
            if recent_overlap >= self.MIN_TOOLS_FOR_TRANSITION:
                self._transition_to(detected, confidence=0.6 + (recent_overlap * 0.1))

    def _transition_to(self, new_stage: ConversationStage, confidence: float = 0.5) -> None:
        """Transition to a new stage.

        Args:
            new_stage: Stage to transition to
            confidence: Confidence in this transition

        Note: Transitions are rate-limited by TRANSITION_COOLDOWN_SECONDS to
        prevent rapid stage thrashing observed in analysis tasks.
        """
        import time

        old_stage = self.state.stage

        # Don't transition backwards unless confidence is high
        # Use class constant for threshold
        if new_stage.value < old_stage.value and confidence < self.BACKWARD_TRANSITION_THRESHOLD:
            return

        if new_stage != old_stage:
            # Enforce cooldown to prevent stage thrashing
            current_time = time.time()
            time_since_last = current_time - self._last_transition_time

            if time_since_last < self.TRANSITION_COOLDOWN_SECONDS:
                logger.debug(
                    f"Stage transition blocked by cooldown: {old_stage.name} -> {new_stage.name} "
                    f"(waited {time_since_last:.1f}s, need {self.TRANSITION_COOLDOWN_SECONDS}s)"
                )
                return

            logger.info(
                f"Stage transition: {old_stage.name} -> {new_stage.name} "
                f"(confidence: {confidence:.2f})"
            )
            self.state.stage = new_stage
            self.state._stage_confidence = confidence
            self._last_transition_time = current_time
            self._transition_count += 1

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
                if tool_name in self._get_tools_for_stage(stage):
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
                if tool_name in self._get_tools_for_stage(stage):
                    return 0.08  # Medium boost for adjacent stage tools

        return 0.0  # No boost

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the state machine to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationStateMachine":
        """Restore state machine from a dictionary.

        Args:
            data: Dictionary containing serialized state machine.

        Returns:
            Restored ConversationStateMachine instance.
        """
        machine = cls()
        if "state" in data:
            machine.state = ConversationState.from_dict(data["state"])
        return machine
