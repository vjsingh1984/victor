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

MIGRATION NOTICE: State storage is migrating to canonical system.

For state storage, use the canonical state management system:
    - victor.state.ConversationStateManager - Conversation scope state
    - victor.state.get_global_manager() - Unified access to all scopes

This module is kept for its stage detection business logic.
The ConversationStateMachine now uses ConversationStateManager internally
for state storage.

---

Legacy Documentation:

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

Hook Integration (Observer Pattern):
The state machine supports hooks for observing transitions:
- on_enter: Called when entering a stage
- on_exit: Called when exiting a stage
- on_transition: Called on any stage change

Example:
    from victor.observability import StateHookManager

    hooks = StateHookManager()

    @hooks.on_transition
    def log_transition(old, new, ctx):
        print(f"{old} -> {new}")

    machine = ConversationStateMachine(hooks=hooks)

Migration Example:
    # OLD (using ConversationStateMachine for state storage):
    machine = ConversationStateMachine()
    machine.record_tool_execution("read", {"file": "test.py"})
    stage = machine.get_stage()

    # NEW (using canonical state management):
    from victor.state import ConversationStateManager, StateScope

    # For stage detection, still use ConversationStateMachine:
    machine = ConversationStateMachine()
    machine.record_tool_execution("read", {"file": "test.py"})
    stage = machine.get_stage()

    # For state storage, use ConversationStateManager:
    mgr = ConversationStateManager()
    await mgr.set("tool_history", ["read"])
    await mgr.set("observed_files", {"test.py"})

    # OR for unified access:
    from victor.state import get_global_manager
    state = get_global_manager()
    await state.set("tool_history", ["read"], scope=StateScope.CONVERSATION)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, cast

from victor.tools.metadata_registry import get_tools_by_stage as registry_get_tools_by_stage
from victor.core.events import ObservabilityBus

if TYPE_CHECKING:
    from victor.observability.hooks import StateHookManager

logger = logging.getLogger(__name__)


# Import canonical ConversationStage from victor.core.state (single source of truth)
# This import enforces layer boundaries - Framework should not depend on Agent
from victor.core.state import ConversationStage  # noqa: F401

__all__ = ["ConversationStage"]


# Stage ordering for adjacency calculations (since values are strings, not ints)
STAGE_ORDER: Dict[ConversationStage, int] = {
    stage: idx for idx, stage in enumerate(ConversationStage)
}


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

    Supports hooks via the Observer pattern for decoupled state observation:
    - on_enter: Called when entering a stage
    - on_exit: Called when exiting a stage
    - on_transition: Called on any stage change

    Example:
        from victor.observability import StateHookManager

        hooks = StateHookManager()

        @hooks.on_enter("EXECUTION")
        def on_exec_start(stage, context):
            print("Starting execution phase!")

        machine = ConversationStateMachine(hooks=hooks)
    """

    # Minimum seconds between stage transitions (prevents thrashing)
    # Reduced from 5.0 to 2.0 to allow faster progression for SWE-bench tasks
    # while still preventing thrashing
    TRANSITION_COOLDOWN_SECONDS: float = 2.0

    # Minimum tools required to trigger stage transition
    MIN_TOOLS_FOR_TRANSITION: int = 3

    # Confidence threshold for backward stage transitions
    BACKWARD_TRANSITION_THRESHOLD: float = 0.85

    # Maximum reads without edit before forcing READING → EXECUTION transition
    # Prevents infinite exploration in SWE-bench style bug fix tasks
    MAX_READS_WITHOUT_EDIT: int = 7

    def __init__(
        self,
        hooks: Optional["StateHookManager"] = None,
        track_history: bool = True,
        max_history_size: int = 100,
        event_bus: Optional[ObservabilityBus] = None,
        state_manager: Optional[Any] = None,
        use_transition_engine: bool = True,
    ) -> None:
        """Initialize the state machine.

        Args:
            hooks: Optional StateHookManager for transition callbacks.
            track_history: Whether to track transition history.
            max_history_size: Maximum number of transitions to keep in history.
            event_bus: Optional ObservabilityBus instance. If None, uses DI container.
            state_manager: Optional ConversationStateManager for canonical state storage.
                          If provided, state will be synced to the manager.
            use_transition_engine: Whether to use StageTransitionEngine for validation.
        """
        self.state = ConversationState()
        self._last_transition_time: float = 0.0
        self._transition_count: int = 0
        self._hooks = hooks
        self._track_history = track_history
        self._max_history_size = max_history_size
        self._transition_history: List[Dict[str, Any]] = []
        self._event_bus = event_bus or self._get_default_bus()
        self._state_manager = state_manager  # Optional canonical state manager

        # Initialize StageTransitionEngine for validated transitions and tool priorities
        self._use_transition_engine = use_transition_engine
        self._transition_engine: Optional[Any] = None  # Will be set below if needed
        if use_transition_engine:
            from victor.agent.stage_transition_engine import StageTransitionEngine

            self._transition_engine = StageTransitionEngine(
                initial_stage=self.state.stage,
                cooldown_seconds=self.TRANSITION_COOLDOWN_SECONDS,
                event_bus=self._event_bus,
            )
            logger.debug("StageTransitionEngine initialized for ConversationStateMachine")
        else:
            self._transition_engine = None

        # Sync initial state to manager if provided
        if self._state_manager:
            self._sync_state_to_manager()

    def reset(self) -> None:
        """Reset state for a new conversation."""
        self.state = ConversationState()
        self._transition_history.clear()
        self._transition_count = 0
        self._last_transition_time = 0.0

        # Reset transition engine if initialized
        if self._transition_engine:
            self._transition_engine.reset()

        # Reset manager state if provided
        if self._state_manager:
            self._sync_state_to_manager()

    def _get_default_bus(self) -> Optional[ObservabilityBus]:
        """Get default ObservabilityBus from DI container.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    def _sync_state_to_manager(self) -> None:
        """Sync current state to the canonical state manager.

        This is called internally when state changes to keep the
        ConversationStateManager in sync.
        """
        if not self._state_manager:
            return

        try:
            # Sync state to manager (non-blocking)
            # We do this synchronously for compatibility
            state_dict = self.state.to_dict()

            # Store in manager (using internal _state for direct access)
            self._state_manager._state.update(state_dict)
        except Exception as e:
            logger.warning(f"Failed to sync state to manager: {e}")

    def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record tool execution and potentially transition stage.

        Args:
            tool_name: Name of the executed tool
            args: Tool arguments
        """
        self.state.record_tool_execution(tool_name, args)

        # Sync to canonical state manager if provided
        if self._state_manager:
            self._sync_state_to_manager()

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

    def get_current_stage(self) -> ConversationStage:
        """Backward-compatible alias for orchestrator integrations."""
        return self.get_stage()

    def get_valid_transitions(self) -> List[ConversationStage]:
        """Get list of valid transition targets from current stage.

        Returns:
            List of stages that can be transitioned to
        """
        if self._transition_engine:
            result = self._transition_engine.get_valid_transitions()
            return cast(List[ConversationStage], result)
        # Fallback: all stages
        return list(ConversationStage)

    def can_transition_to(self, target_stage: ConversationStage, confidence: float = 0.5) -> bool:
        """Check if transition to target stage is valid.

        Args:
            target_stage: Stage to check
            confidence: Confidence level for backward transitions

        Returns:
            True if transition is valid
        """
        if self._transition_engine:
            result = self._transition_engine.can_transition(target_stage, confidence)
            return cast(bool, result)
        # Fallback: always allow
        return True

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
            logger.debug("_detect_stage_from_tools: No recent tools, returning None")
            return None

        # Score stages based on tool overlap (uses registry + static fallback)
        scores: Dict[ConversationStage, int] = {}

        for stage in ConversationStage:
            stage_tools = self._get_tools_for_stage(stage)
            overlap = len(set(self.state.last_tools) & stage_tools)
            if overlap > 0:
                scores[stage] = overlap

        if scores:
            max_score = max(scores.values())
            tied_stages = [s for s, v in scores.items() if v == max_score]

            # Tie-breaking logic to prevent oscillation:
            # 1. Prefer current stage if it's in the tied group (stability)
            # 2. Otherwise, prefer the most advanced stage (forward progress)
            if len(tied_stages) == 1:
                detected = tied_stages[0]
            elif self.state.stage in tied_stages:
                # Current stage is tied - stay to avoid oscillation
                detected = self.state.stage
                logger.debug("_detect_stage_from_tools: Tie resolved by staying at current stage")
            else:
                # Pick the most advanced (highest in workflow order)
                detected = max(tied_stages, key=lambda s: STAGE_ORDER[s])
                logger.debug(
                    "_detect_stage_from_tools: Tie resolved by picking most advanced stage"
                )

            scores_str = ", ".join(f"{k.name}={v}" for k, v in scores.items())
            logger.debug(
                f"_detect_stage_from_tools: last_tools={self.state.last_tools}, "
                f"scores=[{scores_str}], detected={detected.name}"
            )
            return detected

        logger.debug(
            f"_detect_stage_from_tools: No stage overlap for tools={self.state.last_tools}"
        )
        return None

    def _maybe_transition(self) -> None:
        """Check if we should transition to a new stage."""
        # Force READING → EXECUTION if we've read too many files without editing
        # This prevents infinite exploration in SWE-bench style bug fix tasks
        if self._should_force_execution_transition():
            logger.info(
                f"Forcing READING→EXECUTION: {len(self.state.observed_files)} files read, "
                f"{len(self.state.modified_files)} files modified"
            )
            self._transition_to(ConversationStage.EXECUTION, confidence=0.8)
            return

        detected = self._detect_stage_from_tools()
        if detected and detected != self.state.stage:
            # Only transition if we have strong evidence
            stage_tools = self._get_tools_for_stage(detected)
            recent_overlap = len(set(self.state.last_tools) & stage_tools)

            logger.debug(
                f"_maybe_transition: current={self.state.stage.name}, detected={detected.name}, "
                f"recent_overlap={recent_overlap}, min_threshold={self.MIN_TOOLS_FOR_TRANSITION}, "
                f"stage_tools_count={len(stage_tools)}"
            )

            # Use class constant for minimum tools threshold
            if recent_overlap >= self.MIN_TOOLS_FOR_TRANSITION:
                self._transition_to(detected, confidence=0.6 + (recent_overlap * 0.1))
            else:
                logger.debug(
                    f"_maybe_transition: Transition blocked - overlap {recent_overlap} < threshold {self.MIN_TOOLS_FOR_TRANSITION}"
                )

    def _should_force_execution_transition(self) -> bool:
        """Check if we should force transition from READING to EXECUTION.

        Conditions:
        1. Current stage is READING (or ANALYSIS)
        2. We've observed many files (> MAX_READS_WITHOUT_EDIT)
        3. We haven't modified any files yet

        This prevents the agent from getting stuck in endless exploration
        for SWE-bench style bug fix tasks.

        Returns:
            True if we should force transition to EXECUTION
        """
        # Only force from READING or ANALYSIS stages
        if self.state.stage not in {ConversationStage.READING, ConversationStage.ANALYSIS}:
            return False

        # Only force if we've read many files but haven't edited any
        files_read = len(self.state.observed_files)
        files_modified = len(self.state.modified_files)

        if files_read >= self.MAX_READS_WITHOUT_EDIT and files_modified == 0:
            logger.debug(
                f"_should_force_execution: files_read={files_read} >= {self.MAX_READS_WITHOUT_EDIT}, "
                f"files_modified={files_modified}"
            )
            return True

        return False

    def _transition_to(self, new_stage: ConversationStage, confidence: float = 0.5) -> None:
        """Transition to a new stage.

        Args:
            new_stage: Stage to transition to
            confidence: Confidence in this transition

        Note: Transitions are rate-limited by TRANSITION_COOLDOWN_SECONDS to
        prevent rapid stage thrashing observed in analysis tasks.

        Uses StageTransitionEngine for validation if enabled.

        Hook invocation order (Observer Pattern):
        1. on_exit hooks for old_stage
        2. Stage update
        3. on_transition hooks (old_stage, new_stage)
        4. on_enter hooks for new_stage
        """
        import time

        old_stage = self.state.stage
        current_time: float  # Will be set in both engine and legacy paths

        # Use StageTransitionEngine for validation if enabled
        if self._transition_engine:
            # Check if transition is valid using the engine's graph
            if not self._transition_engine.can_transition(new_stage, confidence):
                logger.debug(
                    f"_transition_to: Transition blocked by StageTransitionEngine: "
                    f"{old_stage.name} -> {new_stage.name} (confidence={confidence:.2f})"
                )
                return

            # Use the engine's transition method (handles cooldown and events)
            if not self._transition_engine.transition_to(new_stage, confidence):
                logger.debug(
                    f"_transition_to: Transition rejected by StageTransitionEngine: "
                    f"{old_stage.name} -> {new_stage.name}"
                )
                return

            # Engine handled the transition, now sync our state
            self.state.stage = self._transition_engine.current_stage
            self.state._stage_confidence = confidence
            self._last_transition_time = self._transition_engine._last_transition_time
            self._transition_count = self._transition_engine.transition_count
            # Get timestamp from engine for history recording
            current_time = self._transition_engine._last_transition_time
        else:
            # Legacy transition logic (without engine)
            # Don't transition backwards unless confidence is high
            if (
                STAGE_ORDER[new_stage] < STAGE_ORDER[old_stage]
                and confidence < self.BACKWARD_TRANSITION_THRESHOLD
            ):
                logger.debug(
                    f"_transition_to: Backward transition blocked {old_stage.name} -> {new_stage.name}, "
                    f"confidence={confidence:.2f} < threshold={self.BACKWARD_TRANSITION_THRESHOLD}"
                )
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

                # Update state
                self.state.stage = new_stage
                self.state._stage_confidence = confidence
                self._last_transition_time = current_time
                self._transition_count += 1

        # Build context for hooks (only if we actually transitioned)
        if self.state.stage != old_stage:
            hook_context = self._build_hook_context(confidence)

            # Fire hooks if registered (Observer Pattern)
            if self._hooks:
                self._hooks.fire_transition(
                    old_stage.name,
                    new_stage.name,
                    hook_context,
                )

            # Sync to canonical state manager if provided
            if self._state_manager:
                self._sync_state_to_manager()

            # Emit STATE event for stage transition (only if using legacy path)
            if not self._transition_engine and self._event_bus:
                try:
                    from victor.core.events.emit_helper import emit_event_sync

                    emit_event_sync(
                        self._event_bus,
                        topic="state.stage_changed",
                        data={
                            "old_stage": old_stage.name,
                            "new_stage": new_stage.name,
                            "confidence": confidence,
                            "transition_count": self._transition_count,
                            "message_count": self.state.message_count,
                            "tools_executed": len(self.state.tool_history),
                            "files_observed": len(self.state.observed_files),
                            "files_modified": len(self.state.modified_files),
                            "category": "state",  # Preserve for observability
                        },
                        source="ConversationStateMachine",
                    )
                except Exception as e:
                    logger.debug(f"Failed to emit stage change event: {e}")

            # Record transition history
            if self._track_history:
                self._record_transition(
                    old_stage, new_stage, confidence, current_time, hook_context
                )

    def _build_hook_context(self, confidence: float) -> Dict[str, Any]:
        """Build context dictionary for hook callbacks.

        Args:
            confidence: Transition confidence.

        Returns:
            Context dictionary with state information.
        """
        return {
            "confidence": confidence,
            "tool_history": list(self.state.tool_history[-10:]),  # Last 10 tools
            "last_tools": list(self.state.last_tools),
            "message_count": self.state.message_count,
            "files_observed": len(self.state.observed_files),
            "files_modified": len(self.state.modified_files),
            "transition_count": self._transition_count,
        }

    def _record_transition(
        self,
        old_stage: ConversationStage,
        new_stage: ConversationStage,
        confidence: float,
        timestamp: float,
        context: Dict[str, Any],
    ) -> None:
        """Record a transition in history.

        Args:
            old_stage: Previous stage.
            new_stage: New stage.
            confidence: Transition confidence.
            timestamp: Unix timestamp.
            context: Transition context.
        """
        from datetime import datetime

        record = {
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
            "confidence": confidence,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "transition_number": self._transition_count,
            "message_count": context.get("message_count", 0),
            "tool_count": len(context.get("tool_history", [])),
        }

        self._transition_history.append(record)

        # Enforce max history size
        if len(self._transition_history) > self._max_history_size:
            self._transition_history.pop(0)

    @property
    def transition_history(self) -> List[Dict[str, Any]]:
        """Get the transition history.

        Returns:
            List of transition records.
        """
        return list(self._transition_history)

    @property
    def transition_count(self) -> int:
        """Get the total number of transitions.

        Returns:
            Number of transitions.
        """
        return self._transition_count

    def get_transitions_summary(self) -> Dict[str, Any]:
        """Get a summary of all transitions.

        Returns:
            Dictionary with transition statistics.
        """
        if not self._transition_history:
            return {
                "total_transitions": 0,
                "unique_paths": 0,
                "transitions_by_stage": {},
                "average_confidence": 0.0,
            }

        # Count transitions by path
        path_counts: Dict[str, int] = {}
        stage_entries: Dict[str, int] = {}
        total_confidence = 0.0

        for record in self._transition_history:
            path = f"{record['from_stage']}->{record['to_stage']}"
            path_counts[path] = path_counts.get(path, 0) + 1
            stage_entries[record["to_stage"]] = stage_entries.get(record["to_stage"], 0) + 1
            total_confidence += record["confidence"]

        return {
            "total_transitions": len(self._transition_history),
            "unique_paths": len(path_counts),
            "transitions_by_path": path_counts,
            "transitions_by_stage": stage_entries,
            "average_confidence": total_confidence / len(self._transition_history),
        }

    def set_hooks(self, hooks: "StateHookManager") -> None:
        """Set or replace the hook manager.

        Args:
            hooks: StateHookManager instance.
        """
        self._hooks = hooks

    def clear_hooks(self) -> None:
        """Remove all hooks."""
        self._hooks = None

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
        current_idx = STAGE_ORDER[self.state.stage]
        for stage in ConversationStage:
            if abs(STAGE_ORDER[stage] - current_idx) <= 1:
                if tool_name in self._get_tools_for_stage(stage):
                    return True

        return False

    def get_tool_priority_boost(self, tool_name: str) -> float:
        """Get priority boost for a tool based on current stage.

        Args:
            tool_name: Name of the tool

        Returns:
            Boost value (0.0 to 0.2) to add to similarity score

        Note:
            If StageTransitionEngine is enabled, uses its detailed
            STAGE_TOOL_PRIORITIES configuration. Falls back to legacy
            logic for tools not in the engine's priorities.
        """
        # Use StageTransitionEngine's detailed priorities if available
        if self._transition_engine:
            multiplier = cast(float, self._transition_engine.get_tool_priority_multiplier(tool_name))
            # If engine has a specific multiplier (not default 1.0), use it
            if multiplier != 1.0:
                # Convert multiplier (1.0-2.0) to boost (0.0-0.2)
                # multiplier 1.0 -> boost 0.0 (no change)
                # multiplier 1.5 -> boost 0.1 (medium boost)
                # multiplier 2.0 -> boost 0.2 (high boost)
                boost = (multiplier - 1.0) * 0.2
                return max(0.0, min(0.2, boost))
            # Tool not in engine's priorities, fall through to legacy logic

        # Legacy boost logic (for tools not in engine or when engine is disabled)
        if tool_name in self.get_stage_tools():
            return 0.15  # High boost for stage-relevant tools

        # Check adjacent stages
        current_idx = STAGE_ORDER[self.state.stage]
        for stage in ConversationStage:
            if abs(STAGE_ORDER[stage] - current_idx) == 1:
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
