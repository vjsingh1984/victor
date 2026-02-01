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

"""Stage transition engine for conversation flow management.

Phase 2.2: Extract StageTransitionEngine.

This module provides centralized stage transition management with:
- Valid transition graph (INITIAL → PLANNING → READING → ...)
- Stage-based tool priority multipliers
- Transition callbacks for coordination
- Event emission on transitions
- Cooldown mechanism to prevent thrashing

Usage:
    from victor.agent.stage_transition_engine import StageTransitionEngine
    from victor.core.state import ConversationStage

    engine = StageTransitionEngine()

    # Check if transition is valid
    if engine.can_transition(ConversationStage.PLANNING):
        engine.transition_to(ConversationStage.PLANNING)

    # Get tool priority for current stage
    multiplier = engine.get_tool_priority_multiplier("read")

    # Register callback for transitions
    engine.register_callback(lambda old, new: print(f"{old} -> {new}"))
"""

from __future__ import annotations

import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)
from collections.abc import Callable

from victor.core.state import ConversationStage
from victor.agent.protocols import StageTransitionProtocol

if TYPE_CHECKING:
    from victor.core.events import ObservabilityBus

logger = logging.getLogger(__name__)


# Stage ordering for adjacency calculations
STAGE_ORDER: dict[ConversationStage, int] = {
    stage: idx for idx, stage in enumerate(ConversationStage)
}

# Transition graph: defines valid FORWARD transitions from each stage
# Backward transitions require high confidence and are checked separately
# Format: stage -> set of valid forward target stages
TRANSITION_GRAPH: dict[ConversationStage, set[ConversationStage]] = {
    ConversationStage.INITIAL: {
        ConversationStage.PLANNING,
        ConversationStage.READING,  # Allow skipping planning for simple tasks
    },
    ConversationStage.PLANNING: {
        ConversationStage.READING,
        ConversationStage.ANALYSIS,  # Allow direct to analysis
    },
    ConversationStage.READING: {
        ConversationStage.ANALYSIS,
        ConversationStage.EXECUTION,  # Allow direct to execution for quick fixes
    },
    ConversationStage.ANALYSIS: {
        ConversationStage.EXECUTION,
    },
    ConversationStage.EXECUTION: {
        ConversationStage.VERIFICATION,
    },
    ConversationStage.VERIFICATION: {
        ConversationStage.COMPLETION,
    },
    ConversationStage.COMPLETION: set(),  # Terminal state
}

# Tool priority multipliers by stage
# Format: stage -> {tool_name: multiplier}
STAGE_TOOL_PRIORITIES: dict[ConversationStage, dict[str, float]] = {
    ConversationStage.INITIAL: {
        "search": 1.2,
        "overview": 1.3,
        "ls": 1.2,
    },
    ConversationStage.PLANNING: {
        "search": 1.4,
        "overview": 1.4,
        "ls": 1.3,
        "read": 1.2,
    },
    ConversationStage.READING: {
        "read": 1.5,
        "grep": 1.3,
        "search": 1.2,
    },
    ConversationStage.ANALYSIS: {
        "read": 1.3,
        "grep": 1.4,
        "search": 1.2,
    },
    ConversationStage.EXECUTION: {
        "edit": 1.5,
        "write": 1.5,
        "bash": 1.3,
        "read": 1.1,
    },
    ConversationStage.VERIFICATION: {
        "bash": 1.5,
        "read": 1.3,
        "grep": 1.2,
    },
    ConversationStage.COMPLETION: {
        "bash": 1.2,  # For final commands like git commit
    },
}


class StageTransitionEngine(StageTransitionProtocol):
    """Engine for managing conversation stage transitions.

    Provides centralized stage transition logic with:
    - Validated transition graph
    - Cooldown mechanism to prevent thrashing
    - Callback support for coordination
    - Event emission for observability
    - Tool priority multipliers per stage

    Attributes:
        current_stage: Current conversation stage
        cooldown_seconds: Minimum time between transitions
        transition_history: List of transition records
    """

    # Default cooldown to prevent stage thrashing (2.0 seconds)
    DEFAULT_COOLDOWN_SECONDS: float = 2.0

    # Confidence threshold for backward transitions
    BACKWARD_TRANSITION_THRESHOLD: float = 0.85

    def __init__(
        self,
        initial_stage: ConversationStage = ConversationStage.INITIAL,
        cooldown_seconds: Optional[float] = None,
        event_bus: Optional["ObservabilityBus"] = None,
    ) -> None:
        """Initialize the stage transition engine.

        Args:
            initial_stage: Starting stage (defaults to INITIAL)
            cooldown_seconds: Minimum time between transitions
            event_bus: Optional event bus for transition events
        """
        self._current_stage = initial_stage
        self._cooldown_seconds = (
            cooldown_seconds if cooldown_seconds is not None else self.DEFAULT_COOLDOWN_SECONDS
        )
        self._event_bus = event_bus
        self._last_transition_time: float = 0.0
        self._callbacks: list[Callable[[ConversationStage, ConversationStage], None]] = []
        self._transition_history: list[dict[str, Any]] = []
        self._transition_count: int = 0

    @property
    def current_stage(self) -> ConversationStage:
        """Get the current conversation stage."""
        return self._current_stage

    @property
    def cooldown_seconds(self) -> float:
        """Get the cooldown period between transitions."""
        return self._cooldown_seconds

    @property
    def transition_history(self) -> list[dict[str, Any]]:
        """Get the transition history."""
        return list(self._transition_history)

    @property
    def transition_count(self) -> int:
        """Get the total number of transitions."""
        return self._transition_count

    @property
    def transition_graph(self) -> dict[ConversationStage, set[ConversationStage]]:
        """Get the transition graph (for testing/inspection)."""
        return TRANSITION_GRAPH

    def can_transition(
        self,
        target_stage: ConversationStage,
        confidence: float = 0.5,
    ) -> bool:
        """Check if transition to target stage is valid.

        A transition is valid if:
        1. Target is in the valid transitions for current stage, OR
        2. It's a backward transition with high confidence (>= 0.85)

        Args:
            target_stage: Stage to check
            confidence: Confidence level for backward transitions

        Returns:
            True if transition is valid
        """
        # Same stage is always valid (no-op)
        if target_stage == self._current_stage:
            return True

        # Check if target is a valid forward transition
        valid_targets = TRANSITION_GRAPH.get(self._current_stage, set())
        if target_stage in valid_targets:
            return True

        # Check if it's a backward transition with high confidence
        current_order = STAGE_ORDER[self._current_stage]
        target_order = STAGE_ORDER[target_stage]

        if target_order < current_order:
            # Backward transition requires high confidence
            return confidence >= self.BACKWARD_TRANSITION_THRESHOLD

        # Forward skip not in graph
        return False

    def get_valid_transitions(self) -> list[ConversationStage]:
        """Get list of valid transition targets from current stage.

        Returns:
            List of stages that can be transitioned to
        """
        valid_targets = TRANSITION_GRAPH.get(self._current_stage, set())
        return list(valid_targets)

    def transition_to(
        self,
        new_stage: ConversationStage,
        confidence: float = 0.5,
    ) -> bool:
        """Transition to a new stage.

        Args:
            new_stage: Stage to transition to
            confidence: Confidence in this transition

        Returns:
            True if transition was successful
        """
        # Same stage is no-op success
        if new_stage == self._current_stage:
            return True

        # Check if transition is valid
        if not self.can_transition(new_stage, confidence):
            logger.debug(
                f"Invalid transition: {self._current_stage.name} -> {new_stage.name} "
                f"(confidence={confidence:.2f})"
            )
            return False

        # Check cooldown
        current_time = time.time()
        time_since_last = current_time - self._last_transition_time

        if time_since_last < self._cooldown_seconds:
            logger.debug(
                f"Transition blocked by cooldown: {self._current_stage.name} -> {new_stage.name} "
                f"(waited {time_since_last:.1f}s, need {self._cooldown_seconds}s)"
            )
            return False

        # Perform transition
        old_stage = self._current_stage
        self._current_stage = new_stage
        self._last_transition_time = current_time
        self._transition_count += 1

        logger.info(
            f"Stage transition: {old_stage.name} -> {new_stage.name} "
            f"(confidence: {confidence:.2f})"
        )

        # Record in history
        self._record_transition(old_stage, new_stage, confidence, current_time)

        # Invoke callbacks
        self._invoke_callbacks(old_stage, new_stage)

        # Emit event
        self._emit_transition_event(old_stage, new_stage, confidence)

        return True

    def get_tool_priority_multiplier(self, tool_name: str) -> float:
        """Get priority multiplier for a tool based on current stage.

        Args:
            tool_name: Name of the tool

        Returns:
            Multiplier value (1.0 = no change)
        """
        stage_priorities = STAGE_TOOL_PRIORITIES.get(self._current_stage, {})
        return stage_priorities.get(tool_name, 1.0)

    def register_callback(
        self,
        callback: Callable[[ConversationStage, ConversationStage], None],
    ) -> None:
        """Register a callback to be called on stage transitions.

        Args:
            callback: Function taking (old_stage, new_stage)
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(
        self,
        callback: Callable[[ConversationStage, ConversationStage], None],
    ) -> None:
        """Unregister a previously registered callback.

        Args:
            callback: Function to unregister
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def reset(self) -> None:
        """Reset the engine to initial state."""
        self._current_stage = ConversationStage.INITIAL
        self._last_transition_time = 0.0
        self._transition_history.clear()
        self._transition_count = 0

    def _record_transition(
        self,
        old_stage: ConversationStage,
        new_stage: ConversationStage,
        confidence: float,
        timestamp: float,
    ) -> None:
        """Record a transition in history.

        Args:
            old_stage: Previous stage
            new_stage: New stage
            confidence: Transition confidence
            timestamp: Unix timestamp
        """
        from datetime import datetime

        record = {
            "from_stage": old_stage,
            "to_stage": new_stage,
            "confidence": confidence,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "transition_number": self._transition_count,
        }

        self._transition_history.append(record)

    def _invoke_callbacks(
        self,
        old_stage: ConversationStage,
        new_stage: ConversationStage,
    ) -> None:
        """Invoke all registered callbacks.

        Args:
            old_stage: Previous stage
            new_stage: New stage
        """
        for callback in self._callbacks:
            try:
                callback(old_stage, new_stage)
            except Exception as e:
                logger.warning(f"Callback error during transition: {e}")

    def _emit_transition_event(
        self,
        old_stage: ConversationStage,
        new_stage: ConversationStage,
        confidence: float,
    ) -> None:
        """Emit transition event to event bus.

        Args:
            old_stage: Previous stage
            new_stage: New stage
            confidence: Transition confidence
        """
        if not self._event_bus:
            return

        try:
            # Import helper for sync event emission
            from victor.core.events.emit_helper import emit_event_sync

            emit_event_sync(
                event_bus=self._event_bus,
                topic="state.stage_changed",
                data={
                    "old_stage": old_stage.name,
                    "new_stage": new_stage.name,
                    "confidence": confidence,
                    "transition_count": self._transition_count,
                },
                source="StageTransitionEngine",
            )
        except Exception as e:
            logger.debug(f"Failed to emit stage transition event: {e}")


__all__ = [
    "StageTransitionEngine",
    "STAGE_ORDER",
    "TRANSITION_GRAPH",
    "STAGE_TOOL_PRIORITIES",
]
