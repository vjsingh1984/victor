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

"""State Coordinator - Coordinates conversation state and stage transitions.

This module extracts state management logic from AgentOrchestrator,
providing a focused interface for:
- Conversation stage tracking (INITIAL -> PLANNING -> READING -> etc.)
- Stage transitions with validation
- Message history management
- State persistence and recovery

Design Philosophy:
- Single Responsibility: Coordinates all state-related operations
- Observable: Emits stage transition events
- Composable: Works with ConversationController and ConversationStateMachine
- Backward Compatible: Maintains API compatibility with orchestrator

Usage:
    coordinator = StateCoordinator(
        conversation_controller=controller,
        state_machine=machine,
    )

    # Get current stage
    stage = coordinator.get_current_stage()

    # Transition to new stage
    success = coordinator.transition_to(ConversationStage.EXECUTION)

    # Get message history
    messages = coordinator.get_message_history()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

from victor.agent.conversation_state import (
    ConversationStage,
    ConversationState,
    STAGE_ORDER,
)

if TYPE_CHECKING:
    from victor.agent.conversation_controller import ConversationController
    from victor.agent.conversation_state import ConversationStateMachine
    from victor.providers.base import Message

logger = logging.getLogger(__name__)


@dataclass
class StateCoordinatorConfig:
    """Configuration for StateCoordinator.

    Attributes:
        enable_auto_transitions: Whether to auto-detect stage transitions
        enable_history_tracking: Whether to track transition history
        max_history_length: Maximum transitions to keep in history
        emit_events: Whether to emit stage change events
    """

    enable_auto_transitions: bool = True
    enable_history_tracking: bool = True
    max_history_length: int = 100
    emit_events: bool = True


@dataclass
class StageTransition:
    """Record of a stage transition.

    Attributes:
        from_stage: Stage before transition
        to_stage: Stage after transition
        reason: Reason for the transition
        tool_name: Tool that triggered transition (if any)
        confidence: Confidence score for the transition
    """

    from_stage: ConversationStage
    to_stage: ConversationStage
    reason: str = ""
    tool_name: Optional[str] = None
    confidence: float = 1.0


@runtime_checkable
class IStateCoordinator(Protocol):
    """Protocol for state coordination operations."""

    def get_current_stage(self) -> ConversationStage: ...
    def transition_to(self, stage: ConversationStage, reason: str = "") -> bool: ...
    def get_message_history(self) -> List[Any]: ...


class StateCoordinator:
    """Coordinates conversation state and stage transitions.

    This class consolidates state management operations that were spread
    across the orchestrator, providing a unified interface for:

    1. Stage Management: Tracks current stage and handles transitions
    2. Message History: Provides access to conversation messages
    3. State Persistence: Serializes/deserializes state for recovery
    4. Transition History: Records stage changes for debugging

    Example:
        coordinator = StateCoordinator(
            conversation_controller=controller,
            state_machine=machine,
        )

        # Check current stage
        if coordinator.get_current_stage() == ConversationStage.READING:
            # In reading phase
            pass

        # Manual transition
        coordinator.transition_to(
            ConversationStage.EXECUTION,
            reason="User requested code changes"
        )

        # Get recent messages
        recent = coordinator.get_recent_messages(limit=10)
    """

    def __init__(
        self,
        conversation_controller: "ConversationController",
        state_machine: Optional["ConversationStateMachine"] = None,
        config: Optional[StateCoordinatorConfig] = None,
        on_stage_change: Optional[Callable[[ConversationStage, ConversationStage], None]] = None,
    ) -> None:
        """Initialize the StateCoordinator.

        Args:
            conversation_controller: Controller for conversation management
            state_machine: Optional state machine for stage detection
            config: Configuration options
            on_stage_change: Callback when stage changes (old, new)
        """
        self._controller = conversation_controller
        self._state_machine = state_machine
        self._config = config or StateCoordinatorConfig()
        self._on_stage_change = on_stage_change

        # Transition history
        self._transition_history: List[StageTransition] = []

        # Track last known stage for change detection
        self._last_stage: Optional[ConversationStage] = None

        logger.debug(
            f"StateCoordinator initialized with auto_transitions={self._config.enable_auto_transitions}"
        )

    @property
    def stage(self) -> ConversationStage:
        """Get the current conversation stage."""
        return self.get_current_stage()

    @property
    def message_count(self) -> int:
        """Get the number of messages in history."""
        return self._controller.message_count

    @property
    def transition_count(self) -> int:
        """Get the number of recorded transitions."""
        return len(self._transition_history)

    def get_current_stage(self) -> ConversationStage:
        """Get the current conversation stage.

        Returns:
            Current ConversationStage
        """
        if self._state_machine:
            return self._state_machine.get_stage()
        return self._controller.stage

    def transition_to(
        self,
        stage: ConversationStage,
        reason: str = "",
        tool_name: Optional[str] = None,
        confidence: float = 1.0,
    ) -> bool:
        """Transition to a new conversation stage.

        Validates the transition and records it in history.

        Args:
            stage: Target stage to transition to
            reason: Reason for the transition
            tool_name: Tool that triggered the transition
            confidence: Confidence score for the transition

        Returns:
            True if transition was successful
        """
        current = self.get_current_stage()

        if current == stage:
            logger.debug(f"Already in stage {stage.value}, no transition needed")
            return True

        # Record transition
        transition = StageTransition(
            from_stage=current,
            to_stage=stage,
            reason=reason,
            tool_name=tool_name,
            confidence=confidence,
        )

        # Perform transition
        success = self._perform_transition(stage)

        if success:
            # Track transition
            if self._config.enable_history_tracking:
                self._transition_history.append(transition)
                if len(self._transition_history) > self._config.max_history_length:
                    self._transition_history.pop(0)

            # Notify callback
            if self._on_stage_change and self._config.emit_events:
                self._on_stage_change(current, stage)

            self._last_stage = stage
            logger.debug(f"Transitioned {current.value} -> {stage.value}: {reason}")

        return success

    def _perform_transition(self, stage: ConversationStage) -> bool:
        """Perform the actual stage transition.

        Args:
            stage: Target stage

        Returns:
            True if successful
        """
        try:
            if self._state_machine:
                self._state_machine.set_stage(stage)
            else:
                # Direct update on controller state
                if hasattr(self._controller, "_state_machine"):
                    self._controller._state_machine.set_stage(stage)
            return True
        except Exception as e:
            logger.warning(f"Failed to transition to {stage.value}: {e}")
            return False

    def get_message_history(self) -> List["Message"]:
        """Get the full message history.

        Returns:
            List of Message objects
        """
        return self._controller.messages

    def get_recent_messages(
        self,
        limit: int = 10,
        include_system: bool = False,
    ) -> List["Message"]:
        """Get recent messages from history.

        Args:
            limit: Maximum messages to return
            include_system: Whether to include system messages

        Returns:
            List of recent Message objects
        """
        messages = self._controller.messages
        if not include_system:
            messages = [m for m in messages if m.role != "system"]
        return messages[-limit:] if limit < len(messages) else messages

    def get_last_user_message(self) -> Optional["Message"]:
        """Get the most recent user message.

        Returns:
            Last user message or None
        """
        for msg in reversed(self._controller.messages):
            if msg.role == "user":
                return msg
        return None

    def get_last_assistant_message(self) -> Optional["Message"]:
        """Get the most recent assistant message.

        Returns:
            Last assistant message or None
        """
        for msg in reversed(self._controller.messages):
            if msg.role == "assistant":
                return msg
        return None

    def get_transition_history(
        self,
        limit: Optional[int] = None,
    ) -> List[StageTransition]:
        """Get stage transition history.

        Args:
            limit: Maximum transitions to return (None for all)

        Returns:
            List of StageTransition records
        """
        if limit is None:
            return list(self._transition_history)
        return self._transition_history[-limit:]

    def get_time_in_stage(self, stage: ConversationStage) -> int:
        """Count how many transitions included a specific stage.

        Args:
            stage: Stage to count

        Returns:
            Number of times this stage was visited
        """
        count = 0
        for transition in self._transition_history:
            if transition.to_stage == stage:
                count += 1
        return count

    def get_stage_sequence(self) -> List[ConversationStage]:
        """Get the sequence of stages visited.

        Returns:
            List of stages in order of visitation
        """
        if not self._transition_history:
            return [self.get_current_stage()]

        stages = [self._transition_history[0].from_stage]
        for transition in self._transition_history:
            stages.append(transition.to_stage)
        return stages

    def is_in_exploration_phase(self) -> bool:
        """Check if currently in exploration phase (INITIAL, PLANNING, READING).

        Returns:
            True if in exploration phase
        """
        stage = self.get_current_stage()
        return stage in {
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
        }

    def is_in_execution_phase(self) -> bool:
        """Check if currently in execution phase.

        Returns:
            True if in execution phase
        """
        return self.get_current_stage() == ConversationStage.EXECUTION

    def is_in_completion_phase(self) -> bool:
        """Check if currently in completion phase.

        Returns:
            True if in completion or verification phase
        """
        stage = self.get_current_stage()
        return stage in {
            ConversationStage.VERIFICATION,
            ConversationStage.COMPLETION,
        }

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current state for persistence.

        Returns:
            Dict containing serialized state
        """
        return {
            "stage": self.get_current_stage().value,
            "message_count": self.message_count,
            "transition_count": self.transition_count,
            "last_transitions": [
                {
                    "from": t.from_stage.value,
                    "to": t.to_stage.value,
                    "reason": t.reason,
                    "tool": t.tool_name,
                }
                for t in self._transition_history[-10:]
            ],
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> bool:
        """Restore state from a snapshot.

        Args:
            snapshot: Previously saved state snapshot

        Returns:
            True if restoration was successful
        """
        try:
            stage_str = snapshot.get("stage", "initial")
            stage = ConversationStage(stage_str)
            return self.transition_to(stage, reason="State restoration")
        except Exception as e:
            logger.warning(f"Failed to restore state: {e}")
            return False

    def clear_history(self) -> None:
        """Clear transition history."""
        self._transition_history.clear()
        logger.debug("Transition history cleared")


def create_state_coordinator(
    conversation_controller: "ConversationController",
    state_machine: Optional["ConversationStateMachine"] = None,
    config: Optional[StateCoordinatorConfig] = None,
) -> StateCoordinator:
    """Factory function to create a StateCoordinator.

    Args:
        conversation_controller: Controller for conversation management
        state_machine: Optional state machine for stage detection
        config: Configuration options

    Returns:
        Configured StateCoordinator instance
    """
    return StateCoordinator(
        conversation_controller=conversation_controller,
        state_machine=state_machine,
        config=config,
    )


__all__ = [
    "StateCoordinator",
    "StateCoordinatorConfig",
    "StageTransition",
    "IStateCoordinator",
    "create_state_coordinator",
]
