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

"""State Coordinator - Unified state management for agent orchestrator.

This module provides a centralized coordinator for all state management,
consolidating access to:

- SessionStateManager: Execution state (tool calls, files, budget)
- ConversationStateMachine: Conversation stage and flow
- Checkpoint state: Serialization/deserialization

The StateCoordinator implements the Observer pattern for state change
notifications, allowing components to react to state transitions.

Design Principles:
- Single Responsibility: Coordinate state access and notifications only
- Interface Segregation: Focused protocols for state operations
- Dependency Inversion: Depend on protocols, not concrete implementations
- Observer Pattern: State change notifications for loose coupling

Usage:
    coordinator = StateCoordinator(
        session_state_manager=session_state,
        conversation_state_machine=conversation_state,
    )

    # Get comprehensive state
    state = coordinator.get_state()

    # Restore state
    coordinator.set_state(restored_state)

    # Transition to new stage
    coordinator.transition_to(ConversationStage.EXECUTION)

    # Subscribe to state changes
    @coordinator.on_state_change
    def handle_change(old_state, new_state):
        print(f"State changed: {old_state} -> {new_state}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
from threading import Lock

if TYPE_CHECKING:
    from victor.agent.session_state_manager import SessionStateManager
    from victor.agent.conversation_state import ConversationStateMachine
    from victor.core.state import ConversationStage

logger = logging.getLogger(__name__)


class StateScope(str, Enum):
    """Scope of state access.

    SESSION: Session-level state (tool calls, files, budget)
    CONVERSATION: Conversation-level state (stage, message history)
    CHECKPOINT: Checkpoint/restore state
    ALL: All state scopes
    """

    SESSION = "session"
    CONVERSATION = "conversation"
    CHECKPOINT = "checkpoint"
    ALL = "all"


@dataclass
class StateChange:
    """Represents a state change event.

    Attributes:
        scope: The scope of the state change
        old_state: Previous state snapshot
        new_state: New state snapshot
        changes: Dict of specific fields that changed
        timestamp: Unix timestamp of the change
    """

    scope: StateScope
    old_state: Dict[str, Any]
    new_state: Dict[str, Any]
    changes: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scope": self.scope.value,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "changes": self.changes,
            "timestamp": self.timestamp,
        }


# Type alias for state change observers
StateObserver = Callable[[StateChange], None]


class StateCoordinator:
    """Unified coordinator for agent state management.

    This coordinator provides a single entry point for all state operations,
    consolidating access to SessionStateManager, ConversationStateMachine,
    and checkpoint state.

    Responsibilities:
    - Get/set comprehensive state across all scopes
    - Delegate state operations to appropriate managers
    - Emit state change notifications (Observer pattern)
    - Track state history for debugging
    - Provide checkpoint serialization

    Thread-safety: State change notifications are thread-safe. State
    operations themselves depend on the underlying managers' thread-safety.
    """

    def __init__(
        self,
        session_state_manager: SessionStateManager,
        conversation_state_machine: Optional[ConversationStateMachine] = None,
        enable_history: bool = True,
        max_history_size: int = 100,
    ):
        """Initialize the state coordinator.

        Args:
            session_state_manager: Manager for session execution state
            conversation_state_machine: Optional manager for conversation stage
            enable_history: Whether to track state change history
            max_history_size: Maximum number of state changes to track
        """
        self._session_state = session_state_manager
        self._conversation_state = conversation_state_machine

        self._enable_history = enable_history
        self._max_history_size = max_history_size
        self._state_history: List[StateChange] = []

        # Observer pattern: list of callbacks for state changes
        self._observers: List[StateObserver] = []
        self._observer_lock = Lock()

        logger.debug("StateCoordinator initialized")

    # ========================================================================
    # State Access
    # ========================================================================

    def get_state(
        self, scope: StateScope = StateScope.ALL, include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Get current state across specified scopes.

        Args:
            scope: Which state scopes to include
            include_metadata: Whether to include metadata like timestamps

        Returns:
            Dictionary containing the requested state
        """
        result: Dict[str, Any] = {}

        if include_metadata:
            result["_metadata"] = {
                "scope": scope.value,
                "timestamp": __import__("time").time(),
            }

        if scope in (StateScope.SESSION, StateScope.ALL):
            result["session"] = self._session_state.get_checkpoint_state()

        if scope in (StateScope.CONVERSATION, StateScope.ALL) and self._conversation_state:
            result["conversation"] = self._conversation_state.to_dict()

        if scope in (StateScope.CHECKPOINT, StateScope.ALL):
            result["checkpoint"] = self._get_checkpoint_state()

        return result

    def set_state(self, state: Dict[str, Any], scope: StateScope = StateScope.ALL) -> None:
        """Restore state from a dictionary.

        Args:
            state: State dictionary from get_state()
            scope: Which scopes to restore
        """
        old_state = self.get_state(scope=scope, include_metadata=False)

        if scope in (StateScope.SESSION, StateScope.ALL):
            if "session" in state:
                self._session_state.apply_checkpoint_state(state["session"])

        if scope in (StateScope.CONVERSATION, StateScope.ALL):
            if "conversation" in state and self._conversation_state:
                # Restore conversation state machine
                from victor.agent.conversation_state import (
                    ConversationStateMachine,
                )

                try:
                    restored = ConversationStateMachine.from_dict(state["conversation"])
                    # Sync state to current machine
                    self._conversation_state.state = restored.state
                    if hasattr(self._conversation_state, "_sync_state_to_manager"):
                        self._conversation_state._sync_state_to_manager()
                except Exception as e:
                    logger.warning(f"Failed to restore conversation state: {e}")

        if scope in (StateScope.CHECKPOINT, StateScope.ALL):
            if "checkpoint" in state:
                self._apply_checkpoint_state(state["checkpoint"])

        new_state = self.get_state(scope=scope, include_metadata=False)

        # Emit state change notification
        self._notify_observers(
            StateChange(
                scope=scope,
                old_state=old_state,
                new_state=new_state,
                changes=self._diff_changes(old_state, new_state),
            )
        )

        logger.debug(f"State restored for scope: {scope.value}")

    # ========================================================================
    # State Transitions (Conversation Stage)
    # ========================================================================

    def transition_to(self, stage: Any) -> bool:
        """Transition to a new conversation stage.

        Args:
            stage: New stage (ConversationStage enum or string)

        Returns:
            True if transition was successful
        """
        if not self._conversation_state:
            logger.debug("Cannot transition - no conversation state machine")
            return False

        # Import ConversationStage for validation
        from victor.core.state import ConversationStage

        # Convert string to enum if needed
        if isinstance(stage, str):
            try:
                stage = ConversationStage[stage.upper()]
            except KeyError:
                logger.warning(f"Invalid stage name: {stage}")
                return False

        old_stage = self._conversation_state.get_stage()

        # Use internal transition method (high confidence to force transition)
        if hasattr(self._conversation_state, "_transition_to"):
            self._conversation_state._transition_to(stage, confidence=1.0)
        else:
            logger.debug("ConversationStateMachine does not support direct transition")
            return False

        new_stage = self._conversation_state.get_stage()

        # Emit notification if stage changed
        if old_stage != new_stage:
            self._notify_observers(
                StateChange(
                    scope=StateScope.CONVERSATION,
                    old_state={"stage": old_stage.name},
                    new_state={"stage": new_stage.name},
                    changes={"stage": (old_stage.name, new_stage.name)},
                )
            )

        return True

    def get_stage(self) -> Optional[str]:
        """Get the current conversation stage.

        Returns:
            Stage name or None if no conversation state machine
        """
        if not self._conversation_state:
            return None
        return self._conversation_state.get_stage().name

    def get_stage_tools(self) -> Set[str]:
        """Get tools relevant to the current stage.

        Returns:
            Set of tool names relevant to current stage
        """
        if not self._conversation_state:
            return set()
        return self._conversation_state.get_stage_tools()

    # ========================================================================
    # Session State Delegation
    # ========================================================================

    @property
    def tool_calls_used(self) -> int:
        """Get number of tool calls used."""
        return self._session_state.tool_calls_used

    @property
    def observed_files(self) -> Set[str]:
        """Get set of observed files."""
        return self._session_state.observed_files

    @property
    def executed_tools(self) -> List[str]:
        """Get list of executed tools."""
        return self._session_state.executed_tools

    @property
    def tool_budget(self) -> int:
        """Get tool budget."""
        return self._session_state.tool_budget

    @tool_budget.setter
    def tool_budget(self, value: int) -> None:
        """Set tool budget."""
        old_budget = self._session_state.tool_budget
        self._session_state.tool_budget = value

        # Emit state change notification
        self._notify_observers(
            StateChange(
                scope=StateScope.SESSION,
                old_state={"tool_budget": old_budget},
                new_state={"tool_budget": value},
                changes={"tool_budget": (old_budget, value)},
            )
        )

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted."""
        return self._session_state.is_budget_exhausted()

    def get_remaining_budget(self) -> int:
        """Get remaining tool budget."""
        return self._session_state.get_remaining_budget()

    def record_tool_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool call execution.

        Args:
            tool_name: Name of the tool being called
            args: Tool arguments dictionary
        """
        self._session_state.record_tool_call(tool_name, args)

        # Also record in conversation state if available
        if self._conversation_state:
            self._conversation_state.record_tool_execution(tool_name, args)

    def increment_tool_calls(self, count: int = 1) -> int:
        """Increment the tool calls counter.

        Args:
            count: Number of calls to add

        Returns:
            New total tool calls count
        """
        old_count = self._session_state.tool_calls_used
        new_count = self._session_state.increment_tool_calls(count)

        # Emit state change notification
        if new_count != old_count:
            self._notify_observers(
                StateChange(
                    scope=StateScope.SESSION,
                    old_state={"tool_calls_used": old_count},
                    new_state={"tool_calls_used": new_count},
                    changes={"tool_calls_used": (old_count, new_count)},
                )
            )

        return new_count

    def record_file_read(self, filepath: str) -> None:
        """Record that a file has been read.

        Args:
            filepath: Path to the file that was read
        """
        self._session_state.record_file_read(filepath)

    # ========================================================================
    # Observer Pattern - State Change Notifications
    # ========================================================================

    def subscribe(self, observer: StateObserver) -> Callable[[], None]:
        """Subscribe to state change notifications.

        Args:
            observer: Callback function that receives StateChange

        Returns:
            Unsubscribe function that removes the observer
        """
        with self._observer_lock:
            self._observers.append(observer)

        def unsubscribe() -> None:
            with self._observer_lock:
                if observer in self._observers:
                    self._observers.remove(observer)

        return unsubscribe

    def unsubscribe(self, observer: StateObserver) -> None:
        """Unsubscribe an observer from state change notifications.

        Args:
            observer: Callback function to remove
        """
        with self._observer_lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def unsubscribe_all(self) -> None:
        """Remove all observers."""
        with self._observer_lock:
            self._observers.clear()

    def _notify_observers(self, change: StateChange) -> None:
        """Notify all observers of a state change.

        Args:
            change: StateChange event to emit
        """
        # Record in history if enabled
        if self._enable_history:
            self._state_history.append(change)
            if len(self._state_history) > self._max_history_size:
                self._state_history.pop(0)

        # Notify observers (thread-safe copy)
        with self._observer_lock:
            observers = list(self._observers)

        for observer in observers:
            try:
                observer(change)
            except Exception as e:
                logger.warning(f"State observer error: {e}")

    # Decorator for observer registration
    def on_state_change(self, func: StateObserver) -> StateObserver:
        """Decorator to register a state change observer.

        Usage:
            @coordinator.on_state_change
            def handle_change(change: StateChange):
                print(f"State changed: {change.scope}")
        """
        self.subscribe(func)
        return func

    # ========================================================================
    # State History
    # ========================================================================

    def get_state_history(self, limit: Optional[int] = None) -> List[StateChange]:
        """Get history of state changes.

        Args:
            limit: Maximum number of changes to return (all if None)

        Returns:
            List of StateChange events, most recent last
        """
        if limit is None:
            return list(self._state_history)
        return self._state_history[-limit:]

    def get_state_changes_count(self) -> int:
        """Get total number of state changes recorded.

        Returns:
            Number of state changes in history
        """
        return len(self._state_history)

    def clear_state_history(self) -> None:
        """Clear state change history."""
        self._state_history.clear()

    # ========================================================================
    # Checkpoint State
    # ========================================================================

    def _get_checkpoint_state(self) -> Dict[str, Any]:
        """Build checkpoint state for serialization.

        Returns:
            Dictionary with state for checkpointing
        """
        return {
            "stage": self.get_stage(),
            "tool_history": list(self.executed_tools),
            "observed_files": list(self.observed_files),
            "tool_calls_used": self.tool_calls_used,
            "tool_budget": self.tool_budget,
        }

    def _apply_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Apply checkpoint state to restore.

        Args:
            state: State dictionary from checkpoint
        """
        # Restore execution tracking
        self._session_state.execution_state.executed_tools = list(state.get("tool_history", []))
        self._session_state.execution_state.observed_files = set(state.get("observed_files", []))
        self._session_state.execution_state.tool_calls_used = state.get("tool_calls_used", 0)

        # Restore stage if present
        stage_name = state.get("stage", "INITIAL")
        if self._conversation_state:
            from victor.core.state import ConversationStage

            try:
                stage = ConversationStage[stage_name]
                # Use internal _transition_to method with high confidence
                if hasattr(self._conversation_state, "_transition_to"):
                    self._conversation_state._transition_to(stage, confidence=1.0)
            except (KeyError, AttributeError):
                logger.debug(f"Could not restore stage: {stage_name}")

    # ========================================================================
    # State Summary
    # ========================================================================

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of current state.

        Returns:
            Dictionary with state summary across all scopes
        """
        summary = {
            "session": self._session_state.get_session_summary(),
        }

        if self._conversation_state:
            conv_summary: Dict[str, Any] = (
                self._conversation_state.get_state_summary()
                if callable(self._conversation_state.get_state_summary)
                else {}
            )
            summary["conversation"] = conv_summary

        summary["state_changes_count"] = self.get_state_changes_count()

        return summary

    # ========================================================================
    # Utilities
    # ========================================================================

    @staticmethod
    def _diff_changes(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between two state dictionaries.

        Args:
            old: Old state dictionary
            new: New state dictionary

        Returns:
            Dictionary of changed fields with (old, new) tuples
        """
        changes: Dict[str, Any] = {}

        # Check for modified values
        for key, new_value in new.items():
            if key not in old:
                changes[key] = (None, new_value)
            elif old[key] != new_value:
                changes[key] = (old[key], new_value)

        # Check for removed keys
        for key in old:
            if key not in new:
                changes[key] = (old[key], None)

        return changes

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"StateCoordinator("
            f"stage={self.get_stage()}, "
            f"tool_calls={self.tool_calls_used}/{self.tool_budget}, "
            f"observers={len(self._observers)})"
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_state_coordinator(
    session_state_manager: SessionStateManager,
    conversation_state_machine: Optional[ConversationStateMachine] = None,
    enable_history: bool = True,
    max_history_size: int = 100,
) -> StateCoordinator:
    """Factory function to create a StateCoordinator.

    Args:
        session_state_manager: Manager for session execution state
        conversation_state_machine: Optional manager for conversation stage
        enable_history: Whether to track state change history
        max_history_size: Maximum number of state changes to track

    Returns:
        Configured StateCoordinator instance
    """
    return StateCoordinator(
        session_state_manager=session_state_manager,
        conversation_state_machine=conversation_state_machine,
        enable_history=enable_history,
        max_history_size=max_history_size,
    )


__all__ = [
    "StateCoordinator",
    "StateScope",
    "StateChange",
    "StateObserver",
    "create_state_coordinator",
]
