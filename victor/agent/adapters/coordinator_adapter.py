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

"""Coordinator Adapter - Coordinates checkpoint and RL operations.

This adapter extracts coordinator-related logic from the orchestrator,
providing a clean interface for checkpoint management and RL reward signaling.

Responsibilities:
- Get checkpoint state from coordinators
- Apply checkpoint state to restore orchestrator
- Send RL reward signals for learning
- Bridge orchestrator and coordinator protocols

Design Patterns:
- Adapter Pattern: Converts between orchestrator and coordinator interfaces
- Single Responsibility: Focuses only on coordinator integration
- Facade Pattern: Provides unified interface for multiple coordinators
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.streaming_controller import StreamingSession
    from victor.agent.coordinators.state_coordinator import StateCoordinator
    from victor.agent.coordinators.evaluation_coordinator import EvaluationCoordinator
    from victor.agent.conversation_controller import ConversationController

# Import StateScope at runtime for use in adapter
try:
    from victor.agent.coordinators.state_coordinator import StateScope
except ImportError:
    # Fallback for testing - create a simple type
    class StateScope:
        """Fallback StateScope for testing."""

        CHECKPOINT = "checkpoint"
        ROLLBACK = "rollback"
        SNAPSHOT = "snapshot"

logger = logging.getLogger(__name__)


class CoordinatorAdapter:
    """Adapter for coordinator integration.

    This adapter encapsulates the logic for interacting with various
    coordinators used in the orchestrator:
    - StateCoordinator: State management and checkpointing
    - EvaluationCoordinator: RL reward signaling and outcome recording
    - CheckpointCoordinator: Auto-checkpointing workflow

    The adapter provides a unified interface for coordinator operations,
    abstracting away the details of each coordinator's protocol.

    Example:
        adapter = CoordinatorAdapter(
            state_coordinator=state_coordinator,
            evaluation_coordinator=evaluation_coordinator,
            conversation_controller=conversation_controller,
        )

        # Get checkpoint state
        state = adapter.get_checkpoint_state()

        # Apply checkpoint state
        adapter.apply_checkpoint_state(state)

        # Send RL reward signal
        adapter.send_rl_reward_signal(session)
    """

    def __init__(
        self,
        state_coordinator: Optional["StateCoordinator"] = None,
        evaluation_coordinator: Optional["EvaluationCoordinator"] = None,
        conversation_controller: Optional["ConversationController"] = None,
    ):
        """Initialize the CoordinatorAdapter.

        Args:
            state_coordinator: Optional state coordinator
            evaluation_coordinator: Optional evaluation coordinator
            conversation_controller: Optional conversation controller
        """
        self._state_coordinator = state_coordinator
        self._evaluation_coordinator = evaluation_coordinator
        self._conversation_controller = conversation_controller

        logger.debug(
            f"CoordinatorAdapter initialized: "
            f"state={'enabled' if state_coordinator else 'disabled'}, "
            f"evaluation={'enabled' if evaluation_coordinator else 'disabled'}, "
            f"conversation={'enabled' if conversation_controller else 'disabled'}"
        )

    def send_rl_reward_signal(self, session: "StreamingSession") -> None:
        """Send reward signal to RL model selector for Q-value updates.

        Converts StreamingSession data into RLOutcome and updates Q-values
        based on session outcome (success, latency, throughput, tool usage).

        Delegates to EvaluationCoordinator for better modularity.

        Args:
            session: StreamingSession with outcome data
        """
        if not self._evaluation_coordinator:
            logger.debug("EvaluationCoordinator not available, skipping RL reward signal")
            return

        try:
            # Note: This must be awaited, but this is a sync method
            # The caller should use an async context or the coordinator should handle this
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, schedule the coroutine
                asyncio.create_task(self._evaluation_coordinator.send_rl_reward_signal(session))
            else:
                # We're in a sync context, run the coroutine
                loop.run_until_complete(self._evaluation_coordinator.send_rl_reward_signal(session))
            logger.debug("RL reward signal sent successfully")
        except Exception as e:
            logger.warning(f"Failed to send RL reward signal: {e}")

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Build a dictionary representing current conversation state for checkpointing.

        Delegates to StateCoordinator for unified state management and merges
        with orchestrator-specific state (modified files, message count).

        Returns:
            Dictionary with checkpoint state
        """
        if not self._state_coordinator:
            logger.debug("StateCoordinator not available, returning empty checkpoint state")
            return {}

        try:
            # Get base state from StateCoordinator
            scope_value = StateScope.CHECKPOINT if isinstance(StateScope, type) else "CHECKPOINT"
            base_state = self._state_coordinator.get_state(scope=scope_value, include_metadata=False)

            # Merge with orchestrator-specific state
            checkpoint_state = {
                **base_state.get("checkpoint", {}),
                "modified_files": list(
                    getattr(self._conversation_controller, "_modified_files", set())
                ),
                "message_count": len(
                    self._conversation_controller.conversation.messages
                    if self._conversation_controller and hasattr(self._conversation_controller, 'conversation')
                    else []
                ),
            }

            logger.debug(
                f"Checkpoint state retrieved: {len(checkpoint_state.get('modified_files', []))} "
                f"modified files, {checkpoint_state.get('message_count', 0)} messages"
            )

            return checkpoint_state
        except Exception as e:
            logger.warning(f"Failed to get checkpoint state: {e}")
            return {}

    def apply_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Apply a checkpoint state to restore the orchestrator.

        Delegates to StateCoordinator for unified state management.

        Args:
            state: State dictionary from checkpoint
        """
        if not self._state_coordinator:
            logger.debug("StateCoordinator not available, skipping checkpoint restore")
            return

        if not state:
            logger.debug("Empty checkpoint state, skipping restore")
            return

        try:
            # Build checkpoint state for StateCoordinator
            checkpoint_state = {
                "stage": state.get("stage", "INITIAL"),
                "tool_history": list(state.get("tool_history", [])),
                "observed_files": list(state.get("observed_files", [])),
                "tool_calls_used": state.get("tool_calls_used", 0),
            }

            # Delegate to StateCoordinator
            # Use string "CHECKPOINT" if StateScope is not available (testing)
            scope = StateScope.CHECKPOINT if StateScope is not None else "CHECKPOINT"
            self._state_coordinator.set_state({"checkpoint": checkpoint_state}, scope=scope)

            logger.debug(
                f"Checkpoint state applied: stage={checkpoint_state['stage']}, "
                f"{len(checkpoint_state['tool_history'])} tools"
            )
        except Exception as e:
            logger.warning(f"Failed to apply checkpoint state: {e}")

    async def record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        """Record outcome for Q-learning feedback.

        Delegates to EvaluationCoordinator for better modularity.

        Args:
            success: Whether the task was completed successfully
            quality_score: Final quality score (0.0-1.0)
            user_satisfied: Whether user seemed satisfied
            completed: Whether task reached completion
        """
        if not self._evaluation_coordinator:
            logger.debug("EvaluationCoordinator not available, skipping outcome recording")
            return

        try:
            await self._evaluation_coordinator.record_intelligent_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=user_satisfied,
                completed=completed,
            )
            logger.debug(
                f"Intelligent outcome recorded: success={success}, quality={quality_score:.2f}"
            )
        except Exception as e:
            logger.warning(f"Failed to record intelligent outcome: {e}")

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def state_coordinator(self) -> Optional["StateCoordinator"]:
        """Get the state coordinator."""
        return self._state_coordinator

    @property
    def evaluation_coordinator(self) -> Optional["EvaluationCoordinator"]:
        """Get the evaluation coordinator."""
        return self._evaluation_coordinator

    @property
    def conversation_controller(self) -> Optional["ConversationController"]:
        """Get the conversation controller."""
        return self._conversation_controller

    @property
    def is_enabled(self) -> bool:
        """Check if any coordinator is enabled."""
        return self._state_coordinator is not None or self._evaluation_coordinator is not None


__all__ = [
    "CoordinatorAdapter",
]
