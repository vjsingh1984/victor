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

"""Checkpoint coordinator for agent orchestration.

This module provides the CheckpointCoordinator which handles all
checkpoint-related operations for the orchestrator including:

- Manual checkpoint saving and restoration
- Automatic checkpointing based on tool execution intervals
- State serialization and deserialization
- Time-travel debugging capabilities

Extracted from AgentOrchestrator as part of SOLID refactoring
to improve modularity and testability.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from victor.storage.checkpoints.manager import ConversationCheckpointManager

logger = logging.getLogger(__name__)


class CheckpointCoordinator:
    """Coordinates checkpoint operations for time-travel debugging.

    This coordinator wraps the ConversationCheckpointManager and provides
    a clean interface for:
    - Manual checkpoint creation with descriptions and tags
    - Checkpoint restoration with state merging
    - Automatic checkpointing at tool execution intervals
    - State serialization and deserialization

    The coordinator uses callback functions for state operations to
    maintain loose coupling with the orchestrator.

    Example:
        coordinator = CheckpointCoordinator(
            checkpoint_manager=manager,
            session_id="session_123",
            get_state_fn=lambda: orchestrator._get_checkpoint_state(),
            apply_state_fn=lambda state: orchestrator._apply_checkpoint_state(state),
        )

        # Manual checkpoint
        cp_id = await coordinator.save_checkpoint(
            description="Before refactoring"
        )

        # Later, restore
        success = await coordinator.restore_checkpoint(cp_id)
    """

    def __init__(
        self,
        checkpoint_manager: Optional["ConversationCheckpointManager"],
        session_id: Optional[str],
        get_state_fn: Callable[[], dict[str, Any]],
        apply_state_fn: Callable[[dict[str, Any]], None],
    ) -> None:
        """Initialize the checkpoint coordinator.

        Args:
            checkpoint_manager: The checkpoint manager instance (can be None if disabled)
            session_id: Current session identifier
            get_state_fn: Function to serialize current state to dict
            apply_state_fn: Function to apply restored state dict to orchestrator
        """
        self._checkpoint_manager = checkpoint_manager
        self._session_id = session_id
        self._get_state_fn = get_state_fn
        self._apply_state_fn = apply_state_fn

    @property
    def checkpoint_manager(self) -> Optional["ConversationCheckpointManager"]:
        """Get the underlying checkpoint manager.

        Returns:
            ConversationCheckpointManager instance or None if disabled
        """
        return self._checkpoint_manager

    @property
    def is_enabled(self) -> bool:
        """Check if checkpointing is enabled.

        Returns:
            True if checkpoint manager is initialized
        """
        return self._checkpoint_manager is not None

    async def save_checkpoint(
        self,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Save a manual checkpoint of the current conversation state.

        Args:
            description: Human-readable description of the checkpoint
            tags: Optional tags for categorization

        Returns:
            Checkpoint ID if saved, None if checkpointing is disabled

        Raises:
            Exception: If checkpoint serialization fails (logged and returns None)
        """
        if not self._checkpoint_manager:
            logger.debug("Checkpoint save skipped - manager not initialized")
            return None

        # Build conversation state for checkpointing
        state = self._get_state_fn()

        try:
            checkpoint_id = await self._checkpoint_manager.save_checkpoint(
                session_id=self._session_id or "default",
                state=state,
                description=description,
                tags=tags or [],
            )
            logger.info(
                f"Checkpoint saved: {checkpoint_id[:20]}... ({description or 'no description'})"
            )
            return checkpoint_id
        except (TypeError, ValueError) as e:
            # Serialization error - state contains non-serializable objects
            logger.warning(f"Failed to save checkpoint (serialization error): {e}")
            return None

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore conversation state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if restored successfully, False otherwise

        Raises:
            OSError/IOError: If checkpoint data is invalid (caught and logged)
        """
        if not self._checkpoint_manager:
            logger.warning("Cannot restore - checkpoint manager not initialized")
            return False

        try:
            state = await self._checkpoint_manager.restore_checkpoint(checkpoint_id)
            self._apply_state_fn(state)
            logger.info(f"Restored checkpoint: {checkpoint_id[:20]}...")
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to restore checkpoint (I/O error): {e}")
            return False
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to restore checkpoint (invalid data): {e}")
            return False

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Trigger auto-checkpoint if interval threshold is met.

        This should be called after tool executions to maintain
        automatic checkpoints at regular intervals.

        Returns:
            Checkpoint ID if auto-checkpoint was created, None otherwise

        Raises:
            Exception: If checkpoint serialization fails (logged and returns None)
        """
        if not self._checkpoint_manager:
            return None

        state = self._get_state_fn()

        try:
            return await self._checkpoint_manager.maybe_auto_checkpoint(
                session_id=self._session_id or "default",
                state=state,
            )
        except (TypeError, ValueError) as e:
            logger.debug(f"Auto-checkpoint failed (serialization error): {e}")
            return None

    def update_session_id(self, session_id: Optional[str]) -> None:
        """Update the session ID for checkpoint operations.

        Args:
            session_id: New session identifier
        """
        self._session_id = session_id
