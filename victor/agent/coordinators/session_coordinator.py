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

"""Session Coordinator - Manages session lifecycle and state coordination.

This module extracts session management from AgentOrchestrator into a focused
coordinator that handles:

- Session ID creation and tracking
- Session lifecycle (create, end, reset, recover)
- Memory session integration
- Checkpoint/recovery coordination
- Session statistics and summary
- Token usage tracking

Design Principles:
- Single Responsibility: Manage session lifecycle only
- Delegation: Use existing components for actual work
- Composable: Works with SessionStateManager, LifecycleManager, etc.
- Observable: Support for session state tracking

Architecture:
    SessionCoordinator sits between AgentOrchestrator and:
    - SessionStateManager: Execution state tracking
    - LifecycleManager: Session lifecycle operations
    - MemoryManager: Persistent session storage
    - ConversationCheckpointManager: Checkpoint save/restore

Usage:
    coordinator = SessionCoordinator(
        session_state_manager=session_state,
        lifecycle_manager=lifecycle,
        memory_manager=memory,
        checkpoint_manager=checkpoint,
    )

    # Create new session
    session_id = coordinator.create_session()

    # Get session info
    stats = coordinator.get_session_stats()

    # End session
    await coordinator.end_session()
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.session_state_manager import SessionStateManager
    from victor.agent.lifecycle_manager import LifecycleManager

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a session.

    Attributes:
        session_id: Unique identifier for the session
        created_at: Timestamp when session was created
        last_activity: Timestamp of last activity
        message_count: Number of messages in session
        tool_calls_used: Number of tool calls made
        is_active: Whether the session is currently active
    """

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    message_count: int = 0
    tool_calls_used: int = 0
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": self.message_count,
            "tool_calls_used": self.tool_calls_used,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionInfo":
        """Create from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            created_at=data.get("created_at", time.time()),
            last_activity=data.get("last_activity", time.time()),
            message_count=data.get("message_count", 0),
            tool_calls_used=data.get("tool_calls_used", 0),
            is_active=data.get("is_active", True),
        )


@dataclass
class SessionCostSummary:
    """Summary of costs for a session.

    Attributes:
        total_cost: Total cost in USD
        input_cost: Cost for input tokens
        output_cost: Cost for output tokens
        total_tokens: Total tokens used
        input_tokens: Input tokens used
        output_tokens: Output tokens used
    """

    total_cost: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_cost": self.total_cost,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class SessionCoordinator:
    """Coordinates session lifecycle and state management.

    This coordinator provides a unified interface for session management,
    delegating to specialized components for specific operations.

    Responsibilities:
    - Session ID creation and tracking
    - Session lifecycle (create, end, reset, recover)
    - Memory session integration
    - Checkpoint/recovery coordination
    - Session statistics and summary

    The coordinator does NOT directly manage:
    - Conversation messages (delegates to ConversationController)
    - Tool execution (delegates to ToolPipeline)
    - Resource cleanup (delegates to LifecycleManager)
    """

    def __init__(
        self,
        session_state_manager: SessionStateManager,
        lifecycle_manager: Optional[LifecycleManager] = None,
        memory_manager: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None,
        cost_tracker: Optional[Any] = None,
    ):
        """Initialize the session coordinator.

        Args:
            session_state_manager: Manager for execution state tracking
            lifecycle_manager: Optional manager for lifecycle operations
            memory_manager: Optional manager for persistent session storage
            checkpoint_manager: Optional manager for checkpoint save/restore
            cost_tracker: Optional tracker for session costs
        """
        self._session_state = session_state_manager
        self._lifecycle_manager = lifecycle_manager
        self._memory_manager = memory_manager
        self._checkpoint_manager = checkpoint_manager
        self._cost_tracker = cost_tracker

        # Current session info
        self._current_session: Optional[SessionInfo] = None
        self._memory_session_id: Optional[str] = None

        logger.debug("SessionCoordinator initialized")

    # ========================================================================
    # Session Lifecycle
    # ========================================================================

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session.

        Args:
            session_id: Optional custom session ID. If not provided,
                       generates a UUID.

        Returns:
            The session ID for the new session
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session-{uuid.uuid4().hex[:16]}"

        # Create session info
        self._current_session = SessionInfo(session_id=session_id)

        # Reset session state for new session
        self._session_state.reset()

        # Initialize memory session if memory manager available
        if self._memory_manager:
            try:
                self._memory_session_id = self._memory_manager.create_session(
                    project_path=getattr(self._memory_manager, "_project_path", None),
                )
                logger.debug(f"Memory session created: {self._memory_session_id}")
            except Exception as e:
                logger.warning(f"Failed to create memory session: {e}")
                self._memory_session_id = None

        logger.info(f"Session created: {session_id}")
        return session_id

    def end_session(self) -> None:
        """End the current session.

        Marks the session as inactive and prepares for cleanup.
        """
        if self._current_session:
            self._current_session.is_active = False
            logger.info(f"Session ended: {self._current_session.session_id}")

        # End memory session if active
        if self._memory_manager and self._memory_session_id:
            try:
                # Memory manager may have an end_session method
                if hasattr(self._memory_manager, "end_session"):
                    self._memory_manager.end_session(self._memory_session_id)
                logger.debug(f"Memory session ended: {self._memory_session_id}")
            except Exception as e:
                logger.warning(f"Failed to end memory session: {e}")

    def reset_session(self, preserve_token_usage: bool = False) -> None:
        """Reset the current session state.

        Args:
            preserve_token_usage: If True, keep accumulated token usage
        """
        # Reset session state (with option to preserve tokens)
        self._session_state.reset(preserve_token_usage=preserve_token_usage)

        # Delegate to lifecycle manager for conversation reset
        if self._lifecycle_manager:
            self._lifecycle_manager.reset_conversation()

        # Update session activity
        if self._current_session:
            self._current_session.last_activity = time.time()

        logger.debug("Session reset")

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous session.

        Args:
            session_id: ID of the session to recover

        Returns:
            True if session was recovered successfully
        """
        if not self._memory_manager:
            logger.warning("Memory manager not available for session recovery")
            return False

        # Delegate to lifecycle manager for recovery
        if self._lifecycle_manager:
            success = self._lifecycle_manager.recover_session(
                session_id=session_id,
                memory_manager=self._memory_manager,
            )

            if success:
                self._memory_session_id = session_id
                self._current_session = SessionInfo(
                    session_id=session_id,
                    is_active=True,
                )
                logger.info(f"Session recovered: {session_id[:8]}...")
            else:
                logger.warning(f"Failed to recover session: {session_id}")

            return success

        return False

    # ========================================================================
    # Session State Access
    # ========================================================================

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._current_session.session_id if self._current_session else None

    @property
    def memory_session_id(self) -> Optional[str]:
        """Get the current memory session ID."""
        return self._memory_session_id

    @property
    def session_state(self) -> SessionStateManager:
        """Get the session state manager."""
        return self._session_state

    @property
    def is_active(self) -> bool:
        """Check if current session is active."""
        return self._current_session.is_active if self._current_session else False

    @property
    def tool_calls_used(self) -> int:
        """Get number of tool calls used in current session."""
        return self._session_state.tool_calls_used

    @property
    def remaining_budget(self) -> int:
        """Get remaining tool budget."""
        return self._session_state.get_remaining_budget()

    @property
    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted."""
        return self._session_state.is_budget_exhausted()

    # ========================================================================
    # Session Statistics
    # ========================================================================

    def get_session_info(self) -> Optional[SessionInfo]:
        """Get current session information.

        Returns:
            SessionInfo if session exists, None otherwise
        """
        return self._current_session

    def get_session_stats(self) -> dict[str, Any]:
        """Get comprehensive session statistics.

        Returns:
            Dictionary with session statistics including:
            - enabled: Whether memory manager is active
            - session_id: Current session ID
            - message_count: Number of messages
            - tool_calls_used: Tool calls made
            - tool_budget: Total tool budget
            - budget_remaining: Remaining tool budget
            - is_active: Whether session is active
            - token_usage: Token usage breakdown
        """
        base_stats: dict[str, Any] = {
            "enabled": bool(self._memory_manager),
            "session_id": self.session_id,
            "memory_session_id": self._memory_session_id,
            "is_active": self.is_active,
            "tool_calls_used": self.tool_calls_used,
            "tool_budget": self._session_state.tool_budget,
            "budget_remaining": self.remaining_budget,
            "budget_exhausted": self.is_budget_exhausted,
            "token_usage": self._session_state.get_token_usage(),
        }

        # Add memory manager stats if available
        if self._memory_manager and self._memory_session_id:
            try:
                memory_stats = self._memory_manager.get_session_stats(self._memory_session_id)
                if memory_stats:
                    base_stats.update(memory_stats)
            except Exception as e:
                logger.warning(f"Failed to get memory stats: {e}")

        return base_stats

    def get_session_summary(self) -> dict[str, Any]:
        """Get a summary of the current session.

        Returns:
            Dictionary with session summary from SessionStateManager
        """
        summary = self._session_state.get_session_summary()
        summary["session_id"] = self.session_id
        summary["memory_session_id"] = self._memory_session_id
        return summary

    def get_session_cost_summary(self) -> dict[str, Any]:
        """Get session cost summary.

        Returns:
            Dictionary with session cost statistics
        """
        if self._cost_tracker and hasattr(self._cost_tracker, "get_summary"):
            summary: dict[str, Any] = self._cost_tracker.get_summary()
            return summary
        return {}

    def get_session_cost_formatted(self) -> str:
        """Get formatted session cost string.

        Returns:
            Cost string like "$0.0123" or "cost n/a"
        """
        if self._cost_tracker and hasattr(self._cost_tracker, "format_inline_cost"):
            cost_str: str = self._cost_tracker.format_inline_cost()
            return cost_str
        return "cost n/a"

    # ========================================================================
    # Checkpoint/Recovery
    # ========================================================================

    async def save_checkpoint(
        self,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Save a checkpoint of current session state.

        Args:
            description: Human-readable description for the checkpoint
            tags: Optional tags for categorization

        Returns:
            Checkpoint ID if saved, None if checkpointing disabled
        """
        if not self._checkpoint_manager:
            logger.debug("Checkpoint save skipped - manager not initialized")
            return None

        # Build conversation state from session state
        state = self._get_checkpoint_state()

        try:
            checkpoint_id: Optional[str] = await self._checkpoint_manager.save_checkpoint(
                session_id=self._memory_session_id or "default",
                state=state,
                description=description,
                tags=tags,
            )
            if checkpoint_id:
                logger.info(f"Checkpoint saved: {checkpoint_id[:20]}...")
            return checkpoint_id
        except (OSError, IOError) as e:
            logger.warning(f"Failed to save checkpoint (I/O error): {e}")
            return None
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to save checkpoint (serialization error): {e}")
            return None

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore session state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if restored successfully, False otherwise
        """
        if not self._checkpoint_manager:
            logger.warning("Cannot restore - checkpoint manager not initialized")
            return False

        try:
            state = await self._checkpoint_manager.restore_checkpoint(checkpoint_id)
            self._apply_checkpoint_state(state)
            logger.info(f"Checkpoint restored: {checkpoint_id[:20]}...")
            return True
        except (OSError, IOError) as e:
            logger.warning(f"Failed to restore checkpoint (I/O error): {e}")
            return False
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to restore checkpoint (invalid data): {e}")
            return False

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Trigger auto-checkpoint if interval threshold is met.

        Returns:
            Checkpoint ID if auto-checkpoint was created, None otherwise
        """
        if not self._checkpoint_manager:
            return None

        state = self._get_checkpoint_state()

        try:
            checkpoint_id: Optional[str] = await self._checkpoint_manager.maybe_auto_checkpoint(
                session_id=self._memory_session_id or "default",
                state=state,
            )
            return checkpoint_id
        except (OSError, IOError) as e:
            logger.debug(f"Auto-checkpoint failed (I/O error): {e}")
            return None
        except (ValueError, TypeError) as e:
            logger.debug(f"Auto-checkpoint failed (serialization error): {e}")
            return None

    def _get_checkpoint_state(self) -> dict[str, Any]:
        """Build a dictionary representing current session state for checkpointing."""
        return {
            "session_id": self.session_id,
            "tool_calls_used": self.tool_calls_used,
            "tool_budget": self._session_state.tool_budget,
            "token_usage": self._session_state.get_token_usage(),
            "observed_files": list(self._session_state.observed_files),
            "executed_tools": list(self._session_state.executed_tools),
        }

    def _apply_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Apply a checkpoint state to restore the session.

        Args:
            state: State dictionary from checkpoint
        """
        # Restore session state
        self._session_state.execution_state.tool_calls_used = state.get("tool_calls_used", 0)
        self._session_state._tool_budget = state.get("tool_budget", self._session_state.tool_budget)

        # Restore token usage
        token_usage = state.get("token_usage", {})
        if token_usage:
            self._session_state.execution_state.token_usage = token_usage

        # Restore observed files and executed tools
        self._session_state.execution_state.observed_files = set(state.get("observed_files", []))
        self._session_state.execution_state.executed_tools = list(state.get("executed_tools", []))

        # Update session activity
        if self._current_session:
            self._current_session.last_activity = time.time()

    # ========================================================================
    # Recent Sessions
    # ========================================================================

    def get_recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent conversation sessions for recovery.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries
        """
        if not self._memory_manager:
            return []

        try:
            sessions = self._memory_manager.list_sessions(limit=limit)
            return [
                {
                    "session_id": s.session_id,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "last_activity": s.last_activity.isoformat() if s.last_activity else None,
                    "project_path": s.project_path,
                    "provider": s.provider,
                    "model": s.model,
                    "message_count": len(s.messages) if hasattr(s, "messages") else 0,
                }
                for s in sessions
            ]
        except Exception as e:
            logger.warning(f"Failed to get recent sessions: {e}")
            return []

    # ========================================================================
    # Token Usage
    # ========================================================================

    def get_token_usage(self) -> dict[str, int]:
        """Get cumulative token usage.

        Returns:
            Dictionary with prompt_tokens, completion_tokens, total_tokens, etc.
        """
        return self._session_state.get_token_usage()

    def update_token_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        """Update cumulative token usage.

        Args:
            prompt_tokens: Input tokens used
            completion_tokens: Output tokens generated
            cache_creation_input_tokens: Tokens used for cache creation
            cache_read_input_tokens: Tokens read from cache
        """
        self._session_state.update_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )

    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking."""
        self._session_state.reset_token_usage()

    # ========================================================================
    # Memory Context
    # ========================================================================

    def get_memory_context(
        self,
        max_tokens: Optional[int] = None,
        messages: Optional[list[Any]] = None,
    ) -> list[dict[str, Any]]:
        """Get token-aware context messages from memory manager.

        Args:
            max_tokens: Override max tokens for this retrieval
            messages: Fallback messages if memory not available

        Returns:
            List of messages in provider format
        """
        if not self._memory_manager or not self._memory_session_id:
            # Fall back to provided messages
            if messages:
                # Convert Message objects to dict if needed
                result_messages: list[dict[str, Any]] = [
                    msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in messages
                ]
                return result_messages
            return []

        try:
            memory_messages: list[dict[str, Any]] = self._memory_manager.get_context_messages(
                session_id=self._memory_session_id,
                max_tokens=max_tokens,
            )
            return memory_messages
        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}, using fallback")
            if messages:
                fallback_messages: list[dict[str, Any]] = [
                    msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in messages
                ]
                return fallback_messages
            return []

    # ========================================================================
    # String Representation
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SessionCoordinator("
            f"session_id={self.session_id}, "
            f"active={self.is_active}, "
            f"tool_calls={self.tool_calls_used}/{self._session_state.tool_budget})"
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_session_coordinator(
    session_state_manager: SessionStateManager,
    lifecycle_manager: Optional[LifecycleManager] = None,
    memory_manager: Optional[Any] = None,
    checkpoint_manager: Optional[Any] = None,
    cost_tracker: Optional[Any] = None,
) -> SessionCoordinator:
    """Factory function to create a SessionCoordinator.

    Args:
        session_state_manager: Manager for execution state tracking
        lifecycle_manager: Optional manager for lifecycle operations
        memory_manager: Optional manager for persistent session storage
        checkpoint_manager: Optional manager for checkpoint save/restore
        cost_tracker: Optional tracker for session costs

    Returns:
        Configured SessionCoordinator instance
    """
    return SessionCoordinator(
        session_state_manager=session_state_manager,
        lifecycle_manager=lifecycle_manager,
        memory_manager=memory_manager,
        checkpoint_manager=checkpoint_manager,
        cost_tracker=cost_tracker,
    )


__all__ = [
    "SessionCoordinator",
    "SessionInfo",
    "SessionCostSummary",
    "create_session_coordinator",
]
