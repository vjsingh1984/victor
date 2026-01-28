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

"""Memory manager for conversation session management.

This module provides a unified interface for conversation memory operations,
extracting memory-related logic from the orchestrator for better separation
of concerns.

Design Principles:
- Protocol-based design for testability (DIP)
- Clean separation between session recovery and memory operations
- Fallback to in-memory conversation when persistent storage is unavailable
- Token-aware context retrieval
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.conversation_memory import ConversationStore
    from victor.agent.message_history import MessageHistory
    from victor.agent.lifecycle_manager import LifecycleManager

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages conversation memory and session persistence.

    This manager provides a unified interface for memory operations,
    wrapping the ConversationStore and providing fallback behavior
    when persistent storage is unavailable.

    Example:
        memory_manager = MemoryManager(
            conversation_store=store,
            session_id=session_id,
        )
        context = memory_manager.get_context(max_tokens=8000)
        stats = memory_manager.get_session_stats()
    """

    def __init__(
        self,
        conversation_store: Optional["ConversationStore"] = None,
        session_id: Optional[str] = None,
        message_history: Optional["MessageHistory"] = None,
    ) -> None:
        """Initialize the memory manager.

        Args:
            conversation_store: Optional ConversationStore for persistent storage
            session_id: Optional session ID for this manager instance
            message_history: Optional in-memory message history for fallback
        """
        self._conversation_store = conversation_store
        self._session_id = session_id
        self._message_history = message_history

    @property
    def is_enabled(self) -> bool:
        """Check if persistent memory is enabled.

        Returns:
            True if conversation_store and session_id are available
        """
        return self._conversation_store is not None and self._session_id is not None

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID.

        Returns:
            Session ID or None if not set
        """
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        """Set the session ID.

        Args:
            value: New session ID
        """
        self._session_id = value

    def get_context(
        self,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get token-aware context messages from memory.

        Uses intelligent pruning to select the most relevant messages
        within token budget. Useful for long conversations.

        Args:
            max_tokens: Override max tokens for this retrieval. If None,
                       uses the default token limit from conversation store.

        Returns:
            List of messages in provider format, where each message is a
            dictionary containing 'role' and 'content' keys.

        Note:
            If memory is not enabled, falls back to returning messages from
            in-memory conversation history. If memory retrieval fails, logs
            a warning and uses in-memory messages as fallback.
        """
        if not self.is_enabled:
            # Fall back to in-memory conversation
            return self._get_in_memory_context()

        try:
            if self._conversation_store is None or self._session_id is None:
                return self._get_in_memory_context()
            return self._conversation_store.get_context_messages(
                session_id=self._session_id,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning(f"Failed to get memory context: {e}, using in-memory")
            return self._get_in_memory_context()

    def _get_in_memory_context(self) -> List[Dict[str, Any]]:
        """Get context from in-memory message history.

        Returns:
            List of messages from in-memory history
        """
        if self._message_history is None:
            return []

        try:
            return [msg.model_dump() for msg in self._message_history.messages]
        except Exception as e:
            logger.warning(f"Failed to get in-memory context: {e}")
            return []

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current memory session.

        Returns:
            Dictionary with session statistics including:
            - enabled: Whether memory manager is active
            - session_id: Current session ID
            - message_count: Number of messages
            - total_tokens: Total token usage
            - max_tokens: Maximum token budget
            - available_tokens: Remaining token budget
            - Other session metadata
        """
        if not self.is_enabled:
            return {
                "enabled": False,
                "session_id": None,
                "message_count": self._get_in_memory_message_count(),
            }

        try:
            if self._conversation_store is None or self._session_id is None:
                return {
                    "enabled": True,
                    "session_id": self._session_id,
                    "error": "Session not available",
                }
            stats = self._conversation_store.get_session_stats(self._session_id)
            if not stats:
                return {
                    "enabled": True,
                    "session_id": self._session_id,
                    "error": "Session not found",
                }

            # Add enabled flag
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.warning(f"Failed to get session stats: {e}")
            return {"enabled": True, "session_id": self._session_id, "error": str(e)}

    def _get_in_memory_message_count(self) -> int:
        """Get message count from in-memory history.

        Returns:
            Number of messages in in-memory history
        """
        if self._message_history is None:
            return 0
        try:
            return len(self._message_history.messages)
        except Exception:
            return 0

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions for recovery.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries
        """
        if not self._conversation_store:
            return []

        try:
            sessions = self._conversation_store.list_sessions(limit=limit)
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

    def add_message(self, role: str, content: str) -> bool:
        """Add a message to persistent storage.

        Args:
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            True if message was persisted successfully
        """
        if not self.is_enabled:
            return False

        try:
            from victor.agent.conversation_memory import MessageRole

            role_map = {
                "user": MessageRole.USER,
                "assistant": MessageRole.ASSISTANT,
                "system": MessageRole.SYSTEM,
            }
            msg_role = role_map.get(role, MessageRole.USER)
            if self._conversation_store is None or self._session_id is None:
                return False
            self._conversation_store.add_message(
                session_id=self._session_id,
                role=msg_role,
                content=content,
            )
            return True
        except Exception as e:
            logger.debug(f"Failed to persist message to memory store: {e}")
            return False


class SessionRecoveryManager:
    """Manages session recovery operations.

    This manager handles session recovery logic, coordinating between
    the MemoryManager and LifecycleManager.

    Example:
        recovery_manager = SessionRecoveryManager(
            memory_manager=memory_manager,
            lifecycle_manager=lifecycle_manager,
        )
        success = recovery_manager.recover_session(session_id)
    """

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        lifecycle_manager: Optional["LifecycleManager"] = None,
    ) -> None:
        """Initialize the session recovery manager.

        Args:
            memory_manager: MemoryManager for session operations
            lifecycle_manager: Optional LifecycleManager for recovery coordination
        """
        self._memory_manager = memory_manager
        self._lifecycle_manager = lifecycle_manager

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session.

        Delegates to LifecycleManager for core recovery logic if available,
        otherwise falls back to MemoryManager for basic recovery.

        Args:
            session_id: ID of the session to recover

        Returns:
            True if session was recovered successfully
        """
        if self._memory_manager is None:
            logger.warning("Memory manager not available, cannot recover session")
            return False

        # Try LifecycleManager first if available
        if self._lifecycle_manager is not None:
            try:
                success = self._lifecycle_manager.recover_session(
                    session_id=session_id,
                    memory_manager=self._memory_manager._conversation_store,
                )

                if success:
                    # Update memory manager session tracking
                    self._memory_manager.session_id = session_id
                    logger.info(f"Recovered session {session_id[:8]}... via LifecycleManager")
                    return success
            except Exception as e:
                logger.debug(f"LifecycleManager recovery failed: {e}, trying direct recovery")

        # Direct recovery through memory manager
        try:
            # Validate session exists
            store = self._memory_manager._conversation_store
            if store is not None:
                stats = store.get_session_stats(session_id)
                if stats is not None:
                    # Update session ID
                    self._memory_manager.session_id = session_id
                    logger.info(f"Recovered session {session_id[:8]}... via direct recovery")
                    return True

                logger.warning(f"Session {session_id} not found")  # type: ignore[unreachable]
                return False
            else:
                logger.warning("Conversation store not available")
                return False
        except Exception as e:
            logger.warning(f"Failed to recover session {session_id}: {e}")
            return False


def create_memory_manager(
    conversation_store: Optional["ConversationStore"] = None,
    session_id: Optional[str] = None,
    message_history: Optional["MessageHistory"] = None,
) -> MemoryManager:
    """Factory function to create a MemoryManager.

    Args:
        conversation_store: Optional ConversationStore for persistent storage
        session_id: Optional session ID for this manager instance
        message_history: Optional in-memory message history for fallback

    Returns:
        MemoryManager instance
    """
    return MemoryManager(
        conversation_store=conversation_store,
        session_id=session_id,
        message_history=message_history,
    )


def create_session_recovery_manager(
    memory_manager: Optional[MemoryManager] = None,
    lifecycle_manager: Optional["LifecycleManager"] = None,
) -> SessionRecoveryManager:
    """Factory function to create a SessionRecoveryManager.

    Args:
        memory_manager: MemoryManager for session operations
        lifecycle_manager: Optional LifecycleManager for recovery coordination

    Returns:
        SessionRecoveryManager instance
    """
    return SessionRecoveryManager(
        memory_manager=memory_manager,
        lifecycle_manager=lifecycle_manager,
    )
