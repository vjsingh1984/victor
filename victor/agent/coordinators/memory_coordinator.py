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

"""Memory coordinator for conversation memory management.

This coordinator manages conversation memory, including context retrieval,
session statistics, and session recovery operations. It provides a unified
interface for memory operations with proper error handling and fallbacks.

Key Features:
- Token-aware context retrieval with pruning
- Session statistics and metadata
- Session recovery with proper state restoration
- Fallback to in-memory messages when memory manager unavailable
- Graceful error handling with logging

Design Patterns:
- SRP: Single responsibility for memory management
- Dependency Inversion: Depends on protocols, not concrete implementations
- Fail-Safe: Graceful degradation when memory manager unavailable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.conversation_memory import ConversationStore


logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Statistics for a memory session.

    Attributes:
        enabled: Whether memory manager is active
        session_id: Current session ID
        message_count: Number of messages in session
        total_tokens: Total token usage
        max_tokens: Maximum token budget
        available_tokens: Remaining token budget
        found: Whether session was found in store
        error: Error message if operation failed
    """

    enabled: bool
    session_id: Optional[str]
    message_count: int
    total_tokens: int = 0
    max_tokens: int = 0
    available_tokens: int = 0
    found: bool = True
    error: Optional[str] = None


@dataclass
class SessionInfo:
    """Information about a conversation session.

    Attributes:
        session_id: Unique session identifier
        created_at: Session creation timestamp
        message_count: Number of messages in session
        total_tokens: Total tokens used
    """

    session_id: str
    created_at: str
    message_count: int
    total_tokens: int


class MemoryCoordinator:
    """Coordinator for conversation memory management.

    This coordinator handles all memory-related operations including context
    retrieval, session statistics, and session recovery. It provides graceful
    fallback when memory manager is unavailable.

    Example:
        ```python
        coordinator = MemoryCoordinator(
            memory_manager=memory_manager,
            session_id="session-123"
        )

        # Get token-aware context
        context = coordinator.get_memory_context(max_tokens=4000)

        # Get session statistics
        stats = coordinator.get_session_stats()

        # Recover session
        success = coordinator.recover_session("session-456")
        ```
    """

    def __init__(
        self,
        memory_manager: Optional[Any] = None,
        session_id: Optional[str] = None,
        conversation_store: Optional["ConversationStore"] = None,
    ):
        """Initialize the memory coordinator.

        Args:
            memory_manager: Memory manager instance (optional)
            session_id: Current session ID (optional)
            conversation_store: Fallback conversation store (optional)
        """
        self._memory_manager = memory_manager
        self._session_id = session_id
        self._conversation_store = conversation_store

    def get_memory_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get token-aware context messages from memory manager.

        Uses intelligent pruning to select the most relevant messages
        within token budget. Useful for long conversations.

        Args:
            max_tokens: Override max tokens for this retrieval. If None,
                       uses the default token limit from memory manager.

        Returns:
            List of messages in provider format, where each message is a
            dictionary containing 'role' and 'content' keys.

        Note:
            If memory manager is not enabled or no session is active,
            falls back to returning messages from in-memory conversation.
            If memory retrieval fails, logs a warning and uses in-memory
            messages as fallback.
        """
        # Fall back to in-memory messages if memory manager is not enabled or no session
        if self._memory_manager is None or not self._session_id:
            return self._get_in_memory_messages()

        # Delegate to MemoryManager with exception handling
        try:
            messages: List[Dict[str, Any]] = self._memory_manager.get_context(max_tokens=max_tokens)
            return messages
        except Exception as e:
            logger.warning(f"Failed to get memory context, falling back to in-memory: {e}")
            return self._get_in_memory_messages()

    def get_session_stats(self, message_count: int = 0) -> MemoryStats:
        """Get statistics for the current memory session.

        Args:
            message_count: Current message count (for fallback)

        Returns:
            MemoryStats with session information including:
            - enabled: Whether memory manager is active
            - session_id: Current session ID
            - message_count: Number of messages
            - total_tokens: Total token usage
            - max_tokens: Maximum token budget
            - available_tokens: Remaining token budget
            - Other session metadata
        """
        # Return basic stats if memory manager not available
        if self._memory_manager is None or not self._session_id:
            return MemoryStats(
                enabled=False,
                session_id=self._session_id,
                message_count=message_count,
            )

        try:
            stats = self._memory_manager.get_session_stats(self._session_id)

            # Handle empty stats (session not found)
            if not stats or not any(k in stats for k in ("message_count", "total_tokens", "found")):
                return MemoryStats(
                    enabled=True,
                    session_id=self._session_id,
                    message_count=message_count,
                    found=False,
                    error="Session not found",
                )

            # Convert to MemoryStats
            return MemoryStats(
                enabled=stats.get("enabled", True),
                session_id=stats.get("session_id", self._session_id),
                message_count=stats.get("message_count", message_count),
                total_tokens=stats.get("total_tokens", 0),
                max_tokens=stats.get("max_tokens", 0),
                available_tokens=stats.get("available_tokens", 0),
                found=stats.get("found", True),
                error=stats.get("error"),
            )

        except Exception as e:
            logger.warning(f"Failed to get session stats: {e}")
            return MemoryStats(
                enabled=True,
                session_id=self._session_id,
                message_count=message_count,
                error=str(e),
            )

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions for recovery.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries

        Note:
            If memory manager is not enabled or retrieval fails, returns empty list.
        """
        # Return empty list if memory manager is not enabled
        if self._memory_manager is None:
            return []

        # Delegate to MemoryManager with exception handling
        try:
            sessions: List[Dict[str, Any]] = self._memory_manager.get_recent_sessions(limit=limit)
            return sessions
        except Exception as e:
            logger.warning(f"Failed to get recent sessions: {e}")
            return []

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session.

        This method updates the coordinator's session_id if recovery succeeds.

        Args:
            session_id: ID of the session to recover

        Returns:
            True if session was recovered successfully
        """
        # Return False if memory manager is not enabled
        if self._memory_manager is None:
            logger.warning("Cannot recover session: memory manager not enabled")
            return False

        try:
            # Delegate to MemoryManager
            success: bool = self._memory_manager.recover_session(session_id)

            if success:
                # Update session tracking
                self._session_id = session_id
                logger.info(f"Recovered session {session_id[:8]}... ")
            else:
                logger.warning(f"Failed to recover session {session_id}")

            return success

        except Exception as e:
            logger.error(f"Error recovering session {session_id}: {e}")
            return False

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Set the current session ID.

        Args:
            session_id: New session ID (or None to clear)
        """
        self._session_id = session_id

    def get_session_id(self) -> Optional[str]:
        """Get the current session ID.

        Returns:
            Current session ID or None if not set
        """
        return self._session_id

    def is_enabled(self) -> bool:
        """Check if memory manager is enabled.

        Returns:
            True if memory manager is available and session is active
        """
        return self._memory_manager is not None and self._session_id is not None

    def _get_in_memory_messages(self) -> List[Dict[str, Any]]:
        """Get messages from in-memory conversation store.

        Returns:
            List of message dictionaries
        """
        if self._conversation_store is None:
            return []

        try:
            messages_list = self._conversation_store.get_recent_messages(1000)
            return [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages_list]
        except Exception as e:
            logger.warning(f"Failed to get in-memory messages: {e}")
            return []


def create_memory_coordinator(
    memory_manager: Optional[Any] = None,
    session_id: Optional[str] = None,
    conversation_store: Optional["ConversationStore"] = None,
) -> MemoryCoordinator:
    """Factory function to create a MemoryCoordinator.

    Args:
        memory_manager: Memory manager instance
        session_id: Initial session ID
        conversation_store: Fallback conversation store

    Returns:
        Configured MemoryCoordinator instance
    """
    return MemoryCoordinator(
        memory_manager=memory_manager,
        session_id=session_id,
        conversation_store=conversation_store,
    )
