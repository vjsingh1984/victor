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

"""Conversation management refinement protocols.

This module contains refined protocols for conversation management following
SOLID principles. These protocols define contracts for:

- Message storage and retrieval
- Context overflow handling
- Session lifecycle management
- Embedding and semantic search

Usage:
    from victor.protocols.agent_conversation_refined import (
        IMessageStore,
        IContextOverflowHandler,
        ISessionManager,
    )
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable


# =============================================================================
# Message Storage Protocols
# =============================================================================


@runtime_checkable
class IMessageStore(Protocol):
    """Protocol for message storage and retrieval.

    Defines interface for persisting and retrieving messages.
    Separated from other conversation concerns to follow ISP.
    """

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to storage.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            **metadata: Additional message metadata
        """
        ...

    def get_messages(self, limit: Optional[int] = None) -> List[Any]:
        """Retrieve messages.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        ...

    def persist(self) -> bool:
        """Persist messages to storage.

        Returns:
            True if persistence succeeded, False otherwise
        """
        ...


# =============================================================================
# Context Management Protocols
# =============================================================================


@runtime_checkable
class IContextOverflowHandler(Protocol):
    """Protocol for context overflow handling.

    Defines interface for detecting and handling context overflow.
    Separated from IMessageStore to follow ISP.
    """

    def check_overflow(self) -> bool:
        """Check if context has overflowed.

        Returns:
            True if overflow detected, False otherwise
        """
        ...

    def handle_compaction(self) -> Optional[Any]:
        """Handle context compaction.

        Returns:
            Compaction result or None
        """
        ...


# =============================================================================
# Session Management Protocols
# =============================================================================


@runtime_checkable
class ISessionManager(Protocol):
    """Protocol for session lifecycle management.

    Defines interface for creating and managing sessions.
    Separated to support different session backends.
    """

    def create_session(self) -> str:
        """Create a new session.

        Returns:
            Session ID
        """
        ...

    def recover_session(self, session_id: str) -> bool:
        """Recover an existing session.

        Args:
            session_id: Session ID to recover

        Returns:
            True if recovery succeeded, False otherwise
        """
        ...

    def persist_session(self) -> bool:
        """Persist session state.

        Returns:
            True if persistence succeeded, False otherwise
        """
        ...


# =============================================================================
# Embedding Management Protocols
# =============================================================================


@runtime_checkable
class IEmbeddingManager(Protocol):
    """Protocol for embedding and semantic search.

    Defines interface for semantic search over conversations.
    Separated because not all conversations need embeddings.
    """

    def initialize_embeddings(self) -> None:
        """Initialize embedding store."""
        ...

    def semantic_search(self, query: str, k: int = 5) -> List[Any]:
        """Perform semantic search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of search results
        """
        ...


__all__ = [
    # Message storage protocols
    "IMessageStore",
    # Context management protocols
    "IContextOverflowHandler",
    # Session management protocols
    "ISessionManager",
    # Embedding management protocols
    "IEmbeddingManager",
]
