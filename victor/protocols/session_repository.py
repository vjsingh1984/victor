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

"""Session repository protocol for UI-database decoupling.

This module defines the SessionRepositoryProtocol, which abstracts
session persistence operations from the UI layer. This enables
Dependency Inversion Principle (DIP) compliance by allowing UI
components to depend on abstractions rather than concrete database
implementations.

The protocol supports both the legacy SessionManager from victor.ui.tui.session
and the newer SQLiteSessionPersistence from victor.agent.sqlite_session_persistence.

Protocol:
    SessionRepositoryProtocol: Interface for session persistence operations

Usage:
    from victor.protocols.session_repository import SessionRepositoryProtocol

    # Type annotation for dependency injection
    def __init__(self, session_repo: SessionRepositoryProtocol):
        self._session_repo = session_repo
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class SessionRepositoryProtocol(Protocol):
    """Protocol for session repository implementations.

    This protocol defines the interface for session persistence operations,
    abstracting the underlying storage mechanism (SQLite, JSON, etc.) from
    UI components. Implementations can use different storage backends while
    conforming to this interface.

    The protocol is designed to support two different session representations:
    1. Legacy Session objects from victor.ui.tui.session (used by TUI)
    2. Dictionary-based session data from SQLiteSessionPersistence (used by CLI)

    Implementations should handle both formats transparently, converting
    as needed for their storage backend.

    Methods:
        get_session: Retrieve a session by ID
        save_session: Save or update a session
        list_sessions: List recent sessions with metadata
        delete_session: Delete a session by ID
        get_latest_session: Get the most recently updated session
        create_session: Factory method for new session instances

    Example:
        class SQLiteSessionRepository:
            '''SQLite implementation of session repository.'''

            def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
                # Query database and return session data
                ...

            def save_session(self, session: Dict[str, Any]) -> None:
                # Save or update session in database
                ...

        # Dependency injection in UI component
        class TUIApp:
            def __init__(self, session_repo: SessionRepositoryProtocol):
                self._session_repo = session_repo

            def load_session(self, session_id: str):
                return self._session_repo.get_session(session_id)
    """

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Get a session by ID.

        Args:
            session_id: Unique session identifier

        Returns:
            Session data as dictionary, or None if not found.
            Dictionary should contain at minimum:
            - id: Session identifier
            - name: Session name/title
            - provider: LLM provider name
            - model: Model name
            - created_at: ISO timestamp of creation
            - updated_at: ISO timestamp of last update
            - messages: List of message dictionaries (for full session data)
            - metadata: Optional additional metadata

        Example:
            session = await repo.get_session("abc123")
            if session:
                print(f"Session: {session['name']}")
                for msg in session.get('messages', []):
                    print(f"  {msg['role']}: {msg['content']}")
        """
        ...

    async def save_session(self, session: dict[str, Any]) -> None:
        """Save or update a session.

        Args:
            session: Session data dictionary with all session fields.
                    Must include 'id' field for updates.

        Raises:
            ValueError: If session data is invalid
            IOError: If save operation fails

        Example:
            await repo.save_session({
                'id': 'abc123',
                'name': 'My Session',
                'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'created_at': '2025-01-14T10:00:00',
                'updated_at': '2025-01-14T11:00:00',
                'messages': [...],
                'metadata': {}
            })
        """
        ...

    async def list_sessions(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List recent sessions with metadata only (not full messages).

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries, ordered by updated_at DESC.
            Each dict contains:
            - id: Session identifier
            - name: Session name/title
            - provider: LLM provider name
            - model: Model name
            - created_at: ISO timestamp
            - updated_at: ISO timestamp
            - message_count: Number of messages in session

        Example:
            sessions = await repo.list_sessions(limit=20)
            for session in sessions:
                print(f"{session['name']}: {session['message_count']} messages")
        """
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: Unique session identifier to delete

        Returns:
            True if session was deleted, False if not found

        Example:
            deleted = await repo.delete_session("abc123")
            if deleted:
                print("Session deleted")
        """
        ...

    async def get_latest_session(self) -> Optional[dict[str, Any]]:
        """Get the most recently updated session.

        Returns:
            Session data dictionary for the latest session, or None if no sessions exist

        Example:
            latest = await repo.get_latest_session()
            if latest:
                print(f"Resuming: {latest['name']}")
        """
        ...

    def create_session(
        self,
        provider: str = "",
        model: str = "",
        name: str = "",
    ) -> dict[str, Any]:
        """Factory method to create a new session instance.

        This is a synchronous factory method that creates a session
        object/dictionary but does not persist it. Use save_session()
        to persist after creation.

        Args:
            provider: LLM provider name
            model: Model name
            name: Optional session name/title

        Returns:
            New session dictionary with unique ID, ready for population and saving

        Example:
            session = repo.create_session(
                provider="anthropic",
                model="claude-sonnet-4-5",
                name="My Session"
            )
            # Populate with messages...
            await repo.save_session(session)
        """
        ...


__all__ = [
    "SessionRepositoryProtocol",
]
