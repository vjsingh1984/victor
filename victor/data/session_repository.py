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

"""Concrete session repository implementation.

This module provides SQLiteSessionRepository, which implements
SessionRepositoryProtocol using the existing SessionManager from
victor.ui.tui.session. This acts as an adapter layer, allowing
UI components to depend on the repository protocol while the
implementation wraps the legacy SessionManager.

Future implementations could use SQLiteSessionPersistence directly
or other storage backends (PostgreSQL, Redis, etc.) while still
conforming to the SessionRepositoryProtocol.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.protocols.session_repository import SessionRepositoryProtocol


class SQLiteSessionRepository:
    """SQLite implementation of session repository.

    This repository wraps the legacy SessionManager from victor.ui.tui.session,
    providing an async interface that conforms to SessionRepositoryProtocol.

    The wrapper approach allows gradual migration:
    1. UI components use SessionRepositoryProtocol
    2. This repository wraps existing SessionManager
    3. Future versions can use SQLiteSessionPersistence or other backends
       without changing UI code

    Example:
        from victor.data.session_repository import SQLiteSessionRepository

        repo = SQLiteSessionRepository()

        # Create and save session
        session = repo.create_session(
            provider="anthropic",
            model="claude-sonnet-4-5",
            name="My Session"
        )
        await repo.save_session(session)

        # List sessions
        sessions = await repo.list_sessions(limit=10)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the session repository.

        Args:
            db_path: Path to database (legacy parameter, now uses unified database)
        """
        from victor.ui.tui.session import SessionManager

        self._manager = SessionManager(db_path=db_path)

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID.

        Args:
            session_id: Unique session identifier

        Returns:
            Session data as dictionary, or None if not found
        """
        from victor.ui.tui.session import Session

        session = self._manager.load(session_id)
        if session is None:
            return None

        # Convert Session object to dictionary
        return self._session_to_dict(session)

    async def save_session(self, session: Dict[str, Any]) -> None:
        """Save or update a session.

        Args:
            session: Session data dictionary

        Raises:
            ValueError: If session data is invalid
        """
        from victor.ui.tui.session import Session

        # Convert dict to Session object
        session_obj = self._dict_to_session(session)

        # Save using SessionManager
        self._manager.save(session_obj)

    async def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent sessions with metadata.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata dictionaries
        """
        # SessionManager.list_sessions returns list[dict[str, Any]]
        sessions = self._manager.list_sessions(limit=limit)

        # The returned dictionaries already have the required format
        # but use 'id' instead of 'session_id' in some places
        return sessions

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: Unique session identifier

        Returns:
            True if deleted, False if not found
        """
        return self._manager.delete(session_id)

    async def get_latest_session(self) -> Optional[Dict[str, Any]]:
        """Get the most recently updated session.

        Returns:
            Session data dictionary, or None if no sessions
        """
        from victor.ui.tui.session import Session

        session = self._manager.get_latest()
        if session is None:
            return None

        return self._session_to_dict(session)

    def create_session(
        self,
        provider: str = "",
        model: str = "",
        name: str = "",
    ) -> Dict[str, Any]:
        """Factory method to create a new session.

        Args:
            provider: LLM provider name
            model: Model name
            name: Optional session name

        Returns:
            New session dictionary
        """
        from victor.ui.tui.session import Session

        # Use SessionManager factory method
        session = self._manager.create_session(
            provider=provider,
            model=model,
            name=name,
        )

        return self._session_to_dict(session)

    def _session_to_dict(self, session: Any) -> Dict[str, Any]:
        """Convert Session object to dictionary.

        Args:
            session: Session object from SessionManager

        Returns:
            Session data as dictionary
        """
        # Session objects have to_dict() method
        if hasattr(session, "to_dict"):
            from typing import cast
            return cast(Dict[str, Any], session.to_dict())

        # Fallback: try to convert manually
        return {
            "id": getattr(session, "id", str(uuid.uuid4())),
            "name": getattr(session, "name", ""),
            "provider": getattr(session, "provider", ""),
            "model": getattr(session, "model", ""),
            "created_at": getattr(session, "created_at", datetime.now().isoformat()),
            "updated_at": getattr(session, "updated_at", datetime.now().isoformat()),
            "messages": [
                msg.to_dict() if hasattr(msg, "to_dict") else msg
                for msg in getattr(session, "messages", [])
            ],
            "metadata": getattr(session, "metadata", {}),
        }

    def _dict_to_session(self, session_dict: Dict[str, Any]) -> Any:
        """Convert dictionary to Session object.

        Args:
            session_dict: Session data dictionary

        Returns:
            Session object for SessionManager
        """
        from victor.ui.tui.session import Session

        # Session has from_dict() class method
        return Session.from_dict(session_dict)


__all__ = [
    "SQLiteSessionRepository",
]
