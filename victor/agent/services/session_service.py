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

"""Session service implementation.

Extracts session management from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Session creation and initialization
- Session state management
- Session persistence and restoration
- Session cleanup and disposal
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SessionInfoImpl:
    """Implementation of session information."""

    def __init__(
        self,
        session_id: str,
        created_at: datetime,
        last_activity: datetime,
        message_count: int = 0,
        tool_calls: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.created_at = created_at
        self.last_activity = last_activity
        self.message_count = message_count
        self.tool_calls = tool_calls
        self.metadata = metadata or {}


class SessionService:
    """Service for session lifecycle and management.

    Extracted from AgentOrchestrator to handle:
    - Session creation and initialization
    - Session state management
    - Session persistence and restoration
    - Session cleanup and disposal

    This service follows SOLID principles:
    - SRP: Only handles session operations
    - OCP: Extensible through storage abstraction
    - LSP: Implements SessionServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        service = SessionService()
        session_id = await service.create_session({"user_id": "123"})
        info = await service.get_session(session_id)
    """

    def __init__(
        self,
        storage: Optional[Any] = None,
        session_timeout_seconds: int = 3600,
    ):
        """Initialize the session service.

        Args:
            storage: Optional storage backend for persistence
            session_timeout_seconds: Default session timeout
        """
        self._storage = storage
        self._session_timeout = session_timeout_seconds
        self._sessions: Dict[str, SessionInfoImpl] = {}
        self._current_session_id: Optional[str] = None
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    async def create_session(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new session.

        Args:
            metadata: Optional session metadata

        Returns:
            Session ID for the new session
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()

        session = SessionInfoImpl(
            session_id=session_id,
            created_at=now,
            last_activity=now,
            metadata=metadata or {},
        )

        self._sessions[session_id] = session
        self._current_session_id = session_id

        # Persist if storage available
        if self._storage:
            await self._storage.save(session_id, session)

        self._logger.info(f"Created session: {session_id}")
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionInfoImpl]:
        """Get information about a session.

        Args:
            session_id: Session ID to retrieve

        Returns:
            SessionInfo if session exists, None otherwise
        """
        # Check in-memory cache first
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try loading from storage
        if self._storage:
            session = await self._storage.load(session_id)
            if session:
                self._sessions[session_id] = session
                return session

        return None

    async def update_session(
        self,
        session_id: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Update session metadata.

        Args:
            session_id: Session ID to update
            metadata: Metadata to update

        Returns:
            True if session was updated, False if not found
        """
        session = await self.get_session(session_id)
        if session is None:
            return False

        # Update metadata
        session.metadata.update(metadata)
        session.last_activity = datetime.now()

        # Persist if storage available
        if self._storage:
            await self._storage.save(session_id, session)

        return True

    async def close_session(self, session_id: str) -> bool:
        """Close and cleanup a session.

        Args:
            session_id: Session ID to close

        Returns:
            True if session was closed, False if not found
        """
        if session_id not in self._sessions:
            return False

        # Remove from memory
        del self._sessions[session_id]

        # Clear from storage if available
        if self._storage:
            await self._storage.delete(session_id)

        # Clear current if it was the active session
        if self._current_session_id == session_id:
            self._current_session_id = None

        self._logger.info(f"Closed session: {session_id}")
        return True

    def get_current_session_id(self) -> Optional[str]:
        """Get the current active session ID.

        Returns:
            Current session ID, or None if no active session
        """
        return self._current_session_id

    async def list_sessions(
        self,
        state: Optional[str] = None,
        limit: int = 100,
    ) -> List[SessionInfoImpl]:
        """List sessions.

        Args:
            state: Optional state filter (not yet implemented)
            limit: Maximum number of sessions to return

        Returns:
            List of session information
        """
        sessions = list(self._sessions.values())
        return sessions[:limit]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session completely.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if not found
        """
        return await self.close_session(session_id)

    async def get_session_metrics(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Get detailed metrics for a session.

        Args:
            session_id: Session ID to get metrics for

        Returns:
            Dictionary with session metrics
        """
        session = await self.get_session(session_id)
        if session is None:
            return {}

        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": session.message_count,
            "tool_calls": session.tool_calls,
            "metadata": session.metadata,
        }

    def is_healthy(self) -> bool:
        """Check if the session service is healthy.

        Returns:
            True if the service is healthy
        """
        return True
