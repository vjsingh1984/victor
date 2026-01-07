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

"""SQLite-based session persistence for Victor conversations.

This module provides SQLiteSessionPersistence, which stores sessions
and messages in the project database (.victor/project.db), eliminating
duplication with JSON file storage.

Key Features:
- Single source of truth for sessions in SQLite
- Fast queries and filtering
- Efficient storage with indexing
- Fallback to JSON for migration/import/export
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.providers.base import Message

logger = logging.getLogger(__name__)


class SQLiteSessionPersistence:
    """SQLite-based session persistence.

    Stores conversation sessions and messages in the project database,
    providing fast queries and eliminating JSON file duplication.

    Tables Used:
    - sessions: id, name, provider, model, profile, data, created_at, updated_at
    - messages: id, session_id, role, content, tool_calls, created_at

    Example:
        persistence = SQLiteSessionPersistence()

        # Save session
        session_id = persistence.save_session(
            conversation=message_history,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            title="Code refactoring session"
        )

        # List sessions
        sessions = persistence.list_sessions(limit=10)

        # Load session
        session = persistence.load_session(session_id)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize SQLite session persistence.

        Args:
            db_path: Path to project database (default: .victor/project.db)
        """
        from victor.config.settings import get_project_paths
        from victor.core.database import get_project_database

        if db_path:
            self._db_path = db_path
        else:
            self._db_path = get_project_paths().project_root / ".victor" / "project.db"

        self._db = get_project_database()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure required tables exist."""
        # Tables should already be created by DatabaseManager
        # Just verify connection
        try:
            result = self._db.query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions', 'messages')"
            )
            tables = [row[0] for row in result] if result else []

            if "sessions" not in tables or "messages" not in tables:
                logger.warning("Sessions or messages table missing, initializing...")
                self._init_tables()
        except Exception as e:
            logger.error(f"Error checking tables: {e}")
            self._init_tables()

    def _init_tables(self) -> None:
        """Initialize sessions and messages tables."""
        from victor.core.schema import Schema

        try:
            # Create sessions table
            self._db.execute(Schema.CONV_SESSION)
            # Create messages table
            self._db.execute(Schema.CONV_MESSAGE)
            logger.info("Initialized sessions and messages tables")
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")

    def save_session(
        self,
        conversation: Any,  # MessageHistory or dict
        model: str,
        provider: str,
        profile: str = "default",
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        conversation_state: Optional[Any] = None,
        tool_selection_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a session to SQLite database.

        Args:
            conversation: MessageHistory instance or dict
            model: Model name used in the session
            provider: Provider name
            profile: Profile name
            session_id: Optional custom session ID (auto-generated if not provided)
            title: Optional custom title (auto-generated if not provided)
            tags: Optional list of tags
            conversation_state: Optional ConversationStateMachine state
            tool_selection_stats: Optional tool usage statistics

        Returns:
            The session ID
        """
        # Convert conversation to dict if needed
        if hasattr(conversation, "to_dict"):
            conversation_data = conversation.to_dict()
        elif isinstance(conversation, dict):
            conversation_data = conversation
        else:
            conversation_data = {"messages": []}

        # Generate session ID if not provided
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate title if not provided
        if not title:
            title = self._generate_title(conversation_data)

        # Get messages
        messages = conversation_data.get("messages", [])
        message_count = len(messages)

        # Timestamps
        now = datetime.now().isoformat()

        # Prepare session data (JSON format for storage)
        session_data = {
            "metadata": {
                "session_id": session_id,
                "created_at": now,
                "updated_at": now,
                "model": model,
                "provider": provider,
                "profile": profile,
                "message_count": message_count,
                "title": title,
                "tags": tags or [],
            },
            "conversation": conversation_data,
            "conversation_state": (
                conversation_state.to_dict() if hasattr(conversation_state, "to_dict") else conversation_state
            ),
            "tool_selection_stats": tool_selection_stats,
        }

        try:
            # Insert or replace session
            self._db.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, name, provider, model, profile, data, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    title,
                    provider,
                    model,
                    profile,
                    json.dumps(session_data),
                    now,
                    now,
                ),
            )

            # Delete existing messages for this session (if updating)
            self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

            # Insert messages
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls")

                # Serialize tool_calls if present
                tool_calls_json = json.dumps(tool_calls) if tool_calls else None

                self._db.execute(
                    """INSERT INTO messages (session_id, role, content, tool_calls)
                       VALUES (?, ?, ?, ?)""",
                    (session_id, role, content, tool_calls_json),
                )

            logger.info(f"Saved session {session_id} to SQLite ({message_count} messages)")
            return session_id

        except Exception as e:
            logger.error(f"Failed to save session to SQLite: {e}")
            return ""

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session from SQLite.

        Args:
            session_id: Session ID to load

        Returns:
            Session dictionary or None if not found
        """
        try:
            # Load session metadata
            result = self._db.query(
                "SELECT id, name, provider, model, profile, data, created_at, updated_at "
                "FROM sessions WHERE id = ?",
                (session_id,),
            )

            rows = list(result) if result else []
            if not rows:
                logger.warning(f"Session not found: {session_id}")
                return None

            row = rows[0]
            session_data = json.loads(row[6])  # data column

            logger.info(f"Loaded session {session_id} from SQLite")
            return session_data

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List sessions from SQLite.

        Args:
            limit: Maximum number of sessions to return
            offset: Offset for pagination

        Returns:
            List of session dictionaries with metadata
        """
        try:
            result = self._db.query(
                """SELECT id, name, provider, model, profile, created_at, updated_at
                   FROM sessions
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            )

            sessions = []
            if result:
                for row in result:
                    # Get message count
                    count_result = self._db.query(
                        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (row[0],)
                    )
                    message_count = count_result[0][0] if count_result else 0

                    sessions.append({
                        "session_id": row[0],
                        "title": row[1],
                        "provider": row[2],
                        "model": row[3],
                        "profile": row[4],
                        "created_at": row[5],
                        "updated_at": row[6],
                        "message_count": message_count,
                    })

            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from SQLite.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            # Delete messages first (foreign key)
            self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            # Delete session
            self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

            logger.info(f"Deleted session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def search_sessions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search sessions by title or content.

        Args:
            query: Search query string
            limit: Maximum results

        Returns:
            List of matching sessions
        """
        try:
            search_pattern = f"%{query}%"

            # Search in session titles
            result = self._db.query(
                """SELECT id, name, provider, model, profile, created_at, updated_at
                   FROM sessions
                   WHERE name LIKE ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (search_pattern, limit),
            )

            sessions = []
            if result:
                for row in result:
                    count_result = self._db.query(
                        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (row[0],)
                    )
                    message_count = count_result[0][0] if count_result else 0

                    sessions.append({
                        "session_id": row[0],
                        "title": row[1],
                        "provider": row[2],
                        "model": row[3],
                        "profile": row[4],
                        "created_at": row[5],
                        "updated_at": row[6],
                        "message_count": message_count,
                    })

            return sessions

        except Exception as e:
            logger.error(f"Failed to search sessions: {e}")
            return []

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session.

        Args:
            session_id: Session ID

        Returns:
            List of message dictionaries
        """
        try:
            result = self._db.query(
                """SELECT role, content, tool_calls, created_at
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY created_at ASC""",
                (session_id,),
            )

            messages = []
            if result:
                for row in result:
                    msg = {
                        "role": row[0],
                        "content": row[1],
                        "created_at": row[3],
                    }

                    # Parse tool_calls if present
                    if row[2]:
                        try:
                            msg["tool_calls"] = json.loads(row[2])
                        except:
                            msg["tool_calls"] = None

                    messages.append(msg)

            return messages

        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {e}")
            return []

    def _generate_title(self, conversation_data: Dict[str, Any]) -> str:
        """Generate a title from the first user message.

        Args:
            conversation_data: Serialized conversation data

        Returns:
            A title string (truncated to 50 chars)
        """
        messages = conversation_data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                # Truncate to 50 chars
                if len(content) > 50:
                    return content[:47] + "..."
                return content
        return "Untitled Session"


def get_sqlite_session_persistence() -> SQLiteSessionPersistence:
    """Get the SQLite session persistence instance.

    Returns:
        SQLiteSessionPersistence instance
    """
    return SQLiteSessionPersistence()


__all__ = [
    "SQLiteSessionPersistence",
    "get_sqlite_session_persistence",
]
