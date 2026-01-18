"""Session persistence for Victor TUI.

Provides auto-save/restore functionality for conversations,
allowing users to resume where they left off.

Database:
    Uses the unified database at ~/.victor/victor.db via victor.core.database.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from victor.core.database import get_database
from victor.ui.common.constants import (
    CURSOR_COL_INDEX,
    CURSOR_ROW_INDEX,
    SESSION_CREATED_AT_INDEX,
    SESSION_DATA_INDEX,
    SESSION_ID_INDEX,
    SESSION_MODEL_INDEX,
    SESSION_NAME_INDEX,
    SESSION_PROVIDER_INDEX,
    SESSION_UPDATED_AT_INDEX,
)
from victor.core.schema import Tables


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # user, assistant, system, error
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """A conversation session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    provider: str = ""
    model: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    messages: list[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_message(self, role: str, content: str, **metadata) -> Message:
        """Add a message to the session."""
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.updated_at = datetime.now().isoformat()
        return msg

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            metadata=data.get("metadata", {}),
        )

    def to_markdown(self) -> str:
        """Export session to markdown format."""
        lines = [
            f"# Victor Session: {self.name or self.id[:8]}",
            "",
            f"**Provider**: {self.provider}",
            f"**Model**: {self.model}",
            f"**Created**: {self.created_at}",
            f"**Updated**: {self.updated_at}",
            "",
            "---",
            "",
        ]

        for msg in self.messages:
            role_display = {
                "user": "You",
                "assistant": "Victor",
                "system": "System",
                "error": "Error",
            }.get(msg.role, msg.role.title())

            lines.append(f"### {role_display}")
            lines.append("")
            lines.append(msg.content)
            lines.append("")

        return "\n".join(lines)


class SessionManager:
    """Manages session persistence using SQLite.

    This class now supports dependency injection of a SessionRepositoryProtocol
    for decoupling from database implementation. If no repository is provided,
    it uses direct database access for backward compatibility.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        repository: Optional[Any] = None,
    ):
        """Initialize session manager.

        Args:
            db_path: Path to SQLite database - legacy, now uses unified database
            repository: Optional SessionRepositoryProtocol for dependency injection
        """
        # Store repository if provided (DIP compliance)
        self._repository = repository

        # Use unified database from victor.core.database (legacy path)
        if self._repository is None:
            self._db_manager = get_database()
            self.db_path = self._db_manager.db_path
            self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._repository is not None:
            raise RuntimeError("Cannot get DB connection when using repository pattern")
        return self._db_manager.get_connection()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {Tables.UI_SESSION} (
                id TEXT PRIMARY KEY,
                name TEXT,
                provider TEXT,
                model TEXT,
                profile TEXT,
                created_at TEXT,
                updated_at TEXT,
                data TEXT
            )
        """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_ui_session_updated
            ON {Tables.UI_SESSION}(updated_at DESC)
        """
        )
        conn.commit()

    def save(self, session: Session) -> None:
        """Save or update a session."""
        import asyncio

        session.updated_at = datetime.now().isoformat()

        # Use repository if available
        if self._repository is not None:
            # Convert Session to dict and save via repository
            session_dict = session.to_dict()

            # Handle sync/async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create task
                    asyncio.create_task(self._repository.save_session(session_dict))
                else:
                    # We're in sync context, run async
                    loop.run_until_complete(self._repository.save_session(session_dict))
            except RuntimeError:
                # No event loop, create new one
                asyncio.run(self._repository.save_session(session_dict))
            return

        # Legacy direct database access
        conn = self._get_conn()
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.UI_SESSION}
            (id, name, provider, model, created_at, updated_at, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session.id,
                session.name,
                session.provider,
                session.model,
                session.created_at,
                session.updated_at,
                json.dumps(session.to_dict()),
            ),
        )
        conn.commit()

    def load(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        import asyncio

        # Use repository if available
        if self._repository is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # In async context, need to handle differently
                    # For now, fall back to creating new loop
                    session_dict = asyncio.run(self._repository.get_session(session_id))
                else:
                    session_dict = loop.run_until_complete(self._repository.get_session(session_id))
            except RuntimeError:
                session_dict = asyncio.run(self._repository.get_session(session_id))

            if session_dict is None:
                return None
            return Session.from_dict(session_dict)

        # Legacy direct database access
        conn = self._get_conn()
        cursor = conn.execute(
            f"SELECT data FROM {Tables.UI_SESSION} WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row:
            # When selecting only data column, it's at index 0
            return Session.from_dict(json.loads(row[0]))
        return None

    def get_latest(self) -> Optional[Session]:
        """Get the most recently updated session."""
        import asyncio

        # Use repository if available
        if self._repository is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    session_dict = asyncio.run(self._repository.get_latest_session())
                else:
                    session_dict = loop.run_until_complete(self._repository.get_latest_session())
            except RuntimeError:
                session_dict = asyncio.run(self._repository.get_latest_session())

            if session_dict is None:
                return None
            return Session.from_dict(session_dict)

        # Legacy direct database access
        conn = self._get_conn()
        cursor = conn.execute(
            f"SELECT data FROM {Tables.UI_SESSION} ORDER BY updated_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            # When selecting only data column, it's at index 0
            return Session.from_dict(json.loads(row[0]))
        return None

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent sessions (metadata only, not full messages).

        Returns:
            List of session metadata dicts with id, name, provider, model,
            created_at, updated_at, and message_count.
        """
        import asyncio

        # Use repository if available
        if self._repository is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    sessions = asyncio.run(self._repository.list_sessions(limit=limit))
                else:
                    sessions = loop.run_until_complete(self._repository.list_sessions(limit=limit))
            except RuntimeError:
                sessions = asyncio.run(self._repository.list_sessions(limit=limit))
            return sessions

        # Legacy direct database access
        conn = self._get_conn()
        cursor = conn.execute(
            f"""
            SELECT id, name, provider, model, created_at, updated_at, data
            FROM {Tables.UI_SESSION}
            ORDER BY updated_at DESC
            LIMIT ?
        """,
            (limit,),
        )
        sessions = []
        for row in cursor:
            data = json.loads(row[SESSION_DATA_INDEX])
            sessions.append(
                {
                    "id": row[SESSION_ID_INDEX],
                    "name": row[SESSION_NAME_INDEX] or f"Session {row[SESSION_ID_INDEX][:8]}",
                    "provider": row[SESSION_PROVIDER_INDEX],
                    "model": row[SESSION_MODEL_INDEX],
                    "created_at": row[SESSION_CREATED_AT_INDEX],
                    "updated_at": row[SESSION_UPDATED_AT_INDEX],
                    "message_count": len(data.get("messages", [])),
                }
            )
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session by ID.

        Returns:
            True if session was deleted, False if not found.
        """
        import asyncio

        # Use repository if available
        if self._repository is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    deleted = asyncio.run(self._repository.delete_session(session_id))
                else:
                    deleted = loop.run_until_complete(self._repository.delete_session(session_id))
            except RuntimeError:
                deleted = asyncio.run(self._repository.delete_session(session_id))
            return deleted

        # Legacy direct database access
        conn = self._get_conn()
        cursor = conn.execute(
            f"DELETE FROM {Tables.UI_SESSION} WHERE id = ?",
            (session_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def export_markdown(self, session_id: str, output_path: Path) -> bool:
        """Export a session to markdown file.

        Args:
            session_id: Session ID to export
            output_path: Path to write markdown file

        Returns:
            True if exported successfully, False if session not found.
        """
        session = self.load(session_id)
        if not session:
            return False

        output_path.write_text(session.to_markdown())
        return True

    def create_session(
        self,
        provider: str = "",
        model: str = "",
        name: str = "",
    ) -> Session:
        """Create a new session.

        Args:
            provider: Provider name
            model: Model name
            name: Optional session name

        Returns:
            New Session instance (not yet saved)
        """
        return Session(
            name=name,
            provider=provider,
            model=model,
        )
