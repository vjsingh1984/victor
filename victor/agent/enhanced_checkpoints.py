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

"""Enhanced checkpoint system with SQLite backend for time-travel debugging.

This module extends the git-based checkpoint system with fine-grained state
persistence using SQLite, enabling true time-travel debugging similar to LangGraph.

**Design Pattern**: Memento pattern with dual backend (Git + SQLite)

**Key Features**:
1. Git backend: Working tree snapshots (existing GitCheckpointManager)
2. SQLite backend: Conversation/execution state snapshots
3. Automatic checkpointing: Every N tool calls
4. State serialization: Full conversation context and execution state
5. Time-travel debugging: Fork sessions, restore to any checkpoint

**Usage**:
    manager = EnhancedCheckpointManager(auto_checkpoint=True, checkpoint_interval=5)

    # Manual checkpoint
    cp = manager.save_checkpoint("Before refactoring", state={"context": {...}})

    # Restore checkpoint
    state = manager.load_checkpoint(cp.id)

    # List checkpoints
    checkpoints = manager.list_checkpoints()

    # Fork a session from checkpoint
    new_session_id = manager.fork_session(cp.id)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.providers.base import Message

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Serialized state for checkpointing.

    Attributes:
        session_id: Unique session identifier
        messages: Conversation messages
        context: Conversation context (variables, metadata)
        execution_state: Workflow/agent execution state
        tool_calls: Tool call history
        timestamp: When checkpoint was created
        metadata: Additional checkpoint metadata
    """

    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    execution_state: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "context": self.context,
            "execution_state": self.execution_state,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        return cls(
            session_id=data["session_id"],
            messages=data.get("messages", []),
            context=data.get("context", {}),
            execution_state=data.get("execution_state", {}),
            tool_calls=data.get("tool_calls", []),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Checkpoint:
    """Enhanced checkpoint with both git and state information.

    Attributes:
        id: Unique checkpoint identifier
        timestamp: When checkpoint was created
        description: Human-readable description
        git_checkpoint_id: Git stash checkpoint ID (if available)
        state: Serialized checkpoint state
        size_bytes: Approximate size in bytes
    """

    id: str
    timestamp: datetime
    description: str
    git_checkpoint_id: Optional[str]
    state: CheckpointState
    size_bytes: int


class StateSerializer:
    """Serializes and deserializes conversation/execution state.

    Handles conversion of complex objects to/from JSON-serializable format.
    """

    @staticmethod
    def serialize_messages(messages: List["Message"]) -> List[Dict[str, Any]]:
        """Serialize messages to JSON-serializable format.

        Args:
            messages: List of Message objects

        Returns:
            List of dictionaries with message data
        """
        serialized = []
        for msg in messages:
            serialized.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": getattr(msg, "tool_calls", None),
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                }
            )
        return serialized

    @staticmethod
    def deserialize_messages(data: List[Dict[str, Any]]) -> List["Message"]:
        """Deserialize messages from JSON format.

        Args:
            data: List of message dictionaries

        Returns:
            List of Message objects
        """
        from victor.providers.base import Message

        messages = []
        for msg_data in data:
            msg = Message(
                role=msg_data["role"],
                content=msg_data["content"],
            )
            if msg_data.get("tool_calls"):
                msg.tool_calls = msg_data["tool_calls"]
            if msg_data.get("tool_call_id"):
                msg.tool_call_id = msg_data["tool_call_id"]
            messages.append(msg)
        return messages

    @staticmethod
    def serialize_context(context: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize context to JSON-serializable format.

        Args:
            context: Context dictionary

        Returns:
            JSON-serializable dictionary
        """
        # Convert datetime objects to ISO format
        serialized = {}
        for key, value in context.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif hasattr(value, "to_dict"):
                serialized[key] = value.to_dict()
            elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serialized[key] = value
            else:
                # Fallback: convert to string
                serialized[key] = str(value)
        return serialized

    @staticmethod
    def serialize_execution_state(state: Any) -> Dict[str, Any]:
        """Serialize execution state to JSON format.

        Args:
            state: Execution state object

        Returns:
            JSON-serializable dictionary
        """
        if hasattr(state, "to_dict"):
            return state.to_dict()
        elif isinstance(state, dict):
            return StateSerializer.serialize_context(state)
        else:
            return {"state": str(state)}


class SQLiteCheckpointBackend:
    """SQLite backend for fine-grained checkpoint storage.

    Stores serialized checkpoint state in SQLite database for fast
    retrieval and querying.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database (default: ~/.victor/checkpoints.db)
        """
        if db_path is None:
            # Default to ~/.victor/checkpoints.db
            victor_dir = Path.home() / ".victor"
            victor_dir.mkdir(exist_ok=True)
            db_path = victor_dir / "checkpoints.db"

        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    description TEXT,
                    state_json TEXT NOT NULL,
                    size_bytes INTEGER,
                    git_checkpoint_id TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON checkpoints(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON checkpoints(timestamp DESC)")
            conn.commit()

    def save_checkpoint(
        self,
        checkpoint_id: str,
        session_id: str,
        description: str,
        state: CheckpointState,
        git_checkpoint_id: Optional[str] = None,
    ) -> None:
        """Save checkpoint to database.

        Args:
            checkpoint_id: Unique checkpoint identifier
            session_id: Session identifier
            description: Human-readable description
            state: Checkpoint state
            git_checkpoint_id: Associated git checkpoint ID
        """
        state_json = json.dumps(state.to_dict())
        size_bytes = len(state_json.encode("utf-8"))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO checkpoints
                (id, session_id, timestamp, description, state_json, size_bytes, git_checkpoint_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    session_id,
                    state.timestamp.isoformat(),
                    description,
                    state_json,
                    size_bytes,
                    git_checkpoint_id,
                ),
            )
            conn.commit()

    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointState]:
        """Load checkpoint state from database.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            CheckpointState if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT state_json FROM checkpoints WHERE id = ?",
                (checkpoint_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            state_dict = json.loads(row[0])
            return CheckpointState.from_dict(state_dict)

    def list_checkpoints(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List checkpoints from database.

        Args:
            session_id: Filter by session ID (optional)
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoint dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            if session_id:
                cursor = conn.execute(
                    """
                    SELECT id, session_id, timestamp, description, size_bytes, git_checkpoint_id
                    FROM checkpoints
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, session_id, timestamp, description, size_bytes, git_checkpoint_id
                    FROM checkpoints
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            checkpoints = []
            for row in cursor.fetchall():
                checkpoints.append(
                    {
                        "id": row[0],
                        "session_id": row[1],
                        "timestamp": row[2],
                        "description": row[3],
                        "size_bytes": row[4],
                        "git_checkpoint_id": row[5],
                    }
                )

            return checkpoints

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from database.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM checkpoints WHERE id = ?",
                (checkpoint_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_old_checkpoints(self, keep_count: int = 20) -> int:
        """Delete old checkpoints, keeping N most recent per session.

        Args:
            keep_count: Number of checkpoints to keep per session

        Returns:
            Number of checkpoints deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            # Delete checkpoints that are not in the top N for their session
            cursor = conn.execute(
                f"""
                DELETE FROM checkpoints
                WHERE id NOT IN (
                    SELECT id FROM checkpoints AS c1
                    WHERE session_id = checkpoints.session_id
                    ORDER BY timestamp DESC
                    LIMIT {keep_count}
                )
                """
            )
            conn.commit()
            return cursor.rowcount


class EnhancedCheckpointManager:
    """Enhanced checkpoint manager with dual backend (Git + SQLite).

    Combines git-based working tree snapshots with SQLite-based state
    persistence for comprehensive time-travel debugging.

    Attributes:
        auto_checkpoint: Automatically create checkpoints
        checkpoint_interval: Create checkpoint every N tool calls
        _tool_call_count: Counter for auto-checkpointing
    """

    def __init__(
        self,
        repo_path: str = ".",
        auto_checkpoint: bool = False,
        checkpoint_interval: int = 5,
        db_path: Optional[Path] = None,
    ):
        """Initialize enhanced checkpoint manager.

        Args:
            repo_path: Path to git repository
            auto_checkpoint: Enable automatic checkpointing
            checkpoint_interval: Checkpoint every N tool calls
            db_path: Path to SQLite database
        """
        from victor.agent.checkpoints import GitCheckpointManager

        self.git_manager = GitCheckpointManager(repo_path=repo_path)
        self.sqlite_backend = SQLiteCheckpointBackend(db_path=db_path)

        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self._tool_call_count = 0

    def save_checkpoint(
        self,
        description: str,
        session_id: str,
        state: CheckpointState,
    ) -> Checkpoint:
        """Save a checkpoint with both git and state persistence.

        Args:
            description: Human-readable description
            session_id: Session identifier
            state: Checkpoint state to persist

        Returns:
            Checkpoint object with metadata

        Example:
            >>> manager = EnhancedCheckpointManager()
            >>> state = CheckpointState(session_id="abc", messages=[...])
            >>> cp = manager.save_checkpoint("Before refactoring", session_id="abc", state=state)
            >>> print(cp.id)
        """
        # Generate unique checkpoint ID
        checkpoint_id = f"checkpoint_{uuid.uuid4().hex[:12]}"

        # Create git checkpoint for working tree
        try:
            git_checkpoint = self.git_manager.create(description)
            git_checkpoint_id = git_checkpoint.id
        except Exception as e:
            logger.warning(f"Failed to create git checkpoint: {e}")
            git_checkpoint_id = None

        # Save state to SQLite
        self.sqlite_backend.save_checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            description=description,
            state=state,
            git_checkpoint_id=git_checkpoint_id,
        )

        logger.info(f"Saved checkpoint {checkpoint_id}: {description}")

        return Checkpoint(
            id=checkpoint_id,
            timestamp=state.timestamp,
            description=description,
            git_checkpoint_id=git_checkpoint_id,
            state=state,
            size_bytes=0,  # Would be calculated by backend
        )

    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointState]:
        """Load checkpoint state from database.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            CheckpointState if found, None otherwise
        """
        return self.sqlite_backend.load_checkpoint(checkpoint_id)

    def restore_checkpoint(self, checkpoint_id: str, restore_git: bool = True) -> bool:
        """Restore to a checkpoint state.

        Args:
            checkpoint_id: Checkpoint to restore
            restore_git: Also restore git working tree (default: True)

        Returns:
            True if restore succeeded
        """
        # Load state
        state = self.load_checkpoint(checkpoint_id)
        if not state:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False

        # Restore git state if requested
        if restore_git and state.metadata.get("git_checkpoint_id"):
            try:
                self.git_manager.rollback(state.metadata["git_checkpoint_id"])
            except Exception as e:
                logger.warning(f"Failed to restore git state: {e}")

        logger.info(f"Restored checkpoint {checkpoint_id}")
        return True

    def list_checkpoints(
        self,
        session_id: Optional[str] = None,
        include_git: bool = True,
    ) -> List[Dict[str, Any]]:
        """List all checkpoints.

        Args:
            session_id: Filter by session ID
            include_git: Include git checkpoints in results

        Returns:
            List of checkpoint information
        """
        # Get SQLite checkpoints
        checkpoints = self.sqlite_backend.list_checkpoints(session_id=session_id)

        # Optionally include git-only checkpoints
        if include_git:
            git_checkpoints = self.git_manager.list_checkpoints()
            for git_cp in git_checkpoints:
                # Check if already in list
                if not any(cp.get("git_checkpoint_id") == git_cp.id for cp in checkpoints):
                    checkpoints.append(
                        {
                            "id": git_cp.id,
                            "session_id": "N/A",
                            "timestamp": git_cp.timestamp.isoformat(),
                            "description": git_cp.description,
                            "git_only": True,
                        }
                    )

        return checkpoints

    def fork_session(self, checkpoint_id: str) -> str:
        """Create a new session forked from a checkpoint.

        Args:
            checkpoint_id: Checkpoint to fork from

        Returns:
            New session ID
        """
        # Load checkpoint state
        state = self.load_checkpoint(checkpoint_id)
        if not state:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        # Create new session with checkpoint state
        new_session_id = f"forked_{uuid.uuid4().hex[:12]}"

        # Create new checkpoint in new session
        self.save_checkpoint(
            description=f"Forked from {checkpoint_id}",
            session_id=new_session_id,
            state=CheckpointState(
                session_id=new_session_id,
                messages=state.messages.copy(),
                context=state.context.copy(),
                execution_state=state.execution_state.copy(),
                tool_calls=state.tool_calls.copy(),
                metadata={"forked_from": checkpoint_id},
            ),
        )

        logger.info(f"Forked session {new_session_id} from checkpoint {checkpoint_id}")
        return new_session_id

    def record_tool_call(self) -> Optional[str]:
        """Record a tool call and auto-checkpoint if needed.

        Should be called after each tool execution.

        Returns:
            Checkpoint ID if auto-checkpoint was created, None otherwise
        """
        if not self.auto_checkpoint:
            return None

        self._tool_call_count += 1

        if self._tool_call_count >= self.checkpoint_interval:
            self._tool_call_count = 0

            # Create auto-checkpoint
            checkpoint_id = f"auto_{uuid.uuid4().hex[:12]}"
            state = CheckpointState(
                session_id="current",
                metadata={"auto_checkpoint": True, "tool_call_count": self._tool_call_count},
            )

            self.save_checkpoint(
                description=f"Auto-checkpoint after {self.checkpoint_interval} tool calls",
                session_id="current",
                state=state,
            )

            logger.info(f"Auto-checkpoint created: {checkpoint_id}")
            return checkpoint_id

        return None

    def cleanup_old(self, keep_count: int = 20) -> Dict[str, int]:
        """Clean up old checkpoints from both backends.

        Args:
            keep_count: Number of checkpoints to keep

        Returns:
            Dictionary with cleanup counts
        """
        # Clean up SQLite checkpoints
        sqlite_removed = self.sqlite_backend.delete_old_checkpoints(keep_count=keep_count)

        # Clean up git checkpoints
        git_removed = self.git_manager.cleanup_old(keep_count=keep_count)

        logger.info(f"Cleaned up {sqlite_removed} SQLite and {git_removed} git checkpoints")

        return {"sqlite": sqlite_removed, "git": git_removed}


__all__ = [
    "CheckpointState",
    "Checkpoint",
    "StateSerializer",
    "SQLiteCheckpointBackend",
    "EnhancedCheckpointManager",
]
