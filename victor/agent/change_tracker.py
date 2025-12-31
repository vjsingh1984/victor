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

"""File change tracking system for undo/redo functionality.

This module provides a robust system for tracking file modifications made by
Victor, enabling undo/redo operations similar to OpenCode's /undo and /redo.
"""

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of file changes."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


@dataclass
class FileChange:
    """Represents a single file change."""

    id: str
    change_type: ChangeType
    file_path: str
    timestamp: float
    tool_name: str
    tool_args: Dict[str, Any]
    original_content: Optional[str] = None  # Content before change
    new_content: Optional[str] = None  # Content after change
    original_path: Optional[str] = None  # For renames
    checksum_before: Optional[str] = None
    checksum_after: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None  # Links to conversation message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "change_type": self.change_type.value,
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "original_content": self.original_content,
            "new_content": self.new_content,
            "original_path": self.original_path,
            "checksum_before": self.checksum_before,
            "checksum_after": self.checksum_after,
            "session_id": self.session_id,
            "message_id": self.message_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileChange":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            change_type=ChangeType(data["change_type"]),
            file_path=data["file_path"],
            timestamp=data["timestamp"],
            tool_name=data["tool_name"],
            tool_args=data.get("tool_args", {}),
            original_content=data.get("original_content"),
            new_content=data.get("new_content"),
            original_path=data.get("original_path"),
            checksum_before=data.get("checksum_before"),
            checksum_after=data.get("checksum_after"),
            session_id=data.get("session_id"),
            message_id=data.get("message_id"),
        )


@dataclass
class ChangeGroup:
    """A group of related changes (e.g., from a single tool execution)."""

    id: str
    changes: List[FileChange] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    description: str = ""
    tool_name: str = ""
    undone: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "changes": [c.to_dict() for c in self.changes],
            "timestamp": self.timestamp,
            "description": self.description,
            "tool_name": self.tool_name,
            "undone": self.undone,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChangeGroup":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            changes=[FileChange.from_dict(c) for c in data.get("changes", [])],
            timestamp=data.get("timestamp", time.time()),
            description=data.get("description", ""),
            tool_name=data.get("tool_name", ""),
            undone=data.get("undone", False),
        )


class FileChangeHistory:
    """Tracks file changes for undo/redo functionality.

    Uses SQLite for persistent storage and maintains an in-memory
    stack for fast undo/redo operations.
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        max_history: int = 10000,
        session_id: Optional[str] = None,
        project_path: Optional[Path] = None,
    ):
        """Initialize the file change history.

        Args:
            storage_dir: Deprecated, ignored. Uses consolidated project.db.
            max_history: Maximum number of change groups to keep (default 10000 for historical analysis)
            session_id: Current session identifier
            project_path: Path to project root for database access.
        """
        from victor.config.settings import get_project_paths
        from victor.core.database import get_project_database

        self.storage_dir = storage_dir or get_project_paths().changes_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.session_id = session_id or self._generate_session_id()

        # In-memory stacks for fast operations
        self._undo_stack: List[ChangeGroup] = []
        self._redo_stack: List[ChangeGroup] = []

        # Current change group being built
        self._current_group: Optional[ChangeGroup] = None

        # Use consolidated project.db via ProjectDatabaseManager
        self._db = get_project_database(project_path)
        self._db_path = self._db.db_path
        self._init_database()

        # Load recent history
        self._load_recent_history()

        logger.info(f"ChangeTracker initialized with session {self.session_id}")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{int(time.time() * 1000)}"

    def _init_database(self) -> None:
        """Initialize database tables with correct schema."""
        conn = self._db.get_connection()

        # Check if tables need recreation (wrong schema)
        if self._needs_schema_rebuild(conn):
            conn.execute("DROP TABLE IF EXISTS file_changes")
            conn.execute("DROP TABLE IF EXISTS change_groups")
            conn.commit()

        # Create tables with correct schema
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS change_groups (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                timestamp REAL,
                description TEXT,
                tool_name TEXT,
                undone INTEGER DEFAULT 0,
                data TEXT
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_changes (
                id TEXT PRIMARY KEY,
                group_id TEXT,
                change_type TEXT,
                file_path TEXT,
                timestamp REAL,
                tool_name TEXT,
                original_content TEXT,
                new_content TEXT,
                original_path TEXT,
                checksum_before TEXT,
                checksum_after TEXT,
                FOREIGN KEY (group_id) REFERENCES change_groups(id)
            )
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_groups_session
            ON change_groups(session_id)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_groups_timestamp
            ON change_groups(timestamp DESC)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_changes_path
            ON file_changes(file_path)
        """
        )

        conn.commit()

    def _needs_schema_rebuild(self, conn: sqlite3.Connection) -> bool:
        """Check if tables exist with wrong schema and need rebuild."""
        try:
            # Check change_groups columns
            cursor = conn.execute("PRAGMA table_info(change_groups)")
            group_cols = {row[1] for row in cursor.fetchall()}
            required_group = {
                "id",
                "session_id",
                "timestamp",
                "description",
                "tool_name",
                "undone",
                "data",
            }
            if group_cols and not required_group.issubset(group_cols):
                return True

            # Check file_changes columns AND types (id must be TEXT, not INTEGER)
            cursor = conn.execute("PRAGMA table_info(file_changes)")
            change_cols = {row[1]: row[2] for row in cursor.fetchall()}
            if change_cols:
                # Check required columns exist
                required_change = {"id", "group_id", "change_type", "file_path", "timestamp"}
                if not required_change.issubset(change_cols.keys()):
                    return True
                # Check id column type is TEXT (not INTEGER)
                if change_cols.get("id", "").upper() != "TEXT":
                    return True

            return False
        except sqlite3.OperationalError:
            return False

    def _load_recent_history(self) -> None:
        """Load recent change history from database."""
        conn = self._db.get_connection()
        cursor = conn.cursor()

        # Load recent non-undone groups
        cursor.execute(
            """
            SELECT data FROM change_groups
            WHERE session_id = ? AND undone = 0
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (self.session_id, self.max_history),
        )

        rows = cursor.fetchall()
        for (data,) in reversed(rows):  # Reverse to maintain order
            try:
                group = ChangeGroup.from_dict(json.loads(data))
                self._undo_stack.append(group)
            except Exception as e:
                logger.warning(f"Failed to load change group: {e}")

        # Connection managed by ProjectDatabaseManager
        logger.debug(f"Loaded {len(self._undo_stack)} change groups from history")

    @staticmethod
    def compute_checksum(content: str) -> str:
        """Compute MD5 checksum of content."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def begin_change_group(self, tool_name: str, description: str = "") -> str:
        """Begin a new change group.

        Args:
            tool_name: Name of the tool making changes
            description: Human-readable description

        Returns:
            Group ID
        """
        group_id = f"grp_{int(time.time() * 1000)}_{tool_name}"
        self._current_group = ChangeGroup(
            id=group_id,
            tool_name=tool_name,
            description=description,
            timestamp=time.time(),
        )
        logger.debug(f"Started change group: {group_id}")
        return group_id

    def record_change(
        self,
        file_path: str,
        change_type: ChangeType,
        original_content: Optional[str] = None,
        new_content: Optional[str] = None,
        tool_name: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
        original_path: Optional[str] = None,
    ) -> FileChange:
        """Record a file change.

        Args:
            file_path: Path to the file being changed
            change_type: Type of change
            original_content: File content before change
            new_content: File content after change
            tool_name: Name of tool making the change
            tool_args: Arguments passed to the tool
            original_path: Original path (for renames)

        Returns:
            The recorded FileChange
        """
        change_id = f"chg_{int(time.time() * 1000)}_{os.path.basename(file_path)}"

        change = FileChange(
            id=change_id,
            change_type=change_type,
            file_path=str(file_path),
            timestamp=time.time(),
            tool_name=tool_name or (self._current_group.tool_name if self._current_group else ""),
            tool_args=tool_args or {},
            original_content=original_content,
            new_content=new_content,
            original_path=original_path,
            checksum_before=self.compute_checksum(original_content) if original_content else None,
            checksum_after=self.compute_checksum(new_content) if new_content else None,
            session_id=self.session_id,
        )

        if self._current_group:
            self._current_group.changes.append(change)
        else:
            # Auto-create a group for standalone changes
            self.begin_change_group(tool_name or "unknown", f"Changed {file_path}")
            if self._current_group:  # Always true after begin_change_group
                self._current_group.changes.append(change)
            self.commit_change_group()

        logger.debug(f"Recorded change: {change_type.value} on {file_path}")
        return change

    def commit_change_group(self) -> Optional[ChangeGroup]:
        """Commit the current change group.

        Returns:
            The committed ChangeGroup, or None if no current group
        """
        if not self._current_group or not self._current_group.changes:
            self._current_group = None
            return None

        group = self._current_group
        self._current_group = None

        # Add to undo stack
        self._undo_stack.append(group)

        # Clear redo stack on new change
        self._redo_stack.clear()

        # Persist to database
        self._save_group(group)

        # Trim history if needed
        self._trim_history()

        logger.info(f"Committed change group: {group.id} with {len(group.changes)} changes")
        return group

    def _save_group(self, group: ChangeGroup) -> None:
        """Save a change group to the database."""
        conn = self._db.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO change_groups
            (id, session_id, timestamp, description, tool_name, undone, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                group.id,
                self.session_id,
                group.timestamp,
                group.description,
                group.tool_name,
                1 if group.undone else 0,
                json.dumps(group.to_dict()),
            ),
        )

        # Also save individual changes for querying
        for change in group.changes:
            cursor.execute(
                """
                INSERT OR REPLACE INTO file_changes
                (id, group_id, change_type, file_path, timestamp, tool_name,
                 original_content, new_content, original_path, checksum_before, checksum_after)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    change.id,
                    group.id,
                    change.change_type.value,
                    change.file_path,
                    change.timestamp,
                    change.tool_name,
                    change.original_content,
                    change.new_content,
                    change.original_path,
                    change.checksum_before,
                    change.checksum_after,
                ),
            )

        conn.commit()
        # Connection managed by ProjectDatabaseManager

    def _trim_history(self) -> None:
        """Trim history to max_history size."""
        while len(self._undo_stack) > self.max_history:
            oldest = self._undo_stack.pop(0)
            # Optionally delete from database
            self._delete_group(oldest.id)

    def _delete_group(self, group_id: str) -> None:
        """Delete a change group from the database."""
        conn = self._db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM file_changes WHERE group_id = ?", (group_id,))
        cursor.execute("DELETE FROM change_groups WHERE id = ?", (group_id,))
        conn.commit()
        # Connection managed by ProjectDatabaseManager

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def undo(self) -> Tuple[bool, str, List[str]]:
        """Undo the last change group.

        Returns:
            Tuple of (success, message, list of affected files)
        """
        if not self._undo_stack:
            return False, "Nothing to undo", []

        group = self._undo_stack.pop()
        affected_files = []
        errors = []

        # Reverse changes in reverse order
        for change in reversed(group.changes):
            try:
                self._reverse_change(change)
                affected_files.append(change.file_path)
            except Exception as e:
                errors.append(f"{change.file_path}: {e}")
                logger.error(f"Failed to undo change on {change.file_path}: {e}")

        # Mark as undone and move to redo stack
        group.undone = True
        self._redo_stack.append(group)
        self._save_group(group)

        if errors:
            return False, f"Partial undo with errors: {'; '.join(errors)}", affected_files

        # Generate summary
        tool_name = group.tool_name
        file_count = len(affected_files)
        timestamp = datetime.fromtimestamp(group.timestamp).strftime("%H:%M:%S")

        return (
            True,
            f"Undid '{tool_name}' ({file_count} file{'s' if file_count > 1 else ''}) from {timestamp}",
            affected_files,
        )

    def redo(self) -> Tuple[bool, str, List[str]]:
        """Redo the last undone change group.

        Returns:
            Tuple of (success, message, list of affected files)
        """
        if not self._redo_stack:
            return False, "Nothing to redo", []

        group = self._redo_stack.pop()
        affected_files = []
        errors = []

        # Apply changes in original order
        for change in group.changes:
            try:
                self._apply_change(change)
                affected_files.append(change.file_path)
            except Exception as e:
                errors.append(f"{change.file_path}: {e}")
                logger.error(f"Failed to redo change on {change.file_path}: {e}")

        # Mark as not undone and move back to undo stack
        group.undone = False
        self._undo_stack.append(group)
        self._save_group(group)

        if errors:
            return False, f"Partial redo with errors: {'; '.join(errors)}", affected_files

        tool_name = group.tool_name
        file_count = len(affected_files)

        return (
            True,
            f"Redid '{tool_name}' ({file_count} file{'s' if file_count > 1 else ''})",
            affected_files,
        )

    def _reverse_change(self, change: FileChange) -> None:
        """Reverse a single file change."""
        path = Path(change.file_path)

        if change.change_type == ChangeType.CREATE:
            # Reverse create = delete
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted {path} (reverse of create)")

        elif change.change_type == ChangeType.DELETE:
            # Reverse delete = restore
            if change.original_content is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(change.original_content)
                logger.debug(f"Restored {path} (reverse of delete)")

        elif change.change_type == ChangeType.MODIFY:
            # Reverse modify = restore original content
            if change.original_content is not None:
                path.write_text(change.original_content)
                logger.debug(f"Restored original content of {path}")

        elif change.change_type == ChangeType.RENAME:
            # Reverse rename = rename back
            if change.original_path and path.exists():
                original = Path(change.original_path)
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(original))
                logger.debug(f"Renamed {path} back to {original}")

    def _apply_change(self, change: FileChange) -> None:
        """Apply a single file change (for redo)."""
        path = Path(change.file_path)

        if change.change_type == ChangeType.CREATE:
            if change.new_content is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(change.new_content)
                logger.debug(f"Created {path} (redo)")

        elif change.change_type == ChangeType.DELETE:
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted {path} (redo)")

        elif change.change_type == ChangeType.MODIFY:
            if change.new_content is not None:
                path.write_text(change.new_content)
                logger.debug(f"Modified {path} (redo)")

        elif change.change_type == ChangeType.RENAME:
            if change.original_path:
                original = Path(change.original_path)
                if original.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(original), str(path))
                    logger.debug(f"Renamed {original} to {path} (redo)")

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent change history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of change group summaries
        """
        history = []
        for group in reversed(self._undo_stack[-limit:]):
            history.append(
                {
                    "id": group.id,
                    "timestamp": datetime.fromtimestamp(group.timestamp).isoformat(),
                    "tool_name": group.tool_name,
                    "description": group.description,
                    "file_count": len(group.changes),
                    "files": [c.file_path for c in group.changes[:5]],  # First 5 files
                    "undone": group.undone,
                }
            )
        return history

    def get_file_history(self, file_path: str, limit: int = 10) -> List[FileChange]:
        """Get change history for a specific file.

        Args:
            file_path: Path to the file
            limit: Maximum number of entries

        Returns:
            List of FileChange objects
        """
        conn = self._db.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, change_type, file_path, timestamp, tool_name,
                   original_content, new_content, original_path,
                   checksum_before, checksum_after
            FROM file_changes
            WHERE file_path = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (str(file_path), limit),
        )

        changes = []
        for row in cursor.fetchall():
            changes.append(
                FileChange(
                    id=row[0],
                    change_type=ChangeType(row[1]),
                    file_path=row[2],
                    timestamp=row[3],
                    tool_name=row[4],
                    tool_args={},
                    original_content=row[5],
                    new_content=row[6],
                    original_path=row[7],
                    checksum_before=row[8],
                    checksum_after=row[9],
                )
            )

        # Connection managed by ProjectDatabaseManager
        return changes

    def clear_history(self) -> int:
        """Clear all change history.

        Returns:
            Number of groups cleared
        """
        count = len(self._undo_stack) + len(self._redo_stack)

        self._undo_stack.clear()
        self._redo_stack.clear()

        conn = self._db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM file_changes WHERE 1=1")
        cursor.execute("DELETE FROM change_groups WHERE 1=1")
        conn.commit()
        # Connection managed by ProjectDatabaseManager

        logger.info(f"Cleared {count} change groups from history")
        return count


# Global instance for easy access
_file_change_history: Optional[FileChangeHistory] = None


def get_change_tracker() -> FileChangeHistory:
    """Get or create the global file change history instance.

    Note: Function name kept for backward compatibility.
    """
    global _file_change_history
    if _file_change_history is None:
        _file_change_history = FileChangeHistory()
    return _file_change_history


def set_change_tracker(tracker: FileChangeHistory) -> None:
    """Set the global file change history instance.

    Note: Function name kept for backward compatibility.
    """
    global _file_change_history
    _file_change_history = tracker


def reset_change_tracker() -> None:
    """Reset the global file change history instance.

    Used primarily for testing to ensure a fresh history between tests.

    Note: Function name kept for backward compatibility.
    """
    global _file_change_history
    _file_change_history = None
