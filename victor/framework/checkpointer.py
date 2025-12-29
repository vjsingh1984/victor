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

"""Checkpointer implementations for StateGraph persistence.

Provides various storage backends for checkpointing graph execution state,
enabling resumption of interrupted workflows and audit trails.

Implementations:
    - SQLiteCheckpointer: File-based SQLite storage (default)
    - AsyncSQLiteCheckpointer: Async-safe SQLite wrapper
    - JSONFileCheckpointer: Simple JSON file storage

Design Principles:
    - Single Responsibility: Each checkpointer handles one storage backend
    - Interface Segregation: CheckpointerProtocol is minimal and focused
    - Dependency Inversion: Graph depends on protocol, not implementations

Example:
    from victor.framework.checkpointer import SQLiteCheckpointer
    from victor.framework.graph import StateGraph

    checkpointer = SQLiteCheckpointer("~/.victor/checkpoints.db")
    graph = StateGraph(MyState)
    # ... add nodes and edges ...
    app = graph.compile(checkpointer=checkpointer)

    # Execute with automatic checkpointing
    result = await app.invoke(initial_state, thread_id="my-thread")

    # Resume from checkpoint
    result = await app.invoke({}, thread_id="my-thread")  # Resumes from last checkpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.framework.graph import Checkpoint, CheckpointerProtocol

logger = logging.getLogger(__name__)


class SQLiteCheckpointer:
    """SQLite-based checkpointer for graph state persistence.

    Stores checkpoints in a SQLite database file for durability
    and queryability.

    Attributes:
        db_path: Path to SQLite database file
        table_name: Name of the checkpoints table

    Example:
        checkpointer = SQLiteCheckpointer("~/.victor/checkpoints.db")
        await checkpointer.save(checkpoint)
        latest = await checkpointer.load("thread-123")
    """

    def __init__(
        self,
        db_path: str = "~/.victor/graph_checkpoints.db",
        table_name: str = "checkpoints",
    ):
        """Initialize SQLite checkpointer.

        Args:
            db_path: Path to database file (will be created if not exists)
            table_name: Name for checkpoints table
        """
        self.db_path = Path(os.path.expanduser(db_path))
        self.table_name = table_name
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row

            if not self._initialized:
                self._init_schema()
                self._initialized = True

        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                checkpoint_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                state TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_thread_id
            ON {self.table_name}(thread_id)
        """)
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_thread_timestamp
            ON {self.table_name}(thread_id, timestamp DESC)
        """)
        conn.commit()
        logger.debug(f"Initialized checkpoint schema: {self.db_path}")

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to SQLite.

        Args:
            checkpoint: Checkpoint to save
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_sync, checkpoint)

    def _save_sync(self, checkpoint: Checkpoint) -> None:
        """Synchronous save implementation."""
        conn = self._get_connection()
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.table_name}
            (checkpoint_id, thread_id, node_id, state, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                checkpoint.checkpoint_id,
                checkpoint.thread_id,
                checkpoint.node_id,
                json.dumps(checkpoint.state),
                checkpoint.timestamp,
                json.dumps(checkpoint.metadata),
            ),
        )
        conn.commit()
        logger.debug(
            f"Saved checkpoint: {checkpoint.checkpoint_id} "
            f"(thread: {checkpoint.thread_id}, node: {checkpoint.node_id})"
        )

    async def load(self, thread_id: str) -> Optional[Checkpoint]:
        """Load the latest checkpoint for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Latest checkpoint or None if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._load_sync, thread_id)

    def _load_sync(self, thread_id: str) -> Optional[Checkpoint]:
        """Synchronous load implementation."""
        conn = self._get_connection()
        row = conn.execute(
            f"""
            SELECT * FROM {self.table_name}
            WHERE thread_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """,
            (thread_id,),
        ).fetchone()

        if row is None:
            return None

        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            thread_id=row["thread_id"],
            node_id=row["node_id"],
            state=json.loads(row["state"]),
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def list(self, thread_id: str) -> List[Checkpoint]:
        """List all checkpoints for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            List of checkpoints ordered by timestamp (newest first)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_sync, thread_id)

    def _list_sync(self, thread_id: str) -> List[Checkpoint]:
        """Synchronous list implementation."""
        conn = self._get_connection()
        rows = conn.execute(
            f"""
            SELECT * FROM {self.table_name}
            WHERE thread_id = ?
            ORDER BY timestamp DESC
        """,
            (thread_id,),
        ).fetchall()

        return [
            Checkpoint(
                checkpoint_id=row["checkpoint_id"],
                thread_id=row["thread_id"],
                node_id=row["node_id"],
                state=json.loads(row["state"]),
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_sync, checkpoint_id)

    def _delete_sync(self, checkpoint_id: str) -> bool:
        """Synchronous delete implementation."""
        conn = self._get_connection()
        cursor = conn.execute(
            f"DELETE FROM {self.table_name} WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    async def delete_thread(self, thread_id: str) -> int:
        """Delete all checkpoints for a thread.

        Args:
            thread_id: Thread to delete checkpoints for

        Returns:
            Number of checkpoints deleted
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_thread_sync, thread_id)

    def _delete_thread_sync(self, thread_id: str) -> int:
        """Synchronous delete thread implementation."""
        conn = self._get_connection()
        cursor = conn.execute(
            f"DELETE FROM {self.table_name} WHERE thread_id = ?",
            (thread_id,),
        )
        conn.commit()
        return cursor.rowcount

    async def cleanup(self, max_age_hours: int = 24, max_per_thread: int = 10) -> int:
        """Clean up old checkpoints.

        Args:
            max_age_hours: Delete checkpoints older than this
            max_per_thread: Keep only this many per thread

        Returns:
            Number of checkpoints deleted
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._cleanup_sync, max_age_hours, max_per_thread
        )

    def _cleanup_sync(self, max_age_hours: int, max_per_thread: int) -> int:
        """Synchronous cleanup implementation."""
        import time

        conn = self._get_connection()
        deleted = 0

        # Delete old checkpoints
        cutoff = time.time() - (max_age_hours * 3600)
        cursor = conn.execute(
            f"DELETE FROM {self.table_name} WHERE timestamp < ?",
            (cutoff,),
        )
        deleted += cursor.rowcount

        # Keep only max_per_thread per thread
        threads = conn.execute(
            f"SELECT DISTINCT thread_id FROM {self.table_name}"
        ).fetchall()

        for (thread_id,) in threads:
            # Get checkpoint IDs to keep
            keep_ids = conn.execute(
                f"""
                SELECT checkpoint_id FROM {self.table_name}
                WHERE thread_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (thread_id, max_per_thread),
            ).fetchall()
            keep_ids = [row[0] for row in keep_ids]

            if keep_ids:
                placeholders = ",".join("?" * len(keep_ids))
                cursor = conn.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE thread_id = ? AND checkpoint_id NOT IN ({placeholders})
                """,
                    (thread_id, *keep_ids),
                )
                deleted += cursor.rowcount

        conn.commit()
        logger.info(f"Cleaned up {deleted} checkpoints")
        return deleted

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class JSONFileCheckpointer:
    """JSON file-based checkpointer for simple storage.

    Stores each checkpoint as a separate JSON file. Suitable for
    development and debugging.

    Attributes:
        base_dir: Directory to store checkpoint files
    """

    def __init__(self, base_dir: str = "~/.victor/checkpoints"):
        """Initialize JSON file checkpointer.

        Args:
            base_dir: Directory for checkpoint files
        """
        self.base_dir = Path(os.path.expanduser(base_dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_thread_dir(self, thread_id: str) -> Path:
        """Get directory for a thread's checkpoints."""
        thread_dir = self.base_dir / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        return thread_dir

    async def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to JSON file."""
        thread_dir = self._get_thread_dir(checkpoint.thread_id)
        filepath = thread_dir / f"{checkpoint.checkpoint_id}.json"

        data = checkpoint.to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"Saved checkpoint to: {filepath}")

    async def load(self, thread_id: str) -> Optional[Checkpoint]:
        """Load latest checkpoint for thread."""
        thread_dir = self._get_thread_dir(thread_id)

        files = sorted(
            thread_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not files:
            return None

        with open(files[0]) as f:
            data = json.load(f)

        return Checkpoint.from_dict(data)

    async def list(self, thread_id: str) -> List[Checkpoint]:
        """List all checkpoints for thread."""
        thread_dir = self._get_thread_dir(thread_id)

        files = sorted(
            thread_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        checkpoints = []
        for filepath in files:
            with open(filepath) as f:
                data = json.load(f)
            checkpoints.append(Checkpoint.from_dict(data))

        return checkpoints


__all__ = [
    "SQLiteCheckpointer",
    "JSONFileCheckpointer",
]
