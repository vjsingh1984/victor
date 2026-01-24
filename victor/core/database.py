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

"""Unified database managers for Victor.

Includes:
- DatabaseManager: Global singleton for user-level data (sync + async APIs)
- ProjectDatabaseManager: Per-project data

Provides two database scopes to separate user-level and project-level data.

See victor.core.schema for table constants and SQL definitions.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │              GLOBAL DATABASE (~/.victor/victor.db)                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │  User-level data shared across all projects:                        │
    │  ├── RL Learning: rl_outcomes, *_q_values, tool/model preferences   │
    │  ├── Teams: team_composition_stats, team_execution_history          │
    │  ├── Sessions: sessions (TUI session persistence)                   │
    │  ├── Signatures: failed_signatures (loop prevention)                │
    │  └── Cross-Vertical: patterns learned across projects               │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │              PROJECT DATABASE (.victor/project.db)                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Project-specific data (per-repo):                                  │
    │  ├── Graph: symbols, references, definitions                        │
    │  ├── Conversations: project conversation history                    │
    │  ├── Entities: project entity memory                                │
    │  └── Mode Learning: project-specific mode preferences               │
    └─────────────────────────────────────────────────────────────────────┘

Why Two Databases:
    - Keeps user preferences/RL separate from project-specific data
    - Prevents pollution of global learnings with project-specific patterns
    - Allows sharing project.db in team settings if desired
    - Enables independent backup/migration of each scope

Migration:
    Existing databases are automatically migrated on first access:
    - ~/.victor/graph/graph.db → global (RL tables) + project (graph tables)
    - ~/.victor/team_learning.db → global
    - ~/.victor/sessions.db → global
    - ~/.victor/signatures.db → global
    - .victor/conversation.db → project

Usage:
    from victor.core.database import get_database, get_project_database

    # Global database (user-level)
    global_db = get_database()
    global_db.execute("INSERT INTO rl_outcomes ...", params)

    # Project database (repo-level)
    project_db = get_project_database()
    project_db.query("SELECT * FROM symbols WHERE name = ?", ("MyClass",))
"""

from __future__ import annotations

import asyncio
import logging
import queue
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Unified database manager for Victor.

    Provides a single SQLite database for all components, with:
    - Thread-safe connection pooling
    - Automatic migration from legacy databases
    - Table namespace management
    - Connection health monitoring

    Attributes:
        db_path: Path to the unified database file
        _connection: Thread-local connection storage
        _lock: Threading lock for migrations
    """

    # Singleton instance
    _instance: Optional["DatabaseManager"] = None
    _init_lock = threading.Lock()

    # Legacy database paths relative to ~/.victor/
    LEGACY_DATABASES = {
        "graph": "graph/graph.db",
        "team_learning": "team_learning.db",
        "sessions": "sessions.db",
        "signatures": "signatures.db",
        "rl_learning": "rl_learning.db",  # May be empty
    }

    def __new__(cls, db_path: Optional[Path] = None) -> "DatabaseManager":
        """Get or create singleton instance."""
        with cls._init_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database manager.

        Args:
            db_path: Path to database file. Defaults to ~/.victor/victor.db
        """
        if self._initialized:
            return

        self._victor_dir = Path.home() / ".victor"
        self._victor_dir.mkdir(parents=True, exist_ok=True)

        if db_path is None:
            db_path = self._victor_dir / "victor.db"

        self.db_path = db_path
        self._local = threading.local()
        self._migration_lock = threading.Lock()
        self._migrated = False

        # Ensure database exists and run migrations
        self._ensure_database()

        self._initialized = True
        logger.info(f"DatabaseManager initialized: {self.db_path}")

    def _ensure_database(self) -> None:
        """Ensure database exists and is up to date."""
        from victor.core.schema import Tables, Schema

        # Create database if needed
        conn = self._get_raw_connection()

        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

        # Create metadata table using schema constant
        conn.execute(Schema.SYS_METADATA)
        conn.commit()

        # Run migrations if needed
        self._run_migrations(conn)

    def _get_raw_connection(self) -> sqlite3.Connection:
        """Get raw SQLite connection (thread-local)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            Thread-local SQLite connection with Row factory
        """
        return self._get_raw_connection()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database transactions.

        Yields:
            Database connection with automatic commit/rollback

        Example:
            with db.transaction() as conn:
                conn.execute("INSERT INTO ...")
                conn.execute("UPDATE ...")
            # Auto-committed on success, rolled back on exception
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> sqlite3.Cursor:
        """Execute SQL with auto-commit.

        Args:
            sql: SQL statement
            params: Optional parameters

        Returns:
            Cursor with results
        """
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        conn.commit()
        return cursor

    def executemany(
        self,
        sql: str,
        params_list: List[Tuple[Any, ...]],
    ) -> sqlite3.Cursor:
        """Execute SQL for multiple parameter sets.

        Args:
            sql: SQL statement
            params_list: List of parameter tuples

        Returns:
            Cursor
        """
        conn = self.get_connection()
        cursor = conn.executemany(sql, params_list)
        conn.commit()
        return cursor

    def query(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> List[sqlite3.Row]:
        """Execute query and return all rows.

        Args:
            sql: SQL query
            params: Optional parameters

        Returns:
            List of Row objects
        """
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        return cursor.fetchall()

    def query_one(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[sqlite3.Row]:
        """Execute query and return first row.

        Args:
            sql: SQL query
            params: Optional parameters

        Returns:
            Row object or None
        """
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        return cursor.fetchone()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists.

        Args:
            table_name: Name of table

        Returns:
            True if table exists
        """
        row = self.query_one(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return row is not None

    def get_tables(self) -> List[str]:
        """Get list of all tables.

        Returns:
            List of table names
        """
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in rows]

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run database migrations from legacy databases."""
        from victor.core.schema import Tables

        with self._migration_lock:
            if self._migrated:
                return

            # Check if already migrated
            cursor = conn.execute(
                f"SELECT value FROM {Tables.SYS_METADATA} WHERE key = 'migrated_at'"
            )
            row = cursor.fetchone()
            if row:
                logger.debug(f"Database already migrated at {row[0]}")
                self._migrated = True
                return

            # Migrate from legacy databases
            migrated_count = 0
            for name, rel_path in self.LEGACY_DATABASES.items():
                legacy_path = self._victor_dir / rel_path
                if legacy_path.exists() and legacy_path.stat().st_size > 0:
                    try:
                        migrated = self._migrate_from_legacy(conn, name, legacy_path)
                        migrated_count += migrated
                        logger.info(f"Migrated {migrated} tables from {name}")
                    except Exception as e:
                        logger.warning(f"Failed to migrate {name}: {e}")

            # Record migration timestamp
            now = datetime.now().isoformat()
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {Tables.SYS_METADATA} (key, value, updated_at)
                VALUES ('migrated_at', ?, ?)
                """,
                (now, now),
            )

            self._migrated = True
            if migrated_count > 0:
                logger.info(f"Database migration complete: {migrated_count} tables")

        # Run schema version migrations after legacy migrations
        self._run_schema_migrations(conn)

    def _migrate_from_legacy(
        self,
        conn: sqlite3.Connection,
        source_name: str,
        legacy_path: Path,
    ) -> int:
        """Migrate tables from a legacy database.

        Args:
            conn: Target database connection
            source_name: Name of source database (for logging)
            legacy_path: Path to legacy database

        Returns:
            Number of tables migrated
        """
        # Attach legacy database
        conn.execute(f"ATTACH DATABASE ? AS legacy_{source_name}", (str(legacy_path),))

        try:
            # Get tables from legacy database (excluding sqlite internal tables)
            cursor = conn.execute(
                f"""
                SELECT name FROM legacy_{source_name}.sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
            )
            tables = [row[0] for row in cursor.fetchall()]

            migrated = 0
            for table in tables:
                # Skip if table already exists in target
                if self.table_exists(table):
                    # Check if source has data and target is empty
                    source_count = conn.execute(
                        f"SELECT COUNT(*) FROM legacy_{source_name}.{table}"
                    ).fetchone()[0]
                    target_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

                    if source_count > 0 and target_count == 0:
                        # Copy data from source to existing target
                        conn.execute(
                            f"""
                            INSERT INTO {table}
                            SELECT * FROM legacy_{source_name}.{table}
                        """
                        )
                        logger.debug(f"Copied {source_count} rows from legacy {table}")
                        migrated += 1
                    continue

                # Get table schema
                schema_row = conn.execute(
                    f"""
                    SELECT sql FROM legacy_{source_name}.sqlite_master
                    WHERE type='table' AND name=?
                """,
                    (table,),
                ).fetchone()

                if schema_row and schema_row[0]:
                    # Create table in target
                    try:
                        conn.execute(schema_row[0])
                    except sqlite3.OperationalError as e:
                        if "already exists" not in str(e):
                            raise

                    # Copy data
                    conn.execute(
                        f"""
                        INSERT INTO {table}
                        SELECT * FROM legacy_{source_name}.{table}
                    """
                    )

                    # Copy indexes
                    idx_cursor = conn.execute(
                        f"""
                        SELECT sql FROM legacy_{source_name}.sqlite_master
                        WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
                    """,
                        (table,),
                    )
                    for idx_row in idx_cursor:
                        try:
                            conn.execute(idx_row[0])
                        except sqlite3.OperationalError:
                            pass  # Index may already exist

                    migrated += 1
                    logger.debug(f"Migrated table: {table}")

            conn.commit()
            return migrated

        finally:
            conn.execute(f"DETACH DATABASE legacy_{source_name}")

    def _run_schema_migrations(self, conn: sqlite3.Connection) -> None:
        """Run schema version migrations.

        Handles adding new columns to existing tables (e.g., profile column to sessions).

        Args:
            conn: Database connection
        """
        from victor.core.schema import Tables, CURRENT_SCHEMA_VERSION

        # Get current schema version
        cursor = conn.execute(
            f"SELECT value FROM {Tables.SYS_METADATA} WHERE key = 'schema_version'"
        )
        row = cursor.fetchone()
        current_version = int(row[0]) if row else 1

        if current_version >= CURRENT_SCHEMA_VERSION:
            logger.debug(f"Schema version {current_version} is up to date")
            return

        logger.info(f"Migrating schema from version {current_version} to {CURRENT_SCHEMA_VERSION}")

        # Run version migrations
        for target_version in range(current_version + 1, CURRENT_SCHEMA_VERSION + 1):
            self._migrate_schema_to(conn, target_version)

        # Update schema version
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.SYS_METADATA} (key, value, updated_at)
            VALUES ('schema_version', ?, datetime('now'))
            """,
            (str(CURRENT_SCHEMA_VERSION),),
        )
        conn.commit()
        logger.info(f"Schema migrated to version {CURRENT_SCHEMA_VERSION}")

    def _migrate_schema_to(self, conn: sqlite3.Connection, version: int) -> None:
        """Migrate schema to a specific version.

        Args:
            conn: Database connection
            version: Target schema version
        """
        # Version 2: Add profile column to sessions table
        if version == 2:
            self._add_column_if_missing(
                conn,
                "sessions",
                "profile",
                "TEXT DEFAULT 'default'",
            )

    def _add_column_if_missing(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        """Add a column to a table if it doesn't exist.

        Args:
            conn: Database connection
            table: Table name
            column: Column name
            definition: Column definition (e.g., "TEXT DEFAULT 'default'")
        """
        # Check if column exists
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = {row["name"] for row in cursor.fetchall()}

        if column in columns:
            logger.debug(f"Column {table}.{column} already exists")
            return

        # Add column
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            conn.commit()
            logger.info(f"Added column {table}.{column}")
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to add column {table}.{column}: {e}")

    def backup(self, backup_path: Optional[Path] = None) -> Path:
        """Create a backup of the database.

        Args:
            backup_path: Optional custom backup path

        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self._victor_dir / "backups" / f"victor_{timestamp}.db"

        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Use SQLite backup API
        conn = self.get_connection()
        backup_conn = sqlite3.connect(str(backup_path))
        try:
            conn.backup(backup_conn)
            logger.info(f"Database backed up to: {backup_path}")
        finally:
            backup_conn.close()

        return backup_path

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dict with table counts and sizes
        """
        stats: Dict[str, Any] = {
            "path": str(self.db_path),
            "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "tables": {},
        }

        for table in self.get_tables():
            if table.startswith("_"):
                continue
            count = self.query_one(f"SELECT COUNT(*) FROM {table}")
            stats["tables"][table] = count[0] if count else 0

        return stats

    def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        conn = self.get_connection()
        conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # =========================================================================
    # Async API (wraps sync methods via thread pool)
    # =========================================================================

    def _ensure_async_state(self) -> None:
        """Initialize async-related state if not already done."""
        if not hasattr(self, "_write_queue"):
            self._write_queue: queue.Queue[Tuple[str, Optional[Tuple[Any, ...]]]] = queue.Queue()
            self._write_lock = threading.Lock()
            self._batch_size = 100
            self._flush_interval = 5.0
            self._flush_task: Optional[asyncio.Task] = None
            self._is_flushing = False

    async def execute_async(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> sqlite3.Cursor:
        """Execute SQL asynchronously.

        Runs the sync execute in a thread pool to avoid blocking.

        Args:
            sql: SQL statement
            params: Optional parameters

        Returns:
            Cursor with results
        """
        return await asyncio.to_thread(self.execute, sql, params)

    async def executemany_async(
        self,
        sql: str,
        params_list: List[Tuple[Any, ...]],
    ) -> sqlite3.Cursor:
        """Execute SQL for multiple parameter sets asynchronously.

        Args:
            sql: SQL statement
            params_list: List of parameter tuples

        Returns:
            Cursor
        """
        return await asyncio.to_thread(self.executemany, sql, params_list)

    async def query_async(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> List[sqlite3.Row]:
        """Execute query asynchronously.

        Args:
            sql: SQL query
            params: Optional parameters

        Returns:
            List of Row objects
        """
        return await asyncio.to_thread(self.query, sql, params)

    async def query_one_async(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[sqlite3.Row]:
        """Execute query and return first row asynchronously.

        Args:
            sql: SQL query
            params: Optional parameters

        Returns:
            Row object or None
        """
        return await asyncio.to_thread(self.query_one, sql, params)

    # =========================================================================
    # Write Batching
    # =========================================================================

    def queue_write(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        """Queue a write operation for batched execution.

        Writes are batched and executed together for better performance.
        Call flush_writes() to execute all queued writes.

        Args:
            sql: SQL statement (INSERT, UPDATE, DELETE)
            params: Optional parameters
        """
        self._ensure_async_state()
        self._write_queue.put((sql, params))

        # Auto-flush if queue is large
        if self._write_queue.qsize() >= self._batch_size:
            self.flush_writes_sync()

    def flush_writes_sync(self) -> int:
        """Flush all queued writes synchronously.

        Returns:
            Number of writes executed
        """
        self._ensure_async_state()

        with self._write_lock:
            if self._is_flushing:
                return 0
            self._is_flushing = True

        try:
            writes: List[Tuple[str, Optional[Tuple[Any, ...]]]] = []

            # Drain the queue
            while not self._write_queue.empty():
                try:
                    writes.append(self._write_queue.get_nowait())
                except Exception:
                    break

            if not writes:
                return 0

            # Execute in a transaction
            conn = self.get_connection()
            try:
                for sql, params in writes:
                    conn.execute(sql, params or ())
                conn.commit()
                logger.debug(f"Flushed {len(writes)} batched writes")
                return len(writes)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to flush writes: {e}")
                # Re-queue failed writes
                for write in writes:
                    self._write_queue.put(write)
                raise

        finally:
            self._is_flushing = False

    async def flush_writes_async(self) -> int:
        """Flush all queued writes asynchronously.

        Returns:
            Number of writes executed
        """
        return await asyncio.to_thread(self.flush_writes_sync)

    def get_write_queue_size(self) -> int:
        """Get number of pending writes.

        Returns:
            Queue size
        """
        self._ensure_async_state()
        return self._write_queue.qsize()

    async def start_auto_flush(self, interval_seconds: Optional[float] = None) -> None:
        """Start background task that periodically flushes writes.

        Args:
            interval_seconds: Flush interval (default 5.0)
        """
        self._ensure_async_state()

        if interval_seconds is not None:
            self._flush_interval = interval_seconds

        async def _flush_loop() -> None:
            while True:
                await asyncio.sleep(self._flush_interval)
                try:
                    await self.flush_writes_async()
                except Exception as e:
                    logger.warning(f"Auto-flush error: {e}")

        self._flush_task = asyncio.create_task(_flush_loop())

    async def stop_auto_flush(self) -> None:
        """Stop background flush task and flush remaining writes."""
        if hasattr(self, "_flush_task") and self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush
        await self.flush_writes_async()


class ProjectDatabaseManager:
    """Project-level database manager for repo-specific data.

    Manages project-scoped data in .victor/project.db:
    - Graph data (symbols, references, definitions)
    - Conversation memory
    - Entity memory
    - Project-specific mode learning

    Unlike the global DatabaseManager, this is NOT a singleton since
    different projects have different databases.

    Attributes:
        db_path: Path to the project database file
        project_dir: Path to the project's .victor directory
    """

    # Legacy project databases to migrate
    LEGACY_PROJECT_DATABASES = {
        "conversation": "conversation.db",
        "entities": "entities.db",
        "mode_learning": "mode_learning.db",
        "profile_learning": "profile_learning.db",
        "changes": "changes/changes.db",
        "graph": "graph/graph.db",
    }

    # Tables that should stay in project database (not global)
    PROJECT_TABLES = {
        # Graph tables
        "graph_node",
        "graph_edge",
        "graph_file_mtime",
        "graph_node_fts",
        # Conversation tables
        "messages",
        "sessions",
        "context_sizes",
        "context_summaries",
        # Mode learning tables
        "rl_mode_q",
        "rl_mode_task",
        "rl_mode_history",
        # Profile learning tables
        "interaction_history",
        "profile_metrics",
        # Changes tables
        "change_groups",
        "file_changes",
    }

    def __init__(self, project_path: Optional[Path] = None, force_rebuild: bool = False):
        """Initialize project database manager.

        Args:
            project_path: Path to project root. If None, uses current directory.
            force_rebuild: If True, rebuild corrupted database.
        """
        if project_path is None:
            project_path = Path.cwd()

        self.project_dir = project_path / ".victor"
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.project_dir / "project.db"
        self._local = threading.local()
        self._migration_lock = threading.Lock()
        self._migrated = False

        # Ensure database exists
        self._ensure_database(force_rebuild=force_rebuild)

        logger.info(f"ProjectDatabaseManager initialized: {self.db_path}")

    def _ensure_database(self, force_rebuild: bool = False) -> None:
        """Ensure database exists and is up to date.

        Args:
            force_rebuild: If True, delete corrupted database and recreate.
        """
        from victor.core.schema import Schema

        try:
            conn = self._get_raw_connection()

            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")

            # Create metadata table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _project_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """
            )

            # Create project-level tables (graph, etc.)
            for schema_sql in Schema.get_project_schemas():
                conn.execute(schema_sql)

            # Create project-level indexes
            for index_sql in Schema.get_project_indexes():
                conn.executescript(index_sql)

            conn.commit()

            # Run migrations if needed
            self._run_migrations(conn)

        except sqlite3.DatabaseError as e:
            error_msg = str(e).lower()
            if "malformed" in error_msg or "corrupted" in error_msg or "database disk image" in error_msg:
                if force_rebuild:
                    logger.warning(f"Database corrupted: {e}. Rebuilding...")
                    self._rebuild_database()
                    # Try again after rebuild
                    return self._ensure_database(force_rebuild=False)
                else:
                    raise RuntimeError(
                        f"Project database is corrupted. Run 'victor index --force' to rebuild.\n"
                        f"Database path: {self.db_path}\n"
                        f"Original error: {e}"
                    ) from e
            raise

    def _rebuild_database(self) -> None:
        """Rebuild the corrupted database from scratch.

        This deletes the corrupted database file and recreates it.
        """
        import shutil

        logger.warning(f"Rebuilding corrupted database: {self.db_path}")

        # Close existing connection
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

        # Delete corrupted database files
        files_to_remove = [self.db_path]
        # Also remove WAL and journal files
        for suffix in ["-wal", "-shm", "-journal"]:
            journal_file = self.db_path.with_suffix(self.db_path.suffix + suffix)
            files_to_remove.append(journal_file)

        removed_files = []
        for file_path in files_to_remove:
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed_files.append(file_path.name)
                except Exception as e:
                    logger.debug(f"Could not remove {file_path}: {e}")

        if removed_files:
            logger.info(f"Removed corrupted database files: {', '.join(removed_files)}")

        # Reset migrated flag so migrations will run
        self._migrated = False

    def _get_raw_connection(self) -> sqlite3.Connection:
        """Get raw SQLite connection (thread-local)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return self._get_raw_connection()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database transactions."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> sqlite3.Cursor:
        """Execute SQL with auto-commit."""
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        conn.commit()
        return cursor

    def query(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> List[sqlite3.Row]:
        """Execute query and return all rows."""
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        return cursor.fetchall()

    def query_one(
        self,
        sql: str,
        params: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[sqlite3.Row]:
        """Execute query and return first row."""
        conn = self.get_connection()
        cursor = conn.execute(sql, params or ())
        return cursor.fetchone()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        row = self.query_one(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return row is not None

    def get_tables(self) -> List[str]:
        """Get list of all tables."""
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in rows]

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run database migrations from legacy project databases."""
        with self._migration_lock:
            if self._migrated:
                return

            # Check if already migrated
            cursor = conn.execute("SELECT value FROM _project_metadata WHERE key = 'migrated_at'")
            row = cursor.fetchone()
            if row:
                logger.debug(f"Project database already migrated at {row[0]}")
                self._migrated = True
                return

            # Migrate from legacy databases
            migrated_count = 0
            for name, rel_path in self.LEGACY_PROJECT_DATABASES.items():
                legacy_path = self.project_dir / rel_path
                if legacy_path.exists() and legacy_path.stat().st_size > 0:
                    try:
                        migrated = self._migrate_project_tables(conn, name, legacy_path)
                        migrated_count += migrated
                        if migrated > 0:
                            logger.info(f"Migrated {migrated} tables from {name}")
                    except Exception as e:
                        logger.warning(f"Failed to migrate {name}: {e}")

            # Record migration timestamp
            now = datetime.now().isoformat()
            conn.execute(
                """
                INSERT OR REPLACE INTO _project_metadata (key, value, updated_at)
                VALUES ('migrated_at', ?, ?)
                """,
                (now, now),
            )
            conn.commit()

            self._migrated = True
            if migrated_count > 0:
                logger.info(f"Project database migration complete: {migrated_count} tables")

        # Run schema version migrations after legacy migrations
        self._run_schema_migrations(conn)

    def _migrate_project_tables(
        self,
        conn: sqlite3.Connection,
        source_name: str,
        legacy_path: Path,
    ) -> int:
        """Migrate project-specific tables from a legacy database."""
        # Attach legacy database
        conn.execute(f"ATTACH DATABASE ? AS legacy_{source_name}", (str(legacy_path),))

        try:
            # Get tables from legacy database
            cursor = conn.execute(
                f"""
                SELECT name FROM legacy_{source_name}.sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """
            )
            tables = [row[0] for row in cursor.fetchall()]

            migrated = 0
            for table in tables:
                # Only migrate project-specific tables
                if table not in self.PROJECT_TABLES and not table.startswith("_"):
                    continue

                # Skip if table already exists
                if self.table_exists(table):
                    continue

                # Get table schema
                schema_row = conn.execute(
                    f"""
                    SELECT sql FROM legacy_{source_name}.sqlite_master
                    WHERE type='table' AND name=?
                """,
                    (table,),
                ).fetchone()

                if schema_row and schema_row[0]:
                    try:
                        conn.execute(schema_row[0])
                    except sqlite3.OperationalError as e:
                        if "already exists" not in str(e):
                            raise

                    # Copy data
                    conn.execute(
                        f"""
                        INSERT INTO {table}
                        SELECT * FROM legacy_{source_name}.{table}
                    """
                    )

                    migrated += 1
                    logger.debug(f"Migrated project table: {table}")

            conn.commit()
            return migrated

        finally:
            conn.execute(f"DETACH DATABASE legacy_{source_name}")

    def _run_schema_migrations(self, conn: sqlite3.Connection) -> None:
        """Run schema version migrations for project database.

        Handles adding new columns to existing tables.

        Args:
            conn: Database connection
        """
        from victor.core.schema import CURRENT_SCHEMA_VERSION

        # Get current schema version
        cursor = conn.execute("SELECT value FROM _project_metadata WHERE key = 'schema_version'")
        row = cursor.fetchone()
        current_version = int(row[0]) if row else 1

        if current_version >= CURRENT_SCHEMA_VERSION:
            logger.debug(f"Project schema version {current_version} is up to date")
            return

        logger.info(
            f"Migrating project schema from version {current_version} to {CURRENT_SCHEMA_VERSION}"
        )

        # Run version migrations
        for target_version in range(current_version + 1, CURRENT_SCHEMA_VERSION + 1):
            self._migrate_schema_to(conn, target_version)

        # Update schema version
        conn.execute(
            """
            INSERT OR REPLACE INTO _project_metadata (key, value, updated_at)
            VALUES ('schema_version', ?, datetime('now'))
            """,
            (str(CURRENT_SCHEMA_VERSION),),
        )
        conn.commit()
        logger.info(f"Project schema migrated to version {CURRENT_SCHEMA_VERSION}")

    def _migrate_schema_to(self, conn: sqlite3.Connection, version: int) -> None:
        """Migrate project schema to a specific version.

        Args:
            conn: Database connection
            version: Target schema version
        """
        # Version 2 migrations for project tables can be added here
        # Currently no project-specific column additions needed
        pass

    def _add_column_if_missing(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        """Add a column to a table if it doesn't exist.

        Args:
            conn: Database connection
            table: Table name
            column: Column name
            definition: Column definition (e.g., "TEXT DEFAULT 'default'")
        """
        # Check if column exists
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = {row["name"] for row in cursor.fetchall()}

        if column in columns:
            logger.debug(f"Column {table}.{column} already exists")
            return

        # Add column
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            conn.commit()
            logger.info(f"Added column {table}.{column}")
        except sqlite3.OperationalError as e:
            logger.warning(f"Failed to add column {table}.{column}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats: Dict[str, Any] = {
            "path": str(self.db_path),
            "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "tables": {},
        }

        for table in self.get_tables():
            if table.startswith("_"):
                continue
            count = self.query_one(f"SELECT COUNT(*) FROM {table}")
            stats["tables"][table] = count[0] if count else 0

        return stats

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Module-level singleton accessors
_database: Optional[DatabaseManager] = None
_project_databases: Dict[str, ProjectDatabaseManager] = {}


def get_database(db_path: Optional[Path] = None) -> DatabaseManager:
    """Get the global database manager instance.

    Args:
        db_path: Optional custom database path (only used on first call)

    Returns:
        DatabaseManager singleton instance
    """
    global _database
    if _database is None:
        _database = DatabaseManager(db_path)
    return _database


def get_project_database(project_path: Optional[Path] = None) -> ProjectDatabaseManager:
    """Get project database manager for a specific project.

    Args:
        project_path: Path to project root. If None, uses current directory.

    Returns:
        ProjectDatabaseManager instance for the project
    """
    global _project_databases
    if project_path is None:
        project_path = Path.cwd()

    project_key = str(project_path.resolve())

    # Check environment variable for force rebuild (set by CLI --force)
    import os
    force_rebuild = os.environ.get("VICTOR_DATABASE_FORCE_REBUILD") == "1"

    if project_key not in _project_databases:
        _project_databases[project_key] = ProjectDatabaseManager(project_path, force_rebuild=force_rebuild)
    return _project_databases[project_key]


def reset_database() -> None:
    """Reset the global database instance (for testing)."""
    global _database
    if _database is not None:
        _database.close()
        _database = None
    DatabaseManager._instance = None


def reset_project_database(project_path: Optional[Path] = None) -> None:
    """Reset project database instance (for testing)."""
    global _project_databases
    if project_path is None:
        project_path = Path.cwd()

    project_key = str(project_path.resolve())
    if project_key in _project_databases:
        _project_databases[project_key].close()
        del _project_databases[project_key]


def reset_all_databases() -> None:
    """Reset all database instances (for testing)."""
    global _database, _project_databases

    if _database is not None:
        _database.close()
        _database = None
    DatabaseManager._instance = None

    for db in _project_databases.values():
        db.close()
    _project_databases.clear()


__all__ = [
    "DatabaseManager",
    "ProjectDatabaseManager",
    "get_database",
    "get_project_database",
    "reset_database",
    "reset_project_database",
    "reset_all_databases",
]
