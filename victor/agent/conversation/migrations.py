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

"""Database migration system for ConversationStore.

This module provides schema migration support to handle database schema
changes over time without losing data.
"""

import sqlite3
import logging
from typing import Optional, List, Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class Migration:
    """Represents a single database migration."""

    def __init__(
        self,
        version: str,
        description: str,
        up_sql: str,
        down_sql: Optional[str] = None,
    ):
        """Initialize a migration.

        Args:
            version: Schema version (e.g., "0.3.0")
            description: Human-readable description
            up_sql: SQL to apply migration
            down_sql: SQL to rollback migration (optional)
        """
        self.version = version
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql


class MigrationRunner:
    """Runs database migrations in order."""

    def __init__(self, db_path: str):
        """Initialize migration runner.

        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.migrations: List[Migration] = []

    def register(self, migration: Migration) -> None:
        """Register a migration.

        Args:
            migration: Migration to register
        """
        self.migrations.append(migration)
        # Sort by version
        self.migrations.sort(key=lambda m: m.version)

    def get_current_version(self) -> Optional[str]:
        """Get current schema version from database.

        Returns:
            Current version or None if no schema_version table
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Check if schema_version table exists first
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                # Create schema_version table if it doesn't exist
                from victor.agent.conversation.migrations import ensure_schema_version_table
                ensure_schema_version_table(conn)

            cursor.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    def run_migrations(self) -> None:
        """Run all pending migrations."""
        current_version = self.get_current_version()

        for migration in self.migrations:
            if current_version is None or migration.version > current_version:
                logger.info(
                    f"Running migration {migration.version}: {migration.description}"
                )
                self._apply_migration(migration)

    def _apply_migration(self, migration: Migration) -> None:
        """Apply a single migration.

        Args:
            migration: Migration to apply
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Execute migration in transaction
            with conn:
                # Apply migration SQL
                conn.executescript(migration.up_sql)

                # Update schema version
                conn.execute(
                    """
                    INSERT INTO schema_version (version, applied_at)
                    VALUES (?, ?)
                    """,
                    (migration.version, datetime.utcnow().isoformat()),
                )

            logger.info(f"Migration {migration.version} applied successfully")
        except Exception as e:
            conn.rollback()
            logger.error(f"Migration {migration.version} failed: {e}")
            raise
        finally:
            conn.close()


# Define migrations
MIGRATIONS = [
    Migration(
        version="0.3.0",
        description="Add timestamp column to messages table and align schemas",
        up_sql="""
            -- This migration is handled by custom Python logic below
            -- due to the need for conditional schema changes
        """,
    ),
]


def apply_migration_0_3_0(db_path: str) -> bool:
    """Apply migration 0.3.0 - Add timestamp column and align schemas.

    This is a custom migration that checks the current schema and applies
    appropriate changes based on what exists.

    Args:
        db_path: Path to database file

    Returns:
        True if migration was applied, False if already up-to-date
    """
    conn = sqlite3.connect(db_path)
    try:
        # Check current messages table schema
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(messages)")
        columns = {col[1]: col[2] for col in cursor.fetchall()}

        # Check if we need to migrate (has created_at but no timestamp)
        needs_migration = "created_at" in columns and "timestamp" not in columns

        if not needs_migration:
            # Check if timestamp column exists but other columns are missing
            if "timestamp" in columns:
                # Check if we need to add other columns
                missing_columns = []
                required_columns = {
                    "token_count": "INTEGER",
                    "priority": "INTEGER",
                    "tool_name": "TEXT",
                    "tool_call_id": "TEXT",
                    "metadata": "TEXT",
                }

                for col_name, col_type in required_columns.items():
                    if col_name not in columns:
                        missing_columns.append((col_name, col_type))

                if missing_columns:
                    # Add missing columns
                    for col_name, col_type in missing_columns:
                        cursor.execute(f"ALTER TABLE messages ADD COLUMN {col_name} {col_type}")
                    conn.commit()
                    logger.info(f"Added {len(missing_columns)} missing columns to messages table")
                    return True
                else:
                    logger.info("Messages table schema is already up-to-date")
                    return False
            else:
                logger.info("Messages table schema is already up-to-date")
                return False

        # Perform migration from created_at to timestamp schema
        logger.info("Migrating messages table from created_at to timestamp schema")

        # Add timestamp column
        cursor.execute("ALTER TABLE messages ADD COLUMN timestamp TIMESTAMP")

        # Migrate data from created_at to timestamp
        cursor.execute("UPDATE messages SET timestamp = created_at WHERE timestamp IS NULL")

        # Add missing columns
        cursor.execute("ALTER TABLE messages ADD COLUMN token_count INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE messages ADD COLUMN priority INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE messages ADD COLUMN tool_call_id TEXT")
        cursor.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")

        # Migrate tool_calls to tool_name
        cursor.execute("UPDATE messages SET tool_name = tool_calls WHERE tool_calls IS NOT NULL")

        # Now recreate the table with proper schema (SQLite doesn't support ALTER COLUMN)
        cursor.execute("""
            CREATE TABLE messages_new (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                priority INTEGER NOT NULL DEFAULT 0,
                tool_name TEXT,
                tool_call_id TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            )
        """)

        # Copy data, converting INTEGER id to TEXT if needed
        cursor.execute("""
            INSERT INTO messages_new (
                id, session_id, role, content, timestamp,
                token_count, priority, tool_name, tool_call_id, metadata
            )
            SELECT
                CASE
                    WHEN typeof(id) = 'integer' THEN cast(id as text)
                    ELSE id
                END as id,
                session_id, role, content,
                COALESCE(timestamp, created_at) as timestamp,
                COALESCE(token_count, 0),
                COALESCE(priority, 0),
                tool_name,
                tool_call_id,
                metadata
            FROM messages
        """)

        # Drop old table and rename
        cursor.execute("DROP TABLE messages")
        cursor.execute("ALTER TABLE messages_new RENAME TO messages")

        # Recreate indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_priority ON messages(session_id, priority DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_tool_results ON messages(session_id, role, tool_name, timestamp DESC) WHERE role IN ('tool_call', 'tool_result')")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_exchange ON messages(session_id, role, timestamp) WHERE role IN ('user', 'assistant')")

        conn.commit()
        logger.info("Successfully migrated messages table to timestamp schema")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Migration 0.3.0 failed: {e}")
        raise
    finally:
        conn.close()


def migrate_database(db_path: str) -> None:
    """Migrate a database to the latest schema version.

    Args:
        db_path: Path to database file
    """
    # Run custom migration 0.3.0 for timestamp column
    try:
        apply_migration_0_3_0(db_path)
    except Exception as e:
        logger.warning(f"Migration 0.3.0 failed (may already be applied): {e}")

    # Update schema version to 0.3.0
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema_version_table(conn)

        # Check if version 0.3.0 is already recorded
        cursor = conn.cursor()
        cursor.execute(
            "SELECT version FROM schema_version WHERE version = ?",
            ("0.3.0",)
        )
        if cursor.fetchone() is None:
            # Insert version record
            cursor.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                ("0.3.0", datetime.utcnow().isoformat()),
            )
            conn.commit()
            logger.info(f"Database {db_path} upgraded to schema version 0.3.0")
    finally:
        conn.close()


def ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    """Ensure schema_version table exists.

    Args:
        conn: Database connection
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL
        );
        """
    )
