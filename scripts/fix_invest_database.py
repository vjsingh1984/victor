#!/usr/bin/env python3
"""Quick fix for victor-invest database schema issues.

This script recreates ONLY the conversation-related tables with the correct schema
while preserving all graph data (graph_node, graph_edge, file_changes, etc.).

Usage:
    cd victor-invest
    python ../codingagent/scripts/fix_invest_database.py
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime

def fix_database(db_path: Path = Path(".victor/project.db")):
    """Fix database schema by recreating conversation tables."""

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False

    # Backup
    backup_path = db_path.with_suffix('.db.backup')
    shutil.copy2(db_path, backup_path)
    print(f"✅ Created backup: {backup_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check what needs to be fixed
        cursor.execute("PRAGMA table_info(sessions)")
        sessions_cols = {col[1]: col[2] for col in cursor.fetchall()}

        needs_fix = 'id' in sessions_cols and 'session_id' not in sessions_cols

        if not needs_fix:
            print("ℹ️  Database schema appears to be already correct")
            return True

        print("🔧 Fixing database schema...")

        # Drop old conversation-related tables (graph data stays intact)
        old_tables = [
            'sessions', 'messages', 'context_sizes', 'context_summaries',
            'model_families', 'model_sizes', 'providers'
        ]

        for table in old_tables:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"  ✓ Dropped old {table}")
            except Exception as e:
                print(f"  ⚠️  Could not drop {table}: {e}")

        # Recreate tables with correct schema (from ConversationStore)
        _create_correct_schema(cursor)

        conn.commit()
        print("✅ Database schema fixed successfully!")

        # Verify graph data is preserved
        cursor.execute("SELECT COUNT(*) FROM graph_node")
        node_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM graph_edge")
        edge_count = cursor.fetchone()[0]

        print(f"\n📊 Graph data preserved:")
        print(f"  • {node_count} graph nodes")
        print(f"  • {edge_count} graph edges")

        return True

    except Exception as e:
        print(f"❌ Fix failed: {e}")
        # Restore from backup
        shutil.copy2(backup_path, db_path)
        print(f"🔄 Restored from backup")
        return False
    finally:
        conn.close()

def _create_correct_schema(cursor: sqlite3.Cursor):
    """Create tables with correct schema matching ConversationStore."""

    # Providers
    cursor.execute("""
        CREATE TABLE providers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT
        )
    """)

    # Model families
    cursor.execute("""
        CREATE TABLE model_families (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            provider_id INTEGER,
            FOREIGN KEY (provider_id) REFERENCES providers(id)
                ON DELETE SET NULL
        )
    """)

    # Model sizes
    cursor.execute("""
        CREATE TABLE model_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            family_id INTEGER,
            num_parameters INTEGER,
            FOREIGN KEY (family_id) REFERENCES model_families(id)
                ON DELETE SET NULL
        )
    """)

    # Context sizes
    cursor.execute("""
        CREATE TABLE context_sizes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            min_tokens INTEGER,
            max_tokens INTEGER
        )
    """)

    # Sessions
    cursor.execute("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            provider_id INTEGER,
            model_family_id INTEGER,
            model_size_id INTEGER,
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            FOREIGN KEY (provider_id) REFERENCES providers(id)
                ON DELETE SET NULL,
            FOREIGN KEY (model_family_id) REFERENCES model_families(id)
                ON DELETE SET NULL,
            FOREIGN KEY (model_size_id) REFERENCES model_sizes(id)
                ON DELETE SET NULL
        )
    """)

    # Messages
    cursor.execute("""
        CREATE TABLE messages (
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

    # Context summaries
    cursor.execute("""
        CREATE TABLE context_summaries (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            messages_summarized TEXT,
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                ON DELETE CASCADE
        )
    """)

    # Indexes
    cursor.execute("CREATE INDEX idx_messages_session_time ON messages(session_id, timestamp)")
    cursor.execute("CREATE INDEX idx_messages_priority ON messages(session_id, priority DESC)")
    cursor.execute("CREATE INDEX idx_summaries_session ON context_summaries(session_id, created_at DESC)")

    # Schema version
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL
        )
    """)
    cursor.execute(
        "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
        ("0.3.0", datetime.now().isoformat())
    )

if __name__ == "__main__":
    import sys

    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".victor/project.db")

    print(f"🔧 Fixing database: {db_path}\n")
    success = fix_database(db_path)

    sys.exit(0 if success else 1)
