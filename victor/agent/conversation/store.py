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

"""
SQLite-based conversation store with token-aware context management.

This module provides persistent conversation state management with:
- SQLite persistence for session recovery
- Token-aware context window management
- Priority-based message pruning
- Semantic relevance scoring for context selection

The primary class is `ConversationStore`.

For simpler alternatives:
- In-memory: see `victor.agent.message_history.MessageHistory`
- JSON file-based: see `victor.agent.session.SessionPersistence`

Usage:
    store = ConversationStore()
    session = store.create_session(project_path="/path/to/project")

    # Add messages
    store.add_message(session.session_id, MessageRole.USER, "Hello")
    store.add_message(session.session_id, MessageRole.ASSISTANT, "Hi there!")

    # Get context for provider
    messages = store.get_context_messages(session.session_id)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import asyncio
import logging
import re
import sqlite3

from victor.core.async_utils import run_sync
from victor.core.json_utils import json_dumps, json_loads
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
import uuid

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore

logger = logging.getLogger(__name__)


# Canonical enums from conversation/types.py
from victor.agent.conversation.types import ConversationMessage, MessageRole, MessagePriority

# ML metadata extracted to victor/agent/ml_metadata.py
from victor.agent.ml_metadata import (  # noqa: F401
    ContextSize,
    ModelFamily,
    ModelMetadata,
    ModelSize,
    get_known_model_context,
    get_known_model_params,
    parse_model_metadata,
)

# ConversationMessage is imported from types.py — single canonical definition.
# Previously duplicated here; consolidated to avoid field drift.


@dataclass
class ConversationSession:
    """A conversation session with context management."""

    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Context management
    max_tokens: int = 100000  # Claude's context window
    reserved_tokens: int = 4096  # Reserved for response
    current_tokens: int = 0

    # Session metadata
    project_path: Optional[str] = None
    active_files: List[str] = field(default_factory=list)
    tool_usage_count: int = 0

    # Provider info (original fields)
    provider: Optional[str] = None
    model: Optional[str] = None
    profile: Optional[str] = None  # User-facing profile name (e.g., "groq-fast")

    # ML/RL-friendly derived fields for multi-dimensional learning
    model_family: Optional[ModelFamily] = None  # Architecture family
    model_size: Optional[ModelSize] = None  # Size category
    model_params_b: Optional[float] = None  # Parameters in billions (numeric)
    context_size: Optional[ContextSize] = None  # Context window category
    context_tokens: Optional[int] = None  # Actual context window tokens
    tool_capable: bool = False  # Whether model supports tool calling
    is_moe: bool = False  # Mixture of Experts architecture
    is_reasoning: bool = False  # Explicit reasoning model (R1, o1)

    @property
    def available_tokens(self) -> int:
        """Get available tokens for new messages."""
        return self.max_tokens - self.reserved_tokens - self.current_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "max_tokens": self.max_tokens,
            "reserved_tokens": self.reserved_tokens,
            "current_tokens": self.current_tokens,
            "project_path": self.project_path,
            "active_files": self.active_files,
            "tool_usage_count": self.tool_usage_count,
            "provider": self.provider,
            "model": self.model,
            "profile": self.profile,
            # ML-friendly fields
            "model_family": self.model_family.value if self.model_family else None,
            "model_size": self.model_size.value if self.model_size else None,
            "model_params_b": self.model_params_b,
            "context_size": self.context_size.value if self.context_size else None,
            "context_tokens": self.context_tokens,
            "tool_capable": self.tool_capable,
            "is_moe": self.is_moe,
            "is_reasoning": self.is_reasoning,
        }


class ConversationStore:
    """
    SQLite-based conversation store with intelligent context pruning.

    Features:
    - SQLite persistence for session recovery
    - Token-aware context window management
    - Priority-based message pruning
    - Semantic relevance scoring for context selection
    - Normalized schema for efficient ML/RL aggregation queries

    Schema Design:
    - Lookup tables (model_families, model_sizes, context_sizes, providers)
      store categorical values with INTEGER PRIMARY KEYs
    - Sessions table uses INTEGER FK columns for efficient joins and aggregation
    - Indexes on FK columns for fast GROUP BY queries

    For simpler in-memory storage, see MessageHistory.
    For JSON file-based persistence, see SessionPersistence.

    Usage:
        store = ConversationStore()
        session = store.create_session(project_path="/path/to/project")
        store.add_message(session.session_id, MessageRole.USER, "Hello")
        messages = store.get_context_messages(session.session_id)
    """

    # Schema version for migration tracking (semver format: major.minor.patch)
    SCHEMA_VERSION = "0.3.0"  # Aligned with migration 0.3.0 (normalized schema)

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_context_tokens: int = 100000,
        response_reserve: int = 4096,
        chars_per_token: int = 4,
    ):
        """Initialize the conversation memory manager.

        Args:
            db_path: Path to SQLite database. Defaults to {project}/.victor/project.db
            max_context_tokens: Maximum tokens in context window
            response_reserve: Tokens reserved for response
            chars_per_token: Approximate characters per token for estimation
        """
        from victor.config.settings import get_project_paths

        paths = get_project_paths()
        self.db_path = db_path or paths.project_db
        self.max_context_tokens = max_context_tokens
        self.response_reserve = response_reserve
        self.chars_per_token = chars_per_token

        # Migration check: rename conversation.db to project.db if it exists
        legacy_db = paths.project_victor_dir / "conversation.db"
        if not db_path and legacy_db.exists() and not self.db_path.exists():
            try:
                legacy_db.rename(self.db_path)
                logger.info(f"Migrated legacy database {legacy_db} to {self.db_path}")
            except Exception as e:
                logger.warning(f"Failed to migrate legacy database: {e}")

        # In-memory session cache
        self._sessions: Dict[str, ConversationSession] = {}

        # ID lookup caches for normalized tables (populated lazily)
        self._model_family_ids: Dict[str, int] = {}
        self._model_size_ids: Dict[str, int] = {}
        self._context_size_ids: Dict[str, int] = {}
        self._provider_ids: Dict[str, int] = {}

        # Initialize database
        self._init_database()

        logger.debug(
            f"ConversationStore initialized. "
            f"DB: {self.db_path}, Max tokens: {max_context_tokens}"
        )

    def _get_connection(self) -> sqlite3.Connection:
        """Get a SQLite connection with optimized settings for concurrent access.

        Returns a connection with:
        - 30-second timeout for handling concurrent access
        - WAL mode for better read/write concurrency
        - Optimized pragmas for performance
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        # Ensure WAL mode is set (in case it wasn't during init)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_database(self) -> None:
        """Initialize SQLite database for persistence with normalized schema.

        Schema v2 Design (Normalized for ML/RL):
        - Lookup tables with INTEGER PKs for categorical values
        - Sessions table uses INTEGER FKs for efficient joins
        - Indexes on all FK columns for fast aggregation queries

        Database Architecture:
        - Primary: .victor/project.db (all project-specific data)
        - Legacy: Auto-migrate conversation.db → project.db if exists
        - Global: ~/.victor/victor.db (providers, models, cross-project settings)

        Migration Strategy:
        - Run pending migrations before schema initialization
        - Support legacy conversation.db → project.db migration
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Run database migrations first (before schema version check)
        try:
            from victor.agent.conversation.migrations import migrate_database

            migrate_database(str(self.db_path))
            logger.debug(f"Database migrations completed for {self.db_path}")
        except Exception as e:
            logger.warning(f"Database migration failed (will continue with existing schema): {e}")

        with sqlite3.connect(self.db_path) as conn:
            # SQLite performance optimizations
            conn.execute("PRAGMA journal_mode = WAL")  # Write-ahead logging
            conn.execute("PRAGMA synchronous = NORMAL")  # Balanced durability/speed
            conn.execute("PRAGMA cache_size = -2000")  # 2MB cache in RAM
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL")  # Auto space reclaim
            conn.execute("PRAGMA temp_store = MEMORY")  # Temp tables in memory

            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Create schema version tracking table (stores semver strings)
            from victor.agent.conversation.migrations import ensure_schema_version_table

            ensure_schema_version_table(conn)

            # Check if current schema version is applied
            cursor = conn.execute(
                "SELECT version FROM schema_version WHERE version = ?",
                (self.SCHEMA_VERSION,),
            )
            version_applied = cursor.fetchone() is not None

            if not version_applied:
                # First, migrate existing sessions table if it exists
                # This must happen BEFORE _apply_normalized_schema since that creates
                # indexes on columns that need to exist
                self._migrate_sessions_table(conn)

                # Apply normalized schema
                self._apply_normalized_schema(conn)
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (self.SCHEMA_VERSION, datetime.now().isoformat()),
                )
                conn.commit()
                logger.info(f"Database schema upgraded to v{self.SCHEMA_VERSION}")
            else:
                # Load ID caches for existing database
                self._load_lookup_caches(conn)

                # Run migration for any missing columns (in case of partial upgrade)
                self._migrate_sessions_table(conn)
                conn.commit()

        logger.debug(f"Database initialized at {self.db_path}")

    def _apply_normalized_schema(self, conn: sqlite3.Connection) -> None:
        """Apply the normalized schema with lookup tables.

        Creates:
        - model_families: Lookup table for model architecture families
        - model_sizes: Lookup table for parameter size categories
        - context_sizes: Lookup table for context window categories
        - providers: Lookup table for LLM provider names
        - sessions: Main table with INTEGER FKs
        - messages: Message storage with session FK
        - context_summaries: Compaction summaries
        """
        # Create lookup tables
        conn.executescript("""
            -- Model family lookup (llama, qwen, claude, gpt, etc.)
            CREATE TABLE IF NOT EXISTS model_families (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                provider_id INTEGER
            );

            -- Model size lookup (tiny, small, medium, large, xlarge, xxlarge)
            CREATE TABLE IF NOT EXISTS model_sizes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                family_id INTEGER,
                num_parameters INTEGER,
                FOREIGN KEY (family_id) REFERENCES model_families(id)
                    ON DELETE SET NULL
            );

            -- Context size lookup (small, medium, large, xlarge)
            CREATE TABLE IF NOT EXISTS context_sizes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                min_tokens INTEGER,
                max_tokens INTEGER
            );

            -- Provider lookup (anthropic, openai, ollama, groq, etc.)
            CREATE TABLE IF NOT EXISTS providers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            );

            -- Sessions table with normalized FK columns
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                last_activity TIMESTAMP NOT NULL,
                project_path TEXT,
                model TEXT,
                profile TEXT,
                max_tokens INTEGER DEFAULT 100000,
                reserved_tokens INTEGER DEFAULT 4096,
                metadata TEXT,
                -- Normalized FK columns for ML/RL
                provider_id INTEGER REFERENCES providers(id),
                model_family_id INTEGER REFERENCES model_families(id),
                model_size_id INTEGER REFERENCES model_sizes(id),
                context_size_id INTEGER REFERENCES context_sizes(id),
                -- Numeric columns for direct queries
                model_params_b REAL,
                context_tokens INTEGER,
                tool_capable INTEGER DEFAULT 0,
                is_moe INTEGER DEFAULT 0,
                is_reasoning INTEGER DEFAULT 0,
                -- Token accounting: actual API-returned counts (NULL until first API response)
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                cached_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                cost_usd_micros INTEGER DEFAULT 0
            );

            -- Messages table
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                token_count INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                tool_name TEXT,
                tool_call_id TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            -- Context summaries for pruned conversations
            CREATE TABLE IF NOT EXISTS context_summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                messages_summarized TEXT,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            -- Indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_messages_session_time
            ON messages(session_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_messages_priority
            ON messages(session_id, priority DESC);

            CREATE INDEX IF NOT EXISTS idx_summaries_session
            ON context_summaries(session_id, created_at DESC);

            -- Indexes on FK columns for ML/RL aggregation
            CREATE INDEX IF NOT EXISTS idx_sessions_provider
            ON sessions(provider_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_model_family
            ON sessions(model_family_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_model_size
            ON sessions(model_size_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_context_size
            ON sessions(context_size_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_tool_capable
            ON sessions(tool_capable);

            -- Composite indexes for common ML queries
            CREATE INDEX IF NOT EXISTS idx_sessions_family_size
            ON sessions(model_family_id, model_size_id);

            CREATE INDEX IF NOT EXISTS idx_sessions_provider_family
            ON sessions(provider_id, model_family_id);

            -- FTS5 virtual table for fast full-text search on messages
            -- Enables O(log n) keyword search vs O(n) linear scan
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                message_id UNINDEXED,
                session_id UNINDEXED,
                content='messages',
                content_rowid='rowid'
            );

            -- Triggers to keep FTS index in sync with messages table
            CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content, message_id, session_id)
                VALUES (NEW.rowid, NEW.content, NEW.id, NEW.session_id);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, message_id, session_id)
                VALUES ('delete', OLD.rowid, OLD.content, OLD.id, OLD.session_id);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, content, message_id, session_id)
                VALUES ('delete', OLD.rowid, OLD.content, OLD.id, OLD.session_id);
                INSERT INTO messages_fts(rowid, content, message_id, session_id)
                VALUES (NEW.rowid, NEW.content, NEW.id, NEW.session_id);
            END;

            -- Index for tool result retrieval (high-value messages for context)
            CREATE INDEX IF NOT EXISTS idx_messages_tool_results
            ON messages(session_id, role, tool_name, timestamp DESC)
            WHERE role IN ('tool_call', 'tool');

            -- Index for user-assistant exchanges
            CREATE INDEX IF NOT EXISTS idx_messages_exchange
            ON messages(session_id, role, timestamp)
            WHERE role IN ('user', 'assistant');
            """)

        # Populate lookup tables with enum values
        self._populate_lookup_tables(conn)

        # Load caches
        self._load_lookup_caches(conn)

        # Apply hybrid compaction schema enhancements
        self._apply_hybrid_compaction_schema(conn)

        # Auto-maintenance: vacuum if fragmented, cleanup stale sessions
        self._auto_maintenance()

    def _auto_maintenance(self) -> None:
        """Run lightweight maintenance on startup.

        - Purge test model sessions and empty sessions
        - Reclaim space if freelist exceeds threshold
        """
        _FREELIST_VACUUM_THRESHOLD = 100  # pages (~400KB at 4KB/page)

        try:
            # Cleanup stale/test sessions
            self.cleanup_stale_sessions(
                max_age_days=0,  # don't TTL on startup — only purge junk
                purge_test_models=True,
                purge_empty=True,
            )

            # Auto-vacuum if fragmented
            with sqlite3.connect(self.db_path) as conn:
                freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
                if freelist > _FREELIST_VACUUM_THRESHOLD:
                    conn.execute("PRAGMA incremental_vacuum")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    logger.info(f"Auto-vacuum: reclaimed {freelist} free pages")
        except sqlite3.Error as e:
            logger.debug(f"Auto-maintenance skipped: {e}")

    def _apply_hybrid_compaction_schema(self, conn: sqlite3.Connection) -> None:
        """Apply enhanced schema for hybrid compaction system.

        Adds JSON1 extension support, new columns for dual-format storage,
        and analytics tables for monitoring compaction performance.

        This is a non-breaking migration - all new columns have DEFAULT values.
        """
        try:
            # Enable JSON1 extension (built-in to Python's sqlite3)
            conn.execute("SELECT json_extract('{\"test\": 1}', '$.test')")

            # Add new columns to messages table (if not exist)
            messages_columns = [
                ("metadata_json", "TEXT DEFAULT '{}'"),  # JSON metadata
                ("priority", "INTEGER DEFAULT 50"),  # Message priority
            ]

            for col_name, col_def in messages_columns:
                try:
                    conn.execute(f"ALTER TABLE messages ADD COLUMN {col_name} {col_def}")
                    logger.info(f"Added column {col_name} to messages table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Failed to add column {col_name} to messages: {e}")

            # Add new columns to context_summaries table (if not exist)
            summary_columns = [
                ("summary_format", "TEXT DEFAULT 'natural'"),  # 'xml', 'natural', 'both'
                ("summary_xml", "TEXT"),  # Machine-readable XML format
                ("summary_text", "TEXT"),  # Natural language format
                ("summary_json", "TEXT DEFAULT '{}'"),  # Structured summary data
                ("messages_summarized_json", "TEXT DEFAULT '[]'"),  # Native JSON array
                ("strategy_used", "TEXT"),  # 'rule_based', 'llm_based', 'hybrid'
                ("complexity_score", "REAL"),  # 0.0-1.0 complexity score
                ("estimated_tokens_saved", "INTEGER DEFAULT 0"),  # Token savings
            ]

            for col_name, col_def in summary_columns:
                try:
                    conn.execute(f"ALTER TABLE context_summaries ADD COLUMN {col_name} {col_def}")
                    logger.info(f"Added column {col_name} to context_summaries table")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Failed to add column {col_name} to context_summaries: {e}")

            # Create compaction_history table for analytics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compaction_history (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    strategy_used TEXT NOT NULL,
                    message_count_before INTEGER NOT NULL,
                    message_count_after INTEGER NOT NULL,
                    token_count_before INTEGER NOT NULL,
                    token_count_after INTEGER NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    llm_provider TEXT,
                    llm_model TEXT,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Create topic_segments table (future-proofing for topic-aware segmentation)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS topic_segments (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    topic_name TEXT NOT NULL,
                    start_message_id TEXT NOT NULL,
                    end_message_id TEXT,
                    message_count INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (start_message_id) REFERENCES messages(id),
                    FOREIGN KEY (end_message_id) REFERENCES messages(id)
                )
            """)

            # Create indexes for enhanced queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_strategy
                ON context_summaries(strategy_used, created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_compaction_history_session
                ON compaction_history(session_id, created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_compaction_history_strategy
                ON compaction_history(strategy_used, created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_segments_session
                ON topic_segments(session_id, created_at)
            """)

            logger.info("Applied hybrid compaction schema enhancements")

        except sqlite3.Error as e:
            logger.warning(f"Failed to apply hybrid compaction schema: {e}")

    def _populate_lookup_tables(self, conn: sqlite3.Connection) -> None:
        """Populate lookup tables with predefined enum values."""
        # Model families
        families = [
            ("llama", None),  # provider_id will be set later
            ("qwen", None),
            ("mistral", None),
            ("mixtral", None),
            ("claude", None),
            ("gpt", None),
            ("gemini", None),
            ("deepseek", None),
            ("phi", None),
            ("codellama", None),
            ("command", None),
            ("grok", None),
            ("unknown", None),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO model_families (name, provider_id) VALUES (?, ?)",
            families,
        )

        # Model sizes with parameter ranges (using midpoint as num_parameters)
        sizes = [
            ("tiny", None, 0),  # <1B (use midpoint, family_id NULL)
            ("small", None, 4),  # 1-8B (midpoint: 4B)
            ("medium", None, 20),  # 8-32B (midpoint: 20B)
            ("large", None, 51),  # 32-70B (midpoint: 51B)
            ("xlarge", None, 122),  # 70-175B (midpoint: 122B)
            ("xxlarge", None, 175),  # >175B (minimum)
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO model_sizes (name, family_id, num_parameters) VALUES (?, ?, ?)",
            sizes,
        )

        # Context sizes with token ranges
        ctx_sizes = [
            ("small", 0, 8000),  # <8K
            ("medium", 8000, 32000),  # 8K-32K
            ("large", 32000, 128000),  # 32K-128K
            ("xlarge", 128000, 999999999),  # 128K+
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO context_sizes (name, min_tokens, max_tokens) VALUES (?, ?, ?)",
            ctx_sizes,
        )

        # Common providers
        providers = [
            ("anthropic", "Anthropic Claude API"),
            ("openai", "OpenAI API"),
            ("ollama", "Local Ollama server"),
            ("groq", "Groq Cloud API"),
            ("google", "Google Gemini API"),
            ("xai", "xAI Grok API"),
            ("deepseek", "DeepSeek API"),
            ("mistral", "Mistral AI API"),
            ("lmstudio", "Local LMStudio server"),
            ("vllm", "vLLM server"),
            ("together", "Together AI API"),
            ("openrouter", "OpenRouter API"),
            ("moonshot", "Moonshot/Kimi API"),
            ("fireworks", "Fireworks AI API"),
            ("cerebras", "Cerebras API"),
            ("unknown", "Unknown provider"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO providers (name, description) VALUES (?, ?)",
            providers,
        )

        conn.commit()

    def _load_lookup_caches(self, conn: sqlite3.Connection) -> None:
        """Load lookup table IDs into memory caches for fast lookup."""
        # Model families
        rows = conn.execute("SELECT name, id FROM model_families").fetchall()
        self._model_family_ids = {row[0]: row[1] for row in rows}

        # Model sizes
        rows = conn.execute("SELECT name, id FROM model_sizes").fetchall()
        self._model_size_ids = {row[0]: row[1] for row in rows}

        # Context sizes
        rows = conn.execute("SELECT name, id FROM context_sizes").fetchall()
        self._context_size_ids = {row[0]: row[1] for row in rows}

        # Providers
        rows = conn.execute("SELECT name, id FROM providers").fetchall()
        self._provider_ids = {row[0]: row[1] for row in rows}

        logger.debug(
            f"Loaded lookup caches: {len(self._model_family_ids)} families, "
            f"{len(self._provider_ids)} providers"
        )

    def _migrate_sessions_table(self, conn: sqlite3.Connection) -> None:
        """Add missing columns to sessions table for schema migration.

        This handles the case where an old database exists with a sessions table
        that was created before the normalized schema was introduced.
        The CREATE TABLE IF NOT EXISTS doesn't add columns to existing tables,
        so we need to use ALTER TABLE to add missing columns.
        """
        # Check if sessions table exists
        cursor = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        result = cursor.fetchone()
        if not result:
            # No sessions table - nothing to migrate
            return

        table_sql = result[0]

        # Define columns that need to be added if missing
        # Format: (column_name, column_definition)
        new_columns = [
            ("provider_id", "INTEGER"),
            ("model_family_id", "INTEGER"),
            ("model_size_id", "INTEGER"),
            ("context_size_id", "INTEGER"),
            ("model_params_b", "REAL"),
            ("context_tokens", "INTEGER"),
            ("tool_capable", "INTEGER DEFAULT 0"),
            ("is_moe", "INTEGER DEFAULT 0"),
            ("is_reasoning", "INTEGER DEFAULT 0"),
            # Token accounting columns (added for actual API token tracking)
            ("prompt_tokens", "INTEGER DEFAULT 0"),
            ("completion_tokens", "INTEGER DEFAULT 0"),
            ("cached_tokens", "INTEGER DEFAULT 0"),
            ("reasoning_tokens", "INTEGER DEFAULT 0"),
            ("cost_usd_micros", "INTEGER DEFAULT 0"),
        ]

        columns_added = []
        for col_name, col_def in new_columns:
            # Check if column exists in table definition
            if col_name not in table_sql:
                try:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {col_name} {col_def}")
                    columns_added.append(col_name)
                except sqlite3.OperationalError as e:
                    # Column might already exist (race condition or partial migration)
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Failed to add column {col_name} to sessions: {e}")

        if columns_added:
            logger.info(f"Migrated sessions table: added columns {columns_added}")

    def _get_or_create_provider_id(
        self, conn: sqlite3.Connection, provider: Optional[str]
    ) -> Optional[int]:
        """Get or create provider ID, with caching."""
        if not provider:
            return None

        provider_lower = provider.lower()

        # Check cache first
        if provider_lower in self._provider_ids:
            return self._provider_ids[provider_lower]

        # Insert and get ID
        try:
            conn.execute(
                "INSERT OR IGNORE INTO providers (name) VALUES (?)",
                (provider_lower,),
            )
            cursor = conn.execute(
                "SELECT id FROM providers WHERE name = ?",
                (provider_lower,),
            )
            row = cursor.fetchone()
            if row:
                self._provider_ids[provider_lower] = row[0]
                return row[0]
        except sqlite3.Error as e:
            logger.warning(f"Failed to get/create provider ID for {provider}: {e}")

        return None

    def _get_model_family_id(self, family: Optional[ModelFamily]) -> Optional[int]:
        """Get model family ID from cache."""
        if not family:
            return None
        return self._model_family_ids.get(family.value)

    def _get_model_size_id(self, size: Optional[ModelSize]) -> Optional[int]:
        """Get model size ID from cache."""
        if not size:
            return None
        return self._model_size_ids.get(size.value)

    def _get_context_size_id(self, ctx_size: Optional[ContextSize]) -> Optional[int]:
        """Get context size ID from cache."""
        if not ctx_size:
            return None
        return self._context_size_ids.get(ctx_size.value)

    def create_session(
        self,
        session_id: Optional[str] = None,
        project_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        profile: Optional[str] = None,
        tool_capable: bool = False,
    ) -> ConversationSession:
        """Create a new conversation session.

        Args:
            session_id: Optional session ID. Generated if not provided.
            project_path: Path to the project being worked on
            provider: LLM provider name
            model: Model identifier
            max_tokens: Override max context tokens
            profile: User-facing profile name (e.g., "groq-fast")
            tool_capable: Whether the model supports tool calling

        Returns:
            New ConversationSession instance
        """
        if session_id is None:
            session_id = self._generate_session_id()

        # Auto-populate ML fields from model metadata
        model_family = None
        model_size = None
        model_params_b = None
        context_size = None
        context_tokens = None
        is_moe = False
        is_reasoning = False

        if model:
            # Get known values first
            known_context = get_known_model_context(model)
            known_params = get_known_model_params(model)

            # Parse metadata from model name
            metadata = parse_model_metadata(
                model,
                provider=provider,
                known_context=known_context,
                known_params_b=known_params,
            )

            model_family = metadata.model_family
            model_size = metadata.model_size
            model_params_b = metadata.model_params_b
            context_size = metadata.context_size
            context_tokens = metadata.context_tokens
            is_moe = metadata.is_moe
            is_reasoning = metadata.is_reasoning

        session = ConversationSession(
            session_id=session_id,
            project_path=project_path,
            provider=provider,
            model=model,
            profile=profile,
            max_tokens=max_tokens or self.max_context_tokens,
            reserved_tokens=self.response_reserve,
            # ML-friendly fields
            model_family=model_family,
            model_size=model_size,
            model_params_b=model_params_b,
            context_size=context_size,
            context_tokens=context_tokens,
            tool_capable=tool_capable,
            is_moe=is_moe,
            is_reasoning=is_reasoning,
        )

        self._sessions[session_id] = session
        self._persist_session(session)

        fam = model_family.value if model_family else "unknown"
        sz = model_size.value if model_size else "unknown"
        logger.info(f"Created session {session_id} for project: " f"{project_path} [{fam}/{sz}]")

        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ConversationSession or None if not found
        """
        # Check cache first
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try to load from database
        session = self._load_session(session_id)
        if session:
            self._sessions[session_id] = session

        return session

    def list_sessions(
        self,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[ConversationSession]:
        """List recent sessions with JOINs to lookup tables.

        Args:
            project_path: Filter by project path
            limit: Maximum sessions to return

        Returns:
            List of sessions ordered by last activity
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            base_query = """
                SELECT
                    s.*,
                    p.name AS provider_name,
                    mf.name AS model_family_name,
                    ms.name AS model_size_name,
                    cs.name AS context_size_name
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                LEFT JOIN model_families mf ON s.model_family_id = mf.id
                LEFT JOIN model_sizes ms ON s.model_size_id = ms.id
                LEFT JOIN context_sizes cs ON s.context_size_id = cs.id
            """

            if project_path:
                rows = conn.execute(
                    f"""
                    {base_query}
                    WHERE s.project_path = ?
                    ORDER BY s.last_activity DESC
                    LIMIT ?
                    """,
                    (project_path, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    {base_query}
                    ORDER BY s.last_activity DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            sessions = []
            for row in rows:
                session = self._session_from_row(row)
                sessions.append(session)

            return sessions

    def _add_message_impl(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        priority: Optional[MessagePriority],
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        tool_calls: Optional[List],
    ) -> ConversationMessage:
        """Shared implementation for adding a message.

        This contains all the common logic for both sync and async versions.
        The only difference between sync/async is how persistence is handled.

        Args:
            session_id: Session identifier
            role: Message role
            content: Message content
            priority: Message priority (auto-determined if None)
            tool_name: Tool name for tool calls
            tool_call_id: Tool call ID
            metadata: Additional metadata
            tool_calls: Tool calls list

        Returns:
            Created ConversationMessage
        """
        session = self._get_or_create_session(session_id)

        # Auto-determine priority based on role
        if priority is None:
            priority = self._determine_priority(role, tool_name)

        # Estimate token count
        token_count = self._estimate_tokens(content)
        trace_metadata = self._build_trace_metadata(
            role=role,
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata,
            tool_calls=tool_calls,
        )
        merged_metadata = dict(metadata or {})
        merged_metadata.update(trace_metadata)

        message = ConversationMessage(
            id=self._generate_message_id(),
            role=role,
            content=content,
            timestamp=datetime.now(),
            token_count=token_count,
            priority=priority,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=merged_metadata,
            tool_calls=tool_calls,
        )

        session.messages.append(message)
        session.current_tokens += token_count
        session.last_activity = datetime.now()

        # Track tool usage
        if role in (MessageRole.TOOL_CALL, MessageRole.TOOL):
            session.tool_usage_count += 1

        # Check if pruning is needed
        if session.current_tokens > (session.max_tokens - session.reserved_tokens):
            self._prune_context(session)

        return message

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        priority: Optional[MessagePriority] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List] = None,
    ) -> ConversationMessage:
        """Add a message to the conversation.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system, etc.)
            content: Message content
            priority: Message priority for pruning. Auto-determined if not provided.
            tool_name: Tool name for tool calls/results
            tool_call_id: Tool call ID for correlation
            metadata: Additional metadata
            tool_calls: Tool calls list for assistant messages (OpenAI spec)

        Returns:
            Created ConversationMessage
        """
        # Call shared implementation
        message = self._add_message_impl(
            session_id, role, content, priority, tool_name, tool_call_id, metadata, tool_calls
        )

        # Persist (sync SQLite I/O)
        self._persist_message(session_id, message)
        self._update_session_activity(session_id)

        # NOTE: Lazy embedding - embeddings created on search, not on add
        # This reduces write overhead and file proliferation

        logger.debug(
            f"Added {role.value} message to {session_id}. " f"Tokens: {message.token_count}"
        )

        return message

    def add_system_message(
        self,
        session_id: str,
        content: str,
    ) -> ConversationMessage:
        """Add a system message with CRITICAL priority.

        Args:
            session_id: Session identifier
            content: System prompt content

        Returns:
            Created ConversationMessage
        """
        return self.add_message(
            session_id,
            MessageRole.SYSTEM,
            content,
            priority=MessagePriority.CRITICAL,
        )

    def get_context_messages(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get messages formatted for the provider, respecting token limits.

        Args:
            session_id: Session identifier
            max_tokens: Override max tokens for this retrieval
            include_system: Whether to include system messages

        Returns:
            List of messages in provider format
        """
        session = self.get_session(session_id)
        if not session:
            return []

        max_tokens = max_tokens or (session.max_tokens - session.reserved_tokens)

        # Score and select messages
        scored_messages = self._score_messages(session.messages)

        # Select messages within token budget
        selected = []
        token_budget = max_tokens

        for msg, _score in scored_messages:
            if msg.token_count <= token_budget:
                selected.append(msg)
                token_budget -= msg.token_count

        # Sort by timestamp for coherent conversation
        selected.sort(key=lambda m: m.timestamp)

        # Filter system messages if requested
        if not include_system:
            selected = [m for m in selected if m.role != MessageRole.SYSTEM]

        logger.debug(
            f"Selected {len(selected)} messages for context. "
            f"Tokens used: {max_tokens - token_budget}/{max_tokens}"
        )

        return [msg.to_provider_format() for msg in selected]

    def get_recent_messages(
        self,
        session_id: str,
        count: int = 10,
    ) -> List[ConversationMessage]:
        """Get the most recent messages.

        Args:
            session_id: Session identifier
            count: Number of messages to return

        Returns:
            List of recent messages
        """
        session = self.get_session(session_id)
        if not session:
            return []

        return session.messages[-count:]

    def clear_session(self, session_id: str):
        """Clear all messages from a session.

        Args:
            session_id: Session identifier
        """
        session = self.get_session(session_id)
        if session:
            session.messages.clear()
            session.current_tokens = 0

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (session_id,),
                )

            logger.info(f"Cleared session {session_id}")

    def delete_session(self, session_id: str):
        """Delete a session and all its messages.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        logger.info(f"Deleted session {session_id}")

    def vacuum(self) -> Dict[str, Any]:
        """Reclaim unused space in the database.

        Performs incremental auto-vacuum plus a checkpoint to merge WAL file.
        This is useful after bulk deletions to reduce database file size.

        Returns:
            Dictionary with before/after file sizes and space freed
        """
        import os

        before_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0

        with sqlite3.connect(self.db_path) as conn:
            # Run incremental auto-vacuum
            conn.execute("PRAGMA incremental_vacuum")
            # Checkpoint WAL to main database
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        after_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0
        freed = before_size - after_size

        logger.info(
            f"Database vacuum: {before_size / 1024:.1f}KB -> {after_size / 1024:.1f}KB "
            f"(freed {freed / 1024:.1f}KB)"
        )

        return {
            "before_bytes": before_size,
            "after_bytes": after_size,
            "freed_bytes": freed,
        }

    # =========================================================================
    # DATABASE CLEANUP
    # =========================================================================

    # Known test model names that should not exist in production DB
    TEST_MODEL_NAMES = frozenset(
        {
            "test-model",
            "dummy",
            "fake",
            "dummy-model",
            "specific-tool-model",
            "unknown-model",
            "totally-unknown-model-xyz",
            "mock-model",
            "test-provider-model",
            "gpt-oss:20b",
        }
    )

    def cleanup_stale_sessions(
        self,
        max_age_days: int = 30,
        purge_test_models: bool = True,
        purge_empty: bool = True,
    ) -> Dict[str, int]:
        """Remove stale, test, and empty sessions from SQLite.

        This prevents unbounded database growth from test artifacts,
        abandoned sessions, and expired history.

        Args:
            max_age_days: Delete sessions older than this (0 = skip)
            purge_test_models: Delete sessions with test model names
            purge_empty: Delete sessions that have zero messages

        Returns:
            Dictionary with counts of deleted sessions by category
        """
        deleted = {"test_models": 0, "empty": 0, "stale": 0}

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # 1. Purge test model sessions
                if purge_test_models:
                    placeholders = ",".join("?" * len(self.TEST_MODEL_NAMES))
                    cursor = conn.execute(
                        f"DELETE FROM sessions WHERE model IN " f"({placeholders})",
                        list(self.TEST_MODEL_NAMES),
                    )
                    deleted["test_models"] = cursor.rowcount

                # 2. Purge empty sessions (no messages)
                if purge_empty:
                    cursor = conn.execute(
                        "DELETE FROM sessions WHERE session_id NOT IN "
                        "(SELECT DISTINCT session_id FROM messages)"
                    )
                    deleted["empty"] = cursor.rowcount

                # 3. Purge sessions older than max_age_days
                if max_age_days > 0:
                    from datetime import timedelta

                    cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
                    cursor = conn.execute(
                        "DELETE FROM sessions " "WHERE last_activity < ?",
                        (cutoff,),
                    )
                    deleted["stale"] = cursor.rowcount

            total = sum(deleted.values())
            if total > 0:
                logger.info(
                    f"Cleaned up {total} sessions: "
                    f"{deleted['test_models']} test, "
                    f"{deleted['empty']} empty, "
                    f"{deleted['stale']} stale"
                )
                # Also clear in-memory caches for deleted sessions
                self._sessions.clear()

        except sqlite3.Error as e:
            logger.warning(f"Session cleanup failed: {e}")

        return deleted

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics including file size and record counts.

        Returns:
            Dictionary with database stats
        """
        import os

        file_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0

        with sqlite3.connect(self.db_path) as conn:
            session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

            # Get WAL mode info
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

        return {
            "file_path": str(self.db_path),
            "file_size_bytes": file_size,
            "file_size_kb": file_size / 1024,
            "session_count": session_count,
            "message_count": message_count,
            "journal_mode": journal_mode,
        }

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary of session statistics
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        role_counts = {}
        for msg in session.messages:
            role_counts[msg.role.value] = role_counts.get(msg.role.value, 0) + 1

        return {
            "session_id": session_id,
            "message_count": len(session.messages),
            "total_tokens": session.current_tokens,
            "available_tokens": session.available_tokens,
            "utilization": session.current_tokens / session.max_tokens,
            "role_distribution": role_counts,
            "tool_usage_count": session.tool_usage_count,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "duration_seconds": (session.last_activity - session.created_at).total_seconds(),
        }

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get existing session or create new one."""
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id=session_id)
        return session

    def _determine_priority(
        self,
        role: MessageRole,
        tool_name: Optional[str],
    ) -> MessagePriority:
        """Determine message priority based on role and context."""
        canonical_tool_name = canonicalize_core_tool_name(tool_name) if tool_name else None

        if role == MessageRole.SYSTEM:
            return MessagePriority.CRITICAL

        if role == MessageRole.USER:
            return MessagePriority.HIGH

        if role == MessageRole.ASSISTANT:
            return MessagePriority.HIGH

        if role == MessageRole.TOOL:
            # File contents and search results are valuable context
            if canonical_tool_name in ("read", "code_search", "ls"):
                return MessagePriority.HIGH
            return MessagePriority.MEDIUM

        if role == MessageRole.TOOL_CALL:
            return MessagePriority.MEDIUM

        return MessagePriority.MEDIUM

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count from content.

        Uses the fast native token counter when available and falls back
        to word-based estimation.
        """
        from victor.processing.native.tokenizer import count_tokens_fast

        return count_tokens_fast(content)

    def _build_trace_metadata(
        self,
        role: MessageRole,
        content: str,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        tool_calls: Optional[List],
    ) -> Dict[str, Any]:
        """Build persisted metadata for semantic and execution retrieval traces."""
        existing = dict(metadata or {})
        semantic_text = str(existing.get("memory_semantic_text") or content)
        trace_kind = (
            "execution"
            if self._is_execution_trace_message(
                role,
                tool_name,
                tool_call_id,
                tool_calls,
            )
            else "semantic"
        )
        trace_metadata: Dict[str, Any] = {
            "memory_trace_kind": trace_kind,
            "memory_semantic_text": semantic_text,
        }
        if trace_kind == "execution":
            trace_metadata["memory_execution_text"] = str(
                existing.get("memory_execution_text")
                or self._build_execution_trace_text(
                    role=role,
                    content=content,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    tool_calls=tool_calls,
                )
            )
        return trace_metadata

    def _is_execution_trace_message(
        self,
        role: MessageRole,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        tool_calls: Optional[List],
    ) -> bool:
        """Return whether a message should participate in execution-trace retrieval."""
        return (
            role in (MessageRole.TOOL, MessageRole.TOOL_CALL)
            or bool(tool_name)
            or bool(tool_call_id)
            or bool(tool_calls)
        )

    def _build_execution_trace_text(
        self,
        role: MessageRole,
        content: str,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        tool_calls: Optional[List],
    ) -> str:
        """Create a compact lexical representation for execution-oriented recall."""
        parts: List[str] = []
        canonical_tool_name = canonicalize_core_tool_name(tool_name) if tool_name else None
        extracted_tool_name = self._extract_trace_attribute(content, "tool")
        effective_tool_name = canonical_tool_name or extracted_tool_name

        if effective_tool_name:
            parts.append(f"tool {effective_tool_name}")
        else:
            role_value = role.value if hasattr(role, "value") else str(role)
            parts.append(role_value.replace("_", " "))

        if tool_call_id:
            parts.append(tool_call_id)

        for attr_name in ("path", "query", "pattern", "symbol", "file", "name"):
            attr_value = self._extract_trace_attribute(content, attr_name)
            if attr_value:
                parts.append(f"{attr_name} {attr_value}")

        if tool_calls:
            parts.append(str(tool_calls))

        body_text = re.sub(r"<[^>]+>", " ", content)
        body_text = re.sub(r"\s+", " ", body_text).strip()
        if body_text:
            parts.append(body_text[:500])

        trace_text = " ".join(part for part in parts if part).strip()
        return trace_text or content[:500]

    @staticmethod
    def _extract_trace_attribute(content: str, attribute_name: str) -> Optional[str]:
        """Extract an XML-style attribute value from stored tool-output markup."""
        match = re.search(rf'{attribute_name}="([^"]+)"', content)
        if not match:
            return None
        return match.group(1).strip() or None

    @staticmethod
    def _trace_tokens(text: str) -> List[str]:
        """Tokenize trace text for lightweight overlap scoring."""
        return re.findall(r"[a-z0-9_]+", text.lower())

    def get_message_trace_kind(self, message: ConversationMessage) -> str:
        """Return the retrieval trace kind for a message."""
        metadata = getattr(message, "metadata", {}) or {}
        trace_kind = metadata.get("memory_trace_kind")
        if trace_kind in {"semantic", "execution"}:
            return trace_kind
        role = message.role if isinstance(message.role, MessageRole) else MessageRole(message.role)
        if self._is_execution_trace_message(
            role,
            getattr(message, "tool_name", None),
            getattr(message, "tool_call_id", None),
            getattr(message, "tool_calls", None),
        ):
            return "execution"
        return "semantic"

    def get_message_trace_text(
        self,
        message: ConversationMessage,
        trace_kind: Optional[str] = None,
    ) -> str:
        """Return the retrieval text for the requested trace kind."""
        metadata = getattr(message, "metadata", {}) or {}
        resolved_kind = trace_kind or self.get_message_trace_kind(message)
        if resolved_kind == "execution":
            existing = metadata.get("memory_execution_text")
            if existing:
                return str(existing)
            role = (
                message.role if isinstance(message.role, MessageRole) else MessageRole(message.role)
            )
            return self._build_execution_trace_text(
                role=role,
                content=message.content,
                tool_name=getattr(message, "tool_name", None),
                tool_call_id=getattr(message, "tool_call_id", None),
                tool_calls=getattr(message, "tool_calls", None),
            )
        return str(metadata.get("memory_semantic_text") or message.content)

    def _score_execution_trace(
        self,
        query: str,
        trace_text: str,
    ) -> float:
        """Score execution traces by lexical overlap with the current request."""
        query_tokens = set(self._trace_tokens(query))
        trace_tokens = set(self._trace_tokens(trace_text))
        if not query_tokens or not trace_tokens:
            return 0.0

        overlap = query_tokens & trace_tokens
        if not overlap:
            return 0.0

        score = len(overlap) / len(query_tokens)
        if query.lower() in trace_text.lower():
            score += 0.15
        return min(score, 1.0)

    def get_dual_trace_relevant_messages(
        self,
        session_id: str,
        query: str,
        semantic_limit: int = 5,
        execution_limit: int = 3,
        min_similarity: float = 0.3,
    ) -> Dict[str, List[Tuple[ConversationMessage, float]]]:
        """Retrieve semantic and execution traces in separate relevance buckets."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Cannot call get_dual_trace_relevant_messages from async context. "
                "Use await aget_dual_trace_relevant_messages(...) instead."
            )

        return run_sync(
            self.aget_dual_trace_relevant_messages(
                session_id=session_id,
                query=query,
                semantic_limit=semantic_limit,
                execution_limit=execution_limit,
                min_similarity=min_similarity,
            )
        )

    async def aget_dual_trace_relevant_messages(
        self,
        session_id: str,
        query: str,
        semantic_limit: int = 5,
        execution_limit: int = 3,
        min_similarity: float = 0.3,
    ) -> Dict[str, List[Tuple[ConversationMessage, float]]]:
        """Async dual-trace retrieval with semantic and execution lanes."""
        semantic_results = await self.aget_semantically_relevant_messages(
            session_id=session_id,
            query=query,
            limit=max(semantic_limit, 1),
            min_similarity=min_similarity,
            exclude_recent=0,
        )
        semantic_bucket = [
            (message, score)
            for message, score in semantic_results
            if self.get_message_trace_kind(message) == "semantic"
        ][:semantic_limit]

        execution_bucket = await asyncio.to_thread(
            self._get_execution_trace_matches,
            session_id,
            query,
            execution_limit,
        )

        return {
            "semantic": semantic_bucket,
            "execution": execution_bucket,
        }

    def _get_execution_trace_matches(
        self,
        session_id: str,
        query: str,
        limit: int,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve execution traces using a lightweight lexical scorer."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                """,
                (session_id,),
            ).fetchall()

        scored_messages: List[Tuple[ConversationMessage, float]] = []
        for row in rows:
            message = self._message_from_row(row)
            if self.get_message_trace_kind(message) != "execution":
                continue

            trace_text = self.get_message_trace_text(message, "execution")
            score = self._score_execution_trace(query, trace_text)
            if score <= 0:
                continue
            scored_messages.append((message, score))

        scored_messages.sort(key=lambda item: item[1], reverse=True)
        return scored_messages[:limit]

    def _score_messages(
        self,
        messages: List[ConversationMessage],
    ) -> List[tuple[ConversationMessage, float]]:
        """Score messages for context selection.

        Delegates to canonical score_messages() with STORE_WEIGHTS
        (priority 40% + recency 60%). Includes Rust-accelerated batch scoring.
        """
        if not messages:
            return []

        from victor.agent.conversation.scoring import score_messages, STORE_WEIGHTS
        from victor.agent.conversation.types import (
            ConversationMessage as CanonicalMessage,
        )

        # Convert store messages to canonical type for scoring
        canonical_msgs = [
            CanonicalMessage(
                role=msg.role.value,
                content=msg.content,
                id=msg.id,
                timestamp=msg.timestamp,
                token_count=msg.token_count,
                priority=msg.priority,
                metadata=msg.metadata,
                tool_name=msg.tool_name,
                tool_call_id=msg.tool_call_id,
            )
            for msg in messages
        ]

        scored = score_messages(canonical_msgs, weights=STORE_WEIGHTS)

        # Map back to original store messages
        canonical_to_store = {id(cm): msg for cm, msg in zip(canonical_msgs, messages)}
        return [(canonical_to_store[id(cm)], s) for cm, s in scored]

    def _prune_context(self, session: ConversationSession):
        """Prune conversation context to fit within token limits.

        Strategy:
        1. Keep all CRITICAL priority messages
        2. Score remaining messages
        3. Keep highest scoring messages within budget
        4. Delete pruned messages from SQLite to prevent unbounded growth
        """
        target_tokens = int((session.max_tokens - session.reserved_tokens) * 0.8)

        logger.info(
            f"Pruning session {session.session_id}. "
            f"Current: {session.current_tokens}, Target: {target_tokens}"
        )

        # Separate by priority
        critical = [m for m in session.messages if m.priority == MessagePriority.CRITICAL]
        others = [m for m in session.messages if m.priority != MessagePriority.CRITICAL]

        # Sort others by score
        scored_others = self._score_messages(others)

        # Select within budget
        kept = list(critical)
        kept_ids = {m.id for m in kept}
        current_tokens = sum(m.token_count for m in kept)

        for msg, _ in scored_others:
            if current_tokens + msg.token_count <= target_tokens:
                kept.append(msg)
                kept_ids.add(msg.id)
                current_tokens += msg.token_count

        # Identify pruned messages and delete from SQLite
        pruned_ids = [m.id for m in session.messages if m.id not in kept_ids]
        pruned_count = len(pruned_ids)

        if pruned_ids:
            self._delete_messages_from_db(session.session_id, pruned_ids)

        # Update session
        session.messages = sorted(kept, key=lambda m: m.timestamp)
        session.current_tokens = current_tokens

        logger.info(
            f"Pruned {pruned_count} messages (deleted from DB). "
            f"Remaining: {len(session.messages)}, Tokens: {current_tokens}"
        )

    def _delete_messages_from_db(self, session_id: str, message_ids: List[str]) -> None:
        """Delete pruned messages from SQLite in batches.

        Args:
            session_id: Session identifier (for logging)
            message_ids: List of message IDs to delete
        """
        batch_size = 500
        try:
            with sqlite3.connect(self.db_path) as conn:
                for i in range(0, len(message_ids), batch_size):
                    batch = message_ids[i : i + batch_size]
                    placeholders = ",".join("?" * len(batch))
                    conn.execute(
                        f"DELETE FROM messages WHERE id IN ({placeholders})",
                        batch,
                    )
            logger.debug(
                f"Deleted {len(message_ids)} pruned messages from DB " f"for session {session_id}"
            )
        except sqlite3.Error as e:
            logger.warning(f"Failed to delete pruned messages from DB: {e}")

    def _persist_session(self, session: ConversationSession):
        """Persist session to database using normalized FK columns."""
        with sqlite3.connect(self.db_path) as conn:
            # Get or create provider ID
            provider_id = self._get_or_create_provider_id(conn, session.provider)
            session_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }

            # Get FK IDs from cached lookups
            model_family_id = self._get_model_family_id(session.model_family)
            model_size_id = self._get_model_size_id(session.model_size)
            context_size_id = self._get_context_size_id(session.context_size)

            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, created_at, last_activity, project_path,
                 model, profile, max_tokens, reserved_tokens, metadata,
                 provider_id, model_family_id, model_size_id, context_size_id,
                 model_params_b, context_tokens, tool_capable, is_moe, is_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    session.project_path,
                    session.model,
                    session.profile,
                    session.max_tokens,
                    session.reserved_tokens,
                    json_dumps(
                        {
                            "active_files": session.active_files,
                            "tool_usage_count": session.tool_usage_count,
                        }
                    ),
                    # Normalized FK columns
                    provider_id,
                    model_family_id,
                    model_size_id,
                    context_size_id,
                    # Numeric columns
                    session.model_params_b,
                    session.context_tokens,
                    1 if session.tool_capable else 0,
                    1 if session.is_moe else 0,
                    1 if session.is_reasoning else 0,
                ),
            )
            if "provider" in session_columns:
                conn.execute(
                    "UPDATE sessions SET provider = ? WHERE session_id = ?",
                    (session.provider, session.session_id),
                )

    # Max chars for tool output content stored in SQLite.
    # Full content is only needed during the active session (in-memory).
    # Historical records keep a truncated version to save space.
    _TOOL_OUTPUT_STORE_LIMIT = 8000

    def _sanitize_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata dict for JSON serialization.

        Converts non-JSON-serializable objects to safe string representations.
        This handles edge cases like Ellipsis, Path objects, etc.

        Args:
            metadata: Raw metadata dict

        Returns:
            Sanitized metadata dict that is JSON-serializable
        """
        if not metadata:
            return {}

        def _sanitize_value(value: Any) -> Any:
            """Recursively sanitize a value for JSON serialization."""
            # Handle Ellipsis (...)
            if value is ...:
                return "<ellipsis>"
            # Handle Path objects
            elif hasattr(value, "__fspath__"):  # Path-like objects
                return str(value)
            # Handle types that can't be JSON serialized
            elif isinstance(value, type(lambda: None)):  # Functions/lambdas
                return f"<function: {getattr(value, '__name__', 'lambda')}>"
            # Handle objects with __dict__
            elif hasattr(value, "__dict__") and not isinstance(
                value, (str, bytes, dict, list, tuple, int, float, bool)
            ):
                return f"<object: {value.__class__.__name__}>"
            # Recursively handle lists and dicts
            elif isinstance(value, dict):
                return {k: _sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [_sanitize_value(v) for v in value]
            else:
                return value

        try:
            # Try to serialize as-is first
            from victor.core.json_utils import json_dumps

            json_dumps(metadata)
            return metadata
        except (TypeError, ValueError):
            # If that fails, sanitize the metadata
            logger.debug(f"Sanitizing metadata for JSON serialization: {list(metadata.keys())}")
            return {k: _sanitize_value(v) for k, v in metadata.items()}

    def _persist_message(self, session_id: str, message: ConversationMessage):
        """Persist message to database.

        Tool output messages (tool_call, tool_result, and user-role
        TOOL_OUTPUT blocks) are truncated to _TOOL_OUTPUT_STORE_LIMIT
        chars when stored, since full content is only needed during the
        active session where it lives in memory.
        """
        content = message.content
        # Truncate large tool outputs for storage
        if len(content) > self._TOOL_OUTPUT_STORE_LIMIT and (
            message.role in (MessageRole.TOOL_CALL, MessageRole.TOOL)
            or (message.role == MessageRole.USER and content.startswith("<TOOL_OUTPUT"))
        ):
            content = (
                content[: self._TOOL_OUTPUT_STORE_LIMIT]
                + f"\n\n[... truncated from {len(message.content)} chars "
                f"for storage]"
            )

        # Merge tool_calls into metadata for persistence so assistant
        # messages that requested tool calls can be fully reconstructed.
        meta = dict(message.metadata) if message.metadata else {}
        if message.tool_calls:
            meta["tool_calls"] = message.tool_calls

        # Sanitize metadata for JSON serialization (handles Ellipsis, Path objects, etc.)
        meta = self._sanitize_metadata_for_json(meta)

        # Use helper method for connection with proper timeout and WAL mode
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO messages
                (id, session_id, role, content, timestamp, token_count,
                 priority, tool_name, tool_call_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    session_id,
                    message.role.value,
                    content,
                    message.timestamp.isoformat(),
                    message.token_count,
                    message.priority.value,
                    message.tool_name,
                    message.tool_call_id,
                    json_dumps(meta),
                ),
            )

    def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET last_activity = ? " "WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )

    def update_session_token_usage(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd_micros: int = 0,
    ) -> None:
        """Persist cumulative API token usage for a session.

        Accumulates (+=) into the existing row so repeated calls are additive.
        Safe to call after every turn — uses INSERT OR IGNORE + UPDATE pattern
        to handle sessions not yet persisted to DB.

        Args:
            session_id: Session to update
            prompt_tokens: Actual prompt tokens from API response
            completion_tokens: Actual completion tokens
            cached_tokens: Cache-hit tokens (reduces effective cost)
            reasoning_tokens: Extended reasoning tokens (if applicable)
            cost_usd_micros: Cost in USD micros (1e-6 USD)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE sessions
                    SET prompt_tokens     = COALESCE(prompt_tokens, 0)     + ?,
                        completion_tokens = COALESCE(completion_tokens, 0) + ?,
                        cached_tokens     = COALESCE(cached_tokens, 0)     + ?,
                        reasoning_tokens  = COALESCE(reasoning_tokens, 0)  + ?,
                        cost_usd_micros   = COALESCE(cost_usd_micros, 0)   + ?,
                        last_activity     = ?
                    WHERE session_id = ?
                    """,
                    (
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens,
                        reasoning_tokens,
                        cost_usd_micros,
                        __import__("datetime").datetime.now().isoformat(),
                        session_id,
                    ),
                )
        except Exception as e:
            logger.debug("update_session_token_usage skipped: %s", e)

    def update_message_token_count(
        self,
        session_id: str,
        message_id: str,
        actual_token_count: int,
    ) -> None:
        """Update an individual message's token count with actual API value.

        Replaces the estimated token count set during add_message() with the
        actual token count from the API response. This enables accurate per-
        message token tracking for analytics and cost estimation.

        Args:
            session_id: Session containing the message
            message_id: Message to update
            actual_token_count: Actual token count from API response
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE messages
                    SET token_count = ?
                    WHERE session_id = ? AND message_id = ?
                    """,
                    (actual_token_count, session_id, message_id),
                )
                conn.commit()

            # Also update in-memory session if loaded
            session = self._sessions.get(session_id)
            if session:
                for msg in session.messages:
                    if msg.id == message_id:
                        # Adjust session total: remove estimate, add actual
                        session.current_tokens -= msg.token_count
                        msg.token_count = actual_token_count
                        session.current_tokens += actual_token_count
                        break
        except Exception as e:
            logger.debug("update_message_token_count skipped: %s", e)

    def get_session_token_stats(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive token statistics for a session.

        Returns token usage, cost estimates, cache hit rate, and reasoning
        token usage from the sessions table.

        Args:
            session_id: Session to query

        Returns:
            Dict with keys: prompt_tokens, completion_tokens, cached_tokens,
            reasoning_tokens, total_tokens, cache_hit_rate, cost_usd_micros,
            or None if session not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens, cost_usd_micros
                    FROM sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                prompt_tokens = row["prompt_tokens"] or 0
                completion_tokens = row["completion_tokens"] or 0
                cached_tokens = row["cached_tokens"] or 0
                reasoning_tokens = row["reasoning_tokens"] or 0
                cost_usd_micros = row["cost_usd_micros"] or 0

                total_tokens = prompt_tokens + completion_tokens
                cache_hit_rate = (cached_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0.0

                return {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                    "cache_hit_rate": round(cache_hit_rate, 2),
                    "cost_usd_micros": cost_usd_micros,
                    "cost_usd": cost_usd_micros / 1_000_000,
                }
        except Exception as e:
            logger.debug("get_session_token_stats error: %s", e)
            return None

    def get_total_token_usage(
        self,
        hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get aggregate token usage across all sessions.

        Useful for monitoring overall usage and cost estimation.

        Args:
            hours: Only include sessions from the last N hours. None = all time.

        Returns:
            Dict with aggregate statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if hours is not None:
                    cursor = conn.execute(
                        """
                        SELECT
                            COUNT(*) as session_count,
                            COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                            COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                            COALESCE(SUM(cached_tokens), 0) as total_cached_tokens,
                            COALESCE(SUM(reasoning_tokens), 0) as total_reasoning_tokens,
                            COALESCE(SUM(cost_usd_micros), 0) as total_cost_usd_micros
                        FROM sessions
                        WHERE datetime(last_activity) >= datetime('now', '-' || ? || ' hours')
                        """,
                        (hours,),
                    )
                else:
                    cursor = conn.execute("""
                        SELECT
                            COUNT(*) as session_count,
                            COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                            COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                            COALESCE(SUM(cached_tokens), 0) as total_cached_tokens,
                            COALESCE(SUM(reasoning_tokens), 0) as total_reasoning_tokens,
                            COALESCE(SUM(cost_usd_micros), 0) as total_cost_usd_micros
                        FROM sessions
                        """)

                row = cursor.fetchone()
                total_prompt = row["total_prompt_tokens"] or 0
                total_completion = row["total_completion_tokens"] or 0
                total_cached = row["total_cached_tokens"] or 0

                return {
                    "session_count": row["session_count"] or 0,
                    "total_prompt_tokens": total_prompt,
                    "total_completion_tokens": total_completion,
                    "total_cached_tokens": total_cached,
                    "total_reasoning_tokens": row["total_reasoning_tokens"] or 0,
                    "total_tokens": total_prompt + total_completion,
                    "total_cost_usd_micros": row["total_cost_usd_micros"] or 0,
                    "total_cost_usd": (row["total_cost_usd_micros"] or 0) / 1_000_000,
                    "cache_hit_rate": (
                        round(total_cached / total_prompt * 100, 2) if total_prompt > 0 else 0.0
                    ),
                }
        except Exception as e:
            logger.debug("get_total_token_usage error: %s", e)
            return {
                "session_count": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cached_tokens": 0,
                "total_reasoning_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd_micros": 0,
                "total_cost_usd": 0.0,
                "cache_hit_rate": 0.0,
            }

    # -----------------------------------------------------------------
    # Async variants — offload blocking SQLite I/O to thread pool
    # Use these from async contexts to avoid blocking the event loop.
    # -----------------------------------------------------------------

    async def add_message_async(
        self,
        session_id: str,
        role: "MessageRole",
        content: str,
        priority: Optional["MessagePriority"] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List] = None,
    ) -> "ConversationMessage":
        """Async variant of add_message.

        In-memory work runs on the calling coroutine. Blocking
        SQLite I/O is offloaded to the default thread executor.
        """
        import asyncio

        # Call shared implementation
        message = self._add_message_impl(
            session_id, role, content, priority, tool_name, tool_call_id, metadata, tool_calls
        )

        # Persist (async SQLite I/O - offloaded to thread pool)
        await asyncio.to_thread(self._persist_message, session_id, message)
        await asyncio.to_thread(self._update_session_activity, session_id)

        session = self._sessions.get(session_id)
        total_tokens = session.current_tokens if session else 0
        logger.debug(
            "Added %s message to %s (async). " "Tokens: %d, Total: %d",
            role.value,
            session_id,
            message.token_count,
            total_tokens,
        )
        return message

    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from database with JOINs to lookup tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # JOIN with lookup tables to get enum names
            row = conn.execute(
                """
                SELECT
                    s.*,
                    p.name AS provider_name,
                    mf.name AS model_family_name,
                    ms.name AS model_size_name,
                    cs.name AS context_size_name
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                LEFT JOIN model_families mf ON s.model_family_id = mf.id
                LEFT JOIN model_sizes ms ON s.model_size_id = ms.id
                LEFT JOIN context_sizes cs ON s.context_size_id = cs.id
                WHERE s.session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if not row:
                return None

            session = self._session_from_row(row)

            # Load messages
            messages = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY timestamp
                """,
                (session_id,),
            ).fetchall()

            for msg_row in messages:
                session.messages.append(self._message_from_row(msg_row))

            session.current_tokens = sum(m.token_count for m in session.messages)

            return session

    def _session_from_row(self, row: sqlite3.Row) -> ConversationSession:
        """Create session from database row with normalized FK lookups."""
        metadata = json_loads(row["metadata"] or "{}")
        row_keys = row.keys()

        # Get provider from joined table or NULL
        provider = row["provider_name"] if "provider_name" in row_keys else None

        # Parse model family from joined lookup table
        model_family = None
        family_name = row["model_family_name"] if "model_family_name" in row_keys else None
        if family_name:
            try:
                model_family = ModelFamily(family_name)
            except ValueError:
                model_family = ModelFamily.UNKNOWN

        # Parse model size from joined lookup table
        model_size = None
        size_name = row["model_size_name"] if "model_size_name" in row_keys else None
        if size_name:
            try:
                model_size = ModelSize(size_name)
            except ValueError:
                model_size = ModelSize.MEDIUM

        # Parse context size from joined lookup table
        context_size = None
        ctx_name = row["context_size_name"] if "context_size_name" in row_keys else None
        if ctx_name:
            try:
                context_size = ContextSize(ctx_name)
            except ValueError:
                context_size = ContextSize.MEDIUM

        return ConversationSession(
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activity=datetime.fromisoformat(row["last_activity"]),
            project_path=row["project_path"],
            provider=provider,
            model=row["model"],
            max_tokens=row["max_tokens"],
            reserved_tokens=row["reserved_tokens"],
            active_files=metadata.get("active_files", []),
            tool_usage_count=metadata.get("tool_usage_count", 0),
            profile=row["profile"] if "profile" in row_keys else None,
            model_family=model_family,
            model_size=model_size,
            model_params_b=(row["model_params_b"] if "model_params_b" in row_keys else None),
            context_size=context_size,
            context_tokens=(row["context_tokens"] if "context_tokens" in row_keys else None),
            tool_capable=(bool(row["tool_capable"]) if "tool_capable" in row_keys else False),
            is_moe=bool(row["is_moe"]) if "is_moe" in row_keys else False,
            is_reasoning=(bool(row["is_reasoning"]) if "is_reasoning" in row_keys else False),
        )

    def _message_from_row(self, row: sqlite3.Row) -> ConversationMessage:
        """Create message from database row."""
        return ConversationMessage(
            id=row["id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            token_count=row["token_count"],
            priority=MessagePriority(row["priority"]),
            tool_name=row["tool_name"],
            tool_call_id=row["tool_call_id"],
            metadata=json_loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID."""
        return f"session_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _generate_message_id() -> str:
        """Generate unique message ID."""
        return f"msg_{uuid.uuid4().hex[:12]}"

    # =========================================================================
    # SEMANTIC RETRIEVAL METHODS
    # For enhanced context compaction using embeddings
    # =========================================================================

    def set_embedding_service(self, service: "EmbeddingService") -> None:
        """Set the embedding service for semantic operations.

        Args:
            service: EmbeddingService instance for computing embeddings
        """
        self._embedding_service = service

    def set_embedding_store(self, store: "ConversationEmbeddingStore") -> None:
        """Set the LanceDB embedding store for efficient vector search.

        When set, message embeddings are synced to LanceDB on add_message(),
        and semantic retrieval uses LanceDB vector search instead of
        on-the-fly embedding computation.

        Args:
            store: ConversationEmbeddingStore instance
        """
        self._embedding_store = store
        logger.info("ConversationStore: LanceDB embedding store configured")

    def get_semantically_relevant_messages(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_recent: int = 5,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve messages semantically relevant to a query from the full history.

        This enables enhanced context compaction by finding relevant historical
        messages that may have been pruned from in-memory context.

        Uses LanceDB vector search when available (O(log n)), falling back to
        on-the-fly embedding computation (O(n)) if not configured.

        Args:
            session_id: Session to search in
            query: Query text to find similar messages for
            limit: Maximum messages to return
            min_similarity: Minimum cosine similarity threshold (0-1)
            exclude_recent: Skip N most recent messages (already in context)

        Returns:
            List of (message, similarity_score) tuples sorted by similarity
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Cannot call get_semantically_relevant_messages from async context. "
                "Use await aget_semantically_relevant_messages(...) instead."
            )

        return run_sync(
            self.aget_semantically_relevant_messages(
                session_id=session_id,
                query=query,
                limit=limit,
                min_similarity=min_similarity,
                exclude_recent=exclude_recent,
            )
        )

    async def aget_semantically_relevant_messages(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_recent: int = 5,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Async variant of semantic message retrieval for runtime code paths."""
        # Try LanceDB vector search first (much faster)
        if hasattr(self, "_embedding_store") and self._embedding_store is not None:
            return await self._aget_relevant_messages_via_lancedb(
                session_id, query, limit, min_similarity, exclude_recent
            )

        # Fall back to on-the-fly embedding (slower but works without LanceDB)
        return await asyncio.to_thread(
            self._get_relevant_messages_via_embedding,
            session_id,
            query,
            limit,
            min_similarity,
            exclude_recent,
        )

    def _get_relevant_messages_via_lancedb(
        self,
        session_id: str,
        query: str,
        limit: int,
        min_similarity: float,
        exclude_recent: int,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve relevant messages using LanceDB vector search.

        Much faster than on-the-fly embedding - O(log n) vs O(n).
        """
        return run_sync(
            self._aget_relevant_messages_via_lancedb(
                session_id=session_id,
                query=query,
                limit=limit,
                min_similarity=min_similarity,
                exclude_recent=exclude_recent,
            )
        )

    def _get_excluded_message_ids(self, session_id: str, exclude_recent: int) -> List[str]:
        """Fetch recent message IDs that should be excluded from semantic search."""
        # Get recent message IDs to exclude
        exclude_ids: List[str] = []
        if exclude_recent > 0:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                recent_rows = conn.execute(
                    """
                    SELECT id FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, exclude_recent),
                ).fetchall()
                exclude_ids = [row["id"] for row in recent_rows]
        return exclude_ids

    async def _aget_relevant_messages_via_lancedb(
        self,
        session_id: str,
        query: str,
        limit: int,
        min_similarity: float,
        exclude_recent: int,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve relevant messages using the async embedding store path."""
        exclude_ids = self._get_excluded_message_ids(session_id, exclude_recent)

        search_results = await self._embedding_store.search_similar(
            query=query,
            session_id=session_id,
            limit=limit,
            min_similarity=min_similarity,
            exclude_message_ids=exclude_ids,
        )

        # Fetch full messages from SQLite for the matching IDs
        if not search_results:
            return []

        result_ids = [r.message_id for r in search_results]
        similarity_map = {r.message_id: r.similarity for r in search_results}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(result_ids))
            rows = conn.execute(
                f"""
                SELECT * FROM messages
                WHERE id IN ({placeholders})
                """,
                result_ids,
            ).fetchall()

        # Build result list with similarity scores
        results: List[Tuple[ConversationMessage, float]] = []
        for row in rows:
            message = self._message_from_row(row)
            similarity = similarity_map.get(message.id, 0.0)
            results.append((message, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _get_relevant_messages_via_embedding(
        self,
        session_id: str,
        query: str,
        limit: int,
        min_similarity: float,
        exclude_recent: int,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve relevant messages using on-the-fly embedding computation.

        Fallback when LanceDB is not available. Slower but works without setup.
        """
        if not hasattr(self, "_embedding_service") or self._embedding_service is None:
            logger.warning("No embedding service set for semantic retrieval")
            return []

        try:
            # Load all messages from session (excluding recent ones already in context)
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    """,
                    (session_id,),
                ).fetchall()

            if not rows or len(rows) <= exclude_recent:
                return []

            # Skip recent messages (already in active context)
            historical_rows = rows[exclude_recent:]

            # Get query embedding
            query_embedding = self._embedding_service.embed_text_sync(query[:2000])

            # Compute embeddings and similarities for historical messages
            scored_messages: List[Tuple[ConversationMessage, float]] = []

            for row in historical_rows:
                message = self._message_from_row(row)

                # Skip very short messages
                if len(message.content) < 20:
                    continue

                # Compute similarity
                msg_embedding = self._embedding_service.embed_text_sync(message.content[:2000])
                similarity = self._cosine_similarity(
                    query_embedding.tolist(), msg_embedding.tolist()
                )

                if similarity >= min_similarity:
                    scored_messages.append((message, similarity))

            # Sort by similarity (descending) and limit
            scored_messages.sort(key=lambda x: x[1], reverse=True)
            return scored_messages[:limit]

        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")
            return []

    def get_relevant_summaries(
        self,
        session_id: str,
        query: str,
        limit: int = 3,
        min_similarity: float = 0.25,
    ) -> List[Tuple[str, float]]:
        """Retrieve context summaries relevant to current query.

        Searches the context_summaries table for summaries that might
        contain relevant historical context.

        Args:
            session_id: Session to search in
            query: Query text to find relevant summaries for
            limit: Maximum summaries to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (summary_text, similarity_score) tuples
        """
        if not hasattr(self, "_embedding_service") or self._embedding_service is None:
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT summary, created_at FROM context_summaries
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 20
                    """,
                    (session_id,),
                ).fetchall()

            if not rows:
                return []

            query_embedding = self._embedding_service.embed(query[:2000])
            scored_summaries: List[Tuple[str, float]] = []

            for row in rows:
                summary = row["summary"]
                if len(summary) < 10:
                    continue

                summary_embedding = self._embedding_service.embed(summary[:2000])
                similarity = self._cosine_similarity(query_embedding, summary_embedding)

                if similarity >= min_similarity:
                    scored_summaries.append((summary, similarity))

            scored_summaries.sort(key=lambda x: x[1], reverse=True)
            return scored_summaries[:limit]

        except Exception as e:
            logger.warning(f"Summary retrieval failed: {e}")
            return []

    def store_compaction_summary(
        self,
        session_id: str,
        summary: str,
        messages_summarized: List[str],
    ) -> None:
        """Store a compaction summary for later retrieval.

        When context is compacted, store a summary of what was removed
        so it can be retrieved later if needed.

        Args:
            session_id: Session the summary belongs to
            summary: Summary text describing compacted content
            messages_summarized: List of message IDs that were summarized
        """
        if not summary or len(summary) < 5:
            return

        token_count = len(summary) // self.chars_per_token

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO context_summaries
                (id, session_id, summary, token_count, messages_summarized, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"sum_{uuid.uuid4().hex[:12]}",
                    session_id,
                    summary,
                    token_count,
                    json_dumps(messages_summarized),
                    datetime.now().isoformat(),
                ),
            )

        logger.debug(f"Stored compaction summary for session {session_id}")

    def store_compaction_summary_enhanced(
        self,
        session_id: str,
        summary_xml: str,
        summary_text: str,
        summary_json: Dict[str, Any],
        messages_summarized: List[str],
        strategy_used: str,
        complexity_score: float,
        tokens_saved: int,
        duration_ms: int,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> str:
        """Store enhanced compaction summary with dual formats.

        Stores both XML (machine-readable) and natural language formats,
        along with metadata for analytics and monitoring.

        Args:
            session_id: Session the summary belongs to
            summary_xml: Machine-readable XML format
            summary_text: Natural language format
            summary_json: Structured summary data as dict
            messages_summarized: List of message IDs that were summarized
            strategy_used: 'rule_based', 'llm_based', or 'hybrid'
            complexity_score: 0.0-1.0 complexity score
            tokens_saved: Estimated tokens saved
            duration_ms: Compaction duration in milliseconds
            llm_provider: LLM provider used (if applicable)
            llm_model: LLM model used (if applicable)
            success: Whether compaction succeeded
            error_message: Error message if failed

        Returns:
            Summary ID
        """
        summary_id = f"sum_{uuid.uuid4().hex[:12]}"

        with sqlite3.connect(self.db_path) as conn:
            # Determine format
            if summary_xml and summary_text:
                summary_format = "both"
            elif summary_xml:
                summary_format = "xml"
            else:
                summary_format = "natural"

            # Store enhanced summary
            conn.execute(
                """
                INSERT INTO context_summaries (
                    id, session_id, summary_format, summary_xml, summary_text,
                    summary_json, messages_summarized, messages_summarized_json,
                    strategy_used, complexity_score, estimated_tokens_saved,
                    token_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary_id,
                    session_id,
                    summary_format,
                    summary_xml,
                    summary_text,
                    json_dumps(summary_json),
                    json_dumps(messages_summarized),
                    json_dumps(messages_summarized),
                    strategy_used,
                    complexity_score,
                    tokens_saved,
                    len(summary_xml or summary_text) // self.chars_per_token,
                    datetime.now().isoformat(),
                ),
            )

            # Log to compaction_history for analytics
            history_id = f"hist_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """
                INSERT INTO compaction_history (
                    id, session_id, strategy_used, message_count_before,
                    message_count_after, token_count_before, token_count_after,
                    duration_ms, llm_provider, llm_model, success, error_message,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    history_id,
                    session_id,
                    strategy_used,
                    len(messages_summarized),  # message_count_before
                    0,  # message_count_after (not tracked here)
                    tokens_saved
                    + len(summary_xml or summary_text)
                    // self.chars_per_token,  # token_count_before (estimated)
                    len(summary_xml or summary_text) // self.chars_per_token,  # token_count_after
                    duration_ms,
                    llm_provider,
                    llm_model,
                    success,
                    error_message,
                    datetime.now().isoformat(),
                ),
            )

        logger.debug(
            f"Stored enhanced compaction summary for session {session_id} "
            f"(strategy={strategy_used}, tokens_saved={tokens_saved})"
        )

        return summary_id

    def get_compaction_summaries(
        self,
        session_id: str,
        format_preference: str = "both",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve compaction summaries with format preference.

        Args:
            session_id: Session to retrieve summaries for
            format_preference: 'xml', 'natural', or 'both'
            limit: Maximum number of summaries to retrieve

        Returns:
            List of summary dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            if format_preference == "xml":
                cursor = conn.execute(
                    """
                    SELECT id, summary_xml as summary, strategy_used,
                           complexity_score, estimated_tokens_saved, created_at
                    FROM context_summaries
                    WHERE session_id = ? AND summary_xml IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            elif format_preference == "natural":
                cursor = conn.execute(
                    """
                    SELECT id, summary_text as summary, strategy_used,
                           complexity_score, estimated_tokens_saved, created_at
                    FROM context_summaries
                    WHERE session_id = ? AND summary_text IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            else:  # both
                cursor = conn.execute(
                    """
                    SELECT id, summary_xml, summary_text, strategy_used,
                           complexity_score, estimated_tokens_saved, created_at
                    FROM context_summaries
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )

            summaries = []
            for row in cursor.fetchall():
                summary = {
                    "id": row[0],
                    "summary_xml": row[1] if format_preference != "natural" else None,
                    "summary_text": row[2] if format_preference == "natural" else row[2],
                    "strategy_used": row[3],
                    "complexity_score": row[4],
                    "estimated_tokens_saved": row[5],
                    "created_at": row[6],
                }
                if format_preference == "both":
                    summary["summary_xml"] = row[1]
                    summary["summary_text"] = row[2]
                    summary["strategy_used"] = row[3]
                summaries.append(summary)

            return summaries

    def get_compaction_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve compaction history for analytics.

        Args:
            session_id: Session to retrieve history for
            limit: Maximum number of events to retrieve

        Returns:
            List of compaction event dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, session_id, strategy_used, message_count_before,
                       message_count_after, token_count_before, token_count_after,
                       duration_ms, llm_provider, llm_model, success, error_message,
                       created_at
                FROM compaction_history
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            )

            history = []
            for row in cursor.fetchall():
                history.append(
                    {
                        "id": row[0],
                        "session_id": row[1],
                        "strategy_used": row[2],
                        "message_count_before": row[3],
                        "message_count_after": row[4],
                        "token_count_before": row[5],
                        "token_count_after": row[6],
                        "duration_ms": row[7],
                        "llm_provider": row[8],
                        "llm_model": row[9],
                        "success": bool(row[10]),
                        "error_message": row[11],
                        "created_at": row[12],
                    }
                )

            return history

    def store_compaction_history(
        self,
        session_id: str,
        strategy_used: str,
        message_count_before: int,
        message_count_after: int,
        token_count_before: int,
        token_count_after: int,
        duration_ms: int,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> str:
        """Store compaction event to history table for analytics.

        Args:
            session_id: Session ID
            strategy_used: Compaction strategy used (rule_based, llm_based, hybrid)
            message_count_before: Message count before compaction
            message_count_after: Message count after compaction
            token_count_before: Estimated token count before compaction
            token_count_after: Token count after compaction
            duration_ms: Duration of compaction in milliseconds
            llm_provider: LLM provider used (if applicable)
            llm_model: LLM model used (if applicable)
            success: Whether compaction succeeded
            error_message: Error message if failed

        Returns:
            History event ID
        """
        history_id = f"hist_{uuid.uuid4().hex[:12]}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO compaction_history (
                    id, session_id, strategy_used, message_count_before,
                    message_count_after, token_count_before, token_count_after,
                    duration_ms, llm_provider, llm_model, success, error_message,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    history_id,
                    session_id,
                    strategy_used,
                    message_count_before,
                    message_count_after,
                    token_count_before,
                    token_count_after,
                    duration_ms,
                    llm_provider,
                    llm_model,
                    success,
                    error_message,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        logger.debug(
            f"Stored compaction history event {history_id} for session {session_id} "
            f"(strategy={strategy_used}, duration_ms={duration_ms}, success={success})"
        )

        return history_id

    def get_compaction_statistics(
        self,
        session_id: Optional[str] = None,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Get aggregate compaction statistics.

        Args:
            session_id: Optional session ID to filter by (None = all sessions)
            limit: Maximum number of events to analyze

        Returns:
            Dictionary with aggregate statistics:
            - total_compactions: Total number of compaction events
            - success_rate: Overall success rate (0.0-1.0)
            - avg_duration_ms: Average duration in milliseconds
            - avg_tokens_saved: Average tokens saved
            - strategy_counts: Count per strategy
            - strategy_success_rates: Success rate per strategy
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build query with optional session filter
            session_filter = "WHERE session_id = ?" if session_id else ""
            session_params = (session_id,) if session_id else ()

            # Get total statistics
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_compactions,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_ms) as avg_duration_ms,
                    AVG(token_count_before - token_count_after) as avg_tokens_saved
                FROM compaction_history
                {session_filter}
                LIMIT ?
                """,
                session_params + (limit,),
            ).fetchone()

            # Get counts per strategy
            strategy_rows = conn.execute(
                f"""
                SELECT
                    strategy_used,
                    COUNT(*) as count,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_ms) as avg_duration_ms,
                    AVG(token_count_before - token_count_after) as avg_tokens_saved
                FROM compaction_history
                {session_filter}
                GROUP BY strategy_used
                LIMIT ?
                """,
                session_params + (limit,),
            ).fetchall()

            strategy_counts = {}
            strategy_success_rates = {}
            strategy_durations = {}
            strategy_savings = {}

            for row in strategy_rows:
                strategy = row["strategy_used"]
                strategy_counts[strategy] = row["count"]
                strategy_success_rates[strategy] = row["success_rate"]
                strategy_durations[strategy] = row["avg_duration_ms"]
                strategy_savings[strategy] = row["avg_tokens_saved"]

            return {
                "total_compactions": row["total_compactions"] if row else 0,
                "success_rate": row["success_rate"] if row else 0.0,
                "avg_duration_ms": row["avg_duration_ms"] if row else 0.0,
                "avg_tokens_saved": row["avg_tokens_saved"] if row else 0.0,
                "strategy_counts": strategy_counts,
                "strategy_success_rates": strategy_success_rates,
                "strategy_durations": strategy_durations,
                "strategy_savings": strategy_savings,
            }

    def get_compaction_performance_trends(
        self,
        session_id: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get time-series performance trends for compaction.

        Args:
            session_id: Optional session ID to filter by (None = all sessions)
            hours_back: Number of hours to look back (default: 24)
            limit: Maximum number of events to return

        Returns:
            List of compaction events with timestamp, ordered by time
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Calculate cutoff time
            from datetime import timedelta

            cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()

            # Build query with optional session filter
            session_filter = "AND session_id = ?" if session_id else ""
            session_params = (session_id,) if session_id else ()

            rows = conn.execute(
                f"""
                SELECT
                    id, session_id, strategy_used, message_count_before,
                    message_count_after, token_count_before, token_count_after,
                    duration_ms, success, error_message, created_at
                FROM compaction_history
                WHERE created_at >= ?
                {session_filter}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (cutoff_time,) + session_params + (limit,),
            ).fetchall()

            trends = []
            for row in rows:
                trends.append(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "strategy_used": row["strategy_used"],
                        "message_count_before": row["message_count_before"],
                        "message_count_after": row["message_count_after"],
                        "token_count_before": row["token_count_before"],
                        "token_count_after": row["token_count_after"],
                        "tokens_saved": row["token_count_before"] - row["token_count_after"],
                        "duration_ms": row["duration_ms"],
                        "success": bool(row["success"]),
                        "error_message": row["error_message"],
                        "created_at": row["created_at"],
                    }
                )

            return trends

    def get_strategy_comparison(
        self,
        limit: int = 1000,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare performance across compaction strategies.

        Args:
            limit: Maximum number of events to analyze per strategy

        Returns:
            Dictionary mapping strategy names to their performance metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT
                    strategy_used,
                    COUNT(*) as total_compactions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_compactions,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_ms) as avg_duration_ms,
                    MIN(duration_ms) as min_duration_ms,
                    MAX(duration_ms) as max_duration_ms,
                    AVG(token_count_before - token_count_after) as avg_tokens_saved,
                    MIN(token_count_before - token_count_after) as min_tokens_saved,
                    MAX(token_count_before - token_count_after) as max_tokens_saved
                FROM compaction_history
                GROUP BY strategy_used
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            comparison = {}
            for row in rows:
                strategy = row["strategy_used"]
                comparison[strategy] = {
                    "total_compactions": row["total_compactions"],
                    "successful_compactions": row["successful_compactions"],
                    "success_rate": row["success_rate"],
                    "avg_duration_ms": row["avg_duration_ms"],
                    "min_duration_ms": row["min_duration_ms"],
                    "max_duration_ms": row["max_duration_ms"],
                    "avg_tokens_saved": row["avg_tokens_saved"],
                    "min_tokens_saved": row["min_tokens_saved"],
                    "max_tokens_saved": row["max_tokens_saved"],
                }

            return comparison

    def get_historical_tool_results(
        self,
        session_id: str,
        tool_names: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[ConversationMessage]:
        """Retrieve historical tool results from the session.

        Useful for finding relevant previous tool outputs that might
        inform the current task.

        Args:
            session_id: Session to search in
            tool_names: Optional list of tool names to filter by
            limit: Maximum results to return

        Returns:
            List of tool result messages
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if tool_names:
                placeholders = ",".join("?" * len(tool_names))
                rows = conn.execute(
                    f"""
                    SELECT * FROM messages
                    WHERE session_id = ?
                    AND role = 'tool'
                    AND tool_name IN ({placeholders})
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, *tool_names, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ?
                    AND role = 'tool'
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()

            return [self._message_from_row(row) for row in rows]

    def search_messages_fts(
        self,
        session_id: str,
        query: str,
        limit: int = 20,
        roles: Optional[List[str]] = None,
    ) -> List[ConversationMessage]:
        """Search messages using FTS5 full-text search.

        Uses SQLite FTS5 for O(log n) keyword search instead of O(n) linear scan.
        This is ~20x faster than linear content matching for large conversations.

        Args:
            session_id: Session to search in
            query: Search query (supports FTS5 syntax: AND, OR, NOT, phrases)
            limit: Maximum results to return
            roles: Optional list of roles to filter by (user, assistant, tool_result)

        Returns:
            List of matching messages ordered by relevance (BM25 rank)

        Example:
            # Simple keyword search
            results = memory.search_messages_fts(session_id, "authentication error")

            # FTS5 phrase search
            results = memory.search_messages_fts(session_id, '"login failed"')

            # FTS5 boolean query
            results = memory.search_messages_fts(session_id, "error AND NOT warning")
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build the query based on role filtering
            if roles:
                placeholders = ",".join("?" * len(roles))
                sql = f"""
                    SELECT m.*, bm25(messages_fts) AS rank
                    FROM messages_fts fts
                    JOIN messages m ON fts.message_id = m.id
                    WHERE fts.session_id = ?
                    AND messages_fts MATCH ?
                    AND m.role IN ({placeholders})
                    ORDER BY rank
                    LIMIT ?
                """
                params = (session_id, query, *roles, limit)
            else:
                sql = """
                    SELECT m.*, bm25(messages_fts) AS rank
                    FROM messages_fts fts
                    JOIN messages m ON fts.message_id = m.id
                    WHERE fts.session_id = ?
                    AND messages_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """
                params = (session_id, query, limit)

            try:
                rows = conn.execute(sql, params).fetchall()
                return [self._message_from_row(row) for row in rows]
            except sqlite3.OperationalError as e:
                # FTS5 may not be available in all SQLite builds
                logger.warning(f"FTS5 search failed, falling back to LIKE: {e}")
                return self._fallback_content_search(conn, session_id, query, limit, roles)

    def _fallback_content_search(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        query: str,
        limit: int,
        roles: Optional[List[str]] = None,
    ) -> List[ConversationMessage]:
        """Fallback content search using LIKE when FTS5 is unavailable."""
        # Simple tokenization for LIKE search
        terms = query.split()
        like_patterns = [f"%{term}%" for term in terms]

        if roles:
            role_placeholders = ",".join("?" * len(roles))
            where_clause = f"session_id = ? AND role IN ({role_placeholders})"
            params = [session_id, *roles]
        else:
            where_clause = "session_id = ?"
            params = [session_id]

        # Build AND conditions for each term
        for pattern in like_patterns:
            where_clause += " AND content LIKE ?"
            params.append(pattern)

        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT * FROM messages
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [self._message_from_row(row) for row in rows]

    def rebuild_fts_index(self, session_id: Optional[str] = None) -> int:
        """Rebuild FTS5 index from messages table.

        Useful after schema migrations or to fix corrupted indexes.

        Args:
            session_id: Optional session to rebuild (all if None)

        Returns:
            Number of messages indexed
        """
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing FTS content
            if session_id:
                conn.execute(
                    """
                    INSERT INTO messages_fts(messages_fts, rowid, content, message_id, session_id)
                    SELECT 'delete', rowid, content, id, session_id
                    FROM messages WHERE session_id = ?
                    """,
                    (session_id,),
                )
                # Rebuild for specific session
                cursor = conn.execute(
                    """
                    INSERT INTO messages_fts(rowid, content, message_id, session_id)
                    SELECT rowid, content, id, session_id
                    FROM messages WHERE session_id = ?
                    """,
                    (session_id,),
                )
            else:
                # Full rebuild - delete all then reinsert
                conn.execute("DELETE FROM messages_fts")
                cursor = conn.execute("""
                    INSERT INTO messages_fts(rowid, content, message_id, session_id)
                    SELECT rowid, content, id, session_id FROM messages
                    """)

            count = cursor.rowcount
            conn.commit()
            logger.info(f"Rebuilt FTS5 index: {count} messages indexed")
            return count

    # =========================================================================
    # ML/RL AGGREGATION METHODS
    # Efficient queries using normalized FK columns for training data extraction
    # =========================================================================

    def get_provider_stats(self) -> List[Dict[str, Any]]:
        """Get session statistics grouped by provider.

        Returns aggregated stats for ML/RL analysis of provider performance.
        Uses normalized provider_id FK for efficient GROUP BY.

        Returns:
            List of dicts with provider, session_count, total_messages,
            avg_messages_per_session, tool_capable_pct
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    p.name AS provider,
                    COUNT(DISTINCT s.session_id) AS session_count,
                    COUNT(m.id) AS total_messages,
                    ROUND(COUNT(m.id) * 1.0 / COUNT(DISTINCT s.session_id), 1)
                        AS avg_messages_per_session,
                    ROUND(AVG(s.tool_capable) * 100, 1)
                        AS tool_capable_pct
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE p.name IS NOT NULL
                GROUP BY p.name
                ORDER BY session_count DESC
                """).fetchall()

            return [dict(row) for row in rows]

    def get_model_family_stats(self) -> List[Dict[str, Any]]:
        """Get session statistics grouped by model family.

        Returns aggregated stats for ML/RL analysis of model architecture performance.
        Uses normalized model_family_id FK for efficient GROUP BY.

        Returns:
            List of dicts with model_family, session_count, total_messages,
            avg_params_b, tool_capable_pct, moe_pct, reasoning_pct
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    mf.name AS model_family,
                    COUNT(DISTINCT s.session_id) AS session_count,
                    COUNT(m.id) AS total_messages,
                    ROUND(AVG(s.model_params_b), 1) AS avg_params_b,
                    ROUND(AVG(s.tool_capable) * 100, 1) AS tool_capable_pct,
                    ROUND(AVG(s.is_moe) * 100, 1) AS moe_pct,
                    ROUND(AVG(s.is_reasoning) * 100, 1) AS reasoning_pct
                FROM sessions s
                LEFT JOIN model_families mf ON s.model_family_id = mf.id
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE mf.name IS NOT NULL
                GROUP BY mf.name
                ORDER BY session_count DESC
                """).fetchall()

            return [dict(row) for row in rows]

    def get_model_size_stats(self) -> List[Dict[str, Any]]:
        """Get session statistics grouped by model size category.

        Returns aggregated stats for ML/RL analysis of model size impact.
        Uses normalized model_size_id FK for efficient GROUP BY.

        Returns:
            List of dicts with model_size, session_count, total_messages,
            avg_context_tokens, tool_capable_pct
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    ms.name AS model_size,
                    COUNT(DISTINCT s.session_id) AS session_count,
                    COUNT(m.id) AS total_messages,
                    ROUND(AVG(s.context_tokens), 0) AS avg_context_tokens,
                    ROUND(AVG(s.tool_capable) * 100, 1) AS tool_capable_pct
                FROM sessions s
                LEFT JOIN model_sizes ms ON s.model_size_id = ms.id
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE ms.name IS NOT NULL
                GROUP BY ms.name
                ORDER BY
                    CASE ms.name
                        WHEN 'tiny' THEN 1
                        WHEN 'small' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'large' THEN 4
                        WHEN 'xlarge' THEN 5
                        WHEN 'xxlarge' THEN 6
                    END
                """).fetchall()

            return [dict(row) for row in rows]

    def get_rl_training_data(
        self,
        limit: int = 1000,
        min_messages: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get comprehensive session data for RL training.

        Returns fully denormalized session data with all ML-friendly features
        for training reinforcement learning models on conversation quality.

        Args:
            limit: Maximum sessions to return
            min_messages: Minimum messages per session to include

        Returns:
            List of dicts with all normalized columns plus computed metrics:
            - provider, model, profile
            - model_family, model_size, context_size
            - model_params_b, context_tokens
            - tool_capable, is_moe, is_reasoning
            - message_count, user_messages, assistant_messages, tool_messages
            - session_duration_minutes
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.model,
                    s.profile,
                    p.name AS provider,
                    mf.name AS model_family,
                    ms.name AS model_size,
                    cs.name AS context_size,
                    s.model_params_b,
                    s.context_tokens,
                    s.tool_capable,
                    s.is_moe,
                    s.is_reasoning,
                    COUNT(m.id) AS message_count,
                    SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS user_messages,
                    SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
                    SUM(CASE WHEN m.role IN ('tool_call', 'tool')
                        THEN 1 ELSE 0 END) AS tool_messages,
                    ROUND(
                        (JULIANDAY(MAX(m.timestamp)) - JULIANDAY(MIN(m.timestamp))) * 24 * 60,
                        1
                    ) AS session_duration_minutes
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                LEFT JOIN model_families mf ON s.model_family_id = mf.id
                LEFT JOIN model_sizes ms ON s.model_size_id = ms.id
                LEFT JOIN context_sizes cs ON s.context_size_id = cs.id
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                HAVING COUNT(m.id) >= ?
                ORDER BY s.last_activity DESC
                LIMIT ?
                """,
                (min_messages, limit),
            ).fetchall()

            return [dict(row) for row in rows]

    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get high-level aggregation summary for ML/RL dashboard.

        Returns:
            Dict with counts and breakdowns for quick overview
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total counts
            session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

            # Provider breakdown
            provider_counts = conn.execute("""
                SELECT p.name, COUNT(*) AS count
                FROM sessions s
                JOIN providers p ON s.provider_id = p.id
                GROUP BY p.name
                ORDER BY count DESC
                LIMIT 5
                """).fetchall()

            # Model family breakdown
            family_counts = conn.execute("""
                SELECT mf.name, COUNT(*) AS count
                FROM sessions s
                JOIN model_families mf ON s.model_family_id = mf.id
                GROUP BY mf.name
                ORDER BY count DESC
                LIMIT 5
                """).fetchall()

            # Tool capability stats
            tool_stats = conn.execute("""
                SELECT
                    SUM(CASE WHEN tool_capable = 1 THEN 1 ELSE 0 END) AS tool_capable,
                    SUM(CASE WHEN tool_capable = 0 THEN 1 ELSE 0 END) AS not_tool_capable
                FROM sessions
                """).fetchone()

            return {
                "total_sessions": session_count,
                "total_messages": message_count,
                "top_providers": [(row[0], row[1]) for row in provider_counts],
                "top_families": [(row[0], row[1]) for row in family_counts],
                "tool_capable_sessions": tool_stats[0] or 0,
                "not_tool_capable_sessions": tool_stats[1] or 0,
            }

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Uses Rust-accelerated implementation with NumPy fallback.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            from victor.processing.native import cosine_similarity

            return cosine_similarity(vec1, vec2)
        except Exception as e:
            logger.debug("Cosine similarity computation failed, returning 0.0: %s", e)
        return 0.0


# Convenience function to get global store
_global_store: Optional[ConversationStore] = None


def get_conversation_manager() -> ConversationStore:
    """Get the global conversation store.

    Note: Function name kept for backward compatibility.
    """
    global _global_store
    if _global_store is None:
        _global_store = ConversationStore()
    return _global_store
