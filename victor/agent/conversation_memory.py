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
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import json
import logging
import sqlite3
import uuid

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message roles in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class MessagePriority(Enum):
    """Priority levels for context pruning.

    Higher values indicate higher priority (kept longer).
    """

    CRITICAL = 100  # System prompts, current task
    HIGH = 75  # Recent tool results, code context
    MEDIUM = 50  # Previous exchanges
    LOW = 25  # Old context, summaries
    EPHEMERAL = 0  # Can be dropped immediately


class ModelFamily(Enum):
    """Model architecture family for ML/RL feature extraction."""

    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    CLAUDE = "claude"
    GPT = "gpt"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    PHI = "phi"
    CODELLAMA = "codellama"
    COMMAND = "command"  # Cohere
    GROK = "grok"
    UNKNOWN = "unknown"


class ModelSize(Enum):
    """Model parameter size category for coarse comparison."""

    TINY = "tiny"  # <1B params
    SMALL = "small"  # 1-8B
    MEDIUM = "medium"  # 8-32B
    LARGE = "large"  # 32-70B
    XLARGE = "xlarge"  # 70-175B
    XXLARGE = "xxlarge"  # >175B


class ContextSize(Enum):
    """Context window size category."""

    SMALL = "small"  # <8K tokens
    MEDIUM = "medium"  # 8K-32K
    LARGE = "large"  # 32K-128K
    XLARGE = "xlarge"  # 128K+


# =============================================================================
# MODEL METADATA PARSER
# Extracts ML-friendly features from model name strings
# =============================================================================

import re  # noqa: E402
from dataclasses import dataclass as _dataclass  # noqa: E402


@_dataclass
class ModelMetadata:
    """Parsed model metadata for ML/RL feature extraction."""

    model_family: ModelFamily
    model_size: ModelSize
    model_params_b: Optional[float]  # Parameters in billions
    context_size: ContextSize
    context_tokens: Optional[int]  # Actual context window
    is_moe: bool  # Mixture of Experts
    is_reasoning: bool  # Explicit reasoning model


# Model family detection patterns (order matters - more specific first)
_MODEL_FAMILY_PATTERNS = [
    (r"codellama|code-llama", ModelFamily.CODELLAMA),
    (r"deepseek[-_]?r1|deepseek[-_]?coder", ModelFamily.DEEPSEEK),
    (r"deepseek", ModelFamily.DEEPSEEK),
    (r"mixtral", ModelFamily.MIXTRAL),
    (r"mistral", ModelFamily.MISTRAL),
    (r"llama[-_]?3\.3|llama3\.3|llama[-_]?3\.1|llama3[-_]?[12]?", ModelFamily.LLAMA),
    (r"llama", ModelFamily.LLAMA),
    (r"qwen[-_]?2\.5|qwen2\.5|qwen[-_]?3|qwen3", ModelFamily.QWEN),
    (r"qwen", ModelFamily.QWEN),
    (r"claude[-_]?3|claude[-_]?opus|claude[-_]?sonnet|claude[-_]?haiku", ModelFamily.CLAUDE),
    (r"claude", ModelFamily.CLAUDE),
    (r"gpt[-_]?4|gpt[-_]?3\.5|chatgpt|openai", ModelFamily.GPT),
    (r"gemini|palm|bard", ModelFamily.GEMINI),
    (r"phi[-_]?[234]|phi[-_]?mini", ModelFamily.PHI),
    (r"command[-_]?r|cohere", ModelFamily.COMMAND),
    (r"grok", ModelFamily.GROK),
]

# MoE model patterns
_MOE_PATTERNS = [r"mixtral", r"8x7b", r"8x22b", r"moe", r"mixture"]

# MoE effective parameters (total active params)
_MOE_EFFECTIVE_PARAMS = {
    "8x7b": 46.7,  # Mixtral 8x7B
    "8x22b": 141.0,  # Mixtral 8x22B
}

# Reasoning model patterns
_REASONING_PATTERNS = [r"deepseek[-_]?r1", r"o1[-_]?", r"r1[-_]?", r"reasoning"]

# Context size patterns (extract from model name like "32k", "128k")
_CONTEXT_PATTERN = re.compile(r"(\d+)k", re.IGNORECASE)

# Parameter size patterns (extract from model name like "70b", "8b", "7b")
_PARAM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*b(?:illion)?", re.IGNORECASE)
_PARAM_PATTERN_ALT = re.compile(r"[-_:](\d+(?:\.\d+)?)b(?!yte)", re.IGNORECASE)


def parse_model_metadata(
    model: str,
    provider: Optional[str] = None,
    known_context: Optional[int] = None,
    known_params_b: Optional[float] = None,
) -> ModelMetadata:
    """Parse model name to extract ML-friendly metadata.

    Args:
        model: Model name string (e.g., "llama-3.3-70b-versatile")
        provider: Optional provider name for disambiguation
        known_context: Optional known context window size
        known_params_b: Optional known parameter count in billions

    Returns:
        ModelMetadata with extracted features

    Examples:
        >>> parse_model_metadata("llama-3.3-70b-versatile")
        ModelMetadata(family=LLAMA, size=LARGE, params_b=70.0, ...)

        >>> parse_model_metadata("mixtral-8x7b-32768")
        ModelMetadata(family=MIXTRAL, size=MEDIUM, is_moe=True, ...)

        >>> parse_model_metadata("deepseek-r1:32b")
        ModelMetadata(family=DEEPSEEK, size=MEDIUM, is_reasoning=True, ...)
    """
    model_lower = model.lower()

    # 1. Detect model family
    model_family = ModelFamily.UNKNOWN
    for pattern, family in _MODEL_FAMILY_PATTERNS:
        if re.search(pattern, model_lower):
            model_family = family
            break

    # Provider-based fallback
    if model_family == ModelFamily.UNKNOWN and provider:
        provider_lower = provider.lower()
        if "anthropic" in provider_lower or "claude" in provider_lower:
            model_family = ModelFamily.CLAUDE
        elif "openai" in provider_lower:
            model_family = ModelFamily.GPT
        elif "google" in provider_lower or "gemini" in provider_lower:
            model_family = ModelFamily.GEMINI
        elif "groq" in provider_lower:
            # Groq primarily serves Llama models
            model_family = ModelFamily.LLAMA
        elif "xai" in provider_lower or "grok" in provider_lower:
            model_family = ModelFamily.GROK

    # 2. Extract parameter count
    params_b = known_params_b
    if params_b is None:
        # Check for MoE patterns first (special handling)
        for moe_pattern, moe_params in _MOE_EFFECTIVE_PARAMS.items():
            if moe_pattern in model_lower:
                params_b = moe_params
                break

        # Try various patterns for non-MoE
        if params_b is None:
            match = _PARAM_PATTERN.search(model_lower)
            if not match:
                match = _PARAM_PATTERN_ALT.search(model_lower)
            if match:
                params_b = float(match.group(1))

        # Special cases for well-known models
        if params_b is None:
            if "gpt-4" in model_lower:
                params_b = 175.0  # Estimated
            elif "gpt-3.5" in model_lower:
                params_b = 175.0  # Estimated
            elif "claude-3-opus" in model_lower:
                params_b = 200.0  # Estimated
            elif "claude-3-sonnet" in model_lower or "claude-3.5-sonnet" in model_lower:
                params_b = 70.0  # Estimated
            elif "claude-3-haiku" in model_lower:
                params_b = 20.0  # Estimated

    # 3. Categorize model size
    if params_b is not None:
        if params_b < 1:
            model_size = ModelSize.TINY
        elif params_b < 8:
            model_size = ModelSize.SMALL
        elif params_b < 32:
            model_size = ModelSize.MEDIUM
        elif params_b < 70:
            model_size = ModelSize.LARGE
        elif params_b < 175:
            model_size = ModelSize.XLARGE
        else:
            model_size = ModelSize.XXLARGE
    else:
        # Default based on model family
        model_size = ModelSize.MEDIUM  # Safe default

    # 4. Extract context window size
    context_tokens = known_context
    if context_tokens is None:
        match = _CONTEXT_PATTERN.search(model_lower)
        if match:
            context_tokens = int(match.group(1)) * 1024  # Convert K to actual tokens

        # Well-known context windows
        if context_tokens is None:
            if model_family == ModelFamily.CLAUDE:
                context_tokens = 200000
            elif model_family == ModelFamily.GPT:
                context_tokens = 128000 if "gpt-4" in model_lower else 16000
            elif model_family == ModelFamily.GEMINI:
                context_tokens = 1000000  # Gemini 1.5 Pro
            elif "128k" in model_lower or "128000" in model_lower:
                context_tokens = 128000

    # Categorize context size
    if context_tokens is not None:
        if context_tokens < 8000:
            context_size = ContextSize.SMALL
        elif context_tokens < 32000:
            context_size = ContextSize.MEDIUM
        elif context_tokens < 128000:
            context_size = ContextSize.LARGE
        else:
            context_size = ContextSize.XLARGE
    else:
        context_size = ContextSize.MEDIUM  # Safe default

    # 5. Detect MoE architecture
    is_moe = any(re.search(pattern, model_lower) for pattern in _MOE_PATTERNS)

    # 6. Detect reasoning models
    is_reasoning = any(re.search(pattern, model_lower) for pattern in _REASONING_PATTERNS)

    return ModelMetadata(
        model_family=model_family,
        model_size=model_size,
        model_params_b=params_b,
        context_size=context_size,
        context_tokens=context_tokens,
        is_moe=is_moe,
        is_reasoning=is_reasoning,
    )


# Known model context windows (for accuracy)
_KNOWN_CONTEXT_WINDOWS = {
    "llama-3.3-70b-versatile": 128000,
    "llama-3.1-8b-instant": 128000,
    "mixtral-8x7b-32768": 32768,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-3-haiku": 200000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    "gemini-pro": 32768,
    "gemini-1.5-pro": 1000000,
    "deepseek-chat": 128000,
    "deepseek-coder": 128000,
    "qwen2.5-coder:32b": 128000,
    "qwen3:32b": 40960,
}

# Known model parameters (for accuracy)
_KNOWN_MODEL_PARAMS = {
    "llama-3.3-70b-versatile": 70.0,
    "llama-3.1-8b-instant": 8.0,
    "mixtral-8x7b-32768": 46.7,  # MoE effective
    "deepseek-r1:32b": 32.0,
    "deepseek-r1:70b": 70.0,
    "qwen2.5-coder:32b": 32.0,
    "qwen3:32b": 32.0,
}


def get_known_model_context(model: str) -> Optional[int]:
    """Get known context window for a model.

    Args:
        model: Model name string

    Returns:
        Context window in tokens, or None if unknown
    """
    model_lower = model.lower()
    for known_model, context in _KNOWN_CONTEXT_WINDOWS.items():
        if known_model in model_lower or model_lower in known_model:
            return context
    return None


def get_known_model_params(model: str) -> Optional[float]:
    """Get known parameter count for a model.

    Args:
        model: Model name string

    Returns:
        Parameter count in billions, or None if unknown
    """
    model_lower = model.lower()
    for known_model, params in _KNOWN_MODEL_PARAMS.items():
        if known_model in model_lower or model_lower in known_model:
            return params
    return None


@dataclass
class ConversationMessage:
    """A single message in the conversation."""

    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    token_count: int
    priority: MessagePriority = MessagePriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For tool calls/results
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_provider_format(self) -> Dict[str, Any]:
        """Convert to provider message format."""
        base = {
            "role": (
                self.role.value
                if self.role in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM)
                else "assistant"
            ),
            "content": self.content,
        }

        # Handle tool-related messages
        if self.role == MessageRole.TOOL_RESULT and self.tool_call_id:
            base["tool_call_id"] = self.tool_call_id

        return base

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "priority": self.priority.value,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_count=data["token_count"],
            priority=MessagePriority(data["priority"]),
            tool_name=data.get("tool_name"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata", {}),
        )


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
    SCHEMA_VERSION = "0.2.0"  # Aligned with Victor 0.2.0 release

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_context_tokens: int = 100000,
        response_reserve: int = 4096,
        chars_per_token: int = 4,
    ):
        """Initialize the conversation memory manager.

        Args:
            db_path: Path to SQLite database. Defaults to {project}/.victor/conversation.db
            max_context_tokens: Maximum tokens in context window
            response_reserve: Tokens reserved for response
            chars_per_token: Approximate characters per token for estimation
        """
        from victor.config.settings import get_project_paths

        self.db_path = db_path or get_project_paths().conversation_db
        self.max_context_tokens = max_context_tokens
        self.response_reserve = response_reserve
        self.chars_per_token = chars_per_token

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

    def _init_database(self) -> None:
        """Initialize SQLite database for persistence with normalized schema.

        Schema v2 Design (Normalized for ML/RL):
        - Lookup tables with INTEGER PKs for categorical values
        - Sessions table uses INTEGER FKs for efficient joins
        - Indexes on all FK columns for fast aggregation queries
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMP NOT NULL
                )
                """
            )

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
        conn.executescript(
            """
            -- Model family lookup (llama, qwen, claude, gpt, etc.)
            CREATE TABLE IF NOT EXISTS model_families (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            );

            -- Model size lookup (tiny, small, medium, large, xlarge, xxlarge)
            CREATE TABLE IF NOT EXISTS model_sizes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                min_params_b REAL,
                max_params_b REAL
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
                is_reasoning INTEGER DEFAULT 0
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
            WHERE role IN ('tool_call', 'tool_result');

            -- Index for user-assistant exchanges
            CREATE INDEX IF NOT EXISTS idx_messages_exchange
            ON messages(session_id, role, timestamp)
            WHERE role IN ('user', 'assistant');
            """
        )

        # Populate lookup tables with enum values
        self._populate_lookup_tables(conn)

        # Load caches
        self._load_lookup_caches(conn)

    def _populate_lookup_tables(self, conn: sqlite3.Connection) -> None:
        """Populate lookup tables with predefined enum values."""
        # Model families
        families = [
            ("llama", "Meta Llama models"),
            ("qwen", "Alibaba Qwen models"),
            ("mistral", "Mistral AI dense models"),
            ("mixtral", "Mistral AI MoE models"),
            ("claude", "Anthropic Claude models"),
            ("gpt", "OpenAI GPT models"),
            ("gemini", "Google Gemini models"),
            ("deepseek", "DeepSeek models"),
            ("phi", "Microsoft Phi models"),
            ("codellama", "Meta Code Llama models"),
            ("command", "Cohere Command models"),
            ("grok", "xAI Grok models"),
            ("unknown", "Unknown model family"),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO model_families (name, description) VALUES (?, ?)",
            families,
        )

        # Model sizes with parameter ranges
        sizes = [
            ("tiny", 0, 1),  # <1B
            ("small", 1, 8),  # 1-8B
            ("medium", 8, 32),  # 8-32B
            ("large", 32, 70),  # 32-70B
            ("xlarge", 70, 175),  # 70-175B
            ("xxlarge", 175, 999999),  # >175B
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO model_sizes (name, min_params_b, max_params_b) VALUES (?, ?, ?)",
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
                provider_id = int(row[0])
                self._provider_ids[provider_lower] = provider_id
                return provider_id
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

        logger.info(
            f"Created session {session_id} for project: {project_path} "
            f"[{model_family.value if model_family else 'unknown'}/{model_size.value if model_size else 'unknown'}]"
        )

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

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        priority: Optional[MessagePriority] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
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

        Returns:
            Created ConversationMessage
        """
        session = self._get_or_create_session(session_id)

        # Auto-determine priority based on role
        if priority is None:
            priority = self._determine_priority(role, tool_name)

        # Estimate token count
        token_count = self._estimate_tokens(content)

        message = ConversationMessage(
            id=self._generate_message_id(),
            role=role,
            content=content,
            timestamp=datetime.now(),
            token_count=token_count,
            priority=priority,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
        )

        session.messages.append(message)
        session.current_tokens += token_count
        session.last_activity = datetime.now()

        # Track tool usage
        if role in (MessageRole.TOOL_CALL, MessageRole.TOOL_RESULT):
            session.tool_usage_count += 1

        # Check if pruning is needed
        if session.current_tokens > (session.max_tokens - session.reserved_tokens):
            self._prune_context(session)

        # Persist
        self._persist_message(session_id, message)
        self._update_session_activity(session_id)

        # NOTE: Lazy embedding - embeddings created on search, not on add
        # This reduces write overhead and file proliferation

        logger.debug(
            f"Added {role.value} message to {session_id}. "
            f"Tokens: {token_count}, Total: {session.current_tokens}"
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

    def clear_session(self, session_id: str) -> None:
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

    def delete_session(self, session_id: str) -> None:
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

        role_counts: Dict[str, int] = {}
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
    # ML/RL AGGREGATION METHODS
    # Efficient GROUP BY queries on normalized FK columns for learning
    # =========================================================================

    def get_model_family_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated statistics by model family for RL feature extraction.

        Returns:
            Dictionary mapping model family names to their stats:
            - session_count: Number of sessions
            - total_messages: Total messages across sessions
            - avg_tool_usage: Average tool calls per session
            - avg_context_tokens: Average context window used
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    mf.name AS family_name,
                    COUNT(s.session_id) AS session_count,
                    AVG(s.context_tokens) AS avg_context_tokens,
                    AVG(s.model_params_b) AS avg_params_b,
                    SUM(CASE WHEN s.tool_capable THEN 1 ELSE 0 END) AS tool_capable_count
                FROM sessions s
                LEFT JOIN model_families mf ON s.model_family_id = mf.id
                WHERE mf.name IS NOT NULL
                GROUP BY mf.id
                ORDER BY session_count DESC
                """
            ).fetchall()

            result = {}
            for row in rows:
                result[row["family_name"]] = {
                    "session_count": row["session_count"],
                    "avg_context_tokens": row["avg_context_tokens"],
                    "avg_params_b": row["avg_params_b"],
                    "tool_capable_count": row["tool_capable_count"],
                }
            return result

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated statistics by provider for RL feature extraction.

        Returns:
            Dictionary mapping provider names to their stats
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    p.name AS provider_name,
                    COUNT(s.session_id) AS session_count,
                    COUNT(DISTINCT s.model) AS unique_models,
                    SUM(CASE WHEN s.tool_capable THEN 1 ELSE 0 END) AS tool_capable_count,
                    SUM(CASE WHEN s.is_moe THEN 1 ELSE 0 END) AS moe_count
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                WHERE p.name IS NOT NULL
                GROUP BY p.id
                ORDER BY session_count DESC
                """
            ).fetchall()

            result = {}
            for row in rows:
                result[row["provider_name"]] = {
                    "session_count": row["session_count"],
                    "unique_models": row["unique_models"],
                    "tool_capable_count": row["tool_capable_count"],
                    "moe_count": row["moe_count"],
                }
            return result

    def get_model_size_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated statistics by model size for RL feature extraction.

        Returns:
            Dictionary mapping size categories to their stats
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    ms.name AS size_name,
                    COUNT(s.session_id) AS session_count,
                    AVG(s.model_params_b) AS avg_params_b,
                    AVG(s.context_tokens) AS avg_context_tokens,
                    SUM(CASE WHEN s.is_reasoning THEN 1 ELSE 0 END) AS reasoning_count
                FROM sessions s
                LEFT JOIN model_sizes ms ON s.model_size_id = ms.id
                WHERE ms.name IS NOT NULL
                GROUP BY ms.id
                ORDER BY ms.id
                """
            ).fetchall()

            result = {}
            for row in rows:
                result[row["size_name"]] = {
                    "session_count": row["session_count"],
                    "avg_params_b": row["avg_params_b"],
                    "avg_context_tokens": row["avg_context_tokens"],
                    "reasoning_count": row["reasoning_count"],
                }
            return result

    def get_rl_training_data(
        self,
        limit: int = 1000,
        filter_tool_capable: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Get session data formatted for RL training.

        Returns a list of feature vectors suitable for reinforcement learning.
        Uses INTEGER FK columns for efficient filtering.

        Args:
            limit: Maximum records to return
            filter_tool_capable: Optional filter for tool-capable models only

        Returns:
            List of dictionaries with normalized feature vectors
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            where_clause = ""
            params: List[Any] = [limit]

            if filter_tool_capable is not None:
                where_clause = "WHERE s.tool_capable = ?"
                params = [1 if filter_tool_capable else 0, limit]

            rows = conn.execute(
                f"""
                SELECT
                    s.session_id,
                    p.name AS provider,
                    mf.name AS model_family,
                    ms.name AS model_size,
                    cs.name AS context_size,
                    s.model_params_b,
                    s.context_tokens,
                    s.tool_capable,
                    s.is_moe,
                    s.is_reasoning,
                    -- FK IDs for efficient embedding lookup
                    s.provider_id,
                    s.model_family_id,
                    s.model_size_id,
                    s.context_size_id
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                LEFT JOIN model_families mf ON s.model_family_id = mf.id
                LEFT JOIN model_sizes ms ON s.model_size_id = ms.id
                LEFT JOIN context_sizes cs ON s.context_size_id = cs.id
                {where_clause}
                ORDER BY s.last_activity DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

            return [dict(row) for row in rows]

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
        if role == MessageRole.SYSTEM:
            return MessagePriority.CRITICAL

        if role == MessageRole.USER:
            return MessagePriority.HIGH

        if role == MessageRole.ASSISTANT:
            return MessagePriority.HIGH

        if role == MessageRole.TOOL_RESULT:
            # File contents and search results are valuable context
            if tool_name in ("read_file", "code_search", "list_directory"):
                return MessagePriority.HIGH
            return MessagePriority.MEDIUM

        if role == MessageRole.TOOL_CALL:
            return MessagePriority.MEDIUM

        return MessagePriority.MEDIUM

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count from content.

        Uses a simple character-based estimation.
        For production, consider using tiktoken or provider-specific tokenizers.
        """
        return len(content) // self.chars_per_token + 1

    def _score_messages(
        self,
        messages: List[ConversationMessage],
    ) -> List[tuple[ConversationMessage, float]]:
        """Score messages for context selection.

        Scoring factors:
        - Priority level (40%)
        - Recency (60%)
        """
        if not messages:
            return []

        now = datetime.now()
        max_age = max((now - msg.timestamp).total_seconds() for msg in messages) or 1

        scored = []
        for msg in messages:
            # Priority score (0-1)
            priority_score = msg.priority.value / 100

            # Recency score (0-1, more recent = higher)
            age = (now - msg.timestamp).total_seconds()
            recency_score = 1 - (age / max_age)

            # Combined score
            score = (priority_score * 0.4) + (recency_score * 0.6)

            scored.append((msg, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _prune_context(self, session: ConversationSession) -> None:
        """Prune conversation context to fit within token limits.

        Strategy:
        1. Keep all CRITICAL priority messages
        2. Score remaining messages
        3. Keep highest scoring messages within budget
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
        current_tokens = sum(m.token_count for m in kept)

        for msg, _ in scored_others:
            if current_tokens + msg.token_count <= target_tokens:
                kept.append(msg)
                current_tokens += msg.token_count

        # Calculate pruned count
        pruned_count = len(session.messages) - len(kept)

        # Update session
        session.messages = sorted(kept, key=lambda m: m.timestamp)
        session.current_tokens = current_tokens

        logger.info(
            f"Pruned {pruned_count} messages. "
            f"Remaining: {len(session.messages)}, Tokens: {current_tokens}"
        )

    def _persist_session(self, session: ConversationSession) -> None:
        """Persist session to database using normalized FK columns."""
        with sqlite3.connect(self.db_path) as conn:
            # Get or create provider ID
            provider_id = self._get_or_create_provider_id(conn, session.provider)

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
                    json.dumps(
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

    def _persist_message(self, session_id: str, message: ConversationMessage) -> None:
        """Persist message to database."""
        with sqlite3.connect(self.db_path) as conn:
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
                    message.content,
                    message.timestamp.isoformat(),
                    message.token_count,
                    message.priority.value,
                    message.tool_name,
                    message.tool_call_id,
                    json.dumps(message.metadata),
                ),
            )

    def _update_session_activity(self, session_id: str) -> None:
        """Update session last activity timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )

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
        metadata = json.loads(row["metadata"] or "{}")
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
            model_params_b=row["model_params_b"] if "model_params_b" in row_keys else None,
            context_size=context_size,
            context_tokens=row["context_tokens"] if "context_tokens" in row_keys else None,
            tool_capable=bool(row["tool_capable"]) if "tool_capable" in row_keys else False,
            is_moe=bool(row["is_moe"]) if "is_moe" in row_keys else False,
            is_reasoning=bool(row["is_reasoning"]) if "is_reasoning" in row_keys else False,
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
            metadata=json.loads(row["metadata"] or "{}"),
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
        # Try LanceDB vector search first (much faster)
        if hasattr(self, "_embedding_store") and self._embedding_store is not None:
            return self._get_relevant_messages_via_lancedb(
                session_id, query, limit, min_similarity, exclude_recent
            )

        # Fall back to on-the-fly embedding (slower but works without LanceDB)
        return self._get_relevant_messages_via_embedding(
            session_id, query, limit, min_similarity, exclude_recent
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
        import asyncio

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

        # Run async search
        async def _search() -> List[Any]:
            return await self._embedding_store.search_similar(
                query=query,
                session_id=session_id,
                limit=limit,
                min_similarity=min_similarity,
                exclude_message_ids=exclude_ids,
            )

        try:
            # Check if we're in an async context
            asyncio.get_running_loop()
            # If we are, raise an error to force caller to use async API
            raise RuntimeError(
                "Cannot call _get_relevant_messages_via_lancedb from async context. "
                "Use the async version of this method instead."
            )
        except RuntimeError as e:
            if "async context" in str(e):
                # Re-raise our custom error
                raise
            # No running loop, safe to use asyncio.run()
            search_results = asyncio.run(_search())

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

    async def get_relevant_summaries(
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

            query_embedding = await self._embedding_service.embed_text(query[:2000])
            scored_summaries: List[Tuple[str, float]] = []

            for row in rows:
                summary = row["summary"]
                if len(summary) < 10:
                    continue

                summary_embedding = await self._embedding_service.embed_text(summary[:2000])
                # Convert numpy arrays to lists for cosine_similarity
                similarity = self._cosine_similarity(
                    list(query_embedding) if hasattr(query_embedding, "tolist") else query_embedding,
                    list(summary_embedding) if hasattr(summary_embedding, "tolist") else summary_embedding,
                )

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
                    json.dumps(messages_summarized),
                    datetime.now().isoformat(),
                ),
            )

        logger.debug(f"Stored compaction summary for session {session_id}")

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
                    AND role = 'tool_result'
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
                    AND role = 'tool_result'
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
            params: List[Any] = [session_id, *roles]
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
                cursor = conn.execute(
                    """
                    INSERT INTO messages_fts(rowid, content, message_id, session_id)
                    SELECT rowid, content, id, session_id FROM messages
                    """
                )

            count = cursor.rowcount
            conn.commit()
            logger.info(f"Rebuilt FTS5 index: {count} messages indexed")
            return count

    # =========================================================================
    # ML/RL AGGREGATION METHODS
    # Efficient queries using normalized FK columns for training data extraction
    # =========================================================================

    def get_provider_stats_list(self) -> List[Dict[str, Any]]:
        """Get session statistics grouped by provider.

        Returns aggregated stats for ML/RL analysis of provider performance.
        Uses normalized provider_id FK for efficient GROUP BY.

        Returns:
            List of dicts with provider, session_count, total_messages,
            avg_messages_per_session, tool_capable_pct
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    p.name AS provider,
                    COUNT(DISTINCT s.session_id) AS session_count,
                    COUNT(m.id) AS total_messages,
                    ROUND(COUNT(m.id) * 1.0 / COUNT(DISTINCT s.session_id), 1) AS avg_messages_per_session,
                    ROUND(AVG(s.tool_capable) * 100, 1) AS tool_capable_pct
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE p.name IS NOT NULL
                GROUP BY p.name
                ORDER BY session_count DESC
                """
            ).fetchall()

            return [dict(row) for row in rows]

    def get_model_family_stats_list(self) -> List[Dict[str, Any]]:
        """Get session statistics grouped by model family.

        Returns aggregated stats for ML/RL analysis of model architecture performance.
        Uses normalized model_family_id FK for efficient GROUP BY.

        Returns:
            List of dicts with model_family, session_count, total_messages,
            avg_params_b, tool_capable_pct, moe_pct, reasoning_pct
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
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
                """
            ).fetchall()

            return [dict(row) for row in rows]

    def get_model_size_stats_list(self) -> List[Dict[str, Any]]:
        """Get session statistics grouped by model size category.

        Returns aggregated stats for ML/RL analysis of model size impact.
        Uses normalized model_size_id FK for efficient GROUP BY.

        Returns:
            List of dicts with model_size, session_count, total_messages,
            avg_context_tokens, tool_capable_pct
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
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
                """
            ).fetchall()

            return [dict(row) for row in rows]

    def get_rl_training_data_list(
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
                    SUM(CASE WHEN m.role IN ('tool_call', 'tool_result') THEN 1 ELSE 0 END) AS tool_messages,
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
            provider_counts = conn.execute(
                """
                SELECT p.name, COUNT(*) AS count
                FROM sessions s
                JOIN providers p ON s.provider_id = p.id
                GROUP BY p.name
                ORDER BY count DESC
                LIMIT 5
                """
            ).fetchall()

            # Model family breakdown
            family_counts = conn.execute(
                """
                SELECT mf.name, COUNT(*) AS count
                FROM sessions s
                JOIN model_families mf ON s.model_family_id = mf.id
                GROUP BY mf.name
                ORDER BY count DESC
                LIMIT 5
                """
            ).fetchall()

            # Tool capability stats
            tool_stats = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN tool_capable = 1 THEN 1 ELSE 0 END) AS tool_capable,
                    SUM(CASE WHEN tool_capable = 0 THEN 1 ELSE 0 END) AS not_tool_capable
                FROM sessions
                """
            ).fetchone()

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
        except Exception:
            pass
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
