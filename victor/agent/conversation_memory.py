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
    from victor.embeddings.service import EmbeddingService
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

    # Provider info
    provider: Optional[str] = None
    model: Optional[str] = None

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
        }


class ConversationStore:
    """
    SQLite-based conversation store with intelligent context pruning.

    Features:
    - SQLite persistence for session recovery
    - Token-aware context window management
    - Priority-based message pruning
    - Semantic relevance scoring for context selection

    For simpler in-memory storage, see MessageHistory.
    For JSON file-based persistence, see SessionPersistence.

    Usage:
        store = ConversationStore()
        session = store.create_session(project_path="/path/to/project")
        store.add_message(session.session_id, MessageRole.USER, "Hello")
        messages = store.get_context_messages(session.session_id)
    """

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

        # Initialize database
        self._init_database()

        logger.debug(
            f"ConversationStore initialized. "
            f"DB: {self.db_path}, Max tokens: {max_context_tokens}"
        )

    def _init_database(self) -> None:
        """Initialize SQLite database for persistence."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                -- Sessions table
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    last_activity TIMESTAMP NOT NULL,
                    project_path TEXT,
                    provider TEXT,
                    model TEXT,
                    max_tokens INTEGER DEFAULT 100000,
                    reserved_tokens INTEGER DEFAULT 4096,
                    metadata TEXT
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

                -- Indexes for efficient queries
                CREATE INDEX IF NOT EXISTS idx_messages_session_time
                ON messages(session_id, timestamp);

                CREATE INDEX IF NOT EXISTS idx_messages_priority
                ON messages(session_id, priority DESC);

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

                CREATE INDEX IF NOT EXISTS idx_summaries_session
                ON context_summaries(session_id, created_at DESC);
            """
            )

        logger.debug(f"Database initialized at {self.db_path}")

    def create_session(
        self,
        session_id: Optional[str] = None,
        project_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ConversationSession:
        """Create a new conversation session.

        Args:
            session_id: Optional session ID. Generated if not provided.
            project_path: Path to the project being worked on
            provider: LLM provider name
            model: Model identifier
            max_tokens: Override max context tokens

        Returns:
            New ConversationSession instance
        """
        if session_id is None:
            session_id = self._generate_session_id()

        session = ConversationSession(
            session_id=session_id,
            project_path=project_path,
            provider=provider,
            model=model,
            max_tokens=max_tokens or self.max_context_tokens,
            reserved_tokens=self.response_reserve,
        )

        self._sessions[session_id] = session
        self._persist_session(session)

        logger.info(f"Created session {session_id} for project: {project_path}")

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
        """List recent sessions.

        Args:
            project_path: Filter by project path
            limit: Maximum sessions to return

        Returns:
            List of sessions ordered by last activity
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if project_path:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions
                    WHERE project_path = ?
                    ORDER BY last_activity DESC
                    LIMIT ?
                    """,
                    (project_path, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions
                    ORDER BY last_activity DESC
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

        # Sync to LanceDB embedding store if configured
        self._sync_message_embedding(session_id, message)

        logger.debug(
            f"Added {role.value} message to {session_id}. "
            f"Tokens: {token_count}, Total: {session.current_tokens}"
        )

        return message

    def _sync_message_embedding(
        self,
        session_id: str,
        message: "ConversationMessage",
    ) -> None:
        """Sync a message embedding to LanceDB (async, fire-and-forget).

        Args:
            session_id: Session ID
            message: Message to sync
        """
        if not hasattr(self, "_embedding_store") or self._embedding_store is None:
            return

        # Skip very short messages
        if len(message.content) < 20:
            return

        import asyncio

        async def _sync():
            try:
                await self._embedding_store.add_message_embedding(
                    message_id=message.id,
                    session_id=session_id,
                    role=message.role.value,
                    content=message.content,
                    timestamp=message.timestamp,
                )
            except Exception as e:
                logger.warning(f"Failed to sync message embedding: {e}")

        # Run async embedding sync in background
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_sync())
        except RuntimeError:
            # No running loop, run synchronously
            asyncio.run(_sync())

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

        for msg, score in scored_messages:
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

    def _prune_context(self, session: ConversationSession):
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

    def _persist_session(self, session: ConversationSession):
        """Persist session to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, created_at, last_activity, project_path,
                 provider, model, max_tokens, reserved_tokens, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    session.project_path,
                    session.provider,
                    session.model,
                    session.max_tokens,
                    session.reserved_tokens,
                    json.dumps(
                        {
                            "active_files": session.active_files,
                            "tool_usage_count": session.tool_usage_count,
                        }
                    ),
                ),
            )

    def _persist_message(self, session_id: str, message: ConversationMessage):
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

    def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )

    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
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
        """Create session from database row."""
        metadata = json.loads(row["metadata"] or "{}")

        return ConversationSession(
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activity=datetime.fromisoformat(row["last_activity"]),
            project_path=row["project_path"],
            provider=row["provider"],
            model=row["model"],
            max_tokens=row["max_tokens"],
            reserved_tokens=row["reserved_tokens"],
            active_files=metadata.get("active_files", []),
            tool_usage_count=metadata.get("tool_usage_count", 0),
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
        async def _search():
            return await self._embedding_store.search_similar(
                query=query,
                session_id=session_id,
                limit=limit,
                min_similarity=min_similarity,
                exclude_message_ids=exclude_ids,
            )

        try:
            loop = asyncio.get_running_loop()
            future = asyncio.ensure_future(_search())
            search_results = loop.run_until_complete(future)
        except RuntimeError:
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

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            import numpy as np
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                return float(dot / (norm1 * norm2))
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
