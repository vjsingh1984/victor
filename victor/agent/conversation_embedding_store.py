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

"""LanceDB-based conversation embedding store for semantic retrieval.

This module provides vector storage for conversation messages using LanceDB:
- Embeddings are pre-computed and stored for efficient retrieval
- Supports semantic search across conversation history
- Syncs with SQLite ConversationStore for authoritative message data

Architecture:
- SQLite (ConversationStore): Authoritative source for messages, metadata, sessions
- LanceDB (ConversationEmbeddingStore): Vector embeddings for fast semantic search

Storage location: {project}/.victor/embeddings/conversations/

Usage:
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore
    from victor.embeddings.service import EmbeddingService

    embedding_service = EmbeddingService.get_instance()
    store = ConversationEmbeddingStore(embedding_service)
    await store.initialize()

    # Add message embedding
    await store.add_message_embedding(
        message_id="msg_abc123",
        session_id="session_xyz",
        role="user",
        content="How do I implement authentication?",
    )

    # Search for similar messages
    results = await store.search_similar(
        query="auth login implementation",
        session_id="session_xyz",
        limit=10,
    )
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    import lancedb
    import pyarrow as pa  # noqa: F401 - Required by LanceDB

    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

if TYPE_CHECKING:
    from victor.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class ConversationSearchResult:
    """Result from semantic conversation search."""

    message_id: str
    session_id: str
    role: str
    content_preview: str  # First 200 chars for reference
    similarity: float
    timestamp: Optional[datetime] = None

    def __repr__(self) -> str:
        return (
            f"ConversationSearchResult(id={self.message_id}, "
            f"role={self.role}, sim={self.similarity:.3f})"
        )


class ConversationEmbeddingStore:
    """LanceDB-based vector store for conversation embeddings.

    Provides efficient semantic search over conversation history by:
    - Pre-computing and storing message embeddings
    - Using LanceDB's fast ANN search
    - Supporting session-scoped queries
    - Automatic pruning of old messages beyond MAX_MESSAGES

    Key benefits over on-the-fly embedding:
    - O(1) embedding lookup instead of O(n) computation
    - Sub-millisecond vector search via LanceDB indices
    - Persistent storage survives process restarts
    - Bounded storage via automatic pruning

    Schema:
        - id: str (message ID)
        - session_id: str (session the message belongs to)
        - role: str (user/assistant/system/tool)
        - content_preview: str (first 500 chars for reference)
        - vector: list[float] (384-dim embedding)
        - timestamp: str (ISO format)
    """

    # LanceDB table name for conversation embeddings
    TABLE_NAME = "conversation_embeddings"

    # Maximum content length to store (for preview)
    MAX_CONTENT_PREVIEW = 500

    # Maximum messages to keep in the store (prunes oldest when exceeded)
    # 10,000 messages ≈ 15MB storage (384-dim embeddings)
    MAX_MESSAGES = 10_000

    # Prune when this many messages over limit (batch pruning for efficiency)
    PRUNE_BATCH_SIZE = 500

    def __init__(
        self,
        embedding_service: "EmbeddingService",
        db_path: Optional[Path] = None,
    ):
        """Initialize the conversation embedding store.

        Args:
            embedding_service: Shared EmbeddingService instance for generating embeddings
            db_path: Path to LanceDB directory. Defaults to {project}/.victor/embeddings/conversations/
        """
        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB not available. Install with: pip install lancedb pyarrow")

        self._embedding_service = embedding_service
        self._db_path = db_path
        self._db: Optional[Any] = None  # LanceDB connection
        self._table: Optional[Any] = None  # LanceDB table
        self._initialized = False

    @property
    def dimension(self) -> int:
        """Get embedding dimension from the embedding service."""
        return self._embedding_service.dimension

    @property
    def is_initialized(self) -> bool:
        """Check if the store is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize LanceDB connection and table.

        Creates the database directory and table if they don't exist.
        """
        if self._initialized:
            return

        # Determine database path
        if self._db_path is None:
            from victor.config.settings import get_project_paths

            paths = get_project_paths()
            self._db_path = paths.embeddings_dir / "conversations"

        # Ensure directory exists
        self._db_path.mkdir(parents=True, exist_ok=True)

        # Connect to LanceDB
        logger.info(f"[ConversationEmbeddingStore] Connecting to LanceDB at {self._db_path}")
        self._db = lancedb.connect(str(self._db_path))

        # Open or verify table exists
        existing_tables = self._db.table_names()
        if self.TABLE_NAME in existing_tables:
            self._table = self._db.open_table(self.TABLE_NAME)
            logger.info(f"[ConversationEmbeddingStore] Opened existing table '{self.TABLE_NAME}'")
        else:
            logger.info(
                f"[ConversationEmbeddingStore] Table '{self.TABLE_NAME}' will be created on first insert"
            )

        self._initialized = True
        logger.info("[ConversationEmbeddingStore] Initialized successfully")

    def _create_table_with_first_record(self, record: Dict[str, Any]) -> None:
        """Create the table with the first record (LanceDB requires data to infer schema)."""
        self._table = self._db.create_table(self.TABLE_NAME, data=[record])
        logger.info(f"[ConversationEmbeddingStore] Created table '{self.TABLE_NAME}'")

    async def add_message_embedding(
        self,
        message_id: str,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add a message embedding to the store.

        Computes the embedding and stores it with metadata.

        Args:
            message_id: Unique message identifier (from SQLite)
            session_id: Session the message belongs to
            role: Message role (user/assistant/system/tool)
            content: Full message content (will be embedded)
            timestamp: Message timestamp (defaults to now)
        """
        if not self._initialized:
            await self.initialize()

        # Skip very short messages (not useful for semantic search)
        if len(content.strip()) < 10:
            logger.debug(f"[ConversationEmbeddingStore] Skipping short message {message_id}")
            return

        start_time = time.perf_counter()

        # Compute embedding
        # Truncate very long content for embedding (model has token limits)
        embed_content = content[:2000]
        embedding = await self._embedding_service.embed_text(embed_content)

        # Prepare record
        record = {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content_preview": content[: self.MAX_CONTENT_PREVIEW],
            "vector": embedding.tolist(),
            "timestamp": (timestamp or datetime.now()).isoformat(),
        }

        # Add to table (create if needed)
        if self._table is None:
            self._create_table_with_first_record(record)
        else:
            self._table.add([record])

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"[ConversationEmbeddingStore] Added message {message_id}: "
            f"role={role}, chars={len(content)}, time={elapsed*1000:.2f}ms"
        )

    async def add_messages_batch(
        self,
        messages: List[Dict[str, Any]],
    ) -> int:
        """Add multiple message embeddings in batch.

        More efficient than adding one by one.

        Args:
            messages: List of dicts with keys: message_id, session_id, role, content, timestamp

        Returns:
            Number of messages successfully added
        """
        if not self._initialized:
            await self.initialize()

        if not messages:
            return 0

        # Filter out very short messages
        valid_messages = [m for m in messages if len(m.get("content", "").strip()) >= 10]
        if not valid_messages:
            return 0

        start_time = time.perf_counter()

        # Extract content for batch embedding
        contents = [m["content"][:2000] for m in valid_messages]

        # Batch embed
        embeddings = await self._embedding_service.embed_batch(contents)

        # Prepare records
        records = []
        for msg, embedding in zip(valid_messages, embeddings, strict=False):
            record = {
                "id": msg["message_id"],
                "session_id": msg["session_id"],
                "role": msg["role"],
                "content_preview": msg["content"][: self.MAX_CONTENT_PREVIEW],
                "vector": embedding.tolist(),
                "timestamp": (
                    (msg.get("timestamp") or datetime.now()).isoformat()
                    if isinstance(msg.get("timestamp"), datetime)
                    else msg.get("timestamp", datetime.now().isoformat())
                ),
            }
            records.append(record)

        # Add to table
        if self._table is None:
            self._create_table_with_first_record(records[0])
            if len(records) > 1:
                self._table.add(records[1:])
        else:
            self._table.add(records)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[ConversationEmbeddingStore] Batch added {len(records)} messages "
            f"in {elapsed*1000:.2f}ms ({elapsed*1000/len(records):.2f}ms/msg)"
        )

        # Prune old messages if over limit
        await self._prune_old_messages()

        return len(records)

    async def search_similar(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> List[ConversationSearchResult]:
        """Search for messages semantically similar to a query.

        Args:
            query: Query text to search for
            session_id: Optional session to scope search (None = all sessions)
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            exclude_message_ids: Message IDs to exclude from results

        Returns:
            List of ConversationSearchResult sorted by similarity (descending)
        """
        if not self._initialized:
            await self.initialize()

        if self._table is None:
            logger.debug("[ConversationEmbeddingStore] No table exists, returning empty results")
            return []

        start_time = time.perf_counter()

        # Embed query
        query_embedding = await self._embedding_service.embed_text(query[:2000])

        # Search in LanceDB
        search_query = self._table.search(query_embedding.tolist()).limit(
            limit * 2
        )  # Over-fetch for filtering

        # Apply session filter if provided
        if session_id:
            search_query = search_query.where(f"session_id = '{session_id}'")

        # Execute search
        results = search_query.to_list()

        # Convert to result objects with filtering
        search_results: List[ConversationSearchResult] = []
        exclude_set = set(exclude_message_ids or [])

        for result in results:
            message_id = result.get("id", "")

            # Skip excluded messages
            if message_id in exclude_set:
                continue

            # Convert distance to similarity (LanceDB returns L2 distance by default)
            distance = result.get("_distance", 0.0)
            # For cosine distance: similarity = 1 - distance
            # For L2 distance: similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + distance)

            # Filter by minimum similarity
            if similarity < min_similarity:
                continue

            # Parse timestamp
            timestamp_str = result.get("timestamp")
            timestamp = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    pass

            search_results.append(
                ConversationSearchResult(
                    message_id=message_id,
                    session_id=result.get("session_id", ""),
                    role=result.get("role", ""),
                    content_preview=result.get("content_preview", ""),
                    similarity=similarity,
                    timestamp=timestamp,
                )
            )

            if len(search_results) >= limit:
                break

        elapsed = time.perf_counter() - start_time
        logger.debug(
            f"[ConversationEmbeddingStore] Search complete: "
            f"query_len={len(query)}, results={len(search_results)}, "
            f"time={elapsed*1000:.2f}ms"
        )

        return search_results

    async def search_by_role(
        self,
        query: str,
        role: str,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[ConversationSearchResult]:
        """Search for messages of a specific role similar to a query.

        Useful for finding relevant tool results, user questions, etc.

        Args:
            query: Query text
            role: Role to filter by (user/assistant/tool/system)
            session_id: Optional session filter
            limit: Maximum results

        Returns:
            List of matching results
        """
        if not self._initialized:
            await self.initialize()

        if self._table is None:
            return []

        # Embed query
        query_embedding = await self._embedding_service.embed_text(query[:2000])

        # Build search with role filter
        where_clause = f"role = '{role}'"
        if session_id:
            where_clause += f" AND session_id = '{session_id}'"

        results = (
            self._table.search(query_embedding.tolist()).where(where_clause).limit(limit).to_list()
        )

        search_results: List[ConversationSearchResult] = []
        for result in results:
            distance = result.get("_distance", 0.0)
            similarity = 1.0 / (1.0 + distance)

            timestamp_str = result.get("timestamp")
            timestamp = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    pass

            search_results.append(
                ConversationSearchResult(
                    message_id=result.get("id", ""),
                    session_id=result.get("session_id", ""),
                    role=result.get("role", ""),
                    content_preview=result.get("content_preview", ""),
                    similarity=similarity,
                    timestamp=timestamp,
                )
            )

        return search_results

    async def delete_message(self, message_id: str) -> bool:
        """Delete a message embedding by ID.

        Args:
            message_id: Message ID to delete

        Returns:
            True if deleted, False if not found or error
        """
        if not self._initialized:
            await self.initialize()

        if self._table is None:
            return False

        try:
            self._table.delete(f"id = '{message_id}'")
            logger.debug(f"[ConversationEmbeddingStore] Deleted message {message_id}")
            return True
        except Exception as e:
            logger.warning(f"[ConversationEmbeddingStore] Failed to delete {message_id}: {e}")
            return False

    async def delete_session(self, session_id: str) -> int:
        """Delete all message embeddings for a session.

        Args:
            session_id: Session ID to delete

        Returns:
            Number of messages deleted (estimated)
        """
        if not self._initialized:
            await self.initialize()

        if self._table is None:
            return 0

        try:
            # Count before deletion
            count_before = self._table.count_rows()

            # Delete
            self._table.delete(f"session_id = '{session_id}'")

            # Count after
            count_after = self._table.count_rows()
            deleted = count_before - count_after

            logger.info(
                f"[ConversationEmbeddingStore] Deleted {deleted} messages from session {session_id}"
            )
            return deleted
        except Exception as e:
            logger.warning(
                f"[ConversationEmbeddingStore] Failed to delete session {session_id}: {e}"
            )
            return 0

    async def _prune_old_messages(self) -> int:
        """Prune oldest messages if over MAX_MESSAGES limit.

        Uses batch pruning for efficiency - only prunes when PRUNE_BATCH_SIZE
        over the limit to avoid frequent small deletions.

        Returns:
            Number of messages pruned
        """
        if self._table is None:
            return 0

        try:
            count = self._table.count_rows()

            # Only prune if we're significantly over limit
            if count <= self.MAX_MESSAGES + self.PRUNE_BATCH_SIZE:
                return 0

            # Calculate how many to delete
            delete_count = count - self.MAX_MESSAGES

            # Get oldest message IDs by timestamp
            # LanceDB doesn't have direct "ORDER BY ... LIMIT" so we query all and sort
            df = self._table.to_pandas()
            if len(df) == 0:
                return 0

            # Sort by timestamp ascending and take oldest
            df = df.sort_values("timestamp", ascending=True)
            oldest_ids = df.head(delete_count)["id"].tolist()

            if not oldest_ids:
                return 0

            # Delete in batches to avoid huge SQL strings
            deleted = 0
            batch_size = 100
            for i in range(0, len(oldest_ids), batch_size):
                batch_ids = oldest_ids[i : i + batch_size]
                # Build delete condition
                id_list = ", ".join(f"'{id}'" for id in batch_ids)
                self._table.delete(f"id IN ({id_list})")
                deleted += len(batch_ids)

            logger.info(
                f"[ConversationEmbeddingStore] Pruned {deleted} old messages "
                f"(was {count}, now {count - deleted}, limit {self.MAX_MESSAGES})"
            )
            return deleted

        except Exception as e:
            logger.warning(f"[ConversationEmbeddingStore] Pruning failed: {e}")
            return 0

    async def compact(self) -> bool:
        """Compact the LanceDB table to reclaim space after deletions.

        Returns:
            True if compaction succeeded
        """
        if self._table is None:
            return False

        try:
            self._table.compact_files()
            self._table.cleanup_old_versions(older_than=None, delete_unverified=True)
            logger.info("[ConversationEmbeddingStore] Compaction complete")
            return True
        except Exception as e:
            logger.warning(f"[ConversationEmbeddingStore] Compaction failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics.

        Returns:
            Dictionary with stats
        """
        if not self._initialized:
            await self.initialize()

        count = 0
        if self._table is not None:
            try:
                count = self._table.count_rows()
            except (AttributeError, RuntimeError, ValueError):
                pass

        return {
            "store": "conversation_embedding_store",
            "backend": "lancedb",
            "total_messages": count,
            "max_messages": self.MAX_MESSAGES,
            "usage_pct": (count / self.MAX_MESSAGES * 100) if self.MAX_MESSAGES > 0 else 0,
            "embedding_dimension": self.dimension,
            "embedding_model": self._embedding_service.model_name,
            "db_path": str(self._db_path),
            "table_name": self.TABLE_NAME,
        }

    async def close(self) -> None:
        """Clean up resources."""
        self._table = None
        self._db = None
        self._initialized = False
        logger.info("[ConversationEmbeddingStore] Closed")


# Module-level singleton instance
_embedding_store: Optional[ConversationEmbeddingStore] = None


async def get_conversation_embedding_store(
    embedding_service: Optional["EmbeddingService"] = None,
) -> ConversationEmbeddingStore:
    """Get or create the global ConversationEmbeddingStore instance.

    Args:
        embedding_service: EmbeddingService to use. If None, uses the singleton.

    Returns:
        Initialized ConversationEmbeddingStore instance
    """
    global _embedding_store

    if _embedding_store is None:
        if embedding_service is None:
            from victor.embeddings.service import EmbeddingService

            embedding_service = EmbeddingService.get_instance()

        _embedding_store = ConversationEmbeddingStore(embedding_service)
        await _embedding_store.initialize()

    return _embedding_store
