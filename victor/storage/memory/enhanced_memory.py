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

"""Enhanced memory system with long-term storage and intelligent retrieval.

This module extends the existing memory system with:
- Long-term memory persistence
- Memory summarization and compression
- Advanced relevance scoring
- Memory clustering and organization
- Automatic memory consolidation
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MemoryPriority(str, Enum):
    """Memory importance levels."""

    CRITICAL = "critical"  # Essential information
    HIGH = "high"  # Important information
    MEDIUM = "medium"  # Normal information
    LOW = "low"  # Less important
    EPHEMERAL = "ephemeral"  # Temporary, can be forgotten


class MemoryType(str, Enum):
    """Types of memories."""

    FACT = "fact"  # Factual information
    PROCEDURE = "procedure"  # How-to knowledge
    CONTEXT = "context"  # Contextual information
    CONVERSATION = "conversation"  # Conversation history
    REFLECTION = "reflection"  # Agent reflections
    EXPERIENCE = "experience"  # Learned experiences
    PREFERENCE = "preference"  # User preferences


@dataclass
class Memory:
    """Enhanced memory entry.

    Attributes:
        id: Unique memory identifier
        content: Memory content
        memory_type: Type of memory
        priority: Importance level
        importance: Importance score (0-1)
        access_count: Number of times accessed
        last_accessed: Last access timestamp
        created_at: Creation timestamp
        expires_at: Optional expiration time
        tags: Associated tags
        embeddings: Vector embedding (optional)
        metadata: Additional metadata
    """

    id: str
    content: str
    memory_type: MemoryType = MemoryType.CONTEXT
    priority: MemoryPriority = MemoryPriority.MEDIUM
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    embeddings: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data.get("memory_type", MemoryType.CONTEXT)),
            priority=MemoryPriority(data.get("priority", MemoryPriority.MEDIUM)),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", time.time()),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemoryCluster:
    """Cluster of related memories.

    Attributes:
        cluster_id: Unique cluster identifier
        memories: List of memory IDs in cluster
        centroid: Cluster centroid (if using embeddings)
        label: Cluster label
        created_at: Creation timestamp
    """

    cluster_id: str
    memories: List[str]
    centroid: Optional[List[float]] = None
    label: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class MemoryConfig:
    """Configuration for enhanced memory.

    Attributes:
        storage_path: Path for persistent storage
        max_memories: Maximum number of memories
        max_age_days: Maximum age in days
        consolidation_threshold: Threshold for consolidation
        enable_summarization: Enable automatic summarization
        enable_clustering: Enable memory clustering
        embedding_model: Embedding model to use
    """

    storage_path: Optional[str] = None
    max_memories: int = 10000
    max_age_days: int = 365
    consolidation_threshold: float = 0.8
    enable_summarization: bool = True
    enable_clustering: bool = True
    embedding_model: str = "default"


class EnhancedMemory:
    """Enhanced memory system with long-term storage.

    Example:
        from victor.storage.memory import EnhancedMemory, MemoryConfig

        memory = EnhancedMemory(
            config=MemoryConfig(
                storage_path=".victor/memory",
                enable_summarization=True
            )
        )

        # Store memory
        await memory.store(
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
            priority=MemoryPriority.HIGH
        )

        # Retrieve relevant memories
        memories = await memory.retrieve_relevant(
            query="user interface preferences",
            limit=5
        )

        # Summarize memories
        summary = await memory.summarize_memories(
            memory_type=MemoryType.CONVERSATION
        )
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize enhanced memory.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self._memories: Dict[str, Memory] = {}
        self._clusters: Dict[str, MemoryCluster] = {}
        self._embeddings_service = None

        # Load from storage if configured
        if self.config.storage_path:
            self._load_from_storage()

    async def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.CONTEXT,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_days: Optional[int] = None,
    ) -> Memory:
        """Store memory with metadata.

        Args:
            content: Memory content
            memory_type: Type of memory
            priority: Priority level
            tags: Associated tags
            metadata: Additional metadata
            ttl_days: Time-to-live in days

        Returns:
            Created Memory
        """
        # Generate ID
        memory_id = self._generate_id(content)

        # Calculate importance
        importance = self._calculate_importance(content, priority)

        # Calculate expiration
        expires_at = None
        if ttl_days:
            expires_at = time.time() + (ttl_days * 86400)

        # Create memory
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            priority=priority,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            expires_at=expires_at,
        )

        # Generate embedding if available
        if self.config.enable_clustering:
            memory.embeddings = await self._generate_embedding(content)

        # Store
        self._memories[memory_id] = memory

        # Persist if configured
        if self.config.storage_path:
            await self._persist_memory(memory)

        # Check if consolidation needed
        await self._maybe_consolidate()

        return memory

    async def retrieve(
        self,
        memory_id: str,
        update_access: bool = True,
    ) -> Optional[Memory]:
        """Retrieve memory by ID.

        Args:
            memory_id: Memory identifier
            update_access: Whether to update access stats

        Returns:
            Memory if found
        """
        memory = self._memories.get(memory_id)

        if memory and update_access:
            memory.access_count += 1
            memory.last_accessed = time.time()

        return memory

    async def retrieve_relevant(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
    ) -> List[Memory]:
        """Retrieve relevant memories.

        Args:
            query: Query string
            limit: Maximum number of memories
            memory_type: Filter by memory type
            min_importance: Minimum importance threshold

        Returns:
            List of relevant memories
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)

        # Score memories
        scored_memories = []

        for memory in self._memories.values():
            # Filter by type
            if memory_type and memory.memory_type != memory_type:
                continue

            # Filter by importance
            if memory.importance < min_importance:
                continue

            # Filter expired
            if memory.expires_at and memory.expires_at < time.time():
                continue

            # Calculate relevance score
            relevance = self._calculate_relevance(query, memory, query_embedding)

            scored_memories.append((relevance, memory))

        # Sort by relevance and return top
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        return [memory for _, memory in scored_memories[:limit]]

    async def search(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Search memories by query and filters.

        Args:
            query: Search query
            tags: Filter by tags
            memory_type: Filter by memory type
            limit: Maximum results

        Returns:
            List of matching memories
        """
        results = []

        for memory in self._memories.values():
            # Text match
            if query.lower() not in memory.content.lower():
                continue

            # Tag filter
            if tags and not any(tag in memory.tags for tag in tags):
                continue

            # Type filter
            if memory_type and memory.memory_type != memory_type:
                continue

            results.append(memory)

            if len(results) >= limit:
                break

        return results

    async def summarize_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        since_days: Optional[int] = None,
    ) -> str:
        """Summarize memories.

        Args:
            memory_type: Type of memories to summarize
            since_days: Only summarize recent memories

        Returns:
            Summary string
        """
        # Filter memories
        memories = list(self._memories.values())

        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]

        if since_days:
            cutoff = time.time() - (since_days * 86400)
            memories = [m for m in memories if m.created_at > cutoff]

        if not memories:
            return "No memories to summarize."

        # Sort by importance
        memories.sort(key=lambda m: m.importance, reverse=True)

        # Extract key information
        key_memories = memories[:10]  # Top 10

        summary_parts = []
        for memory in key_memories:
            summary_parts.append(f"- {memory.content[:100]}...")

        summary = f"Summary of {len(memories)} memories:\n" + "\n".join(summary_parts)

        return summary

    async def consolidate_memories(self) -> int:
        """Consolidate old/less important memories.

        Returns:
            Number of memories consolidated
        """
        # Get memories to consolidate
        memories_to_consolidate = []

        for memory in self._memories.values():
            # Old and low importance
            age = time.time() - memory.created_at
            if age > (self.config.max_age_days * 86400) and memory.importance < 0.3:
                memories_to_consolidate.append(memory)

        # Remove consolidated memories
        for memory in memories_to_consolidate:
            del self._memories[memory.id]

        # Update storage
        if self.config.storage_path:
            await self._save_to_storage()

        return len(memories_to_consolidate)

    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        priority: Optional[MemoryPriority] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update existing memory.

        Args:
            memory_id: Memory to update
            content: New content
            priority: New priority
            tags: New tags
            metadata: New metadata

        Returns:
            True if updated
        """
        memory = self._memories.get(memory_id)

        if not memory:
            return False

        if content is not None:
            memory.content = content

        if priority is not None:
            memory.priority = priority
            memory.importance = self._calculate_importance(content or "", priority)

        if tags is not None:
            memory.tags = tags

        if metadata is not None:
            memory.metadata.update(metadata)

        # Persist if configured
        if self.config.storage_path:
            await self._persist_memory(memory)

        return True

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory.

        Args:
            memory_id: Memory to delete

        Returns:
            True if deleted
        """
        if memory_id in self._memories:
            del self._memories[memory_id]

            if self.config.storage_path:
                await self._save_to_storage()

            return True

        return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Statistics dict
        """
        memories_by_type: Dict[str, int] = {}
        memories_by_priority: Dict[str, int] = {}

        for memory in self._memories.values():
            memories_by_type[memory.memory_type.value] = (
                memories_by_type.get(memory.memory_type.value, 0) + 1
            )
            memories_by_priority[memory.priority.value] = (
                memories_by_priority.get(memory.priority.value, 0) + 1
            )

        return {
            "total_memories": len(self._memories),
            "by_type": memories_by_type,
            "by_priority": memories_by_priority,
            "total_clusters": len(self._clusters),
        }

    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content (non-cryptographic, for ID generation only)."""
        hash_input = f"{content}_{time.time()}"
        return hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()

    def _calculate_importance(self, content: str, priority: MemoryPriority) -> float:
        """Calculate importance score."""
        base_scores = {
            MemoryPriority.CRITICAL: 0.9,
            MemoryPriority.HIGH: 0.7,
            MemoryPriority.MEDIUM: 0.5,
            MemoryPriority.LOW: 0.3,
            MemoryPriority.EPHEMERAL: 0.1,
        }

        base_score = base_scores.get(priority, 0.5)

        # Adjust based on content length (longer = potentially more important)
        length_factor = min(len(content) / 500, 1.0) * 0.1

        return min(base_score + length_factor, 1.0)

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
        # Placeholder for embedding generation
        # In production, use actual embedding service
        return None

    def _calculate_relevance(
        self, query: str, memory: Memory, query_embedding: Optional[List[float]]
    ) -> float:
        """Calculate relevance score."""
        # Simple keyword matching
        query_words = set(query.lower().split())
        content_words = set(memory.content.lower().split())

        # Keyword overlap
        overlap = len(query_words & content_words) / len(query_words) if query_words else 0

        # Combine with importance and recency
        age = time.time() - memory.created_at
        recency = max(0, 1 - age / (30 * 86400))  # Decay over 30 days

        # Tags match
        tag_match = 1.0 if query_words & set(memory.tags) else 0.0

        # Combined score
        relevance = 0.5 * overlap + 0.3 * memory.importance + 0.1 * recency + 0.1 * tag_match

        return relevance

    async def _maybe_consolidate(self) -> None:
        """Check if consolidation is needed."""
        if len(self._memories) > self.config.max_memories:
            await self.consolidate_memories()

    async def _persist_memory(self, memory: Memory) -> None:
        """Persist memory to storage."""
        if not self.config.storage_path:
            return

        storage_path = Path(self.config.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        memory_file = storage_path / f"{memory.id}.json"

        with open(memory_file, "w") as f:
            json.dump(memory.to_dict(), f, indent=2)

    def _load_from_storage(self) -> None:
        """Load memories from storage."""
        if not self.config.storage_path:
            return

        storage_path = Path(self.config.storage_path)

        if not storage_path.exists():
            return

        for memory_file in storage_path.glob("*.json"):
            try:
                with open(memory_file, "r") as f:
                    data = json.load(f)
                    memory = Memory.from_dict(data)
                    self._memories[memory.id] = memory
            except Exception as e:
                logger.warning(f"Failed to load memory from {memory_file}: {e}")

    async def _save_to_storage(self) -> None:
        """Save all memories to storage."""
        if not self.config.storage_path:
            return

        storage_path = Path(self.config.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Save each memory
        for memory in self._memories.values():
            await self._persist_memory(memory)

        # Clean up orphaned files
        stored_ids = {f"{m}.json" for m in self._memories.keys()}
        for memory_file in storage_path.glob("*.json"):
            if memory_file.name not in stored_ids:
                memory_file.unlink()
