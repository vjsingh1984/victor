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

"""Episodic memory system for agentic AI.

This module implements an episodic memory system that stores and retrieves
agent experiences (episodes) with context, actions, and outcomes.

Key Features:
- Store episodes with context, actions, outcomes, rewards
- Vector similarity search for relevant episodes
- Temporal decay for episode importance
- Memory consolidation from episodic to semantic
- Automatic forgetting of old episodes
- Context-aware retrieval
- Reward-based episode ranking
- Efficient similarity search with MemoryIndex

Architecture:
    EpisodicMemory
    ├── Episode Storage (EpisodeMemory with embeddings)
    ├── MemoryIndex (for fast similarity search)
    ├── Embedding Generation (via EmbeddingService)
    ├── Consolidation (episodic → semantic)
    ├── Temporal Decay (importance adjustment)
    └── Forgetting (age-based eviction)

Usage:
    from victor.agent.memory import EpisodicMemory, Episode, MemoryStats

    # Create episodic memory
    memory = EpisodicMemory()

    # Store an episode
    episode = Episode(
        inputs={"query": "fix authentication bug"},
        actions=["read_file", "edit_file"],
        outcomes={"success": True, "files_changed": 2},
        rewards=10.0
    )
    episode_id = memory.store_episode(episode)

    # Recall relevant episodes
    relevant = memory.recall_relevant("authentication error", k=5)

    # Recall recent episodes
    recent = memory.recall_recent(n=10)

    # Recall by outcome
    successful = memory.recall_by_outcome("success")

    # Get statistics
    stats = memory.get_memory_statistics()
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService
    from victor.agent.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Represents a single episodic memory (agent experience).

    An episode captures the complete context, actions, and outcomes of an
    agent's experience, enabling learning and retrieval.

    Attributes:
        id: Unique episode identifier (UUID)
        timestamp: When the episode occurred
        inputs: Initial inputs/context (query, state, environment)
        actions: List of actions taken during the episode
        outcomes: Results and outcomes of actions
        rewards: Reward signal for reinforcement learning (optional)
        embedding: Vector embedding for similarity search (optional)
        context: Additional context information
        importance: Subjective importance score (0-1, default 0.5)
        access_count: Number of times this episode was accessed
        last_accessed: Last time this episode was recalled
        decay_factor: Temporal decay factor (0-1, adjusted over time)

    Example:
        episode = Episode(
            inputs={"query": "fix authentication bug", "files": ["auth.py"]},
            actions=["read_file", "edit_file", "run_tests"],
            outcomes={"success": True, "tests_passed": 5},
            rewards=10.0,
            context={"task_type": "bugfix", "complexity": "medium"}
        )
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    inputs: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    rewards: float = 0.0
    embedding: Optional[np.ndarray] = None
    context: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    decay_factor: float = 1.0

    def to_text(self) -> str:
        """Convert episode to text for embedding generation.

        Returns:
            Text representation of the episode
        """
        parts = []

        # Inputs/Context
        if self.inputs:
            inputs_str = ", ".join(f"{k}={v}" for k, v in self.inputs.items())
            parts.append(f"Inputs: {inputs_str}")
        else:
            parts.append("Inputs:")

        # Actions
        if self.actions:
            actions_str = ", ".join(self.actions)
            parts.append(f"Actions: {actions_str}")
        else:
            parts.append("Actions:")

        # Outcomes
        if self.outcomes:
            outcomes_str = ", ".join(f"{k}={v}" for k, v in self.outcomes.items())
            parts.append(f"Outcomes: {outcomes_str}")
        else:
            parts.append("Outcomes:")

        # Rewards
        if self.rewards != 0.0:
            parts.append(f"Reward: {self.rewards}")

        return " | ".join(parts)

    def compute_effective_importance(self, current_time: Optional[datetime] = None) -> float:
        """Compute effective importance with temporal decay.

        Args:
            current_time: Current time for decay calculation (defaults to now)

        Returns:
            Effective importance score (0-1)
        """
        if current_time is None:
            current_time = datetime.utcnow()

        # Apply temporal decay based on age
        age_hours = (current_time - self.timestamp).total_seconds() / 3600

        # Decay factor reduces by 1% per hour (configurable)
        decayed_importance = self.importance * (self.decay_factor**age_hours)

        # Boost based on rewards (positive rewards increase importance)
        reward_boost = min(abs(self.rewards) / 100.0, 0.3)  # Max 30% boost

        # Boost based on access frequency
        access_boost = min(self.access_count * 0.05, 0.2)  # Max 20% boost

        effective = min(decayed_importance + reward_boost + access_boost, 1.0)
        return float(max(effective, 0.0))

    def __repr__(self) -> str:
        """String representation of episode."""
        return (
            f"Episode(id={self.id[:8]}..., "
            f"timestamp={self.timestamp.isoformat()}, "
            f"actions={len(self.actions)}, "
            f"reward={self.rewards})"
        )


@dataclass
class EpisodeMemory:
    """Memory container for episode with embedding and metadata.

    This wraps an Episode with additional indexing information for
    efficient retrieval and similarity search.

    Attributes:
        episode: The episode data
        embedding_id: ID for embedding index lookup
        created_at: When this memory was created
        updated_at: When this memory was last updated
        access_count: Number of times accessed
        last_accessed: Last access timestamp
        metadata: Additional metadata for filtering
    """

    episode: Episode
    embedding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        self.updated_at = datetime.utcnow()


@dataclass
class MemoryStats:
    """Statistics about episodic memory system.

    Attributes:
        total_episodes: Current number of episodes
        total_stored: Total episodes ever stored
        total_forgotten: Total episodes forgotten
        max_episodes: Maximum storage capacity
        utilization_pct: Storage utilization percentage
        avg_importance: Average importance score
        avg_reward: Average reward
        avg_access_count: Average access count
        oldest_episode: Timestamp of oldest episode
        newest_episode: Timestamp of newest episode
        total_embeddings: Number of embeddings stored
        index_size: Memory index size
        decay_rate: Current decay rate
        consolidation_count: Number of consolidations performed
    """

    total_episodes: int = 0
    total_stored: int = 0
    total_forgotten: int = 0
    max_episodes: int = 0
    utilization_pct: float = 0.0
    avg_importance: float = 0.0
    avg_reward: float = 0.0
    avg_access_count: float = 0.0
    oldest_episode: Optional[str] = None
    newest_episode: Optional[str] = None
    total_embeddings: int = 0
    index_size: int = 0
    decay_rate: float = 0.01
    consolidation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all statistics
        """
        return {
            "total_episodes": self.total_episodes,
            "total_stored": self.total_stored,
            "total_forgotten": self.total_forgotten,
            "max_episodes": self.max_episodes,
            "utilization_pct": self.utilization_pct,
            "avg_importance": self.avg_importance,
            "avg_reward": self.avg_reward,
            "avg_access_count": self.avg_access_count,
            "oldest_episode": self.oldest_episode,
            "newest_episode": self.newest_episode,
            "total_embeddings": self.total_embeddings,
            "index_size": self.index_size,
            "decay_rate": self.decay_rate,
            "consolidation_count": self.consolidation_count,
        }


@dataclass
class MemoryIndex:
    """Index for fast similarity search over episodes.

    Maintains a vector index and metadata index for efficient retrieval.

    Attributes:
        embeddings: Dictionary of episode_id -> embedding vector
        metadata_index: Inverted index for metadata filtering
        action_index: Inverted index for action-based lookup
        outcome_index: Inverted index for outcome-based lookup
        temporal_index: List of (timestamp, episode_id) for temporal queries
        dimension: Embedding dimension
    """

    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata_index: Dict[str, Any] = field(
        default_factory=lambda: {"by_key": {}}
    )  # key -> value -> [episode_ids]
    action_index: Dict[str, List[str]] = field(default_factory=dict)  # action -> [episode_ids]
    outcome_index: Dict[str, List[str]] = field(default_factory=dict)  # outcome -> [episode_ids]
    temporal_index: List[Tuple[datetime, str]] = field(default_factory=list)
    dimension: int = 384  # Default embedding dimension

    def add_embedding(self, episode_id: str, embedding: np.ndarray) -> None:
        """Add an embedding to the index.

        Args:
            episode_id: Episode ID
            embedding: Embedding vector
        """
        self.embeddings[episode_id] = embedding
        if self.dimension == 384 and len(embedding) != 384:
            self.dimension = len(embedding)

    def remove_embedding(self, episode_id: str) -> None:
        """Remove an embedding from the index.

        Args:
            episode_id: Episode ID to remove
        """
        self.embeddings.pop(episode_id, None)

    def get_embedding(self, episode_id: str) -> Optional[np.ndarray]:
        """Get embedding for an episode.

        Args:
            episode_id: Episode ID

        Returns:
            Embedding vector or None
        """
        return self.embeddings.get(episode_id)

    def index_metadata(self, episode_id: str, metadata: Dict[str, Any]) -> None:
        """Index episode metadata for filtering.

        Args:
            episode_id: Episode ID
            metadata: Metadata dictionary
        """
        # metadata_index structure: {"by_key": {key: {value: [episode_ids]}}}
        if "by_key" not in self.metadata_index:
            self.metadata_index["by_key"] = {}

        by_key = self.metadata_index["by_key"]
        for key, value in metadata.items():
            if key not in by_key:
                by_key[key] = {}

            value_str = str(value)
            if value_str not in by_key[key]:
                by_key[key][value_str] = []

            if episode_id not in by_key[key][value_str]:
                by_key[key][value_str].append(episode_id)

    def index_actions(self, episode_id: str, actions: List[str]) -> None:
        """Index episode actions for action-based lookup.

        Args:
            episode_id: Episode ID
            actions: List of actions
        """
        for action in actions:
            if action not in self.action_index:
                self.action_index[action] = []
            if episode_id not in self.action_index[action]:
                self.action_index[action].append(episode_id)

    def index_outcomes(self, episode_id: str, outcomes: Dict[str, Any]) -> None:
        """Index episode outcomes for outcome-based lookup.

        Args:
            episode_id: Episode ID
            outcomes: Outcomes dictionary
        """
        for key in outcomes.keys():
            if key not in self.outcome_index:
                self.outcome_index[key] = []
            if episode_id not in self.outcome_index[key]:
                self.outcome_index[key].append(episode_id)

    def index_timestamp(self, episode_id: str, timestamp: datetime) -> None:
        """Index episode timestamp for temporal queries.

        Args:
            episode_id: Episode ID
            timestamp: Episode timestamp
        """
        self.temporal_index.append((timestamp, episode_id))
        # Keep temporal index sorted
        self.temporal_index.sort(key=lambda x: x[0], reverse=True)

    def remove_from_indexes(self, episode_id: str) -> None:
        """Remove episode from all indexes.

        Args:
            episode_id: Episode ID to remove
        """
        # Remove from embedding index
        self.remove_embedding(episode_id)

        # Remove from metadata index
        if "by_key" in self.metadata_index:
            by_key = self.metadata_index["by_key"]
            for key in list(by_key.keys()):
                for value in list(by_key[key].keys()):
                    if episode_id in by_key[key][value]:
                        by_key[key][value].remove(episode_id)

        # Remove from action index
        for action in self.action_index:
            if episode_id in self.action_index[action]:
                self.action_index[action].remove(episode_id)

        # Remove from outcome index
        for outcome in self.outcome_index:
            if episode_id in self.outcome_index[outcome]:
                self.outcome_index[outcome].remove(episode_id)

        # Remove from temporal index
        self.temporal_index = [(ts, eid) for ts, eid in self.temporal_index if eid != episode_id]

    def query_by_metadata(self, filters: Dict[str, Any]) -> List[str]:
        """Query episodes by metadata filters.

        Args:
            filters: Metadata key-value filters

        Returns:
            List of matching episode IDs
        """
        if not filters:
            return []

        result_sets = []
        if "by_key" in self.metadata_index:
            by_key = self.metadata_index["by_key"]
            for key, value in filters.items():
                if key in by_key:
                    value_str = str(value)
                    if value_str in by_key[key]:
                        result_sets.append(set(by_key[key][value_str]))

        if not result_sets:
            return []

        # Intersection of all filters
        return list(set.intersection(*result_sets)) if result_sets else []

    def query_by_action(self, action: str) -> List[str]:
        """Query episodes that contain an action.

        Args:
            action: Action name

        Returns:
            List of episode IDs
        """
        return self.action_index.get(action, [])

    def query_by_outcome(self, outcome: str) -> List[str]:
        """Query episodes that have an outcome key.

        Args:
            outcome: Outcome key

        Returns:
            List of episode IDs
        """
        return self.outcome_index.get(outcome, [])

    def query_by_timerange(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> List[str]:
        """Query episodes by time range.

        Args:
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            List of episode IDs
        """
        result = []
        for timestamp, episode_id in self.temporal_index:
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            result.append(episode_id)
        return result

    def find_similar(
        self, query_embedding: np.ndarray, k: int = 5, min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Find similar episodes by cosine similarity.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (episode_id, similarity) tuples
        """
        similarities = []

        for episode_id, embedding in self.embeddings.items():
            # Cosine similarity
            similarity = float(
                np.dot(query_embedding, embedding)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8)
            )

            if similarity >= min_similarity:
                similarities.append((episode_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]


class EpisodicMemory:
    """Episodic memory system for storing and retrieving agent experiences.

    This class provides a complete episodic memory implementation with:
    - Episode storage with embeddings
    - Vector similarity search via MemoryIndex
    - Filtering by actions, outcomes, metadata
    - Temporal decay for episode importance
    - Memory consolidation
    - Automatic forgetting
    - Reward-based ranking
    - Context-aware retrieval

    Attributes:
        embedding_service: Service for generating embeddings
        max_episodes: Maximum number of episodes to store (default: 10,000)
        decay_rate: Hourly decay rate for importance (default: 0.01 = 1%)
        consolidation_threshold: Minimum episodes for consolidation (default: 100)

    Example:
        from victor.agent.memory import EpisodicMemory, Episode

        memory = EpisodicMemory(max_episodes=5000)

        # Store episode
        episode = Episode(
            inputs={"query": "fix bug"},
            actions=["edit_file"],
            outcomes={"success": True},
            rewards=10.0
        )
        episode_id = memory.store_episode(episode)

        # Recall relevant episodes
        relevant = memory.recall_relevant("how to fix bugs", k=5)

        # Recall recent episodes
        recent = memory.recall_recent(n=10)

        # Recall by outcome
        successful = memory.recall_by_outcome("success")
    """

    def __init__(
        self,
        embedding_service: Optional["EmbeddingService"] = None,
        max_episodes: int = 10000,
        decay_rate: float = 0.01,
        consolidation_threshold: int = 100,
    ):
        """Initialize episodic memory.

        Args:
            embedding_service: Service for generating embeddings (optional)
            max_episodes: Maximum number of episodes to store
            decay_rate: Hourly decay rate for importance (0-1)
            consolidation_threshold: Minimum episodes for consolidation
        """
        from victor.storage.embeddings.service import get_embedding_service

        self._embedding_service = embedding_service or get_embedding_service()
        self._max_episodes = max_episodes
        self._decay_rate = decay_rate
        self._consolidation_threshold = consolidation_threshold

        # Episode storage (id -> EpisodeMemory)
        self._episodes: Dict[str, EpisodeMemory] = {}

        # Memory index for fast search
        self._index = MemoryIndex(dimension=self._embedding_service.dimension)

        # Statistics
        self._total_episodes_stored = 0
        self._total_episodes_forgotten = 0
        self._consolidation_count = 0

    @property
    def episode_count(self) -> int:
        """Get current number of stored episodes."""
        return len(self._episodes)

    @property
    def total_stored(self) -> int:
        """Get total number of episodes ever stored."""
        return self._total_episodes_stored

    @property
    def total_forgotten(self) -> int:
        """Get total number of episodes forgotten."""
        return self._total_episodes_forgotten

    async def store_episode(self, episode: Episode) -> str:
        """Store an episode in memory.

        Generates embedding if not provided, indexes the episode for
        efficient retrieval, and manages storage limits.

        Args:
            episode: Episode to store

        Returns:
            Episode ID

        Raises:
            ValueError: If episode is invalid
        """
        if not episode.id:
            episode.id = str(uuid.uuid4())

        # Generate embedding if not provided
        if episode.embedding is None:
            text = episode.to_text()
            try:
                episode.embedding = await self._embedding_service.embed_text(text)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for episode {episode.id}: {e}")
                # Use zero embedding as fallback
                episode.embedding = np.zeros(self._embedding_service.dimension, dtype=np.float32)

        # Create episode memory wrapper
        episode_memory = EpisodeMemory(episode=episode, metadata=episode.context.copy())

        # Store episode
        self._episodes[episode.id] = episode_memory
        self._total_episodes_stored += 1

        # Index episode for fast retrieval
        self._index.add_embedding(episode.id, episode.embedding)
        self._index.index_metadata(episode.id, episode.context)
        self._index.index_actions(episode.id, episode.actions)
        self._index.index_outcomes(episode.id, episode.outcomes)
        self._index.index_timestamp(episode.id, episode.timestamp)

        # Enforce storage limit (evict oldest/least important)
        if len(self._episodes) > self._max_episodes:
            await self._evict_episodes()

        logger.debug(f"Stored episode {episode.id[:8]}... (total: {len(self._episodes)})")

        return episode.id

    async def _evict_episodes(self) -> None:
        """Evict episodes to enforce storage limit.

        Uses a combination of effective importance, recency, and access count
        to decide which episodes to evict. Evicts low-importance, old,
        rarely accessed episodes first.
        """
        current_time = datetime.utcnow()

        # Sort by effective importance (ascending)
        sorted_episodes = sorted(
            self._episodes.items(),
            key=lambda item: item[1].episode.compute_effective_importance(current_time),
        )

        # Evict 10% of episodes if over limit
        num_to_evict = max(
            1, len(self._episodes) - self._max_episodes + int(self._max_episodes * 0.1)
        )

        for episode_id, _ in sorted_episodes[:num_to_evict]:
            await self._remove_episode(episode_id)

        logger.info(
            f"Evicted {num_to_evict} episodes (total forgotten: {self._total_episodes_forgotten})"
        )

    async def _remove_episode(self, episode_id: str) -> None:
        """Remove an episode from memory.

        Args:
            episode_id: Episode ID to remove
        """
        if episode_id in self._episodes:
            # Remove from storage
            del self._episodes[episode_id]

            # Remove from indexes
            self._index.remove_from_indexes(episode_id)

            # Update statistics
            self._total_episodes_forgotten += 1

    async def recall_relevant(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Episode]:
        """Recall episodes relevant to a query.

        Uses vector similarity search to find the most relevant episodes.
        Applies context filters if provided. Updates access statistics.

        Args:
            query: Natural language query
            context: Optional additional context for filtering
            k: Maximum number of episodes to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of relevant episodes, ordered by similarity (descending)
        """
        if not self._episodes:
            return []

        # Generate query embedding
        try:
            query_embedding = await self._embedding_service.embed_text(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
            # Fallback: return most recent episodes
            return await self.recall_recent(n=k)

        # Find similar episodes via index
        similar_episodes = self._index.find_similar(
            query_embedding, k=k * 2, min_similarity=min_similarity
        )

        # Apply context filters if provided
        if context:
            filtered_ids = self._index.query_by_metadata(context)
            similar_episodes = [(eid, sim) for eid, sim in similar_episodes if eid in filtered_ids]

        # Get top-k episodes and update access statistics
        episodes = []
        for episode_id, similarity in similar_episodes[:k]:
            if episode_id in self._episodes:
                episode_memory = self._episodes[episode_id]
                episode_memory.update_access()

                # Update episode access count
                episode_memory.episode.access_count += 1
                episode_memory.episode.last_accessed = datetime.utcnow()

                episodes.append(episode_memory.episode)

        logger.debug(f"Recalled {len(episodes)} episodes for query: {query[:50]}...")

        return episodes

    async def recall_recent(self, n: int = 10) -> List[Episode]:
        """Recall the most recent episodes.

        Args:
            n: Maximum number of episodes to return

        Returns:
            List of recent episodes, ordered by timestamp (descending)
        """
        # Get recent episode IDs from temporal index
        recent_ids = [eid for _, eid in self._index.temporal_index[:n]]

        # Retrieve episodes
        episodes = []
        for episode_id in recent_ids:
            if episode_id in self._episodes:
                episode_memory = self._episodes[episode_id]
                episode_memory.update_access()
                episodes.append(episode_memory.episode)

        return episodes

    async def recall_by_outcome(self, outcome: str) -> List[Episode]:
        """Recall episodes that have a specific outcome.

        Args:
            outcome: Outcome key to filter by

        Returns:
            List of episodes with the specified outcome
        """
        # Query by outcome index
        episode_ids = self._index.query_by_outcome(outcome)

        # Retrieve episodes
        episodes = []
        for episode_id in episode_ids:
            if episode_id in self._episodes:
                episode_memory = self._episodes[episode_id]
                episode_memory.update_access()
                episodes.append(episode_memory.episode)

        # Sort by timestamp (most recent first)
        episodes.sort(key=lambda ep: ep.timestamp, reverse=True)

        logger.debug(f"Recalled {len(episodes)} episodes with outcome: {outcome}")

        return episodes

    async def recall_by_action(self, action: str) -> List[Episode]:
        """Recall episodes that contain a specific action.

        Args:
            action: Action name to filter by

        Returns:
            List of episodes with the specified action
        """
        # Query by action index
        episode_ids = self._index.query_by_action(action)

        # Retrieve episodes
        episodes = []
        for episode_id in episode_ids:
            if episode_id in self._episodes:
                episode_memory = self._episodes[episode_id]
                episode_memory.update_access()
                episodes.append(episode_memory.episode)

        # Sort by timestamp (most recent first)
        episodes.sort(key=lambda ep: ep.timestamp, reverse=True)

        logger.debug(f"Recalled {len(episodes)} episodes with action: {action}")

        return episodes

    async def recall_by_metadata(self, filters: Dict[str, Any]) -> List[Episode]:
        """Recall episodes matching metadata filters.

        Args:
            filters: Metadata key-value filters

        Returns:
            List of matching episodes
        """
        # Query by metadata index
        episode_ids = self._index.query_by_metadata(filters)

        # Retrieve episodes
        episodes = []
        for episode_id in episode_ids:
            if episode_id in self._episodes:
                episode_memory = self._episodes[episode_id]
                episode_memory.update_access()
                episodes.append(episode_memory.episode)

        # Sort by timestamp (most recent first)
        episodes.sort(key=lambda ep: ep.timestamp, reverse=True)

        logger.debug(f"Recalled {len(episodes)} episodes matching metadata: {filters}")

        return episodes

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID.

        Args:
            episode_id: Episode ID

        Returns:
            Episode if found, None otherwise
        """
        if episode_id in self._episodes:
            episode_memory = self._episodes[episode_id]
            episode_memory.update_access()
            return episode_memory.episode
        return None

    async def consolidate_memories(
        self,
        consolidation_fn: Optional[Callable[[List[Episode]], Dict[str, Dict[str, Any]]]] = None,
    ) -> "SemanticMemory":
        """Consolidate episodic memories into semantic knowledge.

        Extracts general knowledge from specific episodes, optionally using
        a custom consolidation function. Default consolidation extracts
        successful action patterns, common outcomes, and frequent contexts.

        Args:
            consolidation_fn: Optional custom consolidation function

        Returns:
            SemanticMemory with consolidated knowledge
        """
        from victor.agent.memory.semantic_memory import SemanticMemory

        semantic_memory = SemanticMemory(embedding_service=self._embedding_service)

        if len(self._episodes) < self._consolidation_threshold:
            logger.info(
                f"Not enough episodes for consolidation "
                f"({len(self._episodes)} < {self._consolidation_threshold})"
            )
            return semantic_memory

        # Get all episodes
        all_episodes = [mem.episode for mem in self._episodes.values()]

        # Default consolidation: extract patterns and outcomes
        if consolidation_fn is None:
            consolidation_fn = self._default_consolidation

        # Consolidate episodes
        knowledge_items = consolidation_fn(all_episodes)

        # Store knowledge in semantic memory
        for fact, metadata in knowledge_items.items():
            await semantic_memory.store_knowledge(fact, metadata=metadata)

        self._consolidation_count += 1
        logger.info(
            f"Consolidated {len(knowledge_items)} knowledge items "
            f"from {len(all_episodes)} episodes"
        )

        return semantic_memory

    def _default_consolidation(self, episodes: List[Episode]) -> Dict[str, Dict[str, Any]]:
        """Default consolidation function.

        Extracts common patterns from episodes:
        - Successful action sequences
        - Common outcomes
        - Frequent contexts
        - High-reward patterns

        Args:
            episodes: Episodes to consolidate

        Returns:
            Dictionary of {fact: metadata}
        """
        knowledge: Dict[str, Dict[str, Any]] = {}

        if not episodes:
            return knowledge

        # Extract successful action sequences
        successful_episodes = [ep for ep in episodes if ep.outcomes.get("success")]
        if successful_episodes:
            # Most common successful actions
            action_counts: Dict[str, int] = {}
            for ep in successful_episodes:
                for action in ep.actions:
                    action_counts[action] = action_counts.get(action, 0) + 1

            # Top actions become knowledge
            top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for action, count in top_actions:
                if count >= len(successful_episodes) * 0.3:  # Used in >= 30% of successful episodes
                    fact = f"Action '{action}' is often successful (used in {count}/{len(successful_episodes)} episodes)"
                    knowledge[fact] = {
                        "type": "successful_action",
                        "usage_count": count,
                        "success_rate": count / len(successful_episodes),
                    }

        # Extract high-reward patterns
        if any(ep.rewards > 0 for ep in episodes):
            high_reward_episodes = [ep for ep in episodes if ep.rewards > 0]
            if high_reward_episodes:
                avg_reward = sum(ep.rewards for ep in high_reward_episodes) / len(
                    high_reward_episodes
                )
                fact = f"High-reward episodes average reward: {avg_reward:.2f}"
                knowledge[fact] = {
                    "type": "reward_pattern",
                    "average_reward": avg_reward,
                    "episode_count": len(high_reward_episodes),
                }

        # Extract common outcomes
        outcome_keys: Dict[str, int] = {}
        for ep in episodes:
            for key in ep.outcomes.keys():
                outcome_keys[key] = outcome_keys.get(key, 0) + 1

        # Common outcomes become knowledge
        for outcome_key, count in outcome_keys.items():
            if count >= len(episodes) * 0.5:  # Present in >= 50% of episodes
                fact = f"Episodes often have '{outcome_key}' as an outcome"
                knowledge[fact] = {"type": "common_outcome", "frequency": count}

        # Extract common context patterns
        context_values: Dict[str, int] = {}
        for ep in episodes:
            for key, value in ep.context.items():
                if isinstance(value, str):
                    context_key = f"context.{key}"
                    context_values[context_key] = context_values.get(context_key, 0) + 1

        # Common contexts become knowledge
        for context_key, count in context_values.items():
            if count >= len(episodes) * 0.5:  # Present in >= 50% of episodes
                fact = f"Episodes often involve '{context_key}'"
                knowledge[fact] = {"type": "common_context", "frequency": count}

        return knowledge

    def get_memory_statistics(self) -> MemoryStats:
        """Get comprehensive memory system statistics.

        Returns:
            MemoryStats with detailed statistics
        """
        if not self._episodes:
            return MemoryStats(
                total_episodes=0,
                total_stored=self._total_episodes_stored,
                total_forgotten=self._total_episodes_forgotten,
                max_episodes=self._max_episodes,
                utilization_pct=0.0,
                decay_rate=self._decay_rate,
                consolidation_count=self._consolidation_count,
            )

        episodes = [mem.episode for mem in self._episodes.values()]
        importances = [ep.importance for ep in episodes]
        rewards = [ep.rewards for ep in episodes]
        access_counts = [ep.access_count for ep in episodes]

        return MemoryStats(
            total_episodes=len(self._episodes),
            total_stored=self._total_episodes_stored,
            total_forgotten=self._total_episodes_forgotten,
            max_episodes=self._max_episodes,
            utilization_pct=len(self._episodes) / self._max_episodes * 100,
            avg_importance=sum(importances) / len(importances),
            avg_reward=sum(rewards) / len(rewards),
            avg_access_count=sum(access_counts) / len(access_counts),
            oldest_episode=min(ep.timestamp for ep in episodes).isoformat(),
            newest_episode=max(ep.timestamp for ep in episodes).isoformat(),
            total_embeddings=len(self._index.embeddings),
            index_size=len(self._index.temporal_index),
            decay_rate=self._decay_rate,
            consolidation_count=self._consolidation_count,
        )

    async def clear_old_memories(self, max_age: timedelta) -> int:
        """Forget episodes older than max_age.

        High-importance episodes (importance >= 0.8) are preserved
        regardless of age.

        Args:
            max_age: Maximum age of episodes to keep

        Returns:
            Number of episodes forgotten
        """
        cutoff_time = datetime.utcnow() - max_age

        to_forget = [
            ep_id
            for ep_id, ep_mem in self._episodes.items()
            if ep_mem.episode.timestamp < cutoff_time and ep_mem.episode.importance < 0.8
        ]

        for ep_id in to_forget:
            await self._remove_episode(ep_id)

        logger.info(f"Forgot {len(to_forget)} episodes older than {max_age}")

        return len(to_forget)

    async def apply_temporal_decay(self) -> None:
        """Apply temporal decay to all episodes.

        Reduces the effective importance of old episodes based on
        the decay rate. This should be called periodically.
        """
        current_time = datetime.utcnow()

        for episode_memory in self._episodes.values():
            episode = episode_memory.episode

            # Calculate age in hours
            age_hours = (current_time - episode.timestamp).total_seconds() / 3600

            # Apply decay factor
            decay_factor = (1.0 - self._decay_rate) ** age_hours
            episode.decay_factor = decay_factor

        logger.debug(f"Applied temporal decay to {len(self._episodes)} episodes")

    async def clear(self) -> None:
        """Clear all episodes from memory."""
        count = len(self._episodes)
        self._episodes.clear()
        self._index = MemoryIndex(dimension=self._embedding_service.dimension)
        logger.info(f"Cleared {count} episodes from memory")


def create_episodic_memory(
    embedding_service: Optional["EmbeddingService"] = None,
    max_episodes: int = 10000,
    decay_rate: float = 0.01,
    consolidation_threshold: int = 100,
) -> EpisodicMemory:
    """Factory function to create an EpisodicMemory instance.

    Args:
        embedding_service: Optional embedding service
        max_episodes: Maximum number of episodes to store
        decay_rate: Hourly decay rate for importance (0-1)
        consolidation_threshold: Minimum episodes for consolidation

    Returns:
        EpisodicMemory instance
    """
    return EpisodicMemory(
        embedding_service=embedding_service,
        max_episodes=max_episodes,
        decay_rate=decay_rate,
        consolidation_threshold=consolidation_threshold,
    )
