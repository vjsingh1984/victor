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

"""Unified Memory Coordinator for federated search across memory systems.

Provides a single interface for querying multiple memory backends:
- Entity memory (code entities, relationships)
- Conversation memory (message history)
- Embedding memory (semantic vectors)
- Graph memory (entity relationships)
- RL experience (tool selection patterns)

Design Patterns:
- Strategy Pattern: Pluggable ranking strategies
- Adapter Pattern: Memory providers wrap existing systems
- Repository Pattern: Unified query interface

Example:
    coordinator = UnifiedMemoryCoordinator()
    coordinator.register_provider(ConversationMemoryAdapter(store))
    coordinator.register_provider(EntityMemoryAdapter(entity_memory))

    # Federated search across all providers
    results = await coordinator.search_all(
        query="authentication",
        limit=20,
    )

    # Filter by memory type
    results = await coordinator.search_all(
        query="login function",
        memory_types=[MemoryType.ENTITY, MemoryType.EMBEDDING],
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class MemoryType(Enum):
    """Types of memory systems for federated queries.

    Each type represents a distinct memory backend with its own
    storage characteristics and query patterns.
    """

    ENTITY = auto()
    """Entity memory - code entities, symbols, definitions."""

    CONVERSATION = auto()
    """Conversation memory - message history, context."""

    EMBEDDING = auto()
    """Embedding memory - semantic vector storage."""

    GRAPH = auto()
    """Graph memory - entity relationships, traversal."""

    RL_EXPERIENCE = auto()
    """RL experience - tool selection patterns, outcomes."""

    CODE = auto()
    """Code memory - file contents, chunks."""


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class MemoryResult:
    """Unified result from any memory system.

    Provides a common format for results regardless of source,
    enabling consistent ranking and deduplication.

    Attributes:
        source: Which memory system produced this result
        content: The actual content (text, entity, etc.)
        relevance: Relevance score 0.0-1.0 (higher is more relevant)
        id: Unique identifier for deduplication
        metadata: Additional source-specific information
        timestamp: When the content was stored (for recency ranking)
    """

    source: MemoryType
    content: Any
    relevance: float
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.id:
            content_str = str(self.content)[:100]
            # MD5 used for memory ID generation, not security
            self.id = hashlib.md5(
                f"{self.source.name}:{content_str}".encode(), usedforsecurity=False
            ).hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"MemoryResult(source={self.source.name}, "
            f"relevance={self.relevance:.3f}, id={self.id[:8]}...)"
        )


@dataclass
class MemoryQuery:
    """Query specification for memory search.

    Encapsulates all query parameters for consistent handling
    across memory providers.

    Attributes:
        query: Search query text
        limit: Maximum results to return
        memory_types: Filter to specific memory types
        filters: Additional provider-specific filters
        min_relevance: Minimum relevance threshold
        session_id: Optional session context
        include_metadata: Whether to include full metadata
    """

    query: str
    limit: int = 20
    memory_types: Optional[list[MemoryType]] = None
    filters: Optional[dict[str, Any]] = None
    min_relevance: float = 0.0
    session_id: Optional[str] = None
    include_metadata: bool = True


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class MemoryProviderProtocol(Protocol):
    """Protocol for memory system adapters.

    Each memory system should implement this protocol to integrate
    with the unified coordinator. The protocol is intentionally
    minimal to allow diverse backend implementations.

    Example:
        class ConversationMemoryAdapter(MemoryProviderProtocol):
            @property
            def memory_type(self) -> MemoryType:
                return MemoryType.CONVERSATION

            async def search(self, query: MemoryQuery) -> List[MemoryResult]:
                # Query conversation store
                messages = await self._store.search(query.query)
                return [self._to_result(m) for m in messages]
    """

    @property
    @abstractmethod
    def memory_type(self) -> MemoryType:
        """The type of memory this provider handles.

        Returns:
            MemoryType enum value
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: MemoryQuery,
    ) -> list[MemoryResult]:
        """Search this memory system.

        Args:
            query: Search query specification

        Returns:
            List of memory results sorted by relevance
        """
        ...

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store a value in this memory system.

        Optional - not all memory systems support direct storage.

        Args:
            key: Unique identifier for the value
            value: Content to store
            metadata: Additional metadata
        """
        pass

    async def get(self, key: str) -> Optional[MemoryResult]:
        """Get a specific item by key.

        Optional - not all memory systems support direct access.

        Args:
            key: Unique identifier

        Returns:
            Memory result or None if not found
        """
        return None

    def is_available(self) -> bool:
        """Check if this provider is available and functional.

        Returns:
            True if provider is ready for queries
        """
        return True


@runtime_checkable
class RankingStrategyProtocol(Protocol):
    """Protocol for result ranking strategies.

    Different strategies can be used to rank results from
    multiple memory sources based on application needs.
    """

    @abstractmethod
    def rank(
        self,
        results: list[MemoryResult],
        query: MemoryQuery,
        limit: int,
    ) -> list[MemoryResult]:
        """Rank and deduplicate results.

        Args:
            results: Raw results from all providers
            query: Original query (for context)
            limit: Maximum results to return

        Returns:
            Ranked, deduplicated results
        """
        ...


# =============================================================================
# Ranking Strategies
# =============================================================================


class RelevanceRankingStrategy:
    """Rank results purely by relevance score."""

    def rank(
        self,
        results: list[MemoryResult],
        query: MemoryQuery,
        limit: int,
    ) -> list[MemoryResult]:
        """Rank by relevance score descending."""
        # Deduplicate by ID
        seen: set[str] = set()
        unique_results = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)

        # Sort by relevance
        sorted_results = sorted(
            unique_results,
            key=lambda r: r.relevance,
            reverse=True,
        )

        # Filter by min relevance
        if query.min_relevance > 0:
            sorted_results = [r for r in sorted_results if r.relevance >= query.min_relevance]

        return sorted_results[:limit]


class RecencyRankingStrategy:
    """Rank results by recency (most recent first)."""

    def rank(
        self,
        results: list[MemoryResult],
        query: MemoryQuery,
        limit: int,
    ) -> list[MemoryResult]:
        """Rank by timestamp descending (most recent first)."""
        # Deduplicate
        seen: set[str] = set()
        unique_results = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)

        # Sort by timestamp (None goes last)
        sorted_results = sorted(
            unique_results,
            key=lambda r: r.timestamp or 0,
            reverse=True,
        )

        return sorted_results[:limit]


class HybridRankingStrategy:
    """Combine relevance and recency with configurable weights."""

    def __init__(
        self,
        relevance_weight: float = 0.7,
        recency_weight: float = 0.3,
        recency_decay_hours: float = 24.0,
    ):
        """Initialize hybrid strategy.

        Args:
            relevance_weight: Weight for relevance score (0-1)
            recency_weight: Weight for recency score (0-1)
            recency_decay_hours: Hours after which recency score is 0.5
        """
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.recency_decay_hours = recency_decay_hours

    def _recency_score(self, timestamp: Optional[float]) -> float:
        """Calculate recency score with exponential decay."""
        if timestamp is None:
            return 0.5  # Default for missing timestamp

        age_hours = (time.time() - timestamp) / 3600
        # Exponential decay: score = 1.0 at age 0, 0.5 at decay_hours
        decay_rate = 0.693 / self.recency_decay_hours  # ln(2) / decay_hours
        return max(0.0, min(1.0, 2 ** (-decay_rate * age_hours)))

    def rank(
        self,
        results: list[MemoryResult],
        query: MemoryQuery,
        limit: int,
    ) -> list[MemoryResult]:
        """Rank by weighted combination of relevance and recency."""
        # Deduplicate
        seen: set[str] = set()
        unique_results = []
        for r in results:
            if r.id not in seen:
                seen.add(r.id)
                unique_results.append(r)

        # Calculate combined scores
        scored_results = []
        for r in unique_results:
            recency = self._recency_score(r.timestamp)
            combined = self.relevance_weight * r.relevance + self.recency_weight * recency
            scored_results.append((combined, r))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Extract results
        ranked = [r for _, r in scored_results]

        # Filter by min relevance
        if query.min_relevance > 0:
            ranked = [r for r in ranked if r.relevance >= query.min_relevance]

        return ranked[:limit]


# =============================================================================
# Unified Memory Coordinator
# =============================================================================


class UnifiedMemoryCoordinator:
    """Federated search across multiple memory systems.

    Provides a unified interface for querying diverse memory backends
    with configurable ranking and deduplication.

    Features:
    - Parallel search across providers
    - Configurable ranking strategies
    - Automatic deduplication
    - Memory type filtering
    - Provider health tracking

    Example:
        coordinator = UnifiedMemoryCoordinator()

        # Register providers
        coordinator.register_provider(ConversationMemoryAdapter(store))
        coordinator.register_provider(EntityMemoryAdapter(entity_mem))

        # Search all memory
        results = await coordinator.search_all("authentication")

        # Search specific types
        results = await coordinator.search_all(
            query="login function",
            memory_types=[MemoryType.ENTITY, MemoryType.CODE],
        )
    """

    def __init__(
        self,
        ranking_strategy: Optional[RankingStrategyProtocol] = None,
    ):
        """Initialize coordinator.

        Args:
            ranking_strategy: Strategy for ranking results (default: HybridRankingStrategy)
        """
        self._providers: dict[MemoryType, MemoryProviderProtocol] = {}
        self._ranking_strategy = ranking_strategy or HybridRankingStrategy()
        self._query_count = 0
        self._error_count = 0

    def register_provider(self, provider: MemoryProviderProtocol) -> None:
        """Register a memory provider.

        Args:
            provider: Provider implementing MemoryProviderProtocol

        Raises:
            ValueError: If provider for this type already registered
        """
        memory_type = provider.memory_type
        if memory_type in self._providers:
            logger.warning(f"Replacing existing provider for {memory_type.name}")
        self._providers[memory_type] = provider
        logger.info(f"Registered memory provider: {memory_type.name}")

    def unregister_provider(self, memory_type: MemoryType) -> bool:
        """Unregister a memory provider.

        Args:
            memory_type: Type of provider to remove

        Returns:
            True if provider was removed, False if not found
        """
        if memory_type in self._providers:
            del self._providers[memory_type]
            logger.info(f"Unregistered memory provider: {memory_type.name}")
            return True
        return False

    def get_provider(self, memory_type: MemoryType) -> Optional[MemoryProviderProtocol]:
        """Get a specific provider.

        Args:
            memory_type: Type of provider to get

        Returns:
            Provider or None if not registered
        """
        return self._providers.get(memory_type)

    def get_registered_types(self) -> list[MemoryType]:
        """Get list of registered memory types.

        Returns:
            List of registered MemoryType values
        """
        return list(self._providers.keys())

    def set_ranking_strategy(self, strategy: RankingStrategyProtocol) -> None:
        """Set the ranking strategy.

        Args:
            strategy: New ranking strategy to use
        """
        self._ranking_strategy = strategy

    async def search_all(
        self,
        query: str,
        limit: int = 20,
        memory_types: Optional[list[Any]] = None,
        session_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        min_relevance: float = 0.0,
    ) -> list[Any]:
        """Federated search across registered providers.

        Searches all matching providers in parallel and returns
        ranked, deduplicated results.

        Args:
            query: Search query
            limit: Maximum results to return
            memory_types: Filter to specific memory types (default: all)
            session_id: Optional session context
            filters: Additional provider-specific filters
            min_relevance: Minimum relevance threshold

        Returns:
            Ranked, deduplicated results from all sources
        """
        self._query_count += 1

        # Build query object
        memory_query = MemoryQuery(
            query=query,
            limit=limit * 2,  # Get more for deduplication
            memory_types=memory_types,
            filters=filters,
            min_relevance=min_relevance,
            session_id=session_id,
        )

        # Determine which providers to query
        target_types = memory_types or list(self._providers.keys())
        target_providers = [
            self._providers[t]
            for t in target_types
            if t in self._providers and self._providers[t].is_available()
        ]

        if not target_providers:
            logger.warning("No available providers for query")
            return []

        # Parallel search
        search_tasks = [
            self._search_provider(provider, memory_query) for provider in target_providers
        ]

        results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Flatten and filter errors
        all_results: list[MemoryResult] = []
        for i, results in enumerate(results_lists):
            if isinstance(results, Exception):
                self._error_count += 1
                logger.warning(f"Provider {target_providers[i].memory_type.name} failed: {results}")
            elif isinstance(results, list):
                all_results.extend(results)

        # Rank and deduplicate
        ranked_results = self._ranking_strategy.rank(all_results, memory_query, limit)

        logger.debug(
            f"Search '{query[:50]}...' returned {len(ranked_results)} results "
            f"from {len(target_providers)} providers"
        )

        return ranked_results

    async def _search_provider(
        self,
        provider: MemoryProviderProtocol,
        query: MemoryQuery,
    ) -> list[MemoryResult]:
        """Search a single provider with error handling."""
        try:
            return await provider.search(query)
        except Exception as e:
            logger.error(f"Error searching {provider.memory_type.name}: {e}")
            raise

    async def search_type(
        self,
        memory_type: MemoryType,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[MemoryResult]:
        """Search a specific memory type.

        Convenience method for single-type searches.

        Args:
            memory_type: Type of memory to search
            query: Search query
            limit: Maximum results
            **kwargs: Additional search parameters

        Returns:
            Results from the specified memory type
        """
        return await self.search_all(
            query=query,
            limit=limit,
            memory_types=[memory_type],
            **kwargs,
        )

    async def get(
        self,
        memory_type: MemoryType,
        key: str,
    ) -> Optional[MemoryResult]:
        """Get a specific item from a memory type.

        Args:
            memory_type: Type of memory
            key: Item identifier

        Returns:
            Memory result or None if not found
        """
        provider = self._providers.get(memory_type)
        if provider is None:
            return None
        return await provider.get(key)

    async def store(
        self,
        memory_type: MemoryType,
        key: str,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Store a value in a specific memory type.

        Args:
            memory_type: Type of memory
            key: Unique identifier
            value: Content to store
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        provider = self._providers.get(memory_type)
        if provider is None:
            logger.warning(f"No provider for {memory_type.name}")
            return False

        try:
            await provider.store(key, value, metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to store in {memory_type.name}: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics.

        Returns:
            Dict with query counts, error rates, provider status
        """
        return {
            "query_count": self._query_count,
            "error_count": self._error_count,
            "error_rate": (self._error_count / max(1, self._query_count)),
            "providers": {
                t.name: {
                    "available": p.is_available(),
                }
                for t, p in self._providers.items()
            },
        }


# =============================================================================
# Factory Functions
# =============================================================================


_coordinator_instance: Optional[UnifiedMemoryCoordinator] = None


def create_memory_coordinator(
    ranking_strategy: Optional[RankingStrategyProtocol] = None,
) -> UnifiedMemoryCoordinator:
    """Create a new memory coordinator.

    Args:
        ranking_strategy: Optional ranking strategy

    Returns:
        New UnifiedMemoryCoordinator instance
    """
    return UnifiedMemoryCoordinator(ranking_strategy=ranking_strategy)


def get_memory_coordinator() -> UnifiedMemoryCoordinator:
    """Get or create the singleton memory coordinator.

    Returns:
        Global UnifiedMemoryCoordinator instance
    """
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = create_memory_coordinator()
    return _coordinator_instance


def reset_memory_coordinator() -> None:
    """Reset the singleton coordinator (for testing)."""
    global _coordinator_instance
    _coordinator_instance = None


__all__ = [
    # Enums
    "MemoryType",
    # Data types
    "MemoryResult",
    "MemoryQuery",
    # Protocols
    "MemoryProviderProtocol",
    "RankingStrategyProtocol",
    # Strategies
    "RelevanceRankingStrategy",
    "RecencyRankingStrategy",
    "HybridRankingStrategy",
    # Coordinator
    "UnifiedMemoryCoordinator",
    # Factory
    "create_memory_coordinator",
    "get_memory_coordinator",
    "reset_memory_coordinator",
]
