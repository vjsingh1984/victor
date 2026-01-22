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

"""Semantic memory system for agentic AI.

This module implements a semantic memory system that stores and queries
factual knowledge with knowledge graph support.

Key Features:
- Store and query factual knowledge
- Vector similarity search for relevant knowledge
- Knowledge graph with relationships
- Knowledge updates and linking
- Citation tracking
- Conflict resolution and fact merging
- Integration with ISemanticSearch protocol

Architecture:
    SemanticMemory
    ├── Knowledge Storage (fact + embedding + metadata)
    ├── Similarity Search (via EmbeddingService)
    ├── Knowledge Graph (nodes + edges)
    ├── Relationship Management (links between knowledge)
    ├── Citation Tracking (source attribution)
    └── Conflict Resolution (merge contradictory facts)

Usage:
    from victor.agent.memory import SemanticMemory

    # Create semantic memory
    memory = SemanticMemory()

    # Store knowledge with source and confidence
    fact_id = memory.store_knowledge(
        "Python uses asyncio for asynchronous programming",
        confidence=0.95,
        source="official_docs"
    )

    # Query knowledge
    results = memory.query_knowledge("async concurrency", k=5)

    # Update confidence
    memory.update_knowledge(fact_id, new_confidence=0.98)

    # Get related facts
    related = memory.get_related_facts(fact_id)

    # Merge conflicting facts
    merged_id = memory.merge_facts([fact_id1, fact_id2])

    # Export as networkx graph
    import networkx as nx
    graph = memory.export_knowledge_graph()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeTriple:
    """Represents a knowledge graph triple (subject, predicate, object).

    This is a canonical representation for knowledge graph relationships,
    compatible with RDF and semantic web standards.

    Attributes:
        subject: Source entity or knowledge ID
        predicate: Relationship type (e.g., "related_to", "prerequisite", "implies")
        object: Target entity or knowledge ID
        weight: Strength of relationship (0-1, default 0.5)
        created_at: When the triple was created
        metadata: Additional metadata about the triple

    Example:
        triple = KnowledgeTriple(
            subject="Python",
            predicate="uses",
            object="asyncio",
            weight=0.9,
            metadata={"source": "official_docs"}
        )
    """

    subject: str
    predicate: str
    object: str
    weight: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """String representation of triple."""
        return f"KnowledgeTriple({self.subject} --[{self.predicate}:{self.weight:.2f}]--> {self.object})"

    def to_link(self, source_id: str, target_id: str) -> "KnowledgeLink":
        """Convert triple to KnowledgeLink.

        Args:
            source_id: Source knowledge ID
            target_id: Target knowledge ID

        Returns:
            KnowledgeLink instance
        """
        return KnowledgeLink(
            source_id=source_id,
            target_id=target_id,
            relation=self.predicate,
            weight=self.weight,
            created_at=self.created_at,
            metadata=self.metadata.copy(),
        )

    @classmethod
    def from_link(cls, link: "KnowledgeLink", subject: str, obj: str) -> "KnowledgeTriple":
        """Create triple from KnowledgeLink.

        Args:
            link: KnowledgeLink instance
            subject: Subject entity string
            obj: Object entity string

        Returns:
            KnowledgeTriple instance
        """
        return cls(
            subject=subject,
            predicate=link.relation,
            object=obj,
            weight=link.weight,
            created_at=link.created_at,
            metadata=link.metadata.copy(),
        )


@dataclass
class KnowledgeLink:
    """Represents a relationship between two knowledge items.

    Attributes:
        source_id: Source knowledge ID
        target_id: Target knowledge ID
        relation: Type of relationship (e.g., "related_to", "prerequisite", "implies")
        weight: Strength of relationship (0-1, default 0.5)
        created_at: When the link was created
        metadata: Additional metadata about the link
    """

    source_id: str
    target_id: str
    relation: str
    weight: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """String representation of link."""
        return f"KnowledgeLink({self.source_id[:8]}... --[{self.relation}]--> {self.target_id[:8]}...)"


@dataclass
class Knowledge:
    """Represents a single piece of semantic knowledge.

    Attributes:
        id: Unique knowledge identifier (UUID)
        fact: The factual knowledge (text)
        embedding: Vector embedding for similarity search
        metadata: Additional metadata (category, tags, importance, etc.)
        created_at: When the knowledge was created
        updated_at: When the knowledge was last updated
        links: List of links to other knowledge
        access_count: Number of times this knowledge was accessed
        confidence: Confidence in the knowledge (0-1, default 1.0)
        source: Source of the knowledge (e.g., "user", "documentation", "code")
        citations: List of citation IDs or references supporting this fact

    Example:
        knowledge = Knowledge(
            fact="Python uses asyncio for asynchronous programming",
            metadata={"category": "programming", "language": "Python"},
            source="official_docs",
            citations=["https://docs.python.org/3/library/asyncio.html"]
        )
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fact: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    links: List[KnowledgeLink] = field(default_factory=list)
    access_count: int = 0
    confidence: float = 1.0
    source: str = "unknown"
    citations: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """String representation of knowledge."""
        return f"Knowledge(id={self.id[:8]}..., fact={self.fact[:50]}..., confidence={self.confidence:.2f})"


@dataclass
class KnowledgeGraph:
    """Represents a knowledge graph with nodes and edges.

    Attributes:
        nodes: Dictionary of knowledge_id -> Knowledge
        edges: List of KnowledgeLink relationships
    """

    nodes: Dict[str, Knowledge] = field(default_factory=dict)
    edges: List[KnowledgeLink] = field(default_factory=list)

    def add_node(self, knowledge: Knowledge) -> None:
        """Add a node to the graph."""
        self.nodes[knowledge.id] = knowledge

    def add_edge(self, link: KnowledgeLink) -> None:
        """Add an edge to the graph."""
        self.edges.append(link)

    def get_neighbors(self, knowledge_id: str) -> List[Tuple[Knowledge, KnowledgeLink]]:
        """Get neighboring nodes and edges.

        Args:
            knowledge_id: Knowledge ID

        Returns:
            List of (neighbor_knowledge, link) tuples
        """
        neighbors = []
        for edge in self.edges:
            if edge.source_id == knowledge_id and edge.target_id in self.nodes:
                neighbors.append((self.nodes[edge.target_id], edge))
        return neighbors

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation.

        Returns:
            Dictionary with nodes and edges
        """
        return {
            "nodes": [
                {
                    "id": k.id,
                    "fact": k.fact,
                    "metadata": k.metadata,
                    "confidence": k.confidence,
                }
                for k in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "relation": e.relation,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            },
        }


class SemanticMemory:
    """Semantic memory system for storing and querying factual knowledge.

    This class provides a complete semantic memory implementation with:
    - Knowledge storage with embeddings
    - Vector similarity search
    - Knowledge graph construction
    - Relationship management
    - Knowledge updates

    Attributes:
        embedding_service: Service for generating embeddings
        max_knowledge: Maximum number of knowledge items to store (default: 10,000)

    Example:
        from victor.agent.memory import SemanticMemory

        memory = SemanticMemory(max_knowledge=5000)

        # Store knowledge
        fact_id = memory.store_knowledge(
            "Python uses asyncio for concurrency"
        )

        # Query knowledge
        results = memory.query_knowledge("async programming", k=5)

        # Link related knowledge
        other_id = memory.store_knowledge("asyncio is part of Python standard library")
        memory.link_knowledge(fact_id, other_id, "related_to")

        # Export knowledge graph
        graph = memory.export_knowledge_graph()
    """

    def __init__(
        self,
        embedding_service: Optional["EmbeddingService"] = None,
        max_knowledge: int = 10000,
    ):
        """Initialize semantic memory.

        Args:
            embedding_service: Service for generating embeddings (optional)
            max_knowledge: Maximum number of knowledge items to store
        """
        from victor.storage.embeddings.service import get_embedding_service

        self._embedding_service = embedding_service or get_embedding_service()
        self._max_knowledge = max_knowledge

        # Knowledge storage (id -> Knowledge)
        self._knowledge: Dict[str, Knowledge] = {}

        # Embedding storage for similarity search (id -> embedding)
        self._embeddings: Dict[str, np.ndarray] = {}

        # Knowledge graph
        self._graph = KnowledgeGraph()

        # Statistics
        self._total_knowledge_stored = 0

    @property
    def knowledge_count(self) -> int:
        """Get current number of stored knowledge items."""
        return len(self._knowledge)

    @property
    def total_stored(self) -> int:
        """Get total number of knowledge items ever stored."""
        return self._total_knowledge_stored

    async def store_knowledge(
        self,
        fact: str,
        confidence: float = 1.0,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        citations: Optional[List[str]] = None,
    ) -> str:
        """Store a piece of knowledge.

        Generates embedding for the fact and tracks citations.

        Args:
            fact: Factual knowledge (text)
            confidence: Confidence in the knowledge (0-1)
            source: Source of the knowledge (e.g., "user", "documentation", "code")
            metadata: Optional metadata (category, tags, etc.)
            citations: Optional list of citation IDs or URLs

        Returns:
            Knowledge ID

        Raises:
            ValueError: If fact is empty or confidence is out of range
        """
        if not fact or not fact.strip():
            raise ValueError("Fact cannot be empty")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        knowledge_id = str(uuid.uuid4())

        # Generate embedding
        try:
            embedding = await self._embedding_service.embed_text(fact)
        except Exception as e:
            logger.warning(f"Failed to generate embedding for knowledge: {e}")
            embedding = np.zeros(self._embedding_service.dimension, dtype=np.float32)

        # Create knowledge
        knowledge = Knowledge(
            id=knowledge_id,
            fact=fact,
            embedding=embedding,
            metadata=metadata or {},
            confidence=confidence,
            source=source,
            citations=citations or [],
        )

        # Store knowledge
        self._knowledge[knowledge_id] = knowledge
        self._embeddings[knowledge_id] = embedding
        self._graph.add_node(knowledge)
        self._total_knowledge_stored += 1

        # Enforce storage limit
        if len(self._knowledge) > self._max_knowledge:
            await self._evict_knowledge()

        logger.debug(f"Stored knowledge {knowledge_id[:8]}...: {fact[:50]}... (confidence={confidence:.2f})")

        return knowledge_id

    async def _evict_knowledge(self) -> None:
        """Evict low-confidence or least-accessed knowledge."""
        # Sort by (confidence, access_count)
        # Low confidence, rarely accessed first
        sorted_knowledge = sorted(
            self._knowledge.values(),
            key=lambda k: (k.confidence, k.access_count),
        )

        # Evict 10% of knowledge if over limit
        num_to_evict = max(
            1,
            len(self._knowledge) - self._max_knowledge + int(self._max_knowledge * 0.1),
        )

        for knowledge in sorted_knowledge[:num_to_evict]:
            self._remove_knowledge(knowledge.id)

        logger.info(f"Evicted {num_to_evict} knowledge items")

    def _remove_knowledge(self, knowledge_id: str) -> None:
        """Remove knowledge and its links.

        Args:
            knowledge_id: Knowledge ID to remove
        """
        if knowledge_id in self._knowledge:
            # Remove knowledge
            del self._knowledge[knowledge_id]
            del self._embeddings[knowledge_id]

            # Remove from graph
            if knowledge_id in self._graph.nodes:
                del self._graph.nodes[knowledge_id]

            # Remove edges
            self._graph.edges = [
                e for e in self._graph.edges if e.source_id != knowledge_id and e.target_id != knowledge_id
            ]

            logger.debug(f"Removed knowledge {knowledge_id[:8]}...")

    async def query_knowledge(
        self,
        query: str,
        k: int = 5,
        min_similarity: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
        filter_source: Optional[str] = None,
    ) -> List[Knowledge]:
        """Query knowledge by semantic similarity.

        Args:
            query: Natural language query
            k: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            filter_metadata: Optional metadata filters
            filter_source: Optional source filter (e.g., "official_docs")

        Returns:
            List of relevant Knowledge objects, ordered by similarity
        """
        if not self._knowledge:
            return []

        # Generate query embedding
        try:
            query_embedding = await self._embedding_service.embed_text(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
            return []

        # Compute similarities
        similarities = []
        for knowledge_id, knowledge_embedding in self._embeddings.items():
            knowledge = self._knowledge[knowledge_id]

            # Apply metadata filters
            if filter_metadata:
                if not all(knowledge.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            # Apply source filter
            if filter_source and knowledge.source != filter_source:
                continue

            similarity = float(
                self._embedding_service.cosine_similarity(query_embedding, knowledge_embedding)
            )
            if similarity >= min_similarity:
                similarities.append((knowledge_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top-k knowledge
        top_k = similarities[:k]
        results = []
        for knowledge_id, similarity in top_k:
            knowledge = self._knowledge[knowledge_id]
            knowledge.access_count += 1
            results.append(knowledge)

        logger.debug(f"Found {len(results)} knowledge items for query: {query[:50]}...")

        return results

    def get_knowledge(self, knowledge_id: str) -> Optional[Knowledge]:
        """Get specific knowledge by ID.

        Args:
            knowledge_id: Knowledge ID

        Returns:
            Knowledge if found, None otherwise
        """
        knowledge = self._knowledge.get(knowledge_id)
        if knowledge:
            knowledge.access_count += 1
        return knowledge

    async def update_knowledge(
        self,
        knowledge_id: str,
        new_confidence: Optional[float] = None,
    ) -> bool:
        """Update knowledge confidence score.

        Args:
            knowledge_id: Knowledge ID to update
            new_confidence: New confidence value (0-1)

        Returns:
            True if updated, False if not found or invalid confidence

        Raises:
            ValueError: If new_confidence is out of range
        """
        if knowledge_id not in self._knowledge:
            return False

        if new_confidence is not None and not 0.0 <= new_confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        knowledge = self._knowledge[knowledge_id]

        if new_confidence is not None:
            knowledge.confidence = new_confidence
            knowledge.updated_at = datetime.utcnow()
            logger.debug(f"Updated confidence for knowledge {knowledge_id[:8]}... to {new_confidence:.2f}")

        return True

    async def update_knowledge_fact(
        self,
        knowledge_id: str,
        new_fact: str,
        new_confidence: Optional[float] = None,
    ) -> bool:
        """Update knowledge fact text and optionally confidence.

        Regenerates embedding for the new fact.

        Args:
            knowledge_id: Knowledge ID to update
            new_fact: New fact text
            new_confidence: Optional new confidence value (0-1)

        Returns:
            True if updated, False if not found

        Raises:
            ValueError: If new_fact is empty or new_confidence is out of range
        """
        if knowledge_id not in self._knowledge:
            return False

        if not new_fact or not new_fact.strip():
            raise ValueError("Fact cannot be empty")

        if new_confidence is not None and not 0.0 <= new_confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        knowledge = self._knowledge[knowledge_id]

        # Update fact and timestamp
        knowledge.fact = new_fact
        knowledge.updated_at = datetime.utcnow()

        if new_confidence is not None:
            knowledge.confidence = new_confidence

        # Regenerate embedding
        try:
            embedding = await self._embedding_service.embed_text(new_fact)
            knowledge.embedding = embedding
            self._embeddings[knowledge_id] = embedding
        except Exception as e:
            logger.warning(f"Failed to regenerate embedding: {e}")

        logger.debug(f"Updated knowledge {knowledge_id[:8]}... with new fact: {new_fact[:50]}...")

        return True

    def link_knowledge(self, id1: str, id2: str, relation: str, weight: float = 0.5) -> bool:
        """Create a link between two knowledge items.

        Args:
            id1: Source knowledge ID
            id2: Target knowledge ID
            relation: Type of relationship
            weight: Strength of relationship (0-1)

        Returns:
            True if link created, False if knowledge not found
        """
        if id1 not in self._knowledge or id2 not in self._knowledge:
            return False

        # Check if link already exists
        for edge in self._graph.edges:
            if edge.source_id == id1 and edge.target_id == id2 and edge.relation == relation:
                # Update weight
                edge.weight = weight
                return True

        # Create new link
        link = KnowledgeLink(source_id=id1, target_id=id2, relation=relation, weight=weight)

        # Add to knowledge items
        self._knowledge[id1].links.append(link)

        # Add to graph
        self._graph.add_edge(link)

        logger.debug(f"Linked {id1[:8]}... to {id2[:8]}... with relation '{relation}'")

        return True

    async def add_relationship(self, triple: KnowledgeTriple) -> bool:
        """Add a relationship between knowledge items using a KnowledgeTriple.

        This method provides a more expressive API for creating knowledge graph
        relationships using the KnowledgeTriple dataclass, which includes
        metadata and supports semantic web standards.

        Args:
            triple: KnowledgeTriple containing subject, predicate, object, weight, and metadata

        Returns:
            True if relationship created, False if knowledge not found

        Raises:
            ValueError: If triple is invalid

        Example:
            triple = KnowledgeTriple(
                subject=fact_id_1,
                predicate="related_to",
                object=fact_id_2,
                weight=0.8,
                metadata={"source": "inference"}
            )
            await memory.add_relationship(triple)
        """
        if not triple.subject or not triple.object:
            raise ValueError("KnowledgeTriple must have non-empty subject and object")

        if triple.subject not in self._knowledge or triple.object not in self._knowledge:
            logger.warning(
                f"Cannot create relationship: one or both knowledge IDs not found "
                f"(subject={triple.subject[:8]}..., object={triple.object[:8]}...)"
            )
            return False

        # Convert triple to link and add to knowledge graph
        link = triple.to_link(triple.subject, triple.object)

        # Check if link already exists
        for edge in self._graph.edges:
            if (edge.source_id == link.source_id and
                edge.target_id == link.target_id and
                edge.relation == link.relation):
                # Update weight and metadata
                edge.weight = link.weight
                edge.metadata.update(link.metadata)
                logger.debug(f"Updated existing relationship: {triple}")
                return True

        # Add to knowledge items
        self._knowledge[triple.subject].links.append(link)

        # Add to graph
        self._graph.add_edge(link)

        logger.debug(
            f"Added relationship: {triple.subject[:8]}... --[{triple.predicate}:{triple.weight:.2f}]--> "
            f"{triple.object[:8]}..."
        )

        return True

    def get_related_knowledge(self, knowledge_id: str, max_depth: int = 1) -> List[Knowledge]:
        """Get knowledge related to the given knowledge ID.

        Follows links in the knowledge graph up to max_depth.

        Args:
            knowledge_id: Knowledge ID
            max_depth: Maximum link depth to follow (default: 1)

        Returns:
            List of related knowledge
        """
        if knowledge_id not in self._knowledge:
            return []

        visited: Set[str] = set()
        related: List[Knowledge] = []

        def dfs(current_id: str, depth: int) -> None:
            """Depth-first search for related knowledge."""
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)

            # Get neighbors
            neighbors = self._graph.get_neighbors(current_id)
            for neighbor, link in neighbors:
                if neighbor.id not in visited:
                    related.append(neighbor)
                    dfs(neighbor.id, depth + 1)

        dfs(knowledge_id, 0)

        logger.debug(f"Found {len(related)} related knowledge items for {knowledge_id[:8]}...")

        return related

    def get_related_facts(self, fact_id: str, relation_type: Optional[str] = None, max_depth: int = 1) -> List[Knowledge]:
        """Get facts related to the given fact ID, optionally filtering by relation type.

        This is an alias for get_related_knowledge with additional filtering.

        Args:
            fact_id: Knowledge/fact ID
            relation_type: Optional relation type to filter (e.g., "prerequisite", "related_to")
            max_depth: Maximum link depth to follow (default: 1)

        Returns:
            List of related Knowledge objects
        """
        if fact_id not in self._knowledge:
            return []

        visited: Set[str] = set()
        related: List[Knowledge] = []

        def dfs(current_id: str, depth: int) -> None:
            """Depth-first search for related knowledge."""
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)

            # Get neighbors
            neighbors = self._graph.get_neighbors(current_id)
            for neighbor, link in neighbors:
                if neighbor.id not in visited:
                    # Filter by relation type if specified
                    if relation_type is None or link.relation == relation_type:
                        related.append(neighbor)
                    dfs(neighbor.id, depth + 1)

        dfs(fact_id, 0)

        logger.debug(
            f"Found {len(related)} related facts for {fact_id[:8]}... "
            f"(relation={relation_type or 'any'})"
        )

        return related

    async def merge_facts(self, fact_ids: List[str], merge_strategy: str = "weighted_average") -> str:
        """Merge multiple facts into a single consolidated fact.

        This method combines contradictory or complementary facts into a single
        knowledge item with aggregated confidence and merged citations.

        Args:
            fact_ids: List of fact IDs to merge
            merge_strategy: Strategy for merging confidence scores:
                - "weighted_average": Weight by original confidence (default)
                - "max": Use maximum confidence
                - "min": Use minimum confidence
                - "average": Simple average

        Returns:
            ID of the merged knowledge item

        Raises:
            ValueError: If fact_ids is empty or contains invalid IDs
        """
        if not fact_ids:
            raise ValueError("Cannot merge empty list of facts")

        # Validate all fact IDs
        facts = []
        for fid in fact_ids:
            if fid not in self._knowledge:
                raise ValueError(f"Fact ID {fid} not found in knowledge base")
            facts.append(self._knowledge[fid])

        if len(facts) == 1:
            logger.debug("Only one fact provided, returning existing ID")
            return facts[0].id

        # Merge facts by combining their text
        merged_fact_text = self._merge_fact_texts(facts)

        # Calculate merged confidence based on strategy
        merged_confidence = self._merge_confidences(facts, merge_strategy)

        # Merge all citations
        merged_citations = list(set([cite for fact in facts for cite in fact.citations]))

        # Merge all sources
        merged_sources = list(set([fact.source for fact in facts]))
        merged_source = ", ".join(merged_sources) if merged_sources else "unknown"

        # Merge metadata
        merged_metadata: Dict[str, Any] = {}
        for fact in facts:
            merged_metadata.update(fact.metadata)
        merged_metadata["merged_from"] = fact_ids
        merged_metadata["merge_strategy"] = merge_strategy
        merged_metadata["merge_count"] = len(facts)

        # Store merged fact
        merged_id = await self.store_knowledge(
            fact=merged_fact_text,
            confidence=merged_confidence,
            source=merged_source,
            metadata=merged_metadata,
            citations=merged_citations,
        )

        # Create links from original facts to merged fact
        for fid in fact_ids:
            self.link_knowledge(fid, merged_id, "merged_into", weight=1.0)

        # Remove original facts
        for fid in fact_ids:
            if fid != merged_id:  # Don't remove if it's the same ID
                self._remove_knowledge(fid)

        logger.info(
            f"Merged {len(facts)} facts into {merged_id[:8]}... "
            f"(confidence={merged_confidence:.2f}, strategy={merge_strategy})"
        )

        return merged_id

    def _merge_fact_texts(self, facts: List[Knowledge]) -> str:
        """Merge fact texts into a consolidated statement.

        Args:
            facts: List of Knowledge objects to merge

        Returns:
            Merged fact text
        """
        if len(facts) == 1:
            return facts[0].fact

        # Check if facts are similar (may be contradictory)
        # For now, concatenate with "and" or merge semantically
        fact_texts = [f.fact for f in facts]

        # If facts are short, combine with semicolon
        if all(len(fact) < 100 for fact in fact_texts):
            return "; ".join(fact_texts)

        # For longer facts, create a merged statement
        return "Merged knowledge: " + " | ".join(fact_texts)

    def _merge_confidences(self, facts: List[Knowledge], strategy: str) -> float:
        """Merge confidence scores based on strategy.

        Args:
            facts: List of Knowledge objects
            strategy: Merge strategy

        Returns:
            Merged confidence score (0-1)
        """
        confidences = [f.confidence for f in facts]

        if strategy == "max":
            return max(confidences)
        elif strategy == "min":
            return min(confidences)
        elif strategy == "average":
            return sum(confidences) / len(confidences)
        elif strategy == "weighted_average":
            # Weight by access count (more accessed = more reliable)
            access_counts = [f.access_count for f in facts]
            total_access = sum(access_counts)

            if total_access == 0:
                return sum(confidences) / len(confidences)

            weighted_sum = sum(c * a for c, a in zip(confidences, access_counts))
            return weighted_sum / total_access
        else:
            logger.warning(f"Unknown merge strategy '{strategy}', using average")
            return sum(confidences) / len(confidences)

    def export_knowledge_graph(self) -> KnowledgeGraph:
        """Export the knowledge graph.

        Returns:
            KnowledgeGraph with all nodes and edges
        """
        return self._graph

    def export_networkx_graph(self) -> nx.DiGraph:
        """Export knowledge graph as NetworkX DiGraph.

        This creates a NetworkX directed graph with nodes as knowledge items
        and edges as relationships. Node attributes include fact, confidence,
        source, and metadata. Edge attributes include relation type and weight.

        Returns:
            networkx.DiGraph: NetworkX directed graph representation

        Example:
            import matplotlib.pyplot as plt

            graph = memory.export_networkx_graph()

            # Visualize
            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, with_labels=True, node_size=1000)
            plt.show()

            # Analyze
            print(f"Nodes: {graph.number_of_nodes()}")
            print(f"Edges: {graph.number_of_edges()}")
            print(f"Density: {nx.density(graph)}")

            # Find central nodes
            centrality = nx.degree_centrality(graph)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        """
        G = nx.DiGraph()

        # Add nodes with attributes
        for knowledge_id, knowledge in self._knowledge.items():
            G.add_node(
                knowledge_id,
                fact=knowledge.fact[:100] + "..." if len(knowledge.fact) > 100 else knowledge.fact,
                confidence=knowledge.confidence,
                source=knowledge.source,
                access_count=knowledge.access_count,
                created_at=knowledge.created_at.isoformat(),
                **knowledge.metadata,
            )

        # Add edges with attributes
        for link in self._graph.edges:
            G.add_edge(
                link.source_id,
                link.target_id,
                relation=link.relation,
                weight=link.weight,
                created_at=link.created_at.isoformat(),
            )

        logger.debug(
            f"Exported NetworkX graph with {G.number_of_nodes()} nodes "
            f"and {G.number_of_edges()} edges"
        )

        return G

    async def export_to_json(self, filepath: Union[str, Path]) -> None:
        """Export knowledge base to JSON file.

        Args:
            filepath: Path to JSON file for export

        Raises:
            IOError: If file cannot be written
        """
        filepath = Path(filepath)

        # Prepare data
        data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "statistics": self.get_statistics(),
            "knowledge": [
                {
                    "id": k.id,
                    "fact": k.fact,
                    "confidence": k.confidence,
                    "source": k.source,
                    "citations": k.citations,
                    "metadata": k.metadata,
                    "created_at": k.created_at.isoformat(),
                    "updated_at": k.updated_at.isoformat(),
                    "access_count": k.access_count,
                }
                for k in self._knowledge.values()
            ],
            "links": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "weight": e.weight,
                    "created_at": e.created_at.isoformat(),
                    "metadata": e.metadata,
                }
                for e in self._graph.edges
            ],
        }

        # Write to file
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(self._knowledge)} knowledge items to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export knowledge to JSON: {e}")
            raise IOError(f"Failed to export to {filepath}: {e}")

    async def import_from_json(self, filepath: Union[str, Path]) -> int:
        """Import knowledge base from JSON file.

        Args:
            filepath: Path to JSON file to import

        Returns:
            Number of knowledge items imported

        Raises:
            IOError: If file cannot be read
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Knowledge file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Import knowledge
            imported_count = 0
            for item in data.get("knowledge", []):
                knowledge_id = item["id"]

                # Create knowledge object
                knowledge = Knowledge(
                    id=knowledge_id,
                    fact=item["fact"],
                    confidence=item.get("confidence", 1.0),
                    source=item.get("source", "imported"),
                    citations=item.get("citations", []),
                    metadata=item.get("metadata", {}),
                    created_at=datetime.fromisoformat(item["created_at"]),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                    access_count=item.get("access_count", 0),
                )

                # Generate embedding
                try:
                    embedding = await self._embedding_service.embed_text(knowledge.fact)
                    knowledge.embedding = embedding
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for imported knowledge: {e}")
                    embedding = np.zeros(self._embedding_service.dimension, dtype=np.float32)

                # Store
                self._knowledge[knowledge_id] = knowledge
                self._embeddings[knowledge_id] = embedding
                self._graph.add_node(knowledge)
                imported_count += 1

            # Import links
            for item in data.get("links", []):
                link = KnowledgeLink(
                    source_id=item["source_id"],
                    target_id=item["target_id"],
                    relation=item["relation"],
                    weight=item.get("weight", 0.5),
                    created_at=datetime.fromisoformat(item["created_at"]),
                    metadata=item.get("metadata", {}),
                )

                # Add to knowledge items
                if link.source_id in self._knowledge:
                    self._knowledge[link.source_id].links.append(link)
                    self._graph.add_edge(link)

            logger.info(f"Imported {imported_count} knowledge items from {filepath}")

            return imported_count
        except Exception as e:
            logger.error(f"Failed to import knowledge from JSON: {e}")
            raise IOError(f"Failed to import from {filepath}: {e}")

    async def save_to_sqlite(self, filepath: Union[str, Path]) -> None:
        """Save knowledge base to SQLite database.

        Args:
            filepath: Path to SQLite database file

        Raises:
            IOError: If database cannot be written
        """
        filepath = Path(filepath)

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(filepath))
            cursor = conn.cursor()

            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    fact TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL,
                    metadata TEXT,
                    citations TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (source_id) REFERENCES knowledge (id),
                    FOREIGN KEY (target_id) REFERENCES knowledge (id)
                )
            """)

            # Clear existing data
            cursor.execute("DELETE FROM knowledge")
            cursor.execute("DELETE FROM links")

            # Insert knowledge
            for knowledge in self._knowledge.values():
                cursor.execute(
                    """
                    INSERT INTO knowledge
                    (id, fact, confidence, source, created_at, updated_at, access_count, metadata, citations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        knowledge.id,
                        knowledge.fact,
                        knowledge.confidence,
                        knowledge.source,
                        knowledge.created_at.isoformat(),
                        knowledge.updated_at.isoformat(),
                        knowledge.access_count,
                        json.dumps(knowledge.metadata),
                        json.dumps(knowledge.citations),
                    ),
                )

            # Insert links
            for link in self._graph.edges:
                cursor.execute(
                    """
                    INSERT INTO links
                    (source_id, target_id, relation, weight, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        link.source_id,
                        link.target_id,
                        link.relation,
                        link.weight,
                        link.created_at.isoformat(),
                        json.dumps(link.metadata),
                    ),
                )

            conn.commit()
            conn.close()

            logger.info(f"Saved knowledge base to SQLite database: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save to SQLite: {e}")
            raise IOError(f"Failed to save to {filepath}: {e}")

    async def load_from_sqlite(self, filepath: Union[str, Path]) -> int:
        """Load knowledge base from SQLite database.

        Args:
            filepath: Path to SQLite database file

        Returns:
            Number of knowledge items loaded

        Raises:
            IOError: If database cannot be read
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Database file not found: {filepath}")

        try:
            conn = sqlite3.connect(str(filepath))
            cursor = conn.cursor()

            # Load knowledge
            cursor.execute("SELECT * FROM knowledge")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            loaded_count = 0
            for row in rows:
                item = dict(zip(columns, row))

                knowledge_id = item["id"]
                knowledge = Knowledge(
                    id=knowledge_id,
                    fact=item["fact"],
                    confidence=item["confidence"],
                    source=item["source"],
                    metadata=json.loads(item["metadata"]) if item["metadata"] else {},
                    citations=json.loads(item["citations"]) if item["citations"] else [],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                    access_count=item["access_count"],
                )

                # Generate embedding
                try:
                    embedding = await self._embedding_service.embed_text(knowledge.fact)
                    knowledge.embedding = embedding
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for loaded knowledge: {e}")
                    embedding = np.zeros(self._embedding_service.dimension, dtype=np.float32)

                # Store
                self._knowledge[knowledge_id] = knowledge
                self._embeddings[knowledge_id] = embedding
                self._graph.add_node(knowledge)
                loaded_count += 1

            # Load links
            cursor.execute("SELECT * FROM links")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            for row in rows:
                item = dict(zip(columns, row))

                link = KnowledgeLink(
                    source_id=item["source_id"],
                    target_id=item["target_id"],
                    relation=item["relation"],
                    weight=item["weight"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    metadata=json.loads(item["metadata"]) if item["metadata"] else {},
                )

                # Add to knowledge items
                if link.source_id in self._knowledge:
                    self._knowledge[link.source_id].links.append(link)
                    self._graph.add_edge(link)

            conn.close()

            logger.info(f"Loaded {loaded_count} knowledge items from SQLite database: {filepath}")

            return loaded_count
        except Exception as e:
            logger.error(f"Failed to load from SQLite: {e}")
            raise IOError(f"Failed to load from {filepath}: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with statistics including knowledge count, confidence
            distribution, graph metrics, source breakdown, and citation stats
        """
        if not self._knowledge:
            return {
                "knowledge_count": 0,
                "total_stored": self._total_knowledge_stored,
                "graph_nodes": 0,
                "graph_edges": 0,
            }

        confidences = [k.confidence for k in self._knowledge.values()]
        access_counts = [k.access_count for k in self._knowledge.values()]
        sources = [k.source for k in self._knowledge.values()]
        citation_counts = [len(k.citations) for k in self._knowledge.values()]

        # Calculate graph statistics
        link_counts = [len(k.links) for k in self._knowledge.values()]

        # Source breakdown
        source_counts: Dict[str, int] = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            "knowledge_count": len(self._knowledge),
            "total_stored": self._total_knowledge_stored,
            "max_knowledge": self._max_knowledge,
            "utilization": len(self._knowledge) / self._max_knowledge,
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "avg_access_count": sum(access_counts) / len(access_counts),
            "max_access_count": max(access_counts),
            "graph_nodes": len(self._graph.nodes),
            "graph_edges": len(self._graph.edges),
            "avg_links_per_node": sum(link_counts) / len(link_counts) if link_counts else 0,
            "oldest_knowledge": min(k.created_at for k in self._knowledge.values()).isoformat(),
            "newest_knowledge": max(k.created_at for k in self._knowledge.values()).isoformat(),
            "source_breakdown": source_counts,
            "total_citations": sum(citation_counts),
            "avg_citations_per_fact": sum(citation_counts) / len(citation_counts) if citation_counts else 0,
            "facts_with_citations": sum(1 for c in citation_counts if c > 0),
        }

    async def clear(self) -> None:
        """Clear all knowledge from memory."""
        count = len(self._knowledge)
        self._knowledge.clear()
        self._embeddings.clear()
        self._graph = KnowledgeGraph()
        logger.info(f"Cleared {count} knowledge items from memory")


def create_semantic_memory(
    embedding_service: Optional["EmbeddingService"] = None,
    max_knowledge: int = 10000,
) -> SemanticMemory:
    """Factory function to create a SemanticMemory instance.

    Args:
        embedding_service: Optional embedding service
        max_knowledge: Maximum number of knowledge items to store

    Returns:
        SemanticMemory instance
    """
    return SemanticMemory(
        embedding_service=embedding_service,
        max_knowledge=max_knowledge,
    )
