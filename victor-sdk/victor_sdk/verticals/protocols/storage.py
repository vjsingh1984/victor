"""Storage protocol definitions for external verticals.

Promoted from victor.storage.graph and victor.storage.vector_stores
so external verticals can type-hint against these protocols without
importing from victor-ai internals.

All types are plain dataclasses with zero dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

# =============================================================================
# Graph Storage Data Types
# =============================================================================


@dataclass
class GraphNodeData:
    """A code symbol node in the graph.

    Mirrors victor.storage.graph.protocol.GraphNode as a plain dataclass.
    """

    node_id: str
    type: str  # function, class, file, module
    name: str
    file: str
    line: Optional[int] = None
    end_line: Optional[int] = None
    lang: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_id: Optional[str] = None
    embedding_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdgeData:
    """A directed edge between graph nodes.

    Mirrors victor.storage.graph.protocol.GraphEdge.
    """

    src: str
    dst: str
    type: str  # CALLS, REFERENCES, CONTAINS, INHERITS
    weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Embedding/Vector Storage Data Types
# =============================================================================


@dataclass
class EmbeddingSearchResultData:
    """A single embedding search result.

    Mirrors victor.storage.vector_stores.base.EmbeddingSearchResult.
    """

    file_path: str
    content: str
    score: float
    symbol_name: Optional[str] = None
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfigData:
    """Configuration for embedding/vector store.

    Subset of victor.storage.vector_stores.EmbeddingConfig (no secrets).
    """

    vector_store: str = "lancedb"
    persist_directory: Optional[str] = None
    distance_metric: str = "cosine"
    embedding_model_type: str = "sentence_transformer"
    embedding_model_name: str = "all-MiniLM-L6-v2"


# =============================================================================
# Storage Protocols
# =============================================================================


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Protocol for graph store implementations.

    External verticals should type-hint against this instead of
    importing from victor.storage.graph.protocol.
    """

    async def upsert_nodes(self, nodes: Iterable[Any]) -> None: ...

    async def upsert_edges(self, edges: Iterable[Any]) -> None: ...

    async def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[Any]: ...

    async def find_nodes(self, **kwargs: Any) -> List[Any]: ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations.

    External verticals should type-hint against this instead of
    importing from victor.storage.vector_stores.
    """

    async def index_document(
        self,
        doc_id: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    async def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Any]: ...

    async def delete_document(self, doc_id: str) -> None: ...


@runtime_checkable
class EmbeddingServiceProtocol(Protocol):
    """Protocol for embedding service implementations."""

    async def embed_text(self, text: str) -> List[float]: ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


__all__ = [
    # Data types
    "GraphNodeData",
    "GraphEdgeData",
    "EmbeddingSearchResultData",
    "EmbeddingConfigData",
    # Protocols
    "GraphStoreProtocol",
    "VectorStoreProtocol",
    "EmbeddingServiceProtocol",
]
