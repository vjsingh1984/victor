# Graph store protocol for per-repo symbol graphs
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Protocol


@dataclass
class GraphNode:
    """Represents a code symbol with metadata (body read from file via line numbers)."""

    node_id: str  # stable id, e.g., symbol hash
    type: str  # e.g., function, class, file, module
    name: str
    file: str
    line: int | None = None
    end_line: int | None = None  # end line - use with line to read body from file
    lang: str | None = None
    signature: str | None = None  # function/method signature
    docstring: str | None = None  # extracted docstring
    parent_id: str | None = None  # for nested symbols (methods in classes)
    embedding_ref: str | None = None  # key to vector store entry
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Directed edge between nodes."""

    src: str
    dst: str
    type: str  # e.g., CALLS, REFERENCES, CONTAINS, INHERITS
    weight: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphStoreProtocol(Protocol):
    """Interface for pluggable graph stores."""

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        ...

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        ...

    async def get_neighbors(
        self, node_id: str, edge_types: Iterable[str] | None = None, max_depth: int = 1
    ) -> List[GraphEdge]:
        ...

    async def find_nodes(
        self, *, name: str | None = None, type: str | None = None, file: str | None = None
    ) -> List[GraphNode]:
        ...

    async def search_symbols(
        self, query: str, *, limit: int = 20, symbol_types: Iterable[str] | None = None
    ) -> List[GraphNode]:
        """Full-text search across symbol names, signatures, bodies, and docstrings."""
        ...

    async def get_node_by_id(self, node_id: str) -> GraphNode | None:
        """Get a single node by its ID."""
        ...

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get all symbols in a specific file."""
        ...

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        """Record file modification time for staleness tracking."""
        ...

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        """Get files that have changed since last index."""
        ...

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes and edges for a specific file (for incremental reindex)."""
        ...

    async def delete_by_repo(self) -> None:
        """Clear current repo graph (per-repo store)."""
        ...

    async def stats(self) -> Dict[str, Any]:
        ...

    async def get_all_edges(self) -> List[GraphEdge]:
        """Get all edges in the graph (bulk retrieval for loading into memory)."""
        ...
