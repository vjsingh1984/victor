# Graph store protocol for per-repo symbol graphs
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Protocol


@dataclass
class GraphNode:
    """Represents a code symbol or file."""

    node_id: str  # stable id, e.g., symbol hash
    type: str  # e.g., function, class, file, module
    name: str
    file: str
    line: int | None = None
    lang: str | None = None
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

    async def delete_by_repo(self) -> None:
        """Clear current repo graph (per-repo store)."""
        ...

    async def stats(self) -> Dict[str, Any]:
        ...
