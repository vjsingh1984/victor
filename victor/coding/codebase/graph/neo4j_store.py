# Neo4j-backed GraphStoreProtocol implementation (optional dependency).
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from victor.coding.codebase.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol


class Neo4jGraphStore(GraphStoreProtocol):
    """Placeholder Neo4j graph store (bolt/http drivers required)."""

    def __init__(self, _path: Path) -> None:
        try:
            import neo4j  # type: ignore  # pragma: no cover
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("neo4j python driver is required for Neo4jGraphStore") from exc
        # TODO: wire bolt URI/auth via settings and implement persistence
        self._not_ready()

    def _not_ready(self) -> None:
        raise NotImplementedError("Neo4jGraphStore is not implemented yet")

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        self._not_ready()

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        self._not_ready()

    async def get_neighbors(
        self, node_id: str, edge_types: Optional[Iterable[str]] = None, max_depth: int = 1
    ) -> List[GraphEdge]:
        self._not_ready()

    async def find_nodes(
        self, *, name: str | None = None, type: str | None = None, file: str | None = None
    ) -> List[GraphNode]:
        self._not_ready()

    async def delete_by_repo(self) -> None:
        self._not_ready()

    async def stats(self) -> Dict[str, Any]:
        self._not_ready()
