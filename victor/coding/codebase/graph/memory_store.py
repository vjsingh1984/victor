# Simple in-memory GraphStoreProtocol implementation (non-persistent).
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from victor.coding.codebase.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol


class MemoryGraphStore(GraphStoreProtocol):
    """In-memory graph store useful for testing or ephemeral runs."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[tuple[str, str, str], GraphEdge] = {}

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for n in nodes:
            self._nodes[n.node_id] = n

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        for e in edges:
            self._edges[(e.src, e.dst, e.type)] = e

    async def get_neighbors(
        self, node_id: str, edge_types: Optional[Iterable[str]] = None, max_depth: int = 1
    ) -> List[GraphEdge]:
        types = set(edge_types) if edge_types else None
        return [
            edge
            for (src, _dst, etype), edge in self._edges.items()
            if src == node_id and (types is None or etype in types)
        ]

    async def find_nodes(
        self, *, name: str | None = None, type: str | None = None, file: str | None = None
    ) -> List[GraphNode]:
        results: List[GraphNode] = []
        for node in self._nodes.values():
            if name is not None and node.name != name:
                continue
            if type is not None and node.type != type:
                continue
            if file is not None and node.file != file:
                continue
            results.append(node)
        return results

    async def delete_by_repo(self) -> None:
        self._nodes.clear()
        self._edges.clear()

    async def stats(self) -> Dict[str, Any]:
        return {"nodes": len(self._nodes), "edges": len(self._edges), "path": ":memory:"}
