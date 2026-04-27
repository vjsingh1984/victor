# Simple in-memory GraphStoreProtocol implementation (non-persistent).
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from victor.storage.graph.protocol import (
    GraphEdge,
    GraphNode,
    GraphStoreProtocol,
    GraphTraversalDirection,
)


class MemoryGraphStore(GraphStoreProtocol):
    """In-memory graph store useful for testing or ephemeral runs."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[tuple[str, str, str], GraphEdge] = {}
        self._file_mtimes: Dict[str, float] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for n in nodes:
            self._nodes[n.node_id] = n

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        for e in edges:
            self._edges[(e.src, e.dst, e.type)] = e

    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[Iterable[str]] = None,
        *,
        direction: GraphTraversalDirection = "both",
        max_depth: int = 1,
    ) -> List[GraphEdge]:
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"Unsupported graph traversal direction: {direction}")
        if max_depth < 1:
            return []

        allowed_types = set(edge_types) if edge_types else None
        frontier = {node_id}
        visited_nodes = {node_id}
        seen_edges: Dict[tuple[str, str, str], GraphEdge] = {}

        for _depth in range(max_depth):
            next_frontier: set[str] = set()
            for edge in self._edges.values():
                if allowed_types is not None and edge.type not in allowed_types:
                    continue

                traversed = False
                if direction in {"out", "both"} and edge.src in frontier:
                    traversed = True
                    next_frontier.add(edge.dst)
                if direction in {"in", "both"} and edge.dst in frontier:
                    traversed = True
                    next_frontier.add(edge.src)

                if traversed:
                    seen_edges[(edge.src, edge.dst, edge.type)] = edge

            next_frontier -= visited_nodes
            if not next_frontier:
                break
            visited_nodes.update(next_frontier)
            frontier = next_frontier

        return sorted(seen_edges.values(), key=lambda edge: (edge.src, edge.dst, edge.type))

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
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

    async def search_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        symbol_types: Optional[Iterable[str]] = None,
    ) -> List[GraphNode]:
        query_lower = query.lower()
        allowed_types = set(symbol_types) if symbol_types else None
        matches: List[GraphNode] = []
        for node in self._nodes.values():
            if allowed_types is not None and node.type not in allowed_types:
                continue
            haystacks = [
                node.name,
                node.signature or "",
                node.docstring or "",
            ]
            if any(query_lower in value.lower() for value in haystacks if value):
                matches.append(node)
        matches.sort(key=lambda node: (node.file, node.line or 0, node.name))
        return matches[:limit]

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    async def get_all_nodes(self) -> List[GraphNode]:
        return sorted(self._nodes.values(), key=lambda node: (node.file, node.line or 0, node.name))

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        return sorted(
            [node for node in self._nodes.values() if node.file == file],
            key=lambda node: (node.line or 0, node.name),
        )

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        self._file_mtimes[file] = mtime

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        return [
            file
            for file, current_mtime in file_mtimes.items()
            if self._file_mtimes.get(file, float("-inf")) < current_mtime
        ]

    async def delete_by_file(self, file: str) -> None:
        node_ids = {node.node_id for node in self._nodes.values() if node.file == file}
        self._nodes = {
            node_id: node for node_id, node in self._nodes.items() if node_id not in node_ids
        }
        self._edges = {
            key: edge
            for key, edge in self._edges.items()
            if edge.src not in node_ids and edge.dst not in node_ids
        }
        self._file_mtimes.pop(file, None)

    async def delete_by_repo(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._file_mtimes.clear()

    async def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "path": ":memory:",
        }

    async def get_all_edges(self) -> List[GraphEdge]:
        return sorted(self._edges.values(), key=lambda edge: (edge.src, edge.dst, edge.type))
