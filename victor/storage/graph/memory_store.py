# Simple in-memory GraphStoreProtocol implementation (non-persistent).
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from victor.storage.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol


class MemoryGraphStore(GraphStoreProtocol):
    """In-memory graph store useful for testing or ephemeral runs."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[tuple[str, str, str], GraphEdge] = {}
        self._file_mtimes: Dict[str, float] = {}

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

    async def search_symbols(
        self, query: str, *, limit: int = 20, symbol_types: Iterable[str] | None = None
    ) -> List[GraphNode]:
        """Full-text search across symbol names, signatures, bodies, and docstrings."""
        query_lower = query.lower()
        results: List[GraphNode] = []
        for node in self._nodes.values():
            if symbol_types and node.type not in symbol_types:
                continue
            if (
                query_lower in node.name.lower()
                or (node.signature and query_lower in node.signature.lower())
                or (node.docstring and query_lower in node.docstring.lower())
            ):
                results.append(node)
                if len(results) >= limit:
                    break
        return results

    async def get_node_by_id(self, node_id: str) -> GraphNode | None:
        """Get a single node by its ID."""
        return self._nodes.get(node_id)

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get all symbols in a specific file."""
        return [node for node in self._nodes.values() if node.file == file]

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        """Record file modification time for staleness tracking."""
        self._file_mtimes[file] = mtime

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        """Get files that have changed since last index."""
        stale: List[str] = []
        for file, mtime in file_mtimes.items():
            if file not in self._file_mtimes or self._file_mtimes[file] < mtime:
                stale.append(file)
        return stale

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes and edges for a specific file (for incremental reindex)."""
        # Delete nodes for this file
        node_ids_to_delete = [node_id for node_id, node in self._nodes.items() if node.file == file]
        for node_id in node_ids_to_delete:
            del self._nodes[node_id]

        # Delete edges connected to these nodes
        edges_to_delete = [
            (src, dst, etype)
            for (src, dst, etype) in self._edges
            if src in node_ids_to_delete or dst in node_ids_to_delete
        ]
        for edge_key in edges_to_delete:
            del self._edges[edge_key]

    async def delete_by_repo(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._file_mtimes.clear()

    async def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "path": ":memory:",
            "files": len(self._file_mtimes),
        }

    async def get_all_edges(self) -> List[GraphEdge]:
        """Get all edges in the graph (bulk retrieval for loading into memory)."""
        return list(self._edges.values())
