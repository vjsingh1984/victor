# DuckDB-backed GraphStoreProtocol implementation (optional dependency).
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from victor.storage.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol


class DuckDBGraphStore(GraphStoreProtocol):
    """Embedded DuckDB graph store for per-repo symbol graphs.

    Uses a single-file DuckDB database with simple tables for nodes and edges.
    """

    def __init__(self, db_path: Path) -> None:
        try:
            import duckdb
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("duckdb must be installed to use DuckDBGraphStore") from exc

        self.duckdb = duckdb
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._ensure_schema()

    def _connect(self) -> Any:
        return self.duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    type TEXT,
                    name TEXT,
                    file TEXT,
                    line BIGINT,
                    lang TEXT,
                    embedding_ref TEXT,
                    metadata TEXT
                );
                CREATE TABLE IF NOT EXISTS edges (
                    src TEXT,
                    dst TEXT,
                    type TEXT,
                    weight DOUBLE,
                    metadata TEXT,
                    PRIMARY KEY (src, dst, type)
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_type_name ON nodes(type, name);
                CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file);
                CREATE INDEX IF NOT EXISTS idx_edges_src_type ON edges(src, type);
                CREATE INDEX IF NOT EXISTS idx_edges_dst_type ON edges(dst, type);
                """
            )
        finally:
            conn.close()

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        rows = [
            (
                n.node_id,
                n.type,
                n.name,
                n.file,
                n.line,
                n.lang,
                n.embedding_ref,
                json.dumps(n.metadata),
            )
            for n in nodes
        ]
        if not rows:
            return
        async with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    """
                    INSERT INTO nodes(node_id, type, name, file, line, lang, embedding_ref, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        type=excluded.type,
                        name=excluded.name,
                        file=excluded.file,
                        line=excluded.line,
                        lang=excluded.lang,
                        embedding_ref=excluded.embedding_ref,
                        metadata=excluded.metadata
                    """,
                    rows,
                )
            finally:
                conn.close()

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        rows = [
            (
                e.src,
                e.dst,
                e.type,
                e.weight,
                json.dumps(e.metadata),
            )
            for e in edges
        ]
        if not rows:
            return
        async with self._lock:
            conn = self._connect()
            try:
                conn.executemany(
                    """
                    INSERT INTO edges(src, dst, type, weight, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(src, dst, type) DO UPDATE SET
                        weight=excluded.weight,
                        metadata=excluded.metadata
                    """,
                    rows,
                )
            finally:
                conn.close()

    async def get_neighbors(
        self, node_id: str, edge_types: Optional[Iterable[str]] = None, max_depth: int = 1
    ) -> List[GraphEdge]:
        params: list[Any] = [node_id]
        type_clause = ""
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            type_clause = f" AND type IN ({placeholders})"
            params.extend(edge_types)
        query = f"SELECT src, dst, type, weight, metadata FROM edges WHERE src=?{type_clause}"
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(query, params)
                return [
                    GraphEdge(
                        src=row[0],
                        dst=row[1],
                        type=row[2],
                        weight=row[3],
                        metadata=json.loads(row[4]) if row[4] else {},
                    )
                    for row in cur.fetchall()
                ]
            finally:
                conn.close()

    async def find_nodes(
        self, *, name: str | None = None, type: str | None = None, file: str | None = None
    ) -> List[GraphNode]:
        clauses = []
        params: list[Any] = []
        if name:
            clauses.append("name = ?")
            params.append(name)
        if type:
            clauses.append("type = ?")
            params.append(type)
        if file:
            clauses.append("file = ?")
            params.append(file)
        where = " AND ".join(clauses) if clauses else "1=1"
        query = f"SELECT node_id, type, name, file, line, lang, embedding_ref, metadata FROM nodes WHERE {where}"
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(query, params)
                return [
                    GraphNode(
                        node_id=row[0],
                        type=row[1],
                        name=row[2],
                        file=row[3],
                        line=row[4],
                        lang=row[5],
                        embedding_ref=row[6],
                        metadata=json.loads(row[7]) if row[7] else {},
                    )
                    for row in cur.fetchall()
                ]
            finally:
                conn.close()

    async def search_symbols(
        self, query: str, *, limit: int = 20, symbol_types: Iterable[str] | None = None
    ) -> List[GraphNode]:
        """Full-text search across symbol names, signatures, bodies, and docstrings."""
        clauses = ["name LIKE ?"]
        params: list[Any] = [f"%{query}%"]
        if symbol_types:
            placeholders = ",".join("?" for _ in symbol_types)
            clauses.append(f"type IN ({placeholders})")
            params.extend(symbol_types)

        where = " AND ".join(clauses)
        query_str = f"SELECT node_id, type, name, file, line, lang, embedding_ref, metadata FROM nodes WHERE {where} LIMIT ?"
        params.append(limit)

        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(query_str, params)
                return [
                    GraphNode(
                        node_id=row[0],
                        type=row[1],
                        name=row[2],
                        file=row[3],
                        line=row[4],
                        lang=row[5],
                        embedding_ref=row[6],
                        metadata=json.loads(row[7]) if row[7] else {},
                    )
                    for row in cur.fetchall()
                ]
            finally:
                conn.close()

    async def get_node_by_id(self, node_id: str) -> GraphNode | None:
        """Get a single node by its ID."""
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT node_id, type, name, file, line, lang, embedding_ref, metadata FROM nodes WHERE node_id=?",
                    [node_id]
                )
                row = cur.fetchone()
                if not row:
                    return None
                return GraphNode(
                    node_id=row[0],
                    type=row[1],
                    name=row[2],
                    file=row[3],
                    line=row[4],
                    lang=row[5],
                    embedding_ref=row[6],
                    metadata=json.loads(row[7]) if row[7] else {},
                )
            finally:
                conn.close()

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get all symbols in a specific file."""
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT node_id, type, name, file, line, lang, embedding_ref, metadata FROM nodes WHERE file=?",
                    [file]
                )
                return [
                    GraphNode(
                        node_id=row[0],
                        type=row[1],
                        name=row[2],
                        file=row[3],
                        line=row[4],
                        lang=row[5],
                        embedding_ref=row[6],
                        metadata=json.loads(row[7]) if row[7] else {},
                    )
                    for row in cur.fetchall()
                ]
            finally:
                conn.close()

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        """Record file modification time for staleness tracking."""
        # For DuckDB, we'd need to create a file_mtimes table
        # For now, we'll skip this implementation
        pass

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        """Get files that have changed since last index."""
        # For DuckDB, we'd need to check against file_mtimes table
        # For now, return all files as potentially stale
        return list(file_mtimes.keys())

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes and edges for a specific file (for incremental reindex)."""
        async with self._lock:
            conn = self._connect()
            try:
                # Delete nodes for this file
                conn.execute("DELETE FROM edges WHERE src IN (SELECT node_id FROM nodes WHERE file=?)", [file])
                conn.execute("DELETE FROM edges WHERE dst IN (SELECT node_id FROM nodes WHERE file=?)", [file])
                conn.execute("DELETE FROM nodes WHERE file=?", [file])
            finally:
                conn.close()

    async def delete_by_repo(self) -> None:
        async with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM edges")
                conn.execute("DELETE FROM nodes")
            finally:
                conn.close()

    async def stats(self) -> Dict[str, Any]:
        async with self._lock:
            conn = self._connect()
            try:
                node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
                edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "path": str(self.db_path),
                    "backend": "duckdb",
                }
            finally:
                conn.close()

    async def get_all_edges(self) -> List[GraphEdge]:
        """Get all edges in the graph (bulk retrieval for loading into memory)."""
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("SELECT src, dst, type, weight, metadata FROM edges")
                return [
                    GraphEdge(
                        src=row[0],
                        dst=row[1],
                        type=row[2],
                        weight=row[3],
                        metadata=json.loads(row[4]) if row[4] else {},
                    )
                    for row in cur.fetchall()
                ]
            finally:
                conn.close()
