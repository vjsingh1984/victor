# SQLite-backed GraphStoreProtocol implementation.
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from victor.codebase.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER,
    lang TEXT,
    embedding_ref TEXT,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL,
    metadata TEXT,
    PRIMARY KEY (src, dst, type)
);
CREATE INDEX IF NOT EXISTS idx_nodes_type_name ON nodes(type, name);
CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file);
CREATE INDEX IF NOT EXISTS idx_edges_src_type ON edges(src, type);
CREATE INDEX IF NOT EXISTS idx_edges_dst_type ON edges(dst, type);
"""


class SqliteGraphStore(GraphStoreProtocol):
    """Embedded SQLite graph store for per-repo symbol graphs."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        # sqlite3 is threadsafe with check_same_thread=False when guarded; use async wrapper.
        self._lock = asyncio.Lock()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
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
                conn.commit()
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
                conn.commit()
            finally:
                conn.close()

    async def get_neighbors(
        self, node_id: str, edge_types: Optional[Iterable[str]] = None, max_depth: int = 1
    ) -> List[GraphEdge]:
        # For now, depth=1; deeper traversal can be added with recursive CTE.
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

    async def delete_by_repo(self) -> None:
        async with self._lock:
            conn = self._connect()
            try:
                conn.executescript("DELETE FROM edges; DELETE FROM nodes;")
                conn.commit()
            finally:
                conn.close()

    async def stats(self) -> Dict[str, Any]:
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("SELECT COUNT(*) FROM nodes")
                node_count = cur.fetchone()[0]
                cur = conn.execute("SELECT COUNT(*) FROM edges")
                edge_count = cur.fetchone()[0]
                return {"nodes": node_count, "edges": edge_count, "path": str(self.db_path)}
            finally:
                conn.close()
