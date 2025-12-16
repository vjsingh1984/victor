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

-- Core symbol/node table (body read from file via line numbers, not stored here)
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    file TEXT NOT NULL,
    line INTEGER,
    end_line INTEGER,
    lang TEXT,
    signature TEXT,
    docstring TEXT,
    parent_id TEXT,
    embedding_ref TEXT,
    metadata TEXT,
    FOREIGN KEY (parent_id) REFERENCES nodes(node_id)
);

-- Graph edges table
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL,
    metadata TEXT,
    PRIMARY KEY (src, dst, type)
);

-- File modification times for staleness tracking
CREATE TABLE IF NOT EXISTS file_mtimes (
    file TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    indexed_at REAL NOT NULL
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_nodes_type_name ON nodes(type, name);
CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_src_type ON edges(src, type);
CREATE INDEX IF NOT EXISTS idx_edges_dst_type ON edges(dst, type);
CREATE INDEX IF NOT EXISTS idx_file_mtimes_mtime ON file_mtimes(mtime);
"""

# FTS5 schema for full-text search on symbols (body not stored - read from file)
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    node_id,
    name,
    signature,
    docstring,
    content='nodes',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, node_id, name, signature, docstring)
    VALUES (new.rowid, new.node_id, new.name, new.signature, new.docstring);
END;

CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, node_id, name, signature, docstring)
    VALUES ('delete', old.rowid, old.node_id, old.name, old.signature, old.docstring);
END;

CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, node_id, name, signature, docstring)
    VALUES ('delete', old.rowid, old.node_id, old.name, old.signature, old.docstring);
    INSERT INTO nodes_fts(rowid, node_id, name, signature, docstring)
    VALUES (new.rowid, new.node_id, new.name, new.signature, new.docstring);
END;
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
            # Add new columns if upgrading from older schema
            self._migrate_schema(conn)
            # Create FTS5 table separately (handles IF NOT EXISTS)
            try:
                conn.executescript(FTS_SCHEMA)
            except sqlite3.OperationalError:
                # FTS5 might not be available on all SQLite builds
                pass
            conn.commit()
        finally:
            conn.close()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add new columns if upgrading from older schema."""
        cursor = conn.execute("PRAGMA table_info(nodes)")
        columns = {row[1] for row in cursor.fetchall()}

        # Note: body column deliberately NOT added - body is read from file via line numbers
        new_columns = [
            ("end_line", "INTEGER"),
            ("signature", "TEXT"),
            ("docstring", "TEXT"),
            ("parent_id", "TEXT"),
        ]
        for col_name, col_type in new_columns:
            if col_name not in columns:
                try:
                    conn.execute(f"ALTER TABLE nodes ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        rows = [
            (
                n.node_id,
                n.type,
                n.name,
                n.file,
                n.line,
                n.end_line,
                n.lang,
                n.signature,
                n.docstring,
                n.parent_id,
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
                    INSERT INTO nodes(node_id, type, name, file, line, end_line, lang,
                                     signature, docstring, parent_id, embedding_ref, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        type=excluded.type,
                        name=excluded.name,
                        file=excluded.file,
                        line=excluded.line,
                        end_line=excluded.end_line,
                        lang=excluded.lang,
                        signature=excluded.signature,
                        docstring=excluded.docstring,
                        parent_id=excluded.parent_id,
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

    def _row_to_node(self, row: tuple) -> GraphNode:
        """Convert a database row to a GraphNode (body read from file, not DB)."""
        return GraphNode(
            node_id=row[0],
            type=row[1],
            name=row[2],
            file=row[3],
            line=row[4],
            end_line=row[5],
            lang=row[6],
            signature=row[7],
            docstring=row[8],
            parent_id=row[9],
            embedding_ref=row[10],
            metadata=json.loads(row[11]) if row[11] else {},
        )

    _NODE_COLS = "node_id, type, name, file, line, end_line, lang, signature, docstring, parent_id, embedding_ref, metadata"

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
        query = f"SELECT {self._NODE_COLS} FROM nodes WHERE {where}"
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(query, params)
                return [self._row_to_node(row) for row in cur.fetchall()]
            finally:
                conn.close()

    async def delete_by_repo(self) -> None:
        """Clear all nodes, edges, and file mtimes for this repo (full rebuild)."""
        async with self._lock:
            conn = self._connect()
            try:
                conn.executescript("DELETE FROM edges; DELETE FROM nodes; DELETE FROM file_mtimes;")
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
                # Count symbols with body text
                cur = conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE body IS NOT NULL AND body != ''"
                )
                body_count = cur.fetchone()[0]
                # Count indexed files
                cur = conn.execute("SELECT COUNT(*) FROM file_mtimes")
                file_count = cur.fetchone()[0]
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "nodes_with_body": body_count,
                    "indexed_files": file_count,
                    "path": str(self.db_path),
                }
            except sqlite3.OperationalError:
                # file_mtimes table might not exist yet
                return {"nodes": node_count, "edges": edge_count, "path": str(self.db_path)}
            finally:
                conn.close()

    async def search_symbols(
        self, query: str, *, limit: int = 20, symbol_types: Optional[Iterable[str]] = None
    ) -> List[GraphNode]:
        """Full-text search across symbol names, signatures, and docstrings.

        Note: Body content is not stored in the DB (read from file via line numbers).
        For body search, use semantic search via embeddings.
        """
        async with self._lock:
            conn = self._connect()
            try:
                # Try FTS5 first
                try:
                    type_clause = ""
                    params: list[Any] = [query, limit]
                    if symbol_types:
                        types = list(symbol_types)
                        type_clause = f"AND n.type IN ({','.join('?' for _ in types)})"
                        params = [query] + types + [limit]

                    # FTS5 searches name, signature, docstring (not body - that's in file)
                    fts_query = f"""
                        SELECT n.node_id, n.type, n.name, n.file, n.line, n.end_line, n.lang,
                               n.signature, n.docstring, n.parent_id, n.embedding_ref, n.metadata
                        FROM nodes_fts fts
                        JOIN nodes n ON fts.node_id = n.node_id
                        WHERE nodes_fts MATCH ?
                        {type_clause}
                        ORDER BY rank
                        LIMIT ?
                    """
                    cur = conn.execute(fts_query, params)
                    return [self._row_to_node(row) for row in cur.fetchall()]
                except sqlite3.OperationalError:
                    # FTS5 not available, fall back to LIKE
                    pass

                # Fallback: LIKE search (no body - use embeddings for body search)
                like_pattern = f"%{query}%"
                type_clause = ""
                params = [like_pattern, like_pattern, like_pattern, limit]
                if symbol_types:
                    types = list(symbol_types)
                    type_clause = f"AND type IN ({','.join('?' for _ in types)})"
                    params = [like_pattern, like_pattern, like_pattern] + types + [limit]

                like_query = f"""
                    SELECT {self._NODE_COLS}
                    FROM nodes
                    WHERE (name LIKE ? OR signature LIKE ? OR docstring LIKE ?)
                    {type_clause}
                    LIMIT ?
                """
                cur = conn.execute(like_query, params)
                return [self._row_to_node(row) for row in cur.fetchall()]
            finally:
                conn.close()

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a single node by its ID."""
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"SELECT {self._NODE_COLS} FROM nodes WHERE node_id = ?", (node_id,)
                )
                row = cur.fetchone()
                return self._row_to_node(row) if row else None
            finally:
                conn.close()

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get all symbols in a specific file."""
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"SELECT {self._NODE_COLS} FROM nodes WHERE file = ? ORDER BY line", (file,)
                )
                return [self._row_to_node(row) for row in cur.fetchall()]
            finally:
                conn.close()

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        """Record file modification time for staleness tracking."""
        import time

        async with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO file_mtimes(file, mtime, indexed_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(file) DO UPDATE SET
                        mtime=excluded.mtime,
                        indexed_at=excluded.indexed_at
                    """,
                    (file, mtime, time.time()),
                )
                conn.commit()
            finally:
                conn.close()

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        """Get files that have changed since last index."""
        stale = []
        async with self._lock:
            conn = self._connect()
            try:
                for file, current_mtime in file_mtimes.items():
                    cur = conn.execute("SELECT mtime FROM file_mtimes WHERE file = ?", (file,))
                    row = cur.fetchone()
                    if row is None or row[0] < current_mtime:
                        stale.append(file)
                return stale
            finally:
                conn.close()

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes and edges for a specific file (for incremental reindex)."""
        async with self._lock:
            conn = self._connect()
            try:
                # Get node IDs for this file
                cur = conn.execute("SELECT node_id FROM nodes WHERE file = ?", (file,))
                node_ids = [row[0] for row in cur.fetchall()]

                if node_ids:
                    placeholders = ",".join("?" for _ in node_ids)
                    # Delete edges involving these nodes
                    conn.execute(f"DELETE FROM edges WHERE src IN ({placeholders})", node_ids)
                    conn.execute(f"DELETE FROM edges WHERE dst IN ({placeholders})", node_ids)
                    # Delete the nodes
                    conn.execute(f"DELETE FROM nodes WHERE node_id IN ({placeholders})", node_ids)

                # Delete file mtime record
                conn.execute("DELETE FROM file_mtimes WHERE file = ?", (file,))
                conn.commit()
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
