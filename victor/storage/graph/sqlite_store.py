# SQLite-backed GraphStoreProtocol implementation.
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

from victor.core.schema import Tables
from victor.storage.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol

if TYPE_CHECKING:
    from victor.core.database import ProjectDatabaseManager

logger = logging.getLogger(__name__)

# Table names from centralized schema
_NODE_TABLE = Tables.GRAPH_NODE
_EDGE_TABLE = Tables.GRAPH_EDGE
_MTIME_TABLE = Tables.GRAPH_FILE_MTIME
_FTS_TABLE = f"{_NODE_TABLE}_fts"

# Schema for consolidated database
SCHEMA = f"""
PRAGMA journal_mode=WAL;

-- Core symbol/node table (body read from file via line numbers, not stored here)
CREATE TABLE IF NOT EXISTS {_NODE_TABLE} (
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
    FOREIGN KEY (parent_id) REFERENCES {_NODE_TABLE}(node_id)
);

-- Graph edges table
CREATE TABLE IF NOT EXISTS {_EDGE_TABLE} (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    type TEXT NOT NULL,
    weight REAL,
    metadata TEXT,
    PRIMARY KEY (src, dst, type)
);

-- File modification times for staleness tracking
CREATE TABLE IF NOT EXISTS {_MTIME_TABLE} (
    file TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    indexed_at REAL NOT NULL
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_{_NODE_TABLE}_type_name ON {_NODE_TABLE}(type, name);
CREATE INDEX IF NOT EXISTS idx_{_NODE_TABLE}_file ON {_NODE_TABLE}(file);
CREATE INDEX IF NOT EXISTS idx_{_NODE_TABLE}_parent ON {_NODE_TABLE}(parent_id);
CREATE INDEX IF NOT EXISTS idx_{_EDGE_TABLE}_src_type ON {_EDGE_TABLE}(src, type);
CREATE INDEX IF NOT EXISTS idx_{_EDGE_TABLE}_dst_type ON {_EDGE_TABLE}(dst, type);
CREATE INDEX IF NOT EXISTS idx_{_MTIME_TABLE}_mtime ON {_MTIME_TABLE}(mtime);
"""

# FTS5 schema for full-text search on symbols
FTS_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE} USING fts5(
    node_id,
    name,
    signature,
    docstring,
    content='{_NODE_TABLE}',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS {_NODE_TABLE}_ai AFTER INSERT ON {_NODE_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}(rowid, node_id, name, signature, docstring)
    VALUES (new.rowid, new.node_id, new.name, new.signature, new.docstring);
END;

CREATE TRIGGER IF NOT EXISTS {_NODE_TABLE}_ad AFTER DELETE ON {_NODE_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, node_id, name, signature, docstring)
    VALUES ('delete', old.rowid, old.node_id, old.name, old.signature, old.docstring);
END;

CREATE TRIGGER IF NOT EXISTS {_NODE_TABLE}_au AFTER UPDATE ON {_NODE_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, node_id, name, signature, docstring)
    VALUES ('delete', old.rowid, old.node_id, old.name, old.signature, old.docstring);
    INSERT INTO {_FTS_TABLE}(rowid, node_id, name, signature, docstring)
    VALUES (new.rowid, new.node_id, new.name, new.signature, new.docstring);
END;
"""


class SqliteGraphStore(GraphStoreProtocol):
    """Embedded SQLite graph store using consolidated project.db."""

    def __init__(self, project_path: Optional[Path] = None) -> None:
        """Initialize graph store.

        Args:
            project_path: Path to project root. If None, uses current directory.
        """
        from victor.core.database import get_project_database

        self._db = get_project_database(project_path)
        self.db_path = self._db.db_path
        self._ensure_schema()
        self._lock = asyncio.Lock()

    def _connect(self) -> sqlite3.Connection:
        """Get database connection."""
        return self._db.get_connection()

    def _ensure_schema(self) -> None:
        conn = self._connect()
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

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add new columns if upgrading from older schema."""
        cursor = conn.execute(f"PRAGMA table_info({_NODE_TABLE})")
        columns = {row[1] for row in cursor.fetchall()}

        new_columns = [
            ("end_line", "INTEGER"),
            ("signature", "TEXT"),
            ("docstring", "TEXT"),
            ("parent_id", "TEXT"),
        ]
        for col_name, col_type in new_columns:
            if col_name not in columns:
                try:
                    conn.execute(f"ALTER TABLE {_NODE_TABLE} ADD COLUMN {col_name} {col_type}")
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
            conn.executemany(
                f"""
                INSERT INTO {_NODE_TABLE}(node_id, type, name, file, line, end_line, lang,
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
            conn.executemany(
                f"""
                INSERT INTO {_EDGE_TABLE}(src, dst, type, weight, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(src, dst, type) DO UPDATE SET
                    weight=excluded.weight,
                    metadata=excluded.metadata
                """,
                rows,
            )
            conn.commit()

    async def get_neighbors(
        self, node_id: str, edge_types: Optional[Iterable[str]] = None, max_depth: int = 1
    ) -> List[GraphEdge]:
        params: list[Any] = [node_id]
        type_clause = ""
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            type_clause = f" AND type IN ({placeholders})"
            params.extend(edge_types)
        query = f"SELECT src, dst, type, weight, metadata FROM {_EDGE_TABLE} WHERE src=?{type_clause}"
        async with self._lock:
            conn = self._connect()
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

    def _row_to_node(self, row: tuple) -> GraphNode:
        """Convert a database row to a GraphNode."""
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
        query = f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} WHERE {where}"
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(query, params)
            return [self._row_to_node(row) for row in cur.fetchall()]

    async def delete_by_repo(self) -> None:
        """Clear all nodes, edges, and file mtimes for this repo (full rebuild)."""
        async with self._lock:
            conn = self._connect()
            conn.executescript(f"DELETE FROM {_EDGE_TABLE}; DELETE FROM {_NODE_TABLE}; DELETE FROM {_MTIME_TABLE};")
            conn.commit()

    async def stats(self) -> Dict[str, Any]:
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM {_NODE_TABLE}")
                node_count = cur.fetchone()[0]
                cur = conn.execute(f"SELECT COUNT(*) FROM {_EDGE_TABLE}")
                edge_count = cur.fetchone()[0]
                cur = conn.execute(f"SELECT COUNT(*) FROM {_MTIME_TABLE}")
                file_count = cur.fetchone()[0]
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "indexed_files": file_count,
                    "path": str(self.db_path),
                }
            except sqlite3.OperationalError:
                return {"nodes": node_count, "edges": edge_count, "path": str(self.db_path)}

    async def search_symbols(
        self, query: str, *, limit: int = 20, symbol_types: Optional[Iterable[str]] = None
    ) -> List[GraphNode]:
        """Full-text search across symbol names, signatures, and docstrings."""
        async with self._lock:
            conn = self._connect()
            # Try FTS5 first
            try:
                type_clause = ""
                params: list[Any] = [query, limit]
                if symbol_types:
                    types = list(symbol_types)
                    type_clause = f"AND n.type IN ({','.join('?' for _ in types)})"
                    params = [query] + types + [limit]

                fts_query = f"""
                    SELECT n.node_id, n.type, n.name, n.file, n.line, n.end_line, n.lang,
                           n.signature, n.docstring, n.parent_id, n.embedding_ref, n.metadata
                    FROM {_FTS_TABLE} fts
                    JOIN {_NODE_TABLE} n ON fts.node_id = n.node_id
                    WHERE {_FTS_TABLE} MATCH ?
                    {type_clause}
                    ORDER BY rank
                    LIMIT ?
                """
                cur = conn.execute(fts_query, params)
                return [self._row_to_node(row) for row in cur.fetchall()]
            except sqlite3.OperationalError:
                pass

            # Fallback: LIKE search
            like_pattern = f"%{query}%"
            type_clause = ""
            params = [like_pattern, like_pattern, like_pattern, limit]
            if symbol_types:
                types = list(symbol_types)
                type_clause = f"AND type IN ({','.join('?' for _ in types)})"
                params = [like_pattern, like_pattern, like_pattern] + types + [limit]

            like_query = f"""
                SELECT {self._NODE_COLS}
                FROM {_NODE_TABLE}
                WHERE (name LIKE ? OR signature LIKE ? OR docstring LIKE ?)
                {type_clause}
                LIMIT ?
            """
            cur = conn.execute(like_query, params)
            return [self._row_to_node(row) for row in cur.fetchall()]

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a single node by its ID."""
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(
                f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} WHERE node_id = ?", (node_id,)
            )
            row = cur.fetchone()
            return self._row_to_node(row) if row else None

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get all symbols in a specific file."""
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(
                f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} WHERE file = ? ORDER BY line", (file,)
            )
            return [self._row_to_node(row) for row in cur.fetchall()]

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        """Record file modification time for staleness tracking."""
        import time

        async with self._lock:
            conn = self._connect()
            conn.execute(
                f"""
                INSERT INTO {_MTIME_TABLE}(file, mtime, indexed_at)
                VALUES (?, ?, ?)
                ON CONFLICT(file) DO UPDATE SET
                    mtime=excluded.mtime,
                    indexed_at=excluded.indexed_at
                """,
                (file, mtime, time.time()),
            )
            conn.commit()

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        """Get files that have changed since last index."""
        stale = []
        async with self._lock:
            conn = self._connect()
            for file, current_mtime in file_mtimes.items():
                cur = conn.execute(f"SELECT mtime FROM {_MTIME_TABLE} WHERE file = ?", (file,))
                row = cur.fetchone()
                if row is None or row[0] < current_mtime:
                    stale.append(file)
            return stale

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes and edges for a specific file (for incremental reindex)."""
        async with self._lock:
            conn = self._connect()
            # Get node IDs for this file
            cur = conn.execute(f"SELECT node_id FROM {_NODE_TABLE} WHERE file = ?", (file,))
            node_ids = [row[0] for row in cur.fetchall()]

            if node_ids:
                placeholders = ",".join("?" for _ in node_ids)
                conn.execute(f"DELETE FROM {_EDGE_TABLE} WHERE src IN ({placeholders})", node_ids)
                conn.execute(f"DELETE FROM {_EDGE_TABLE} WHERE dst IN ({placeholders})", node_ids)
                conn.execute(f"DELETE FROM {_NODE_TABLE} WHERE node_id IN ({placeholders})", node_ids)

            conn.execute(f"DELETE FROM {_MTIME_TABLE} WHERE file = ?", (file,))
            conn.commit()

    async def get_all_edges(self) -> List[GraphEdge]:
        """Get all edges in the graph (bulk retrieval for loading into memory)."""
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(f"SELECT src, dst, type, weight, metadata FROM {_EDGE_TABLE}")
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
