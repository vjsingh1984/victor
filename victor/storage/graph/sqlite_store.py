# SQLite-backed GraphStoreProtocol implementation.
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterable, List, Optional, Set

from victor.core.schema import Tables
from victor.storage.graph.protocol import (
    GraphEdge,
    GraphNode,
    GraphQueryResult,
    GraphStoreProtocol,
    GraphTraversalDirection,
)

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

    async def initialize(self) -> None:
        """Ensure schema exists for compatibility with higher-level stores."""
        self._ensure_schema()

    async def close(self) -> None:
        """Release store resources without closing the shared project database.

        ``ProjectDatabaseManager`` is a shared singleton per project path.
        Closing it here invalidates other live components on the same thread.
        Test isolation and process shutdown already clean up the shared manager.
        """
        logger.debug("SqliteGraphStore close skipped shared project database shutdown")

    def _connect(self) -> sqlite3.Connection:
        """Get database connection."""
        return self._db.get_connection()

    def _canonical_file_path(self, file: str | Path) -> str:
        """Normalize file paths so equivalent aliases map to one graph key."""
        path = Path(file).expanduser()
        try:
            return str(path.resolve(strict=False))
        except OSError:
            return str(path.absolute())

    def _file_path_variants(self, file: str | Path) -> List[str]:
        """Return raw plus canonical path forms for compatibility lookups."""
        raw_path = str(file)
        canonical_path = self._canonical_file_path(raw_path)
        if canonical_path == raw_path:
            return [raw_path]
        return [raw_path, canonical_path]

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

        # v4 columns (legacy)
        new_columns = [
            ("end_line", "INTEGER"),
            ("signature", "TEXT"),
            ("docstring", "TEXT"),
            ("parent_id", "TEXT"),
        ]
        # v5 columns (CCG and Graph RAG)
        new_columns.extend(
            [
                ("ast_kind", "TEXT"),
                ("scope_id", "TEXT"),
                ("statement_type", "TEXT"),
                ("requirement_id", "TEXT"),
                ("visibility", "TEXT"),
            ]
        )
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
                # v5 CCG fields
                n.ast_kind,
                n.scope_id,
                n.statement_type,
                n.requirement_id,
                n.visibility,
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
                                 signature, docstring, parent_id, embedding_ref, metadata,
                                 ast_kind, scope_id, statement_type, requirement_id, visibility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    metadata=excluded.metadata,
                    ast_kind=excluded.ast_kind,
                    scope_id=excluded.scope_id,
                    statement_type=excluded.statement_type,
                    requirement_id=excluded.requirement_id,
                    visibility=excluded.visibility
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

        allowed_types = list(edge_types) if edge_types else []
        frontier = {node_id}
        visited_nodes = {node_id}
        seen_edges: Dict[tuple[str, str, str], GraphEdge] = {}

        async with self._lock:
            conn = self._connect()
            for _depth in range(max_depth):
                traversed_edges: List[tuple[GraphEdge, str]] = []

                if direction in {"out", "both"}:
                    traversed_edges.extend(
                        self._select_frontier_edges(
                            conn,
                            frontier=frontier,
                            edge_types=allowed_types,
                            node_column="src",
                            neighbor_column="dst",
                        )
                    )
                if direction in {"in", "both"}:
                    traversed_edges.extend(
                        self._select_frontier_edges(
                            conn,
                            frontier=frontier,
                            edge_types=allowed_types,
                            node_column="dst",
                            neighbor_column="src",
                        )
                    )

                next_frontier: set[str] = set()
                for edge, neighbor_id in traversed_edges:
                    seen_edges[(edge.src, edge.dst, edge.type)] = edge
                    next_frontier.add(neighbor_id)

                next_frontier -= visited_nodes
                if not next_frontier:
                    break
                visited_nodes.update(next_frontier)
                frontier = next_frontier

        return sorted(seen_edges.values(), key=lambda edge: (edge.src, edge.dst, edge.type))

    def _select_frontier_edges(
        self,
        conn: sqlite3.Connection,
        *,
        frontier: set[str],
        edge_types: list[str],
        node_column: str,
        neighbor_column: str,
    ) -> List[tuple[GraphEdge, str]]:
        if not frontier:
            return []

        frontier_params = list(frontier)
        frontier_placeholders = ",".join("?" for _ in frontier_params)
        params: list[Any] = frontier_params
        type_clause = ""

        if edge_types:
            type_placeholders = ",".join("?" for _ in edge_types)
            type_clause = f" AND type IN ({type_placeholders})"
            params.extend(edge_types)

        query = f"""
            SELECT src, dst, type, weight, metadata, {neighbor_column}
            FROM {_EDGE_TABLE}
            WHERE {node_column} IN ({frontier_placeholders}){type_clause}
        """
        cur = conn.execute(query, params)
        return [
            (
                GraphEdge(
                    src=row[0],
                    dst=row[1],
                    type=row[2],
                    weight=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                ),
                row[5],
            )
            for row in cur.fetchall()
        ]

    def _row_to_node(self, row: tuple) -> GraphNode:
        """Convert a database row to a GraphNode."""
        # Handle both v4 (12 columns) and v5 (17 columns) schemas
        row_len = len(row)
        node = GraphNode(
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
            metadata=json.loads(row[11]) if row_len > 11 and row[11] else {},
        )
        # v5 CCG fields (optional for backward compatibility)
        if row_len > 12:
            node.ast_kind = row[12]
        if row_len > 13:
            node.scope_id = row[13]
        if row_len > 14:
            node.statement_type = row[14]
        if row_len > 15:
            node.requirement_id = row[15]
        if row_len > 16:
            node.visibility = row[16]
        return node

    _NODE_COLS = "node_id, type, name, file, line, end_line, lang, signature, docstring, parent_id, embedding_ref, metadata, ast_kind, scope_id, statement_type, requirement_id, visibility"

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
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
            file_variants = self._file_path_variants(file)
            if len(file_variants) == 1:
                clauses.append("file = ?")
            else:
                placeholders = ",".join("?" for _ in file_variants)
                clauses.append(f"file IN ({placeholders})")
            params.extend(file_variants)
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
            conn.executescript(
                f"DELETE FROM {_EDGE_TABLE}; DELETE FROM {_NODE_TABLE}; DELETE FROM {_MTIME_TABLE};"
            )
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
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "path": str(self.db_path),
                }

    async def search_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        symbol_types: Optional[Iterable[str]] = None,
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
                f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} WHERE node_id = ?",
                (node_id,),
            )
            row = cur.fetchone()
            return self._row_to_node(row) if row else None

    async def get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes in the graph."""
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(
                f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} ORDER BY file, line, name"
            )
            return [self._row_to_node(row) for row in cur.fetchall()]

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get all symbols in a specific file."""
        file_variants = self._file_path_variants(file)
        placeholders = ",".join("?" for _ in file_variants)
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(
                f"""
                SELECT {self._NODE_COLS}
                FROM {_NODE_TABLE}
                WHERE file IN ({placeholders})
                ORDER BY line
                """,
                file_variants,
            )
            return [self._row_to_node(row) for row in cur.fetchall()]

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        """Record file modification time for staleness tracking."""
        import time

        file = self._canonical_file_path(file)
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
                file_variants = self._file_path_variants(file)
                placeholders = ",".join("?" for _ in file_variants)
                cur = conn.execute(
                    f"SELECT MAX(mtime) FROM {_MTIME_TABLE} WHERE file IN ({placeholders})",
                    file_variants,
                )
                row = cur.fetchone()
                recorded_mtime = None if row is None else row[0]
                if recorded_mtime is None or recorded_mtime < current_mtime:
                    stale.append(file)
            return stale

    async def get_indexed_files(self) -> List[str]:
        """Get the set of files currently tracked for graph staleness."""
        async with self._lock:
            conn = self._connect()
            cur = conn.execute(f"SELECT file FROM {_MTIME_TABLE} ORDER BY file")
            return [str(row[0]) for row in cur.fetchall()]

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes and edges for a specific file (for incremental reindex)."""
        file_variants = self._file_path_variants(file)
        placeholders = ",".join("?" for _ in file_variants)
        async with self._lock:
            conn = self._connect()
            # Get node IDs for this file
            cur = conn.execute(
                f"SELECT node_id FROM {_NODE_TABLE} WHERE file IN ({placeholders})",
                file_variants,
            )
            node_ids = [row[0] for row in cur.fetchall()]

            if node_ids:
                placeholders = ",".join("?" for _ in node_ids)
                conn.execute(f"DELETE FROM {_EDGE_TABLE} WHERE src IN ({placeholders})", node_ids)
                conn.execute(f"DELETE FROM {_EDGE_TABLE} WHERE dst IN ({placeholders})", node_ids)
                conn.execute(
                    f"DELETE FROM {_NODE_TABLE} WHERE node_id IN ({placeholders})",
                    node_ids,
                )

            file_placeholders = ",".join("?" for _ in file_variants)
            conn.execute(
                f"DELETE FROM {_MTIME_TABLE} WHERE file IN ({file_placeholders})",
                file_variants,
            )
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

    # ===========================================
    # v5: Lazy loading methods (PH4-006)
    # ===========================================

    async def iter_nodes(
        self,
        *,
        batch_size: int = 100,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> AsyncIterator[List[GraphNode]]:
        """Iterate over nodes in batches for memory-efficient processing."""
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
        query = (
            f"SELECT {self._NODE_COLS} FROM {_NODE_TABLE} WHERE {where} ORDER BY file, line, name"
        )

        async with self._lock:
            conn = self._connect()
            cur = conn.execute(query, params)

            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yield [self._row_to_node(row) for row in rows]

    async def iter_edges(
        self,
        *,
        batch_size: int = 100,
        edge_types: Iterable[str] | None = None,
    ) -> AsyncIterator[List[GraphEdge]]:
        """Iterate over edges in batches for memory-efficient processing."""
        query = f"SELECT src, dst, type, weight, metadata FROM {_EDGE_TABLE}"
        params: list[Any] = []

        if edge_types:
            types = list(edge_types)
            placeholders = ",".join("?" for _ in types)
            query += f" WHERE type IN ({placeholders})"
            params.extend(types)

        query += " ORDER BY src, dst, type"

        async with self._lock:
            conn = self._connect()
            cur = conn.execute(query, params)

            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yield [
                    GraphEdge(
                        src=row[0],
                        dst=row[1],
                        type=row[2],
                        weight=row[3],
                        metadata=json.loads(row[4]) if row[4] else {},
                    )
                    for row in rows
                ]

    async def iter_neighbors(
        self,
        node_id: str,
        *,
        batch_size: int = 50,
        edge_types: Iterable[str] | None = None,
        direction: GraphTraversalDirection = "out",
    ) -> AsyncIterator[List[GraphEdge]]:
        """Iterate over neighbors in batches for memory-efficient traversal."""
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"Unsupported graph traversal direction: {direction}")

        # Build query for single-hop neighbors
        if direction == "out":
            node_column = "src"
            neighbor_column = "dst"
        elif direction == "in":
            node_column = "dst"
            neighbor_column = "src"
        else:  # both
            # For both directions, we'll use two queries
            async for batch in self._iter_neighbors_direction(
                node_id, batch_size, edge_types, "src", "dst"
            ):
                yield batch
            async for batch in self._iter_neighbors_direction(
                node_id, batch_size, edge_types, "dst", "src"
            ):
                yield batch
            return

        async for batch in self._iter_neighbors_direction(
            node_id, batch_size, edge_types, node_column, neighbor_column
        ):
            yield batch

    async def _iter_neighbors_direction(
        self,
        node_id: str,
        batch_size: int,
        edge_types: Iterable[str] | None,
        node_column: str,
        neighbor_column: str,
    ) -> AsyncIterator[List[GraphEdge]]:
        """Helper for iterating neighbors in a specific direction."""
        query = f"""
            SELECT src, dst, type, weight, metadata, {neighbor_column}
            FROM {_EDGE_TABLE}
            WHERE {node_column} = ?
        """
        params: list[Any] = [node_id]

        if edge_types:
            types = list(edge_types)
            placeholders = ",".join("?" for _ in types)
            query += f" AND type IN ({placeholders})"
            params.extend(types)

        query += f" ORDER BY {neighbor_column}, type"

        async with self._lock:
            conn = self._connect()
            cur = conn.execute(query, params)

            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                yield [
                    GraphEdge(
                        src=row[0],
                        dst=row[1],
                        type=row[2],
                        weight=row[3],
                        metadata=json.loads(row[4]) if row[4] else {},
                    )
                    for row in rows
                ]

    # ===========================================
    # v5: Parallel traversal methods (PH4-007)
    # ===========================================

    async def get_neighbors_batch(
        self,
        node_ids: List[str],
        *,
        edge_types: Iterable[str] | None = None,
        direction: GraphTraversalDirection = "out",
    ) -> Dict[str, List[GraphEdge]]:
        """Get neighbors for multiple nodes in parallel.

        Uses asyncio.gather() to fetch neighbors concurrently.

        Args:
            node_ids: List of node IDs to get neighbors for
            edge_types: Optional filter by edge types
            direction: Traversal direction

        Returns:
            Dict mapping node_id to list of neighboring edges
        """
        if not node_ids:
            return {}

        # Create tasks for each node
        tasks = [
            self.get_neighbors(
                node_id,
                edge_types=edge_types,
                direction=direction,
                max_depth=1,
            )
            for node_id in node_ids
        ]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict
        neighbor_map: Dict[str, List[GraphEdge]] = {}
        for node_id, result in zip(node_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"Error getting neighbors for {node_id}: {result}")
                neighbor_map[node_id] = []
            else:
                neighbor_map[node_id] = result

        return neighbor_map

    async def multi_hop_traverse_parallel(
        self,
        start_node_ids: List[str],
        max_hops: int = 2,
        edge_types: Iterable[str] | None = None,
        max_nodes: int = 100,
        max_workers: int = 4,
    ) -> GraphQueryResult:
        """Parallel multi-hop traversal from multiple start nodes.

        Performs BFS traversal but processes multiple frontier nodes
        in parallel using asyncio.gather().

        Args:
            start_node_ids: Multiple starting node IDs
            max_hops: Maximum hop distance
            edge_types: Optional filter by edge types
            max_nodes: Maximum total nodes to return
            max_workers: Maximum parallel workers

        Returns:
            GraphQueryResult with traversed nodes and edges
        """
        import time
        from victor.storage.graph.protocol import GraphQueryResult

        start_time = time.time()

        if not start_node_ids:
            return GraphQueryResult(nodes=[], edges=[], query="parallel_traversal")

        # Initialize BFS state
        visited: Set[str] = set(start_node_ids)
        frontier: List[str] = list(start_node_ids)
        all_edges: Dict[tuple[str, str, str], GraphEdge] = {}
        all_nodes: Dict[str, GraphNode] = {}

        # Get initial nodes
        for node_id in start_node_ids:
            node = await self.get_node_by_id(node_id)
            if node:
                all_nodes[node_id] = node

        # Parallel BFS
        for hop in range(max_hops):
            if not frontier or len(all_nodes) >= max_nodes:
                break

            # Process frontier in parallel batches
            batch_size = max_workers
            results: Dict[str, List[GraphEdge]] = {}

            for i in range(0, len(frontier), batch_size):
                batch = frontier[i : i + batch_size]

                # Fetch neighbors for this batch in parallel
                batch_results = await self.get_neighbors_batch(
                    batch,
                    edge_types=edge_types,
                    direction="out",
                )

                results.update(batch_results)

            # Process results
            next_frontier: Set[str] = set()

            for node_id, neighbors in results.items():
                for edge in neighbors:
                    # Store edge
                    edge_key = (edge.src, edge.dst, edge.type)
                    if edge_key not in all_edges:
                        all_edges[edge_key] = edge

                    # Track neighbor
                    neighbor_id = edge.dst
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)

                        # Get neighbor node details
                        if len(all_nodes) < max_nodes:
                            node = await self.get_node_by_id(neighbor_id)
                            if node:
                                all_nodes[neighbor_id] = node

                        # Stop if we've reached max_nodes
                        if len(all_nodes) >= max_nodes:
                            break

                if len(all_nodes) >= max_nodes:
                    break

            frontier = list(next_frontier)

        # Build result
        return GraphQueryResult(
            nodes=list(all_nodes.values()),
            edges=list(all_edges.values()),
            query=f"parallel_traverse_{len(start_node_ids)}_seeds",
            execution_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "start_nodes": start_node_ids,
                "max_hops": max_hops,
                "hops_completed": hop + 1 if hop < max_hops else max_hops,
                "max_workers": max_workers,
            },
        )
