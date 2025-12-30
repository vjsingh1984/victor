# SQLite-backed GraphStoreProtocol implementation.
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

from victor.storage.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol

if TYPE_CHECKING:
    from victor.core.database import ProjectDatabaseManager

logger = logging.getLogger(__name__)

# Legacy schema for standalone database (backwards compatibility)
LEGACY_SCHEMA = """
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
CREATE VIRTUAL TABLE IF NOT EXISTS {nodes_table}_fts USING fts5(
    node_id,
    name,
    signature,
    docstring,
    content='{nodes_table}',
    content_rowid='rowid'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS {nodes_table}_ai AFTER INSERT ON {nodes_table} BEGIN
    INSERT INTO {nodes_table}_fts(rowid, node_id, name, signature, docstring)
    VALUES (new.rowid, new.node_id, new.name, new.signature, new.docstring);
END;

CREATE TRIGGER IF NOT EXISTS {nodes_table}_ad AFTER DELETE ON {nodes_table} BEGIN
    INSERT INTO {nodes_table}_fts({nodes_table}_fts, rowid, node_id, name, signature, docstring)
    VALUES ('delete', old.rowid, old.node_id, old.name, old.signature, old.docstring);
END;

CREATE TRIGGER IF NOT EXISTS {nodes_table}_au AFTER UPDATE ON {nodes_table} BEGIN
    INSERT INTO {nodes_table}_fts({nodes_table}_fts, rowid, node_id, name, signature, docstring)
    VALUES ('delete', old.rowid, old.node_id, old.name, old.signature, old.docstring);
    INSERT INTO {nodes_table}_fts(rowid, node_id, name, signature, docstring)
    VALUES (new.rowid, new.node_id, new.name, new.signature, new.docstring);
END;
"""


class SqliteGraphStore(GraphStoreProtocol):
    """Embedded SQLite graph store for per-repo symbol graphs.

    Supports two modes:
    1. Standalone mode: Uses a dedicated graph.db file (legacy behavior)
    2. Consolidated mode: Uses ProjectDatabaseManager with project.db

    By default (db_path=None), uses consolidated mode with project.db.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        project_path: Optional[Path] = None,
        use_consolidated: bool = True,
    ) -> None:
        """Initialize graph store.

        Args:
            db_path: Path to standalone database file. If provided, uses legacy mode.
            project_path: Path to project root for consolidated mode.
            use_consolidated: Whether to use consolidated project.db (default True).
        """
        from victor.core.schema import Tables

        self._project_db: Optional["ProjectDatabaseManager"] = None
        self._use_consolidated = use_consolidated and db_path is None

        if self._use_consolidated:
            # Use consolidated project.db via ProjectDatabaseManager
            from victor.core.database import get_project_database

            self._project_db = get_project_database(project_path)
            self.db_path = self._project_db.db_path
            # Use new table names from schema
            self._nodes_table = Tables.GRAPH_NODE
            self._edges_table = Tables.GRAPH_EDGE
            self._mtimes_table = Tables.GRAPH_FILE_MTIME
            logger.info(f"SqliteGraphStore using consolidated database: {self.db_path}")
        else:
            # Legacy standalone mode
            if db_path is None:
                raise ValueError("db_path required when use_consolidated=False")
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # Use legacy table names for backwards compatibility
            self._nodes_table = "nodes"
            self._edges_table = "edges"
            self._mtimes_table = "file_mtimes"
            self._ensure_schema()
            logger.info(f"SqliteGraphStore using standalone database: {self.db_path}")

        # sqlite3 is threadsafe with check_same_thread=False when guarded; use async wrapper.
        self._lock = asyncio.Lock()

        # Setup FTS for the tables
        self._ensure_fts()

    def _connect(self) -> sqlite3.Connection:
        """Get database connection.

        In consolidated mode, uses ProjectDatabaseManager's connection.
        In standalone mode, creates a new connection.
        """
        if self._use_consolidated and self._project_db is not None:
            return self._project_db.get_connection()
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _ensure_schema(self) -> None:
        """Create schema for standalone mode (legacy)."""
        conn = self._connect()
        try:
            conn.executescript(LEGACY_SCHEMA)
            # Add new columns if upgrading from older schema
            self._migrate_schema(conn)
            conn.commit()
        finally:
            if not self._use_consolidated:
                conn.close()

    def _ensure_fts(self) -> None:
        """Create FTS5 table for full-text search."""
        conn = self._connect()
        try:
            # Format FTS schema with correct table name
            fts_sql = FTS_SCHEMA.format(nodes_table=self._nodes_table)
            conn.executescript(fts_sql)
            conn.commit()
        except sqlite3.OperationalError as e:
            # FTS5 might not be available on all SQLite builds
            logger.debug(f"FTS5 not available: {e}")
        finally:
            if not self._use_consolidated:
                conn.close()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add new columns if upgrading from older schema."""
        cursor = conn.execute(f"PRAGMA table_info({self._nodes_table})")
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
                    conn.execute(
                        f"ALTER TABLE {self._nodes_table} ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass  # Column already exists

    def _close_if_standalone(self, conn: sqlite3.Connection) -> None:
        """Close connection only in standalone mode (not in consolidated mode)."""
        if not self._use_consolidated:
            conn.close()

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
                    f"""
                    INSERT INTO {self._nodes_table}(node_id, type, name, file, line, end_line, lang,
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
                self._close_if_standalone(conn)

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
                    f"""
                    INSERT INTO {self._edges_table}(src, dst, type, weight, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(src, dst, type) DO UPDATE SET
                        weight=excluded.weight,
                        metadata=excluded.metadata
                    """,
                    rows,
                )
                conn.commit()
            finally:
                self._close_if_standalone(conn)

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
        query = f"SELECT src, dst, type, weight, metadata FROM {self._edges_table} WHERE src=?{type_clause}"
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
                self._close_if_standalone(conn)

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
        query = f"SELECT {self._NODE_COLS} FROM {self._nodes_table} WHERE {where}"
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(query, params)
                return [self._row_to_node(row) for row in cur.fetchall()]
            finally:
                self._close_if_standalone(conn)

    async def delete_by_repo(self) -> None:
        """Clear all nodes, edges, and file mtimes for this repo (full rebuild)."""
        async with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    f"DELETE FROM {self._edges_table}; "
                    f"DELETE FROM {self._nodes_table}; "
                    f"DELETE FROM {self._mtimes_table};"
                )
                conn.commit()
            finally:
                self._close_if_standalone(conn)

    async def stats(self) -> Dict[str, Any]:
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM {self._nodes_table}")
                node_count = cur.fetchone()[0]
                cur = conn.execute(f"SELECT COUNT(*) FROM {self._edges_table}")
                edge_count = cur.fetchone()[0]
                # Count indexed files
                cur = conn.execute(f"SELECT COUNT(*) FROM {self._mtimes_table}")
                file_count = cur.fetchone()[0]
                return {
                    "nodes": node_count,
                    "edges": edge_count,
                    "indexed_files": file_count,
                    "path": str(self.db_path),
                    "consolidated": self._use_consolidated,
                }
            except sqlite3.OperationalError:
                # table might not exist yet
                return {"nodes": node_count, "edges": edge_count, "path": str(self.db_path)}
            finally:
                self._close_if_standalone(conn)

    async def search_symbols(
        self, query: str, *, limit: int = 20, symbol_types: Optional[Iterable[str]] = None
    ) -> List[GraphNode]:
        """Full-text search across symbol names, signatures, and docstrings.

        Note: Body content is not stored in the DB (read from file via line numbers).
        For body search, use semantic search via embeddings.
        """
        fts_table = f"{self._nodes_table}_fts"
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
                        FROM {fts_table} fts
                        JOIN {self._nodes_table} n ON fts.node_id = n.node_id
                        WHERE {fts_table} MATCH ?
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
                    FROM {self._nodes_table}
                    WHERE (name LIKE ? OR signature LIKE ? OR docstring LIKE ?)
                    {type_clause}
                    LIMIT ?
                """
                cur = conn.execute(like_query, params)
                return [self._row_to_node(row) for row in cur.fetchall()]
            finally:
                self._close_if_standalone(conn)

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a single node by its ID."""
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"SELECT {self._NODE_COLS} FROM {self._nodes_table} WHERE node_id = ?",
                    (node_id,),
                )
                row = cur.fetchone()
                return self._row_to_node(row) if row else None
            finally:
                self._close_if_standalone(conn)

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        """Get all symbols in a specific file."""
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"SELECT {self._NODE_COLS} FROM {self._nodes_table} WHERE file = ? ORDER BY line",
                    (file,),
                )
                return [self._row_to_node(row) for row in cur.fetchall()]
            finally:
                self._close_if_standalone(conn)

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        """Record file modification time for staleness tracking."""
        import time

        async with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"""
                    INSERT INTO {self._mtimes_table}(file, mtime, indexed_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(file) DO UPDATE SET
                        mtime=excluded.mtime,
                        indexed_at=excluded.indexed_at
                    """,
                    (file, mtime, time.time()),
                )
                conn.commit()
            finally:
                self._close_if_standalone(conn)

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        """Get files that have changed since last index."""
        stale = []
        async with self._lock:
            conn = self._connect()
            try:
                for file, current_mtime in file_mtimes.items():
                    cur = conn.execute(
                        f"SELECT mtime FROM {self._mtimes_table} WHERE file = ?", (file,)
                    )
                    row = cur.fetchone()
                    if row is None or row[0] < current_mtime:
                        stale.append(file)
                return stale
            finally:
                self._close_if_standalone(conn)

    async def delete_by_file(self, file: str) -> None:
        """Delete all nodes and edges for a specific file (for incremental reindex)."""
        async with self._lock:
            conn = self._connect()
            try:
                # Get node IDs for this file
                cur = conn.execute(
                    f"SELECT node_id FROM {self._nodes_table} WHERE file = ?", (file,)
                )
                node_ids = [row[0] for row in cur.fetchall()]

                if node_ids:
                    placeholders = ",".join("?" for _ in node_ids)
                    # Delete edges involving these nodes
                    conn.execute(
                        f"DELETE FROM {self._edges_table} WHERE src IN ({placeholders})",
                        node_ids,
                    )
                    conn.execute(
                        f"DELETE FROM {self._edges_table} WHERE dst IN ({placeholders})",
                        node_ids,
                    )
                    # Delete the nodes
                    conn.execute(
                        f"DELETE FROM {self._nodes_table} WHERE node_id IN ({placeholders})",
                        node_ids,
                    )

                # Delete file mtime record
                conn.execute(f"DELETE FROM {self._mtimes_table} WHERE file = ?", (file,))
                conn.commit()
            finally:
                self._close_if_standalone(conn)

    async def get_all_edges(self) -> List[GraphEdge]:
        """Get all edges in the graph (bulk retrieval for loading into memory)."""
        async with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"SELECT src, dst, type, weight, metadata FROM {self._edges_table}"
                )
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
                self._close_if_standalone(conn)

    @staticmethod
    def migrate_from_legacy(
        legacy_db_path: Path,
        project_path: Optional[Path] = None,
    ) -> Dict[str, int]:
        """Migrate data from legacy graph.db to consolidated project.db.

        Args:
            legacy_db_path: Path to legacy graph/graph.db file
            project_path: Path to project root (defaults to cwd)

        Returns:
            Dict with migration stats (nodes_migrated, edges_migrated, files_migrated)
        """
        from victor.core.schema import Tables
        from victor.core.database import get_project_database

        if not legacy_db_path.exists():
            logger.warning(f"Legacy database not found: {legacy_db_path}")
            return {"nodes_migrated": 0, "edges_migrated": 0, "files_migrated": 0}

        # Get the project database
        project_db = get_project_database(project_path)
        target_conn = project_db.get_connection()

        # Connect to legacy database
        legacy_conn = sqlite3.connect(str(legacy_db_path), check_same_thread=False)
        legacy_conn.row_factory = sqlite3.Row

        try:
            # Check if target already has data
            cursor = target_conn.execute(f"SELECT COUNT(*) FROM {Tables.GRAPH_NODE}")
            existing_count = cursor.fetchone()[0]
            if existing_count > 0:
                logger.info(
                    f"Target database already has {existing_count} nodes, skipping migration"
                )
                return {"nodes_migrated": 0, "edges_migrated": 0, "files_migrated": 0}

            # Migrate nodes
            logger.info("Migrating nodes from legacy database...")
            cursor = legacy_conn.execute(
                "SELECT node_id, type, name, file, line, end_line, lang, "
                "signature, docstring, parent_id, embedding_ref, metadata FROM nodes"
            )
            nodes_data = cursor.fetchall()
            nodes_count = len(nodes_data)

            if nodes_data:
                target_conn.executemany(
                    f"""
                    INSERT OR IGNORE INTO {Tables.GRAPH_NODE}
                    (node_id, type, name, file, line, end_line, lang,
                     signature, docstring, parent_id, embedding_ref, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [tuple(row) for row in nodes_data],
                )
                logger.info(f"Migrated {nodes_count} nodes")

            # Migrate edges
            logger.info("Migrating edges from legacy database...")
            cursor = legacy_conn.execute("SELECT src, dst, type, weight, metadata FROM edges")
            edges_data = cursor.fetchall()
            edges_count = len(edges_data)

            if edges_data:
                target_conn.executemany(
                    f"""
                    INSERT OR IGNORE INTO {Tables.GRAPH_EDGE}
                    (src, dst, type, weight, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [tuple(row) for row in edges_data],
                )
                logger.info(f"Migrated {edges_count} edges")

            # Migrate file mtimes
            logger.info("Migrating file mtimes from legacy database...")
            try:
                cursor = legacy_conn.execute("SELECT file, mtime, indexed_at FROM file_mtimes")
                mtimes_data = cursor.fetchall()
                files_count = len(mtimes_data)

                if mtimes_data:
                    target_conn.executemany(
                        f"""
                        INSERT OR IGNORE INTO {Tables.GRAPH_FILE_MTIME}
                        (file, mtime, indexed_at)
                        VALUES (?, ?, ?)
                        """,
                        [tuple(row) for row in mtimes_data],
                    )
                    logger.info(f"Migrated {files_count} file mtimes")
            except sqlite3.OperationalError:
                files_count = 0
                logger.debug("No file_mtimes table in legacy database")

            target_conn.commit()

            logger.info(
                f"Migration complete: {nodes_count} nodes, {edges_count} edges, "
                f"{files_count} file mtimes"
            )

            return {
                "nodes_migrated": nodes_count,
                "edges_migrated": edges_count,
                "files_migrated": files_count,
            }

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            target_conn.rollback()
            raise
        finally:
            legacy_conn.close()
