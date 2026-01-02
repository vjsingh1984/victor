# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SQLite + LanceDB implementation of UnifiedSymbolStore.

Default backend for local/air-gapped deployments:
- SQLite: Graph storage (nodes, edges) + full-text search (FTS5)
- LanceDB: Vector storage + semantic search

Key Features:
- Unified IDs for graph-embedding correlation
- No content duplication (vectors only in LanceDB, metadata in SQLite)
- Hybrid search (FTS + semantic + PageRank weighting)
- Batch operations for performance
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.storage.unified.protocol import (
    GraphEdge,
    GraphNode,
    SearchMode,
    SearchParams,
    SearchResult,
    UnifiedEdge,
    UnifiedId,
    UnifiedSymbol,
)

logger = logging.getLogger(__name__)


class SqliteLanceDBStore:
    """Unified symbol store using SQLite for graph + LanceDB for vectors.

    Architecture:
        SQLite (.victor/project.db):
            - graph_node: Symbol metadata, FTS index
            - graph_edge: Relationships
            - graph_file_mtime: Staleness tracking

        LanceDB (.victor/embeddings/):
            - embeddings table: (id, vector, file_path, symbol_type)
            - NO content storage (deduplicated - lookup via graph)

    Unified ID Scheme:
        - Format: {type}:{repo_relative_path}:{symbol_name}
        - Used as primary key in BOTH stores
        - Example: symbol:victor/tools/graph_tool.py:find_symbols
    """

    def __init__(
        self,
        repo_root: Path,
        persist_directory: Optional[Path] = None,
        embedding_model_type: str = "sentence-transformers",
        embedding_model_name: str = "all-MiniLM-L12-v2",
        **kwargs: Any,
    ):
        """Initialize store.

        Args:
            repo_root: Root directory of the repository
            persist_directory: Override storage location (default: {repo_root}/.victor)
            embedding_model_type: Embedding model type (sentence-transformers, ollama, openai)
            embedding_model_name: Model name (default: all-MiniLM-L12-v2)
        """
        self.repo_root = repo_root.resolve()
        self.persist_directory = persist_directory or (self.repo_root / ".victor")
        self.embedding_model_type = embedding_model_type
        self.embedding_model_name = embedding_model_name
        self._extra_config = kwargs

        self._graph_store = None
        self._vector_store = None
        self._embedding_model = None
        self._initialized = False

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self, repo_root: Optional[Path] = None) -> None:
        """Initialize both stores."""
        if self._initialized:
            return

        if repo_root:
            self.repo_root = repo_root.resolve()
            self.persist_directory = self.repo_root / ".victor"

        # Ensure directories exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite graph store
        from victor.storage.graph.sqlite_store import SqliteGraphStore

        graph_db_path = self.persist_directory / "project.db"
        self._graph_store = SqliteGraphStore(str(graph_db_path))
        await self._graph_store.initialize()

        # Initialize embedding model
        from victor.coding.codebase.embeddings.models import (
            EmbeddingModelConfig,
            create_embedding_model,
        )

        model_config = EmbeddingModelConfig(
            model_type=self.embedding_model_type,
            model_name=self.embedding_model_name,
        )
        self._embedding_model = create_embedding_model(model_config)
        await self._embedding_model.initialize()

        # Initialize LanceDB vector store (vectors only, no content)
        await self._init_vector_store()

        self._initialized = True
        logger.info(
            f"Initialized SqliteLanceDBStore for {self.repo_root} "
            f"(graph: SQLite, vectors: LanceDB, model: {self.embedding_model_name})"
        )

    async def _init_vector_store(self) -> None:
        """Initialize LanceDB for vector storage."""
        try:
            import lancedb

            embeddings_dir = self.persist_directory / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            self._vector_store = lancedb.connect(str(embeddings_dir))
            self._vector_table = None

            # Check if table exists
            if "symbols" in self._vector_store.table_names():
                self._vector_table = self._vector_store.open_table("symbols")
        except ImportError:
            logger.warning("LanceDB not available. Semantic search disabled.")
            self._vector_store = None

    async def close(self) -> None:
        """Clean up resources."""
        if self._embedding_model:
            await self._embedding_model.close()
            self._embedding_model = None
        if self._graph_store:
            await self._graph_store.close()
            self._graph_store = None
        self._vector_store = None
        self._vector_table = None
        self._initialized = False

    # =========================================================================
    # Unified ID Helpers
    # =========================================================================

    def make_symbol_id(self, rel_path: str, symbol_name: str) -> str:
        """Create unified ID for symbol.

        Uses repo-relative path to avoid conflicts when multiple files
        have the same name (e.g., protocol.py in different modules).
        """
        return str(UnifiedId.for_symbol(rel_path, symbol_name))

    def make_file_id(self, rel_path: str) -> str:
        """Create unified ID for file."""
        return str(UnifiedId.for_file(rel_path))

    def parse_id(self, unified_id: str) -> UnifiedId:
        """Parse unified ID string."""
        return UnifiedId.from_string(unified_id)

    # =========================================================================
    # Indexing (writes to both graph AND vector store)
    # =========================================================================

    async def index_symbol(
        self,
        symbol: UnifiedSymbol,
        embedding_text: str,
    ) -> None:
        """Index a single symbol (graph + embedding)."""
        if not self._initialized:
            await self.initialize()

        # Create graph node
        node = GraphNode(
            node_id=symbol.unified_id,
            type=symbol.type,
            name=symbol.name,
            file=symbol.file_path,
            line=symbol.line,
            end_line=symbol.end_line,
            lang=symbol.lang,
            signature=symbol.signature,
            docstring=symbol.docstring,
            parent_id=symbol.parent_id,
            embedding_ref=symbol.unified_id,  # Link to vector store
            metadata=symbol.metadata,
        )
        await self._graph_store.upsert_nodes([node])

        # Create embedding (vector only, no content)
        if self._vector_store and embedding_text:
            vector = await self._embedding_model.embed_text(embedding_text)
            await self._upsert_vector(
                doc_id=symbol.unified_id,
                vector=vector,
                metadata={
                    "file_path": symbol.file_path,
                    "symbol_type": symbol.type,
                    "line_number": symbol.line,
                },
            )

    async def index_symbols_batch(
        self,
        symbols: List[Tuple[UnifiedSymbol, str]],
        batch_size: int = 500,
    ) -> int:
        """Batch index symbols. Returns count indexed."""
        if not self._initialized:
            await self.initialize()

        if not symbols:
            return 0

        # Prepare graph nodes
        nodes = []
        embedding_items = []

        for symbol, embedding_text in symbols:
            nodes.append(
                GraphNode(
                    node_id=symbol.unified_id,
                    type=symbol.type,
                    name=symbol.name,
                    file=symbol.file_path,
                    line=symbol.line,
                    end_line=symbol.end_line,
                    lang=symbol.lang,
                    signature=symbol.signature,
                    docstring=symbol.docstring,
                    parent_id=symbol.parent_id,
                    embedding_ref=symbol.unified_id,
                    metadata=symbol.metadata,
                )
            )

            if embedding_text:
                embedding_items.append(
                    {
                        "id": symbol.unified_id,
                        "text": embedding_text,
                        "metadata": {
                            "file_path": symbol.file_path,
                            "symbol_type": symbol.type,
                            "line_number": symbol.line,
                        },
                    }
                )

        # Batch upsert to graph
        await self._graph_store.upsert_nodes(nodes)

        # Batch upsert to vector store
        if self._vector_store and embedding_items:
            # Process in batches
            indexed = 0
            for i in range(0, len(embedding_items), batch_size):
                batch = embedding_items[i : i + batch_size]
                texts = [item["text"] for item in batch]

                # Batch embed
                vectors = await self._embedding_model.embed_batch(texts)

                # Prepare LanceDB documents (no content, just vector + metadata)
                lance_docs = []
                for item, vector in zip(batch, vectors, strict=False):
                    lance_docs.append(
                        {
                            "id": item["id"],
                            "vector": vector,
                            **item["metadata"],
                        }
                    )

                # Upsert to LanceDB
                if self._vector_table is None:
                    self._vector_table = self._vector_store.create_table("symbols", data=lance_docs)
                else:
                    self._vector_table.add(lance_docs)

                indexed += len(batch)
                if indexed % 5000 == 0:
                    logger.info(f"Indexed {indexed}/{len(embedding_items)} embeddings...")

            return indexed

        return len(nodes)

    async def index_edge(self, edge: UnifiedEdge) -> None:
        """Index an edge."""
        if not self._initialized:
            await self.initialize()

        graph_edge = GraphEdge(
            src=edge.src_id,
            dst=edge.dst_id,
            type=edge.type,
            weight=edge.weight,
            metadata=edge.metadata,
        )
        await self._graph_store.upsert_edges([graph_edge])

    async def index_edges_batch(self, edges: List[UnifiedEdge]) -> int:
        """Batch index edges. Returns count indexed."""
        if not self._initialized:
            await self.initialize()

        if not edges:
            return 0

        graph_edges = [
            GraphEdge(
                src=e.src_id,
                dst=e.dst_id,
                type=e.type,
                weight=e.weight,
                metadata=e.metadata,
            )
            for e in edges
        ]
        await self._graph_store.upsert_edges(graph_edges)
        return len(graph_edges)

    async def _upsert_vector(
        self,
        doc_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert single vector to LanceDB."""
        if not self._vector_store:
            return

        doc = {
            "id": doc_id,
            "vector": vector,
            **(metadata or {}),
        }

        if self._vector_table is None:
            self._vector_table = self._vector_store.create_table("symbols", data=[doc])
        else:
            self._vector_table.add([doc])

    # =========================================================================
    # Unified Search
    # =========================================================================

    async def search(self, params: SearchParams) -> List[SearchResult]:
        """Unified search combining keyword, semantic, and graph.

        Algorithm:
        1. If HYBRID or SEMANTIC: vector search for similar symbols
        2. If HYBRID or KEYWORD: FTS search in graph
        3. Weight results by graph importance (PageRank)
        4. Merge and rank by combined score
        5. Optionally add caller/callee context
        """
        if not self._initialized:
            await self.initialize()

        results: Dict[str, SearchResult] = {}

        # Semantic search
        if params.mode in (SearchMode.HYBRID, SearchMode.SEMANTIC):
            semantic_results = await self._semantic_search(
                params.query,
                limit=params.limit * 2,  # Over-fetch for merging
                threshold=params.similarity_threshold,
            )
            for r in semantic_results:
                results[r.symbol.unified_id] = r

        # Keyword search (FTS)
        if params.mode in (SearchMode.HYBRID, SearchMode.KEYWORD):
            keyword_results = await self._keyword_search(
                params.query,
                limit=params.limit * 2,
                symbol_types=params.symbol_types,
            )
            for r in keyword_results:
                if r.symbol.unified_id in results:
                    # Merge scores
                    existing = results[r.symbol.unified_id]
                    existing.keyword_score = r.keyword_score
                    # Recalculate combined score
                    existing.score = self._combine_scores(
                        semantic=existing.semantic_score,
                        keyword=existing.keyword_score,
                        graph=existing.graph_score,
                        semantic_weight=params.semantic_weight,
                        graph_weight=params.graph_weight,
                    )
                else:
                    results[r.symbol.unified_id] = r

        # Sort by combined score
        sorted_results = sorted(
            results.values(),
            key=lambda r: r.score,
            reverse=True,
        )[: params.limit]

        # Optionally include neighbors
        if params.include_neighbors:
            for result in sorted_results:
                symbol = result.symbol
                # Fetch callers/callees
                callers = await self.get_callers(symbol.unified_id, max_depth=1)
                callees = await self.get_callees(symbol.unified_id, max_depth=1)
                symbol.callers = [c.unified_id for c in callers[:5]]
                symbol.callees = [c.unified_id for c in callees[:5]]

        return sorted_results

    async def search_semantic(
        self,
        query: str,
        limit: int = 20,
        threshold: float = 0.25,
    ) -> List[SearchResult]:
        """Pure semantic search (vector similarity)."""
        return await self._semantic_search(query, limit, threshold)

    async def search_keyword(
        self,
        query: str,
        limit: int = 20,
        symbol_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Pure keyword search (FTS)."""
        return await self._keyword_search(query, limit, symbol_types)

    async def _semantic_search(
        self,
        query: str,
        limit: int,
        threshold: float,
    ) -> List[SearchResult]:
        """Internal semantic search."""
        if not self._vector_store or not self._vector_table:
            return []

        # Embed query
        query_vector = await self._embedding_model.embed_text(query)

        # Search LanceDB
        results = self._vector_table.search(query_vector).limit(limit).to_list()

        search_results = []
        for r in results:
            # Convert distance to similarity score
            distance = r.get("_distance", 0.0)
            score = 1.0 / (1.0 + distance)

            if score < threshold:
                continue

            # Get full symbol from graph
            unified_id = r.get("id", "")
            symbol = await self.get_symbol(unified_id)
            if not symbol:
                continue

            search_results.append(
                SearchResult(
                    symbol=symbol,
                    score=score,
                    match_type="semantic",
                    semantic_score=score,
                )
            )

        return search_results

    async def _keyword_search(
        self,
        query: str,
        limit: int,
        symbol_types: Optional[List[str]],
    ) -> List[SearchResult]:
        """Internal keyword search using graph FTS."""
        if not self._graph_store:
            return []

        nodes = await self._graph_store.search_symbols(
            query=query,
            limit=limit,
            symbol_types=symbol_types,
        )

        results = []
        for node in nodes:
            symbol = self._node_to_symbol(node)
            results.append(
                SearchResult(
                    symbol=symbol,
                    score=1.0,  # FTS doesn't provide scores
                    match_type="keyword",
                    keyword_score=1.0,
                )
            )

        return results

    def _combine_scores(
        self,
        semantic: Optional[float],
        keyword: Optional[float],
        graph: Optional[float],
        semantic_weight: float,
        graph_weight: float,
    ) -> float:
        """Combine multiple scores into single relevance score."""
        total = 0.0
        weight_sum = 0.0

        if semantic is not None:
            total += semantic * semantic_weight
            weight_sum += semantic_weight

        if keyword is not None:
            keyword_weight = 1.0 - semantic_weight - graph_weight
            total += keyword * keyword_weight
            weight_sum += keyword_weight

        if graph is not None:
            total += graph * graph_weight
            weight_sum += graph_weight

        return total / weight_sum if weight_sum > 0 else 0.0

    # =========================================================================
    # Graph Queries
    # =========================================================================

    async def get_symbol(self, unified_id: str) -> Optional[UnifiedSymbol]:
        """Get symbol by unified ID."""
        if not self._initialized:
            await self.initialize()

        node = await self._graph_store.get_node_by_id(unified_id)
        if not node:
            return None

        return self._node_to_symbol(node)

    async def get_symbols_in_file(self, rel_path: str) -> List[UnifiedSymbol]:
        """Get all symbols in a file."""
        if not self._initialized:
            await self.initialize()

        nodes = await self._graph_store.get_nodes_by_file(rel_path)
        return [self._node_to_symbol(n) for n in nodes]

    async def get_callers(
        self,
        unified_id: str,
        max_depth: int = 1,
    ) -> List[UnifiedSymbol]:
        """Get functions that call this symbol."""
        if not self._initialized:
            await self.initialize()

        edges = await self._graph_store.get_neighbors(
            unified_id,
            edge_types=["CALLS"],
            max_depth=max_depth,
        )

        # Get incoming edges (callers)
        caller_ids = [e.src for e in edges if e.dst == unified_id]
        callers = []
        for cid in caller_ids:
            symbol = await self.get_symbol(cid)
            if symbol:
                callers.append(symbol)

        return callers

    async def get_callees(
        self,
        unified_id: str,
        max_depth: int = 1,
    ) -> List[UnifiedSymbol]:
        """Get functions called by this symbol."""
        if not self._initialized:
            await self.initialize()

        edges = await self._graph_store.get_neighbors(
            unified_id,
            edge_types=["CALLS"],
            max_depth=max_depth,
        )

        # Get outgoing edges (callees)
        callee_ids = [e.dst for e in edges if e.src == unified_id]
        callees = []
        for cid in callee_ids:
            symbol = await self.get_symbol(cid)
            if symbol:
                callees.append(symbol)

        return callees

    async def get_related(
        self,
        unified_id: str,
        edge_types: Optional[List[str]] = None,
    ) -> List[Tuple[UnifiedSymbol, str]]:
        """Get related symbols with relationship type."""
        if not self._initialized:
            await self.initialize()

        edges = await self._graph_store.get_neighbors(
            unified_id,
            edge_types=edge_types,
        )

        results = []
        for edge in edges:
            related_id = edge.dst if edge.src == unified_id else edge.src
            symbol = await self.get_symbol(related_id)
            if symbol:
                results.append((symbol, edge.type))

        return results

    # =========================================================================
    # Semantic + Graph Combined Queries
    # =========================================================================

    async def find_similar_symbols(
        self,
        unified_id: str,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Find semantically similar symbols to given symbol."""
        if not self._initialized:
            await self.initialize()

        # Get the symbol
        symbol = await self.get_symbol(unified_id)
        if not symbol:
            return []

        # Build query text from symbol
        query_parts = [symbol.name]
        if symbol.signature:
            query_parts.append(symbol.signature)
        if symbol.docstring:
            query_parts.append(symbol.docstring[:200])

        query = " ".join(query_parts)
        return await self.search_semantic(query, limit=limit)

    async def semantic_blast_radius(
        self,
        unified_id: str,
        similarity_threshold: float = 0.5,
    ) -> List[SearchResult]:
        """Find symbols that might be affected by changes (graph + semantic).

        Combines:
        1. Graph callers (direct impact)
        2. Semantically similar code (might need similar changes)
        """
        if not self._initialized:
            await self.initialize()

        results: Dict[str, SearchResult] = {}

        # Get graph callers (structural impact)
        callers = await self.get_callers(unified_id, max_depth=2)
        for caller in callers:
            results[caller.unified_id] = SearchResult(
                symbol=caller,
                score=0.9,  # High confidence - direct callers
                match_type="graph",
                graph_score=0.9,
            )

        # Get semantically similar (semantic impact)
        similar = await self.find_similar_symbols(unified_id, limit=20)
        for r in similar:
            if r.symbol.unified_id == unified_id:
                continue  # Skip self
            if (r.semantic_score or 0) < similarity_threshold:
                continue

            if r.symbol.unified_id in results:
                # Merge - add semantic score
                existing = results[r.symbol.unified_id]
                existing.semantic_score = r.semantic_score
                existing.score = max(existing.score, r.score)
            else:
                results[r.symbol.unified_id] = r

        return sorted(results.values(), key=lambda r: r.score, reverse=True)

    # =========================================================================
    # Maintenance
    # =========================================================================

    async def delete_file(self, rel_path: str) -> None:
        """Delete all symbols for a file (graph + embeddings)."""
        if not self._initialized:
            await self.initialize()

        # Delete from graph
        await self._graph_store.delete_by_file(rel_path)

        # Delete from vector store (by file_path prefix)
        if self._vector_table:
            # LanceDB delete by predicate
            self._vector_table.delete(f"file_path = '{rel_path}'")

    async def delete_all(self) -> None:
        """Clear entire store."""
        if not self._initialized:
            await self.initialize()

        await self._graph_store.delete_by_repo()

        if self._vector_store and "symbols" in self._vector_store.table_names():
            self._vector_store.drop_table("symbols")
            self._vector_table = None

    async def stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        if not self._initialized:
            await self.initialize()

        graph_stats = await self._graph_store.stats()

        vector_count = 0
        if self._vector_table:
            try:
                vector_count = self._vector_table.count_rows()
            except Exception:
                pass

        return {
            "backend": "sqlite+lancedb",
            "repo_root": str(self.repo_root),
            "graph": graph_stats,
            "vectors": {
                "count": vector_count,
                "model": self.embedding_model_name,
            },
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _node_to_symbol(self, node: GraphNode) -> UnifiedSymbol:
        """Convert GraphNode to UnifiedSymbol."""
        return UnifiedSymbol(
            unified_id=node.node_id,
            name=node.name,
            type=node.type,
            file_path=node.file,
            line=node.line,
            end_line=node.end_line,
            lang=node.lang,
            signature=node.signature,
            docstring=node.docstring,
            parent_id=node.parent_id,
            metadata=node.metadata,
        )


__all__ = ["SqliteLanceDBStore"]
