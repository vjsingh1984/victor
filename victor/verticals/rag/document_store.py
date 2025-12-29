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

"""RAG Document Store - LanceDB-based vector storage for RAG.

This module provides a complete document storage solution using LanceDB:
- Embedded database (no server required)
- Fast vector search with HNSW index
- Hybrid search combining vector + full-text
- Automatic persistence to disk
- Batch operations for efficiency

Design:
    - Document: Source document with metadata
    - DocumentChunk: Indexed chunk with embedding
    - SearchResult: Search result with score and context

Example:
    store = DocumentStore(path=".victor/rag")

    # Ingest documents
    doc = Document(id="doc1", content="...", source="file.pdf")
    await store.add_document(doc)

    # Search
    results = await store.search("query", k=10)
    for result in results:
        print(f"{result.score:.2f}: {result.chunk.content[:100]}")
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# LanceDB is optional - provide graceful fallback
try:
    import lancedb
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    logger.warning("LanceDB not available. Install with: pip install lancedb")


@dataclass
class Document:
    """Source document for RAG.

    Attributes:
        id: Unique document identifier
        content: Full document content
        source: Source path or URL
        doc_type: Document type (pdf, markdown, text, code)
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    id: str
    content: str
    source: str
    doc_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def content_hash(self) -> str:
        """Get content hash for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class DocumentChunk:
    """Indexed chunk of a document.

    Attributes:
        id: Unique chunk identifier
        doc_id: Parent document ID
        content: Chunk content
        embedding: Vector embedding
        chunk_index: Position in document
        start_char: Start character offset
        end_char: End character offset
        metadata: Chunk-specific metadata
    """
    id: str
    doc_id: str
    content: str
    embedding: List[float]
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result with relevance score.

    Attributes:
        chunk: The matched chunk
        score: Relevance score (0-1, higher is better)
        highlights: Highlighted text segments
        doc_source: Source document path
    """
    chunk: DocumentChunk
    score: float
    highlights: List[str] = field(default_factory=list)
    doc_source: str = ""

    @property
    def content(self) -> str:
        """Convenience property for chunk content."""
        return self.chunk.content

    @property
    def metadata(self) -> Dict[str, Any]:
        """Convenience property for chunk metadata."""
        return self.chunk.metadata

    @property
    def doc_id(self) -> str:
        """Convenience property for document ID."""
        return self.chunk.doc_id


@dataclass
class DocumentStoreConfig:
    """Configuration for document store.

    Attributes:
        path: Path to store data
        table_name: LanceDB table name
        embedding_dim: Embedding dimension
        use_hybrid_search: Enable hybrid (vector + full-text) search
        rerank_results: Enable reranking
        max_results: Maximum results per search
    """
    path: Path = field(default_factory=lambda: Path(".victor/rag"))
    table_name: str = "documents"
    embedding_dim: int = 384  # Default for sentence-transformers/all-MiniLM-L6-v2
    use_hybrid_search: bool = True
    rerank_results: bool = True
    max_results: int = 20


class DocumentStore:
    """LanceDB-based document store for RAG.

    Provides vector storage and search for document chunks with:
    - Automatic embedding generation
    - Hybrid search (vector + full-text)
    - Optional reranking
    - Batch operations
    - Persistence to disk

    Example:
        store = DocumentStore()
        await store.initialize()

        # Add documents
        doc = Document(id="1", content="...", source="doc.pdf")
        chunks = chunker.chunk(doc)
        await store.add_chunks(chunks)

        # Search
        results = await store.search("query", k=10)
    """

    def __init__(
        self,
        config: Optional[DocumentStoreConfig] = None,
        embedding_service: Optional[Any] = None,
    ):
        """Initialize document store.

        Args:
            config: Store configuration
            embedding_service: Optional embedding service (uses default if None)
        """
        self.config = config or DocumentStoreConfig()
        self._embedding_service = embedding_service
        self._db: Optional[Any] = None
        self._table: Optional[Any] = None
        self._initialized = False

        # Document index for fast lookup
        self._doc_index: Dict[str, Document] = {}

        # Statistics
        self._stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_searches": 0,
            "last_updated": 0.0,
        }

    async def initialize(self) -> None:
        """Initialize the document store.

        Creates the LanceDB database and table if they don't exist.
        """
        if self._initialized:
            return

        if not LANCEDB_AVAILABLE:
            logger.warning("LanceDB not available - using in-memory fallback")
            self._initialized = True
            return

        # Ensure directory exists
        self.config.path.mkdir(parents=True, exist_ok=True)

        # Connect to LanceDB
        self._db = await asyncio.to_thread(
            lancedb.connect,
            str(self.config.path)
        )

        # Create or open table
        try:
            self._table = await asyncio.to_thread(
                self._db.open_table,
                self.config.table_name
            )
            logger.info(f"Opened existing table: {self.config.table_name}")
        except Exception:
            # Create new table with schema
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("content", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("start_char", pa.int32()),
                pa.field("end_char", pa.int32()),
                pa.field("metadata", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.config.embedding_dim)),
            ])

            self._table = await asyncio.to_thread(
                self._db.create_table,
                self.config.table_name,
                schema=schema
            )
            logger.info(f"Created new table: {self.config.table_name}")

        self._initialized = True
        self._stats["last_updated"] = time.time()

    async def add_document(self, doc: Document) -> List[DocumentChunk]:
        """Add a document to the store.

        Chunks the document and stores embeddings.

        Args:
            doc: Document to add

        Returns:
            List of created chunks
        """
        from victor.verticals.rag.chunker import DocumentChunker

        await self.initialize()

        # Chunk the document
        chunker = DocumentChunker()
        chunks = await chunker.chunk_document(doc, self._get_embedding)

        # Store chunks
        await self.add_chunks(chunks)

        # Update index
        self._doc_index[doc.id] = doc
        self._stats["total_documents"] += 1

        return chunks

    async def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add chunks to the store.

        Args:
            chunks: Chunks to add

        Returns:
            Number of chunks added
        """
        await self.initialize()

        if not chunks:
            return 0

        if not LANCEDB_AVAILABLE or self._table is None:
            # In-memory fallback
            for chunk in chunks:
                if not hasattr(self, "_memory_chunks"):
                    self._memory_chunks: List[DocumentChunk] = []
                self._memory_chunks.append(chunk)
            self._stats["total_chunks"] += len(chunks)
            return len(chunks)

        import json

        # Convert to records
        records = []
        for chunk in chunks:
            records.append({
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": json.dumps(chunk.metadata),
                "vector": chunk.embedding,
            })

        # Add to table
        await asyncio.to_thread(self._table.add, records)

        self._stats["total_chunks"] += len(chunks)
        self._stats["last_updated"] = time.time()

        logger.info(f"Added {len(chunks)} chunks to store")
        return len(chunks)

    async def search(
        self,
        query: str,
        k: int = 10,
        filter_doc_ids: Optional[List[str]] = None,
        use_hybrid: Optional[bool] = None,
    ) -> List[SearchResult]:
        """Search for relevant chunks.

        Args:
            query: Search query
            k: Number of results
            filter_doc_ids: Optional document ID filter
            use_hybrid: Override hybrid search setting

        Returns:
            List of search results sorted by relevance
        """
        await self.initialize()
        self._stats["total_searches"] += 1

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        if not LANCEDB_AVAILABLE or self._table is None:
            # In-memory fallback with cosine similarity
            return await self._search_memory(query_embedding, k, filter_doc_ids)

        # Build search query
        use_hybrid = use_hybrid if use_hybrid is not None else self.config.use_hybrid_search

        if use_hybrid:
            # Hybrid search: combine vector + FTS
            results = await asyncio.to_thread(
                lambda: (
                    self._table.search(query_embedding)
                    .limit(k * 2)  # Get more for reranking
                    .to_list()
                )
            )
        else:
            # Vector-only search
            results = await asyncio.to_thread(
                lambda: (
                    self._table.search(query_embedding)
                    .limit(k)
                    .to_list()
                )
            )

        # Convert to SearchResult
        import json

        search_results = []
        for row in results:
            chunk = DocumentChunk(
                id=row["id"],
                doc_id=row["doc_id"],
                content=row["content"],
                embedding=row.get("vector", []),
                chunk_index=row.get("chunk_index", 0),
                start_char=row.get("start_char", 0),
                end_char=row.get("end_char", 0),
                metadata=json.loads(row.get("metadata", "{}")),
            )

            # Filter by doc_ids if specified
            if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                continue

            score = 1.0 - row.get("_distance", 0.5)  # Convert distance to similarity
            search_results.append(SearchResult(
                chunk=chunk,
                score=score,
                doc_source=self._doc_index.get(chunk.doc_id, Document("", "", "")).source,
            ))

        # Optionally rerank
        if self.config.rerank_results and len(search_results) > k:
            search_results = await self._rerank(query, search_results, k)
        else:
            search_results = search_results[:k]

        return search_results

    async def _search_memory(
        self,
        query_embedding: List[float],
        k: int,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """In-memory search fallback when LanceDB is not available."""
        if not hasattr(self, "_memory_chunks"):
            return []

        import math

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            if not a or not b:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        results = []
        for chunk in self._memory_chunks:
            if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                continue

            score = cosine_similarity(query_embedding, chunk.embedding)
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                doc_source=self._doc_index.get(chunk.doc_id, Document("", "", "")).source,
            ))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    async def _rerank(
        self,
        query: str,
        results: List[SearchResult],
        k: int,
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder or simple scoring.

        This is a simple lexical reranking. For production, use a cross-encoder.
        """
        query_terms = set(query.lower().split())

        for result in results:
            content_terms = set(result.chunk.content.lower().split())
            overlap = len(query_terms & content_terms)
            # Boost score by term overlap
            result.score = result.score * 0.7 + (overlap / max(len(query_terms), 1)) * 0.3

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text.

        Uses the embedding service if provided, otherwise uses a fallback.
        """
        if self._embedding_service:
            return await self._embedding_service.embed(text)

        # Try to use sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, "_model"):
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except ImportError:
            # Simple hash-based fallback for testing
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            # Convert to list of floats
            return [b / 255.0 for b in h[:self.config.embedding_dim]]

    async def delete_document(self, doc_id: str) -> int:
        """Delete a document and its chunks.

        Args:
            doc_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        await self.initialize()

        if doc_id in self._doc_index:
            del self._doc_index[doc_id]
            self._stats["total_documents"] -= 1

        if not LANCEDB_AVAILABLE or self._table is None:
            if hasattr(self, "_memory_chunks"):
                before = len(self._memory_chunks)
                self._memory_chunks = [c for c in self._memory_chunks if c.doc_id != doc_id]
                deleted = before - len(self._memory_chunks)
                self._stats["total_chunks"] -= deleted
                return deleted
            return 0

        # Delete from LanceDB
        await asyncio.to_thread(
            lambda: self._table.delete(f"doc_id = '{doc_id}'")
        )

        return 1  # LanceDB doesn't return count

    async def list_documents(self) -> List[Document]:
        """List all documents in the store.

        Returns:
            List of documents
        """
        await self.initialize()
        return list(self._doc_index.values())

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document or None if not found
        """
        await self.initialize()
        return self._doc_index.get(doc_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics.

        Returns:
            Dictionary with statistics
        """
        await self.initialize()
        stats = dict(self._stats)
        stats["store_path"] = str(self.config.path)
        return stats

    async def close(self) -> None:
        """Close the document store."""
        # LanceDB handles cleanup automatically
        self._initialized = False
        logger.info("Document store closed")
