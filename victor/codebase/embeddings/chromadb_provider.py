"""ChromaDB embedding provider for development/testing.

ChromaDB is a lightweight, easy-to-use embedding database perfect for:
- Local development
- Small to medium codebases (< 100k documents)
- Quick prototyping

Install: pip install chromadb sentence-transformers
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from victor.codebase.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    SearchResult,
)


class ChromaDBProvider(BaseEmbeddingProvider):
    """ChromaDB embedding provider.

    Uses ChromaDB for vector storage and sentence-transformers for embeddings.

    Features:
    - In-memory or persistent storage
    - Automatic embedding generation
    - Metadata filtering
    - Easy setup (no external services)

    Limitations:
    - Not optimized for very large datasets (>100k docs)
    - Single-machine only
    - CPU-based inference (slower than GPU)
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize ChromaDB provider.

        Args:
            config: Embedding configuration
        """
        super().__init__(config)

        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB not available. Install with: "
                "pip install chromadb sentence-transformers"
            )

        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_model: Optional[SentenceTransformer] = None

    async def initialize(self) -> None:
        """Initialize ChromaDB and load embedding model."""
        if self._initialized:
            return

        print(f"🔧 Initializing ChromaDB provider (model: {self.config.model})")

        # Initialize ChromaDB client
        if self.config.persist_directory:
            persist_dir = Path(self.config.persist_directory).expanduser()
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.Client(
                Settings(
                    persist_directory=str(persist_dir),
                    anonymized_telemetry=False,
                )
            )
            print(f"📁 Using persistent storage: {persist_dir}")
        else:
            # In-memory mode (good for testing)
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False)
            )
            print("💾 Using in-memory storage")

        # Get or create collection
        collection_name = self.config.extra_config.get("collection_name", "victor_codebase")
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
            print(f"📚 Collection: {collection_name}")
        except Exception as e:
            print(f"⚠️  Warning: Could not create collection: {e}")
            self.collection = self.client.get_collection(collection_name)

        # Load embedding model (runs in executor to avoid blocking)
        print(f"🤖 Loading embedding model: {self.config.model}...")
        loop = asyncio.get_event_loop()
        self.embedding_model = await loop.run_in_executor(
            None, SentenceTransformer, self.config.model
        )

        self._initialized = True
        print("✅ ChromaDB provider initialized!")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self._initialized:
            await self.initialize()

        # Run embedding in executor (CPU-bound)
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self.embedding_model.encode, text
        )
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (optimized).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()

        # Batch encoding is much faster than individual
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.embedding_model.encode, texts
        )
        return [emb.tolist() for emb in embeddings]

    async def index_document(
        self, doc_id: str, content: str, metadata: Dict[str, Any]
    ) -> None:
        """Index a single document.

        Args:
            doc_id: Unique document identifier
            content: Content to index
            metadata: Additional metadata
        """
        if not self._initialized:
            await self.initialize()

        # Generate embedding
        embedding = await self.embed_text(content)

        # Add to ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Batch index multiple documents (optimized).

        Args:
            documents: List of documents with id, content, metadata
        """
        if not self._initialized:
            await self.initialize()

        if not documents:
            return

        print(f"📝 Indexing {len(documents)} documents...")

        # Extract data
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # Batch generate embeddings (much faster)
        embeddings = await self.embed_batch(contents)

        # Batch add to ChromaDB
        batch_size = self.config.extra_config.get("batch_size", 100)

        for i in range(0, len(documents), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_contents = contents[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_contents,
                metadatas=batch_metadatas,
            )

        print(f"✅ Indexed {len(documents)} documents")

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query
            limit: Maximum results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        # Generate query embedding
        query_embedding = await self.embed_text(query)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                # Convert distance to similarity score (0-1, higher is better)
                # For cosine distance: similarity = 1 - distance
                score = 1.0 - distance

                search_results.append(
                    SearchResult(
                        file_path=metadata.get("file_path", ""),
                        symbol_name=metadata.get("symbol_name"),
                        content=content,
                        score=score,
                        line_number=metadata.get("line_number"),
                        metadata=metadata,
                    )
                )

        return search_results

    async def delete_document(self, doc_id: str) -> None:
        """Delete a document from index.

        Args:
            doc_id: Document identifier
        """
        if not self._initialized:
            await self.initialize()

        self.collection.delete(ids=[doc_id])

    async def clear_index(self) -> None:
        """Clear entire index."""
        if not self._initialized:
            await self.initialize()

        # Delete collection and recreate
        collection_name = self.collection.name
        self.client.delete_collection(name=collection_name)

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )

        print("🗑️  Cleared index")

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            await self.initialize()

        count = self.collection.count()

        return {
            "provider": "chromadb",
            "total_documents": count,
            "model_name": self.config.model,
            "dimension": self.config.dimension,
            "distance_metric": self.config.distance_metric,
            "collection_name": self.collection.name,
            "persist_directory": self.config.persist_directory,
        }

    async def close(self) -> None:
        """Clean up resources."""
        if self.client and self.config.persist_directory:
            # ChromaDB auto-persists, no explicit save needed
            pass

        self._initialized = False
        print("👋 ChromaDB provider closed")
