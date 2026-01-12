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

"""LanceDB embedding provider for production deployments.

LanceDB is a modern, fast, embedded vector database perfect for:
- Production deployments (scales to billions of vectors)
- Fast similarity search with disk-based indices
- Serverless and embedded mode
- Zero-copy integration with Apache Arrow

Install: pip install lancedb

For embedding models:
- Ollama: pip install httpx (then: ollama pull qwen3-embedding:8b)
- Sentence-transformers: pip install sentence-transformers
- OpenAI: pip install openai (requires API key)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import lancedb

    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

from victor.storage.vector_stores.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingSearchResult,
)
from victor.storage.vector_stores.models import (
    BaseEmbeddingModel,
    EmbeddingModelConfig,
    create_embedding_model,
)


class LanceDBProvider(BaseEmbeddingProvider):
    """LanceDB embedding provider.

    Uses LanceDB for vector storage with pluggable embedding models.

    Features:
    - Disk-based storage with mmap for efficiency
    - Scales to billions of vectors
    - Fast ANN search with disk-based indices
    - Automatic embedding generation
    - Metadata filtering
    - Support for multiple embedding models (Ollama, OpenAI, Sentence-transformers, Cohere)
    - Zero-copy reads via Apache Arrow

    Advantages over ChromaDB:
    - Better performance for large datasets (>100k docs)
    - Lower memory footprint (disk-based)
    - Faster search with ANN indices
    - Production-ready scalability
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize LanceDB provider.

        Args:
            config: Embedding configuration
        """
        super().__init__(config)

        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB not available. Install with: pip install lancedb")

        self.db = None
        self.table = None
        self.embedding_model: Optional[BaseEmbeddingModel] = None

    async def initialize(self) -> None:
        """Initialize LanceDB and load embedding model."""
        if self._initialized:
            return

        # Get embedding model configuration from EmbeddingConfig
        model_type = self.config.embedding_model_type
        model_name = self.config.embedding_model_name
        api_key = self.config.embedding_api_key
        dimension = self.config.extra_config.get("dimension", 4096)
        batch_size = self.config.extra_config.get("batch_size", 16)

        # Create embedding model config
        embedding_config = EmbeddingModelConfig(
            model_type=model_type,
            model_name=model_name,
            dimension=dimension,
            api_key=api_key,
            batch_size=batch_size,
        )

        # Initialize embedding model
        self.embedding_model = create_embedding_model(embedding_config)
        await self.embedding_model.initialize()

        print("🔧 Initializing LanceDB provider")
        print("📦 Vector Store: LanceDB")
        print(f"🤖 Embedding Model: {model_name} ({model_type})")

        # Setup LanceDB
        persist_dir = self.config.persist_directory
        if persist_dir:
            persist_dir = Path(persist_dir).expanduser()
            persist_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Using persistent storage: {persist_dir}")
        else:
            # LanceDB requires a directory, use centralized path
            from victor.config.settings import get_project_paths

            persist_dir = get_project_paths().global_embeddings_dir / "lancedb"
            persist_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Using default storage: {persist_dir}")

        # Connect to LanceDB
        self.db = lancedb.connect(str(persist_dir))

        # Get or create table
        table_name = self.config.extra_config.get("table_name", "embeddings")
        print(f"📚 Table: {table_name}")

        # Check if table exists
        try:
            existing_tables = self.db.table_names()
        except AttributeError:
            # Fallback for older LanceDB versions
            existing_tables = []
            try:
                existing_tables = (
                    self.db.list_tables().tables if hasattr(self.db, "list_tables") else []
                )
            except Exception:
                # If list_tables also fails, try to open table directly and catch exception
                existing_tables = []

        if table_name not in existing_tables:
            # Create empty table with schema
            # We'll add data later when indexing
            print(f"📝 Creating new table: {table_name}")
        else:
            self.table = self.db.open_table(table_name)
            print(f"📖 Opened existing table: {table_name}")

        self._initialized = True
        print("✅ LanceDB provider initialized!")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self._initialized:
            await self.initialize()

        return await self.embedding_model.embed_text(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()

        return await self.embedding_model.embed_batch(texts)

    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a single document.

        Args:
            doc_id: Document identifier
            content: Document content
            metadata: Optional metadata
        """
        if not self._initialized:
            await self.initialize()

        # Generate embedding
        embedding = await self.embed_text(content)

        # Prepare document
        document = {
            "id": doc_id,
            "vector": embedding,
            "content": content,
            **(metadata or {}),
        }

        # Add to table (create if doesn't exist)
        table_name = self.config.extra_config.get("table_name", "embeddings")
        if self.table is None:
            self.table = self.db.create_table(table_name, data=[document])
        else:
            self.table.add([document])

    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index multiple documents in batch.

        Args:
            documents: List of documents with id, content, metadata
        """
        if not self._initialized:
            await self.initialize()

        if not documents:
            return

        # Generate embeddings in batch
        contents = [doc["content"] for doc in documents]
        embeddings = await self.embed_batch(contents)

        # Prepare documents for insertion
        lance_docs = []
        for doc, embedding in zip(documents, embeddings, strict=False):
            lance_doc = {
                "id": doc["id"],
                "vector": embedding,
                "content": doc["content"],
                **doc.get("metadata", {}),
            }
            lance_docs.append(lance_doc)

        # Add to table (create if doesn't exist)
        table_name = self.config.extra_config.get("table_name", "embeddings")
        if self.table is None:
            self.table = self.db.create_table(table_name, data=lance_docs)
        else:
            self.table.add(lance_docs)

    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EmbeddingSearchResult]:
        """Search for similar documents.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        if self.table is None:
            return []

        # Generate query embedding
        query_embedding = await self.embed_text(query)

        # Search in LanceDB
        results = self.table.search(query_embedding).limit(limit)

        # Apply metadata filters if provided
        if filter_metadata:
            for key, value in filter_metadata.items():
                results = results.where(f"{key} = '{value}'")

        # Execute search
        search_results = []
        for result in results.to_list():
            # LanceDB returns distance (lower is better)
            # Convert to similarity score (0-1, higher is better)
            distance = result.get("_distance", 0.0)
            score = 1.0 / (1.0 + distance)  # Convert distance to similarity

            search_results.append(
                EmbeddingSearchResult(
                    file_path=result.get("file_path", ""),
                    symbol_name=result.get("symbol_name"),
                    content=result.get("content", ""),
                    score=score,
                    line_number=result.get("line_number"),
                    metadata={k: v for k, v in result.items() if not k.startswith("_")},
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

        if self.table is None:
            return

        self.table.delete(f"id = '{doc_id}'")

    async def delete_by_file(self, file_path: str) -> int:
        """Delete all documents from a specific file.

        Used for incremental updates - when a file changes, we delete all
        its chunks and re-index.

        Args:
            file_path: Relative file path to delete documents for

        Returns:
            Number of documents deleted
        """
        if not self._initialized:
            await self.initialize()

        if self.table is None:
            return 0

        # Count documents before deletion
        try:
            count_before = self.table.count_rows()
        except (AttributeError, RuntimeError, ValueError):
            count_before = 0

        # Delete documents with matching file_path
        # LanceDB uses SQL-like predicates
        self.table.delete(f"file_path = '{file_path}'")

        # Count documents after deletion
        try:
            count_after = self.table.count_rows()
        except (AttributeError, RuntimeError, ValueError):
            count_after = 0

        return count_before - count_after

    async def clear_index(self) -> None:
        """Clear entire index."""
        if not self._initialized:
            await self.initialize()

        # Drop and recreate table
        table_name = self.config.extra_config.get("table_name", "embeddings")
        if table_name in self.db.list_tables().tables:
            self.db.drop_table(table_name)

        self.table = None
        print("🗑️  Cleared index")

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with statistics
        """
        if not self._initialized:
            await self.initialize()

        count = 0
        if self.table is not None:
            try:
                count = self.table.count_rows()
            except (AttributeError, RuntimeError, ValueError):
                count = 0

        return {
            "provider": "lancedb",
            "total_documents": count,
            "embedding_model_type": self.config.embedding_model_type,
            "embedding_model_name": self.config.embedding_model_name,
            "dimension": self.embedding_model.get_dimension() if self.embedding_model else 4096,
            "distance_metric": self.config.distance_metric,
            "table_name": self.config.extra_config.get("table_name", "embeddings"),
            "persist_directory": self.config.persist_directory,
        }

    async def close(self) -> None:
        """Clean up resources."""
        if self.embedding_model:
            await self.embedding_model.close()
            self.embedding_model = None

        # LanceDB connections are lightweight, no explicit cleanup needed
        self.db = None
        self.table = None
        self._initialized = False
