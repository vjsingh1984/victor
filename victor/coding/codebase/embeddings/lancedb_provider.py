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
    import lancedb  # type: ignore[import-untyped]

    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

from victor.coding.codebase.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingSearchResult,
)
from victor.coding.codebase.embeddings.models import (
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
        # Store rebuild flag from config (for corruption recovery)
        self._rebuild_on_corruption = config.extra_config.get("rebuild_on_corruption", False)

    async def initialize(self, rebuild_on_corruption: bool = False) -> None:
        """Initialize LanceDB and load embedding model.

        Args:
            rebuild_on_corruption: If True, rebuild database if corrupted.
                If not provided, uses the value from config.
        """
        if self._initialized:
            return

        # Use parameter if provided, otherwise use stored value from config
        should_rebuild = rebuild_on_corruption or self._rebuild_on_corruption

        # Get embedding model configuration from EmbeddingConfig
        model_type = self.config.embedding_model_type
        model_name = self.config.embedding_model
        api_key = self.config.embedding_api_key
        dimension = self.config.extra_config.get("dimension", 4096)
        batch_size = self.config.extra_config.get("batch_size", 16)

        # Create embedding model config
        embedding_config = EmbeddingModelConfig(
            embedding_type=model_type,
            embedding_model=model_name,
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
        persist_dir_path: Path
        persist_dir = self.config.persist_directory
        if persist_dir:
            persist_dir_path = Path(persist_dir).expanduser()
            persist_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Using persistent storage: {persist_dir_path}")
        else:
            # LanceDB requires a directory, use centralized path
            from victor.config.settings import get_project_paths

            persist_dir_path = get_project_paths().global_embeddings_dir / "lancedb"
            persist_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Using default storage: {persist_dir_path}")

        # Check for database corruption and handle it
        table_name = self.config.extra_config.get("table_name", "embeddings")
        db_path = persist_dir_path / f"{table_name}.lance"

        # If rebuild requested or DB is corrupted, clean up
        if should_rebuild or db_path.exists():
            try:
                # Try to connect first to check if DB is valid
                self.db = lancedb.connect(str(persist_dir_path))
                if self.db is None:
                    raise RuntimeError("Failed to connect to LanceDB")
                existing_tables = self.db.list_tables().tables

                # Try to open the table to verify it's not corrupted
                if table_name in existing_tables:
                    try:
                        test_table = self.db.open_table(table_name)
                        # Try a simple operation to verify integrity
                        _ = test_table.count_rows()
                        self.table = test_table
                        print(f"📖 Opened existing table: {table_name}")
                    except (RuntimeError, Exception) as e:
                        error_msg = str(e).lower()
                        if "malformed" in error_msg or "corrupted" in error_msg or "database disk image" in error_msg:
                            print(f"⚠️  Database corrupted: {e}")
                            if should_rebuild:
                                print("🔧 Rebuilding corrupted database...")
                                self._corrupted_db_cleanup(persist_dir_path, table_name)
                                self.db = lancedb.connect(str(persist_dir_path))
                                if self.db is None:
                                    raise RuntimeError("Failed to reconnect to LanceDB")
                                existing_tables = self.db.list_tables().tables
                            else:
                                raise RuntimeError(
                                    f"Database is corrupted. Run 'victor index --force' to rebuild.\n"
                                    f"Original error: {e}"
                                ) from e
                        else:
                            raise
                else:
                    print(f"📝 Creating new table: {table_name}")

            except Exception as e:
                error_msg = str(e).lower()
                if "malformed" in error_msg or "corrupted" in error_msg or "database disk image" in error_msg:
                    if should_rebuild:
                        print(f"⚠️  Database corrupted during connection: {e}")
                        print("🔧 Rebuilding corrupted database...")
                        self._corrupted_db_cleanup(persist_dir_path, table_name)
                        self.db = lancedb.connect(str(persist_dir_path))
                        if self.db is None:
                            raise RuntimeError("Failed to reconnect to LanceDB")
                        existing_tables = self.db.list_tables().tables
                    else:
                        raise RuntimeError(
                            f"Database is corrupted. Run 'victor index --force' to rebuild.\n"
                            f"Original error: {e}"
                        ) from e
                else:
                    raise
        else:
            # Fresh database
            self.db = lancedb.connect(str(persist_dir_path))
            if self.db is None:
                raise RuntimeError("Failed to connect to LanceDB")
            existing_tables = self.db.list_tables().tables
            print(f"📝 Creating new table: {table_name}")

        self._initialized = True
        print("✅ LanceDB provider initialized!")

    def _corrupted_db_cleanup(self, persist_dir: Path, table_name: str) -> None:
        """Clean up corrupted database files.

        Args:
            persist_dir: Directory containing the database
            table_name: Name of the corrupted table
        """
        import shutil

        # Close existing connection
        if self.db:
            try:
                del self.db
            except Exception:
                pass
            self.db = None
        self.table = None

        # Remove corrupted database files
        db_pattern = f"{table_name}.*"
        removed_files = []
        for file in persist_dir.glob(db_pattern):
            try:
                if file.is_file():
                    file.unlink()
                    removed_files.append(file.name)
                elif file.is_dir():
                    shutil.rmtree(file)
                    removed_files.append(file.name + "/")
            except Exception as e:
                print(f"⚠️  Could not remove {file}: {e}")

        # Also remove any journal/wal files
        for suffix in ["-journal", "-wal", "-shm"]:
            journal_file = persist_dir / f"{table_name}.lance{suffix}"
            if journal_file.exists():
                try:
                    journal_file.unlink()
                    removed_files.append(journal_file.name)
                except Exception:
                    pass

        if removed_files:
            print(f"🗑️  Removed corrupted files: {', '.join(removed_files)}")
        print("✅ Corrupted database cleaned up, will rebuild on next index")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self._initialized:
            await self.initialize()

        if self.embedding_model is None:
            raise RuntimeError("Embedding model not initialized")

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

        if self.embedding_model is None:
            raise RuntimeError("Embedding model not initialized")

        return await self.embedding_model.embed_batch(texts)

    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a single document.

        Note: Content is NOT stored in LanceDB to avoid duplication.
        Use unified_id to lookup content from graph store.

        Args:
            doc_id: Document identifier (unified ID for graph correlation)
            content: Document content (used for embedding, not stored)
            metadata: Optional metadata
        """
        if not self._initialized:
            await self.initialize()

        # Generate embedding
        embedding = await self.embed_text(content)

        # Prepare document - NO content storage (deduplication)
        # Content can be looked up via unified_id from graph store
        document = {
            "id": doc_id,
            "vector": embedding,
            # "content": content,  # REMOVED - deduplicated, lookup via graph
            **(metadata or {}),
        }

        # Add to table (create if doesn't exist)
        table_name = self.config.extra_config.get("table_name", "embeddings")
        if self.table is None:
            if self.db is None:
                raise RuntimeError("Database not initialized")
            self.table = self.db.create_table(table_name, data=[document])
        else:
            self.table.add([document])

    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index multiple documents in batch.

        Note: Content is NOT stored in LanceDB to avoid duplication.
        Use unified_id to lookup content from graph store.

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

        # Prepare documents for insertion - NO content storage (deduplication)
        lance_docs = []
        for doc, embedding in zip(documents, embeddings, strict=False):
            lance_doc = {
                "id": doc["id"],
                "vector": embedding,
                # "content": doc["content"],  # REMOVED - deduplicated, lookup via graph
                **doc.get("metadata", {}),
            }
            lance_docs.append(lance_doc)

        # Add to table (create if doesn't exist)
        table_name = self.config.extra_config.get("table_name", "embeddings")
        if self.table is None:
            if self.db is None:
                raise RuntimeError("Database not initialized")
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

        Note: Content is not stored in LanceDB. Use the returned id (unified_id)
        to lookup full content from the graph store.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results (content field contains unified_id for lookup)
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

            # unified_id is stored in the "id" field
            unified_id = result.get("id", "")

            search_results.append(
                EmbeddingSearchResult(
                    file_path=result.get("file_path", ""),
                    symbol_name=result.get("symbol_name"),
                    # Content not stored - use unified_id for graph lookup
                    content=unified_id,  # Store unified_id for graph correlation
                    score=score,
                    line_number=result.get("line_number"),
                    metadata={
                        "unified_id": unified_id,  # Explicit correlation key
                        **{k: v for k, v in result.items() if not k.startswith("_")},
                    },
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
        if self.db is None:
            return
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
            "embedding_model": self.config.embedding_model,
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
