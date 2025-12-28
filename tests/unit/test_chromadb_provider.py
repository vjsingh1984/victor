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

"""Unit tests for ChromaDB embedding provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.vector_stores.base import EmbeddingConfig, SearchResult
from victor.vector_stores.chromadb_provider import ChromaDBProvider


@pytest.fixture
def chroma_config():
    """Create ChromaDB config for testing."""
    return EmbeddingConfig(
        vector_store="chromadb",
        persist_directory="/tmp/test_chromadb",
        distance_metric="cosine",
        embedding_model_type="ollama",
        embedding_model_name="qwen3-embedding:8b",
        embedding_api_key="http://localhost:11434",
        extra_config={
            "collection_name": "test_collection",
            "dimension": 4096,
            "batch_size": 8,
        },
    )


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client and collection."""
    # Check if chromadb is available first
    import importlib.util

    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb not installed")

    # Patch chromadb.Client directly instead of through module
    with patch("chromadb.Client") as mock_client_class:
        with patch("victor.codebase.embeddings.chromadb_provider.CHROMADB_AVAILABLE", True):
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.name = "test_collection"
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client
            yield mock_client_class, mock_client, mock_collection


class TestChromaDBProvider:
    """Tests for ChromaDBProvider."""

    def test_initialization(self, chroma_config):
        """Test provider initialization."""
        with patch("victor.codebase.embeddings.chromadb_provider.CHROMADB_AVAILABLE", True):
            provider = ChromaDBProvider(chroma_config)

            assert provider.config == chroma_config
            assert provider.client is None
            assert provider.collection is None
            assert provider.embedding_model is None
            assert not provider._initialized

    def test_initialization_chromadb_not_available(self, chroma_config):
        """Test initialization when ChromaDB not available."""
        with patch("victor.codebase.embeddings.chromadb_provider.CHROMADB_AVAILABLE", False):
            with pytest.raises(ImportError, match="ChromaDB not available"):
                ChromaDBProvider(chroma_config)

    @pytest.mark.asyncio
    async def test_initialize_persistent(self, chroma_config, mock_chromadb):
        """Test initialization with persistent storage."""
        mock_chroma, mock_client, mock_collection = mock_chromadb

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            assert provider._initialized
            assert provider.client == mock_client
            assert provider.collection == mock_collection
            assert provider.embedding_model == mock_embedding_model
            mock_embedding_model.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_in_memory(self, mock_chromadb):
        """Test initialization with in-memory storage."""
        config = EmbeddingConfig(
            vector_store="chromadb",
            persist_directory=None,  # In-memory
            embedding_model_type="ollama",
            embedding_model_name="qwen3-embedding:8b",
            embedding_api_key="http://localhost:11434",
            extra_config={"dimension": 4096},
        )

        mock_chroma, mock_client, mock_collection = mock_chromadb

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(config)
            await provider.initialize()

            assert provider._initialized

    @pytest.mark.asyncio
    async def test_embed_text(self, chroma_config, mock_chromadb):
        """Test single text embedding."""
        mock_chroma, mock_client, mock_collection = mock_chromadb

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_embedding_model.embed_text.return_value = [0.1] * 4096
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            result = await provider.embed_text("test text")

            assert len(result) == 4096
            mock_embedding_model.embed_text.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_embed_batch(self, chroma_config, mock_chromadb):
        """Test batch embedding."""
        mock_chroma, mock_client, mock_collection = mock_chromadb

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_embedding_model.embed_batch.return_value = [[0.1] * 4096, [0.2] * 4096]
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            texts = ["text1", "text2"]
            results = await provider.embed_batch(texts)

            assert len(results) == 2
            assert all(len(emb) == 4096 for emb in results)
            mock_embedding_model.embed_batch.assert_called_once_with(texts)

    @pytest.mark.asyncio
    async def test_index_document(self, chroma_config, mock_chromadb):
        """Test indexing single document."""
        mock_chroma, mock_client, mock_collection = mock_chromadb

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_embedding_model.embed_text.return_value = [0.1] * 4096
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            await provider.index_document(
                doc_id="test_id", content="test content", metadata={"file": "test.py"}
            )

            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args[1]
            assert call_args["ids"] == ["test_id"]
            assert call_args["documents"] == ["test content"]
            assert len(call_args["embeddings"]) == 1
            assert len(call_args["embeddings"][0]) == 4096

    @pytest.mark.asyncio
    async def test_index_documents(self, chroma_config, mock_chromadb):
        """Test batch indexing documents."""
        mock_chroma, mock_client, mock_collection = mock_chromadb

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_embedding_model.embed_batch.return_value = [[0.1] * 4096, [0.2] * 4096]
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            documents = [
                {"id": "doc1", "content": "content1", "metadata": {"file": "test1.py"}},
                {"id": "doc2", "content": "content2", "metadata": {"file": "test2.py"}},
            ]

            await provider.index_documents(documents)

            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args[1]
            assert len(call_args["ids"]) == 2
            assert len(call_args["documents"]) == 2
            assert len(call_args["embeddings"]) == 2

    @pytest.mark.asyncio
    async def test_search_similar(self, chroma_config, mock_chromadb):
        """Test semantic similarity search."""
        mock_chroma, mock_client, mock_collection = mock_chromadb

        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["content1", "content2"]],
            "metadatas": [[{"file_path": "test1.py"}, {"file_path": "test2.py"}]],
            "distances": [[0.1, 0.2]],
        }

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_embedding_model.embed_text.return_value = [0.1] * 4096
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            results = await provider.search_similar("query text", limit=2)

            assert len(results) == 2
            assert isinstance(results[0], SearchResult)
            assert results[0].file_path == "test1.py"
            assert results[0].content == "content1"
            assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_delete_document(self, chroma_config, mock_chromadb):
        """Test deleting document."""
        mock_chroma, mock_client, mock_collection = mock_chromadb

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            await provider.delete_document("doc_id")

            mock_collection.delete.assert_called_once_with(ids=["doc_id"])

    @pytest.mark.asyncio
    async def test_clear_index(self, chroma_config, mock_chromadb):
        """Test clearing entire index."""
        mock_chroma, mock_client, mock_collection = mock_chromadb

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            await provider.clear_index()

            # Verify collection was deleted and recreated
            mock_client.delete_collection.assert_called_once_with(name="test_collection")
            mock_client.create_collection.assert_called()

    @pytest.mark.asyncio
    async def test_get_stats(self, chroma_config, mock_chromadb):
        """Test getting index statistics."""
        mock_chroma, mock_client, mock_collection = mock_chromadb
        mock_collection.count.return_value = 42

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            # get_dimension is synchronous, not async
            mock_embedding_model.get_dimension = MagicMock(return_value=4096)
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            stats = await provider.get_stats()

            assert stats["provider"] == "chromadb"
            assert stats["total_documents"] == 42
            assert stats["embedding_model_type"] == "ollama"
            assert stats["embedding_model_name"] == "qwen3-embedding:8b"
            assert stats["dimension"] == 4096
            assert stats["distance_metric"] == "cosine"

    @pytest.mark.asyncio
    async def test_close(self, chroma_config, mock_chromadb):
        """Test cleanup."""
        mock_chroma, mock_client, mock_collection = mock_chromadb

        with patch(
            "victor.codebase.embeddings.chromadb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension.return_value = 4096
            mock_embedding_model.close = AsyncMock()
            mock_create.return_value = mock_embedding_model

            provider = ChromaDBProvider(chroma_config)
            await provider.initialize()

            await provider.close()

            mock_embedding_model.close.assert_called_once()
            assert not provider._initialized
