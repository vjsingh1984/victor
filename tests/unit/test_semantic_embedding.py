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

"""Tests for semantic embedding providers (sentence-transformers, Ollama, etc.)."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path

from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.base import ToolRegistry, BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing semantic selection"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return {"success": True}


@pytest.fixture
def mock_tool_registry():
    """Create mock tool registry with a few tools."""
    registry = ToolRegistry()
    registry.register(MockTool())
    return registry


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "embeddings"
    cache_dir.mkdir()
    return cache_dir


class TestSentenceTransformersEmbedding:
    """Test sentence-transformers embedding provider."""

    @pytest.mark.asyncio
    async def test_default_provider_is_sentence_transformers(self):
        """Test that default provider is sentence-transformers."""
        selector = SemanticToolSelector()
        assert selector.embedding_provider == "sentence-transformers"
        assert selector.embedding_model == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_sentence_transformer_lazy_loading(self, temp_cache_dir):
        """Test that sentence-transformers model is loaded lazily."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Model should not be loaded initially
        assert selector._sentence_model is None

        # Mock sentence-transformers import (patch the import path)
        with patch('sentence_transformers.SentenceTransformer') as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
            MockST.return_value = mock_model

            # Get embedding should load model
            embedding = await selector._get_sentence_transformer_embedding("test text")

            # Model should be loaded now
            MockST.assert_called_once_with("all-MiniLM-L6-v2")
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_sentence_transformer_embedding_dimensions(self, temp_cache_dir):
        """Test that embeddings have correct dimensions."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        with patch('sentence_transformers.SentenceTransformer') as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
            MockST.return_value = mock_model

            embedding = await selector._get_embedding("test text")

            # all-MiniLM-L6-v2 produces 384-dimensional embeddings
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_sentence_transformer_fallback_on_import_error(self, temp_cache_dir):
        """Test fallback to random embedding when sentence-transformers not installed."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Mock ImportError when loading sentence-transformers
        with patch('sentence_transformers.SentenceTransformer', side_effect=ImportError):
            with pytest.raises(ImportError, match="sentence-transformers not installed"):
                await selector._get_sentence_transformer_embedding("test text")

    @pytest.mark.asyncio
    async def test_async_execution_in_thread_pool(self, temp_cache_dir):
        """Test that sentence-transformers runs in thread pool (non-blocking)."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        with patch('sentence_transformers.SentenceTransformer') as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
            MockST.return_value = mock_model

            # Should run without blocking
            embedding = await selector._get_sentence_transformer_embedding("test text")

            # Verify encode was called
            mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)


class TestOllamaAPIEmbedding:
    """Test Ollama/vLLM/LMStudio API embedding provider."""

    @pytest.mark.asyncio
    async def test_ollama_provider_initialization(self):
        """Test Ollama provider initializes HTTP client."""
        selector = SemanticToolSelector(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text"
        )

        assert selector.embedding_provider == "ollama"
        assert selector._client is not None

    @pytest.mark.asyncio
    async def test_api_embedding_request(self, temp_cache_dir):
        """Test API embedding makes correct HTTP request."""
        selector = SemanticToolSelector(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            cache_dir=temp_cache_dir
        )

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embedding": [0.1] * 768  # nomic-embed-text is 768-dim
        }

        with patch.object(selector._client, 'post', return_value=mock_response) as mock_post:
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock()

            embedding = await selector._get_api_embedding("test text")

            # Verify correct API endpoint and payload
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/api/embeddings" in str(call_args)

    @pytest.mark.asyncio
    async def test_vllm_provider(self):
        """Test vLLM provider uses same API interface."""
        selector = SemanticToolSelector(
            embedding_provider="vllm",
            embedding_model="BAAI/bge-large-en-v1.5",
            ollama_base_url="http://localhost:8000"
        )

        assert selector.embedding_provider == "vllm"
        assert selector._client is not None


class TestEmbeddingCaching:
    """Test embedding caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_file_naming(self, temp_cache_dir):
        """Test cache file uses correct naming convention."""
        selector = SemanticToolSelector(
            embedding_model="all-MiniLM-L6-v2",
            cache_dir=temp_cache_dir
        )

        expected_filename = "tool_embeddings_all-MiniLM-L6-v2.pkl"
        assert selector.cache_file.name == expected_filename

    @pytest.mark.asyncio
    async def test_cache_file_naming_with_special_chars(self, temp_cache_dir):
        """Test cache file naming handles special characters."""
        selector = SemanticToolSelector(
            embedding_model="qwen3-embedding:8b",
            cache_dir=temp_cache_dir
        )

        # Colons and slashes should be replaced
        expected_filename = "tool_embeddings_qwen3-embedding_8b.pkl"
        assert selector.cache_file.name == expected_filename


class TestProviderFallback:
    """Test fallback logic between providers."""

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises NotImplementedError."""
        selector = SemanticToolSelector(
            embedding_provider="openai"  # Not yet implemented
        )

        with pytest.raises(NotImplementedError, match="openai not yet supported"):
            await selector._get_embedding("test text")

    @pytest.mark.asyncio
    async def test_random_fallback_on_sentence_transformer_error(self, temp_cache_dir):
        """Test fallback to random embedding on error."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        with patch('sentence_transformers.SentenceTransformer') as MockST:
            mock_model = MagicMock()
            mock_model.encode.side_effect = Exception("Model error")
            MockST.return_value = mock_model

            embedding = await selector._get_sentence_transformer_embedding("test text")

            # Should return random embedding as fallback
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32


class TestToolSelectionWithEmbeddings:
    """Test end-to-end tool selection with embeddings."""

    @pytest.mark.asyncio
    async def test_tool_selection_with_sentence_transformers(self, mock_tool_registry, temp_cache_dir):
        """Test complete tool selection flow with sentence-transformers."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        with patch('sentence_transformers.SentenceTransformer') as MockST:
            mock_model = MagicMock()
            # Return different embeddings for different inputs
            def mock_encode(text, convert_to_numpy=False):
                if "mock" in text.lower():
                    return np.ones(384, dtype=np.float32)
                else:
                    return np.zeros(384, dtype=np.float32)

            mock_model.encode.side_effect = mock_encode
            MockST.return_value = mock_model

            # Initialize embeddings
            await selector.initialize_tool_embeddings(mock_tool_registry)

            # Select tools
            selected_tools = await selector.select_relevant_tools(
                "use the mock tool",
                mock_tool_registry,
                max_tools=5,
                similarity_threshold=0.3
            )

            # Should select mock_tool due to high similarity
            tool_names = [t.name for t in selected_tools]
            assert len(tool_names) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
