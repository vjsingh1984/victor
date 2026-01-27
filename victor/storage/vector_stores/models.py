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

"""Embedding model providers (separate from vector stores).

This module handles GENERATING embeddings (converting text to vectors).
The vector stores (ChromaDB, LanceDB, etc.) handle STORING and SEARCHING.

This separation allows mixing and matching:
- OpenAI embeddings + FAISS storage
- Sentence-transformers + LanceDB storage
- Cohere embeddings + ChromaDB storage
"""

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService

    try:
        from openai import AsyncOpenAI as AsyncOpenAIType  # noqa: F401
    except ImportError:
        AsyncOpenAIType = None

    try:
        import cohere  # type: ignore[import-not-found]
    except ImportError:
        pass

    try:
        import httpx
    except ImportError:
        pass


class EmbeddingModelConfig(BaseModel):
    """Configuration for embedding model."""

    model_config = ConfigDict(protected_namespaces=())

    embedding_type: str = Field(
        default="sentence-transformers",
        description="Embedding provider type (sentence-transformers, openai, cohere, ollama, etc.)",
        validation_alias=AliasChoices("embedding_type", "model_type"),
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Specific model name",
        validation_alias=AliasChoices("embedding_model", "model_name"),
    )
    dimension: int = Field(
        default=384, description="Embedding dimension (auto-detected if possible)"
    )
    api_key: Optional[str] = Field(default=None, description="API key for cloud providers")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")

    @property
    def model_type(self) -> str:
        """Backwards-compatible alias for embedding_type."""
        return self.embedding_type

    @model_type.setter
    def model_type(self, value: str) -> None:
        self.embedding_type = value

    @property
    def model_name(self) -> str:
        """Backwards-compatible alias for embedding_model."""
        return self.embedding_model

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.embedding_model = value


class BaseEmbeddingModel(ABC):
    """Abstract base for embedding models.

    Handles converting text -> vectors.
    Does NOT handle storage/search (that's the vector store's job).
    """

    def __init__(self, config: EmbeddingModelConfig):
        """Initialize embedding model.

        Args:
            config: Model configuration
        """
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model (load weights, connect to API, etc.)."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch optimized).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


class SentenceTransformerModel(BaseEmbeddingModel):
    """Sentence-transformers embedding model (local, CPU/GPU).

    Uses the shared EmbeddingService singleton for memory efficiency.
    This allows sharing the model with IntentClassifier and SemanticToolSelector.

    Pros:
    - Free
    - Runs locally
    - No API limits
    - Many pre-trained models available
    - Shares model with other components (saves ~80MB per model)

    Cons:
    - Slower than cloud APIs (unless you have GPU)
    - Requires downloading models
    - CPU inference can be slow for large batches

    Good for: Development, privacy-sensitive data, offline use
    """

    def __init__(self, config: EmbeddingModelConfig):
        """Initialize sentence-transformers model."""
        super().__init__(config)
        self._embedding_service: Optional["EmbeddingService"] = None

    async def initialize(self) -> None:
        """Initialize using shared EmbeddingService singleton."""
        if self._initialized:
            return

        # Import here to avoid circular imports
        from victor.storage.embeddings.service import EmbeddingService

        print(f"🤖 Loading sentence-transformer model: {self.config.embedding_model}")

        # Use shared EmbeddingService singleton for memory efficiency
        # This shares the model with IntentClassifier and SemanticToolSelector
        self._embedding_service = EmbeddingService.get_instance(
            model_name=self.config.embedding_model
        )

        # Ensure model is loaded (lazy loading)
        self._embedding_service._ensure_model_loaded()

        self._initialized = True
        print(f"✅ Model loaded (shared via EmbeddingService)! Dimension: {self.get_dimension()}")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        if not self._initialized:
            await self.initialize()

        # Use async method from EmbeddingService
        if self._embedding_service is None:
            raise RuntimeError("EmbeddingService not initialized")
        embedding = await self._embedding_service.embed_text(text)
        return list(embedding.tolist())

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch optimized)."""
        if not self._initialized:
            await self.initialize()

        # Use async batch method from EmbeddingService
        if self._embedding_service is None:
            raise RuntimeError("EmbeddingService not initialized")
        embeddings = await self._embedding_service.embed_batch(texts)
        return [list(emb.tolist()) for emb in embeddings]

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedding_service is not None:
            return self._embedding_service.dimension
        return self.config.dimension

    async def close(self) -> None:
        """Clean up resources.

        Note: We don't reset EmbeddingService singleton here as it may be
        shared by other components (IntentClassifier, SemanticToolSelector).
        """
        self._embedding_service = None
        self._initialized = False


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embedding model (cloud API).

    Pros:
    - Very fast
    - High quality embeddings
    - Scalable
    - No local resources needed

    Cons:
    - Costs money ($0.0001 per 1K tokens for text-embedding-3-small)
    - Requires internet
    - Rate limits
    - Data sent to OpenAI

    Good for: Production, when cost is acceptable
    """

    def __init__(self, config: EmbeddingModelConfig):
        """Initialize OpenAI embedding model."""
        super().__init__(config)
        self.client: Optional["AsyncOpenAIType"] = None

    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        if self._initialized:
            return

        try:
            from openai import AsyncOpenAI as AsyncOpenAIImpl
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        if not self.config.api_key:
            raise ValueError("OpenAI API key required")

        self.client = AsyncOpenAIImpl(api_key=self.config.api_key)
        self._initialized = True

        print(f"✅ OpenAI embedding model initialized: {self.config.embedding_model}")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        if not self._initialized:
            await self.initialize()

        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        response = await self.client.embeddings.create(
            model=self.config.embedding_model, input=text
        )
        embedding = response.data[0].embedding
        # Type assertion for OpenAI's dynamic response
        assert isinstance(embedding, list)
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self._initialized:
            await self.initialize()

        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")
        # OpenAI API handles batching internally (up to 2048 texts per request)
        response = await self.client.embeddings.create(
            model=self.config.embedding_model, input=texts
        )
        embeddings = [item.embedding for item in response.data]
        # Type assertion for OpenAI's dynamic response
        for emb in embeddings:
            assert isinstance(emb, list)
        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension based on model."""
        # OpenAI model dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.config.embedding_model, self.config.dimension)

    async def close(self) -> None:
        """Clean up resources."""
        self.client = None
        self._initialized = False


class CohereEmbeddingModel(BaseEmbeddingModel):
    """Cohere embedding model (cloud API).

    Pros:
    - Fast and high quality
    - Good multilingual support
    - Competitive pricing

    Cons:
    - Costs money
    - Requires internet
    - Data sent to Cohere

    Good for: Production, multilingual use cases
    """

    def __init__(self, config: EmbeddingModelConfig):
        """Initialize Cohere embedding model."""
        super().__init__(config)
        self.client: Optional["cohere.AsyncClient"] = None

    async def initialize(self) -> None:
        """Initialize Cohere client."""
        if self._initialized:
            return

        try:
            import cohere
        except ImportError:
            raise ImportError("cohere not installed. Install with: pip install cohere")

        if not self.config.api_key:
            raise ValueError("Cohere API key required")

        self.client = cohere.AsyncClient(api_key=self.config.api_key)
        self._initialized = True

        print(f"✅ Cohere embedding model initialized: {self.config.embedding_model}")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding using Cohere API."""
        if not self._initialized:
            await self.initialize()

        if self.client is None:
            raise RuntimeError("Cohere client not initialized")
        response = await self.client.embed(texts=[text], model=self.config.embedding_model)
        embedding = response.embeddings[0]
        # Type assertion for Cohere's dynamic response
        assert isinstance(embedding, list)
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self._initialized:
            await self.initialize()

        if self.client is None:
            raise RuntimeError("Cohere client not initialized")
        response = await self.client.embed(texts=texts, model=self.config.embedding_model)
        embeddings = response.embeddings
        # Type assertion for Cohere's dynamic response
        for emb in embeddings:
            assert isinstance(emb, list)
        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        # Cohere model dimensions
        dimensions = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
        }
        return dimensions.get(self.config.embedding_model, self.config.dimension)

    async def close(self) -> None:
        """Clean up resources."""
        self.client = None
        self._initialized = False


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """Ollama embedding model (local, high-performance).

    Pros:
    - Free and runs locally (no API costs)
    - High accuracy models (Qwen3-Embedding:8b is #1 on MTEB multilingual)
    - No API limits or rate limiting
    - Large context windows (40K tokens for Qwen3)
    - Full data privacy (embeddings stay local)
    - No internet required after model download

    Cons:
    - Requires Ollama to be running locally
    - Larger disk space for bigger models (4.7GB for Qwen3:8b)
    - CPU/GPU dependent performance

    Good for: Production deployments, accuracy-driven use cases, privacy,
              offline work, cost-sensitive applications

    Supported Models:
    - qwen3-embedding:8b (RECOMMENDED) - 70.58 MTEB, 40K context, 4096-dim
    - qwen3-embedding:4b - Fast variant, 32K context, 2560-dim (user-definable 32-2560)
    - gte-Qwen2-7B-instruct - Instruction-tuned, large model, 3584-dim
    - bge-m3 - Multi-functional retrieval, 8K context, 1024-dim
    - mxbai-embed-large - SOTA for Bert-large size, 512 context, 1024-dim
    - nomic-embed-text - Beats OpenAI ada-002, 2K context, 768-dim
    """

    def __init__(self, config: EmbeddingModelConfig):
        """Initialize Ollama embedding model."""
        super().__init__(config)
        self.base_url = (
            config.api_key or "http://localhost:11434"
        )  # Reuse api_key field for base_url
        self.client: Optional["httpx.AsyncClient"] = None

    async def initialize(self) -> None:
        """Initialize Ollama client and verify model availability."""
        if self._initialized:
            return

        try:
            import httpx
        except ImportError:
            raise ImportError("httpx not installed. Install with: pip install httpx")

        # Create async HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=120.0  # Longer timeout for large models
        )

        print(f"🤖 Initializing Ollama embedding model: {self.config.embedding_model}")
        print(f"🔗 Ollama server: {self.base_url}")

        # Verify model is available by testing with a small prompt
        try:
            response = await self.client.post(
                "/api/embeddings", json={"model": self.config.embedding_model, "prompt": "test"}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise RuntimeError(
                    f"❌ Ollama model '{self.config.embedding_model}' not found.\n"
                    f"   Pull it with: ollama pull {self.config.embedding_model}\n"
                    f"   Available models: ollama list"
                )
            else:
                raise RuntimeError(f"❌ Ollama API error: {e}")
        except httpx.ConnectError:
            raise RuntimeError(
                f"❌ Cannot connect to Ollama at {self.base_url}\n"
                f"   Make sure Ollama is running: ollama serve\n"
                f"   Or check if it's running on a different port"
            )

        self._initialized = True
        print(f"✅ Ollama embedding model ready: {self.config.embedding_model}")
        print(f"📊 Embedding dimension: {self.get_dimension()}")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text using Ollama.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not self._initialized:
            await self.initialize()

        if self.client is None:
            raise RuntimeError("Ollama client not initialized")
        try:
            response = await self.client.post(
                "/api/embeddings", json={"model": self.config.embedding_model, "prompt": text}
            )
            response.raise_for_status()
            result = response.json()
            embedding = result.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError(f"Expected list embedding, got {type(embedding)}")
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (optimized with concurrent requests).

        Ollama doesn't have a native batch API, so we use asyncio.gather
        to send concurrent requests for better performance.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()

        # Use asyncio.gather for concurrent requests
        # This is much faster than sequential processing
        tasks = [self.embed_text(text) for text in texts]
        embeddings = await asyncio.gather(*tasks)
        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension based on model.

        Returns:
            Embedding dimension for the model
        """
        # Ollama model dimensions (based on MTEB research and official docs)
        dimensions = {
            # Qwen3 family (RECOMMENDED for production - highest dimensional)
            "qwen3-embedding:8b": 4096,
            "qwen3-embedding:4b": 2560,  # User-definable 32-2560
            "qwen3-embedding:0.6b": 4096,
            # High-dimensional instruction-tuned models
            "gte-Qwen2-7B-instruct": 3584,
            "gte-qwen2-7b-instruct": 3584,
            # Other high-quality models
            "snowflake-arctic-embed2": 1024,
            "bge-m3": 1024,
            "mxbai-embed-large": 1024,
            "nomic-embed-text": 768,
            "nomic-embed-text:v1.5": 768,
            # Smaller models
            "all-minilm": 384,
            "all-minilm:l6-v2": 384,
        }

        # Check for exact match first
        if self.config.embedding_model in dimensions:
            return dimensions[self.config.embedding_model]

        # Check for partial match (e.g., "qwen3-embedding" matches "qwen3-embedding:8b")
        for model_key, dim in dimensions.items():
            if self.config.embedding_model in model_key or model_key in self.config.embedding_model:
                return dim

        # Fall back to config dimension
        return self.config.dimension

    async def close(self) -> None:
        """Clean up HTTP client resources."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._initialized = False


# Model Registry
_embedding_models = {
    "sentence-transformers": SentenceTransformerModel,
    "openai": OpenAIEmbeddingModel,
    "cohere": CohereEmbeddingModel,
    "ollama": OllamaEmbeddingModel,
}


def create_embedding_model(config: EmbeddingModelConfig) -> BaseEmbeddingModel:
    """Factory function to create embedding model.

    Args:
        config: Model configuration

    Returns:
        Embedding model instance

    Raises:
        ValueError: If model type not recognized
    """
    model_class = _embedding_models.get(config.embedding_type)
    if not model_class:
        available = ", ".join(_embedding_models.keys())
        raise ValueError(
            f"Unknown embedding model type: {config.embedding_type}. " f"Available: {available}"
        )

    return model_class(config)  # type: ignore[abstract]
