"""Base embedding provider interface.

This module separates concerns:
1. **Embedding Model**: Generates vectors from text (sentence-transformers, OpenAI, etc.)
2. **Vector Store**: Stores and searches vectors (ChromaDB, ProximaDB, FAISS, etc.)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding system.

    This configures BOTH the embedding model (text -> vector) and vector store (storage/search).
    """

    # Vector Store Configuration
    vector_store: str = Field(
        default="chromadb",
        description="Vector store provider (chromadb, proximadb, faiss, etc.)"
    )
    persist_directory: Optional[str] = Field(
        default=None, description="Directory to persist vector store"
    )
    distance_metric: str = Field(
        default="cosine", description="Distance metric (cosine, euclidean, dot)"
    )

    # Embedding Model Configuration
    embedding_model_type: str = Field(
        default="sentence-transformers",
        description="Embedding model type (sentence-transformers, openai, cohere)"
    )
    embedding_model_name: str = Field(
        default="all-mpnet-base-v2",
        description="Specific embedding model name (all-mpnet-base-v2 = best quality local model)"
    )
    embedding_api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud embedding providers"
    )

    # Provider-specific configuration
    extra_config: Dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific configuration"
    )


class SearchResult(BaseModel):
    """Result from semantic search."""

    file_path: str = Field(description="Path to the file")
    symbol_name: Optional[str] = Field(default=None, description="Symbol name if applicable")
    content: str = Field(description="Content that matched")
    score: float = Field(description="Relevance score (0-1, higher is better)")
    line_number: Optional[int] = Field(default=None, description="Line number in file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseEmbeddingProvider(ABC):
    """Abstract base class for vector store providers.

    This handles STORAGE and SEARCH of embeddings.
    Embedding generation is delegated to a pluggable BaseEmbeddingModel.

    Separation of concerns:
    - Embedding Model: text -> vector (sentence-transformers, OpenAI, etc.)
    - Vector Store: vector storage/search (ChromaDB, ProximaDB, FAISS, etc.)

    This allows mixing and matching:
    - OpenAI embeddings + FAISS storage
    - Sentence-transformers + ProximaDB storage
    - Cohere embeddings + ChromaDB storage
    """

    def __init__(self, config: EmbeddingConfig, embedding_model: Optional["BaseEmbeddingModel"] = None):
        """Initialize provider with configuration.

        Args:
            config: Embedding configuration
            embedding_model: Optional pre-configured embedding model.
                           If None, will be created from config.
        """
        self.config = config
        self.embedding_model: Optional["BaseEmbeddingModel"] = embedding_model
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider.

        This should:
        - Load embedding models
        - Connect to databases
        - Set up any required resources

        Should be idempotent (safe to call multiple times).
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (list of floats)
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (optimized batch operation).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def index_document(
        self, doc_id: str, content: str, metadata: Dict[str, Any]
    ) -> None:
        """Index a single document.

        Args:
            doc_id: Unique identifier for the document
            content: Content to index
            metadata: Additional metadata (file_path, symbol_name, etc.)
        """
        pass

    @abstractmethod
    async def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Batch index multiple documents (optimized).

        Args:
            documents: List of documents, each with:
                - id: Unique identifier
                - content: Content to index
                - metadata: Additional metadata
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for documents similar to query.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_metadata: Optional metadata filters (e.g., {"file_path": "src/"})

        Returns:
            List of search results, sorted by relevance (highest first)
        """
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Delete a document from the index.

        Args:
            doc_id: Document identifier to delete
        """
        pass

    @abstractmethod
    async def clear_index(self) -> None:
        """Clear the entire index (remove all documents)."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index.

        Returns:
            Dictionary with stats like:
            - total_documents: Number of indexed documents
            - total_embeddings: Number of embedding vectors
            - index_size_mb: Size of index in MB
            - model_name: Name of embedding model
        """
        pass

    async def close(self) -> None:
        """Clean up resources (close connections, etc.).

        Subclasses should override if cleanup is needed.
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(provider={self.config.provider}, model={self.config.model})"
