# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Generic vector storage module for Victor framework.

This module provides a plugin-based architecture for vector storage with
multiple backend and embedding model options:

**Vector Store Backends:**
- **LanceDBProvider**: Production, fast, disk-based (default)
- **ChromaDBProvider**: Development, lightweight
- **ProximaDBProvider**: Production, fast vector + graph (recommended)

**Embedding Models:**
- **SentenceTransformerModel**: Local, free, CPU/GPU
- **OllamaEmbeddingModel**: Local via Ollama (Qwen3, etc.)
- **OpenAIEmbeddingModel**: Cloud, fast, costs money
- **CohereEmbeddingModel**: Cloud, multilingual

Example:
    ```python
    from victor.vector_stores import EmbeddingConfig, EmbeddingRegistry

    # Production: Local embeddings + LanceDB
    config = EmbeddingConfig(
        vector_store="lancedb",
        embedding_model_type="sentence-transformers",
        embedding_model_name="all-MiniLM-L12-v2",
        persist_directory="~/.victor/embeddings"
    )

    # Get provider
    provider = EmbeddingRegistry.get_provider(config)
    await provider.initialize()

    # Index and search
    await provider.index_documents(documents)
    results = await provider.search("query", k=10)
    ```

This is a core framework module available to all verticals, not just coding.
"""

from victor.vector_stores.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    SearchResult,
)
from victor.vector_stores.models import (
    BaseEmbeddingModel,
    EmbeddingModelConfig,
    SentenceTransformerModel,
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    CohereEmbeddingModel,
    create_embedding_model,
)
from victor.vector_stores.registry import EmbeddingRegistry

__all__ = [
    # Base classes
    "BaseEmbeddingProvider",
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "EmbeddingModelConfig",
    "SearchResult",
    # Embedding models
    "SentenceTransformerModel",
    "OllamaEmbeddingModel",
    "OpenAIEmbeddingModel",
    "CohereEmbeddingModel",
    "create_embedding_model",
    # Registry
    "EmbeddingRegistry",
]
