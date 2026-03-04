# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Embedding system for Victor codebase intelligence.

Re-exports generic vector storage infrastructure from victor-core
(victor.storage.vector_stores) for backward compatibility.

For new code, prefer importing directly from victor.storage.vector_stores:
    from victor.storage.vector_stores import EmbeddingConfig, EmbeddingRegistry
"""

# Re-export from victor-core for backward compatibility
from victor.storage.vector_stores import (
    # Base classes
    BaseEmbeddingProvider,
    BaseEmbeddingModel,
    EmbeddingConfig,
    EmbeddingModelConfig,
    EmbeddingSearchResult,
    # Embedding models
    SentenceTransformerModel,
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    CohereEmbeddingModel,
    create_embedding_model,
    # Registry
    EmbeddingRegistry,
)

__all__ = [
    # Base classes (from victor-core)
    "BaseEmbeddingProvider",
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "EmbeddingModelConfig",
    "EmbeddingSearchResult",
    # Embedding models (from victor-core)
    "SentenceTransformerModel",
    "OllamaEmbeddingModel",
    "OpenAIEmbeddingModel",
    "CohereEmbeddingModel",
    "create_embedding_model",
    # Registry (from victor-core)
    "EmbeddingRegistry",
]
