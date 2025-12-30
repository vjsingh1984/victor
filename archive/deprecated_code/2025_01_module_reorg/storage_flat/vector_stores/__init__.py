# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Generic vector storage module for Victor framework.

This module has moved to victor.storage.vector_stores.
Import from victor.storage.vector_stores instead for new code.

This module provides backward-compatible re-exports.
"""

# Re-export from new location for backward compatibility
from victor.storage.vector_stores.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    SearchResult,
)
from victor.storage.vector_stores.models import (
    BaseEmbeddingModel,
    EmbeddingModelConfig,
    SentenceTransformerModel,
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    CohereEmbeddingModel,
    create_embedding_model,
)
from victor.storage.vector_stores.registry import EmbeddingRegistry

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
