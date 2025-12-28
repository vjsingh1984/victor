# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Embedding system for codebase intelligence.

.. deprecated:: 0.3.0
    The generic vector storage has moved to ``victor.vector_stores``.
    Please update your imports. This shim will be removed in version 0.5.0.

For new code, prefer importing directly from victor.vector_stores:
    from victor.vector_stores import EmbeddingConfig, EmbeddingRegistry

For code-specific AST-aware chunking:
    from victor_coding.codebase.embeddings.chunker import ASTAwareChunker
"""

import warnings

warnings.warn(
    "Importing from 'victor.codebase.embeddings' is deprecated. "
    "Please use 'victor.vector_stores' for vector storage or "
    "'victor_coding.codebase.embeddings' for code-specific features. "
    "This compatibility shim will be removed in version 0.5.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from victor.vector_stores (the canonical location for generic parts)
from victor.vector_stores import (
    BaseEmbeddingProvider,
    BaseEmbeddingModel,
    EmbeddingConfig,
    EmbeddingModelConfig,
    SearchResult,
    SentenceTransformerModel,
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    CohereEmbeddingModel,
    create_embedding_model,
    EmbeddingRegistry,
)

__all__ = [
    "BaseEmbeddingProvider",
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "EmbeddingModelConfig",
    "SearchResult",
    "SentenceTransformerModel",
    "OllamaEmbeddingModel",
    "OpenAIEmbeddingModel",
    "CohereEmbeddingModel",
    "create_embedding_model",
    "EmbeddingRegistry",
]
