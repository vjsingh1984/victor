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

"""Embedding system for Victor codebase intelligence.

This module provides a plugin-based architecture for embeddings with two layers:

1. **Embedding Models** (text -> vector):
   - SentenceTransformerModel: Local, free, CPU/GPU
   - OpenAIEmbeddingModel: Cloud, fast, costs money
   - CohereEmbeddingModel: Cloud, multilingual

2. **Vector Stores** (storage + search):
   - ChromaDB: Development, lightweight
   - ProximaDB: Production, custom-built for scale
   - FAISS: Fast CPU-based search

Mix and match for your needs:
```python
# Development: Local embeddings + ChromaDB
config = EmbeddingConfig(
    vector_store="chromadb",
    embedding_model_type="sentence-transformers",
    embedding_model_name="all-MiniLM-L6-v2"
)

# Production: OpenAI embeddings + ProximaDB
config = EmbeddingConfig(
    vector_store="proximadb",
    embedding_model_type="openai",
    embedding_model_name="text-embedding-3-small",
    embedding_api_key=os.getenv("OPENAI_API_KEY")
)
```
"""

from victor.codebase.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    SearchResult,
)
from victor.codebase.embeddings.models import (
    BaseEmbeddingModel,
    CohereEmbeddingModel,
    EmbeddingModelConfig,
    OpenAIEmbeddingModel,
    SentenceTransformerModel,
    create_embedding_model,
)
from victor.codebase.embeddings.registry import EmbeddingRegistry

__all__ = [
    # Base classes
    "BaseEmbeddingProvider",
    "BaseEmbeddingModel",
    "EmbeddingConfig",
    "EmbeddingModelConfig",
    "SearchResult",
    # Embedding models
    "SentenceTransformerModel",
    "OpenAIEmbeddingModel",
    "CohereEmbeddingModel",
    "create_embedding_model",
    # Registry
    "EmbeddingRegistry",
]
