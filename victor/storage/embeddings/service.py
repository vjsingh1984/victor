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

"""Shared embedding service singleton for Victor.

This module provides a single embedding service instance that:
- Loads the sentence-transformers model once
- Provides embedding generation for all components
- Supports both sync and async embedding generation
- Manages memory efficiently with a single model instance
"""

import asyncio
import hashlib
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Import TRACE level from debug_logger (initializes the level on import)
from victor.agent.debug_logger import TRACE

# Note: Similarity operations use NumPy+BLAS directly (not Rust).
# Benchmarks show NumPy is 10x faster due to BLAS optimization and
# avoiding Python→list→Rust→list→Python FFI overhead.
# See tests/benchmark/test_native_performance.py for details.

# Disable tokenizers parallelism BEFORE importing sentence_transformers
# This prevents "bad value(s) in fds_to_keep" errors in async contexts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# Default embedding model - matches unified_embedding_model in settings.py
# BAAI/bge-small-en-v1.5: 130MB, 384-dim, ~6ms, MTEB 62.2
# - Excellent for code search (trained on code-related tasks)
# - CPU-optimized for consumer-grade hardware
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingService:
    """Singleton embedding service for Victor.

    Provides a shared embedding model instance that can be used by:
    - SemanticToolSelector (tool selection)
    - IntentClassifier (continuation vs completion detection)
    - Codebase search (code semantic search)

    Benefits:
    - Single model instance (saves ~80MB memory per model)
    - Thread-safe embedding generation
    - Lazy loading (model loads on first use)
    - Consistent embeddings across all components

    Usage:
        # Get the singleton instance
        service = EmbeddingService.get_instance()

        # Generate embeddings
        embedding = await service.embed_text("Hello world")
        embeddings = await service.embed_batch(["Hello", "World"])

        # Sync version (for non-async contexts)
        embedding = service.embed_text_sync("Hello world")
    """

    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
    ):
        """Initialize embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Recommended options (all CPU-optimized, 384 dims):
                - "BAAI/bge-small-en-v1.5" (default, 130MB, MTEB 62.2, best for code)
                - "thenlper/gte-small" (67MB, MTEB 61.4, smallest footprint)
                - "all-MiniLM-L12-v2" (120MB, MTEB 59.8, legacy)
                - "all-MiniLM-L6-v2" (80MB, MTEB 58.8, fastest)
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device
        self._model: Any = None  # SentenceTransformer, lazy loaded
        self._model_lock = threading.Lock()
        self._dimension: Optional[int] = None

        # In-memory embedding cache for repeated texts (e.g., tool descriptions)
        # Key: hash of text, Value: embedding vector
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 1000  # Limit memory usage

        # Shutdown flag to prevent new operations after shutdown initiated
        self._shutdown = False

    @classmethod
    def get_instance(
        cls,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
    ) -> "EmbeddingService":
        """Get or create the singleton embedding service instance.

        Args:
            model_name: Model name (only used on first call)
            device: Device (only used on first call)

        Returns:
            The singleton EmbeddingService instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name, device=device)
                    logger.info(f"Created EmbeddingService singleton with model: {model_name}")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._shutdown = True
                cls._instance._model = None
                cls._instance = None
                logger.info("Reset EmbeddingService singleton")

    def shutdown(self) -> None:
        """Signal shutdown to prevent new embedding operations.

        After calling this, embed_text and embed_batch will return
        zero vectors instead of computing new embeddings. This prevents
        embedding operations from running after shutdown is initiated.
        """
        self._shutdown = True
        logger.debug("[EmbeddingService] Shutdown signaled, new operations will be skipped")

    def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded (lazy loading)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        load_start = time.perf_counter()
                        logger.info(
                            f"[EmbeddingService] Loading model: {self.model_name} "
                            f"(device={self.device or 'auto'})"
                        )
                        logger.debug(
                            "[EmbeddingService] Model specs: "
                            "BAAI/bge-small-en-v1.5=130MB/384-dim/MTEB-62.2, "
                            "thenlper/gte-small=67MB/384-dim, "
                            "all-MiniLM-L6-v2=80MB/384-dim/fastest"
                        )

                        self._model = SentenceTransformer(
                            self.model_name,
                            device=self.device,
                        )

                        # Get embedding dimension from model
                        self._dimension = self._model.get_sentence_embedding_dimension()
                        load_time = time.perf_counter() - load_start

                        logger.info(
                            f"[EmbeddingService] Model loaded successfully: "
                            f"model={self.model_name}, "
                            f"dimension={self._dimension}, "
                            f"device={self._model.device}, "
                            f"load_time={load_time:.2f}s"
                        )
                        logger.debug(
                            "[EmbeddingService] Singleton instance ready. "
                            "Memory footprint ~130MB for bge-small-en-v1.5"
                        )
                    except ImportError as e:
                        raise ImportError(
                            "sentence-transformers not installed. "
                            "Install with: pip install sentence-transformers"
                        ) from e

    @property
    def dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            Embedding dimension (e.g., 384 for MiniLM)
        """
        self._ensure_model_loaded()
        return self._dimension or 384

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def embed_text_sync(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a single text (sync version).

        Args:
            text: Text to embed
            use_cache: Whether to use in-memory cache (default: True)

        Returns:
            Embedding vector as numpy array (float32)
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                self._cache_hits += 1
                logger.log(TRACE, f"[EmbeddingService] cache hit: chars={len(text)}")
                return self._embedding_cache[cache_key]
            self._cache_misses += 1

        self._ensure_model_loaded()

        try:
            start_time = time.perf_counter()
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            elapsed = time.perf_counter() - start_time
            chars_per_sec = len(text) / elapsed if elapsed > 0 else 0

            # Log at TRACE level (very verbose)
            logger.log(
                TRACE,
                f"[EmbeddingService] embed_text: "
                f"chars={len(text)}, "
                f"time={elapsed*1000:.2f}ms, "
                f"chars/sec={chars_per_sec:.0f}",
            )

            result = np.asarray(embedding, dtype=np.float32)

            # Cache the result
            if use_cache:
                # Evict oldest entries if cache is full
                if len(self._embedding_cache) >= self._max_cache_size:
                    # Simple eviction: remove first 10% of entries
                    keys_to_remove = list(self._embedding_cache.keys())[
                        : self._max_cache_size // 10
                    ]
                    for key in keys_to_remove:
                        del self._embedding_cache[key]
                self._embedding_cache[cache_key] = result

            return result
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback (better than crashing)
            return np.zeros(self.dimension, dtype=np.float32)

    def _calculate_optimal_batch_size(self, texts: List[str]) -> int:
        """Calculate optimal batch size based on text lengths.

        Uses adaptive batching to balance memory usage and throughput:
        - Short texts (avg < 256 chars): batch_size=64 (more parallelism)
        - Medium texts (256-1024 chars): batch_size=32 (balanced)
        - Long texts (1024-4096 chars): batch_size=16 (memory-safe)
        - Very long texts (> 4096 chars): batch_size=8 (prevent OOM)

        Args:
            texts: List of texts to embed

        Returns:
            Optimal batch size for the given texts
        """
        if not texts:
            return 32

        avg_length = sum(len(t) for t in texts) / len(texts)
        max_length = max(len(t) for t in texts)

        # Consider both average and max length for safety
        effective_length = (avg_length + max_length) / 2

        if effective_length < 256:
            return 64  # Short texts - maximize throughput
        elif effective_length < 1024:
            return 32  # Medium texts - balanced
        elif effective_length < 4096:
            return 16  # Long texts - memory-conscious
        else:
            return 8  # Very long texts - prevent OOM

    def embed_batch_sync(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts (sync version).

        Uses dynamic batch sizing based on text lengths for optimal
        memory usage and throughput.

        Args:
            texts: List of texts to embed

        Returns:
            2D numpy array of embeddings (shape: [len(texts), dimension])
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._ensure_model_loaded()

        try:
            start_time = time.perf_counter()
            total_chars = sum(len(t) for t in texts)

            # Calculate optimal batch size based on text lengths
            batch_size = self._calculate_optimal_batch_size(texts)

            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )

            elapsed = time.perf_counter() - start_time
            embeddings_per_sec = len(texts) / elapsed if elapsed > 0 else 0
            ms_per_embedding = (elapsed * 1000) / len(texts) if texts else 0
            chars_per_sec = total_chars / elapsed if elapsed > 0 else 0

            logger.debug(
                f"[EmbeddingService] embed_batch: "
                f"count={len(texts)}, "
                f"total_chars={total_chars}, "
                f"batch_size={batch_size}, "
                f"time={elapsed*1000:.2f}ms, "
                f"embeddings/sec={embeddings_per_sec:.1f}, "
                f"ms/embedding={ms_per_embedding:.2f}, "
                f"chars/sec={chars_per_sec:.0f}"
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to generate batch embeddings: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics.

        Returns:
            Dictionary with cache stats: hits, misses, size, hit_rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._embedding_cache),
            "hit_rate": f"{hit_rate:.1%}",
            "max_size": self._max_cache_size,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("[EmbeddingService] Cache cleared")

    async def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a single text (async version).

        Args:
            text: Text to embed
            use_cache: Whether to use in-memory cache (default: True)

        Returns:
            Embedding vector as numpy array (float32)
        """
        # Check shutdown flag before starting operation
        if self._shutdown:
            logger.log(
                TRACE, f"[EmbeddingService] Skipping embed_text (shutdown): chars={len(text)}"
            )
            return np.zeros(self.dimension, dtype=np.float32)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.embed_text_sync(text, use_cache=use_cache)
        )

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts (async version).

        Args:
            texts: List of texts to embed

        Returns:
            2D numpy array of embeddings (shape: [len(texts), dimension])
        """
        # Check shutdown flag before starting operation
        if self._shutdown:
            logger.log(
                TRACE, f"[EmbeddingService] Skipping embed_batch (shutdown): count={len(texts)}"
            )
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch_sync, texts)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Uses NumPy which is already SIMD-optimized via BLAS/LAPACK.
        Benchmarks show NumPy is 10x faster than Rust for this operation
        due to FFI overhead from numpy→list→rust→list→numpy conversion.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (0-1 for normalized vectors, -1 to 1 in general)
        """
        # NumPy with BLAS is faster than Rust for vectorized operations
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def cosine_similarity_matrix(
        query: np.ndarray,
        corpus: np.ndarray,
    ) -> np.ndarray:
        """Calculate cosine similarity between query and all corpus vectors.

        Uses NumPy which is already SIMD-optimized via BLAS/LAPACK.
        Benchmarks show NumPy is 10x faster than Rust for batch operations
        due to FFI overhead from numpy→list→rust→list→numpy conversion.

        Args:
            query: Query vector (shape: [dimension])
            corpus: Corpus matrix (shape: [n_items, dimension])

        Returns:
            Similarity scores (shape: [n_items])
        """
        if corpus.size == 0:
            return np.array([])

        # NumPy with BLAS is faster than Rust for vectorized operations
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        corpus_norms = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)

        # Dot product
        similarities = np.dot(corpus_norms, query_norm)
        return np.asarray(similarities)
