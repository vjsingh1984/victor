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
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import TRACE level from debug_logger (initializes the level on import)
from victor.agent.debug_logger import TRACE

# Import Rust-accelerated embedding operations (optional, with NumPy fallback)
from victor.native.accelerators.embedding_ops import get_embedding_accelerator

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
        use_rust_embeddings: Optional[bool] = None,
        rust_embedding_batch_threshold: Optional[int] = None,
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
            use_rust_embeddings: Enable Rust-accelerated similarity operations (default: from settings)
            rust_embedding_batch_threshold: Minimum batch size for Rust (default: from settings)
        """
        self.model_name = model_name
        self.device = device
        self._model: Any = None  # SentenceTransformer, lazy loaded
        self._model_lock = threading.Lock()
        self._dimension: Optional[int] = None

        # Initialize Rust accelerator for similarity operations
        # Note: Embedding generation still uses sentence-transformers (Python)
        # Only similarity computation uses the accelerator
        try:
            # Get configuration from settings if not provided
            if use_rust_embeddings is None or rust_embedding_batch_threshold is None:
                from victor.config.settings import load_settings

                settings = load_settings()
                if use_rust_embeddings is None:
                    use_rust_embeddings = getattr(settings, "use_rust_embedding_ops", True)
                if rust_embedding_batch_threshold is None:
                    rust_embedding_batch_threshold = getattr(
                        settings, "rust_embedding_batch_threshold", 10
                    )

            self._embedding_accelerator = get_embedding_accelerator(
                force_numpy=not use_rust_embeddings,
                enable_cache=True,
            )

            if self._embedding_accelerator.is_using_rust:
                logger.info(
                    f"[EmbeddingService] Rust-accelerated similarity enabled "
                    f"(threshold={rust_embedding_batch_threshold} vectors)"
                )
            else:
                logger.info("[EmbeddingService] Using NumPy for similarity operations")
        except Exception as e:
            logger.warning(f"[EmbeddingService] Failed to initialize accelerator: {e}")
            self._embedding_accelerator = get_embedding_accelerator(force_numpy=True)

        self._rust_batch_threshold = rust_embedding_batch_threshold or 10

        # In-memory embedding cache for repeated texts (e.g., tool descriptions)
        # Uses OrderedDict for LRU eviction order
        # Key: hash of text, Value: embedding vector
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # Memory-based eviction configuration
        # Each embedding: 384 floats * 4 bytes = 1,536 bytes (~1.5KB)
        # Default limit: 50MB allows ~32,000 embeddings
        self._max_cache_memory_bytes = 50 * 1024 * 1024  # 50MB default
        self._current_cache_memory_bytes = 0
        self._bytes_per_float = 4  # float32

        # Shutdown flag to prevent new operations after shutdown initiated
        self._shutdown = False

    @classmethod
    def get_instance(
        cls,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        use_rust_embeddings: Optional[bool] = None,
        rust_embedding_batch_threshold: Optional[int] = None,
    ) -> "EmbeddingService":
        """Get or create the singleton embedding service instance.

        Args:
            model_name: Model name (only used on first call)
            device: Device (only used on first call)
            use_rust_embeddings: Enable Rust-accelerated similarity (only used on first call)
            rust_embedding_batch_threshold: Batch size threshold for Rust (only used on first call)

        Returns:
            The singleton EmbeddingService instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = cls(
                        model_name=model_name,
                        device=device,
                        use_rust_embeddings=use_rust_embeddings,
                        rust_embedding_batch_threshold=rust_embedding_batch_threshold,
                    )
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

    def _estimate_embedding_memory(self, embedding: np.ndarray) -> int:
        """Estimate memory usage of an embedding in bytes.

        Args:
            embedding: Numpy array embedding

        Returns:
            Estimated memory usage in bytes
        """
        return int(embedding.size * self._bytes_per_float)

    def _evict_cache_for_memory(self, needed_bytes: int) -> int:
        """Evict oldest cache entries to free memory.

        Uses LRU eviction - removes entries from the front of OrderedDict
        until enough memory is freed.

        NOTE: This method should only be called while holding _model_lock.

        Args:
            needed_bytes: Bytes needed for new entry

        Returns:
            Number of entries evicted
        """
        evicted = 0
        target_memory = self._max_cache_memory_bytes - needed_bytes

        while self._current_cache_memory_bytes > target_memory and self._embedding_cache:
            # Pop oldest entry (from front of OrderedDict)
            key, embedding = self._embedding_cache.popitem(last=False)
            freed_bytes = self._estimate_embedding_memory(embedding)
            self._current_cache_memory_bytes -= freed_bytes
            evicted += 1
            logger.log(
                TRACE,
                f"[EmbeddingService] Evicted cache entry: freed={freed_bytes}B, "
                f"current_memory={self._current_cache_memory_bytes}B",
            )

        return evicted

    def embed_text_sync(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a single text (sync version).

        Args:
            text: Text to embed
            use_cache: Whether to use in-memory cache (default: True)

        Returns:
            Embedding vector as numpy array (float32)
        """
        # Check cache first (with lock for thread safety)
        if use_cache:
            cache_key = self._get_cache_key(text)
            with self._model_lock:  # Reuse model_lock for cache protection
                if cache_key in self._embedding_cache:
                    self._cache_hits += 1
                    # Move to end for LRU ordering (most recently used)
                    self._embedding_cache.move_to_end(cache_key)
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

            # Cache the result with memory-based eviction (with lock)
            if use_cache:
                entry_bytes = self._estimate_embedding_memory(result)

                with self._model_lock:  # Protect cache operations
                    # Check if we need to evict entries to make room
                    if self._current_cache_memory_bytes + entry_bytes > self._max_cache_memory_bytes:
                        self._evict_cache_for_memory(entry_bytes)

                    # Add to cache and update memory tracking
                    self._embedding_cache[cache_key] = result
                    self._current_cache_memory_bytes += entry_bytes

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
            Dictionary with cache stats including memory usage
        """
        with self._model_lock:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total if total > 0 else 0.0
            memory_mb = self._current_cache_memory_bytes / (1024 * 1024)
            max_memory_mb = self._max_cache_memory_bytes / (1024 * 1024)
            memory_utilization = (
                self._current_cache_memory_bytes / self._max_cache_memory_bytes
                if self._max_cache_memory_bytes > 0
                else 0.0
            )
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "size": len(self._embedding_cache),
                "hit_rate": f"{hit_rate:.1%}",
                "memory_used_mb": f"{memory_mb:.2f}",
                "max_memory_mb": f"{max_memory_mb:.2f}",
                "memory_utilization": f"{memory_utilization:.1%}",
            }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        with self._model_lock:
            self._embedding_cache.clear()
            self._current_cache_memory_bytes = 0
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

    def cosine_similarity_matrix(
        self,
        query: np.ndarray,
        corpus: np.ndarray,
    ) -> np.ndarray:
        """Calculate cosine similarity between query and all corpus vectors.

        Uses Rust accelerator for large batches, NumPy for small batches.
        Automatically selects backend based on batch size threshold.

        Performance:
            - Small batches (< threshold): NumPy (BLAS-optimized, no FFI overhead)
            - Large batches (>= threshold): Rust (SIMD + parallelization)

        Args:
            query: Query vector (shape: [dimension])
            corpus: Corpus matrix (shape: [n_items, dimension])

        Returns:
            Similarity scores (shape: [n_items])
        """
        if corpus.size == 0:
            return np.array([])

        # Choose backend based on batch size
        batch_size = len(corpus)

        if batch_size >= self._rust_batch_threshold and self._embedding_accelerator.is_using_rust:
            # Use Rust accelerator for large batches
            query_list = query.tolist()
            corpus_list = corpus.tolist()

            similarities = self._embedding_accelerator.batch_cosine_similarity(
                query=query_list,
                embeddings=corpus_list,
            )

            return np.array(similarities, dtype=np.float32)
        else:
            # Use NumPy for small batches (BLAS-optimized)
            query_norm = query / (np.linalg.norm(query) + 1e-9)
            corpus_norms = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)

            # Dot product
            similarities = np.dot(corpus_norms, query_norm)
            return np.asarray(similarities)

    def get_top_k_similar(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k most similar embeddings using Rust partial sort.

        Combines cosine similarity computation with efficient top-k selection.

        Args:
            query_embedding: Query vector (shape: [dimension])
            corpus_embeddings: Corpus matrix (shape: [n_items, dimension])
            k: Number of top results to return

        Returns:
            Tuple of (top_k_similarities, top_k_indices)
                - top_k_similarities: Similarity scores (shape: [k])
                - top_k_indices: Indices in corpus (shape: [k])
        """
        # Compute similarities
        similarities = self.cosine_similarity_matrix(query_embedding, corpus_embeddings)

        # Use Rust accelerator for top-k selection if available
        if self._embedding_accelerator.is_using_rust:
            similarities_list = similarities.tolist()
            top_k_indices_list = self._embedding_accelerator.topk_indices(similarities_list, k)
            top_k_indices = np.array(top_k_indices_list, dtype=np.int64)
            top_k_similarities = np.array([similarities[i] for i in top_k_indices_list], dtype=np.float32)
        else:
            # NumPy argpartition (O(n))
            if k >= len(similarities):
                top_k_indices = np.arange(len(similarities), dtype=np.int64)
                top_k_similarities = similarities
            else:
                top_k_unsorted = np.argpartition(-similarities, k)[:k]
                top_k = top_k_unsorted[np.argsort(-similarities[top_k_unsorted])]
                top_k_similarities = similarities[top_k]
                top_k_indices = top_k

        return top_k_similarities, top_k_indices

    def embed_with_timing(
        self,
        text: str,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Embed text and return timing information.

        Useful for performance monitoring and benchmarking.

        Args:
            text: Text to embed
            use_cache: Whether to use embedding cache

        Returns:
            Tuple of (embedding, elapsed_seconds)
        """
        import time

        start = time.perf_counter()
        embedding = self.embed_text_sync(text, use_cache=use_cache)
        elapsed = time.perf_counter() - start

        backend = "Rust" if self._embedding_accelerator.is_using_rust else "NumPy"
        logger.debug(
            f"[EmbeddingService] Embedded {len(text)} chars in {elapsed*1000:.2f}ms "
            f"(similarity backend: {backend})"
        )

        return embedding, elapsed


def get_embedding_service(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: Optional[str] = None,
    use_rust_embeddings: Optional[bool] = None,
    rust_embedding_batch_threshold: Optional[int] = None,
) -> EmbeddingService:
    """Convenience function to get the singleton embedding service instance.

    This is a convenience wrapper around EmbeddingService.get_instance()
    for easier importing and usage.

    Args:
        model_name: Model name (only used on first call)
        device: Device (only used on first call)
        use_rust_embeddings: Enable Rust-accelerated similarity (only used on first call)
        rust_embedding_batch_threshold: Batch size threshold for Rust (only used on first call)

    Returns:
        The singleton EmbeddingService instance

    Example:
        from victor.storage.embeddings.service import get_embedding_service

        service = get_embedding_service()
        embedding = await service.embed_text("Hello world")
    """
    return EmbeddingService.get_instance(
        model_name=model_name,
        device=device,
        use_rust_embeddings=use_rust_embeddings,
        rust_embedding_batch_threshold=rust_embedding_batch_threshold,
    )
