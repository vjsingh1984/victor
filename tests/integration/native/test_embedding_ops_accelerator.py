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

"""Integration tests for embedding operations accelerator.

Tests the Rust-backed embedding operations with Python fallback,
including cosine similarity, top-k selection, similarity matrix,
and caching behavior.
"""

import pytest
import time
from typing import List

from victor.native.accelerators import (
    EmbeddingOpsAccelerator,
    get_embedding_accelerator,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def accelerator():
    """Create a fresh embedding accelerator instance for each test."""
    return EmbeddingOpsAccelerator(enable_cache=True)


@pytest.fixture
def query_384():
    """Sample 384-dimensional query embedding (e.g., from sentence-transformers)."""
    # Generate a normalized vector for testing
    import random

    random.seed(42)
    vec = [random.uniform(-1, 1) for _ in range(384)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


@pytest.fixture
def query_768():
    """Sample 768-dimensional query embedding."""
    import random

    random.seed(43)
    vec = [random.uniform(-1, 1) for _ in range(768)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


@pytest.fixture
def embeddings_384():
    """Sample 384-dimensional embeddings matrix."""
    import random

    random.seed(44)
    embeddings = []
    for i in range(100):
        vec = [random.uniform(-1, 1) for _ in range(384)]
        norm = sum(x * x for x in vec) ** 0.5
        embeddings.append([x / norm for x in vec])
    return embeddings


@pytest.fixture
def embeddings_768():
    """Sample 768-dimensional embeddings matrix."""
    import random

    random.seed(45)
    embeddings = []
    for i in range(50):
        vec = [random.uniform(-1, 1) for _ in range(768)]
        norm = sum(x * x for x in vec) ** 0.5
        embeddings.append([x / norm for x in vec])
    return embeddings


@pytest.fixture
def similarity_scores():
    """Sample similarity scores for top-k testing."""
    import random

    random.seed(46)
    return [random.uniform(0.0, 1.0) for _ in range(100)]


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestBatchCosineSimilarity:
    """Tests for batch cosine similarity computation."""

    def test_basic_similarity(self, accelerator, query_384, embeddings_384):
        """Test basic cosine similarity computation."""
        similarities = accelerator.batch_cosine_similarity(query_384, embeddings_384)

        assert isinstance(similarities, list)
        assert len(similarities) == len(embeddings_384)
        assert all(isinstance(s, float) for s in similarities)
        # Cosine similarity should be in [-1, 1]
        assert all(-1.0 <= s <= 1.0 for s in similarities)

    def test_different_dimensions(self, accelerator, query_768, embeddings_768):
        """Test with 768-dimensional embeddings."""
        similarities = accelerator.batch_cosine_similarity(query_768, embeddings_768)

        assert len(similarities) == len(embeddings_768)
        assert all(-1.0 <= s <= 1.0 for s in similarities)

    def test_dimension_mismatch_raises_error(self, accelerator, query_384):
        """Test that dimension mismatch raises ValueError."""
        wrong_embeddings = [[0.1, 0.2, 0.3] for _ in range(10)]  # 3-dim

        with pytest.raises(ValueError, match="Dimension mismatch"):
            accelerator.batch_cosine_similarity(query_384, wrong_embeddings)

    def test_empty_embeddings(self, accelerator, query_384):
        """Test with empty embedding list."""
        similarities = accelerator.batch_cosine_similarity(query_384, [])
        assert similarities == []

    def test_identical_vectors_high_similarity(self, accelerator):
        """Test that identical vectors have similarity near 1.0."""
        vec = [0.1, 0.2, 0.3, 0.4]
        embeddings = [vec]

        similarities = accelerator.batch_cosine_similarity(vec, embeddings)
        assert len(similarities) == 1
        assert abs(similarities[0] - 1.0) < 1e-6

    def test_orthogonal_vectors_zero_similarity(self, accelerator):
        """Test that orthogonal vectors have similarity near 0.0."""
        vec1 = [1.0, 0.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0, 0.0]
        embeddings = [vec2]

        similarities = accelerator.batch_cosine_similarity(vec1, embeddings)
        assert len(similarities) == 1
        assert abs(similarities[0] - 0.0) < 1e-6


class TestTopKIndices:
    """Tests for top-k index selection."""

    def test_basic_topk(self, accelerator, similarity_scores):
        """Test basic top-k selection."""
        k = 10
        top_k = accelerator.topk_indices(similarity_scores, k)

        assert isinstance(top_k, list)
        assert len(top_k) == k
        assert all(isinstance(i, int) for i in top_k)
        assert all(0 <= i < len(similarity_scores) for i in top_k)

        # Verify scores are in descending order
        selected_scores = [similarity_scores[i] for i in top_k]
        assert selected_scores == sorted(selected_scores, reverse=True)

    def test_k_equals_list_length(self, accelerator, similarity_scores):
        """Test when k equals the list length."""
        k = len(similarity_scores)
        top_k = accelerator.topk_indices(similarity_scores, k)

        assert len(top_k) == k
        # Should return all indices
        assert set(top_k) == set(range(len(similarity_scores)))

    def test_k_exceeds_list_length(self, accelerator, similarity_scores):
        """Test when k exceeds the list length."""
        k = len(similarity_scores) + 10
        top_k = accelerator.topk_indices(similarity_scores, k)

        # Should return all indices
        assert set(top_k) == set(range(len(similarity_scores)))

    def test_negative_k_raises_error(self, accelerator, similarity_scores):
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError, match="k must be non-negative"):
            accelerator.topk_indices(similarity_scores, -1)

    def test_empty_scores(self, accelerator):
        """Test with empty score list."""
        top_k = accelerator.topk_indices([], 10)
        assert top_k == []

    def test_k_zero(self, accelerator, similarity_scores):
        """Test with k=0."""
        top_k = accelerator.topk_indices(similarity_scores, 0)
        assert top_k == []

    def test_duplicate_scores_stable(self, accelerator):
        """Test that duplicate scores are handled consistently."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        k = 3
        top_k = accelerator.topk_indices(scores, k)

        assert len(top_k) == k
        # All scores are the same, so any 3 indices are valid
        assert set(top_k).issubset(set(range(len(scores))))


class TestSimilarityMatrix:
    """Tests for similarity matrix computation."""

    def test_basic_matrix(self, accelerator, query_384, embeddings_384):
        """Test basic similarity matrix computation."""
        queries = [query_384, query_384]
        corpus = embeddings_384[:10]

        matrix = accelerator.similarity_matrix(queries, corpus)

        assert isinstance(matrix, list)
        assert len(matrix) == len(queries)
        assert all(len(row) == len(corpus) for row in matrix)
        assert all(isinstance(val, float) for row in matrix for val in row)

    def test_matrix_dimensions(self, accelerator):
        """Test matrix shape is correct."""
        queries = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        corpus = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        matrix = accelerator.similarity_matrix(queries, corpus)

        assert len(matrix) == 2  # 2 queries
        assert all(len(row) == 3 for row in matrix)  # 3 corpus items

    def test_empty_queries_returns_empty(self, accelerator, embeddings_384):
        """Test with empty query list."""
        matrix = accelerator.similarity_matrix([], embeddings_384[:10])
        assert matrix == [[]]

    def test_empty_corpus_returns_empty(self, accelerator, query_384):
        """Test with empty corpus."""
        matrix = accelerator.similarity_matrix([query_384], [])
        assert matrix == [[]]

    def test_dimension_mismatch_raises_error(self, accelerator):
        """Test that dimension mismatch raises ValueError."""
        queries = [[0.1, 0.2, 0.3]]
        corpus = [[0.1, 0.2, 0.3, 0.4]]  # Different dimension

        with pytest.raises(ValueError, match="Dimension mismatch"):
            accelerator.similarity_matrix(queries, corpus)


# =============================================================================
# Performance and Caching Tests
# =============================================================================


class TestCaching:
    """Tests for result caching functionality."""

    def test_cache_hit_on_repeat_query(self, accelerator, query_384, embeddings_384):
        """Test that repeated queries hit the cache."""
        # First call - cache miss
        similarities1 = accelerator.batch_cosine_similarity(
            query_384, embeddings_384[:10]
        )
        stats1 = accelerator.cache_stats

        # Second call - cache hit (if Rust is available)
        similarities2 = accelerator.batch_cosine_similarity(
            query_384, embeddings_384[:10]
        )
        stats2 = accelerator.cache_stats

        # Results should be identical
        assert similarities1 == similarities2

        if accelerator.is_using_rust:
            # Cache hits should increase
            assert stats2.similarity_cache_hits >= stats1.similarity_cache_hits

    def test_cache_stats_tracking(self, accelerator, query_384, embeddings_384):
        """Test that cache statistics are tracked correctly."""
        accelerator.batch_cosine_similarity(query_384, embeddings_384[:10])

        stats = accelerator.cache_stats
        assert stats.total_similarity_calls > 0
        assert stats.avg_similarity_ms >= 0

    def test_clear_cache(self, accelerator, query_384, embeddings_384):
        """Test cache clearing."""
        # Populate cache (only works if Rust is available)
        accelerator.batch_cosine_similarity(query_384, embeddings_384[:10])
        cache_size_before = accelerator.get_cache_size()

        if accelerator.is_using_rust:
            assert cache_size_before > 0

        # Clear cache
        accelerator.clear_cache()
        cache_size_after = accelerator.get_cache_size()
        assert cache_size_after == 0

    def test_cache_size(self, accelerator, query_384, embeddings_384):
        """Test cache size reporting."""
        assert accelerator.get_cache_size() == 0

        # Add some entries
        for i in range(5):
            query = [float(i) if j == 0 else 0.0 for j in range(384)]
            accelerator.batch_cosine_similarity(query, embeddings_384[:10])

        # Cache only populates when using Rust
        if accelerator.is_using_rust:
            assert accelerator.get_cache_size() > 0
        else:
            assert accelerator.get_cache_size() == 0

    def test_cache_disabled(self, query_384, embeddings_384):
        """Test with caching disabled."""
        accelerator = EmbeddingOpsAccelerator(enable_cache=False)

        for _ in range(3):
            accelerator.batch_cosine_similarity(query_384, embeddings_384[:10])

        # Cache should be empty
        assert accelerator.get_cache_size() == 0


class TestPerformance:
    """Tests for performance characteristics."""

    def test_batch_similarity_performance(self, accelerator, query_384, embeddings_384):
        """Test that batch similarity is reasonably fast."""
        start = time.perf_counter()
        similarities = accelerator.batch_cosine_similarity(
            query_384, embeddings_384
        )
        duration = time.perf_counter() - start

        # Should complete in reasonable time (< 1 second for 100 embeddings)
        assert duration < 1.0
        assert len(similarities) == len(embeddings_384)

    def test_topk_performance(self, accelerator, similarity_scores):
        """Test that top-k selection is fast."""
        import time

        start = time.perf_counter()
        top_k = accelerator.topk_indices(similarity_scores, 10)
        duration = time.perf_counter() - start

        # Should be very fast (< 10ms)
        assert duration < 0.01
        assert len(top_k) == 10

    def test_similarity_matrix_performance(self, accelerator):
        """Test similarity matrix performance with larger datasets."""
        import random

        random.seed(47)
        queries = [
            [random.uniform(-1, 1) for _ in range(384)] for _ in range(10)
        ]
        corpus = [
            [random.uniform(-1, 1) for _ in range(384)] for _ in range(100)
        ]

        start = time.perf_counter()
        matrix = accelerator.similarity_matrix(queries, corpus)
        duration = time.perf_counter() - start

        # Should complete in reasonable time
        assert duration < 2.0
        assert len(matrix) == 10
        assert all(len(row) == 100 for row in matrix)


# =============================================================================
# Rust vs NumPy Fallback Tests
# =============================================================================


class TestRustVsNumpyFallback:
    """Tests comparing Rust and NumPy implementations."""

    def test_numpy_fallback_produces_same_results(
        self, query_384, embeddings_384
    ):
        """Test that NumPy fallback produces identical results."""
        # Rust implementation (if available)
        rust_accelerator = EmbeddingOpsAccelerator(force_numpy=False)
        rust_similarities = rust_accelerator.batch_cosine_similarity(
            query_384, embeddings_384[:10]
        )

        # NumPy implementation
        numpy_accelerator = EmbeddingOpsAccelerator(force_numpy=True)
        numpy_similarities = numpy_accelerator.batch_cosine_similarity(
            query_384, embeddings_384[:10]
        )

        # Results should be nearly identical
        assert len(rust_similarities) == len(numpy_similarities)
        for r, n in zip(rust_similarities, numpy_similarities):
            assert abs(r - n) < 1e-5

    def test_topk_numpy_fallback_consistent(self, similarity_scores):
        """Test top-k consistency between implementations."""
        rust_accelerator = EmbeddingOpsAccelerator(force_numpy=False)
        rust_topk = rust_accelerator.topk_indices(similarity_scores, 10)

        numpy_accelerator = EmbeddingOpsAccelerator(force_numpy=True)
        numpy_topk = numpy_accelerator.topk_indices(similarity_scores, 10)

        # Should select the same indices
        assert rust_topk == numpy_topk

    def test_is_rust_available(self):
        """Test Rust availability detection."""
        accelerator = EmbeddingOpsAccelerator()
        # Should not crash
        assert isinstance(accelerator.is_rust_available, bool)
        assert isinstance(accelerator.is_using_rust, bool)

    def test_force_numpy_flag(self):
        """Test force_numpy flag works."""
        rust_accel = EmbeddingOpsAccelerator(force_numpy=False)
        numpy_accel = EmbeddingOpsAccelerator(force_numpy=True)

        # If Rust is available, rust_accel should use it
        # numpy_accel should always use NumPy
        if rust_accel.is_rust_available:
            assert rust_accel.is_using_rust
        assert not numpy_accel.is_using_rust


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_embedding_accelerator_returns_singleton(self):
        """Test that get_embedding_accelerator returns the same instance."""
        accel1 = get_embedding_accelerator()
        accel2 = get_embedding_accelerator()

        assert accel1 is accel2

    def test_singleton_with_different_params(self):
        """Test that singleton ignores params after first call."""
        accel1 = get_embedding_accelerator(force_numpy=False)
        accel2 = get_embedding_accelerator(force_numpy=True)

        # Should be the same instance
        assert accel1 is accel2

    def test_singleton_cache_stats_persist(self, query_384, embeddings_384):
        """Test that cache stats persist across singleton calls."""
        accel1 = get_embedding_accelerator()
        accel1.batch_cosine_similarity(query_384, embeddings_384[:10])

        accel2 = get_embedding_accelerator()
        stats2 = accel2.cache_stats

        # Stats should persist
        assert stats2.total_similarity_calls > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_embedding(self, accelerator, query_384):
        """Test with a single embedding."""
        embeddings = [[0.1] * 384]
        similarities = accelerator.batch_cosine_similarity(query_384, embeddings)

        assert len(similarities) == 1
        assert isinstance(similarities[0], float)

    def test_zero_vector_handling(self, accelerator):
        """Test handling of zero vectors."""
        zero_vec = [0.0] * 384
        normal_vec = [0.1] * 384

        # Should not crash or return NaN
        similarities = accelerator.batch_cosine_similarity(
            zero_vec, [normal_vec]
        )
        assert len(similarities) == 1
        # Similarity with zero vector should be 0 (after normalization)
        assert similarities[0] == 0.0

    def test_large_embedding_dimension(self, accelerator):
        """Test with larger embedding dimensions."""
        dim = 1536  # OpenAI text-embedding-ada-002
        query = [0.1] * dim
        embeddings = [[0.2] * dim for _ in range(10)]

        similarities = accelerator.batch_cosine_similarity(query, embeddings)

        assert len(similarities) == 10
        assert all(isinstance(s, float) for s in similarities)

    def test_very_small_k(self, accelerator, similarity_scores):
        """Test with k=1."""
        top_k = accelerator.topk_indices(similarity_scores, 1)

        assert len(top_k) == 1
        # Should be the index of the maximum score
        max_idx = similarity_scores.index(max(similarity_scores))
        assert top_k[0] == max_idx

    def test_negative_scores_topk(self, accelerator):
        """Test top-k with negative scores."""
        scores = [-0.5, -0.3, -0.8, -0.1, -0.9]
        top_k = accelerator.topk_indices(scores, 3)

        # Should select indices with highest (least negative) scores
        selected_scores = [scores[i] for i in top_k]
        assert selected_scores == sorted(selected_scores, reverse=True)
        assert top_k[0] == 3  # Index of -0.1 (highest score)


# =============================================================================
# Integration with Embedding Services
# =============================================================================


class TestRealWorldIntegration:
    """Tests simulating real-world usage patterns."""

    def test_semantic_search_workflow(self, accelerator, query_384, embeddings_384):
        """Test typical semantic search workflow."""
        # 1. Compute similarities
        similarities = accelerator.batch_cosine_similarity(
            query_384, embeddings_384
        )

        # 2. Get top-k results
        k = 5
        top_k = accelerator.topk_indices(similarities, k)

        # 3. Verify results
        assert len(top_k) == k
        top_scores = [similarities[i] for i in top_k]
        assert top_scores == sorted(top_scores, reverse=True)

    def test_batch_query_processing(self, accelerator):
        """Test processing multiple queries in batch."""
        import random

        random.seed(48)
        queries = [
            [random.uniform(-1, 1) for _ in range(384)] for _ in range(5)
        ]
        corpus = [
            [random.uniform(-1, 1) for _ in range(384)] for _ in range(50)
        ]

        # Compute similarity matrix
        matrix = accelerator.similarity_matrix(queries, corpus)

        # Get top-k for each query
        results = []
        for similarities in matrix:
            top_k = accelerator.topk_indices(similarities, k=10)
            results.append(top_k)

        assert len(results) == len(queries)
        assert all(len(r) == 10 for r in results)

    def test_cache_effectiveness_in_repeated_queries(self, accelerator):
        """Test cache effectiveness with repeated query patterns."""
        import random

        random.seed(49)
        query = [random.uniform(-1, 1) for _ in range(384)]
        embeddings = [
            [random.uniform(-1, 1) for _ in range(384)] for _ in range(100)
        ]

        # First query - populate cache
        _ = accelerator.batch_cosine_similarity(query, embeddings)

        # Repeated queries - should hit cache
        for _ in range(10):
            _ = accelerator.batch_cosine_similarity(query, embeddings)

        stats = accelerator.cache_stats

        if accelerator.is_using_rust:
            # Should have cache hits
            assert stats.similarity_cache_hits > 0
            # Cache hit rate should be high
            assert stats.cache_hit_rate > 0.5
