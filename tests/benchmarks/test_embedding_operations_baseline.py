"""
Comprehensive benchmarks for embedding operations.

This module provides baseline performance measurements for core embedding operations
including cosine similarity, top-k selection, cache operations, and matrix operations.
"""

import pytest
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Optional


# =============================================================================
# Cache Implementation for Benchmarking
# =============================================================================

class LRUCache:
    """Simple LRU cache implementation for benchmarking."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        """Get value from cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        """Put value in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self.cache


# =============================================================================
# Cosine Similarity Benchmarks
# =============================================================================

@pytest.mark.benchmark
def test_cosine_similarity_small_batch(benchmark):
    """Benchmark cosine similarity for 10 embeddings using loop-based approach."""
    query = np.random.rand(384).astype(np.float32)
    embeddings = np.random.rand(10, 384).astype(np.float32)

    def compute_batch():
        query_norm = np.linalg.norm(query)
        similarities = []
        for emb in embeddings:
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                sim = np.dot(query, emb) / (query_norm * emb_norm)
                similarities.append(sim)
        return similarities

    result = benchmark.pedantic(compute_batch, rounds=100, iterations=10)
    assert len(result) == 10


@pytest.mark.benchmark
def test_cosine_similarity_medium_batch(benchmark):
    """Benchmark cosine similarity for 50 embeddings using vectorized operations."""
    query = np.random.rand(384).astype(np.float32)
    embeddings = np.random.rand(50, 384).astype(np.float32)

    def compute_batch():
        query_norm = np.linalg.norm(query)
        embeddings_norm = np.linalg.norm(embeddings, axis=1)
        dot_products = np.dot(embeddings, query)
        similarities = dot_products / (query_norm * embeddings_norm)
        return similarities

    result = benchmark.pedantic(compute_batch, rounds=100, iterations=10)
    assert len(result) == 50


@pytest.mark.benchmark
def test_cosine_similarity_large_batch(benchmark):
    """Benchmark cosine similarity for 100 embeddings using vectorized operations."""
    query = np.random.rand(384).astype(np.float32)
    embeddings = np.random.rand(100, 384).astype(np.float32)

    def compute_batch():
        query_norm = np.linalg.norm(query)
        embeddings_norm = np.linalg.norm(embeddings, axis=1)
        dot_products = np.dot(embeddings, query)
        similarities = dot_products / (query_norm * embeddings_norm)
        return similarities

    result = benchmark.pedantic(compute_batch, rounds=100, iterations=10)
    assert len(result) == 100


@pytest.mark.benchmark
def test_cosine_similarity_very_large_batch(benchmark):
    """Benchmark cosine similarity for 500 embeddings using vectorized operations."""
    query = np.random.rand(384).astype(np.float32)
    embeddings = np.random.rand(500, 384).astype(np.float32)

    def compute_batch():
        query_norm = np.linalg.norm(query)
        embeddings_norm = np.linalg.norm(embeddings, axis=1)
        dot_products = np.dot(embeddings, query)
        similarities = dot_products / (query_norm * embeddings_norm)
        return similarities

    result = benchmark.pedantic(compute_batch, rounds=50, iterations=10)
    assert len(result) == 500


@pytest.mark.benchmark
def test_cosine_similarity_pre_normalized(benchmark):
    """Benchmark cosine similarity with pre-normalized embeddings."""
    query = np.random.rand(384).astype(np.float32)
    embeddings = np.random.rand(100, 384).astype(np.float32)

    # Pre-normalize
    query = query / np.linalg.norm(query)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def compute_batch():
        similarities = np.dot(embeddings, query)
        return similarities

    result = benchmark.pedantic(compute_batch, rounds=100, iterations=10)
    assert len(result) == 100


# =============================================================================
# Top-K Selection Benchmarks
# =============================================================================

@pytest.mark.benchmark
def test_topk_selection_sorting_small(benchmark):
    """Benchmark top-k using sorting for small dataset."""
    scores = np.random.rand(50).astype(np.float32)
    k = 5

    def select_topk_sort():
        indexed = np.argsort(scores)[-k:][::-1]
        return indexed

    result = benchmark.pedantic(select_topk_sort, rounds=100, iterations=10)
    assert len(result) == k


@pytest.mark.benchmark
def test_topk_selection_argpartition_small(benchmark):
    """Benchmark top-k using argpartition for small dataset."""
    scores = np.random.rand(50).astype(np.float32)
    k = 5

    def select_topk_partition():
        top_k_unsorted = np.argpartition(-scores, k)[:k]
        top_k = top_k_unsorted[np.argsort(-scores[top_k_unsorted])]
        return top_k

    result = benchmark.pedantic(select_topk_partition, rounds=100, iterations=10)
    assert len(result) == k


@pytest.mark.benchmark
def test_topk_selection_sorting_medium(benchmark):
    """Benchmark top-k using sorting for medium dataset."""
    scores = np.random.rand(100).astype(np.float32)
    k = 10

    def select_topk_sort():
        indexed = np.argsort(scores)[-k:][::-1]
        return indexed

    result = benchmark.pedantic(select_topk_sort, rounds=100, iterations=10)
    assert len(result) == k


@pytest.mark.benchmark
def test_topk_selection_argpartition_medium(benchmark):
    """Benchmark top-k using argpartition for medium dataset."""
    scores = np.random.rand(100).astype(np.float32)
    k = 10

    def select_topk_partition():
        top_k_unsorted = np.argpartition(-scores, k)[:k]
        top_k = top_k_unsorted[np.argsort(-scores[top_k_unsorted])]
        return top_k

    result = benchmark.pedantic(select_topk_partition, rounds=100, iterations=10)
    assert len(result) == k


@pytest.mark.benchmark
def test_topk_selection_sorting_large(benchmark):
    """Benchmark top-k using sorting for large dataset."""
    scores = np.random.rand(500).astype(np.float32)
    k = 20

    def select_topk_sort():
        indexed = np.argsort(scores)[-k:][::-1]
        return indexed

    result = benchmark.pedantic(select_topk_sort, rounds=50, iterations=10)
    assert len(result) == k


@pytest.mark.benchmark
def test_topk_selection_argpartition_large(benchmark):
    """Benchmark top-k using argpartition for large dataset."""
    scores = np.random.rand(500).astype(np.float32)
    k = 20

    def select_topk_partition():
        top_k_unsorted = np.argpartition(-scores, k)[:k]
        top_k = top_k_unsorted[np.argsort(-scores[top_k_unsorted])]
        return top_k

    result = benchmark.pedantic(select_topk_partition, rounds=50, iterations=10)
    assert len(result) == k


@pytest.mark.benchmark
def test_topk_with_scores(benchmark):
    """Benchmark top-k selection with scores extraction."""
    scores = np.random.rand(100).astype(np.float32)
    k = 10

    def select_topk_with_scores():
        top_k_indices = np.argpartition(-scores, k)[:k]
        top_k_scores = scores[top_k_indices]
        # Sort by score
        sorted_order = np.argsort(-top_k_scores)
        return top_k_indices[sorted_order], top_k_scores[sorted_order]

    indices, values = benchmark.pedantic(select_topk_with_scores, rounds=100, iterations=10)
    assert len(indices) == k
    assert len(values) == k


# =============================================================================
# Cache Operation Benchmarks
# =============================================================================

@pytest.mark.benchmark
def test_cache_hit_small(benchmark):
    """Benchmark cache hit operation with small cache."""
    cache = LRUCache(capacity=100)
    embedding = np.random.rand(384).astype(np.float32)
    cache.put("test_key", embedding)

    def cache_lookup():
        return cache.get("test_key")

    result = benchmark.pedantic(cache_lookup, rounds=1000, iterations=10)
    assert result is not None


@pytest.mark.benchmark
def test_cache_hit_large(benchmark):
    """Benchmark cache hit operation with large cache."""
    cache = LRUCache(capacity=10000)
    # Pre-fill cache
    for i in range(1000):
        cache.put(f"key_{i}", np.random.rand(384).astype(np.float32))

    embedding = np.random.rand(384).astype(np.float32)
    cache.put("test_key", embedding)

    def cache_lookup():
        return cache.get("test_key")

    result = benchmark.pedantic(cache_lookup, rounds=1000, iterations=10)
    assert result is not None


@pytest.mark.benchmark
def test_cache_miss(benchmark):
    """Benchmark cache miss operation."""
    cache = LRUCache(capacity=1000)

    def cache_lookup():
        return cache.get("nonexistent_key")

    result = benchmark.pedantic(cache_lookup, rounds=1000, iterations=10)
    assert result is None


@pytest.mark.benchmark
def test_cache_insert(benchmark):
    """Benchmark cache insert operation."""
    cache = LRUCache(capacity=1000)

    def cache_insert():
        embedding = np.random.rand(384).astype(np.float32)
        cache.put(f"key_{np.random.randint(0, 1000)}", embedding)

    benchmark.pedantic(cache_insert, rounds=1000, iterations=10)


@pytest.mark.benchmark
def test_cache_insert_with_eviction(benchmark):
    """Benchmark cache insert operation with eviction."""
    cache = LRUCache(capacity=100)
    # Pre-fill cache
    for i in range(100):
        cache.put(f"key_{i}", np.random.rand(384).astype(np.float32))

    def cache_insert():
        embedding = np.random.rand(384).astype(np.float32)
        cache.put(f"new_key_{np.random.randint(0, 100)}", embedding)

    benchmark.pedantic(cache_insert, rounds=1000, iterations=10)


@pytest.mark.benchmark
def test_cache_hit_rate(benchmark):
    """Benchmark cache with 80% hit rate."""
    cache = LRUCache(capacity=1000)
    # Pre-fill cache
    for i in range(800):
        cache.put(f"key_{i}", np.random.rand(384).astype(np.float32))

    lookup_keys = [f"key_{i}" if i < 800 else f"miss_{i}" for i in range(1000)]

    def mixed_lookups():
        results = []
        for key in lookup_keys:
            result = cache.get(key)
            results.append(result)
        return results

    result = benchmark.pedantic(mixed_lookups, rounds=100, iterations=10)
    assert len(result) == 1000


# =============================================================================
# Matrix Operation Benchmarks
# =============================================================================

@pytest.mark.benchmark
def test_similarity_matrix_small(benchmark):
    """Benchmark computing similarity matrix for small dataset."""
    queries = np.random.rand(5, 384).astype(np.float32)
    corpus = np.random.rand(20, 384).astype(np.float32)

    def compute_matrix():
        # Normalize
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)

        queries_normed = queries / query_norms
        corpus_normed = corpus / corpus_norms

        # Compute matrix
        matrix = np.dot(queries_normed, corpus_normed.T)
        return matrix

    result = benchmark.pedantic(compute_matrix, rounds=100, iterations=10)
    assert result.shape == (5, 20)


@pytest.mark.benchmark
def test_similarity_matrix_medium(benchmark):
    """Benchmark computing similarity matrix for medium dataset."""
    queries = np.random.rand(10, 384).astype(np.float32)
    corpus = np.random.rand(50, 384).astype(np.float32)

    def compute_matrix():
        # Normalize
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)

        queries_normed = queries / query_norms
        corpus_normed = corpus / corpus_norms

        # Compute matrix
        matrix = np.dot(queries_normed, corpus_normed.T)
        return matrix

    result = benchmark.pedantic(compute_matrix, rounds=100, iterations=10)
    assert result.shape == (10, 50)


@pytest.mark.benchmark
def test_similarity_matrix_large(benchmark):
    """Benchmark computing similarity matrix for large dataset."""
    queries = np.random.rand(20, 384).astype(np.float32)
    corpus = np.random.rand(100, 384).astype(np.float32)

    def compute_matrix():
        # Normalize
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)

        queries_normed = queries / query_norms
        corpus_normed = corpus / corpus_norms

        # Compute matrix
        matrix = np.dot(queries_normed, corpus_normed.T)
        return matrix

    result = benchmark.pedantic(compute_matrix, rounds=50, iterations=10)
    assert result.shape == (20, 100)


@pytest.mark.benchmark
def test_batch_normalization(benchmark):
    """Benchmark batch normalization of embeddings."""
    embeddings = np.random.rand(100, 384).astype(np.float32)

    def normalize_batch():
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        return normalized

    result = benchmark.pedantic(normalize_batch, rounds=100, iterations=10)
    assert result.shape == (100, 384)


@pytest.mark.benchmark
def test_matrix_multiplication_efficient(benchmark):
    """Benchmark efficient matrix multiplication for similarity."""
    queries = np.random.rand(10, 384).astype(np.float32)
    corpus = np.random.rand(50, 384).astype(np.float32)

    # Pre-normalize
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)

    def compute_similarity():
        return np.dot(queries, corpus.T)

    result = benchmark.pedantic(compute_similarity, rounds=100, iterations=10)
    assert result.shape == (10, 50)


# =============================================================================
# Combined Operation Benchmarks
# =============================================================================

@pytest.mark.benchmark
def test_full_search_pipeline(benchmark):
    """Benchmark full search pipeline: normalize, compute similarity, top-k."""
    query = np.random.rand(384).astype(np.float32)
    corpus = np.random.rand(100, 384).astype(np.float32)
    k = 10

    def search_pipeline():
        # Normalize query
        query_norm = query / np.linalg.norm(query)

        # Normalize corpus
        corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        corpus_normed = corpus / corpus_norms

        # Compute similarities
        similarities = np.dot(corpus_normed, query_norm)

        # Get top-k
        top_k_unsorted = np.argpartition(-similarities, k)[:k]
        top_k = top_k_unsorted[np.argsort(-similarities[top_k_unsorted])]

        return top_k, similarities[top_k]

    indices, scores = benchmark.pedantic(search_pipeline, rounds=100, iterations=10)
    assert len(indices) == k
    assert len(scores) == k


@pytest.mark.benchmark
def test_batch_search_pipeline(benchmark):
    """Benchmark batch search pipeline with multiple queries."""
    queries = np.random.rand(5, 384).astype(np.float32)
    corpus = np.random.rand(100, 384).astype(np.float32)
    k = 10

    def batch_search():
        # Normalize queries
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries_normed = queries / query_norms

        # Normalize corpus
        corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        corpus_normed = corpus / corpus_norms

        # Compute similarity matrix
        similarity_matrix = np.dot(queries_normed, corpus_normed.T)

        # Get top-k for each query
        results = []
        for i in range(len(queries)):
            top_k_unsorted = np.argpartition(-similarity_matrix[i], k)[:k]
            top_k = top_k_unsorted[np.argsort(-similarity_matrix[i][top_k_unsorted])]
            results.append(top_k)

        return results

    result = benchmark.pedantic(batch_search, rounds=50, iterations=10)
    assert len(result) == 5
    assert all(len(r) == k for r in result)


@pytest.mark.benchmark
def test_cached_search_pipeline(benchmark):
    """Benchmark search pipeline with caching."""
    query = np.random.rand(384).astype(np.float32)
    corpus = np.random.rand(100, 384).astype(np.float32)
    cache = LRUCache(capacity=1000)
    k = 10

    # Pre-cache some corpus embeddings
    for i in range(50):
        cache.put(f"doc_{i}", corpus[i])

    def cached_search():
        # Check cache first
        cached_results = []
        uncached_indices = []

        for i in range(min(20, len(corpus))):
            doc_key = f"doc_{i}"
            if doc_key in cache:
                cached_emb = cache.get(doc_key)
                # Compute similarity
                similarity = np.dot(query, cached_emb)
                cached_results.append((i, similarity))
            else:
                uncached_indices.append(i)

        # Compute for uncached
        if uncached_indices:
            uncached_embeddings = corpus[uncached_indices]
            norms = np.linalg.norm(uncached_embeddings, axis=1)
            query_norm = np.linalg.norm(query)
            similarities = np.dot(uncached_embeddings, query) / (norms * query_norm)
            for idx, sim in zip(uncached_indices, similarities):
                cached_results.append((idx, sim))

        # Sort and get top-k
        cached_results.sort(key=lambda x: x[1], reverse=True)
        top_k = [idx for idx, _ in cached_results[:k]]

        return top_k

    result = benchmark.pedantic(cached_search, rounds=100, iterations=10)
    assert len(result) == k


# =============================================================================
# Performance Comparison Tests
# =============================================================================

@pytest.mark.benchmark
def test_algorithm_comparison_sorting(benchmark):
    """
    Benchmark sorting approach for top-k selection.
    Use this for comparison with partition approach.
    """
    scores = np.random.rand(100).astype(np.float32)
    k = 10

    def sorting_method():
        indexed = np.argsort(scores)[-k:][::-1]
        return indexed

    result = benchmark.pedantic(sorting_method, rounds=100, iterations=10)
    assert len(result) == k


@pytest.mark.benchmark
def test_algorithm_comparison_partition(benchmark):
    """
    Benchmark argpartition approach for top-k selection.
    Use this for comparison with sorting approach.
    """
    scores = np.random.rand(100).astype(np.float32)
    k = 10

    def partition_method():
        top_k_unsorted = np.argpartition(-scores, k)[:k]
        top_k = top_k_unsorted[np.argsort(-scores[top_k_unsorted])]
        return top_k

    result = benchmark.pedantic(partition_method, rounds=100, iterations=10)
    assert len(result) == k


@pytest.mark.benchmark
def test_vectorization_comparison_loop(benchmark):
    """
    Benchmark loop-based cosine similarity computation.
    Use this for comparison with vectorized approach.
    """
    query = np.random.rand(384).astype(np.float32)
    embeddings = np.random.rand(50, 384).astype(np.float32)

    def loop_method():
        query_norm = np.linalg.norm(query)
        similarities = []
        for emb in embeddings:
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                sim = np.dot(query, emb) / (query_norm * emb_norm)
                similarities.append(sim)
        return np.array(similarities)

    result = benchmark.pedantic(loop_method, rounds=100, iterations=10)
    assert len(result) == 50


@pytest.mark.benchmark
def test_vectorization_comparison_vectorized(benchmark):
    """
    Benchmark vectorized cosine similarity computation.
    Use this for comparison with loop-based approach.
    """
    query = np.random.rand(384).astype(np.float32)
    embeddings = np.random.rand(50, 384).astype(np.float32)

    def vectorized_method():
        query_norm = np.linalg.norm(query)
        embeddings_norm = np.linalg.norm(embeddings, axis=1)
        dot_products = np.dot(embeddings, query)
        similarities = dot_products / (query_norm * embeddings_norm)
        return similarities

    result = benchmark.pedantic(vectorized_method, rounds=100, iterations=10)
    assert len(result) == 50


# =============================================================================
# Helper Functions
# =============================================================================

def generate_benchmark_report() -> str:
    """
    Generate a summary report of benchmark results.
    This function should be called after running the benchmarks.
    """
    return """
    # Embedding Operations Baseline Benchmark Report

    ## Test Categories

    ### 1. Cosine Similarity Benchmarks
    - Small batch (10 embeddings): Loop-based approach
    - Medium batch (50 embeddings): Vectorized operations
    - Large batch (100 embeddings): Vectorized operations
    - Very large batch (500 embeddings): Vectorized operations
    - Pre-normalized embeddings: Optimized computation

    ### 2. Top-K Selection Benchmarks
    - Sorting vs Argpartition (O(n log n) vs O(n))
    - Small dataset (50 items, k=5)
    - Medium dataset (100 items, k=10)
    - Large dataset (500 items, k=20)
    - Top-k with score extraction

    ### 3. Cache Operation Benchmarks
    - Cache hit (small and large caches)
    - Cache miss
    - Cache insert
    - Cache insert with eviction
    - Mixed hit/rate (80% hit rate)

    ### 4. Matrix Operation Benchmarks
    - Similarity matrix computation (5x20, 10x50, 20x100)
    - Batch normalization
    - Efficient matrix multiplication

    ### 5. Combined Operation Benchmarks
    - Full search pipeline
    - Batch search pipeline
    - Cached search pipeline

    ## Key Metrics to Monitor

    1. **Cosine Similarity**: Throughput (embeddings/sec)
    2. **Top-K Selection**: Time complexity differences
    3. **Cache Operations**: Hit/miss performance
    4. **Matrix Operations**: Memory bandwidth utilization
    5. **End-to-End**: Full pipeline latency

    ## Running the Benchmarks

    ```bash
    # Run all benchmarks
    pytest tests/benchmarks/test_embedding_operations_baseline.py -v

    # Run specific category
    pytest tests/benchmarks/test_embedding_operations_baseline.py::test_cosine_similarity_large_batch -v

    # Generate comparison report
    pytest tests/benchmarks/test_embedding_operations_baseline.py --benchmark-compare

    # Save results for later comparison
    pytest tests/benchmarks/test_embedding_operations_baseline.py --benchmark-save=baseline
    ```

    ## Expected Performance Patterns

    1. **Vectorized operations** should be 5-10x faster than loop-based
    2. **Argpartition** should outperform sorting for k << n
    3. **Cache hits** should be < 1μs, cache misses > 10μs
    4. **Matrix operations** should scale linearly with batch size
    """


if __name__ == "__main__":
    print(generate_benchmark_report())
