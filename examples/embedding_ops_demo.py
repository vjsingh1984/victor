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

"""Embedding Operations Accelerator Demo - High-performance vector operations.

This demo demonstrates the Rust-accelerated embedding operations
for semantic search, similarity computation, and top-k selection.

Usage:
    # Basic demo with synthetic embeddings
    python examples/embedding_ops_demo.py

    # Compare Rust vs NumPy performance
    python examples/embedding_ops_demo.py --benchmark

    # Test with specific embedding dimensions
    python examples/embedding_ops_demo.py --dimensions 768

Example:
    >>> python examples/embedding_ops_demo.py
    Using Rust-accelerated embedding operations (3-8x faster)
    Generated 100 query vectors (384-dim)
    Generated 1000 corpus vectors (384-dim)
    Batch similarity: 0.23ms
    Top-k selection: 0.01ms
    Top 5 matches: [42, 87, 13, 95, 3]
"""

import argparse
import logging
import random
import time
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def generate_normalized_vector(dim: int, seed: int = None) -> List[float]:
    """Generate a normalized random vector.

    Args:
        dim: Vector dimension
        seed: Random seed for reproducibility

    Returns:
        Normalized vector of dimension dim
    """
    if seed is not None:
        random.seed(seed)

    vec = [random.uniform(-1, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5

    if norm == 0:
        return [0.0] * dim

    return [x / norm for x in vec]


def generate_embedding_matrix(n: int, dim: int, seed: int = None) -> List[List[float]]:
    """Generate a matrix of normalized embeddings.

    Args:
        n: Number of vectors
        dim: Vector dimension
        seed: Random seed

    Returns:
        List of n normalized vectors
    """
    if seed is not None:
        random.seed(seed)

    return [generate_normalized_vector(dim, seed + i) for i in range(n)]


def demo_basic_operations():
    """Demonstrate basic embedding operations."""
    from victor.native.accelerators import get_embedding_accelerator

    logger.info("=" * 70)
    logger.info("Embedding Operations Accelerator Demo")
    logger.info("=" * 70)
    logger.info("")

    # Initialize accelerator
    accelerator = get_embedding_accelerator()
    logger.info(f"Rust available: {accelerator.is_rust_available}")
    logger.info(f"Using Rust: {accelerator.is_using_rust}")
    logger.info(f"Cache enabled: {accelerator._enable_cache}")
    logger.info("")

    # Generate test data (384-dim, like sentence-transformers)
    dim = 384
    logger.info(f"Generating test data ({dim}-dimensional vectors)...")
    query = generate_normalized_vector(dim, seed=42)
    corpus = generate_embedding_matrix(1000, dim, seed=100)
    logger.info(f"Generated 1 query vector and {len(corpus)} corpus vectors")
    logger.info("")

    # 1. Batch Cosine Similarity
    logger.info("1. Batch Cosine Similarity")
    logger.info("-" * 70)
    start = time.perf_counter()
    similarities = accelerator.batch_cosine_similarity(query, corpus)
    duration = (time.perf_counter() - start) * 1000

    logger.info(f"   Computed {len(similarities)} similarities in {duration:.3f}ms")
    logger.info(f"   Average similarity: {sum(similarities) / len(similarities):.4f}")
    logger.info(f"   Max similarity: {max(similarities):.4f}")
    logger.info(f"   Min similarity: {min(similarities):.4f}")
    logger.info("")

    # 2. Top-K Selection
    k = 10
    logger.info(f"2. Top-{k} Selection")
    logger.info("-" * 70)
    start = time.perf_counter()
    top_k = accelerator.topk_indices(similarities, k)
    duration = (time.perf_counter() - start) * 1000

    logger.info(f"   Selected top {k} indices in {duration:.3f}ms")
    logger.info(f"   Top indices: {top_k[:5]}...")
    logger.info(f"   Top scores: {[f'{similarities[i]:.4f}' for i in top_k[:5]]}")
    logger.info("")

    # 3. Similarity Matrix (Batch Processing)
    logger.info("3. Similarity Matrix (Batch Processing)")
    logger.info("-" * 70)
    queries = generate_embedding_matrix(10, dim, seed=200)
    corpus_subset = corpus[:100]

    start = time.perf_counter()
    matrix = accelerator.similarity_matrix(queries, corpus_subset)
    duration = (time.perf_counter() - start) * 1000

    logger.info(f"   Computed {len(queries)}x{len(corpus_subset)} matrix in {duration:.3f}ms")
    logger.info(f"   Matrix shape: {len(matrix)}x{len(matrix[0])}")
    logger.info("")

    # 4. Cache Statistics
    logger.info("4. Cache Statistics")
    logger.info("-" * 70)
    stats = accelerator.cache_stats
    logger.info(f"   Total similarity calls: {stats.total_similarity_calls}")
    logger.info(f"   Total top-k calls: {stats.total_topk_calls}")
    logger.info(f"   Rust calls: {stats.total_rust_calls}")
    logger.info(f"   Fallback calls: {stats.total_fallback_calls}")
    logger.info(f"   Avg similarity time: {stats.avg_similarity_ms:.3f}ms")
    logger.info(f"   Avg top-k time: {stats.avg_topk_ms:.3f}ms")
    logger.info(f"   Cache size: {accelerator.get_cache_size()} entries")
    logger.info("")


def demo_semantic_search():
    """Demonstrate semantic search workflow."""
    from victor.native.accelerators import get_embedding_accelerator

    logger.info("=" * 70)
    logger.info("Semantic Search Workflow Demo")
    logger.info("=" * 70)
    logger.info("")

    accelerator = get_embedding_accelerator()

    # Simulate document embeddings
    dim = 384
    documents = [
        "Python programming tutorial for beginners",
        "Advanced machine learning algorithms",
        "Web development with Django and Flask",
        "Data visualization with Matplotlib",
        "Natural language processing techniques",
        "Database design and SQL optimization",
        "Cloud computing with AWS and Azure",
        "Cybersecurity best practices",
        "Mobile app development with React Native",
        "DevOps and CI/CD pipelines",
    ]

    logger.info(f"Simulating {len(documents)} documents...")
    corpus_embeddings = generate_embedding_matrix(len(documents), dim, seed=300)

    # Query
    query_text = "How to learn Python programming?"
    query_embedding = generate_normalized_vector(dim, seed=42)

    logger.info(f"Query: '{query_text}'")
    logger.info("")

    # Search
    start = time.perf_counter()
    similarities = accelerator.batch_cosine_similarity(query_embedding, corpus_embeddings)
    top_k = accelerator.topk_indices(similarities, k=3)
    duration = (time.perf_counter() - start) * 1000

    # Results
    logger.info("Top 3 matching documents:")
    logger.info("-" * 70)
    for rank, idx in enumerate(top_k, 1):
        score = similarities[idx]
        doc = documents[idx]
        logger.info(f"{rank}. [{score:.4f}] {doc}")

    logger.info("")
    logger.info(f"Search completed in {duration:.3f}ms")
    logger.info("")


def benchmark_rust_vs_numpy(dim: int = 384, num_runs: int = 100):
    """Benchmark Rust vs NumPy implementations.

    Args:
        dim: Embedding dimension
        num_runs: Number of benchmark runs
    """
    from victor.native.accelerators import EmbeddingOpsAccelerator

    logger.info("=" * 70)
    logger.info(f"Performance Benchmark (dim={dim}, runs={num_runs})")
    logger.info("=" * 70)
    logger.info("")

    # Generate test data
    query = generate_normalized_vector(dim, seed=42)
    corpus = generate_embedding_matrix(1000, dim, seed=100)

    # Benchmark Rust implementation
    rust_accelerator = EmbeddingOpsAccelerator(force_numpy=False)
    logger.info(
        f"Rust Implementation: {'Available' if rust_accelerator.is_rust_available else 'Unavailable'}"
    )

    if rust_accelerator.is_using_rust:
        times_rust = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = rust_accelerator.batch_cosine_similarity(query, corpus)
            times_rust.append((time.perf_counter() - start) * 1000)

        avg_rust = sum(times_rust) / len(times_rust)
        logger.info(f"  Avg time: {avg_rust:.3f}ms")
        logger.info(f"  Min time: {min(times_rust):.3f}ms")
        logger.info(f"  Max time: {max(times_rust):.3f}ms")
    else:
        avg_rust = None
        logger.info("  Rust implementation not available")

    logger.info("")

    # Benchmark NumPy implementation
    numpy_accelerator = EmbeddingOpsAccelerator(force_numpy=True)
    logger.info("NumPy Implementation:")

    times_numpy = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = numpy_accelerator.batch_cosine_similarity(query, corpus)
        times_numpy.append((time.perf_counter() - start) * 1000)

    avg_numpy = sum(times_numpy) / len(times_numpy)
    logger.info(f"  Avg time: {avg_numpy:.3f}ms")
    logger.info(f"  Min time: {min(times_numpy):.3f}ms")
    logger.info(f"  Max time: {max(times_numpy):.3f}ms")
    logger.info("")

    # Performance comparison
    if avg_rust:
        speedup = avg_numpy / avg_rust
        logger.info("=" * 70)
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Time saved per query: {avg_numpy - avg_rust:.3f}ms")
        logger.info(f"Time saved per 1000 queries: {(avg_numpy - avg_rust) * 1000 / 1000:.2f}s")
        logger.info("")


def demo_cache_effectiveness():
    """Demonstrate caching effectiveness."""
    from victor.native.accelerators import get_embedding_accelerator

    logger.info("=" * 70)
    logger.info("Cache Effectiveness Demo")
    logger.info("=" * 70)
    logger.info("")

    accelerator = get_embedding_accelerator()
    accelerator.clear_cache()

    # Generate test data
    dim = 384
    queries = [
        generate_normalized_vector(dim, seed=42),
        generate_normalized_vector(dim, seed=43),
        generate_normalized_vector(dim, seed=44),
    ]
    corpus = generate_embedding_matrix(100, dim, seed=100)

    logger.info("Testing cache with 3 repeated queries...")
    logger.info("")

    # First pass - cache misses
    logger.info("First pass (cache misses):")
    start = time.perf_counter()
    for i, query in enumerate(queries):
        _ = accelerator.batch_cosine_similarity(query, corpus)
        logger.info(f"  Query {i+1}: {accelerator.cache_stats.similarity_cache_misses} misses")
    duration_first = (time.perf_counter() - start) * 1000

    logger.info(f"Total time: {duration_first:.3f}ms")
    logger.info("")

    # Second pass - cache hits
    logger.info("Second pass (cache hits):")
    start = time.perf_counter()
    for i, query in enumerate(queries):
        _ = accelerator.batch_cosine_similarity(query, corpus)
        logger.info(f"  Query {i+1}: {accelerator.cache_stats.similarity_cache_hits} hits")
    duration_second = (time.perf_counter() - start) * 1000

    logger.info(f"Total time: {duration_second:.3f}ms")
    logger.info("")

    # Stats
    stats = accelerator.cache_stats
    logger.info("Cache Statistics:")
    logger.info("-" * 70)
    logger.info(f"Total calls: {stats.total_similarity_calls}")
    logger.info(f"Cache hits: {stats.similarity_cache_hits}")
    logger.info(f"Cache misses: {stats.similarity_cache_misses}")
    logger.info(f"Hit rate: {stats.cache_hit_rate:.2%}")
    logger.info(f"Time speedup: {duration_first / duration_second:.2f}x")
    logger.info("")


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description="Embedding Operations Accelerator Demo")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark comparing Rust vs NumPy",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=384,
        help="Embedding dimensions (default: 384)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark runs (default: 100)",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run semantic search demo",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Run cache effectiveness demo",
    )

    args = parser.parse_args()

    if args.benchmark:
        benchmark_rust_vs_numpy(dim=args.dimensions, num_runs=args.runs)
    elif args.search:
        demo_semantic_search()
    elif args.cache:
        demo_cache_effectiveness()
    else:
        # Run all demos
        demo_basic_operations()
        demo_semantic_search()
        demo_cache_effectiveness()


if __name__ == "__main__":
    main()
