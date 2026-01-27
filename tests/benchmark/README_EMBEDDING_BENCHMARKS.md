# Embedding Operations Benchmarks

Comprehensive baseline benchmarks for embedding operations including cosine similarity, top-k selection, caching, and matrix operations.

## Overview

This benchmark suite measures performance characteristics of core embedding operations used in semantic search and similarity matching:

- **Cosine Similarity**: Vectorized vs loop-based computation
- **Top-K Selection**: Sorting vs argpartition algorithms
- **Cache Operations**: Hit/miss performance, LRU eviction
- **Matrix Operations**: Batch similarity computation
- **Full Pipeline**: End-to-end search performance

## Test Structure

### 1. Cosine Similarity Benchmarks

Tests cosine similarity computation across different batch sizes and implementations:

```python
test_cosine_similarity_small_batch      # 10 embeddings, loop-based
test_cosine_similarity_medium_batch     # 50 embeddings, vectorized
test_cosine_similarity_large_batch      # 100 embeddings, vectorized
test_cosine_similarity_very_large_batch # 500 embeddings, vectorized
test_cosine_similarity_pre_normalized   # 100 embeddings, pre-normalized
```

**Key Insights:**
- Vectorized operations are 5-10x faster than loop-based
- Pre-normalization provides ~2x speedup
- Scales linearly with batch size

### 2. Top-K Selection Benchmarks

Compares sorting (O(n log n)) vs argpartition (O(n)) algorithms:

```python
test_topk_selection_sorting_small       # 50 items, k=5
test_topk_selection_argpartition_small  # 50 items, k=5
test_topk_selection_sorting_medium      # 100 items, k=10
test_topk_selection_argpartition_medium # 100 items, k=10
test_topk_selection_sorting_large       # 500 items, k=20
test_topk_selection_argpartition_large  # 500 items, k=20
test_topk_with_scores                   # With score extraction
```

**Key Insights:**
- Argpartition is 2-3x faster for k << n
- Benefit increases with dataset size
- Negligible difference for k ≈ n

### 3. Cache Operation Benchmarks

Tests LRU cache performance:

```python
test_cache_hit_small          # Small cache (100 items)
test_cache_hit_large          # Large cache (10,000 items)
test_cache_miss               # Cache miss scenario
test_cache_insert             # Insert operation
test_cache_insert_with_eviction  # Insert with LRU eviction
test_cache_hit_rate           # 80% hit rate scenario
```

**Key Insights:**
- Cache hits: < 1μs
- Cache misses: > 10μs
- 10x+ speedup for cached queries
- Large cache has minimal overhead

### 4. Matrix Operation Benchmarks

Tests batch similarity computation:

```python
test_similarity_matrix_small   # 5x20 matrix
test_similarity_matrix_medium  # 10x50 matrix
test_similarity_matrix_large   # 20x100 matrix
test_batch_normalization       # Normalize 100 embeddings
test_matrix_multiplication_efficient  # Pre-normalized
```

**Key Insights:**
- Matrix operations scale with O(n*m)
- Pre-normalization is critical
- Memory bandwidth is limiting factor

### 5. Combined Operation Benchmarks

Tests end-to-end search pipeline:

```python
test_full_search_pipeline      # Single query search
test_batch_search_pipeline     # Batch query search
test_cached_search_pipeline    # Search with caching
```

**Key Insights:**
- Full pipeline: 10-100μs for 100 embeddings
- Caching provides 5-10x speedup
- Batch processing is 2-3x more efficient

### 6. Algorithm Comparison

Direct comparison of different approaches:

```python
test_algorithm_comparison_sorting_vs_partition
test_vectorization_comparison_loop_vs_vectorized
```

## Running the Benchmarks

### Basic Usage

```bash
# Run all benchmarks
pytest tests/benchmarks/test_embedding_operations_baseline.py -v

# Run specific category
pytest tests/benchmarks/test_embedding_operations_baseline.py::test_cosine_similarity_large_batch -v

# Run with detailed output
pytest tests/benchmarks/test_embedding_operations_baseline.py -v --benchmark-only

# Run and generate histogram
pytest tests/benchmarks/test_embedding_operations_baseline.py --benchmark-histogram
```

### Saving and Comparing Results

```bash
# Save baseline results
pytest tests/benchmarks/test_embedding_operations_baseline.py --benchmark-save=baseline

# Run again and compare
pytest tests/benchmarks/test_embedding_operations_baseline.py --benchmark-compare=baseline

# Save with specific name
pytest tests/benchmarks/test_embedding_operations_baseline.py --benchmark-save=optimized_v1
```

### Using the Analysis Script

```bash
# Run benchmarks and analyze
python scripts/analyze_embedding_benchmarks.py run --save=baseline
python scripts/analyze_embedding_benchmarks.py analyze --results=.benchmarks/baseline.json

# Generate report to file
python scripts/analyze_embedding_benchmarks.py report --output=benchmark_report.md
```

## Expected Performance

### Cosine Similarity (per query)

| Batch Size | Mean Time | Throughput |
|------------|-----------|------------|
| 10         | ~5 μs     | 200K ops/s |
| 50         | ~20 μs    | 2.5M ops/s |
| 100        | ~40 μs    | 2.5M ops/s |
| 500        | ~200 μs   | 2.5M ops/s |

### Top-K Selection (for 100 items, k=10)

| Algorithm  | Mean Time | Speedup |
|------------|-----------|---------|
| Sorting    | ~15 μs    | 1.0x    |
| Argpartition | ~8 μs  | 1.9x    |

### Cache Operations

| Operation  | Mean Time |
|------------|-----------|
| Hit        | ~0.5 μs   |
| Miss       | ~10 μs    |
| Insert     | ~1 μs     |

### Full Search Pipeline (100 embeddings, k=10)

| Scenario     | Mean Time |
|--------------|-----------|
| No cache     | ~100 μs   |
| With cache   | ~15 μs    |
| Batch (5)    | ~200 μs   |

## Performance Optimization Guidelines

### 1. Use Vectorized Operations

**Bad:**
```python
similarities = []
for emb in embeddings:
    sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb))
    similarities.append(sim)
```

**Good:**
```python
query_norm = np.linalg.norm(query)
emb_norms = np.linalg.norm(embeddings, axis=1)
similarities = np.dot(embeddings, query) / (query_norm * emb_norms)
```

**Speedup: 5-10x**

### 2. Pre-normalize Embeddings

**Bad:**
```python
# Normalize at query time
similarities = []
for emb in embeddings:
    emb_normed = emb / np.linalg.norm(emb)
    sim = np.dot(query, emb_normed)
    similarities.append(sim)
```

**Good:**
```python
# Normalize once during indexing
embeddings_normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# At query time
query_normed = query / np.linalg.norm(query)
similarities = np.dot(embeddings_normed, query_normed)
```

**Speedup: 2x**

### 3. Use Argpartition for Top-K

**Bad:**
```python
top_k_indices = np.argsort(similarities)[-k:][::-1]
```

**Good:**
```python
top_k_unsorted = np.argpartition(-similarities, k)[:k]
top_k_indices = top_k_unsorted[np.argsort(-similarities[top_k_unsorted])]
```

**Speedup: 2-3x for k << n**

### 4. Implement Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(doc_id: str) -> np.ndarray:
    return load_embedding(doc_id)

# Or use custom LRU cache
cache = LRUCache(capacity=1000)
embedding = cache.get(doc_id)
if embedding is None:
    embedding = load_embedding(doc_id)
    cache.put(doc_id, embedding)
```

**Speedup: 10x for cached queries**

### 5. Use Appropriate Data Types

**Bad:**
```python
embeddings = np.random.rand(1000, 384)  # float64
```

**Good:**
```python
embeddings = np.random.rand(1000, 384).astype(np.float32)  # float32
```

**Memory: 2x reduction, Speed: 1.5x faster**

## Troubleshooting

### Benchmarks are slow

- Reduce `rounds` and `iterations` parameters
- Use smaller batch sizes for initial testing
- Close other applications to reduce noise

### High variance in results

- Increase `rounds` for more stable measurements
- Ensure system is idle (no background processes)
- Disable CPU frequency scaling

### Out of memory errors

- Reduce batch sizes in large tests
- Process in smaller chunks
- Monitor memory usage with `--benchmark-memory`

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Benchmark Tests

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest-benchmark

      - name: Run benchmarks
        run: |
          pytest tests/benchmarks/test_embedding_operations_baseline.py \
            --benchmark-only \
            --benchmark-json=benchmark_results.json

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmark_results.json

      - name: Compare with baseline
        if: github.event_name == 'pull_request'
        run: |
          python scripts/analyze_embedding_benchmarks.py report \
            --results=benchmark_results.json \
            --output=benchmark_report.md
```

## Contributing

When adding new benchmarks:

1. Follow existing naming conventions
2. Include multiple batch sizes
3. Add documentation and expected performance
4. Test on different hardware
5. Update this README

## Resources

- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [NumPy Performance Tips](https://numpy.org/doc/stable/reference/routines.performance.html)
- [Python Performance Optimization](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)

## License

MIT License - See LICENSE file for details
