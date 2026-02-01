# Embedding Operations Accelerator - Quick Reference

## Quick Start

```python
from victor.native.accelerators import get_embedding_accelerator

# Get singleton instance
accelerator = get_embedding_accelerator()

# Compute similarities
similarities = accelerator.batch_cosine_similarity(query_embedding, corpus_embeddings)

# Get top-k results
top_k = accelerator.topk_indices(similarities, k=10)

# View statistics
stats = accelerator.cache_stats
print(f"Avg time: {stats.avg_similarity_ms:.3f}ms")
print(f"Cache hit rate: {stats.cache_hit_rate:.2%}")
```

## API Reference

### EmbeddingOpsAccelerator

#### Constructor
```python
accelerator = EmbeddingOpsAccelerator(
    force_numpy=False,        # Force NumPy implementation
    enable_cache=True,         # Enable result caching
    cache_ttl_seconds=3600,    # Cache TTL (default: 1 hour)
)
```

#### Methods

**batch_cosine_similarity(query, embeddings, use_cache=None)**
- Compute cosine similarities between query and embeddings
- Returns: `List[float]`
- Raises: `ValueError` if dimensions mismatch

```python
similarities = accelerator.batch_cosine_similarity(
    query=[0.1, 0.2, ...],      # 384-dim query
    embeddings=[[0.3, 0.4, ...], ...],  # N x 384 corpus
)
```

**topk_indices(scores, k)**
- Select top-k indices using partial sort
- Returns: `List[int]`
- Raises: `ValueError` if k is negative

```python
top_k = accelerator.topk_indices(
    scores=[0.9, 0.3, 0.8, ...],
    k=10,
)
```

**similarity_matrix(queries, corpus)**
- Compute similarity matrix for batch processing
- Returns: `List[List[float]]` (M x N matrix)
- Raises: `ValueError` if dimensions mismatch

```python
matrix = accelerator.similarity_matrix(
    queries=[[0.1, 0.2, ...], ...],    # M x 384
    corpus=[[0.3, 0.4, ...], ...],     # N x 384
)
# matrix[i][j] = similarity(queries[i], corpus[j])
```

**Properties**
```python
accelerator.is_rust_available   # bool: Rust implementation available
accelerator.is_using_rust        # bool: Currently using Rust
accelerator.cache_stats          # EmbeddingCacheStats: Performance stats
accelerator.get_cache_size()    # int: Current cache size
accelerator.clear_cache()        # None: Clear the cache
```

### EmbeddingCacheStats

```python
stats = accelerator.cache_stats

# Properties
stats.similarity_cache_hits      # int: Number of cache hits
stats.similarity_cache_misses    # int: Number of cache misses
stats.topk_cache_hits            # int: Top-k cache hits
stats.topk_cache_misses          # int: Top-k cache misses
stats.total_rust_calls           # int: Total Rust function calls
stats.total_fallback_calls       # int: Total NumPy fallback calls
stats.avg_similarity_ms          # float: Average similarity time (ms)
stats.avg_topk_ms                # float: Average top-k time (ms)
stats.cache_hit_rate             # float: Overall cache hit rate (0-1)
```

## Usage Patterns

### Semantic Search
```python
from victor.native.accelerators import get_embedding_accelerator

accelerator = get_embedding_accelerator(enable_cache=True)

# Compute similarities
similarities = accelerator.batch_cosine_similarity(
    query_embedding,
    document_embeddings,
)

# Get top 10 matches
top_k = accelerator.topk_indices(similarities, k=10)

# Display results
for rank, idx in enumerate(top_k, 1):
    score = similarities[idx]
    document = documents[idx]
    print(f"{rank}. [{score:.4f}] {document}")
```

### Batch Processing
```python
# Process multiple queries efficiently
queries = [query1, query2, query3, ...]
corpus = [doc1, doc2, doc3, ...]

# Compute similarity matrix
matrix = accelerator.similarity_matrix(queries, corpus)

# Get top-k for each query
for i, similarities in enumerate(matrix):
    top_k = accelerator.topk_indices(similarities, k=5)
    print(f"Query {i}: top docs = {top_k}")
```

### Cache Monitoring
```python
accelerator = get_embedding_accelerator()

# Perform operations
for query in queries:
    similarities = accelerator.batch_cosine_similarity(query, corpus)
    top_k = accelerator.topk_indices(similarities, k=10)

# Check performance
stats = accelerator.cache_stats
print(f"Calls: {stats.total_similarity_calls}")
print(f"Avg time: {stats.avg_similarity_ms:.3f}ms")
print(f"Hit rate: {stats.cache_hit_rate:.2%}")
print(f"Cache size: {accelerator.get_cache_size()}")

# Clear cache if needed
accelerator.clear_cache()
```

## Performance Tips

1. **Enable caching for repeated queries**
   ```python
   accelerator = get_embedding_accelerator(enable_cache=True)
   ```

2. **Use batch processing for multiple queries**
   ```python
   # Faster: Single matrix computation
   matrix = accelerator.similarity_matrix(queries, corpus)

   # Slower: Multiple individual computations
   for query in queries:
       similarities = accelerator.batch_cosine_similarity(query, corpus)
   ```

3. **Adjust cache TTL based on query patterns**
   ```python
   # Short TTL for dynamic data
   accelerator = EmbeddingOpsAccelerator(cache_ttl_seconds=300)

   # Long TTL for static data
   accelerator = EmbeddingOpsAccelerator(cache_ttl_seconds=7200)
   ```

4. **Monitor cache effectiveness**
   ```python
   stats = accelerator.cache_stats
   if stats.cache_hit_rate < 0.3:
       print("Consider disabling cache or adjusting TTL")
   ```

## Supported Embedding Dimensions

| Dimension | Model Examples |
|-----------|---------------|
| 384 | sentence-transformers all-MiniLM-L6-v2 |
| 768 | sentence-transformers all-mpnet-base-v2 |
| 1024 | Cohere embed-english-v3.0 |
| 1536 | OpenAI text-embedding-ada-002 |
| Custom | Any dimension (validated at runtime) |

## Error Handling

```python
try:
    # Dimension mismatch
    similarities = accelerator.batch_cosine_similarity(
        query=[0.1] * 384,
        embeddings=[[0.2] * 768],  # Wrong dimension!
    )
except ValueError as e:
    print(f"Dimension error: {e}")

try:
    # Invalid k value
    top_k = accelerator.topk_indices(scores, k=-1)
except ValueError as e:
    print(f"Invalid k: {e}")

# Empty inputs - no error, returns empty list
similarities = accelerator.batch_cosine_similarity(query, [])
assert similarities == []
```

## Testing

```python
# Test basic operations
from victor.native.accelerators import get_embedding_accelerator
import random

accelerator = get_embedding_accelerator()

# Generate test data
query = [random.uniform(-1, 1) for _ in range(384)]
corpus = [[random.uniform(-1, 1) for _ in range(384)] for _ in range(100)]

# Test similarity
similarities = accelerator.batch_cosine_similarity(query, corpus)
assert len(similarities) == 100
assert all(-1.0 <= s <= 1.0 for s in similarities)

# Test top-k
top_k = accelerator.topk_indices(similarities, k=10)
assert len(top_k) == 10

# Test matrix
queries = [query] * 5
matrix = accelerator.similarity_matrix(queries, corpus[:10])
assert len(matrix) == 5
assert all(len(row) == 10 for row in matrix)

print("✓ All tests passed")
```

## Configuration

### Environment Variables
```bash
# Note: These would need to be implemented in settings.py
export VICTOR_USE_RUST_EMBEDDING_OPS=true
export VICTOR_EMBEDDING_CACHE_SIZE=1000
export VICTOR_EMBEDDING_CACHE_TTL=3600
```

### Python Configuration
```python
# Force NumPy implementation
accelerator = EmbeddingOpsAccelerator(force_numpy=True)

# Disable caching
accelerator = EmbeddingOpsAccelerator(enable_cache=False)

# Custom cache TTL
accelerator = EmbeddingOpsAccelerator(cache_ttl_seconds=1800)
```

## Performance Benchmarks

### Current (NumPy fallback)
- Batch similarity (100 embeddings, 384-dim): ~7.7ms
- Top-k selection (100 scores): ~0.08ms
- Similarity matrix (10x100): ~0.87ms

### Expected (Rust implementation)
- Batch similarity: ~0.1-0.5ms (3-8x faster)
- Top-k selection: ~0.01ms (2-3x faster)
- Similarity matrix: 5-10x faster

## Common Issues

**Issue**: Cache hit rate is low
**Solution**: Adjust cache TTL or disable caching for dynamic data

**Issue**: Out of memory errors
**Solution**: Reduce corpus size or disable caching

**Issue**: Slow performance
**Solution**: Install Rust implementation: `pip install victor-ai[native]`

## Integration Examples

### RAG Pipeline
```python
# Fast document retrieval for RAG
accelerator = get_embedding_accelerator()
doc_similarities = accelerator.batch_cosine_similarity(
    query_embedding,
    document_embeddings
)
top_docs = accelerator.topk_indices(doc_similarities, k=5)
retrieved_docs = [documents[i] for i in top_docs]
```

### Vector Database
```python
# Batch lookup for vector database
matrix = accelerator.similarity_matrix(
    query_embeddings,
    database_vectors,
)
# matrix[i][j] = similarity(query[i], db_vector[j])
```

### Document Clustering
```python
# Compute document-to-document similarities
matrix = accelerator.similarity_matrix(
    document_embeddings,
    document_embeddings,
)
# Use matrix for clustering algorithms
```

## Further Reading

- Implementation details: `docs/embedding_ops_implementation.md`
- API documentation: `victor/native/accelerators/embedding_ops.py`
- Test examples: `tests/integration/native/test_embedding_ops_accelerator.py`
- Demo script: `examples/embedding_ops_demo.py`

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
