# Native Accelerators

High-performance Rust-accelerated implementations of critical operations throughout Victor.

## Available Accelerators

### AST Processor

**Location**: `victor.native.accelerators.ast_processor`

**Performance**: 10x faster parsing, 5-8x faster queries, 10-15x parallel symbol extraction

**Usage**:
```python
from victor.native.accelerators.ast_processor import get_ast_processor

processor = get_ast_processor()
tree = processor.parse_to_ast(source_code, language="python")
nodes = processor.execute_query(tree, query, language)
```

**Features**:
- Automatic fallback to Python when Rust unavailable
- Built-in LRU cache with configurable size
- Comprehensive error handling
- Performance monitoring

### Embedding Operations

**Location**: `victor.native.accelerators.embedding_ops`

**Performance**: 3-8x faster similarity computation, 2-3x faster top-k selection, 5-10x faster similarity matrix

**Usage**:
```python
from victor.native.accelerators import get_embedding_accelerator

accelerator = get_embedding_accelerator()

# Batch cosine similarity
similarities = accelerator.batch_cosine_similarity(query_embedding, corpus_embeddings)

# Top-k selection
top_k = accelerator.topk_indices(similarities, k=10)

# Similarity matrix (batch processing)
matrix = accelerator.similarity_matrix(queries, corpus)
```

**Features**:
- SIMD-accelerated vector operations
- Automatic fallback to NumPy
- Result caching with configurable TTL
- Thread-safe operations
- Performance monitoring

**Supported Dimensions**:
- 384-dim (sentence-transformers)
- 768-dim (larger transformers)
- 1536-dim (OpenAI embeddings)
- Any custom dimension

**Example: Semantic Search**:
```python
# Initialize accelerator
accelerator = get_embedding_accelerator(enable_cache=True)

# Compute similarities
query = [0.1, 0.2, ...]  # 384-dim query vector
corpus = [[0.3, 0.4, ...], ...]  # N x 384 corpus
similarities = accelerator.batch_cosine_similarity(query, corpus)

# Get top 10 results
top_k = accelerator.topk_indices(similarities, k=10)
for idx in top_k:
    print(f"Match {idx}: similarity={similarities[idx]:.4f}")
```

**Example: Batch Processing**:
```python
# Process multiple queries efficiently
queries = [[...], [...], ...]  # M x 384
corpus = [[...], [...], ...]   # N x 384

matrix = accelerator.similarity_matrix(queries, corpus)
# matrix[i][j] = similarity(queries[i], corpus[j])
```

**Performance Benchmarks**:
- Batch similarity (100 embeddings, 384-dim):
  - Rust (SIMD): 0.1-0.5ms
  - NumPy: 0.5-2ms
  - Speedup: 3-8x

- Top-k selection (100 scores):
  - Rust (partial sort): 0.01ms
  - NumPy (argpartition): 0.03ms
  - Speedup: 2-3x

- Similarity matrix (10 queries x 100 corpus):
  - Rust (parallel): 5-10x faster
  - NumPy (sequential): baseline

## Configuration

All accelerators respect Victor's settings system:

```yaml
# ~/.victor/profiles.yaml
use_rust_ast_processor: true
ast_cache_size: 1000
```

Or via environment variables:
```bash
export VICTOR_USE_RUST_AST_PROCESSOR=true
export VICTOR_AST_CACHE_SIZE=1000
```

## Installation

Install with native extensions:
```bash
pip install victor-ai[native]
```

Or build from source:
```bash
cd victor/native/rust
maturin develop --release
```

## Architecture

All accelerators follow this pattern:

1. **Protocol-based design**: Define interface first
2. **Rust implementation**: High-performance native code
3. **Python fallback**: Graceful degradation
4. **Singleton pattern**: Efficient resource usage
5. **Observable**: Performance monitoring built-in

## Adding New Accelerators

1. Create Rust implementation in `victor/native/rust/`
2. Create Python wrapper class
3. Add factory function `get_<accelerator>()`
4. Implement protocol interface
5. Add comprehensive error handling
6. Add tests
7. Update documentation

## Testing

Run accelerator tests:
```bash
pytest tests/unit/native/ -v
pytest tests/unit/coding/test_ast_processor_integration.py -v
```

## Performance

See individual accelerator documentation for benchmark results.

## Support

For issues or questions:
- Check `/docs/ast_processor_integration.md` for AST processor details
- Review test files for usage examples
- Check Victor main documentation for architecture
