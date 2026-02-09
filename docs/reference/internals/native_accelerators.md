# Native Accelerators

Victor provides Rust-native accelerators for performance-critical operations,
  delivering 3-10x speedups through SIMD optimizations and parallel processing.

## Overview

All accelerators follow a consistent pattern:
- **Automatic fallback**: Gracefully degrade to Python if Rust is unavailable
- **Observability integration**: Automatic metrics collection via `NativeMetrics`
- **Protocol compliance**: Implement standard protocols for easy substitution
- **Type safety**: Full type hints for IDE support

## Available Accelerators

### ToolSelectorAccelerator

**Purpose**: High-performance tool selection operations

**Performance**: 3-10x faster than NumPy/Python equivalents

**Key Operations**:
- Cosine similarity batch computation
- Top-k selection with partial sort
- Category filtering
- Combined filter and rank

**Use Case**: Selecting relevant tools from a large registry based on semantic similarity

**Documentation**: [Tool Selector Accelerator](./tool_selector_accelerator.md)

```python
from victor.native.rust import get_tool_selector_accelerator

accelerator = get_tool_selector_accelerator()
results = accelerator.filter_and_rank(query, tools, names, categories, category_map, k=10)
```text

### RustSimilarityComputer

**Purpose**: Fast similarity computations for embeddings

**Performance**: 2-5x faster than NumPy (SIMD + parallel)

**Key Operations**:
- Single cosine similarity
- Batch cosine similarity
- Similarity matrix computation
- Top-k similar vectors

**Use Case**: Semantic search, document similarity, recommendation systems

```python
from victor.native.rust import RustSimilarityComputer

computer = RustSimilarityComputer()
similarities = computer.batch_cosine(query, corpus)
```

### RustTextChunker

**Purpose**: Line-aware text chunking

**Performance**: 3-10x faster than Python (SIMD byte counting)

**Key Operations**:
- Chunk with overlap (respecting line boundaries)
- Line counting (SIMD-optimized)
- Line boundary detection
- Line lookup by offset

**Use Case**: Splitting large files into manageable chunks for processing

```python
from victor.native.rust import RustTextChunker

chunker = RustTextChunker()
chunks = chunker.chunk_with_overlap(text, chunk_size=1000, overlap=100)
```text

### RustArgumentNormalizer

**Purpose**: Tool argument normalization

**Performance**: 5-10x faster than Python (optimized JSON parsing)

**Key Operations**:
- Normalize tool arguments
- JSON repair and type coercion
- Schema validation

**Use Case**: Pre-processing tool arguments before execution

```python
from victor.native.rust import RustArgumentNormalizer

normalizer = RustArgumentNormalizer()
normalized = normalizer.normalize_arguments(args, schema)
```

### RustAstIndexer

**Purpose**: AST-based code indexing

**Performance**: 10-50x faster than Python (Tree-sitter C bindings)

**Key Operations**:
- Extract functions
- Extract classes
- Extract imports
- Symbol extraction

**Use Case**: Fast code analysis and symbol extraction

```python
from victor.native.rust import RustAstIndexer

indexer = RustAstIndexer()
functions = indexer.extract_functions(source_code, lang="python")
```text

## Installation

Install Victor with the native extension:

```bash
pip install victor-ai[native]
```

Or build from source:

```bash
pip install victor-ai[native] --no-binary :all:
```text

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  (Tool Selection, Search, Code Analysis, etc.)         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Python Accelerator Wrapper               │
│  - Protocol-compliant interface                         │
│  - Observability hooks (metrics, tracing)               │
│  - Error handling and fallback                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Rust Native Layer                     │
│  - SIMD optimizations (wide crate)                      │
│  - Parallel processing (rayon)                          │
│  - Memory-efficient data structures                     │
└─────────────────────────────────────────────────────────┘
```text

## Observability Integration

All accelerators integrate with Victor's observability infrastructure:

### Automatic Metrics Collection

```python
from victor.native.observability import NativeMetrics

metrics = NativeMetrics.get_instance()

# Get stats for all operations
stats = metrics.get_stats()

# Get stats for specific operation
cosine_stats = metrics.get_stats("cosine_similarity_batch")

# Get summary
summary = metrics.get_summary()
print(f"Rust ratio: {summary['rust_ratio']:.1%}")
print(f"Avg duration: {summary['avg_duration_ms']:.2f}ms")
```

### OpenTelemetry Tracing

```python
from victor.native.observability import traced_native_call

with traced_native_call("tool_selection", {"num_tools": 100}) as span:
    results = accelerator.filter_and_rank(...)
    if span:
        span.set_attribute("results_count", len(results))
```text

## Performance Comparison

### Tool Selection Operations

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Cosine Similarity (100 tools) | 1.0ms | 0.1ms | **10x** |
| Top-K Selection | 0.03ms | 0.01ms | **3x** |
| Category Filtering | 0.05ms | 0.01ms | **5x** |
| Filter + Rank | 0.5ms | 0.12ms | **4x** |

### Similarity Computations

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Single Cosine | 0.01ms | 0.003ms | **3x** |
| Batch Cosine (100) | 1.0ms | 0.2ms | **5x** |
| Similarity Matrix | 100ms | 20ms | **5x** |
| Top-K Similar | 1.2ms | 0.3ms | **4x** |

### Text Chunking

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Count Lines | 0.5ms | 0.05ms | **10x** |
| Find Boundaries | 1.0ms | 0.2ms | **5x** |
| Chunk with Overlap | 2.0ms | 0.4ms | **5x** |

## Error Handling

All accelerators provide graceful fallback to Python:

```python
from victor.native.rust import ToolSelectorAccelerator

accelerator = ToolSelectorAccelerator()

# Automatically uses Rust if available, falls back to Python
try:
    results = accelerator.filter_and_rank(...)
except Exception as e:
    # Fallback already handled internally
    logger.error(f"Operation failed: {e}")
```

### Debugging Backend Issues

```python
accelerator = ToolSelectorAccelerator()

# Check backend
print(f"Backend: {accelerator.backend}")  # "rust" or "python"
print(f"Rust Available: {accelerator.rust_available}")

# Get version if available
if accelerator.rust_available:
    print(f"Version: {accelerator.get_version()}")
```text

## Testing

### Run All Tests

```bash
# Test all accelerators
pytest tests/integration/native/ -v

# Test specific accelerator
pytest tests/integration/native/test_tool_selector_accelerator.py -v

# With coverage
pytest tests/integration/native/ --cov=victor.native.rust --cov-report=html
```

### Force Python Backend

```python
from victor.native.rust import ToolSelectorAccelerator

# Test Python fallback
accelerator = ToolSelectorAccelerator(force_python=True)
assert accelerator.backend == "python"
```text

## Best Practices

### 1. Use Singleton Access

```python
# Good
from victor.native.rust import get_tool_selector_accelerator

accelerator = get_tool_selector_accelerator()

# Avoid
from victor.native.rust import ToolSelectorAccelerator
accelerator = ToolSelectorAccelerator()  # Creates new instance
```

### 2. Pre-compute Embeddings

```python
# Good: Pre-compute once
tool_embeddings = [embed(tool.description) for tool in tools]

# Use multiple times
for query in queries:
    similarities = accelerator.cosine_similarity_batch(embed(query), tool_embeddings)
```text

### 3. Use Combined Operations

```python
# Good: Single operation
results = accelerator.filter_and_rank(query, tools, names, categories, category_map, k=10)

# Avoid: Multiple operations
filtered = accelerator.filter_by_category(names, categories, category_map)
similarities = accelerator.cosine_similarity_batch(query, tools)
# ... manual filtering and ranking
```

### 4. Monitor Performance

```python
from victor.native.observability import NativeMetrics

metrics = NativeMetrics.get_instance()
stats = metrics.get_stats("cosine_similarity_batch")

if stats["rust_ratio"] < 0.8:
    logger.warning("Low Rust usage, check native extension")
```text

## Troubleshooting

### Rust Extension Not Available

**Symptom**: "Using Python fallback" warnings

**Solution**:
```bash
# Reinstall with native extension
pip install victor-ai[native] --force-reinstall

# Or build from source
pip install victor-ai[native] --no-binary :all:
```

### Performance Slower Than Expected

**Symptoms**: Operations not meeting expected performance

**Debugging**:
```python
# Check backend
accelerator = get_tool_selector_accelerator()
print(f"Backend: {accelerator.backend}")

# Check metrics
metrics = NativeMetrics.get_instance()
stats = metrics.get_stats()
print(stats)

# Look for high error rates or low Rust ratio
```text

### Import Errors

**Symptom**: `ImportError: No module named 'victor_native'`

**Solution**:
```bash
# Install native dependencies
pip install victor-ai[native]

# Or install maturin for building
pip install maturin
cd victor/native/python
maturin develop
```

## Contributing

### Adding a New Accelerator

1. **Create Rust implementation** in `victor/native/rust/{module}.py`
2. **Inherit from InstrumentedAccelerator** for automatic metrics
3. **Use `_timed_call()` context manager** for operation tracking
4. **Provide Python fallback** for compatibility
5. **Add integration tests** in `tests/integration/native/test_{module}.py`
6. **Update documentation** with performance characteristics

Example:
```python
from victor.native.observability import InstrumentedAccelerator

class MyAccelerator(InstrumentedAccelerator):
    def __init__(self):
        super().__init__(backend="rust" if _RUST_AVAILABLE else "python")

    def my_operation(self, data):
        with self._timed_call("my_operation"):
            if self._use_rust:
                return victor_native.my_operation(data)
            else:
                return self._python_my_operation(data)
```text

## References

- [Tool Selector Accelerator](./tool_selector_accelerator.md) - Detailed guide for tool selection
- [Observability Guide](../../observability/README.md) - Metrics and tracing
- [Performance Benchmarks](../../performance/benchmark_results.md) - Detailed performance analysis
- [Architecture Overview](../../architecture/overview.md) - Design patterns

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
