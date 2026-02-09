# Tool Selection Accelerator

High-performance tool selection operations using Rust-native implementations for critical path optimizations.

## Overview

The `ToolSelectorAccelerator` provides 3-10x faster tool selection operations through native Rust implementations. It automatically falls back to Python implementations when Rust is unavailable, ensuring compatibility across all environments.

## Performance Characteristics

| Operation | Rust Performance | Python Performance | Speedup |
|-----------|-----------------|-------------------|---------|
| Cosine Similarity (batch) | ~0.1ms for 100 tools | ~1ms for 100 tools | **5-10x** |
| Top-K Selection | ~0.01ms for 100 scores | ~0.03ms for 100 scores | **2-3x** |
| Category Filtering | ~0.01ms for 100 tools | ~0.05ms for 100 tools | **3-5x** |
| Combined Filter+Rank | ~0.12ms for 100 tools | ~0.5ms for 100 tools | **4-5x** |

## Installation

The Rust accelerator is included in the `victor-ai` package with the `native` extra:

```bash
pip install victor-ai[native]
```text

If the Rust extension is not available, the accelerator automatically uses Python fallbacks.

## Quick Start

```python
from victor.native.rust import get_tool_selector_accelerator

# Get accelerator instance
accelerator = get_tool_selector_accelerator()

# Compute similarities
query = [0.1, 0.2, 0.3, ...]  # 384-dimensional query embedding
tools = [[0.4, 0.5, 0.6, ...], ...]  # Tool embeddings
similarities = accelerator.cosine_similarity_batch(query, tools)

# Get top-k results
top_k_indices = accelerator.topk_indices(similarities, k=10)

# Filter by category
tool_names = ["read_file", "write_file", "search_files", ...]
available_categories = {"file_ops", "git"}
tool_category_map = {"read_file": "file_ops", "write_file": "file_ops", ...}
filtered_tools = accelerator.filter_by_category(
    tool_names,
    available_categories,
    tool_category_map
)

# Combined filter and rank (most efficient)
results = accelerator.filter_and_rank(
    query=query,
    tools=tools,
    tool_names=tool_names,
    available_categories=available_categories,
    tool_category_map=tool_category_map,
    k=10
)
# Returns: [("tool_name", similarity), ...]
```

## API Reference

### ToolSelectorAccelerator

Main class for high-performance tool selection.

#### Initialization

```python
from victor.native.rust import ToolSelectorAccelerator

# Auto-detect Rust availability
accelerator = ToolSelectorAccelerator()

# Force Python implementation
accelerator = ToolSelectorAccelerator(force_python=True)

# Check backend
print(accelerator.backend)  # "rust" or "python"
print(accelerator.rust_available)  # True or False
```text

#### Methods

##### cosine_similarity_batch

Compute cosine similarities between query and multiple tools.

```python
similarities = accelerator.cosine_similarity_batch(
    query: List[float],      # Query embedding (384 dimensions)
    tools: List[List[float]] # Tool embeddings
) -> List[float]            # Similarity scores [-1, 1]
```

**Performance:**
- Rust: ~0.1ms for 100 tools
- Python: ~1ms for 100 tools

**Example:**
```python
query = embedding_service.embed("Read and analyze Python files")
tools = [embedding_service.embed(tool.description) for tool in tool_registry]
similarities = accelerator.cosine_similarity_batch(query, tools)

# Find most similar tools
top_tools = sorted(zip(tool_names, similarities), key=lambda x: x[1], reverse=True)[:5]
```text

##### topk_indices

Select top-k indices from scores.

```python
indices = accelerator.topk_indices(
    scores: List[float],  # Similarity scores
    k: int               # Number of top results
) -> List[int]          # Indices with highest scores
```

**Performance:**
- Rust: ~0.01ms for 100 scores
- Python: ~0.03ms for 100 scores

**Example:**
```python
similarities = [0.1, 0.9, 0.5, 0.3, 0.8]
top_3 = accelerator.topk_indices(similarities, k=3)
# Returns: [1, 4, 2] (indices of 0.9, 0.8, 0.5)
```text

##### topk_with_scores

Select top-k (index, score) pairs.

```python
results = accelerator.topk_with_scores(
    scores: List[float],            # Similarity scores
    k: int                         # Number of top results
) -> List[Tuple[int, float]]      # (index, score) pairs
```

**Example:**
```python
similarities = [0.1, 0.9, 0.5, 0.3, 0.8]
top_3 = accelerator.topk_with_scores(similarities, k=3)
# Returns: [(1, 0.9), (4, 0.8), (2, 0.5)]
```text

##### filter_by_category

Filter tools by category membership.

```python
filtered = accelerator.filter_by_category(
    tools: List[str],               # Tool names
    available_categories: Set[str], # Allowed categories
    tool_category_map: Dict[str, str]  # Tool -> category mapping
) -> List[str]                     # Filtered tool names
```

**Performance:**
- Rust: ~0.01ms for 100 tools
- Python: ~0.05ms for 100 tools

**Example:**
```python
tools = ["read_file", "write_file", "search_files", "run_command"]
categories = {"file_ops", "git"}
category_map = {
    "read_file": "file_ops",
    "write_file": "file_ops",
    "search_files": "search",
    "run_command": "execution"
}

filtered = accelerator.filter_by_category(tools, categories, category_map)
# Returns: ["read_file", "write_file"]
```text

##### filter_and_rank

Combined filter and rank operation (most efficient).

```python
results = accelerator.filter_and_rank(
    query: List[float],             # Query embedding
    tools: List[List[float]],       # Tool embeddings
    tool_names: List[str],          # Tool names
    available_categories: Set[str], # Allowed categories
    tool_category_map: Dict[str, str],  # Tool -> category mapping
    k: int                         # Number of top results
) -> List[Tuple[str, float]]      # (tool_name, similarity) pairs
```

**Performance:**
- Combined operation is ~20-30% faster than separate calls
- Reduces computation by filtering before similarity calculation

**Example:**
```python
query = embedding_service.embed("Git operations")
tools = [embedding_service.embed(t.description) for t in tool_registry]
names = [t.name for t in tool_registry]
categories = {"git", "file_ops"}
category_map = {t.name: t.category for t in tool_registry}

# Get top 5 git-related tools
top_5 = accelerator.filter_and_rank(query, tools, names, categories, category_map, k=5)
# Returns: [("git_commit", 0.95), ("create_pull_request", 0.87), ...]
```text

### Singleton Access

Use the singleton accessor for convenient access to the default accelerator:

```python
from victor.native.rust import (
    get_tool_selector_accelerator,
    reset_tool_selector_accelerator
)

# Get default instance (auto-detects Rust)
accelerator = get_tool_selector_accelerator()

# Force Python implementation
python_accelerator = get_tool_selector_accelerator(force_python=True)

# Reset singleton (useful for testing)
reset_tool_selector_accelerator()
```

## Observability Integration

The accelerator integrates with Victor's observability infrastructure for automatic metrics collection:

```python
from victor.native.observability import NativeMetrics

# Get metrics instance
metrics = NativeMetrics.get_instance()

# Get statistics for all operations
stats = metrics.get_stats()
print(stats)

# Get statistics for specific operation
cosine_stats = metrics.get_stats("cosine_similarity_batch")
print(f"Cosine similarity: {cosine_stats}")

# Get summary across all operations
summary = metrics.get_summary()
print(f"Total calls: {summary['total_calls']}")
print(f"Rust ratio: {summary['rust_ratio']:.1%}")
print(f"Average duration: {summary['avg_duration_ms']:.2f}ms")
```text

### Metrics Collected

For each operation, the following metrics are collected:
- **calls_total**: Total number of calls
- **duration_ms_total**: Total duration in milliseconds
- **duration_ms_avg**: Average duration per call
- **errors_total**: Number of errors
- **rust_calls**: Number of Rust backend calls
- **python_calls**: Number of Python fallback calls
- **rust_ratio**: Ratio of Rust to total calls

### Operations Tracked

- `cosine_similarity_batch` / `cosine_similarity_batch_python`
- `topk_indices` / `topk_indices_python`
- `topk_with_scores` / `topk_with_scores_python`
- `filter_by_category` / `filter_by_category_python`

## Advanced Usage

### Custom Embedding Service Integration

```python
from victor.native.rust import get_tool_selector_accelerator
from victor.processing.embeddings import EmbeddingService

class ToolSelector:
    def __init__(self, embedding_service: EmbeddingService):
        self.embeddings = embedding_service
        self.accelerator = get_tool_selector_accelerator()

        # Pre-compute tool embeddings
        self.tools = self._load_tools()
        self.tool_embeddings = [
            self.embeddings.embed(tool.description)
            for tool in self.tools
        ]
        self.tool_names = [tool.name for tool in self.tools]
        self.category_map = {tool.name: tool.category for tool in self.tools}

    def select_tools(
        self,
        query: str,
        categories: Set[str],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Select top-k tools for a query."""
        query_embedding = self.embeddings.embed(query)

        return self.accelerator.filter_and_rank(
            query=query_embedding,
            tools=self.tool_embeddings,
            tool_names=self.tool_names,
            available_categories=categories,
            tool_category_map=self.category_map,
            k=k
        )
```

### Batch Processing

```python
from victor.native.rust import get_tool_selector_accelerator

accelerator = get_tool_selector_accelerator()

# Process multiple queries efficiently
queries = [
    "Read Python files",
    "Execute git commands",
    "Run tests",
]

query_embeddings = [embed(q) for q in queries]

# Batch similarity computation
all_results = []
for query_emb in query_embeddings:
    similarities = accelerator.cosine_similarity_batch(query_emb, tool_embeddings)
    top_k = accelerator.topk_indices(similarities, k=5)
    all_results.append([tool_names[i] for i in top_k])
```text

### Performance Benchmarking

```python
import time
from victor.native.rust import get_tool_selector_accelerator

accelerator = get_tool_selector_accelerator()

# Benchmark cosine similarity
query = [random.random() for _ in range(384)]
tools = [[random.random() for _ in range(384)] for _ in range(100)]

# Warm up
_ = accelerator.cosine_similarity_batch(query, tools)

# Benchmark
start = time.perf_counter()
for _ in range(1000):
    _ = accelerator.cosine_similarity_batch(query, tools)
duration = time.perf_counter() - start

print(f"Backend: {accelerator.backend}")
print(f"Time: {duration:.3f}s for 1000 iterations")
print(f"Average: {duration/1000*1000:.2f}ms per iteration")
```

## Error Handling

The accelerator provides graceful fallback to Python implementations:

```python
from victor.native.rust import ToolSelectorAccelerator

accelerator = ToolSelectorAccelerator()

try:
    # Try Rust implementation
    similarities = accelerator.cosine_similarity_batch(query, tools)
except Exception as e:
    # Automatically falls back to Python if Rust fails
    print(f"Rust failed, using Python: {e}")
    similarities = accelerator._python_cosine_similarity_batch(query, tools)
```text

## Testing

Run the integration tests:

```bash
# Run all tests
pytest tests/integration/native/test_tool_selector_accelerator.py -v

# Run specific test
pytest
  tests/integration/native/test_tool_selector_accelerator.py::TestToolSelectorAccelerator::test_cosine_similarity_batch
  -v

# Run with coverage
pytest tests/integration/native/test_tool_selector_accelerator.py --cov=victor.native.rust.tool_selector
  --cov-report=html
```

### Example Test

```python
from victor.native.rust import get_tool_selector_accelerator

def test_tool_selection():
    accelerator = get_tool_selector_accelerator()

    # Setup
    query = [0.1] * 384
    tools = [[random.random() for _ in range(384)] for _ in range(100)]
    names = [f"tool_{i}" for i in range(100)]
    categories = {"file_ops", "git"}
    category_map = {name: "file_ops" if i % 2 == 0 else "git" for i, name in enumerate(names)}

    # Test
    results = accelerator.filter_and_rank(query, tools, names, categories, category_map, k=10)

    # Verify
    assert len(results) == 10
    assert all(isinstance(pair, tuple) for pair in results)
    assert all(category_map[name] in categories for name, _ in results)
```text

## Performance Tips

1. **Use `filter_and_rank` for combined operations** - 20-30% faster than separate calls
2. **Pre-compute tool embeddings** - Cache embeddings for repeated queries
3. **Batch queries when possible** - Reduces overhead
4. **Filter early** - Use category filtering before similarity computation
5. **Monitor metrics** - Use `NativeMetrics` to track performance

## Troubleshooting

### Rust Not Available

If you see "Using Python fallback for tool selection":

```bash
# Install with native extension
pip install victor-ai[native]

# Or build from source
pip install victor-ai[native] --no-binary :all:
```

### Performance Degradation

If performance is slower than expected:

1. Check backend: `print(accelerator.backend)`
2. Review metrics: `NativeMetrics.get_instance().get_stats()`
3. Ensure embeddings are pre-computed
4. Use appropriate batch sizes

### Import Errors

If you get import errors:

```python
# Check if victor_native is installed
try:
    import victor_native
    print("Rust extension available")
except ImportError:
    print("Rust extension not available, using Python fallback")
```text

## References

- Rust Implementation: `victor/native/rust/tool_selector.py`
- Integration Tests: `tests/integration/native/test_tool_selector_accelerator.py`
- [Observability Guide](../../observability/README.md)
- [Performance Benchmarks](../../performance/benchmark_results.md)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
