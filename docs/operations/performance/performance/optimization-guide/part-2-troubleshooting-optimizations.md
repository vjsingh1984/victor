# Performance Optimization Guide - Part 2

**Part 2 of 3:** Troubleshooting and Optimization Strategies

---

## Navigation

- [Part 1: Improvements & Benchmarking](part-1-improvements-benchmarking.md)
- **[Part 2: Troubleshooting & Optimizations](#)** (Current)
- [Part 3: Monitoring & Guidelines](part-3-monitoring-guidelines.md)
- [**Complete Guide**](../optimization_guide.md)

---

### Running Full Benchmark Suite

```bash
# Run all performance benchmarks
pytest tests/benchmark/test_performance_optimizations.py -v -s -m benchmark

# Run specific benchmark
pytest tests/benchmark/test_performance_optimizations.py::test_batch_tool_execution_performance -v -s

# Run with coverage
pytest tests/benchmark/test_performance_optimizations.py --cov=victor.agent -v
```

### Expected Results

| Benchmark | Metric | Target |
|-----------|--------|--------|
| Batch Tool Execution | Improvement vs sequential | 40%+ |
| LLM Compaction | Time overhead vs truncation | < 2x |
| Summary Cache Hit | Speedup vs cache miss | 10x+ |
| Prompt Building Cache | Improvement vs uncached | 90%+ |
| Overall System | Combined improvement | 20%+ |

### Custom Benchmarks

Create custom benchmarks for your use case:

```python
import asyncio
import time
from victor.agent.tool_executor import ToolExecutor

async def benchmark_my_workload():
    executor = ToolExecutor(tool_registry=registry)

    tool_calls = [
        ("my_tool", {"arg": value})
        for value in range(100)
    ]

    # Benchmark sequential
    start = time.perf_counter()
    for tool_name, args in tool_calls:
        await executor.execute(tool_name, args)
    sequential_time = time.perf_counter() - start

    # Benchmark batch
    start = time.perf_counter()
    await executor.execute_batch(tool_calls, max_concurrency=10)
    batch_time = time.perf_counter() - start

    improvement = ((sequential_time - batch_time) / sequential_time) * 100
    print(f"Improvement: {improvement:.1f}%")

asyncio.run(benchmark_my_workload())
```

---

## Troubleshooting

### Batch Execution Not Showing Improvement

**Problem:** Batch execution is not faster than sequential.

**Possible causes:**
1. Tools are not I/O-bound (CPU-bound tools don't benefit from parallelization)
2. Concurrency limit is too low
3. Dependencies prevent true parallelization

**Solutions:**
```python
# Increase concurrency for I/O-bound tools
await executor.execute_batch(tool_calls, max_concurrency=10)

# Check if tools are truly independent
tool_calls = [
    ("read_file", {"path": "a.txt"}),  # Independent
    ("read_file", {"path": "b.txt"}),  # Independent
    # Avoid: write then read same file (sequential dependency)
]
```

### LLM Compaction Too Slow

**Problem:** LLM-based compaction takes too long.

**Solutions:**
```python
# Use faster model
strategy = LLMCompactionStrategy(
    summarization_model="gpt-4o-mini",  # Fast, not gpt-4
)

# Enable caching
strategy = LLMCompactionStrategy(
    cache_summaries=True,  # Critical for performance
)

# Fall back to truncation for small contexts
strategy = HybridCompactionStrategy(
    small_context_threshold=5000,  # Use truncation for < 5K tokens
)
```

### Low Prompt Cache Hit Rate

**Problem:** Prompt cache hit rate is below 70%.

**Possible causes:**
1. Context objects are not stable (different keys for same logical context)
2. TTL is too short
3. Cache is being invalidated too frequently

**Solutions:**
```python
# Use stable context keys
context = PromptContext({
    "task": "code_review",  # Stable
    "language": "python",   # Stable
    # Avoid: timestamp, random IDs
})

# Increase or remove TTL
coordinator = PromptCoordinator(
    contributors=[...],
    enable_cache=True,
    cache_ttl=None,  # No TTL
)

# Monitor invalidations
stats = coordinator.get_cache_stats()
print(f"Invalidations: {stats['cache_invalidations']}")
```

---

## Performance Tuning Guidelines

### Memory vs Speed Trade-offs

| Configuration | Memory Usage | Speed | Quality |
|---------------|--------------|-------|---------|
| Truncation + No Cache | Low | Fast | Low |
| Truncation + Cache | Medium | Very Fast | Low |
| LLM + No Cache | Medium | Slow | High |
| LLM + Cache | High | Fast | High |
| Hybrid + Cache | High | Adaptive | Adaptive |

### When to Use Each Configuration

**Low Memory Systems (< 4GB RAM):**
```python
# Minimal memory footprint
strategy = TruncationCompactionStrategy()
coordinator = PromptCoordinator(enable_cache=False, cache_ttl=None)
```

**Balanced Systems (4-16GB RAM):**
```python
# Good balance
strategy = HybridCompactionStrategy()
coordinator = PromptCoordinator(enable_cache=True, cache_ttl=1800.0)
```

**High-Performance Systems (> 16GB RAM):**
```python
# Maximum performance
strategy = LLMCompactionStrategy(cache_summaries=True)
coordinator = PromptCoordinator(enable_cache=True, cache_ttl=None)
```

---

## Monitoring and Metrics

### Key Metrics to Track

1. **Batch Execution:**
   - Average batch size
   - Concurrency utilization
   - Speedup vs sequential

2. **Context Compaction:**
   - Compaction frequency
   - Token reduction ratio
   - Cache hit rate (LLM summaries)

3. **Prompt Building:**
   - Cache hit rate
   - Average build time
   - Cache invalidation rate

### Logging Configuration

```python
import logging

# Enable performance logging
logging.getLogger("victor.agent.tool_executor").setLevel(logging.INFO)
logging.getLogger("victor.agent.coordinators.compaction_strategies").setLevel(logging.INFO)
logging.getLogger("victor.agent.coordinators.prompt_coordinator").setLevel(logging.INFO)
```

---

## Future Optimizations

Roadmap for future performance improvements:

1. **Smart batching**: Automatically detect batchable tool calls
2. **Predictive caching**: Pre-cache likely prompts
3. **Incremental summarization**: Update summaries instead of regenerating
4. **Tool result streaming**: Stream large tool results during execution
5. **Parallel LLM calls**: Batch LLM summarization requests

---

## References

- **Batch Tool Execution**: `/victor/agent/tool_executor.py` - `ToolExecutor.execute_batch()`
- **Context Compaction**: `/victor/agent/coordinators/compaction_strategies.py`
- **Prompt Building**: `/victor/agent/coordinators/prompt_coordinator.py`
- **Benchmarks**: `/tests/benchmark/test_performance_optimizations.py`
- **Strategic Plan**: `/docs/parallel_work_streams_plan.md` - Work Stream 3.3

---

## Summary

The performance optimizations implemented in Work Stream 3.3 achieve **20% overall system improvement** through:

1. **Batch Tool Execution** - 40-75% faster for independent tools
2. **Optimized Context Compaction** - Better quality with acceptable overhead
3. **Cached Prompt Building** - 90%+ faster on cache hit

These optimizations are production-ready, well-tested, and configurable for different use cases. Use the configuration examples and benchmarks to tune the system for your specific workload.

**Next Steps:**
1. Run the benchmark suite to verify improvements
2. Configure optimizations based on your workload
3. Monitor cache hit rates and performance metrics
4. Tune parameters for optimal performance/quality trade-off

---

# ADVANCED OPTIMIZATION DETAILS

## 1. Response Caching

### Overview

The response caching system stores LLM responses and retrieves them based on:
- **Exact match** - Via content hashing (SHA256)
- **Semantic similarity** - Via embedding similarity (cosine similarity)

### Features

- Thread-safe operations
- LRU eviction when size limit reached
- TTL-based expiration (configurable)
- Optional persistence to disk
- Performance metrics tracking

### Configuration

```yaml
# In ~/.victor/profiles.yaml or via environment variables
response_cache_enabled: true
response_cache_max_size: 1000
response_cache_ttl: 3600  # 1 hour
response_cache_semantic_enabled: true
response_cache_semantic_threshold: 0.85  # 85% similarity
response_cache_persist_path: ~/.victor/cache/response_cache.json
```

### Usage

```python
from victor.core.cache import get_response_cache
from victor.providers.base import Message, CompletionResponse

# Get global cache instance
cache = get_response_cache()

# Cache a response
messages = [Message(role="user", content="What is 2+2?")]
response = CompletionResponse(content="2+2 equals 4.")
await cache.put(messages, response)

# Retrieve with exact match
cached_response = await cache.get(messages)

# Retrieve with semantic similarity
similar_response = await cache.get_similar(
    messages,
    threshold=0.85  # Minimum similarity
)

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Performance Characteristics

**Cache Hit:**
- Time: ~0.05ms (vs 500-2000ms for API call)
- Speedup: 10,000-40,000x

**Cache Miss:**
- Time: ~0.1ms (hash lookup)
- Overhead: Negligible

## 2. Request Batching
