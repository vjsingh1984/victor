# Tool Selection Caching Strategy

**Track**: Performance Optimization - Tool Selection Caching
**Priority**: MEDIUM (improves hot path performance)
**Status**: Implemented

## Overview

Tool selection is a hot path that's called frequently during agent execution. The caching system provides **24-37%
  latency reduction** through intelligent LRU caching with cache warming and invalidation strategies.

## Performance Improvements

### Benchmark Results

| Benchmark | Latency (ms) | Speedup | Hit Rate |
|-----------|-------------|---------|----------|
| Cold Cache (0% hits) | 0.17 | 1.0x | 0% |
| Warm Cache (100% hits) | 0.13 | 1.32x | 100% |
| Context-Aware Cache | 0.11 | 1.59x | 100% |
| RL Ranking Cache | 0.11 | 1.56x | 100% |

### Memory Usage

- Per entry: ~0.65 KB
- 1000 entries: ~0.87 MB
- Recommended cache size: 500-1000 entries

### Target Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Cache Hit Rate | > 40% | 40-60% |
| Latency Reduction | > 30% | 24-37% |
| Cache Size | 500-1000 entries | 1000 entries |
| Query TTL | 5 minutes | 5 minutes |
| Warm-up Time | < 10 seconds | ~2-3 seconds |

## Architecture

### Cache Types

The tool selection caching system uses **three separate caches** for different use cases:

#### 1. Query Cache (Primary)
- **Purpose**: Cache for simple query-based tool selection
- **Cache Key**: `query_hash + tools_hash + config_hash`
- **TTL**: 5 minutes
- **Max Size**: 1000 entries
- **Expected Hit Rate**: 40-60%

#### 2. Context Cache (Multi-turn Conversations)
- **Purpose**: Cache for context-aware selection with conversation history
- **Cache Key**: `query_hash + tools_hash + recent_conversation_context`
- **TTL**: 5 minutes
- **Max Size**: 500 entries
- **Expected Hit Rate**: 30-40%

#### 3. RL Ranking Cache (Reinforcement Learning)
- **Purpose**: Cache for learned tool rankings
- **Cache Key**: `task_type + tools_hash + hour_bucket + query`
- **TTL**: 1 hour
- **Max Size**: 100 entries
- **Expected Hit Rate**: 60-70%

### LRU Eviction Policy

All caches use **LRU (Least Recently Used)** eviction policy with O(1) operations:

```python
from victor.tools.caches import LRUCache

# Create LRU cache
cache = LRUCache(max_size=1000, ttl_seconds=300, name="query")

# Get entry (moves to end = recently used)
entry = cache.get("cache_key")

# Put entry (evicts oldest if full)
cache.put("cache_key", cache_entry)

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```text

### Cache Key Generation

Cache keys are generated using `CacheKeyGenerator`:

```python
from victor.tools.caches import get_cache_key_generator

key_gen = get_cache_key_generator()

# Calculate tools hash (for invalidation)
tools_hash = key_gen.calculate_tools_hash(tool_registry)

# Generate query cache key
query_key = key_gen.generate_query_key(
    query="read the file",
    tools_hash=tools_hash,
    config_hash="threshold:0.18:model:all-MiniLM-L6-v2"
)

# Generate context cache key
context_key = key_gen.generate_context_key(
    query="read the file",
    tools_hash=tools_hash,
    conversation_history=history
)
```

## Usage

### Basic Usage

```python
from victor.tools.caches import get_tool_selection_cache

# Get global cache instance
cache = get_tool_selection_cache()

# Query cache
result = cache.get_query("cache_key")
if result is None:
    # Cache miss - perform selection
    tools = await select_tools(query)
    # Store in cache
    cache.put_query("cache_key", tool_names, tools=tools)
```text

### Integration with SemanticToolSelector

The `SemanticToolSelector` automatically uses the cache:

```python
from victor.tools.semantic_selector import SemanticToolSelector

selector = SemanticToolSelector()
await selector.initialize_tool_embeddings(tool_registry)

# Cache is automatically checked in select_relevant_tools_with_context
tools = await selector.select_relevant_tools_with_context(
    user_message="read the file",
    tools=tool_registry,
    conversation_history=None,
)
```

### Cache Invalidation

```python
from victor.tools.caches import invalidate_tool_selection_cache

# Invalidate all caches
invalidate_tool_selection_cache()

# Invalidate specific cache type
invalidate_tool_selection_cache(cache_type="query")

# Invalidate specific key
invalidate_tool_selection_cache(cache_type="query", key="specific_key")
```text

### Performance Metrics

```python
from victor.tools.caches import get_tool_selection_cache

cache = get_tool_selection_cache()
stats = cache.get_stats()

print(f"Query cache hit rate: {stats['query']['hit_rate']:.1%}")
print(f"Context cache hit rate: {stats['context']['hit_rate']:.1%}")
print(f"RL cache hit rate: {stats['rl']['hit_rate']:.1%}")
```

## Cache Warming

### Automatic Warmup

The cache is automatically warmed on startup with common query patterns:

```python
from victor.tools.caches import CacheWarmer

warmer = CacheWarmer(cache)
await warmer.warmup_common_queries(tool_registry, semantic_selector)
```text

### Common Query Patterns

The system pre-warms the cache with 24 common queries:
- File operations: "read the file", "write to file", "list directory"
- Code search: "search code", "find classes", "find functions"
- Analysis: "analyze codebase", "refactor", "debug"
- Git: "git commit", "show diff"
- Testing: "run tests", "test changes"
- Editing: "edit files", "modify code"

### Warmup Performance

- **Warmup Time**: 2-3 seconds for 24 queries
- **Queries/Second**: ~8-12 QPS during warmup
- **Memory Impact**: ~15 KB for 24 cached entries

## Cache Invalidation Strategy

### Time-Based (TTL)

All cache entries automatically expire after their TTL:
- Query cache: 5 minutes
- Context cache: 5 minutes
- RL cache: 1 hour

### Config-Based Invalidation

When tool registry changes:
```python
# Notify semantic selector of tools change
semantic_selector.notify_tools_changed()

# This invalidates all caches automatically
```

### Manual Invalidation

```python
# Invalidate all caches
from victor.tools.caches import invalidate_tool_selection_cache
invalidate_tool_selection_cache()
```text

### Background Cleanup

Expired entries are automatically cleaned up every 60 seconds by a background thread:
```python
# Automatic cleanup (runs in background)
cache._cleanup_expired()
```

## Performance Metrics

### Tracking

The cache tracks comprehensive metrics:

```python
stats = cache.get_stats()

# Query cache metrics
{
    "name": "query",
    "size": 450,  # Current number of entries
    "max_size": 1000,
    "hits": 4500,
    "misses": 3000,
    "hit_rate": 0.60,  # 60% hit rate
    "evictions": 150,
    "expirations": 300,
    "utilization": 0.45,  # 45% full
}
```text

### Latency Tracking

Each cache entry tracks its original selection latency:

```python
entry = CacheEntry(
    tool_names=["read", "search"],
    tools=tool_definitions,
    selection_latency_ms=12.5,  # Original selection time
)
```

### Hit Rate Analysis

Target hit rates by cache type:
- **Query cache**: 40-60% (short TTL, high churn)
- **Context cache**: 30-40% (conversational, less repetition)
- **RL cache**: 60-70% (task-based, stable patterns)

## Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks
python scripts/benchmark_tool_selection.py run --group all

# Run specific benchmark
python scripts/benchmark_tool_selection.py run --group cold

# Generate report
python scripts/benchmark_tool_selection.py report --format markdown

# Compare runs
python scripts/benchmark_tool_selection.py compare run1.json run2.json
```text

### Benchmark Scenarios

1. **Cold Cache** (0% hits): Baseline uncached performance
2. **Warm Cache** (100% hits): Best case cached performance
3. **Mixed Cache** (50% hits): Realistic mixed workload
4. **Context Cache**: Multi-turn conversation performance
5. **RL Cache**: Learned ranking performance

### Expected Results

- **Cold Cache**: ~0.17ms per selection
- **Warm Cache**: ~0.13ms per selection (1.32x speedup)
- **Context Cache**: ~0.11ms per selection (1.59x speedup)
- **Memory Usage**: ~0.87 MB for 1000 entries

## Implementation Details

### Thread Safety

All cache operations are thread-safe using `threading.RLock`:

```python
with self._lock:
    entry = self._cache.get(key)
    # ... thread-safe operations
```

### LRU Implementation

Uses `collections.OrderedDict` for O(1) LRU operations:
- **get()**: Move to end (mark as recently used)
- **put()**: Add to end, evict from front if full

### Expiration Check

Entries are checked for expiration on access:
```python
def is_expired(self) -> bool:
    return time.time() - self.timestamp > self.ttl_seconds
```text

## Best Practices

### DO:
- ✅ Use the cache for all tool selection operations
- ✅ Configure TTL based on your use case
- ✅ Monitor hit rates and adjust cache size
- ✅ Warm up cache on startup with common queries
- ✅ Invalidate cache when tool registry changes

### DON'T:
- ❌ Store too large entries (> 1 KB each)
- ❌ Use very long TTL (> 1 hour for query cache)
- ❌ Ignore cache hit rates
- ❌ Forget to invalidate after tool changes

## Configuration

### Cache Sizes

Adjust cache sizes in `victor/tools/caches.py`:

```python
class CacheConfig:
    QUERY_CACHE_MAX_SIZE = 1000  # Increase for more hits
    QUERY_CACHE_TTL_SECONDS = 300  # Adjust based on data freshness
```

### Disable Caching

To disable caching entirely (not recommended):
```python
# Set cache sizes to 0
CacheConfig.QUERY_CACHE_MAX_SIZE = 0
```text

## Troubleshooting

### Low Hit Rate

If hit rate is < 40%:
1. Check if queries are varying too much
2. Increase cache size
3. Adjust TTL
4. Check tools_hash is stable

### High Memory Usage

If memory usage is high:
1. Reduce cache max size
2. Reduce TTL
3. Check cache entry sizes

### Cache Not Working

If cache appears ineffective:
1. Verify cache is enabled
2. Check logs for cache hits/misses
3. Verify tools_hash is calculated correctly
4. Check config_hash changes

## Future Improvements

### Planned Enhancements

1. **Adaptive TTL**: Adjust TTL based on access patterns
2. **Compression**: Compress cache entries to reduce memory
3. **Persistent Cache**: Save cache to disk for faster startup
4. **Machine Learning**: Use ML to predict cache misses
5. **Distributed Cache**: Redis support for multi-process scenarios

### Performance Targets

- **Hit Rate**: > 50% (from 40%)
- **Latency**: > 40% reduction (from 30%)
- **Memory**: < 0.5 MB for 1000 entries (from 0.87 MB)

## References

- **Implementation**: `victor/tools/caches.py`
- **Semantic Selector**: `victor/tools/semantic_selector.py`
- **Benchmark Tests**: `tests/benchmarks/test_tool_selection_benchmark.py`
- **Benchmark Script**: `scripts/benchmark_tool_selection.py`

## Changelog

### Version 0.5.0 (2025-01-18)

- ✅ Implemented LRU cache with O(1) operations
- ✅ Added query, context, and RL caches
- ✅ Implemented cache warming for 24 common queries
- ✅ Added comprehensive metrics tracking
- ✅ Implemented background cleanup thread
- ✅ Added thread-safe operations
- ✅ Achieved 24-37% latency reduction
- ✅ Achieved 40-60% hit rate target

### Version 0.4.0 (2025-01-10)

- ✅ Added query embedding cache in SemanticToolSelector
- ✅ Added category memberships cache
- ✅ Implemented batch embedding generation
- ✅ Added cache warmup on startup

---

**Last Updated**: 2025-01-18
**Maintainer**: @vijay-singh
**Status**: ✅ Implemented and Production Ready

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 5 minutes
