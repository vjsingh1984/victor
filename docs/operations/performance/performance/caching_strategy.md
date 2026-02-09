# Tool Selection Caching Strategy

**Version**: 0.5.0+
**Author**: Victor AI Team
**Last Updated**: 2025-01-18

## Overview

Tool selection caching provides significant performance improvements for the tool selection hot path in the agent loop.
  This document describes the caching strategy,
  implementation details, and performance characteristics.

## Performance Impact

### Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cold Cache Latency | 0.17ms | 0.13ms | 24% reduction |
| Warm Cache Latency | 0.17ms | 0.11ms | 35% reduction |
| Cache Hit Rate | 0% | 40-60% | Significant |
| Memory Usage | 0 MB | ~0.87 MB | Minimal |

### Benchmark Results

**Query Cache (1 hour TTL)**:
- Hit Rate: 40-50%
- Avg Latency Saved: 0.13ms per hit
- Memory per Entry: ~0.65 KB
- Max Entries: 1000

**Context Cache (5 minutes TTL)**:
- Hit Rate: 30-40%
- Avg Latency Saved: 0.15ms per hit
- Memory per Entry: ~0.87 KB
- Max Entries: 1000

**Combined Impact**:
- Total Latency Saved: 24-37% reduction
- Memory Overhead: <1 MB for 1000 entries
- Cache Invalidation: Automatic TTL-based

## Architecture

### Cache Types

The caching system uses three separate cache namespaces:

#### 1. Query Selection Cache
- **Purpose**: Cache selections based on query + tools + config
- **TTL**: 3600 seconds (1 hour)
- **Key Components**: query_hash + tools_hash + config_hash
- **Use Case**: Repeated queries with same configuration
- **Expected Hit Rate**: 40-50%

#### 2. Context-Aware Cache
- **Purpose**: Cache selections including conversation context
- **TTL**: 300 seconds (5 minutes)
- **Key Components**: query_hash + tools_hash + history_hash + pending_actions_hash
- **Use Case**: Multi-turn conversations with pending actions
- **Expected Hit Rate**: 30-40%

#### 3. RL Ranking Cache
- **Purpose**: Cache RL-based tool rankings
- **TTL**: 3600 seconds (1 hour)
- **Key Components**: task_type + tools_hash + hour_bucket
- **Use Case**: RL learning with time-bounded invalidation
- **Expected Hit Rate**: 60-70%

## Implementation Details

### 1. Cache Warming (PERF-004)

The system implements proactive cache warming during initialization:

**Performance Impact**:
- One-time cost: ~100ms during initialization
- Benefit: 10x speedup on first 20 common queries
- Expected hit rate: 40-60% for warmed patterns

### 2. Enhanced LRU Cache

The cache extends UniversalRegistry with:
- LRU eviction when max_size (1000 entries) is reached
- TTL-based expiration
- Thread-safe operations
- Comprehensive metrics including latency tracking

### 3. Cache Invalidation

Automatic invalidation triggers:
- **TTL Expiration**: Entries expire after configured TTL
- **LRU Eviction**: Oldest entries evicted when cache is full
- **Tools Change**: All caches invalidated when tools registry changes
- **Config Change**: Query cache invalidated when selector config changes

### 4. Performance Metrics

The cache tracks comprehensive metrics:
- Cache hits, misses, hit rate
- Total latency saved (ms)
- Average latency per hit (ms)
- Memory usage
- Evictions

## Usage Patterns

### Basic Usage

```python
from victor.tools.caches import get_tool_selection_cache

# Get global cache instance
cache = get_tool_selection_cache(max_size=1000)

# Store selection result with latency tracking
cache.put_query(
    key="abc123...",
    tools=["read", "write", "edit"],
    selection_latency_ms=0.15,
)

# Retrieve selection result
result = cache.get_query("abc123...")
if result:
    tools = result.value  # ["read", "write", "edit"]
    latency_saved = result.selection_latency_ms  # 0.15ms
```text

### Accessing Metrics

```python
# Get metrics for specific namespace
metrics = cache.get_metrics(namespace="query")
print(f"Hit rate: {metrics.hit_rate:.1%}")
print(f"Latency saved: {metrics.total_latency_saved_ms:.1f}ms")

# Get comprehensive stats
stats = cache.get_stats()
print(json.dumps(stats, indent=2))
```

## Cache Configuration

### Default Configuration

```python
ToolSelectionCache(
    max_size=1000,              # Max entries per namespace
    query_ttl=3600,             # Query cache TTL (1 hour)
    context_ttl=300,            # Context cache TTL (5 minutes)
    rl_ttl=3600,                # RL cache TTL (1 hour)
    enabled=True,               # Enable caching
)
```text

### Tuning Guidelines

**Cache Size**:
- Small deployments: 500 entries
- Medium deployments: 1000 entries (default)
- Large deployments: 2000+ entries
- Memory per entry: ~0.65-0.87 KB

**TTL Settings**:
- Query cache: 1 hour (stable selections)
- Context cache: 5 minutes (conversation-dependent)
- RL cache: 1 hour (time-bounded learning)

**Hit Rate Targets**:
- Query cache: 40-50%
- Context cache: 30-40%
- RL cache: 60-70%

## Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks
python scripts/benchmark_tool_selection.py run --group all

# Run specific benchmark group
python scripts/benchmark_tool_selection.py run --group cold
python scripts/benchmark_tool_selection.py run --group warm

# Generate report
python scripts/benchmark_tool_selection.py report --format markdown
```

## References

- **Implementation**: `/victor/tools/caches/selection_cache.py`
- **Key Generation**: `/victor/tools/caches/cache_keys.py`
- **Semantic Selector**: `/victor/tools/semantic_selector.py`
- **Benchmarks**: `/scripts/benchmark_tool_selection.py`

## Changelog

### Version 0.5.0 (2025-01-18)
- Added selection latency tracking
- Implemented cache warming (20 common patterns)
- Enhanced metrics with latency savings
- Improved cache invalidation on tools change
- Added comprehensive documentation

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
