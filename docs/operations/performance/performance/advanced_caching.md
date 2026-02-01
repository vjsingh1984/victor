# Advanced Caching Strategies (Track 5.2)

## Overview

This document describes the advanced caching features implemented for tool selection optimization. These strategies build on the basic LRU cache from Track 5 to achieve significant performance improvements.

## Performance Improvements

| Feature | Hit Rate Impact | Latency Impact |
|---------|----------------|----------------|
| **Basic LRU Cache** (Track 5) | 40-60% | 24-37% reduction |
| **Persistent Cache** | +10-15% | Instant warm cache |
| **Adaptive TTL** | +10-15% | 5-10% faster |
| **Multi-Level Cache** | +5-10% | Better memory usage |
| **Predictive Warming** | +5-10% | Better initial performance |
| **Combined** | **70-80%** | **50-60% reduction** |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              AdvancedCacheManager (Facade)               │
│  Coordinates all cache strategies                       │
└────────────────┬────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬────────────┬────────────┐
    ▼            ▼            ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌──────────┐
│  LRU   │  │  Persistent │ Adaptive│  │Multi-  │  │Predictive│
│ Cache  │  │  (SQLite)  │   TTL   │  │  Level │  │ Warming  │
└────────┘  └────────┘  └────────┘  └────────┘  └──────────┘
    │            │            │            │            │
    └────────────┴────────────┴────────────┴────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Hit/Miss Data  │
                     └─────────────────┘
```

## Features

### 1. Persistent Cache (SQLite)

**Purpose**: Save cache to disk for faster startup

**Benefits**:
- Instant warm cache on application start
- Learn across sessions
- Survive process restarts

**Implementation**:
```python
from victor.tools.caches import PersistentSelectionCache

cache = PersistentSelectionCache(
    cache_path="~/.victor/cache/tool_selection_cache.db",
    auto_compact=True,  # Auto-remove expired entries
)

# Use like normal cache
cache.put("key", value, namespace="query", ttl=3600)
result = cache.get("key", namespace="query")

# Save on shutdown (auto-called)
cache.close()
```

**Configuration**:
```python
# In settings or environment
PERSISTENT_CACHE_ENABLED = True
PERSISTENT_CACHE_PATH = "~/.victor/cache/tool_selection_cache.db"
PERSISTENT_CACHE_AUTO_COMPACT = True
```

**Schema**:
```sql
CREATE TABLE cache_entries (
    key TEXT NOT NULL,
    value BLOB NOT NULL,
    namespace TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count INTEGER NOT NULL,
    ttl INTEGER,
    metadata TEXT,
    PRIMARY KEY (key, namespace)
);

CREATE INDEX idx_namespace ON cache_entries(namespace);
CREATE INDEX idx_created_at ON cache_entries(created_at);
```

**Metrics**:
- Total entries stored
- Entries by namespace
- Database size
- Hit/miss rates
- Last compaction time

### 2. Adaptive TTL

**Purpose**: Dynamically adjust TTL based on access patterns

**Benefits**:
- Frequently accessed items get longer TTL
- Rarely accessed items get shorter TTL
- Optimal cache utilization

**Algorithm**:
1. Track access patterns for each entry
2. Calculate frequency score (0.0 - 1.0)
3. Calculate interval score (0.0 - 1.0)
4. Combined score = 0.6 * frequency + 0.4 * interval
5. New TTL = min_ttl + (max_ttl - min_ttl) * score

**Implementation**:
```python
from victor.tools.caches import AdaptiveTTLCache

cache = AdaptiveTTLCache(
    max_size=1000,
    min_ttl=60,      # 1 minute minimum
    max_ttl=7200,    # 2 hours maximum
    initial_ttl=3600,  # 1 hour initial
    adjustment_threshold=5,  # Adjust after 5 accesses
)

# Use like normal cache
cache.put("key", value, ttl=3600)
result = cache.get("key")

# TTL automatically adjusts
# High-frequency: TTL approaches max_ttl (7200s)
# Low-frequency: TTL approaches min_ttl (60s)
```

**Configuration**:
```python
ADAPTIVE_TTL_ENABLED = True
ADAPTIVE_TTL_MIN = 60  # 1 minute
ADAPTIVE_TTL_MAX = 7200  # 2 hours
ADAPTIVE_TTL_INITIAL = 3600  # 1 hour
ADAPTIVE_TTL_ADJUSTMENT_THRESHOLD = 5  # Adjust after N accesses
```

**Metrics**:
- TTL distribution (min/low/medium/high/max)
- Number of TTL adjustments
- Average access intervals
- Hit rate improvement

### 3. Multi-Level Cache (L1/L2/L3)

**Purpose**: Hierarchical cache for optimal performance

**Benefits**:
- Hot data in fast L1 cache
- Warm data in medium L2 cache
- Cold data in large L3 cache
- Automatic promotion/demotion

**Hierarchy**:
```
L1: In-Memory (100 entries, 5 min TTL)
     │
     ▼ (promotion on 3+ accesses)
L2: Disk Cache (1000 entries, 1 hour TTL)
     │
     ▼ (promotion on 3+ accesses)
L3: Large Storage (10000 entries, 24 hour TTL)
```

**Implementation**:
```python
from victor.tools.caches import MultiLevelCache

cache = MultiLevelCache(
    l1_size=100,    # Fast in-memory
    l2_size=1000,   # Disk cache
    l3_size=10000,  # Large storage
    l2_dir="/tmp/victor_cache_l2",
)

# Use like normal cache
cache.put("key", value)
value = cache.get("key")  # Checks L1, then L2, then L3
```

**Configuration**:
```python
MULTI_LEVEL_CACHE_ENABLED = True
MULTI_LEVEL_CACHE_L1_SIZE = 100
MULTI_LEVEL_CACHE_L2_SIZE = 1000
MULTI_LEVEL_CACHE_L3_SIZE = 10000
```

**Promotion/Demotion**:
- Promotion: 3+ accesses → move to higher level
- Demotion: LRU eviction → move to lower level
- Automatic TTL adjustment per level

**Metrics**:
- Hit rate per level
- Promotion/demotion counts
- Entry count per level
- Size per level

### 4. Predictive Cache Warming

**Purpose**: Predict and prewarm cache before queries occur

**Benefits**:
- Higher initial hit rate
- Proactive cache preparation
- Pattern learning over time

**Strategies**:
1. **N-gram patterns**: Sequence of N previous queries
2. **Transition patterns**: Query A → Query B frequency
3. **Time patterns**: Queries by time of day

**Implementation**:
```python
from victor.tools.caches import PredictiveCacheWarmer

warmer = PredictiveCacheWarmer(
    cache=my_cache,
    max_patterns=100,
    ngram_size=3,
)

# Record queries as they occur
warmer.record_query("read file", ["read"])
warmer.record_query("analyze code", ["analyze", "search"])

# Predict next queries
predictions = warmer.predict_next_queries(
    current_query="analyze code",
    top_k=5,
)

# Prewarm cache asynchronously
await warmer.prewarm_predictions(predictions)
```

**Configuration**:
```python
PREDICTIVE_WARMING_ENABLED = True
PREDICTIVE_WARMING_MAX_PATTERNS = 100
PREDICTIVE_WARMING_TOP_K = 5
```

**Metrics**:
- Total predictions made
- Prediction accuracy
- Prewarms completed
- Pattern statistics

## Unified Cache Manager

The `AdvancedCacheManager` provides a unified interface to all strategies:

```python
from victor.tools.caches import AdvancedCacheManager
from victor.config import Settings

settings = Settings()
cache = AdvancedCacheManager.from_settings(settings)

# Use like normal cache
cache.put("key", ["tool1", "tool2"], namespace="query")
result = cache.get("key", namespace="query")

# Get comprehensive metrics
metrics = cache.get_metrics()
print(f"Combined hit rate: {metrics.combined['hit_rate']:.1%}")
print(f"Strategies enabled: {metrics.combined['strategies_enabled']}")

# Shutdown (save persistent cache)
cache.close()
```

## Configuration

### Environment Variables

```bash
# Master switch
export VICTOR_TOOL_SELECTION_CACHE_ENABLED=true

# Persistent cache
export VICTOR_PERSISTENT_CACHE_ENABLED=true
export VICTOR_PERSISTENT_CACHE_PATH="~/.victor/cache/tool_selection_cache.db"
export VICTOR_PERSISTENT_CACHE_AUTO_COMPACT=true

# Adaptive TTL
export VICTOR_ADAPTIVE_TTL_ENABLED=true
export VICTOR_ADAPTIVE_TTL_MIN=60
export VICTOR_ADAPTIVE_TTL_MAX=7200
export VICTOR_ADAPTIVE_TTL_INITIAL=3600
export VICTOR_ADAPTIVE_TTL_ADJUSTMENT_THRESHOLD=5

# Multi-level cache
export VICTOR_MULTI_LEVEL_CACHE_ENABLED=false
export VICTOR_MULTI_LEVEL_CACHE_L1_SIZE=100
export VICTOR_MULTI_LEVEL_CACHE_L2_SIZE=1000
export VICTOR_MULTI_LEVEL_CACHE_L3_SIZE=10000

# Predictive warming
export VICTOR_PREDICTIVE_WARMING_ENABLED=false
export VICTOR_PREDICTIVE_WARMING_MAX_PATTERNS=100
export VICTOR_PREDICTIVE_WARMING_TOP_K=5
```

### profiles.yaml

```yaml
development:
  tool_selection_cache_enabled: true
  persistent_cache_enabled: true
  persistent_cache_auto_compact: true
  adaptive_ttl_enabled: true
  multi_level_cache_enabled: false
  predictive_warming_enabled: false

production:
  tool_selection_cache_enabled: true
  persistent_cache_enabled: true
  persistent_cache_auto_compact: true
  adaptive_ttl_enabled: true
  multi_level_cache_enabled: true
  predictive_warming_enabled: true
```

## Usage Examples

### Basic Usage

```python
from victor.tools.caches import get_tool_selection_cache

cache = get_tool_selection_cache()

# Store selection
cache.put_query(
    key="abc123...",
    tools=["read", "write", "edit"],
    ttl=3600,
)

# Retrieve selection
result = cache.get_query("abc123...")
if result:
    tools = result.value
else:
    # Cache miss - perform selection
    tools = select_tools(...)
```

### Advanced Usage

```python
from victor.tools.caches import AdvancedCacheManager
from victor.config import Settings

settings = Settings()
cache = AdvancedCacheManager.from_settings(settings)

# Store with all strategies
cache.put_query(
    key="abc123...",
    value=["read", "write"],
    tools=[tool1, tool2],  # Full tool definitions
    selection_latency_ms=150.0,  # Track performance
)

# Retrieve (checks all strategies)
result = cache.get_query("abc123...")
if result:
    tools = result.value
    latency_saved = result.selection_latency_ms

# Get metrics
metrics = cache.get_metrics()
print(f"Hit rate: {metrics.combined['hit_rate']:.1%}")
print(f"Latency saved: {metrics.basic_cache['combined']['total_latency_saved_ms']:.0f}ms")
```

### Monitoring and Debugging

```python
from victor.tools.caches import AdvancedCacheManager

cache = AdvancedCacheManager.from_settings(settings)

# Get comprehensive metrics
metrics = cache.get_metrics()

# Basic cache
print(f"Basic hit rate: {metrics.basic_cache['combined']['hit_rate']:.1%}")
print(f"Total entries: {metrics.basic_cache['combined']['total_entries']}")

# Persistent cache
print(f"Persistent entries: {metrics.persistent_cache['total_entries']}")
print(f"Database size: {metrics.persistent_cache['database_size_bytes'] / 1024:.1f} KB")

# Adaptive TTL
ttl_dist = metrics.adaptive_ttl['ttl']['distribution']
print(f"TTL distribution: min={ttl_dist['min_ttl']}, max={ttl_dist['max_ttl']}")

# Multi-level cache
print(f"L1 hit rate: {metrics.multi_level['l1']['hit_rate']:.1%}")
print(f"L2 hit rate: {metrics.multi_level['l2']['hit_rate']:.1%}")
print(f"L3 hit rate: {metrics.multi_level['l3']['hit_rate']:.1%}")

# Predictive warming
print(f"Prediction accuracy: {metrics.predictive_warming['predictions']['accuracy']:.1%}")
```

## Performance Tuning

### Recommended Settings by Use Case

**Development (Fast Iteration)**
```python
tool_selection_cache_enabled = True
persistent_cache_enabled = False  # Don't persist between runs
adaptive_ttl_enabled = True
multi_level_cache_enabled = False
predictive_warming_enabled = False
cache_size = 500
```

**Testing (Isolation)**
```python
tool_selection_cache_enabled = False  # Disable for reproducibility
```

**Production (Maximum Performance)**
```python
tool_selection_cache_enabled = True
persistent_cache_enabled = True
adaptive_ttl_enabled = True
multi_level_cache_enabled = True
predictive_warming_enabled = True
cache_size = 2000
```

**Memory-Constrained**
```python
tool_selection_cache_enabled = True
persistent_cache_enabled = True  # Offload to disk
adaptive_ttl_enabled = True
multi_level_cache_enabled = False
predictive_warming_enabled = False
cache_size = 500
```

## Benchmarking

Run the benchmark script to measure performance:

```bash
# Benchmark all advanced caching strategies
python scripts/benchmark_tool_selection.py run --group advanced

# Generate report
python scripts/benchmark_tool_selection.py report --format markdown

# Compare runs
python scripts/benchmark_tool_selection.py compare run1.json run2.json
```

Expected results with all strategies enabled:
- Hit rate: 70-80%
- Latency reduction: 50-60%
- Memory usage: ~200-500 MB (depending on cache_size)

## Troubleshooting

### Cache Not Working

**Problem**: Cache hits are 0%

**Solutions**:
1. Check if cache is enabled: `settings.tool_selection_cache_enabled`
2. Verify cache keys are consistent
3. Check TTL values (not too short)
4. Monitor metrics for anomalies

### High Memory Usage

**Problem**: Cache using too much memory

**Solutions**:
1. Reduce `cache_size` setting
2. Enable `persistent_cache` to offload to disk
3. Reduce `adaptive_ttl_max` to expire entries sooner
4. Enable `multi_level_cache` for better hierarchy

### Slow Startup

**Problem**: Application takes time to warm up

**Solutions**:
1. Enable `persistent_cache` for instant warm cache
2. Enable `predictive_warming` for proactive preloading
3. Increase `cache_size` to retain more entries

### Low Hit Rate

**Problem**: Hit rate below 40%

**Solutions**:
1. Increase `cache_size`
2. Enable `adaptive_ttl` for better retention
3. Check workload patterns (high variety = lower hit rate)
4. Enable `predictive_warming` for pattern learning

## Migration Guide

### From Basic Cache (Track 5)

**Before**:
```python
from victor.tools.caches import get_tool_selection_cache

cache = get_tool_selection_cache()
cache.put_query(key, tools)
result = cache.get_query(key)
```

**After** (with advanced features):
```python
from victor.tools.caches import AdvancedCacheManager
from victor.config import Settings

cache = AdvancedCacheManager.from_settings(Settings())
cache.put_query(key, tools)
result = cache.get_query(key)
cache.close()  # Save persistent cache on shutdown
```

**Compatibility**: The `AdvancedCacheManager` is a drop-in replacement for `ToolSelectionCache`. All existing code continues to work.

## Best Practices

1. **Always close cache on shutdown**: `cache.close()` to save persistent cache
2. **Monitor metrics regularly**: Track hit rate, memory usage, latency
3. **Tune for workload**: Adjust settings based on query patterns
4. **Use persistent cache in production**: Instant warm cache on startup
5. **Enable adaptive TTL**: Automatic optimization based on access patterns
6. **Consider multi-level for large datasets**: Better memory utilization
7. **Use predictive warming for predictable workloads**: Proactive cache preparation

## Future Enhancements

Potential improvements for future versions:
- Machine learning-based prediction
- Distributed cache (Redis, Memcached)
- Compression for large entries
- Cache prewarming from historical data
- Automatic cache size tuning
- Cross-process cache sharing

## References

- [Basic Caching Guide](../performance/optimization_guide.md)
- [Tool Selection Architecture](../architecture/overview.md)
- [Performance Best Practices](../performance/optimization_guide.md)
- [Configuration Reference](../reference/configuration/index.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
