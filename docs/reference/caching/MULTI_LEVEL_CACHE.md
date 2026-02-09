# Multi-Level Cache System

## Overview
## Cache Architecture

```mermaid
graph TB
    subgraph["Cache Layer"]
        A[Cache Interface]
        B[In-Memory Cache]
        C[Redis Backend]
    end

    A --> B
    A --> C

    B --> D[TTL Policy]
    C --> E[Persistence]

    F[Application] --> A

    style A fill:#e1f5ff
    style B fill:#e8f5e9
    style C fill:#fff4e1
```



The Multi-Level Cache (MLC) system implements a two-tier caching hierarchy for Victor AI:

- **L1 Cache**: Fast in-memory cache with LRU eviction (small size, sub-millisecond access)
- **L2 Cache**: Persistent cache with TTL-based expiration (larger size, millisecond access)

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     Application                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Multi-Level Cache                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   L1 Cache   │◄───────►│   L2 Cache   │                 │
│  │  (In-Memory) │         │ (Persistent) │                 │
│  │              │         │              │                 │
│  │  Max: 1,000  │         │ Max: 10,000  │                 │
│  │  TTL:  5min  │         │ TTL:  1hr    │                 │
│  └──────────────┘         └──────────────┘                 │
│         │                         │                         │
│         │  Promotion/Demotion     │                         │
│         └─────────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Operation | L1 Hit | L2 Hit | Miss |
|-----------|--------|--------|------|
| Latency | ~0.1ms | ~1-5ms | API call |
| Hit Rate | 40-60% | 20-30% | 10-40% |

**Overall Performance Improvement**: 30-50% latency reduction for cached queries

## Configuration

### Basic Setup

```python
from victor.core.cache import MultiLevelCache, CacheLevelConfig, WritePolicy

cache = MultiLevelCache(
    l1_config=CacheLevelConfig(
        max_size=1000,
        ttl=300,  # 5 minutes
        eviction_policy=EvictionPolicy.LRU,
    ),
    l2_config=CacheLevelConfig(
        max_size=10000,
        ttl=3600,  # 1 hour
        eviction_policy=EvictionPolicy.TTL,
        enable_persistence=True,
        persistence_path=Path.home() / ".victor" / "cache_l2.pkl",
    ),
    write_policy=WritePolicy.WRITE_THROUGH,
)
```text

### Write Policies

#### Write-Through (Default)
Writes to both L1 and L2 synchronously.
- **Pros**: Data safety, immediate persistence
- **Cons**: Slower writes
- **Use when**: Data durability is critical

```python
cache = MultiLevelCache(
    write_policy=WritePolicy.WRITE_THROUGH,
)
```

#### Write-Back
Writes to L1 immediately, L2 on eviction/flush.
- **Pros**: Fast writes, reduced L2 wear
- **Cons**: Risk of data loss on crash
- **Use when**: Write performance is critical

```python
cache = MultiLevelCache(
    write_policy=WritePolicy.WRITE_BACK,
)

# Periodically flush L1 to L2
await cache.flush()
```text

#### Write-Around
Writes to L2 only, bypassing L1.
- **Pros**: Avoids L1 pollution with write-once data
- **Cons**: First read is slower (must go to L2)
- **Use when**: Data is written once but read many times

```python
cache = MultiLevelCache(
    write_policy=WritePolicy.WRITE_AROUND,
)
```

### Eviction Policies

```python
from victor.core.cache import EvictionPolicy

# LRU (Least Recently Used) - Default
config = CacheLevelConfig(eviction_policy=EvictionPolicy.LRU)

# LFU (Least Frequently Used)
config = CacheLevelConfig(eviction_policy=EvictionPolicy.LFU)

# FIFO (First In, First Out)
config = CacheLevelConfig(eviction_policy=EvictionPolicy.FIFO)

# TTL (Time-based expiration)
config = CacheLevelConfig(eviction_policy=EvictionPolicy.TTL)
```text

## Usage Examples

### Basic Operations

```python
# Store value
await cache.set("result_123", computation_result(), namespace="tool")

# Retrieve value (checks L1, then L2)
result = await cache.get("result_123", namespace="tool")

# Delete value
await cache.delete("result_123", namespace="tool")

# Clear namespace
count = await cache.clear_namespace("tool")

# Clear all cache
await cache.clear()
```

### Automatic Promotion/Demotion

```python
# Access patterns drive automatic promotion
# Items accessed frequently in L2 are promoted to L1

# First access: L2 hit
result = await cache.get("key", namespace="tool")  # ~1-5ms

# Subsequent accesses: Promoted to L1
result = await cache.get("key", namespace="tool")  # ~0.1ms
```text

### Namespace Isolation

```python
# Different namespaces are isolated
await cache.set("config", {"key": "value"}, namespace="app")
await cache.set("config", {"key": "other"}, namespace="user")

# Each namespace has separate entries
app_config = await cache.get("config", namespace="app")
user_config = await cache.get("config", namespace="user")
```

## Monitoring and Statistics

```python
# Get comprehensive statistics
stats = cache.get_stats()

print(f"L1 Hit Rate: {stats['l1']['hit_rate']:.1%}")
print(f"L2 Hit Rate: {stats['l2']['hit_rate']:.1%}")
print(f"Combined Hit Rate: {stats['combined_hit_rate']:.1%}")
print(f"Write Policy: {stats['write_policy']}")

# Output:
# L1 Hit Rate: 55.2%
# L2 Hit Rate: 28.1%
# Combined Hit Rate: 83.3%
# Write Policy: write_through
```text

## Performance Tuning

### Sizing Guidelines

| Scenario | L1 Size | L2 Size | Rationale |
|----------|---------|---------|-----------|
| Small workload | 500 | 5000 | Minimal memory footprint |
| Medium workload | 1000 | 10000 | Balanced performance |
| Large workload | 5000 | 50000 | High hit rate |
| Memory-constrained | 500 | 20000 | Small L1, large L2 on disk |

### TTL Guidelines

| Data Type | L1 TTL | L2 TTL | Rationale |
|-----------|--------|--------|-----------|
| Real-time data | 60s | 300s | Freshness critical |
| Tool results | 5min | 1hr | Balance freshness and hit rate |
| Config data | 10min | 24hr | Changes infrequent |
| Static data | 30min | 7 days | Rarely changes |

## Best Practices

### 1. Use Appropriate Namespaces

```python
# Good: Specific namespaces
await cache.set("result", value, namespace="tool.code_analysis")
await cache.set("result", value, namespace="llm.response")

# Avoid: Generic namespace
await cache.set("result", value, namespace="default")  # Too broad
```

### 2. Set Appropriate TTLs

```python
# Short TTL for frequently changing data
await cache.set("result", value, namespace="tool", ttl=60)

# Long TTL for static data
await cache.set("config", value, namespace="app", ttl=86400)
```text

### 3. Monitor Performance

```python
# Regularly check statistics
stats = cache.get_stats()

if stats['combined_hit_rate'] < 0.5:
    logger.warning("Low cache hit rate, consider tuning")

if stats['l1']['hit_rate'] < 0.3:
    logger.info("Low L1 hit rate, consider increasing L1 size")
```

### 4. Handle Errors Gracefully

```python
try:
    result = await cache.get(key, namespace="tool")
    if result is not None:
        return result
except Exception as e:
    logger.error(f"Cache error: {e}")
    # Fall back to computation
    pass

# Compute result
result = compute_value()
await cache.set(key, result, namespace="tool")
return result
```text

## Integration with Other Systems

### Combine with Cache Warming

```python
from victor.core.cache import CacheWarmer

cache = MultiLevelCache(...)
warmer = CacheWarmer(cache=cache, strategy=WarmingStrategy.HYBRID)

# Start background warming
await warmer.start_background_warming()
```

### Add Analytics

```python
from victor.core.cache import CacheAnalytics

cache = MultiLevelCache(...)
analytics = CacheAnalytics(cache=cache, track_hot_keys=True)

# Monitor performance
await analytics.start_monitoring(interval_seconds=60)
```text

### Enable Intelligent Invalidation

```python
from victor.core.cache import CacheInvalidator

cache = MultiLevelCache(...)
invalidator = CacheInvalidator(cache=cache, enable_tagging=True)

# Tag entries
invalidator.tag("key1", "tool", ["python_files", "src"])

# Invalidate by tag
await invalidator.invalidate_tag("python_files")
```

## Troubleshooting

### Low Hit Rate

**Symptoms**: Hit rate < 50%

**Possible Causes**:
1. Cache size too small
2. TTL too short
3. High data churn
4. Poor cache key design

**Solutions**:
1. Increase cache size
2. Increase TTL
3. Use better cache keys
4. Enable cache warming

### High Memory Usage

**Symptoms**: Memory consumption growing

**Possible Causes**:
1. L1 cache too large
2. Insufficient eviction
3. Memory leak

**Solutions**:
1. Reduce L1 size
2. Check TTL settings
3. Monitor memory usage
4. Use L2 with disk persistence

### Slow Cache Access

**Symptoms**: Cache latency > 5ms

**Possible Causes**:
1. L2 cache on slow disk
2. Large cache entries
3. Serialization overhead

**Solutions**:
1. Use faster storage for L2 (SSD)
2. Reduce entry sizes
3. Optimize serialization
4. Increase L1 hit rate

## See Also

- [Cache Warming Guide](CACHE_WARMING.md)
- [Semantic Caching](SEMANTIC_CACHING.md)
- [Cache Invalidation](CACHE_INVALIDATION.md)
- [Cache Analytics](CACHE_ANALYTICS.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
