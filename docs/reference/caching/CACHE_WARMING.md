# Cache Warming Guide

## Overview

Cache warming proactively populates the cache with frequently accessed data to reduce cold starts and improve hit rates. Victor AI's intelligent cache warming system uses multiple strategies to predict and pre-load data before it's needed.

## Benefits

- **Reduced Cold Starts**: 30-50% improvement in first-access latency
- **Higher Hit Rates**: 10-20% increase in overall cache hit rate
- **Better User Experience**: Faster responses from the start
- **Optimal Resource Usage**: Background warming doesn't block requests

## Warming Strategies

### 1. Frequency-Based Warming

Warms most frequently accessed items based on historical access patterns.

**Best for**: Stable workloads with consistent access patterns

```python
from victor.core.cache import CacheWarmer, WarmingStrategy

warmer = CacheWarmer(
    cache=cache,
    strategy=WarmingStrategy.FREQUENCY,
    preload_count=100,
)

# Warm top 100 most frequently accessed items
await warmer.warm_top_items(n=100)
```

**How it works**:
1. Tracks access frequency for each key
2. Identifies top N most frequently accessed keys
3. Pre-loads these items into cache

### 2. Recency-Based Warming

Warms most recently accessed items.

**Best for**: Time-sensitive data, recent workloads

```python
warmer = CacheWarmer(
    cache=cache,
    strategy=WarmingStrategy.RECENCY,
    preload_count=100,
)

# Warm top 100 most recently accessed items
await warmer.warm_top_items(n=100)
```

**How it works**:
1. Tracks access timestamps
2. Identifies most recently accessed keys
3. Pre-loads recent items

### 3. Hybrid Warming (Recommended)

Combines frequency and recency scoring for optimal results.

```python
warmer = CacheWarmer(
    cache=cache,
    strategy=WarmingStrategy.HYBRID,
    preload_count=100,
    recency_weight=0.5,  # Equal weight to frequency and recency
)

# Warm top items using hybrid scoring
await warmer.warm_top_items(n=100)
```

**Scoring Formula**:
```
hybrid_score = recency_weight * recency_score + (1 - recency_weight) * frequency_score
```

**Tuning**:
- `recency_weight=0.0`: Frequency-only (stable data)
- `recency_weight=1.0`: Recency-only (time-sensitive data)
- `recency_weight=0.5`: Balanced (default)

### 4. Time-Based Warming

Schedules warming based on time-of-day patterns.

**Best for**: Workloads with daily patterns (e.g., business hours vs. overnight)

```python
warmer = CacheWarmer(
    cache=cache,
    strategy=WarmingStrategy.TIME_BASED,
    enable_time_based=True,
)

# Warm items for current hour
await warmer.warm_top_items()

# Or predict for specific hour
await warmer.warm_top_items(hour=9)  # 9 AM patterns
```

**How it works**:
1. Tracks access patterns by hour
2. Identifies popular items for each time period
3. Pre-loads items based on current time

### 5. User-Specific Warming

Personalized warming per user or context.

**Best for**: Multi-tenant systems, personalized content

```python
# Record access with user ID
await warmer.record_access(
    key="result_123",
    namespace="tool",
    hit=True,
    user_id="user@example.com",
)

# Warm items for specific user
await warmer.warm_top_items(user_id="user@example.com", n=50)
```

## Configuration

### Basic Configuration

```python
from victor.core.cache import CacheWarmer, WarmingConfig

config = WarmingConfig(
    strategy=WarmingStrategy.HYBRID,
    preload_count=100,
    warm_interval=300,  # 5 minutes
    recency_weight=0.5,
    enable_time_based=True,
    enable_user_specific=True,
)

warmer = CacheWarmer(cache=cache, config=config)
```

### Value Loader

Provide a function to load values for warming:

```python
async def load_tool_result(key: str, namespace: str) -> Any:
    """Load value for cache warming."""
    # Re-compute or fetch from source
    return await expensive_computation(key)

warmer = CacheWarmer(
    cache=cache,
    value_loader=load_tool_result,
)
```

## Usage Examples

### Startup Warming

```python
import asyncio
from victor.core.cache import MultiLevelCache, CacheWarmer

async def startup():
    # Initialize cache
    cache = MultiLevelCache(...)

    # Initialize warmer
    warmer = CacheWarmer(cache=cache, strategy=WarmingStrategy.HYBRID)

    # Load historical patterns
    await warmer.load_patterns_from_history()

    # Warm top items
    count = await warmer.warm_top_items(n=100)
    logger.info(f"Warmed {count} cache items on startup")

    # Start background warming
    await warmer.start_background_warming()

    return cache, warmer

# On application startup
cache, warmer = await startup()
```

### Background Warming

```python
# Start automatic background warming
await warmer.start_background_warming()

# Warming runs every 5 minutes (configurable)
# Periodically warms top items based on strategy

# Stop when shutting down
await warmer.stop_background_warming()
```

### Manual Warming

```python
# Warm specific item
await warmer.warm_item(
    key="result_123",
    value=computation_result(),
    namespace="tool",
    ttl=3600,
)

# Warm top items
count = await warmer.warm_top_items(n=100, namespace="tool")

# Warm for specific user
count = await warmer.warm_top_items(
    user_id="user@example.com",
    n=50,
    namespace="tool",
)
```

### Pattern Persistence

```python
# Save access patterns to disk
await warmer.save_patterns_to_history()

# Load patterns on startup
await warmer.load_patterns_from_history()

# Patterns are persisted to ~/.victor/cache_warming_history.json
```

## Monitoring and Statistics

```python
# Get warming statistics
stats = warmer.get_stats()

print(f"Warming Strategy: {stats['strategy']}")
print(f"Total Patterns: {stats['total_patterns']}")
print(f"Unique Keys: {stats['unique_keys']}")
print(f"Users Tracked: {stats['users_tracked']}")
print(f"Background Running: {stats['background_running']}")
```

## Best Practices

### 1. Choose the Right Strategy

```python
# Stable workload: Frequency-based
warmer = CacheWarmer(cache=cache, strategy=WarmingStrategy.FREQUENCY)

# Time-sensitive: Recency-based
warmer = CacheWarmer(cache=cache, strategy=WarmingStrategy.RECENCY)

# Balanced: Hybrid (recommended)
warmer = CacheWarmer(cache=cache, strategy=WarmingStrategy.HYBRID)
```

### 2. Set Appropriate Warm Intervals

```python
# Frequent warming (1-2 minutes): Fast-changing data
warmer = CacheWarmer(cache=cache, warm_interval=60)

# Moderate warming (5-10 minutes): Balanced
warmer = CacheWarmer(cache=cache, warm_interval=300)

# Infrequent warming (15-30 minutes): Stable data
warmer = CacheWarmer(cache=cache, warm_interval=900)
```

### 3. Record Access Patterns

```python
# Record every cache access for pattern tracking
async def get_with_warming(key: str, namespace: str):
    # Try cache
    value = await cache.get(key, namespace)

    # Record access
    hit = value is not None
    await warmer.record_access(key, namespace, hit=hit)

    return value
```

### 4. Handle Errors Gracefully

```python
# Warming should never fail the application
try:
    count = await warmer.warm_top_items(n=100)
    logger.info(f"Warmed {count} items")
except Exception as e:
    logger.error(f"Cache warming failed: {e}")
    # Continue without warming
```

## Performance Impact

### Startup Warming

| Metric | Without Warming | With Warming | Improvement |
|--------|----------------|--------------|-------------|
| Cold starts | 100% | 30-50% | 50-70% reduction |
| First-access latency | 100-500ms | 10-50ms | 90% reduction |
| Initial hit rate | 0% | 60-80% | Significant |

### Background Warming

| Metric | Without Warming | With Warming | Improvement |
|--------|----------------|--------------|-------------|
| Overall hit rate | 40-50% | 50-60% | 10-20% increase |
| Cache misses | 50-60% | 40-50% | 10-20% reduction |
| Avg latency | 50ms | 30ms | 40% reduction |

### Resource Overhead

| Resource | Overhead |
|----------|----------|
| Memory | ~5-10% additional for tracking |
| CPU | <1% for background warming |
| Disk | ~1-5MB for pattern history |
| Network | Minimal (local operations) |

## Integration Examples

### With Multi-Level Cache

```python
from victor.core.cache import MultiLevelCache, CacheWarmer

cache = MultiLevelCache(...)
warmer = CacheWarmer(cache=cache)

# Warm both L1 and L2
await warmer.warm_top_items(n=100)
```

### With Analytics

```python
from victor.core.cache import CacheWarmer, CacheAnalytics

cache = MultiLevelCache(...)
warmer = CacheWarmer(cache=cache)
analytics = CacheAnalytics(cache=cache)

# Monitor warming effectiveness
await warmer.warm_top_items(n=100)
stats = analytics.get_comprehensive_stats()
print(f"Hit rate after warming: {stats['hit_rate']:.1%}")
```

### With Invalidation

```python
from victor.core.cache import CacheWarmer, CacheInvalidator

cache = MultiLevelCache(...)
warmer = CacheWarmer(cache=cache)
invalidator = CacheInvalidator(cache=cache)

# On invalidation, re-warm affected items
async def on_invalidate(key: str, namespace: str):
    await invalidator.invalidate(key, namespace)
    # Re-warm if important
    await warmer.warm_item(key, reload_value(key), namespace)
```

## Troubleshooting

### Warming Not Effective

**Symptoms**: Hit rate doesn't improve after warming

**Possible Causes**:
1. Wrong strategy for workload
2. Insufficient historical data
3. Too few items warmed
4. Value loader too slow

**Solutions**:
1. Try different warming strategy
2. Collect more access patterns
3. Increase preload_count
4. Optimize value_loader function

### High Resource Usage

**Symptoms**: High CPU or memory during warming

**Possible Causes**:
1. Warm interval too short
2. Too many items warmed
3. Value loader blocking

**Solutions**:
1. Increase warm_interval
2. Reduce preload_count
3. Use async value_loader

### Background Warming Not Running

**Symptoms**: Items not warming automatically

**Possible Causes**:
1. Background task not started
2. Event loop not running
3. Task crashed

**Solutions**:
1. Call start_background_warming()
2. Ensure asyncio event loop running
3. Check logs for errors

## See Also

- [Multi-Level Cache](MULTI_LEVEL_CACHE.md)
- [Semantic Caching](SEMANTIC_CACHING.md)
- [Cache Invalidation](CACHE_INVALIDATION.md)
- [Cache Analytics](CACHE_ANALYTICS.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
