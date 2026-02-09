# Advanced Caching Quick Start

**Get started with advanced caching in 5 minutes**

## Installation

No additional installation needed - all features are included in Victor.

## 1. Basic Adaptive Cache

```python
from victor.tools.caches import AdaptiveLRUCache

# Create cache with auto-sizing
cache = AdaptiveLRUCache(
    initial_size=500,
    max_size=2000,
    target_hit_rate=0.6,
)

# Use like a normal cache
cache.put("key", value)
value = cache.get("key")

# Auto-adjust (call periodically)
if cache.should_adjust():
    result = cache.adjust_size()
    print(f"Adjusted: {result['old_size']} -> {result['new_size']}")
```text

## 2. Predictive Warming

```python
from victor.tools.caches import PredictiveCacheWarmer

warmer = PredictiveCacheWarmer(cache=my_cache)

# Record queries as they occur
warmer.record_query("read file", ["read", "search"])

# Get predictions
predictions = warmer.predict_next_queries(
    current_query="analyze code",
    top_k=5,
)

# Prewarm cache
await warmer.prewarm_predictions(predictions)
```

## 3. Multi-Level Cache

```python
from victor.tools.caches import MultiLevelCache

cache = MultiLevelCache(
    l1_size=100,   # In-memory (fast)
    l2_size=1000,  # Local disk (medium)
    l3_size=10000, # Backup (slow)
)

# Use like a normal cache
cache.put("key", value)
value = cache.get("key")  # Checks L1 → L2 → L3

# Get metrics
metrics = cache.get_metrics()
print(f"Overall hit rate: {metrics['combined']['hit_rate']:.1%}")
```text

## 4. Enable All Optimizations

```python
from victor.tools.caches import (
    AdaptiveLRUCache,
    PredictiveCacheWarmer,
    MultiLevelCache,
)

# Create multi-level cache
cache = MultiLevelCache(l1_size=100, l2_size=1000, l3_size=10000)

# Create adaptive L1
l1_cache = AdaptiveLRUCache(initial_size=100, max_size=200)

# Create predictive warmer
warmer = PredictiveCacheWarmer(cache=cache, max_patterns=500)

# Use in your application
async def process_query(query):
    # Check cache
    cached = cache.get(query)
    if cached:
        return cached

    # Process query
    result = await process(query)

    # Store in cache
    cache.put(query, result)

    # Record for predictions
    warmer.record_query(query, result.tools)

    # Prewarm for next queries
    predictions = warmer.predict_next_queries(query)
    await warmer.prewarm_predictions(predictions)

    # Auto-adjust cache size
    if l1_cache.should_adjust():
        l1_cache.adjust_size()

    return result
```

## 5. Monitor Performance

```python
from victor.tools.caches import MultiLevelCache

cache = MultiLevelCache()

# Get comprehensive metrics
metrics = cache.get_metrics()

print(f"L1 hit rate: {metrics['l1']['hit_rate']:.1%}")
print(f"L2 hit rate: {metrics['l2']['hit_rate']:.1%}")
print(f"L3 hit rate: {metrics['l3']['hit_rate']:.1%}")
print(f"Overall: {metrics['combined']['hit_rate']:.1%}")
print(f"Promotions: {metrics['combined']['total_promotions']}")
print(f"Demotions: {metrics['combined']['total_demotions']}")
```text

## Common Patterns

### Memory-Constrained Systems
```python
cache = MultiLevelCache(
    l1_size=50,
    l2_size=500,
    l3_size=5000,
    l1_ttl=180,   # Shorter TTL
    l2_ttl=1800,
)
```

### High-Performance Systems
```python
cache = MultiLevelCache(
    l1_size=200,
    l2_size=2000,
    l3_size=20000,
    l1_ttl=600,   # Longer TTL
    l2_ttl=7200,
)
```text

### Conservative Predictions
```python
warmer = PredictiveCacheWarmer(max_patterns=200)
predictions = warmer.predict_next_queries(
    query,
    min_confidence=0.4,  # Higher threshold
)
```

### Aggressive Predictions
```python
warmer = PredictiveCacheWarmer(max_patterns=1000)
predictions = warmer.predict_next_queries(
    query,
    min_confidence=0.1,  # Lower threshold
)
```text

## Next Steps

- Read [Advanced Optimization Guide](./advanced_optimization.md) for detailed documentation
- Check [Caching Strategy](./caching_strategy.md) for design decisions
- See [Tool Selection Cache](./tool_selection_cache_implementation.md) for integration details

## Troubleshooting

### Low Hit Rate?
```python
# Increase cache size
cache = AdaptiveLRUCache(initial_size=1000, max_size=5000)

# Increase TTL
cache = MultiLevelCache(l1_ttl=600, l2_ttl=7200)
```

### High Memory Usage?
```python
# Reduce cache sizes
cache = MultiLevelCache(l1_size=50, l2_size=500, l3_size=5000)

# More aggressive adaptive sizing
cache = AdaptiveLRUCache(target_hit_rate=0.5)
```text

### Poor Prediction Accuracy?
```python
# Track more patterns
warmer = PredictiveCacheWarmer(max_patterns=1000)

# Use longer n-grams
warmer = PredictiveCacheWarmer(ngram_size=4)

# Lower confidence threshold
predictions = warmer.predict_next_queries(query, min_confidence=0.1)
```

---

**Last Updated:** February 01, 2026
**Reading Time:** 1 min
