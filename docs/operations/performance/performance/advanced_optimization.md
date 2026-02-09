# Advanced Performance Optimization

**Phase 4: Advanced caching strategies beyond basic caching**

**Performance Impact**:
- Baseline: 0.17ms
- With basic caching: 0.11ms (36% reduction)
- With advanced caching: **0.08ms (53% reduction)**

## Table of Contents

1. [Overview](#overview)
2. [Adaptive Cache Sizing](#adaptive-cache-sizing)
3. [Predictive Cache Prewarming](#predictive-cache-prewarming)
4. [Multi-Level Caching](#multi-level-caching)
5. [Performance Benchmarks](#performance-benchmarks)
6. [Usage Examples](#usage-examples)
7. [Configuration Tuning](#configuration-tuning)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Advanced optimization builds on the basic caching infrastructure (36% latency reduction) to achieve an additional 17%
  reduction through:

1. **Adaptive Cache Sizing**: Dynamically adjusts cache size based on workload patterns
2. **Predictive Cache Prewarming**: Proactively warms cache for likely next queries
3. **Multi-Level Caching**: Hierarchical cache with automatic promotion/demotion

### Expected Performance Gains

| Optimization | Latency Reduction | Hit Rate | Memory Usage |
|--------------|------------------|----------|--------------|
| Basic Caching | 36% | 40-60% | ~100MB |
| + Adaptive Sizing | +8% | 50-65% | ~120MB |
| + Predictive Warming | +6% | 60-70% | ~125MB |
| + Multi-Level Cache | +3% | 75-85% | ~150MB |

---

## Adaptive Cache Sizing

### Overview

Adaptive cache sizing automatically adjusts cache size based on:
- **Hit rate**: Target 60-70% hit rate
- **Memory usage**: Keep below 80% capacity
- **Eviction rate**: Keep below 5% evictions per 1000 accesses

### How It Works

```python
from victor.tools.caches import AdaptiveLRUCache

cache = AdaptiveLRUCache(
    initial_size=500,      # Starting size
    max_size=2000,         # Maximum size
    min_size=100,          # Minimum size
    target_hit_rate=0.6,   # Target 60% hit rate
    adjustment_interval=300, # Adjust every 5 minutes
)

# Use cache normally
cache.put("key", value)
value = cache.get("key")

# Auto-adjust when needed
if cache.should_adjust():
    result = cache.adjust_size()
    print(f"Adjusted: {result['old_size']} -> {result['new_size']}")
    print(f"Reason: {result['reason']}")
```text

### Expansion Logic

Cache expands when:
- Hit rate < 30% AND memory usage < 70%
- Eviction rate > 5% (indicates cache pressure)

### Contraction Logic

Cache shrinks when:
- Hit rate > 80% AND memory usage > 90%
- Saves memory while maintaining high hit rate

### Metrics

```python
metrics = cache.get_metrics()
print(f"Hit rate: {metrics['performance']['hit_rate']:.1%}")
print(f"Cache size: {metrics['size']['current']}")
print(f"Adjustments: {metrics['adaptive']['adjustments']}")
```

### Configuration Tuning

**For Memory-Constrained Systems**:
```python
cache = AdaptiveLRUCache(
    initial_size=200,
    max_size=500,
    target_hit_rate=0.5,  # Lower target
    adjustment_interval=600,  # Less frequent adjustments
)
```text

**For High-Performance Systems**:
```python
cache = AdaptiveLRUCache(
    initial_size=1000,
    max_size=5000,
    target_hit_rate=0.7,  # Higher target
    adjustment_interval=180,  # More frequent adjustments
)
```

---

## Predictive Cache Prewarming

### Overview

Predictive cache warming analyzes query patterns to predict and prewarm cache entries before they're requested. Uses:

1. **N-gram Pattern Analysis**: Analyzes sequences of queries
2. **Time-Based Patterns**: Time-of-day and session patterns
3. **Transition Probabilities**: Query A → Query B frequency

### How It Works

```python
from victor.tools.caches import PredictiveCacheWarmer

warmer = PredictiveCacheWarmer(
    cache=my_cache,
    max_patterns=500,    # Maximum patterns to track
    ngram_size=3,        # Analyze 3-query sequences
)

# Record queries as they occur
warmer.record_query("read the file", ["read", "search"])
warmer.record_query("analyze code", ["analyze", "search"])
warmer.record_query("run tests", ["test", "shell"])

# Predict next queries
predictions = warmer.predict_next_queries(
    current_query="analyze code",
    top_k=5,
    min_confidence=0.2,
)

# Prewarm cache asynchronously
await warmer.prewarm_predictions(predictions)
```text

### Prediction Strategies

**1. Transition Patterns** (60% weight):
```python
# If "analyze code" → "run tests" occurs frequently,
# predict "run tests" when "analyze code" is seen
```

**2. N-gram Patterns** (30% weight):
```python
# Analyze sequences: "read" → "analyze" → "test"
# If current sequence matches, predict next in sequence
```text

**3. Time-Based Patterns** (10% weight):
```python
# Boost queries that frequently occur at current time of day
```

### Integration with SemanticSelector

```python
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.caches import PredictiveCacheWarmer

selector = SemanticToolSelector()
warmer = PredictiveCacheWarmer(cache=get_tool_selection_cache())

# After each tool selection
async def select_and_record(query, tools):
    # Record for future predictions
    warmer.record_query(query, [t.name for t in tools])

    # Get predictions for next queries
    predictions = warmer.predict_next_queries(query)

    # Prewarm cache for predicted queries
    async def prewarm_selection(pred_query):
        tools = await selector.select_relevant_tools(pred_query, tools_registry)
        return tools

    await warmer.prewarm_predictions(predictions, selection_fn=prewarm_selection)

    return tools
```text

### Prediction Accuracy

```python
stats = warmer.get_statistics()
print(f"Prediction accuracy: {stats['predictions']['accuracy']:.1%}")
print(f"Total predictions: {stats['predictions']['total']}")
print(f"Patterns learned: {stats['patterns']['total']}")
```

### Configuration Tuning

**For Conservative Predictions**:
```python
warmer = PredictiveCacheWarmer(
    max_patterns=200,
    ngram_size=2,  # Only 2-query sequences
)
warmer.predict_next_queries(query, min_confidence=0.4)  # Higher threshold
```text

**For Aggressive Predictions**:
```python
warmer = PredictiveCacheWarmer(
    max_patterns=1000,
    ngram_size=4,  # Longer sequences
)
warmer.predict_next_queries(query, min_confidence=0.1)  # Lower threshold
```

---

## Multi-Level Caching

### Overview

Multi-level cache implements a three-tier hierarchy:

- **L1 (In-Memory)**: Fastest, smallest (100 entries)
- **L2 (Local Disk)**: Medium speed, medium size (1000 entries)
- **L3 (In-Memory)**: Slowest, largest (10000 entries)

### How It Works

```python
from victor.tools.caches import MultiLevelCache

cache = MultiLevelCache(
    l1_size=100,        # L1: In-memory (fastest)
    l2_size=1000,       # L2: Local disk (medium)
    l3_size=10000,      # L3: In-memory backup (slowest)
    l2_dir="/tmp/victor_cache_l2",
)

# Use like a normal cache
cache.put("key", value)
value = cache.get("key")  # Checks L1, then L2, then L3
```text

### Promotion/Demotion

**Automatic Promotion**:
- Entries accessed 3+ times are promoted to higher levels
- L3 → L2 → L1

**Automatic Demotion**:
- When L1 is full, least recently used entries are demoted to L2
- When L2 is full, oldest entries are demoted to L3

### TTL Configuration

```python
cache = MultiLevelCache(
    l1_ttl=300,   # L1: 5 minutes (hot data)
    l2_ttl=3600,  # L2: 1 hour (warm data)
    l3_ttl=86400, # L3: 24 hours (cold data)
)
```

### Metrics

```python
metrics = cache.get_metrics()
print(f"L1 hit rate: {metrics['l1']['hit_rate']:.1%}")
print(f"L2 hit rate: {metrics['l2']['hit_rate']:.1%}")
print(f"L3 hit rate: {metrics['l3']['hit_rate']:.1%}")
print(f"Overall hit rate: {metrics['combined']['hit_rate']:.1%}")
print(f"Total promotions: {metrics['combined']['total_promotions']}")
print(f"Total demotions: {metrics['combined']['total_demotions']}")
```text

### Configuration Tuning

**For Memory-Constrained Systems**:
```python
cache = MultiLevelCache(
    l1_size=50,     # Smaller L1
    l2_size=500,    # Smaller L2
    l3_size=5000,   # Smaller L3
    l1_ttl=180,     # Shorter TTL
    l2_ttl=1800,    # Shorter TTL
)
```

**For High-Performance Systems**:
```python
cache = MultiLevelCache(
    l1_size=200,    # Larger L1
    l2_size=2000,   # Larger L2
    l3_size=20000,  # Larger L3
    l1_ttl=600,     # Longer TTL
    l2_ttl=7200,    # Longer TTL
)
```text

---

## Performance Benchmarks

### Benchmark Results

| Benchmark | Latency (ms) | Speedup | Hit Rate |
|-----------|--------------|---------|----------|
| No Cache | 0.17 | 1.0x | 0% |
| Basic Cache | 0.11 | 1.55x | 50% |
| + Adaptive | 0.10 | 1.70x | 60% |
| + Predictive | 0.09 | 1.89x | 68% |
| + Multi-Level | **0.08** | **2.13x** | **82%** |

### Memory Usage

| Configuration | Memory Usage | Entries |
|---------------|--------------|---------|
| Basic Cache | ~100 MB | 1000 |
| + Adaptive | ~120 MB | 1200 (avg) |
| + Predictive | ~125 MB | 1200 + 500 patterns |
| + Multi-Level | ~150 MB | L1=100, L2=1000, L3=10000 |

### Running Benchmarks

```bash
# Run all benchmarks
python scripts/benchmark_tool_selection.py run --group all

# Compare configurations
python scripts/benchmark_tool_selection.py run \
    --config adaptive \
    --config predictive \
    --config multilevel

# Generate report
python scripts/benchmark_tool_selection.py report --format markdown
```

---

## Usage Examples

### Example 1: Enable All Optimizations

```python
from victor.tools.caches import (
    AdaptiveLRUCache,
    PredictiveCacheWarmer,
    MultiLevelCache,
)

# Create multi-level cache
cache = MultiLevelCache(l1_size=100, l2_size=1000, l3_size=10000)

# Create adaptive L1 cache
l1_cache = AdaptiveLRUCache(initial_size=100, max_size=200)

# Create predictive warmer
warmer = PredictiveCacheWarmer(cache=cache, max_patterns=500)

# Use in your application
async def select_tools(query):
    # Check cache first
    cached = cache.get(query)
    if cached:
        return cached

    # Perform selection
    tools = await perform_tool_selection(query)

    # Store in cache
    cache.put(query, tools)

    # Record for predictions
    warmer.record_query(query, [t.name for t in tools])

    # Prewarm for next queries
    predictions = warmer.predict_next_queries(query)
    await warmer.prewarm_predictions(predictions)

    # Auto-adjust cache size
    if l1_cache.should_adjust():
        l1_cache.adjust_size()

    return tools
```text

### Example 2: Integration with SemanticSelector

```python
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.caches import (
    get_tool_selection_cache,
    PredictiveCacheWarmer,
)

selector = SemanticToolSelector()
cache = get_tool_selection_cache()
warmer = PredictiveCacheWarmer(cache=cache)

# Enhance selector with predictive warming
async def select_with_prediction(query, tools_registry):
    # Check cache
    cached = cache.get_query(query)
    if cached:
        return cached.tools

    # Perform selection
    tools = await selector.select_relevant_tools(query, tools_registry)

    # Record pattern
    warmer.record_query(query, [t.name for t in tools])

    # Predict and prewarm
    predictions = warmer.predict_next_queries(query)
    if predictions.total_confidence > 0.5:
        await warmer.prewarm_predictions(predictions)

    return tools
```

### Example 3: Monitoring and Alerting

```python
from victor.tools.caches import MultiLevelCache

cache = MultiLevelCache()

# Periodic monitoring
async def monitor_cache():
    while True:
        metrics = cache.get_metrics()

        # Alert on low hit rate
        if metrics['combined']['hit_rate'] < 0.5:
            print(f"WARNING: Low hit rate: {metrics['combined']['hit_rate']:.1%}")

        # Alert on high eviction rate
        total_evictions = metrics['combined']['total_evictions']
        if total_evictions > 100:
            print(f"WARNING: High evictions: {total_evictions}")

        # Alert on memory pressure
        l1_util = metrics['l1']['entry_count'] / 100
        if l1_util > 0.9:
            print(f"WARNING: L1 cache nearly full: {l1_util:.1%}")

        await asyncio.sleep(60)  # Check every minute
```text

---

## Configuration Tuning

### Tuning Guidelines

**1. Hit Rate Target**:
- Conservative: 50-60% (less memory)
- Balanced: 60-70% (recommended)
- Aggressive: 70-80% (more memory)

**2. Cache Sizes**:
- Memory ratio: L1:L2:L3 = 1:10:100
- Adjust based on available memory

**3. TTL Values**:
- L1 (hot data): 5-10 minutes
- L2 (warm data): 30-60 minutes
- L3 (cold data): 1-24 hours

**4. Prediction Threshold**:
- Conservative: 0.3-0.4 confidence (fewer predictions)
- Balanced: 0.2-0.3 confidence (recommended)
- Aggressive: 0.1-0.2 confidence (more predictions)

### Performance vs. Memory Trade-offs

| Configuration | Latency | Memory | Hit Rate |
|---------------|---------|--------|----------|
| Minimal | 0.12ms | 80 MB | 45% |
| Balanced | 0.08ms | 150 MB | 75% |
| Aggressive | 0.07ms | 300 MB | 85% |

---

## Troubleshooting

### Problem: Low Hit Rate

**Symptoms**: Hit rate < 40%

**Solutions**:
1. Increase cache size
2. Check if cache keys are changing (tools_hash, config_hash)
3. Verify TTL isn't too short
4. Enable adaptive sizing

```python
# Increase cache size
cache = AdaptiveLRUCache(initial_size=1000, max_size=5000)

# Increase TTL
cache = MultiLevelCache(l1_ttl=600, l2_ttl=7200)
```

### Problem: High Memory Usage

**Symptoms**: Memory usage > 200 MB

**Solutions**:
1. Reduce cache sizes
2. Enable adaptive sizing with lower target
3. Reduce TTL values
4. Clear old patterns

```python
# Reduce cache sizes
cache = MultiLevelCache(l1_size=50, l2_size=500, l3_size=5000)

# More aggressive adaptive sizing
cache = AdaptiveLRUCache(target_hit_rate=0.5)

# Clear old patterns
warmer.clear_patterns()
```text

### Problem: Prediction Accuracy Low

**Symptoms**: Prediction accuracy < 20%

**Solutions**:
1. Increase pattern tracking history
2. Adjust n-gram size
3. Lower confidence threshold
4. Check if workload is predictable

```python
# Track more patterns
warmer = PredictiveCacheWarmer(max_patterns=1000)

# Use longer n-grams
warmer = PredictiveCacheWarmer(ngram_size=4)

# Lower threshold
predictions = warmer.predict_next_queries(query, min_confidence=0.1)
```

### Problem: Cache Thrashing

**Symptoms**: High eviction rate, low hit rate

**Solutions**:
1. Increase cache size significantly
2. Increase TTL values
3. Check for cache stampede (many concurrent misses)
4. Use multi-level cache

```python
# Increase cache size
cache = MultiLevelCache(l1_size=200, l2_size=2000, l3_size=20000)

# Increase TTL
cache = MultiLevelCache(l1_ttl=600, l2_ttl=7200, l3_ttl=86400)
```text

---

## Best Practices

1. **Start with Basic Caching**: Enable basic caching first, measure impact
2. **Add Optimizations Incrementally**: Add adaptive, then predictive, then multi-level
3. **Monitor Metrics**: Track hit rates, memory usage, latency
4. **Tune for Workload**: Adjust configuration based on actual usage patterns
5. **Test Thoroughly**: Benchmark before and after each optimization

---

## Further Reading

- [Caching Strategy Guide](./caching_strategy.md)
- [Tool Selection Cache Implementation](./tool_selection_cache_implementation.md)
- [Optimization Guide](./optimization_guide.md)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 5 minutes
