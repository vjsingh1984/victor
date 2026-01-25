# Advanced Caching System for Victor AI 0.5.0

Comprehensive, production-ready caching system with multi-level hierarchy, intelligent warming, semantic similarity, and advanced monitoring.

## Features

### 🚀 Multi-Level Cache
- **L1 Cache**: Fast in-memory cache with sub-millisecond access
- **L2 Cache**: Persistent cache with larger capacity
- **Automatic Promotion/Demotion**: Smart data movement between levels
- **Write Policies**: Write-through, write-back, write-around
- **Eviction Strategies**: LRU, LFU, FIFO, TTL-based

### 🔥 Cache Warming
- **Frequency-Based**: Warm most frequently accessed items
- **Recency-Based**: Warm most recently accessed items
- **Hybrid Strategy**: Combine frequency and recency scoring
- **Time-Based**: Schedule warming based on usage patterns
- **User-Specific**: Personalized warming per user/context
- **Background Warming**: Automatic periodic cache population

### 🧠 Semantic Caching
- **Vector Similarity**: Find semantically similar cached queries
- **Configurable Threshold**: Tune similarity matching (0-1)
- **Batch Processing**: Efficient similarity computation
- **Exact Match Fallback**: Graceful degradation
- **Multiple Embedding Models**: Support for various embedding providers

### 🔄 Cache Invalidation
- **Time-Based (TTL)**: Automatic expiration
- **Event-Based**: React to system events (file changes, config updates)
- **Manual**: Explicit invalidation API
- **Tag-Based**: Group invalidation by tags
- **Dependency Graph**: Cascade invalidation through dependencies
- **Predictive Refresh**: Proactive refresh of hot entries

### 📊 Cache Analytics
- **Performance Metrics**: Hit rate, miss rate, eviction rate, latency
- **Hot Key Detection**: Identify frequently accessed entries
- **Optimization Recommendations**: Actionable improvement suggestions
- **Prometheus Export**: Native metrics export
- **Real-Time Monitoring**: Background monitoring with alerts
- **Dashboard Integration**: Grafana, Streamlit support

## Installation

The caching system is included with Victor AI 0.5.0+:

```bash
pip install victor-ai>=0.5.0
```

## Quick Start

### Basic Multi-Level Cache

```python
from victor.core.cache import MultiLevelCache, CacheLevelConfig

# Create cache
cache = MultiLevelCache(
    l1_config=CacheLevelConfig(max_size=1000, ttl=300),
    l2_config=CacheLevelConfig(max_size=10000, ttl=3600),
)

# Use cache
await cache.set("key", value, namespace="tool")
result = await cache.get("key", namespace="tool")

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['combined_hit_rate']:.1%}")
```

### Cache with Warming

```python
from victor.core.cache import MultiLevelCache, CacheWarmer, WarmingStrategy

cache = MultiLevelCache(...)
warmer = CacheWarmer(cache=cache, strategy=WarmingStrategy.HYBRID)

# Start background warming
await warmer.start_background_warming()
```

### Semantic Caching

```python
from victor.core.cache import SemanticCache
from victor.providers.base import Message

cache = SemanticCache(similarity_threshold=0.85)

# Store response
messages = [Message(role="user", content="How do I parse JSON?")]
await cache.put(messages, response)

# Retrieve with semantic similarity
result = await cache.get_similar(messages)
```

### Complete Setup

```python
from victor.core.cache import (
    MultiLevelCache,
    CacheWarmer,
    SemanticCache,
    CacheInvalidator,
    CacheAnalytics,
)

# Initialize components
cache = MultiLevelCache(...)
warmer = CacheWarmer(cache=cache)
invalidator = CacheInvalidator(cache=cache)
analytics = CacheAnalytics(cache=cache)

# Start services
await warmer.start_background_warming()
await analytics.start_monitoring(interval_seconds=60)

# Use cache
await cache.set("key", value, namespace="tool")
result = await cache.get("key", namespace="tool")

# Monitor performance
stats = analytics.get_comprehensive_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

## Configuration

### YAML Configuration

Add to `~/.victor/profiles.yaml`:

```yaml
cache:
  multi_level:
    enabled: true
    l1:
      type: memory
      max_size: 1000
      ttl: 300
      eviction: lru
    l2:
      type: file
      max_size: 10000
      ttl: 3600
      eviction: ttl
      persistence_path: ~/.victor/cache_l2.pkl
    write_policy: write_through

  warming:
    enabled: true
    strategy: hybrid
    preload_count: 100
    warm_interval: 300
    recency_weight: 0.5

  semantic:
    enabled: true
    similarity_threshold: 0.85
    embedding_model: text-embedding-ada-002
    max_size: 1000
    batch_size: 100

  invalidation:
    strategy: ttl
    default_ttl: 3600
    enable_tagging: true
    enable_dependencies: true

  analytics:
    enabled: true
    track_hot_keys: true
    hot_key_window: 1000
    monitoring_interval: 60
```

### Programmatic Configuration

```python
from victor.core.cache import (
    MultiLevelCache,
    CacheLevelConfig,
    WarmingConfig,
    InvalidationConfig,
)

cache = MultiLevelCache(
    l1_config=CacheLevelConfig(
        max_size=1000,
        ttl=300,
        eviction_policy=EvictionPolicy.LRU,
    ),
    l2_config=CacheLevelConfig(
        max_size=10000,
        ttl=3600,
        eviction_policy=EvictionPolicy.TTL,
    ),
    write_policy=WritePolicy.WRITE_THROUGH,
)

warmer = CacheWarmer(
    cache=cache,
    config=WarmingConfig(
        strategy=WarmingStrategy.HYBRID,
        preload_count=100,
        warm_interval=300,
    ),
)

invalidator = CacheInvalidator(
    cache=cache,
    config=InvalidationConfig(
        strategy=InvalidationStrategy.HYBRID,
        default_ttl=3600,
        enable_tagging=True,
    ),
)
```

## Performance

### Benchmarks

| Metric | Without Cache | With Cache | Improvement |
|--------|---------------|------------|-------------|
| Avg Latency | 100-500ms | 0.1-5ms | **95-99% reduction** |
| Hit Rate | 0% | 60-80% | Significant |
| API Calls | 100% | 20-40% | **60-80% reduction** |
| Cost | $1.00 | $0.20-0.40 | **60-80% savings** |

### Multi-Level Cache Performance

| Level | Hit Rate | Latency | Capacity |
|-------|----------|---------|----------|
| L1 | 40-60% | ~0.1ms | 1,000 entries |
| L2 | 20-30% | ~1-5ms | 10,000 entries |
| Miss | 10-40% | API call | N/A |

### Semantic Cache Performance

| Metric | Value |
|--------|-------|
| Hit Rate (Similarity > 0.85) | 40-60% |
| Semantic Hit Rate | 50-70% of hits |
| Embedding Computation | 50-200ms |
| Similarity Search (1000 entries) | 1-5ms |

## Documentation

- [Multi-Level Cache Guide](MULTI_LEVEL_CACHE.md)
- [Cache Warming Guide](CACHE_WARMING.md)
- [Semantic Caching Guide](SEMANTIC_CACHING.md)
- [Cache Invalidation Guide](CACHE_INVALIDATION.md)
- [Cache Analytics Guide](CACHE_ANALYTICS.md)

## Use Cases

### 1. LLM Response Caching

```python
from victor.core.cache import SemanticCache

cache = SemanticCache(similarity_threshold=0.85)

# Cache LLM responses
async def get_llm_response(messages: List[Message]) -> str:
    # Check cache
    response = await cache.get_similar(messages)
    if response:
        return response.content

    # Call LLM
    response = await llm_provider.chat(messages)

    # Cache response
    await cache.put(messages, response)

    return response.content
```

### 2. Tool Result Caching

```python
from victor.core.cache import MultiLevelCache, CacheInvalidator

cache = MultiLevelCache(...)
invalidator = CacheInvalidator(cache=cache, enable_tagging=True)

async def execute_tool(tool_name: str, **kwargs) -> Any:
    # Generate cache key
    cache_key = f"{tool_name}:{hash(kwargs)}"

    # Check cache
    result = await cache.get(cache_key, namespace="tool")
    if result:
        return result

    # Execute tool
    result = await tool_executor.execute(tool_name, **kwargs)

    # Cache result
    await cache.set(cache_key, result, namespace="tool")
    invalidator.tag(cache_key, "tool", [tool_name, kwargs.get("category")])

    return result
```

### 3. File Analysis Caching

```python
from victor.core.cache import MultiLevelCache, CacheInvalidator

cache = MultiLevelCache(...)
invalidator = CacheInvalidator(cache=cache, enable_dependencies=True)

async def analyze_file(file_path: str) -> Dict[str, Any]:
    cache_key = f"analysis:{file_path}"

    # Check cache
    result = await cache.get(cache_key, namespace="analysis")
    if result:
        return result

    # Analyze file
    result = await analyzer.analyze(file_path)

    # Cache with dependency
    await cache.set(cache_key, result, namespace="analysis")
    invalidator.add_dependency(cache_key, "analysis", file_path)

    return result

# On file change
async def on_file_change(file_path: str):
    await invalidator.invalidate_dependents(file_path)
```

## Best Practices

### 1. Choose the Right Cache Type

- **Multi-Level Cache**: General purpose, balanced performance
- **Semantic Cache**: Natural language queries, LLM responses
- **Response Cache**: Simple key-value caching

### 2. Configure Appropriate TTLs

```python
# Short TTL for frequently changing data
await cache.set("result", value, ttl=60)  # 1 minute

# Medium TTL for balanced freshness
await cache.set("result", value, ttl=3600)  # 1 hour

# Long TTL for stable data
await cache.set("result", value, ttl=86400)  # 1 day
```

### 3. Use Namespaces

```python
# Good: Specific namespaces
await cache.set("result", value, namespace="tool.code_analysis")
await cache.set("result", value, namespace="llm.response")

# Avoid: Generic namespace
await cache.set("result", value, namespace="default")
```

### 4. Monitor Performance

```python
# Regularly check statistics
stats = analytics.get_comprehensive_stats()

if stats['hit_rate'] < 0.5:
    logger.warning("Low hit rate, consider tuning")

if stats['latency']['avg_ms'] > 5.0:
    logger.warning("High latency, investigate")
```

### 5. Handle Errors Gracefully

```python
try:
    result = await cache.get(key, namespace="tool")
    if result:
        return result
except Exception as e:
    logger.error(f"Cache error: {e}")

# Fall back to computation
result = await compute_result()
return result
```

## Troubleshooting

### Low Hit Rate

**Symptoms**: Hit rate < 50%

**Solutions**:
1. Increase cache size
2. Increase TTL
3. Enable cache warming
4. Improve cache key design

### High Memory Usage

**Symptoms**: Memory consumption growing

**Solutions**:
1. Reduce cache size
2. Check TTL settings
3. Monitor with analytics
4. Use L2 with disk persistence

### Slow Performance

**Symptoms**: Cache latency > 5ms

**Solutions**:
1. Use in-memory L1 cache
2. Optimize serialization
3. Increase L1 hit rate
4. Check disk I/O for L2

## Contributing

Contributions are welcome! Please see [Contributing Guide](../contributing/index.md) for guidelines.

## License

Apache License 2.0 - see LICENSE file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/victor-ai/issues)
- **Documentation**: [Victor AI Docs](https://docs.victor-ai.com)
- **Discord**: [Victor AI Community](https://discord.gg/victor-ai)
