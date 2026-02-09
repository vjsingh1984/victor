# Performance Tuning Guide - Part 2

**Part 2 of 2:** Production Tuning, Best Practices, Troubleshooting, and Performance Checklist

---

## Navigation

- [Part 1: Optimization Techniques](part-1-optimization-techniques.md)
- **[Part 2: Production & Best Practices](#)** (Current)
- [**Complete Guide](../PERFORMANCE_TUNING.md)**

---
## Production Tuning

Optimize for production workloads.

### Production Configuration

```python
# Production settings
settings = Settings()

# Performance
settings.lazy_loading_enabled = True
settings.parallel_execution_enabled = True
settings.max_parallel_workers = 8

# Caching
settings.tool_selection_cache_enabled = True
settings.tool_selection_cache_size = 1000
settings.response_cache_enabled = True
settings.response_cache_size = 2000

# Memory
settings.memory_limit_mb = 4000
settings.episodic_memory_max_episodes = 2000
settings.semantic_memory_max_facts = 10000

# Connection pooling
settings.connection_pool_size = 20
settings.connection_pool_max_overflow = 10
```text

### Provider-Specific Tuning

```python
# Anthropic
settings.anthropic_timeout = 30
settings.anthropic_max_retries = 3
settings.anthropic_concurrent_requests = 10

# OpenAI
settings.openai_timeout = 30
settings.openai_max_retries = 3
settings.openai_concurrent_requests = 10

# Local providers (Ollama, vLLM)
settings.local_provider_timeout = 120
settings.local_provider_batch_size = 8
```

### High-Throughput Configuration

```python
# For high-throughput scenarios
settings.max_parallel_workers = 16
settings.connection_pool_size = 50
settings.tool_selection_cache_size = 2000
settings.response_cache_size = 5000

# Disable expensive features
settings.enable_episodic_memory = False  # If not needed
settings.enable_semantic_memory = False
```text

### Low-Latency Configuration

```python
# For low-latency scenarios
settings.lazy_loading_enabled = True
settings.parallel_execution_enabled = True
settings.tool_selection_cache_size = 2000

# Use faster provider
settings.provider = "anthropic"
settings.model = "claude-3-haiku"  # Faster model

# Disable non-critical features
settings.enable_hierarchical_planning = False
```

## Best Practices

### 1. Profile Before Optimizing

```python
# Always profile first
profiler = cProfile.Profile()
profiler.enable()

# ... run operations ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.print_stats(20)

# Identify bottlenecks before optimizing
```text

### 2. Use Appropriate Caching

```python
# Cache frequently accessed data
settings.tool_selection_cache_enabled = True

# Don't cache rarely used data
if usage_pattern == "rarely_repeated":
    settings.response_cache_enabled = False
```

### 3. Monitor Performance

```python
# Continuous monitoring
async def monitor_and_alert():
    metrics = await orchestrator.get_metrics()

    if metrics['avg_latency'] > threshold:
        send_alert("High latency detected")

    if metrics['error_rate'] > 0.05:  # 5%
        send_alert("High error rate detected")
```text

### 4. Scale Resources

```python
# Scale based on load
current_load = await get_current_load()

if current_load > 0.8:
    settings.max_parallel_workers = min(
        settings.max_parallel_workers * 2,
        32  # Max limit
    )
```

### 5. Use Appropriate Providers

```python
# Choose provider based on requirements
if priority == "speed":
    settings.provider = "anthropic"
    settings.model = "claude-3-haiku"  # Fastest
elif priority == "quality":
    settings.provider = "anthropic"
    settings.model = "claude-3-opus"  # Best quality
elif priority == "cost":
    settings.provider = "openai"
    settings.model = "gpt-3.5-turbo"  # Lowest cost
```text

## Troubleshooting

### High Memory Usage

**Problem**: Memory usage growing continuously.

**Solutions**:
1. **Reduce cache sizes**: Lower cache limits
2. **Enable cleanup**: Regular memory cleanup
3. **Check leaks**: Profile for memory leaks
4. **Adjust limits**: Lower max_episodes/max_facts

```python
# Reduce memory footprint
settings.tool_selection_cache_size = 250
settings.episodic_memory_max_episodes = 500

# Enable cleanup
asyncio.create_task(periodic_cleanup())
```

### Slow Response Times

**Problem**: Responses taking too long.

**Solutions**:
1. **Enable caching**: Cache frequent queries
2. **Use parallel execution**: Execute independent operations in parallel
3. **Optimize tool selection**: Reduce max_tools
4. **Use faster provider**: Switch to faster model

```python
# Improve response time
settings.tool_selection_cache_enabled = True
settings.parallel_execution_enabled = True
settings.tool_selection_max_tools = 10
settings.model = "claude-3-haiku"
```text

### High CPU Usage

**Problem**: CPU usage consistently high.

**Solutions**:
1. **Reduce parallelism**: Lower max_parallel_workers
2. **Enable lazy loading**: Defer component initialization
3. **Optimize loops**: Check for inefficient loops
4. **Use batching**: Batch operations when possible

```python
# Reduce CPU usage
settings.max_parallel_workers = 2
settings.lazy_loading_enabled = True
```

### Cache Misses

**Problem**: Low cache hit rate.

**Solutions**:
1. **Increase cache size**: Larger cache
2. **Increase TTL**: Longer cache lifetime
3. **Check patterns**: Analyze access patterns
4. **Optimize keys**: Improve cache key generation

```python
# Improve cache hit rate
settings.tool_selection_cache_size = 1000
settings.tool_selection_cache_ttl = 7200  # 2 hours
```text

## Performance Checklist

Use this checklist to ensure optimal performance:

### Initial Setup
- [ ] Enable lazy loading
- [ ] Configure parallel execution
- [ ] Set appropriate cache sizes
- [ ] Configure memory limits
- [ ] Choose optimal provider/model

### Monitoring
- [ ] Set up performance monitoring
- [ ] Define alert thresholds
- [ ] Track key metrics
- [ ] Regular performance reviews

### Optimization
- [ ] Profile bottlenecks
- [ ] Optimize slow operations
- [ ] Cache frequently accessed data
- [ ] Scale resources as needed

### Maintenance
- [ ] Regular cache cleanup
- [ ] Memory consolidation
- [ ] Review and adjust settings
- [ ] Update dependencies

## Additional Resources

- [API Reference](../api/NEW_CAPABILITIES_API.md)
- [User Guide](../user-guide/index.md)
- [Profiling Guide](../observability/performance_monitoring.md)
- [Production Deployment](../operations/deployment/enterprise.md)
- [Performance Benchmarks](../performance/benchmark_results.md)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
