# Performance Optimization Guide - Part 3

**Part 3 of 3:** Monitoring and Configuration Guidelines

---

## Navigation

- [Part 1: Improvements & Benchmarking](part-1-improvements-benchmarking.md)
- [Part 2: Troubleshooting & Optimizations](part-2-troubleshooting-optimizations.md)
- **[Part 3: Monitoring & Guidelines](#)** (Current)
- [**Complete Guide**](../optimization_guide.md)

---

### Overview

Request batching automatically groups similar requests to reduce overhead:
- Batches are flushed when they reach max size OR timeout expires
- Priority-based execution (HIGH, MEDIUM, LOW)
- Concurrent batch execution with semaphore limiting

### Configuration

```yaml
request_batching_enabled: true
request_batch_max_size: 10
request_batch_timeout: 0.1  # 100ms
request_batch_max_concurrent: 5
```

### Usage

```python
from victor.core.batching import get_llm_batcher, BatchPriority

# Get global LLM batcher
batcher = get_llm_batcher()

# Submit requests (automatically batched)
result1 = await batcher.submit(
    model="claude-3",
    prompt="What is 2+2?",
    priority=BatchPriority.HIGH
)

result2 = await batcher.submit(
    model="claude-3",
    prompt="What is 3+3?",
    priority=BatchPriority.MEDIUM
)

# Results are automatically batched and executed together
```

### Performance Characteristics

**Sequential Execution (5 requests, 100ms each):**
- Total time: 500ms
- Throughput: 10 req/sec

**Batched Execution (batch_size=5, timeout=100ms):**
- Total time: ~100ms
- Throughput: 50 req/sec
- **Speedup: 5x**

## 3. Hot Path Optimizations

### Optimized JSON Serialization

Uses `orjson` for 3-5x faster JSON serialization:

```python
from victor.core.optimizations import json_dumps, json_loads

# Serialize (3-5x faster than standard json)
json_str = json_dumps(obj)

# Deserialize
obj = json_loads(json_str)
```

**Benchmark Results:**
- Standard `json.dumps()`: ~0.45ms
- Optimized `json_dumps()`: ~0.12ms
- **Speedup: 3.75x**

### Lazy Imports

Defer importing heavy modules until first use:

```python
from victor.core.optimizations import lazy_import

# Heavy modules are loaded on first access
numpy = lazy_import("numpy")
pandas = lazy_import("pandas")

# Use normally - module is imported here
arr = numpy.array([1, 2, 3])
```

**Benefits:**
- 20-30% faster startup time
- Reduced memory footprint
- Faster test execution

### Memoization

Cache function results with TTL:

```python
from victor.core.optimizations import ThreadSafeMemoized

@ThreadSafeMemoized(ttl=3600, max_size=128)
def expensive_computation(x, y):
    # Expensive calculation
    return x + y
```

## 4. Performance Monitoring

### Overview

Comprehensive performance monitoring tracks:
- Operation timing with percentiles (p50, p95, p99)
- Error rates
- Cache hit rates
- Hot path identification

### Usage

```python
from victor.core.monitoring import get_performance_monitor, OperationCategory

# Get global monitor
monitor = get_performance_monitor()

# Track an operation
with monitor.track("database_query", OperationCategory.GENERAL):
    result = database.query(...)

# Get statistics
stats = monitor.get_stats("database_query")
print(f"Average time: {stats['avg_time']:.4f}s")
print(f"P95 time: {stats['p95_time']:.4f}s")

# Get hot operations
hot_ops = monitor.get_hot_operations(min_count=10, threshold=1.0)
```

---

# CONFIGURATION

## Enable/Disable Optimizations

In `~/.victor/profiles.yaml`:

```yaml
# Performance optimization settings
response_cache_enabled: true
request_batching_enabled: true
hot_path_optimizations_enabled: true
performance_monitoring_enabled: true

# JSON optimization
use_orjson: true

# Lazy imports
lazy_imports_enabled: true
```

## Environment Variables

```bash
# Disable response caching
export VICTOR_RESPONSE_CACHE_ENABLED=false

# Enable verbose performance logging
export VICTOR_PERFORMANCE_LOG_SLOW_OPERATIONS=true
export VICTOR_PERFORMANCE_SLOW_THRESHOLD=0.5

# Use orjson for JSON
export VICTOR_USE_ORJSON=true
```

---

# BENCHMARKING

## Running Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks/test_performance_optimizations.py -v -m benchmark

# Run specific benchmark
pytest tests/benchmarks/test_performance_optimizations.py::TestJSONSerializationBenchmarks -v

# Run with coverage
pytest tests/benchmarks/ --cov=victor.core --cov-report=html
```

## Expected Results

| Benchmark | Metric | Target |
|-----------|--------|--------|
| JSON Serialization | Speedup vs standard | 3-5x |
| Response Cache Hit | Latency | < 0.1ms |
| Request Batching | Overhead reduction | 20-40% |
| Overall System | Combined improvement | 30-50% |

---

# BEST PRACTICES

## When to Use Caching

**Enable caching for:**
- Read-heavy workloads
- Repeated queries
- Expensive operations
- Reference data lookups

**Disable caching for:**
- Write-heavy workloads
- Unique queries
- Real-time data
- Frequently changing data

## When to Use Batching

**Enable batching for:**
- High-volume operations
- Independent requests
- Non-latency-sensitive operations

**Disable batching for:**
- Low-latency requirements
- Dependent requests
- Streaming operations

## Performance Tuning

1. **Start with defaults** - Default settings work well for most cases
2. **Monitor metrics** - Use performance monitoring to identify bottlenecks
3. **Adjust thresholds** - Tune cache sizes, timeouts, and thresholds based on workload
4. **Profile regularly** - Run benchmarks regularly to catch regressions

---

# REFERENCES

## Advanced Optimizations (New)

- **Response Cache**: `/victor/core/cache/response_cache.py`
- **Request Batching**: `/victor/core/batching/request_batcher.py`
- **Hot Path Optimizations**: `/victor/core/optimizations/hot_path_optimizations.py`
- **Performance Monitoring**: `/victor/core/monitoring/performance_monitor.py`
- **Configuration**: `/victor/config/settings.py`
- **Benchmarks**: `/tests/benchmarks/test_performance_optimizations.py`

## Legacy Optimizations (Previous)

- **Batch Tool Execution**: `/victor/agent/tool_executor.py` - `ToolExecutor.execute_batch()`
- **Context Compaction**: `/victor/agent/coordinators/compaction_strategies.py`
- **Prompt Building**: `/victor/agent/coordinators/prompt_coordinator.py`

---

**Last Updated:** February 01, 2026
**Reading Time:** 10 minutes
