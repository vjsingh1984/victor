# Cache Troubleshooting Guide

**Track 5.3: Advanced Caching in Production**

This guide helps you diagnose and resolve common issues with Victor's advanced caching system.

## Quick Diagnostics

### Health Check Script

Run this script to check cache health:

```python
#!/usr/bin/env python3
"""Cache health check script."""

from victor.tools.caches import AdvancedCacheManager
from victor.config import load_settings
import sys

def check_cache_health():
    """Check cache health and report issues."""
    settings = load_settings()
    cache = AdvancedCacheManager.from_settings(settings)

    if not cache.enabled:
        print("❌ CRITICAL: Caching is disabled")
        return False

    metrics = cache.get_metrics()

    # Check hit rate
    hit_rate = metrics['combined']['hit_rate']
    if hit_rate < 0.4:
        print(f"❌ WARNING: Low hit rate ({hit_rate:.1%})")
    elif hit_rate < 0.6:
        print(f"⚠️  INFO: Moderate hit rate ({hit_rate:.1%})")
    else:
        print(f"✅ OK: Good hit rate ({hit_rate:.1%})")

    # Check persistent cache
    if cache._persistent_cache:
        persistent = metrics['persistent_cache']
        if persistent.get('total_entries', 0) == 0:
            print("⚠️  INFO: Persistent cache is empty (first run?)")
        else:
            print(f"✅ OK: Persistent cache has {persistent['total_entries']} entries")

    # Check adaptive TTL
    if cache._adaptive_ttl_cache:
        adaptive = metrics['adaptive_ttl']
        adjustments = adaptive.get('ttl_adjustments', 0)
        if adjustments == 0:
            print("⚠️  INFO: Adaptive TTL hasn't made any adjustments yet")
        else:
            print(f"✅ OK: Adaptive TTL made {adjustments} adjustments")

    # Check memory usage
    memory_mb = metrics.get('memory_usage_mb', 0)
    if memory_mb > 100:
        print(f"❌ WARNING: High memory usage ({memory_mb:.1f} MB)")
    else:
        print(f"✅ OK: Memory usage ({memory_mb:.1f} MB)")

    return True

if __name__ == "__main__":
    success = check_cache_health()
    sys.exit(0 if success else 1)
```

### Get Detailed Metrics

```python
from victor.tools.caches import AdvancedCacheManager
from victor.config import load_settings
import json

settings = load_settings()
cache = AdvancedCacheManager.from_settings(settings)
metrics = cache.get_metrics()

print(json.dumps(metrics, indent=2, default=str))
```

## Common Issues and Solutions

### Issue 1: Low Cache Hit Rate

**Symptoms:**
- Hit rate < 50%
- High latency
- Frequent cache misses

**Diagnosis:**

```python
# Check cache size utilization
metrics = cache.get_metrics()
total_entries = metrics['combined']['total_entries']
cache_size = settings.tool_selection_cache_size

utilization = total_entries / cache_size
print(f"Cache utilization: {utilization:.1%}")

if utilization > 0.9:
    print("Cache is nearly full - consider increasing size")
```

**Solutions:**

1. **Increase cache size:**
   ```yaml
   VICTOR_CACHE_SIZE: 2000  # Increase from 1000
   ```

2. **Increase TTL:**
   ```yaml
   VICTOR_TOOL_SELECTION_CACHE_QUERY_TTL: 7200  # 2 hours
   VICTOR_TOOL_SELECTION_CACHE_CONTEXT_TTL: 600  # 10 minutes
   ```

3. **Enable adaptive TTL:**
   ```yaml
   VICTOR_ADAPTIVE_TTL_ENABLED: true
   ```

4. **Enable multi-level cache:**
   ```yaml
   VICTOR_MULTI_LEVEL_CACHE_ENABLED: true
   ```

**Verification:**

```python
# Monitor hit rate over 24 hours
import time

hit_rates = []
for i in range(24):
    time.sleep(3600)  # Wait 1 hour
    metrics = cache.get_metrics()
    hit_rate = metrics['combined']['hit_rate']
    hit_rates.append(hit_rate)
    print(f"Hour {i+1}: Hit rate = {hit_rate:.1%}")

# Check trend
if hit_rates[-1] > hit_rates[0] * 1.2:
    print("Hit rate improving - solution working!")
```

### Issue 2: Cache Not Persisting

**Symptoms:**
- Empty cache after restart
- Cache file doesn't exist
- Slow cold starts

**Diagnosis:**

```bash
# Check if cache file exists
kubectl exec -it deployment/victor-ai -- ls -lh /app/.victor/cache/

# Check file permissions
kubectl exec -it deployment/victor-ai -- ls -la /app/.victor/cache/

# Check mount point
kubectl exec -it deployment/victor-ai -- df -h /app/.victor/cache
```

**Solutions:**

1. **Verify persistent cache is enabled:**
   ```yaml
   VICTOR_PERSISTENT_CACHE_ENABLED: "true"
   VICTOR_PERSISTENT_CACHE_PATH: "/app/.victor/cache/tool_selection_cache.db"
   ```

2. **Check PVC is mounted:**
   ```yaml
   volumeMounts:
   - name: victor-cache
     mountPath: /app/.victor/cache

   volumes:
   - name: victor-cache
     persistentVolumeClaim:
       claimName: victor-cache-pvc
   ```

3. **Fix permissions:**
   ```yaml
   securityContext:
     runAsUser: 1000
     fsGroup: 1000
   ```

4. **Check disk space:**
   ```bash
   kubectl exec -it deployment/victor-ai -- df -h
   ```

**Verification:**

```python
# Write to cache
cache.put("test_key", ["tool1", "tool2"], namespace="query")

# Close and reopen cache
cache.close()

cache2 = AdvancedCacheManager.from_settings(settings)
result = cache2.get("test_key", namespace="query")

if result:
    print("✅ Cache persistence working!")
else:
    print("❌ Cache not persisting")
```

### Issue 3: High Memory Usage

**Symptoms:**
- OOMKilled pods
- High memory consumption
- Memory limits exceeded

**Diagnosis:**

```bash
# Check pod memory usage
kubectl top pod -l app=victor-ai -n victor-ai-prod

# Check memory limits
kubectl describe pod <pod-name> -n victor-ai-prod | grep -A 5 Limits
```

**Solutions:**

1. **Reduce cache size:**
   ```yaml
   VICTOR_CACHE_SIZE: 500  # Reduce from 1000
   ```

2. **Enable auto-compaction:**
   ```yaml
   VICTOR_PERSISTENT_CACHE_AUTO_COMPACT: "true"
   ```

3. **Reduce TTL:**
   ```yaml
   VICTOR_TOOL_SELECTION_CACHE_QUERY_TTL: 1800  # 30 minutes
   VICTOR_ADAPTIVE_TTL_MAX: 3600  # 1 hour
   ```

4. **Increase memory limit:**
   ```yaml
   resources:
     limits:
       memory: 4Gi  # Increase from 2Gi
   ```

**Verification:**

```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()

print(f"RSS memory: {memory_info.rss / 1024 / 1024:.1f} MB")
print(f"VMS memory: {memory_info.vms / 1024 / 1024:.1f} MB")

# Check cache memory contribution
metrics = cache.get_metrics()
cache_memory_mb = metrics.get('memory_usage_mb', 0)
print(f"Cache memory: {cache_memory_mb:.1f} MB")
```

### Issue 4: Adaptive TTL Not Adjusting

**Symptoms:**
- TTL not changing
- `ttl_adjustments` counter stuck at 0
- All entries have same TTL

**Diagnosis:**

```python
adaptive_metrics = metrics['adaptive_ttl']
print(f"TTL adjustments: {adaptive_metrics['ttl_adjustments']}")
print(f"TTL distribution: {adaptive_metrics.get('ttl_distribution', {})}")
```

**Solutions:**

1. **Verify adaptive TTL is enabled:**
   ```yaml
   VICTOR_ADAPTIVE_TTL_ENABLED: "true"
   ```

2. **Check adjustment threshold:**
   ```yaml
   VICTOR_ADAPTIVE_TTL_ADJUSTMENT_THRESHOLD: 5  # Lower to 3 for faster adaptation
   ```

3. **Verify cache is being accessed:**
   ```python
   # Check if cache is being used
   hits = metrics['basic_cache']['combined']['hits']
   misses = metrics['basic_cache']['combined']['misses']

   if hits + misses == 0:
       print("Cache not being used - no traffic")
   ```

4. **Check bounds:**
   ```yaml
   # If min and max are too close, no room for adjustment
   VICTOR_ADAPTIVE_TTL_MIN: 60  # 1 minute
   VICTOR_ADAPTIVE_TTL_MAX: 7200  # 2 hours (needs range)
   ```

**Verification:**

```python
# Force some cache accesses
for i in range(10):
    cache.put(f"test_key_{i}", ["tool1"])
    result = cache.get(f"test_key_{i}")
    assert result is not None

# Check if adjustments happened
metrics = cache.get_metrics()
adjustments = metrics['adaptive_ttl']['ttl_adjustments']
print(f"Adjustments after 10 accesses: {adjustments}")
```

### Issue 5: Cache Corruption

**Symptoms:**
- Cache errors in logs
- Invalid data returned
- SQLite database errors

**Diagnosis:**

```bash
# Check logs for cache errors
kubectl logs -f deployment/victor-ai -n victor-ai-prod | grep -i cache

# Check database integrity
kubectl exec -it deployment/victor-ai -- \
  sqlite3 /app/.victor/cache/tool_selection_cache.db "PRAGMA integrity_check;"
```

**Solutions:**

1. **Delete and recreate cache:**
   ```bash
   kubectl exec -it deployment/victor-ai -- \
     rm /app/.victor/cache/tool_selection_cache.db
   ```

2. **Export and reimport (if possible):**
   ```bash
   # Backup
   kubectl cp victor-ai-prod/deployment/victor-ai:/app/.victor/cache/tool_selection_cache.db \
     ./cache-backup.db

   # Recreate on local machine
   sqlite3 cache-backup.db "VACUUM;"

   # Restore
   kubectl cp ./cache-backup.db \
     victor-ai-prod/deployment/victor-ai:/app/.victor/cache/tool_selection_cache.db
   ```

3. **Disable persistent cache temporarily:**
   ```yaml
   VICTOR_PERSISTENT_CACHE_ENABLED: "false"
   ```

**Verification:**

```python
# Test cache operations
try:
    cache.put("corruption_test", ["tool1"])
    result = cache.get("corruption_test")
    assert result == ["tool1"]
    print("✅ Cache operations successful")
except Exception as e:
    print(f"❌ Cache error: {e}")
```

### Issue 6: Slow Cache Performance

**Symptoms:**
- Cache access latency > 10ms
- Overall latency not improving
- Cache lookups taking long

**Diagnosis:**

```python
import time

# Measure cache latency
start = time.time()
for i in range(1000):
    cache.get(f"key_{i % 100}")
end = time.time()

avg_latency_ms = (end - start) / 1000 * 1000
print(f"Average cache latency: {avg_latency_ms:.2f}ms")

if avg_latency_ms > 10:
    print("⚠️  Cache latency is high")
```

**Solutions:**

1. **Check disk I/O (persistent cache):**
   ```bash
   # Use fast SSD storage
   kubectl get storageclass fast-ssd

   # Check PVC storage class
   kubectl get pvc victor-cache-pvc -o jsonpath='{.spec.storageClassName}'
   ```

2. **Reduce cache size (smaller cache = faster):**
   ```yaml
   VICTOR_CACHE_SIZE: 500  # Reduce from 2000
   ```

3. **Disable persistent cache (use in-memory only):**
   ```yaml
   VICTOR_PERSISTENT_CACHE_ENABLED: "false"
   ```

4. **Check for lock contention:**
   ```python
   # Check if multiple threads are blocking
   import threading
   print(f"Active threads: {threading.active_count()}")
   ```

**Verification:**

```python
# Benchmark cache performance
import timeit

def benchmark_cache():
    setup = """
from victor.tools.caches import AdvancedCacheManager
from victor.config import load_settings
settings = load_settings()
cache = AdvancedCacheManager.from_settings(settings)
# Warm up cache
for i in range(100):
    cache.put(f"key_{i}", ["tool1"])
    """
    stmt = "cache.get('key_50')"
    iterations = 10000

    time_taken = timeit.timeit(stmt, setup=setup, number=iterations)
    avg_latency_ms = (time_taken / iterations) * 1000

    print(f"Average latency: {avg_latency_ms:.3f}ms")
    print(f"Throughput: {iterations / time_taken:.0f} ops/sec")

benchmark_cache()
```

### Issue 7: Cache Warming Issues

**Symptoms:**
- Low hit rate on startup
- Takes long to reach steady state
- Predictive warming not working

**Diagnosis:**

```python
# Check cache warming status
if cache._predictive_warmer:
    stats = cache._predictive_warmer.get_statistics()
    print(f"Patterns learned: {stats['patterns_learned']}")
    print(f"Predictions made: {stats['predictions_made']}")
    print(f"Prediction accuracy: {stats['accuracy']:.1%}")
else:
    print("Predictive warming not enabled")
```

**Solutions:**

1. **Enable persistent cache (best solution):**
   ```yaml
   VICTOR_PERSISTENT_CACHE_ENABLED: "true"
   ```

2. **Enable predictive warming:**
   ```yaml
   VICTOR_PREDICTIVE_WARMING_ENABLED: "true"
   VICTOR_PREDICTIVE_WARMING_MAX_PATTERNS: 100
   ```

3. **Preload cache manually:**
   ```python
   # Load common queries at startup
   common_queries = [
       ("read file", ["read_file"]),
       ("search code", ["semantic_code_search"]),
       ("list files", ["list_directory"]),
   ]

   for query, tools in common_queries:
       cache.put_query(query, tools)
   ```

4. **Increase initial TTL:**
   ```yaml
   VICTOR_ADAPTIVE_TTL_INITIAL: 7200  # 2 hours
   ```

**Verification:**

```python
# Monitor warm-up time
import time

start_time = time.time()
target_hit_rate = 0.60

while True:
    time.sleep(60)  # Check every minute
    metrics = cache.get_metrics()
    hit_rate = metrics['combined']['hit_rate']

    elapsed = time.time() - start_time
    print(f"[{elapsed:.0f}s] Hit rate: {hit_rate:.1%}")

    if hit_rate >= target_hit_rate:
        print(f"✅ Warmed up in {elapsed:.0f}s")
        break

    if elapsed > 600:  # 10 minutes timeout
        print("⚠️  Warm-up taking longer than expected")
        break
```

## Debug Mode

Enable debug logging for detailed cache information:

```yaml
- name: VICTOR_LOG_LEVEL
  value: "DEBUG"
- name: VICTOR_CACHE_DEBUG
  value: "true"
```

Check logs:

```bash
kubectl logs -f deployment/victor-ai -n victor-ai-prod | grep -i cache
```

## Performance Baselines

Use these baselines to detect anomalies:

| Metric | Expected | Warning | Critical |
|--------|----------|---------|----------|
| Hit rate | 60-80% | < 50% | < 40% |
| Cache latency | < 5ms | > 10ms | > 20ms |
| Memory usage | 1-5 MB | > 50 MB | > 100 MB |
| TTL adjustments | > 0 | = 0 (after 1h) | N/A |
| Persistent entries | > 100 | = 0 | N/A |

## Emergency Procedures

### Disable Caching Immediately

```bash
kubectl set env deployment/victor-ai \
  VICTOR_TOOL_SELECTION_CACHE_ENABLED=false \
  -n victor-ai-prod
```

### Clear Cache

```bash
kubectl exec -it deployment/victor-ai -n victor-ai-prod -- \
  rm -rf /app/.victor/cache/

# Or restart pods (clears in-memory cache)
kubectl rollout restart deployment/victor-ai -n victor-ai-prod
```

### Rollback Configuration

```bash
# Rollback to previous deployment
kubectl rollout undo deployment/victor-ai -n victor-ai-prod

# Verify rollback
kubectl get pods -n victor-ai-prod
```

## Getting Help

If issues persist after troubleshooting:

1. **Collect diagnostics:**
   ```bash
   # Export metrics
   kubectl exec -it deployment/victor-ai -n victor-ai-prod -- \
     python -c "from victor.tools.caches import AdvancedCacheManager; from victor.config import load_settings; import json; cache = AdvancedCacheManager.from_settings(load_settings()); print(json.dumps(cache.get_metrics(), indent=2, default=str))" > cache-metrics.json

   # Export logs
   kubectl logs deployment/victor-ai -n victor-ai-prod > victor-logs.txt

   # Export cache database
   kubectl cp victor-ai-prod/deployment/victor-ai:/app/.victor/cache/tool_selection_cache.db \
     ./cache-diagnostics.db
   ```

2. **Check documentation:**
   - [Production Caching Guide](production_caching_guide.md)
   - [Cache Tuning Guide](cache_tuning_guide.md)

3. **Open issue:**
   - GitHub: https://github.com/victor-ai/victor/issues
   - Include: cache-metrics.json, victor-logs.txt

## Conclusion

Most cache issues can be resolved by:
1. Checking configuration (enabled vs. disabled)
2. Verifying persistence (PVC mounted, permissions)
3. Adjusting parameters (size, TTL)
4. Monitoring metrics (hit rate, latency)

When in doubt, start simple:
- Enable basic cache + persistent cache
- Monitor for 24-48 hours
- Add adaptive TTL if hit rate < 70%
- Add multi-level cache only if needed
