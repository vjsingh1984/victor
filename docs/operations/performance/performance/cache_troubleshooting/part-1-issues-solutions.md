# Cache Troubleshooting - Part 1

**Part 1 of 2:** Common Issues and Solutions

---

## Navigation

- **[Part 1: Issues & Solutions](#)** (Current)
- [Part 2: Debug & Emergency Procedures](part-2-debug-emergency.md)
- [**Cache Performance Guide**](../cache_performance.md)

---

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
```text

### Get Detailed Metrics

```python
from victor.tools.caches import AdvancedCacheManager
from victor.config import load_settings
import json

settings = load_settings()
cache = AdvancedCacheManager.from_settings(settings)
metrics = cache.get_metrics()

print(json.dumps(metrics, indent=2))
```text

Output:
```json
{
  "combined": {
    "hits": 8432,
    "misses": 1523,
    "hit_rate": 0.847,
    "total_entries": 5234
  },
  "in_memory_cache": {
    "enabled": true,
    "max_size": 1000,
    "current_size": 847,
    "ttl_seconds": 300
  },
  "persistent_cache": {
    "enabled": true,
    "backend": "redis",
    "total_entries": 4387,
    "size_bytes": 52340000
  },
  "adaptive_ttl": {
    "enabled": true,
    "ttl_adjustments": 234,
    "avg_ttl_seconds": 287.3
  },
  "memory_usage_mb": 42.3
}
```text

---

## Common Issues and Solutions

### Issue 1: Low Cache Hit Rate

**Symptoms:**
- Hit rate below 40%
- Frequent cache misses
- Slow response times

**Diagnosis:**
```python
from victor.tools.caches import AdvancedCacheManager

cache = AdvancedCacheManager.from_settings(settings)
metrics = cache.get_metrics()

print(f"Hit rate: {metrics['combined']['hit_rate']:.1%}")
print(f"Total misses: {metrics['combined']['misses']}")
```text

**Possible Causes:**

1. **Cache key mismatch** - Keys not consistent across requests
2. **TTL too short** - Items expiring before reuse
3. **Insufficient cache size** - Cache evicting useful items

**Solutions:**

```python
# Solution 1: Check cache key generation
from victor.tools.caches import CacheKeyGenerator

key_gen = CacheKeyGenerator()
key = key_gen.generate("my_function", arg1="value", arg2=123)
print(f"Cache key: {key}")

# Solution 2: Increase TTL
cache.configure(
    in_memory_ttl=600,  # 10 minutes instead of 5
    persistent_ttl=3600  # 1 hour instead of 30 minutes
)

# Solution 3: Increase cache size
cache.configure(
    in_memory_max_size=2000,  # Double the size
)
```text

### Issue 2: Cache Not Persisting

**Symptoms:**
- Persistent cache shows 0 entries
- Data lost on restart
- Redis shows no data

**Possible Causes:**

1. **Redis connection failed**
2. **Persistence disabled**

**Solutions:**

```python
# Solution 1: Check Redis connection
import redis

try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("✅ Redis is reachable")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")

# Solution 2: Enable persistent cache
cache = AdvancedCacheManager(
    in_memory_max_size=1000,
    persistent_enabled=True,
    persistent_backend="redis",
    persistent_host="localhost",
    persistent_port=6379,
)
```text

### Issue 3: High Memory Usage

**Symptoms:**
- Memory usage > 100 MB
- OOM errors
- System slowdown

**Solutions:**

```python
# Solution 1: Reduce cache size
cache.configure(
    in_memory_max_size=500,  # Reduce from 1000
)

# Solution 2: Enable aggressive eviction
cache.configure(
    eviction_policy="lru",  # Use LRU instead of LFU
    in_memory_ttl=180,  # Shorter TTL
)

# Solution 3: Monitor and clear
if metrics['memory_usage_mb'] > 100:
    cache.clear_in_memory()
    print("Cleared in-memory cache")
```text

### Issue 4: Adaptive TTL Not Adjusting

**Symptoms:**
- TTL adjustments count is 0
- All items have same TTL

**Solutions:**

```python
# Solution 1: Enable adaptive TTL
cache = AdvancedCacheManager(
    adaptive_ttl_enabled=True,
    adaptive_ttl_min_seconds=60,
    adaptive_ttl_max_seconds=3600,
)

# Solution 2: Adjust sensitivity
cache.configure(
    adaptive_ttl_sensitivity=0.5,  # More aggressive adjustments
)
```text

### Issue 5: Cache Corruption

**Symptoms:**
- Deserialization errors
- Inconsistent data
- Panic errors

**Solutions:**

```python
# Solution 1: Clear corrupted cache
cache.clear_all()
print("Cleared all caches")

# Solution 2: Rebuild cache
cache.rebuild()
print("Rebuilt cache from source")

# Solution 3: Verify integrity
if cache.verify_integrity():
    print("✅ Cache integrity verified")
else:
    print("❌ Cache corruption detected")
    cache.clear_all()
```text

### Issue 6: Slow Cache Performance

**Symptoms:**
- Cache operations take >100ms
- Latency spikes

**Solutions:**

```python
# Solution 1: Use local Redis
cache.configure(
    persistent_host="localhost",  # Use local Redis
    persistent_port=6379,
)

# Solution 2: Enable compression
cache.configure(
    compression_enabled=True,
    compression_threshold=1024,  # Compress objects >1KB
)
```text

### Issue 7: Cache Warming Issues

**Symptoms:**
- Poor performance after restart
- Cold start problems

**Solutions:**

```python
# Solution 1: Warm cache on startup
async def warm_cache(cache):
    """Preload common data into cache."""
    common_keys = [
        "config:settings",
        "user:permissions",
        "routes:patterns",
    ]

    for key in common_keys:
        value = await fetch_from_source(key)
        cache.set(key, value, ttl=3600)

# Solution 2: Extend persistent TTL
cache.configure(
    persistent_ttl=86400,  # 24 hours
)
```text

---

## See Also

- [Cache Performance Guide](../cache_performance.md)
- [Cache Architecture](../cache_architecture.md)
- [Debug Mode & Emergency Procedures](part-2-debug-emergency.md)
- [Documentation Home](../../../README.md)


**Reading Time:** 8 min
**Last Updated:** February 09, 2026**
