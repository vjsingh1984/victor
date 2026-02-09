# Cache Troubleshooting - Part 2

**Part 2 of 2:** Debug Mode and Emergency Procedures

---

## Navigation

- [Part 1: Issues & Solutions](part-1-issues-solutions.md)
- **[Part 2: Debug & Emergency Procedures](#)** (Current)
- [**Cache Performance Guide**](../cache_performance.md)

---

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
```text

### Debug Output Format

When debug mode is enabled, cache operations log:

```text
[DEBUG] [CACHE] GET tool_selection:llm:anthropic:claude-3-5-sonnet-20241022 -> HIT (in 2.3ms)
[DEBUG] [CACHE] GET tool_selection:llm:anthropic:claude-3-5-sonnet-20241022 -> MISS (in 1.1ms)
[DEBUG] [CACHE] SET tool_selection:llm:anthropic:claude-3-5-sonnet-20241022 -> TTL=300s
[DEBUG] [CACHE] PERSISTENT_WRITE -> SUCCESS (in 8.4ms)
[DEBUG] [CACHE] ADAPTIVE_TTL -> Adjusted from 300s to 450s (hit rate increased)
```

### Debug Commands

```python
# Python debug utilities
from victor.tools.caches import AdvancedCacheManager
from victor.config import load_settings

cache = AdvancedCacheManager.from_settings(load_settings())

# Enable debug mode programmatically
cache.enable_debug_mode()

# Print all cache keys
print("All cache keys:")
for key in cache.get_all_keys():
    print(f"  - {key}")

# Get cache statistics
stats = cache.get_statistics()
print(f"Total operations: {stats['total_operations']}")
print(f"Breakdown: {stats['operation_breakdown']}")
```text

## Performance Baselines

Use these baselines to detect anomalies:

| Metric | Expected | Warning | Critical |
|--------|----------|---------|----------|
| Hit rate | 60-80% | < 50% | < 40% |
| Cache latency | < 5ms | > 10ms | > 20ms |
| Memory usage | 1-5 MB | > 50 MB | > 100 MB |
| TTL adjustments | > 0 | = 0 (after 1h) | N/A |
| Persistent entries | > 100 | = 0 | N/A |

### Baseline Testing

```python
import time
from victor.tools.caches import AdvancedCacheManager

cache = AdvancedCacheManager.from_settings(settings)

# Test latency
def test_latency(cache, key, value):
    start = time.time()
    cache.set(key, value)
    set_time = (time.time() - start) * 1000

    start = time.time()
    result = cache.get(key)
    get_time = (time.time() - start) * 1000

    return {"set": set_time, "get": get_time}

# Run 100 iterations
latencies = [test_latency(cache, f"test_{i}", f"value_{i}") for i in range(100)]

avg_set = sum(l["set"] for l in latencies) / 100
avg_get = sum(l["get"] for l in latencies) / 100

print(f"Average SET latency: {avg_set:.2f}ms")
print(f"Average GET latency: {avg_get:.2f}ms")
```text

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
```text

### Rollback Configuration

```bash
# Rollback to previous deployment
kubectl rollout undo deployment/victor-ai -n victor-ai-prod

# Verify rollback
kubectl get pods -n victor-ai-prod
```

### Emergency Diagnostics Script

```python
#!/usr/bin/env python3
"""Emergency cache diagnostics script."""

from victor.tools.caches import AdvancedCacheManager
from victor.config import load_settings
import json
import sys

def emergency_diagnostics():
    """Run emergency diagnostics."""
    print("=" * 60)
    print("EMERGENCY CACHE DIAGNOSTICS")
    print("=" * 60)

    try:
        settings = load_settings()
        cache = AdvancedCacheManager.from_settings(settings)

        # Check 1: Cache enabled?
        print("\n[1] Cache Status:")
        if cache.enabled:
            print("  ✅ Cache is ENABLED")
        else:
            print("  ❌ Cache is DISABLED")

        # Check 2: Get metrics
        print("\n[2] Cache Metrics:")
        metrics = cache.get_metrics()
        print(json.dumps(metrics, indent=2, default=str))

        # Check 3: Memory usage
        print("\n[3] Memory Usage:")
        memory_mb = metrics.get('memory_usage_mb', 0)
        if memory_mb > 100:
            print(f"  ❌ HIGH: {memory_mb:.1f} MB")
        elif memory_mb > 50:
            print(f"  ⚠️  WARNING: {memory_mb:.1f} MB")
        else:
            print(f"  ✅ OK: {memory_mb:.1f} MB")

        # Check 4: Hit rate
        print("\n[4] Hit Rate:")
        hit_rate = metrics['combined']['hit_rate']
        if hit_rate < 0.4:
            print(f"  ❌ CRITICAL: {hit_rate:.1%}")
        elif hit_rate < 0.6:
            print(f"  ⚠️  WARNING: {hit_rate:.1%}")
        else:
            print(f"  ✅ OK: {hit_rate:.1%}")

        # Check 5: Persistent cache
        print("\n[5] Persistent Cache:")
        if cache._persistent_cache:
            persistent = metrics['persistent_cache']
            entries = persistent.get('total_entries', 0)
            if entries == 0:
                print("  ⚠️  Empty (first run?)")
            else:
                print(f"  ✅ {entries} entries")
        else:
            print("  ❌ Not enabled")

        # Recommendations
        print("\n[6] Recommendations:")
        if hit_rate < 0.5:
            print("  → Consider increasing TTL")
            print("  → Check cache key consistency")
        if memory_mb > 100:
            print("  → Clear cache immediately")
            print("  → Reduce max_size setting")
        if not cache._persistent_cache:
            print("  → Enable persistent cache")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = emergency_diagnostics()
    sys.exit(0 if success else 1)
```text

## Getting Help

If issues persist after troubleshooting:

### 1. Collect Diagnostics

```bash
# Export metrics
kubectl exec -it deployment/victor-ai -n victor-ai-prod -- \
  python -c "from victor.tools.caches import AdvancedCacheManager; from victor.config import load_settings; import json; cache = AdvancedCacheManager.from_settings(load_settings()); print(json.dumps(cache.get_metrics(), indent=2, default=str))" > cache-metrics.json

# Export logs
kubectl logs deployment/victor-ai -n victor-ai-prod > victor-logs.txt

# Export cache database
kubectl cp victor-ai-prod/deployment/victor-ai:/app/.victor/cache/tool_selection_cache.db \
  ./cache-diagnostics.db
```text

### 2. Check Documentation

- [Production Caching Guide](../production_caching_guide.md)
- [Cache Tuning Guide](../cache_tuning_guide.md)
- [Performance Profiling](../../performance_profiling.md)

### 3. Open Issue

Create an issue at: https://github.com/victor-ai/victor/issues

**Include:**
- `cache-metrics.json` (sanitized)
- `victor-logs.txt` (relevant sections)
- Configuration (YAML, environment variables)
- Steps to reproduce
- Expected vs. actual behavior

## Conclusion

Most cache issues can be resolved by:
1. **Checking configuration** (enabled vs. disabled)
2. **Verifying persistence** (PVC mounted, permissions)
3. **Adjusting parameters** (size, TTL)
4. **Monitoring metrics** (hit rate, latency)

When in doubt, start simple:
- Enable basic cache + persistent cache
- Monitor for 24-48 hours
- Add adaptive TTL if hit rate < 70%
- Add multi-level cache only if needed

### Quick Reference Card

```text
┌─────────────────────────────────────────────────────────────┐
│                  CACHE TROUBLESHOOTING QUICK REF           │
├─────────────────────────────────────────────────────────────┤
│ Hit Rate < 50%? → Increase TTL, check keys                 │
│ Memory > 100MB? → Clear cache, reduce size                  │
│ No persistence? → Check Redis, enable backend               │
│ Slow operations? → Check network, enable compression        │
│ Corruption? → Clear all, rebuild                            │
└─────────────────────────────────────────────────────────────┘
```

---

## See Also

- [Common Issues & Solutions](part-1-issues-solutions.md)
- [Cache Performance Guide](../cache_performance.md)
- [Production Caching Guide](../production_caching_guide.md)
- [Documentation Home](../../../README.md)


**Reading Time:** 6 min
**Last Updated:** February 09, 2026**
