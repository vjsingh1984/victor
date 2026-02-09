# Production Cache Quick Reference

**Track 5.3: Advanced Caching in Production**

Quick reference for production operators managing Victor AI's advanced caching system.

## Cache Status Check

```bash
# Quick health check
kubectl exec -it deployment/victor-ai -n victor-ai-prod -- \
  python -c "
from victor.tools.caches import AdvancedCacheManager
from victor.config import load_settings
settings = load_settings()
cache = AdvancedCacheManager.from_settings(settings)
metrics = cache.get_metrics()
print(f'Hit Rate: {metrics[\"combined\"][\"hit_rate\"]:.1%}')
print(f'Entries: {metrics[\"combined\"][\"total_entries\"]:,}')
print(f'Memory: {metrics.get(\"memory_usage_mb\", 0):.1f}MB')
print(f'Strategies: {sum(metrics[\"combined\"][\"strategies_enabled\"].values())} enabled')
"
```text

## Key Metrics

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Hit Rate | > 60% | < 50% | < 40% |
| Latency | < 5ms | > 10ms | > 20ms |
| Memory | < 10MB | > 50MB | > 100MB |
| Utilization | < 80% | > 90% | > 95% |

## Common Commands

### Enable/Disable Caching
```bash
# Enable
kubectl set env deployment/victor-ai VICTOR_TOOL_SELECTION_CACHE_ENABLED=true -n victor-ai-prod

# Disable
kubectl set env deployment/victor-ai VICTOR_TOOL_SELECTION_CACHE_ENABLED=false -n victor-ai-prod
```

### Clear Cache
```bash
# Clear in-memory cache (restart pods)
kubectl rollout restart deployment/victor-ai -n victor-ai-prod

# Clear persistent cache
kubectl exec -it deployment/victor-ai -n victor-ai-prod -- \
  rm /app/.victor/cache/tool_selection_cache.db
```text

### Monitor Cache
```bash
# One-time check
kubectl exec -it deployment/victor-ai -n victor-ai-prod -- \
  python scripts/monitor_cache.py

# Continuous monitoring
kubectl exec -it deployment/victor-ai -n victor-ai-prod -- \
  python scripts/monitor_cache.py --interval 60
```

### View Logs
```bash
# Cache-related logs
kubectl logs -f deployment/victor-ai -n victor-ai-prod | grep -i cache

# All logs
kubectl logs -f deployment/victor-ai -n victor-ai-prod
```text

## Environment Variables

### Master Switch
```yaml
VICTOR_TOOL_SELECTION_CACHE_ENABLED: "true"  # Master switch
```

### Persistent Cache
```yaml
VICTOR_PERSISTENT_CACHE_ENABLED: "true"
VICTOR_PERSISTENT_CACHE_PATH: "/app/.victor/cache/tool_selection_cache.db"
VICTOR_PERSISTENT_CACHE_AUTO_COMPACT: "true"
```text

### Adaptive TTL
```yaml
VICTOR_ADAPTIVE_TTL_ENABLED: "true"
VICTOR_ADAPTIVE_TTL_MIN: "60"           # 1 minute
VICTOR_ADAPTIVE_TTL_MAX: "7200"         # 2 hours
VICTOR_ADAPTIVE_TTL_INITIAL: "3600"     # 1 hour
VICTOR_ADAPTIVE_TTL_ADJUSTMENT_THRESHOLD: "5"
```

### Cache Size
```yaml
VICTOR_CACHE_SIZE: "2000"                            # Max entries
VICTOR_TOOL_SELECTION_CACHE_QUERY_TTL: "3600"        # 1 hour
VICTOR_TOOL_SELECTION_CACHE_CONTEXT_TTL: "300"       # 5 minutes
```text

## Troubleshooting

### Low Hit Rate (< 50%)
```bash
# Increase cache size
kubectl set env deployment/victor-ai VICTOR_CACHE_SIZE=3000 -n victor-ai-prod

# Increase TTL
kubectl set env deployment/victor-ai VICTOR_TOOL_SELECTION_CACHE_QUERY_TTL=7200 -n victor-ai-prod

# Enable adaptive TTL
kubectl set env deployment/victor-ai VICTOR_ADAPTIVE_TTL_ENABLED=true -n victor-ai-prod
```

### High Memory Usage (> 50MB)
```bash
# Reduce cache size
kubectl set env deployment/victor-ai VICTOR_CACHE_SIZE=1000 -n victor-ai-prod

# Reduce TTL
kubectl set env deployment/victor-ai VICTOR_TOOL_SELECTION_CACHE_QUERY_TTL=1800 -n victor-ai-prod

# Enable auto-compaction
kubectl set env deployment/victor-ai VICTOR_PERSISTENT_CACHE_AUTO_COMPACT=true -n victor-ai-prod
```text

### Cache Not Persisting
```bash
# Check PVC is mounted
kubectl exec -it deployment/victor-ai -n victor-ai-prod -- df -h /app/.victor/cache

# Check persistent cache enabled
kubectl set env deployment/victor-ai VICTOR_PERSISTENT_CACHE_ENABLED=true -n victor-ai-prod

# Check cache file exists
kubectl exec -it deployment/victor-ai -n victor-ai-prod -- ls -lh /app/.victor/cache/
```

### Slow Performance
```bash
# Check disk I/O (use fast SSD)
kubectl get pvc victor-cache-pvc -o jsonpath='{.spec.storageClassName}'

# Reduce cache size (faster lookups)
kubectl set env deployment/victor-ai VICTOR_CACHE_SIZE=1000 -n victor-ai-prod

# Disable persistent cache (use in-memory only)
kubectl set env deployment/victor-ai VICTOR_PERSISTENT_CACHE_ENABLED=false -n victor-ai-prod
```text

## Emergency Procedures

### Immediate Disable
```bash
kubectl set env deployment/victor-ai VICTOR_TOOL_SELECTION_CACHE_ENABLED=false -n victor-ai-prod
```

### Rollback Deployment
```bash
kubectl rollout undo deployment/victor-ai -n victor-ai-prod
```text

### Force Restart
```bash
kubectl rollout restart deployment/victor-ai -n victor-ai-prod
```

## Performance Tuning

### API Server (High Traffic)
```yaml
VICTOR_CACHE_SIZE: "5000"
VICTOR_TOOL_SELECTION_CACHE_QUERY_TTL: "7200"
VICTOR_ADAPTIVE_TTL_ENABLED: "true"
VICTOR_MULTI_LEVEL_CACHE_ENABLED: "true"
```text

### Interactive CLI (Low Traffic)
```yaml
VICTOR_CACHE_SIZE: "1000"
VICTOR_TOOL_SELECTION_CACHE_QUERY_TTL: "3600"
VICTOR_ADAPTIVE_TTL_ENABLED: "true"
```

### Development/Testing
```yaml
VICTOR_CACHE_SIZE: "500"
VICTOR_TOOL_SELECTION_CACHE_QUERY_TTL: "1800"
VICTOR_PERSISTENT_CACHE_ENABLED: "false"
```text

## Monitoring Alerts

### Prometheus Queries
```promql
# Hit rate
rate(victor_cache_hits_total[5m]) / (rate(victor_cache_hits_total[5m]) + rate(victor_cache_misses_total[5m]))

# Latency
histogram_quantile(0.95, rate(victor_cache_access_duration_seconds_bucket[5m]))

# Cache size
victor_cache_entries_total
```

### Alert Rules
```yaml
# Low hit rate alert
rate(victor_cache_hits_total[5m]) / (rate(victor_cache_hits_total[5m]) + rate(victor_cache_misses_total[5m])) < 0.5

# High latency alert
histogram_quantile(0.95, rate(victor_cache_access_duration_seconds_bucket[5m])) > 0.01
```text

## Documentation

- [Production Caching Guide](production_caching_guide.md) - Full deployment guide
- [Cache Tuning Guide](cache_tuning_guide.md) - Performance tuning
- [Cache Troubleshooting Guide](cache_troubleshooting.md) - Common issues
- [Track 5.3 Summary](track_5.3_production_caching_summary.md) - Implementation details

## Support

**Issues:** https://github.com/victor-ai/victor/issues

**Diagnostics:**
```bash
# Export diagnostics
kubectl exec -it deployment/victor-ai -n victor-ai-prod -- \
  python -c "from victor.tools.caches import AdvancedCacheManager; from victor.config import load_settings; import json;
  cache = AdvancedCacheManager.from_settings(load_settings()); print(json.dumps(cache.get_metrics(),
  indent=2, default=str))" > cache-metrics.json

# Export logs
kubectl logs deployment/victor-ai -n victor-ai-prod > victor-logs.txt

# Export cache database
kubectl cp victor-ai-prod/deployment/victor-ai:/app/.victor/cache/tool_selection_cache.db ./cache-diagnostics.db
```

---

## See Also

- [Documentation Home](../../README.md)


**Version:** 0.5.0
**Reading Time:** 3 min
**Last Updated:** 2025-01-21
**Track:** 5.3 - Production Caching
