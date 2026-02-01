# Cache Analytics and Monitoring Guide

## Overview

Cache analytics provides comprehensive monitoring and insights into cache performance, helping you optimize hit rates, identify bottlenecks, and make data-driven tuning decisions.

## Key Metrics

### Hit Rate

Percentage of cache requests that return cached data.

**Calculation**: `hits / (hits + misses)`

**Target**: > 50% for general purpose, > 70% for optimized systems

```python
from victor.core.cache import CacheAnalytics

analytics = CacheAnalytics(cache=cache)
hit_rate = analytics.get_hit_rate()

print(f"Current hit rate: {hit_rate:.1%}")

if hit_rate < 0.5:
    print("Hit rate below target, consider tuning")
```

### Miss Rate

Percentage of cache requests that don't find cached data.

**Calculation**: `misses / (hits + misses)` or `1 - hit_rate`

```python
miss_rate = analytics.get_miss_rate()
print(f"Miss rate: {miss_rate:.1%}")
```

### Eviction Rate

Percentage of cache accesses that result in eviction.

**High eviction rate** suggests:
- Cache size too small
- TTL too short
- High data churn

```python
eviction_rate = analytics.get_eviction_rate()

if eviction_rate > 0.1:
    print("High eviction rate detected")
    print("Consider: Increasing cache size or TTL")
```

### Latency Statistics

Access latency distribution (avg, p50, p95, p99).

```python
latency_stats = analytics.get_latency_stats()

print(f"Average: {latency_stats['avg_ms']:.2f}ms")
print(f"P50: {latency_stats['p50_ms']:.2f}ms")
print(f"P95: {latency_stats['p95_ms']:.2f}ms")
print(f"P99: {latency_stats['p99_ms']:.2f}ms")

# Check for latency issues
if latency_stats['p95_ms'] > 5.0:
    print("High P95 latency detected")
```

## Hot Key Analysis

Identify frequently accessed cache entries.

### Get Hot Keys

```python
# Get top 100 hot keys
hot_keys = analytics.get_hot_keys(top_n=100)

for hot_key in hot_keys[:10]:
    print(f"Key: {hot_key.key}")
    print(f"  Access Count: {hot_key.access_count}")
    print(f"  Hit Count: {hot_key.hit_count}")
    print(f"  Miss Count: {hot_key.miss_count}")
    print(f"  Avg Latency: {hot_key.avg_latency_ms:.2f}ms")
    print()
```

### Hot Key Characteristics

```python
# Identify hot keys with poor performance
for hot_key in hot_keys:
    hit_rate = hot_key.hit_count / hot_key.access_count

    if hot_key.access_count > 100 and hit_rate < 0.5:
        print(f"Poor hit rate for hot key: {hot_key.key}")
        print(f"  Access Count: {hot_key.access_count}")
        print(f"  Hit Rate: {hit_rate:.1%}")
        print("  → Consider increasing TTL or pre-warming")
```

### Heatmap Analysis

```python
import matplotlib.pyplot as plt
from collections import defaultdict

# Group by namespace
namespace_access = defaultdict(int)
for hot_key in hot_keys:
    namespace_access[hot_key.namespace] += hot_key.access_count

# Visualize
plt.bar(namespace_access.keys(), namespace_access.values())
plt.title("Cache Access by Namespace")
plt.xlabel("Namespace")
plt.ylabel("Access Count")
plt.xticks(rotation=45)
plt.show()
```

## Size Utilization

Monitor cache space usage.

### Current Utilization

```python
utilization = analytics.get_size_utilization()

print(f"L1 Utilization: {utilization['l1_utilization']:.1%}")
print(f"L2 Utilization: {utilization['l2_utilization']:.1%}")

# Check for capacity issues
if utilization['l1_utilization'] > 0.9:
    print("L1 cache near capacity")
    print("→ Consider increasing L1 size")
```

### Trend Analysis

```python
# Track utilization over time
utilization_history = []

for i in range(24):  # 24 hours
    time.sleep(3600)  # Wait 1 hour
    utilization = analytics.get_size_utilization()
    utilization_history.append({
        'hour': i,
        'l1': utilization['l1_utilization'],
        'l2': utilization['l2_utilization'],
    })

# Analyze trends
for entry in utilization_history:
    print(f"Hour {entry['hour']}: "
          f"L1={entry['l1']:.1%}, L2={entry['l2']:.1%}")
```

## Optimization Recommendations

Get actionable recommendations for cache improvement.

### Generate Recommendations

```python
recommendations = analytics.get_recommendations()

for rec in recommendations:
    print(f"[{rec.priority.upper()}] {rec.title}")
    print(f"  {rec.description}")
    print(f"  Expected Impact: {rec.expected_impact}")
    print(f"  Action: {rec.action}")
    print()
```

### Recommendation Types

1. **Hit Rate Optimization**
   - Low hit rate (< 30%)
   - Suggests: Increase TTL, enable warming, adjust cache keys

2. **Eviction Rate Optimization**
   - High eviction rate (> 10%)
   - Suggests: Increase cache size, increase TTL

3. **Size Optimization**
   - High utilization (> 90%)
   - Suggests: Increase cache size, enable write-back policy

4. **Latency Optimization**
   - High latency (> 5ms)
   - Suggests: Use in-memory cache, optimize serialization

5. **Warming Optimization**
   - Low hit rate with many keys
   - Suggests: Enable cache warming

### Prioritization

```python
recommendations = analytics.get_recommendations()

# Prioritize by priority
priority_order = ['critical', 'high', 'medium', 'low']

for priority in priority_order:
    critical_recs = [r for r in recommendations if r.priority == priority]
    if critical_recs:
        print(f"\n{'='*60}")
        print(f"{priority.upper()} PRIORITY")
        print('='*60)
        for rec in critical_recs:
            print(f"\n{rec.title}")
            print(f"  {rec.action}")
```

## Real-Time Monitoring

### Background Monitoring

```python
# Start monitoring
await analytics.start_monitoring(interval_seconds=60)

# Monitoring runs in background, checking metrics every 60 seconds
```

### Alert Callbacks

```python
# Register alert callback
def on_alert(alert_type: str, alert_data: dict):
    print(f"ALERT: {alert_type}")
    print(f"Severity: {alert_data['severity']}")
    print(f"Message: {alert_data['message']}")

    # Send to monitoring service
    send_to_prometheus(alert_type, alert_data)
    send_to_slack(alert_type, alert_data)

analytics.register_alert_callback(on_alert)
```

### Alert Types

1. **Low Hit Rate Alert**
   - Triggered when hit rate < 30%
   - Severity: critical if < 20%, warning if 20-30%

2. **High Latency Alert**
   - Triggered when avg latency > 10ms
   - Severity: critical if > 20ms, warning if 10-20ms

3. **High Eviction Rate Alert**
   - Triggered when eviction rate > 20%
   - Severity: warning

## Prometheus Integration

Export metrics for Prometheus monitoring.

### Export Metrics

```python
# Export to Prometheus format
metrics = await analytics.export_to_prometheus()

for metric_name, value in metrics.items():
    print(f"{metric_name} {value}")

# Output:
# cache_hit_rate 0.673
# cache_miss_rate 0.327
# cache_eviction_rate 0.082
# cache_latency_avg_ms 0.45
# cache_latency_p95_ms 1.23
# cache_latency_p99_ms 2.87
# cache_l1_utilization 0.843
# cache_l2_utilization 0.621
# cache_total_accesses 15234
# cache_hot_keys 234
```

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'victor_cache'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8080']
```

### Metrics Endpoint

```python
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge

app = FastAPI()

# Define metrics
cache_hits = Counter('cache_hits_total', 'Total cache hits')
cache_misses = Counter('cache_misses_total', 'Total cache misses')
cache_latency = Histogram('cache_latency_seconds', 'Cache access latency')
cache_size = Gauge('cache_size', 'Current cache size')

@app.get("/metrics")
async def metrics():
    # Update metrics
    stats = analytics.get_comprehensive_stats()

    cache_hits._value._value = stats['cache_stats']['l1']['hits']
    cache_misses._value._value = stats['cache_stats']['l1']['misses']
    cache_size.set(stats['cache_stats']['l1']['size'])

    # Return Prometheus format
    from prometheus_client import generate_latest
    return generate_latest()
```

## Dashboard Visualization

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Victor Cache Analytics",
    "panels": [
      {
        "title": "Hit Rate",
        "targets": [
          {
            "expr": "cache_hit_rate"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Latency",
        "targets": [
          {
            "expr": "cache_latency_avg_ms"
          },
          {
            "expr": "cache_latency_p95_ms"
          },
          {
            "expr": "cache_latency_p99_ms"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Size Utilization",
        "targets": [
          {
            "expr": "cache_l1_utilization"
          },
          {
            "expr": "cache_l2_utilization"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Hot Keys",
        "targets": [
          {
            "expr": "cache_hot_keys"
          }
        ],
        "type": "stat"
      }
    ]
  }
}
```

### Custom Dashboard

```python
import streamlit as st

st.title("Victor Cache Analytics")

# Get statistics
stats = analytics.get_comprehensive_stats()

# Hit rate gauge
st.metric("Hit Rate", f"{stats['hit_rate']:.1%}")

# Latency chart
latency = stats['latency']
col1, col2, col3 = st.columns(3)
col1.metric("Avg", f"{latency['avg_ms']:.2f}ms")
col2.metric("P95", f"{latency['p95_ms']:.2f}ms")
col3.metric("P99", f"{latency['p99_ms']:.2f}ms")

# Size utilization
util = stats['size_utilization']
st.write("### Cache Utilization")
st.write(f"L1: {util['l1_utilization']:.1%}")
st.write(f"L2: {util['l2_utilization']:.1%}")

# Hot keys table
hot_keys = analytics.get_hot_keys(top_n=20)
st.write("### Top 20 Hot Keys")
st.dataframe([
    {
        "Key": k.key,
        "Namespace": k.namespace,
        "Access Count": k.access_count,
        "Hit Rate": f"{k.hit_count/k.access_count:.1%}",
    }
    for k in hot_keys
])
```

## Performance Analysis

### Time-Based Analysis

```python
# Analyze hit rate by hour
hourly_hit_rates = defaultdict(lambda: {'hits': 0, 'misses': 0})

for access in analytics._access_log:
    hour = datetime.fromtimestamp(access[3]).hour
    if access[2]:  # hit
        hourly_hit_rates[hour]['hits'] += 1
    else:  # miss
        hourly_hit_rates[hour]['misses'] += 1

# Calculate rates
for hour in sorted(hourly_hit_rates.keys()):
    data = hourly_hit_rates[hour]
    total = data['hits'] + data['misses']
    hit_rate = data['hits'] / total if total > 0 else 0
    print(f"Hour {hour:2d}: {hit_rate:.1%}")
```

### Namespace Analysis

```python
# Analyze hit rate by namespace
namespace_stats = defaultdict(lambda: {'hits': 0, 'misses': 0})

for access in analytics._access_log:
    key, ns, hit, _ = access
    if hit:
        namespace_stats[ns]['hits'] += 1
    else:
        namespace_stats[ns]['misses'] += 1

# Print summary
for ns in sorted(namespace_stats.keys()):
    data = namespace_stats[ns]
    total = data['hits'] + data['misses']
    hit_rate = data['hits'] / total if total > 0 else 0
    print(f"{ns}: {hit_rate:.1%} ({total} accesses)")
```

## Best Practices

### 1. Monitor Continuously

```python
# Set up continuous monitoring
analytics = CacheAnalytics(cache=cache, track_hot_keys=True)
await analytics.start_monitoring(interval_seconds=60)

# Register alerts
analytics.register_alert_callback(send_to_slack)
analytics.register_alert_callback(send_to_prometheus)
```

### 2. Track Key Metrics

```python
# Focus on key metrics
stats = analytics.get_comprehensive_stats()

key_metrics = {
    'hit_rate': stats['hit_rate'],
    'avg_latency': stats['latency']['avg_ms'],
    'l1_utilization': stats['size_utilization']['l1_utilization'],
}

for metric, value in key_metrics.items():
    print(f"{metric}: {value}")
```

### 3. Act on Recommendations

```python
# Regularly check recommendations
recommendations = analytics.get_recommendations()

# Implement high-priority recommendations first
high_priority = [r for r in recommendations if r.priority == 'high']
for rec in high_priority:
    print(f"Implementing: {rec.action}")
    # Apply recommendation
    apply_recommendation(rec)
```

### 4. Track Trends

```python
# Store historical metrics
history = []

for i in range(24):  # 24 hours
    stats = analytics.get_comprehensive_stats()
    history.append({
        'timestamp': datetime.now(),
        'hit_rate': stats['hit_rate'],
        'latency': stats['latency']['avg_ms'],
    })
    await asyncio.sleep(3600)

# Analyze trends
hit_rates = [h['hit_rate'] for h in history]
if hit_rates[-1] < hit_rates[0] * 0.9:
    print("Hit rate declining by >10%")
```

## Troubleshooting

### Monitoring Not Working

**Symptoms**: No metrics being collected

**Possible Causes**:
1. Monitoring not started
2. Access tracking not enabled
3. Event loop not running

**Solutions**:
1. Call `start_monitoring()`
2. Enable `track_hot_keys=True`
3. Ensure asyncio event loop running

### High Memory Usage

**Symptoms**: Analytics using too much memory

**Possible Causes**:
1. Hot key window too large
2. Latency samples not bounded
3. Historical data not cleaned up

**Solutions**:
1. Reduce `hot_key_window`
2. Check `maxlen` on deques
3. Implement periodic cleanup

### Missing Recommendations

**Symptoms**: No recommendations generated

**Possible Causes**:
1. Insufficient data collected
2. All metrics within normal range
3. Recommendation logic not triggered

**Solutions**:
1. Collect more access data
2. Check if metrics are actually good
3. Adjust recommendation thresholds

## See Also

- [Multi-Level Cache](MULTI_LEVEL_CACHE.md)
- [Cache Warming](CACHE_WARMING.md)
- [Semantic Caching](SEMANTIC_CACHING.md)
- [Cache Invalidation](CACHE_INVALIDATION.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
