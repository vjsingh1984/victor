# Production Metrics and Monitoring Guide - Part 2

**Part 2 of 2:** Monitoring Best Practices, Troubleshooting, and Related Documentation

---

## Navigation

- [Part 1: Metrics & Monitoring](part-1-metrics-monitoring.md)
- **[Part 2: Best Practices, Troubleshooting](#)** (Current)
- [**Complete Guide](../PRODUCTION_METRICS.md)**

---

## Monitoring Best Practices

### 1. Metric Naming

Use consistent naming conventions:

```yaml
# Good: Hierarchical naming
victor_api_requests_total
victor_api_duration_seconds
victor_tools_executed_total

# Avoid: Flat naming
api_requests
tool_duration
```

### 2. Label Usage

Add meaningful labels:

```python
# Good: Rich labels
counter(
    "victor_api_requests_total",
    ["endpoint", "method", "status"],
    endpoint="/chat", method="POST", status="200"
)

# Avoid: Missing context
counter("victor_api_requests_total", ["endpoint"])
```

### 3. Cardinality

Control metric cardinality:

```python
# Good: Bounded labels
endpoint = "/api/v1/chat"  # Low cardinality
status = "200"              # Low cardinality

# Avoid: High cardinality labels
user_id = "12345"            # High cardinality - don't use!
request_id = "abc-def"      # High cardinality - don't use!
```

### 4. Sampling

Use appropriate sampling rates:

```python
# High-throughput metrics
histogram(
    "victor_api_duration_seconds",
    buckets=[0.001, 0.01, 0.1, 1.0],
    sample_rate=0.1  # 10% sampling for high traffic
)
```

---

## Troubleshooting

### Metrics Not Appearing

**Problem:** Metrics not showing in Prometheus.

**Solutions:**
1. **Check endpoint**: Verify `/metrics` endpoint is accessible
2. **Check scrape config**: Ensure Prometheus target is configured
3. **Check labels**: Verify metric labels match scrape config

```bash
# Test metrics endpoint
curl http://localhost:8080/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### High Memory Usage

**Problem:** Prometheus using too much memory.

**Solutions:**
1. **Reduce scrape interval**: Scrape less frequently
2. **Reduce retention**: Keep metrics for shorter period
3. **Filter metrics**: Scrape only needed metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'victor'
    scrape_interval: 30s  # Increase from 15s
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'victor_.*'
        action: keep
```

### Alert Fatigue

**Problem:** Too many alerts, alert fatigue.

**Solutions:**
1. **Adjust thresholds**: Tune alert thresholds
2. **Group alerts**: Use alert grouping
3. **Add inhibition rules**: Suppress during maintenance

```yaml
# Alertmanager configuration
inhibit_rules:
  - source_match:
      severity: 'warning'
    target_match:
      alertname: 'Watchdog'
    equal: ['maintenance']
```

---

## Related Documentation

- [Dashboard Setup](../dashboards/)
- [Observability Guide](../MONITORING_SETUP.md)
- [Performance Tuning](../performance/performance_autotuning.md)

---

**Reading Time:** 1 min
**Last Updated:** February 01, 2026
