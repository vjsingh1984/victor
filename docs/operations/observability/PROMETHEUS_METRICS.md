# Prometheus Metrics Guide

Comprehensive guide for Prometheus metrics collection in Victor AI.

## Overview

Victor AI provides production-ready Prometheus metrics for:
- **Request Metrics**: Request count, latency, error rate
- **LLM Metrics**: Provider calls, token usage, latency
- **Tool Metrics**: Tool executions, errors, duration
- **Cache Metrics**: Hit/miss rates, memory usage
- **System Metrics**: CPU, memory, connections
- **Business Metrics**: Tasks completed, workflows executed

## Installation

Dependencies are included with observability support:

```bash
pip install victor-ai[observability]
```

## Quick Start

### 1. Enable Metrics

```bash
# Enable Prometheus metrics
export VICTOR_METRICS_ENABLED=true

# Set metrics port
export VICTOR_METRICS_PORT=9090
```

### 2. Start Metrics Server

```python
from victor.observability.metrics_endpoint import start_metrics_server

# Start standalone metrics server
start_metrics_server(port=9090)
# Metrics available at http://localhost:9090/metrics
```

### 3. Configure Prometheus

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'victor'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

## Available Metrics

### Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_requests_total` | Counter | endpoint, method | Total requests |
| `victor_request_duration_ms` | Histogram | endpoint, method | Request latency |
| `victor_request_errors_total` | Counter | endpoint, method, error_type | Total errors |

**Example**:
```
victor_requests_total{endpoint="/chat",method="POST"} 1234
victor_request_duration_ms_bucket{endpoint="/chat",method="POST",le="100"} 800
victor_request_errors_total{endpoint="/chat",method="POST",error_type="rate_limit"} 5
```

### LLM Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_llm_requests_total` | Counter | provider, model | Total LLM requests |
| `victor_llm_duration_ms` | Histogram | provider, model | LLM latency |
| `victor_llm_tokens_total` | Counter | provider, model, type | Token usage |
| `victor_llm_errors_total` | Counter | provider, model, error_type | LLM errors |

**Example**:
```
victor_llm_requests_total{provider="anthropic",model="claude-sonnet-4-5"} 500
victor_llm_tokens_total{provider="anthropic",model="claude-sonnet-4-5",type="prompt"} 25000
victor_llm_tokens_total{provider="anthropic",model="claude-sonnet-4-5",type="completion"} 50000
victor_llm_duration_ms_bucket{provider="anthropic",model="claude-sonnet-4-5",le="1000"} 450
```

### Tool Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_tool_executions_total` | Counter | tool_name, status | Tool executions |
| `victor_tool_duration_ms` | Histogram | tool_name | Tool duration |
| `victor_tool_errors_total` | Counter | tool_name, error_type | Tool errors |

**Example**:
```
victor_tool_executions_total{tool_name="read_file",status="success"} 1200
victor_tool_executions_total{tool_name="read_file",status="error"} 5
victor_tool_duration_ms_bucket{tool_name="read_file",le="50"} 1150
```

### Cache Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_cache_hits_total` | Counter | cache_name | Cache hits |
| `victor_cache_misses_total` | Counter | cache_name | Cache misses |
| `victor_cache_hit_rate` | Gauge | cache_name | Hit rate (0-1) |
| `victor_cache_size_bytes` | Gauge | cache_name | Cache size |

**Example**:
```
victor_cache_hits_total{cache_name="tool_selection"} 4500
victor_cache_misses_total{cache_name="tool_selection"} 500
victor_cache_hit_rate{cache_name="tool_selection"} 0.9
victor_cache_size_bytes{cache_name="tool_selection"} 1048576
```

### Coordinator Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_coordinator_executions_total` | Counter | coordinator_name | Coordinator executions |
| `victor_coordinator_duration_seconds` | Histogram | coordinator_name | Duration |
| `victor_coordinator_errors_total` | Counter | coordinator_name | Errors |
| `victor_coordinator_memory_bytes` | Gauge | coordinator_name | Memory usage |
| `victor_coordinator_cpu_percent` | Gauge | coordinator_name | CPU usage |

**Example**:
```
victor_coordinator_executions_total{coordinator_name="ToolCoordinator"} 300
victor_coordinator_duration_seconds_bucket{coordinator_name="ToolCoordinator",le="1"} 250
victor_coordinator_memory_bytes{coordinator_name="ToolCoordinator"} 52428800
```

### System Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_system_cpu_percent` | Gauge | - | CPU usage |
| `victor_system_memory_bytes` | Gauge | type | Memory (rss, vms) |
| `victor_system_uptime_seconds` | Gauge | - | Uptime |
| `victor_active_connections` | Gauge | - | Active connections |

## Recording Metrics

### Using MetricsCollector

```python
from victor.observability.metrics import MetricsCollector

collector = MetricsCollector()

# Record metrics
collector.tool_calls.increment()
collector.model_requests.increment(labels={"provider": "anthropic"})
collector.model_latency.observe(45.2)

# Get summary
summary = collector.get_summary()
print(f"Tool calls: {summary['tool_calls']}")
print(f"Error rate: {summary['tool_error_rate']:.2%}")
```

### Custom Metrics

```python
from victor.observability.prometheus_metrics import get_prometheus_registry

registry = get_prometheus_registry()

# Create counter
counter = registry.counter(
    name="custom_operations_total",
    help="Total custom operations",
    labels={"service": "my_service"},
)
counter.inc()

# Create gauge
gauge = registry.gauge(
    name="queue_size",
    help="Current queue size",
)
gauge.set(10)

# Create histogram
histogram = registry.histogram(
    name="operation_latency_ms",
    help="Operation latency",
    buckets=[10, 50, 100, 500, 1000],
)
histogram.observe(75)
```

## Grafana Dashboards

### Basic Dashboard JSON

```json
{
  "title": "Victor AI Metrics",
  "panels": [
    {
      "title": "Request Rate",
      "targets": [
        {
          "expr": "rate(victor_requests_total[5m])"
        }
      ],
      "type": "graph"
    },
    {
      "title": "Request Latency (p95)",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, rate(victor_request_duration_ms_bucket[5m]))"
        }
      ],
      "type": "graph"
    },
    {
      "title": "Error Rate",
      "targets": [
        {
          "expr": "rate(victor_request_errors_total[5m]) / rate(victor_requests_total[5m])"
        }
      ],
      "type": "graph"
    }
  ]
}
```

### Recommended Queries

**Request rate by endpoint**:
```promql
sum(rate(victor_requests_total{endpoint=~"/api/.*"}[5m])) by (endpoint)
```

**P95 latency by provider**:
```promql
histogram_quantile(0.95, sum(rate(victor_llm_duration_ms_bucket[5m])) by (provider, le))
```

**Error rate by model**:
```promql
sum(rate(victor_llm_errors_total[5m])) by (model) / sum(rate(victor_llm_requests_total[5m])) by (model)
```

**Cache hit rate**:
```promql
victor_cache_hit_rate{cache_name="tool_selection"}
```

**Token usage per provider**:
```promql
sum(rate(victor_llm_tokens_total[5m])) by (provider, type)
```

## Alerting Rules

### Prometheus Alert Rules

```yaml
groups:
  - name: victor_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(victor_request_errors_total[5m]) / rate(victor_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.endpoint }}"

      # High latency
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, rate(victor_request_duration_ms_bucket[5m])) > 5000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency"
          description: "P95 latency is {{ $value }}ms"

      # LLM rate limiting
      - alert: LLMRateLimiting
        expr: |
          rate(victor_llm_errors_total{error_type="rate_limit"}[5m]) > 1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM rate limiting detected"
          description: "{{ $labels.provider }} is being rate limited"

      # Low cache hit rate
      - alert: LowCacheHitRate
        expr: |
          victor_cache_hit_rate < 0.5
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Low cache hit rate"
          description: "Cache {{ $labels.cache_name }} hit rate is {{ $value | humanizePercentage }}"
```

## Best Practices

1. **Use labels effectively**: Include relevant dimensions
2. **Avoid high cardinality**: Don't include user IDs or timestamps
3. **Use histograms for latency**: Capture distribution
4. **Count errors separately**: Don't lose error details
5. **Document metrics**: Add HELP and TYPE metadata
6. **Test queries**: Verify in Prometheus UI
7. **Set up alerts**: Proactive monitoring
8. **Review dashboards**: Regular maintenance

## Performance Tuning

### Scrape Interval

```yaml
# Higher frequency for development
scrape_interval: 15s

# Lower frequency for production
scrape_interval: 60s
```

### Metric Retention

```prometheus
# prometheus.yml
storage:
  tsdb:
    retention.time: 15d
```

### Recording Rules

Pre-compute expensive queries:

```yaml
groups:
  - name: victor_recording
    interval: 30s
    rules:
      - record: job:victor:request_rate:5m
        expr: sum(rate(victor_requests_total[5m])) by (job)
```

## Troubleshooting

### No metrics appearing

1. Check metrics endpoint:
```bash
curl http://localhost:9090/metrics
```

2. Verify Prometheus configuration:
```bash
promtool check config prometheus.yml
```

3. Check Prometheus targets:
```bash
curl http://localhost:9090/api/v1/targets
```

### High memory usage

1. Reduce scrape interval
2. Decrease retention period
3. Use recording rules
4. Filter unnecessary metrics

### Missing labels

1. Verify metric registration
2. Check label consistency
3. Review prometheus logs

## Examples

See `examples/observability/` for complete examples:
- `metrics_collector.py`: Basic metrics collection
- `custom_metrics.py`: Custom metric definitions
- `grafana_dashboard.json`: Complete dashboard
- `alerting_rules.yml`: Production alerting rules
