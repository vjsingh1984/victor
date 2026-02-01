# Performance Monitoring Dashboard

Comprehensive performance monitoring system for Victor AI with real-time metrics, alerting, and visualization.

## Overview

The performance monitoring system provides:

- **Real-time metrics collection** from all Victor AI components
- **Grafana dashboard** for visualization and analysis
- **Prometheus alerts** for proactive monitoring
- **REST API** for programmatic access
- **Multi-dimensional metrics** (latency, throughput, errors, resources)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Victor AI Components                         │
│  Tool Selection Cache │ Provider Pool │ Tool Execution │ etc.  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PerformanceMetricsCollector                         │
│  Aggregates metrics from:                                       │
│  - ToolSelectionCache (hit rate, latency, memory)               │
│  - BootstrapMetrics (startup time, lazy loading)                │
│  - ProviderPoolMetrics (health, latency)                        │
│  - ToolExecutionMetrics (duration, errors)                      │
│  - SystemMetrics (CPU, memory, threads)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
        ┌───────────┐ ┌──────────┐ ┌─────────────┐
        │   REST    │ │ Prometheus│ │  Grafana    │
        │   API     │ │   Export  │ │  Dashboard  │
        └───────────┘ └──────────┘ └─────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  AlertManager  │
                    │  Alerting      │
                    └────────────────┘
```

## Components

### 1. PerformanceMetricsCollector

Centralized metrics collector that aggregates data from all components.

**Location**: `victor/observability/performance_collector.py`

**Key Classes**:
- `ToolSelectionMetrics` - Cache hit rates, latency, entries, utilization
- `CacheMetrics` - Memory usage, operations, hit/miss rates
- `BootstrapMetrics` - Startup time, phase timings, lazy loading
- `ProviderPoolMetrics` - Provider health, request metrics, latency
- `ToolExecutionMetrics` - Execution counts, duration, errors
- `SystemMetrics` - Memory, CPU, uptime, threads

**Usage**:

```python
from victor.observability import get_performance_collector

# Get collector instance
collector = get_performance_collector()

# Register components
collector.register_tool_selection_cache(cache)
collector.register_agent_metrics_collector(metrics_collector)

# Get all metrics
metrics = collector.get_all_metrics()

# Get specific category
cache_metrics = collector.get_cache_metrics()

# Export for Prometheus
prometheus_text = collector.export_prometheus()
```

### 2. REST API

REST endpoints for accessing performance metrics programmatically.

**Location**: `victor/api/endpoints/performance.py`

**Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/performance/summary` | GET | All performance metrics |
| `/api/performance/cache` | GET | Cache metrics only |
| `/api/performance/providers` | GET | Provider pool metrics |
| `/api/performance/tools` | GET | Tool execution metrics |
| `/api/performance/system` | GET | System resource metrics |
| `/api/performance/bootstrap` | GET | Bootstrap/startup metrics |
| `/api/performance/prometheus` | GET | Prometheus format export |

**Integration with FastAPI**:

```python
from fastapi import FastAPI
from victor.api.endpoints.performance import register_performance_routes

app = FastAPI()
register_performance_routes(app)
```

**Example Responses**:

```bash
# Get all metrics
curl http://localhost:8000/api/performance/summary

# Get cache metrics
curl http://localhost:8000/api/performance/cache

# Prometheus format
curl http://localhost:8000/api/performance/prometheus
```

### 3. Grafana Dashboard

Comprehensive dashboard for visualizing performance metrics.

**Location**: `deployment/kubernetes/monitoring/dashboards/victor-performance.json`

**Panels**:

**Row 1: Overview**
- Tool Selection Latency (P95)
- Cache Hit Rate (gauge)
- Memory Usage (graph)
- System Uptime (stat)

**Row 2: Cache Performance**
- Cache Entries by Namespace (graph)
- Cache Hit/Miss Ratio (pie)
- Cache Evictions (graph)
- Cache Utilization (gauge)

**Row 3: Tool Selection**
- Selection Latency P50/P95/P99 (graph)
- Selection Hit Rate by Type (bar)
- Selection Misses Rate (graph)

**Row 4: Provider Pool**
- Provider Health (table)
- Provider Latency (graph)
- Active Providers (stat)

**Row 5: Tool Execution**
- Execution Duration P95/P99 (graph)
- Execution Errors (graph)
- Top Tools by Execution (table)

**Deployment**:

```bash
# Apply dashboard config
kubectl apply -f deployment/kubernetes/monitoring/dashboards/

# Or use Grafana UI to import:
# 1. Open Grafana
# 2. Go to Dashboards -> Import
# 3. Upload victor-performance.json
```

### 4. Prometheus Alerts

Alerting rules for performance thresholds.

**Location**: `deployment/kubernetes/monitoring/performance-alerts.yaml`

**Alert Groups**:

#### Cache Alerts (`victor_performance_cache`)

| Alert | Severity | Threshold | Duration |
|-------|----------|-----------|----------|
| `HighCacheMissRate` | warning | hit_rate < 40% | 5m |
| `CriticalCacheMissRate` | critical | hit_rate < 20% | 10m |
| `HighCacheUtilization` | warning | utilization > 80% | 10m |
| `CriticalCacheUtilization` | critical | utilization > 95% | 5m |
| `HighCacheEvictionRate` | warning | evictions > 10/s | 5m |

#### Tool Selection Alerts (`victor_performance_tool_selection`)

| Alert | Severity | Threshold | Duration |
|-------|----------|-----------|----------|
| `HighToolSelectionLatency` | warning | P95 > 1ms | 10m |
| `CriticalToolSelectionLatency` | critical | P95 > 5ms | 5m |
| `HighToolSelectionMissRate` | warning | miss_rate > 70% | 5m |

#### System Alerts (`victor_performance_system`)

| Alert | Severity | Threshold | Duration |
|-------|----------|-----------|----------|
| `HighMemoryUsage` | warning | memory > 1GB | 10m |
| `CriticalMemoryUsage` | critical | memory > 2GB | 5m |
| `HighCPUUsage` | warning | cpu > 80% | 10m |
| `CriticalCPUUsage` | critical | cpu > 95% | 5m |
| `HighThreadCount` | warning | threads > 100 | 10m |

#### Tool Execution Alerts (`victor_performance_tool_execution`)

| Alert | Severity | Threshold | Duration |
|-------|----------|-----------|----------|
| `HighToolExecutionLatency` | warning | P95 > 1s | 10m |
| `CriticalToolExecutionLatency` | critical | P95 > 5s | 5m |
| `HighToolErrorRate` | warning | error_rate > 5% | 10m |
| `CriticalToolErrorRate` | critical | error_rate > 15% | 5m |

#### Provider Pool Alerts (`victor_performance_provider_pool`)

| Alert | Severity | Threshold | Duration |
|-------|----------|-----------|----------|
| `UnhealthyProviders` | warning | any unhealthy | 5m |
| `HighProviderFailureRate` | warning | failure_rate > 10% | 5m |
| `CriticalProviderFailureRate` | critical | failure_rate > 30% | 2m |
| `HighProviderLatency` | warning | P95 > 10s | 10m |

#### Summary Alerts (`victor_performance_summary`)

| Alert | Severity | Threshold | Duration |
|-------|----------|-----------|----------|
| `PerformanceDegraded` | warning | health_score < 70 | 10m |
| `PerformanceCritical` | critical | health_score < 50 | 5m |

**Deployment**:

```bash
# Apply alert rules
kubectl apply -f deployment/kubernetes/monitoring/performance-alerts.yaml

# Verify in Prometheus
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090
open http://localhost:9090/alerts
```

## Installation & Setup

### Prerequisites

- Kubernetes cluster with monitoring stack
- Prometheus deployed
- Grafana deployed
- AlertManager deployed
- Victor AI deployed with `/metrics` endpoint

### Step 1: Deploy Performance Alerts

```bash
# Apply performance alert rules
kubectl apply -f deployment/kubernetes/monitoring/performance-alerts.yaml

# Verify
kubectl get configmap -n victor-monitoring
```

### Step 2: Update Prometheus Configuration

The Prometheus ConfigMap already references `performance_alerts.yml`. If you need to reload:

```bash
# Restart Prometheus to pick up new rules
kubectl rollout restart deployment/prometheus -n victor-monitoring

# Verify rules are loaded
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/rules
```

### Step 3: Deploy Grafana Dashboard

```bash
# Option 1: Apply as ConfigMap
kubectl apply -f deployment/kubernetes/monitoring/dashboards/

# Option 2: Import via Grafana UI
# 1. Port-forward Grafana
kubectl port-forward -n victor-monitoring svc/grafana 3000:3000

# 2. Open Grafana
open http://localhost:3000

# 3. Go to Dashboards -> Import
# 4. Upload victor-performance.json

# Default credentials: admin/admin (change on first login)
```

### Step 4: Enable Performance API

Ensure your Victor AI deployment includes the performance API:

```python
# In victor/api/server.py
from victor.api.endpoints.performance import register_performance_routes

app = FastAPI(title="Victor AI API")
register_performance_routes(app)
```

### Step 5: Verify Metrics Collection

```bash
# Check metrics endpoint
kubectl port-forward -n victor-production svc/victor-api 8000:8000
curl http://localhost:8000/api/performance/summary

# Check Prometheus scraping
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/targets
```

## Usage

### Viewing Metrics in Grafana

1. **Open Dashboard**:
   ```bash
   kubectl port-forward -n victor-monitoring svc/grafana 3000:3000
   open http://localhost:3000/d/victor-performance
   ```

2. **Key Metrics to Monitor**:
   - **Cache Hit Rate**: Should be > 40% (warning if < 40%, critical if < 20%)
   - **Selection Latency**: P95 should be < 1ms (warning if > 1ms, critical if > 5ms)
   - **Memory Usage**: Should be < 1GB (warning if > 1GB, critical if > 2GB)
   - **Tool Error Rate**: Should be < 5% (warning if > 5%, critical if > 15%)

3. **Time Range Selection**:
   - Use the time picker at top right
   - Common ranges: Last 1 hour, Last 6 hours, Last 24 hours
   - Auto-refresh: 30s default

4. **Drill Down**:
   - Click on any panel to expand
   - Use panel inspector to view queries
   - Filter by labels (namespace, provider, tool)

### Querying Metrics via API

```bash
# Get all metrics
curl http://localhost:8000/api/performance/summary | jq

# Get cache metrics
curl http://localhost:8000/api/performance/cache | jq

# Get tool execution metrics
curl http://localhost:8000/api/performance/tools | jq

# Prometheus format (for custom scraping)
curl http://localhost:8000/api/performance/prometheus
```

### Responding to Alerts

When you receive an alert:

1. **Check the Dashboard**:
   - Open Grafana dashboard
   - Look for correlated metrics
   - Check recent trends

2. **Identify the Root Cause**:
   - **High cache miss rate**: Cache may be too small or TTL too short
   - **High latency**: Database slow, network issues, or overload
   - **High memory**: Memory leak or increased load
   - **High error rate**: Provider issues, tool failures, or bugs

3. **Take Action**:
   - Scale deployment: `kubectl scale deployment/victor-api --replicas=3`
   - Adjust cache size: Update configuration
   - Restart services: `kubectl rollout restart deployment/victor-api`
   - Check logs: `kubectl logs -f deployment/victor-api`

4. **Verify Resolution**:
   - Watch dashboard for improvement
   - Check AlertManager for alert clearance
   - Monitor for 15-30 minutes to ensure stability

## Metrics Reference

### Cache Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_cache_hit_rate` | gauge | namespace | Cache hit rate (0-1) |
| `victor_cache_entries` | gauge | namespace | Number of cache entries |
| `victor_cache_operations_total` | counter | operation | Cache operations (hits, misses, evictions) |
| `victor_cache_utilization` | gauge | namespace | Cache utilization (0-1) |
| `victor_cache_memory_bytes` | gauge | - | Cache memory usage in bytes |
| `victor_cache_memory_mb` | gauge | - | Cache memory usage in MB |

### Tool Selection Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_tool_duration_ms` | histogram | quantile | Tool selection duration |
| `victor_tool_duration_ms_bucket` | histogram | le, quantile | Duration bucket for percentile calculation |
| `victor_tool_executions_total` | counter | status | Tool execution count (success, failure) |
| `victor_tool_error_rate` | gauge | - | Tool execution error rate |

### System Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_system_memory_bytes` | gauge | - | System memory usage in bytes |
| `victor_system_memory_mb` | gauge | - | System memory usage in MB |
| `victor_system_cpu_percent` | gauge | - | CPU usage percentage |
| `victor_system_uptime_seconds` | gauge | - | System uptime in seconds |
| `victor_system_threads` | gauge | - | Active thread count |

### Provider Pool Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `victor_provider_health` | gauge | provider | Provider health status (0-1) |
| `victor_provider_requests_total` | counter | provider | Total provider requests |
| `victor_provider_errors_total` | counter | provider | Total provider errors |
| `victor_provider_latency_ms` | histogram | provider, le | Provider request latency |

## Configuration

### Cache Configuration

Adjust cache settings based on monitoring data:

```yaml
# In config/settings.yaml or environment variables
VICTOR_CACHE_SIZE=1000  # Max entries per namespace
VICTOR_CACHE_QUERY_TTL=3600  # Query cache TTL (1 hour)
VICTOR_CACHE_CONTEXT_TTL=300  # Context cache TTL (5 minutes)
VICTOR_CACHE_RL_TTL=3600  # RL cache TTL (1 hour)
```

**Tuning Guidelines**:
- **High cache miss rate**: Increase `VICTOR_CACHE_SIZE` or TTL values
- **High memory usage**: Decrease `VICTOR_CACHE_SIZE` or TTL values
- **High eviction rate**: Increase `VICTOR_CACHE_SIZE`

### Alert Thresholds

Customize alert thresholds in `performance-alerts.yaml`:

```yaml
# Example: Adjust cache miss rate threshold
- alert: HighCacheMissRate
  expr: victor_cache_hit_rate{namespace="overall"} < 0.3  # Was 0.4
  for: 5m
  labels:
    severity: warning
```

### Grafana Dashboard Customization

1. **Edit Dashboard**:
   - Open dashboard in Grafana
   - Click gear icon -> Settings
   - Modify panel queries, thresholds, layouts

2. **Add Variables**:
   - Settings -> Variables -> Add Variable
   - Example: `namespace` variable for filtering

3. **Export Modified Dashboard**:
   - Share -> Export -> Save to JSON
   - Commit to repository

## Troubleshooting

### Dashboard Shows No Data

**Symptoms**: Panels are empty or show "No data"

**Diagnosis**:
```bash
# Check if Prometheus is scraping
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/targets

# Check if metrics endpoint is accessible
kubectl port-forward -n victor-production svc/victor-api 8000:8000
curl http://localhost:8000/api/performance/prometheus
```

**Solutions**:
1. Verify Prometheus is scraping Victor pods:
   - Check pod annotations: `prometheus.io/scrape: "true"`
   - Check port: `prometheus.io/port: "8000"`
   - Check path: Default is `/metrics` or `/api/performance/prometheus`

2. Restart Prometheus:
   ```bash
   kubectl rollout restart deployment/prometheus -n victor-monitoring
   ```

3. Check network policies:
   ```bash
   kubectl get networkpolicy -n victor-production
   ```

### Alerts Not Firing

**Symptoms**: Alerts defined but not firing when thresholds exceeded

**Diagnosis**:
```bash
# Check if alert rules are loaded
kubectl port-forward -n victor-monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/rules

# Check AlertManager
kubectl port-forward -n victor-monitoring svc/alertmanager 9093:9093
# Open http://localhost:9093/#/alerts
```

**Solutions**:
1. Verify alert rules applied:
   ```bash
   kubectl get configmap performance-alerts -n victor-monitoring
   kubectl rollout restart deployment/prometheus -n victor-monitoring
   ```

2. Check alert expression syntax:
   - Use Prometheus expression browser: http://localhost:9090/graph
   - Test query manually
   - Verify metric names match

3. Check AlertManager routing:
   ```bash
   kubectl get configmap alertmanager-config -n victor-monitoring
   kubectl edit configmap alertmanager-config -n victor-monitoring
   ```

### High Memory Usage

**Symptoms**: Memory usage consistently above warning threshold

**Diagnosis**:
1. Check dashboard "Memory Usage" panel
2. Check "Cache Utilization" panels
3. Check "System Metrics" rows

**Solutions**:
1. Reduce cache size:
   ```yaml
   VICTOR_CACHE_SIZE=500  # Reduce from 1000
   ```

2. Reduce cache TTL:
   ```yaml
   VICTOR_CACHE_QUERY_TTL=1800  # Reduce from 3600
   ```

3. Scale deployment:
   ```bash
   kubectl autoscale deployment/victor-api \
     --cpu-percent=70 --min=2 --max=10
   ```

### Poor Cache Performance

**Symptoms**: Cache hit rate below 40%

**Diagnosis**:
1. Check "Cache Hit Rate" gauge
2. Check "Cache Hit/Miss Ratio" pie chart
3. Check "Cache Entries" graph

**Solutions**:
1. Increase cache size:
   ```yaml
   VICTOR_CACHE_SIZE=2000  # Increase from 1000
   ```

2. Increase cache TTL:
   ```yaml
   VICTOR_CACHE_QUERY_TTL=7200  # Increase from 3600
   ```

3. Analyze workload patterns:
   - Check if queries are diverse (low locality)
   - Check if cache key generation is effective
   - Consider using context cache for repeated patterns

## Best Practices

### Monitoring

1. **Set Up Dashboard Views**:
   - Create separate dashboard views for different teams
   - Use variables to filter by environment/namespace
   - Set up custom annotations for deployments

2. **Establish Baselines**:
   - Monitor system for 1-2 weeks to establish baselines
   - Document typical metric ranges
   - Set alert thresholds based on baseline + buffer

3. **Regular Review**:
   - Review dashboard weekly for trends
   - Adjust alert thresholds based on false positives
   - Archive old data to optimize performance

### Alerting

1. **Use Severity Levels**:
   - **Info**: Informational (startup events)
   - **Warning**: Investigation needed (degraded performance)
   - **Critical**: Immediate action required (service down)

2. **Set Up Notification Routing**:
   ```yaml
   # In AlertManager config
   route:
     receiver: 'default'
     routes:
       - match:
           severity: critical
         receiver: 'pagerduty'
       - match:
           severity: warning
         receiver: 'slack'
   ```

3. **Test Alerts**:
   - Use alert testing tools
   - Verify notification channels work
   - Document runbook procedures

### Performance Optimization

1. **Cache Tuning**:
   - Start with defaults (size=1000, TTL=3600s)
   - Monitor hit rate and memory usage
   - Adjust incrementally (±25% changes)
   - Allow 24-48 hours to evaluate changes

2. **Resource Allocation**:
   - Set resource requests/limits based on monitoring data
   - Use HPA for autoscaling based on CPU/memory
   - Profile memory usage during different workloads

3. **Query Optimization**:
   - Monitor tool selection latency
   - Identify slow queries
   - Consider query caching or optimization

## Advanced Topics

### Custom Metrics

Add custom metrics to the performance collector:

```python
from victor.observability import get_performance_collector

collector = get_performance_collector()

# Add custom metric tracking
class CustomMetrics:
    def __init__(self):
        self.custom_counter = 0
        self.custom_latency = []

    def record_custom_event(self, latency_ms):
        self.custom_counter += 1
        self.custom_latency.append(latency_ms)

# Register with collector
collector._custom_metrics = CustomMetrics()
```

### Dashboard Templating

Use Grafana variables for dynamic dashboards:

```json
{
  "templating": {
    "list": [
      {
        "name": "namespace",
        "query": "label_values(victor_cache_hit_rate, namespace)",
        "type": "query"
      }
    ]
  }
}
```

### Multi-Cluster Monitoring

Monitor multiple Victor AI clusters:

1. **Federated Prometheus**:
   - Deploy central Prometheus
   - Configure federation to scrape cluster Prometheis
   - Aggregate metrics with external labels

2. **Grafana Data Sources**:
   - Add each cluster Prometheus as data source
   - Use dashboard variables to switch clusters
   - Create unified dashboard with nested queries

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [AlertManager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Victor AI Architecture](../architecture/overview.md)
- [Performance Benchmarks](../performance/benchmark_results.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/victor-ai/issues
- Documentation: https://docs.victor-ai.dev
- Community: https://discord.gg/victor-ai
