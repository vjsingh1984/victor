# Performance Monitoring Guide - Part 1

**Part 1 of 2:** Overview, Architecture, Components, Installation & Setup, and Usage

---

## Navigation

- **[Part 1: System & Setup](#)** (Current)
- [Part 2: Metrics & Operations](part-2-metrics-operations.md)
- [**Complete Guide](../performance_monitoring.md)**

---
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

