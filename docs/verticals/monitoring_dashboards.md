# Victor Monitoring Dashboards

**Version**: 1.0
**Date**: 2026-03-31
**Platform**: Grafana + Prometheus + OpenTelemetry

## Overview

This document defines monitoring dashboards for the refactored Victor verticals architecture. Dashboards track vertical loading performance, dependency graph metrics, cache statistics, and system health.

## Table of Contents

1. [Dashboard 1: Vertical Loading Performance](#dashboard-1-vertical-loading-performance)
2. [Dashboard 2: Dependency Graph Metrics](#dashboard-2-dependency-graph-metrics)
3. [Dashboard 3: Cache Performance](#dashboard-3-cache-performance)
4. [Dashboard 4: System Health](#dashboard-4-system-health)
5. [Alert Rules](#alert-rules)
6. [Queries](#queries)

---

## Dashboard 1: Vertical Loading Performance

### Purpose

Monitor vertical loading times, entry point scanning, and error rates.

### Panels

#### 1. Vertical Loading Time (Histogram)

**Type**: Histogram
**Query**:
```promql
# Vertical loading time percentiles
histogram_quantile(0.50, sum(rate(victor_vertical_loading_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(victor_vertical_loading_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.99, sum(rate(victor_vertical_loading_duration_seconds_bucket[5m])) by (le))
```

**Visualization**:
- P50: Green line
- P95: Yellow line
- P99: Red line
- X-axis: Time (seconds)
- Y-axis: Duration

**Alert**: P95 > 100ms (WARNING), P95 > 200ms (CRITICAL)

---

#### 2. Entry Point Scan Duration (Gauge)

**Type**: Gauge
**Query**:
```promql
# Entry point scan duration
victor_entry_point_scan_duration_seconds{job="victor-api"}
```

**Visualization**:
- Single stat panel
- Current value in milliseconds
- Color scale: Green (<20ms), Yellow (20-50ms), Red (>50ms)

**Alert**: Scan > 50ms (WARNING), Scan > 100ms (CRITICAL)

---

#### 3. Verticals Loaded Successfully (Counter)

**Type**: Counter
**Query**:
```promql
# Verticals loaded successfully
sum(rate(victor_verticals_loaded_total{status="success"}[5m])) by (vertical_name)
```

**Visualization**:
- Time series graph
- Group by vertical name
- Y-axis: Loads per second

---

#### 4. Vertical Loading Errors (Counter)

**Type**: Counter
**Query**:
```promql
# Vertical loading errors
sum(rate(victor_verticals_loaded_total{status="error"}[5m])) by (vertical_name, error_type)
```

**Visualization**:
- Time series graph
- Group by vertical name and error type
- Y-axis: Errors per second

**Alert**: Error rate > 0.01/s (WARNING), Error rate > 0.1/s (CRITICAL)

---

#### 5. Vertical Loading Success Rate (Gauge)

**Type**: Gauge
**Query**:
```promql
# Success rate
sum(rate(victor_verticals_loaded_total{status="success"}[5m])) /
sum(rate(victor_verticals_loaded_total[5m]))
```

**Visualization**:
- Single stat panel
- Percentage
- Color scale: Red (<95%), Yellow (95-99%), Green (≥99%)

**Alert**: Success rate < 95% (WARNING), Success rate < 90% (CRITICAL)

---

## Dashboard 2: Dependency Graph Metrics

### Purpose

Monitor dependency graph depth, load order consistency, and circular dependencies.

### Panels

#### 1. Dependency Graph Depth (Gauge)

**Type**: Gauge
**Query**:
```promql
# Dependency graph depth
victor_dependency_graph_depth{job="victor-api"}
```

**Visualization**:
- Single stat panel
- Current depth value
- Color scale: Green (<5), Yellow (5-10), Red (>10)

**Alert**: Depth > 10 (WARNING), Depth > 15 (CRITICAL)

---

#### 2. Load Order Changes (Counter)

**Type**: Counter
**Query**:
```promql
# Load order changes
sum(rate(victor_dependency_graph_load_order_changes_total[5m]))
```

**Visualization**:
- Time series graph
- Y-axis: Changes per second

**Alert**: Changes > 0.1/s (WARNING)

---

#### 3. Circular Dependencies Detected (Counter)

**Type**: Counter
**Query**:
```promql
# Circular dependencies detected
sum(rate(victor_dependency_graph_circular_dependencies_total[5m]))
```

**Visualization**:
- Single stat panel
- Total count (cumulative)
- Color: Green (0), Yellow (>0), Red (>5)

**Alert**: Circular dependencies detected (CRITICAL)

---

#### 4. Missing Dependencies (Counter)

**Type**: Counter
**Query**:
```promql
# Missing dependencies
sum(rate(victor_dependency_graph_missing_dependencies_total[5m])) by (dependency_name)
```

**Visualization**:
- Time series graph
- Group by dependency name
- Y-axis: Detections per second

**Alert**: Missing dependencies > 0 (CRITICAL)

---

#### 5. Dependency Resolution Duration (Histogram)

**Type**: Histogram
**Query**:
```promql
# Dependency resolution duration
histogram_quantile(0.95, sum(rate(victor_dependency_resolution_duration_seconds_bucket[5m])) by (le))
```

**Visualization**:
- P95 line
- Y-axis: Duration (seconds)
- Color scale: Green (<5ms), Yellow (5-10ms), Red (>10ms)

**Alert**: P95 > 10ms (WARNING), P95 > 20ms (CRITICAL)

---

## Dashboard 3: Cache Performance

### Purpose

Monitor cache hit rate, lock contention, and cache size.

### Panels

#### 1. Cache Hit Rate (Gauge)

**Type**: Gauge
**Query**:
```promql
# Cache hit rate
sum(rate(victor_cache_hits_total[5m])) /
(sum(rate(victor_cache_hits_total[5m])) + sum(rate(victor_cache_misses_total[5m])))
```

**Visualization**:
- Single stat panel
- Percentage
- Color scale: Red (<70%), Yellow (70-80%), Green (>80%)

**Alert**: Hit rate < 70% (WARNING), Hit rate < 50% (CRITICAL)

---

#### 2. Cache Hit/Miss Rate (Time Series)

**Type**: Time series
**Query**:
```promql
# Cache hits and misses
sum(rate(victor_cache_hits_total[5m])) by (cache_namespace)
sum(rate(victor_cache_misses_total[5m])) by (cache_namespace)
```

**Visualization**:
- Two lines (hits in green, misses in red)
- Group by cache namespace
- Y-axis: Operations per second

---

#### 3. Cache Size (Gauge)

**Type**: Gauge
**Query**:
```promql
# Cache size
sum(victor_cache_size) by (cache_namespace)
```

**Visualization**:
- Stat panel
- Group by cache namespace
- Y-axis: Number of entries

---

#### 4. Lock Contention Duration (Histogram)

**Type**: Histogram
**Query**:
```promql
# Lock wait duration
histogram_quantile(0.95, sum(rate(victor_cache_lock_wait_duration_seconds_bucket[5m])) by (le))
```

**Visualization**:
- P95 line
- Y-axis: Duration (seconds)
- Color scale: Green (<1ms), Yellow (1-5ms), Red (>5ms)

**Alert**: P95 > 5ms (WARNING), P95 > 10ms (CRITICAL)

---

#### 5. Cache Evictions (Counter)

**Type**: Counter
**Query**:
```promql
# Cache evictions
sum(rate(victor_cache_evictions_total[5m])) by (cache_namespace)
```

**Visualization**:
- Time series graph
- Group by cache namespace
- Y-axis: Evictions per second

**Alert**: Eviction rate > 10/s (WARNING)

---

## Dashboard 4: System Health

### Purpose

Monitor overall system health, including errors, latency, and resource usage.

### Panels

#### 1. Total Verticals Loaded (Counter)

**Type**: Counter
**Query**:
```promql
# Total verticals loaded
sum(increase(victor_verticals_loaded_total{status="success"}[1h]))
```

**Visualization**:
- Single stat panel
- Total count
- Show delta from previous hour

---

#### 2. Error Rate by Component (Time Series)

**Type**: Time series
**Query**:
```promql
# Error rate by component
sum(rate(victor_errors_total[5m])) by (component, error_type)
```

**Visualization**:
- Time series graph
- Group by component and error type
- Y-axis: Errors per second

**Alert**: Any component error rate > 0.01/s (WARNING)

---

#### 3. Startup Duration (Gauge)

**Type**: Gauge
**Query**:
```promql
# Startup duration
victor_startup_duration_seconds{job="victor-api"}
```

**Visualization**:
- Single stat panel
- Current startup time in seconds
- Color scale: Green (<1s), Yellow (1-2s), Red (>2s)

**Alert**: Startup > 2s (WARNING), Startup > 5s (CRITICAL)

---

#### 4. Memory Usage (Gauge)

**Type**: Gauge
**Query**:
```promql
# Memory usage
process_resident_memory_bytes{job="victor-api"} / 1024 / 1024 / 1024
```

**Visualization**:
- Single stat panel
- Current memory in GB
- Color scale: Green (<2GB), Yellow (2-4GB), Red (>4GB)

**Alert**: Memory > 4GB (WARNING), Memory > 8GB (CRITICAL)

---

#### 5. CPU Usage (Gauge)

**Type**: Gauge
**Query**:
```promql
# CPU usage
rate(process_cpu_seconds_total{job="victor-api"}[5m]) * 100
```

**Visualization**:
- Single stat panel
- Current CPU usage percentage
- Color scale: Green (<50%), Yellow (50-80%), Red (>80%)

**Alert**: CPU > 80% (WARNING), CPU > 90% (CRITICAL)

---

## Alert Rules

### Critical Alerts

#### Alert: Vertical Loading Failure

**Condition**: Error rate > 1%
```yaml
alert: vertical_loading_failure
expr: |
  sum(rate(victor_verticals_loaded_total{status="error"}[5m])) /
  sum(rate(victor_verticals_loaded_total[5m])) > 0.01
for: 1m
labels:
  severity: critical
annotations:
  summary: "High vertical loading error rate"
  description: "Vertical loading error rate is {{ $value | humanizePercentage }}"
```

---

#### Alert: Entry Point Scan Too Slow

**Condition**: P95 > 100ms
```yaml
alert: entry_point_scan_slow
expr: |
  histogram_quantile(0.95, victor_entry_point_scan_duration_seconds) > 0.1
for: 5m
labels:
  severity: critical
annotations:
  summary: "Entry point scan too slow"
  description: "P95 scan duration is {{ $value }}s"
```

---

#### Alert: Circular Dependency Detected

**Condition**: Any circular dependency
```yaml
alert: circular_dependency_detected
expr: |
  sum(increase(victor_dependency_graph_circular_dependencies_total[5m])) > 0
for: 1m
labels:
  severity: critical
annotations:
  summary: "Circular dependency detected"
  description: "Circular dependencies prevent vertical loading"
```

---

#### Alert: Cache Hit Rate Too Low

**Condition**: Hit rate < 50%
```yaml
alert: cache_hit_rate_low
expr: |
  sum(rate(victor_cache_hits_total[5m])) /
  (sum(rate(victor_cache_hits_total[5m])) + sum(rate(victor_cache_misses_total[5m]))) < 0.5
for: 10m
labels:
  severity: critical
annotations:
  summary: "Cache hit rate too low"
  description: "Cache hit rate is {{ $value | humanizePercentage }}"
```

---

### Warning Alerts

#### Alert: High P95 Latency

**Condition**: P95 loading time > 100ms
```yaml
alert: high_p95_latency
expr: |
  histogram_quantile(0.95, sum(rate(victor_vertical_loading_duration_seconds_bucket[5m])) by (le)) > 0.1
for: 5m
labels:
  severity: warning
annotations:
  summary: "High P95 vertical loading latency"
  description: "P95 latency is {{ $value }}s"
```

---

#### Alert: Dependency Graph Deep

**Condition**: Depth > 10
```yaml
alert: dependency_graph_deep
expr: |
  victor_dependency_graph_depth > 10
for: 5m
labels:
  severity: warning
annotations:
  summary: "Dependency graph is deep"
  description: "Graph depth is {{ $value }}"
```

---

## Queries

### PromQL Queries

#### Vertical Loading Performance

```promql
# P50 loading time by vertical
histogram_quantile(0.50, sum(rate(victor_vertical_loading_duration_seconds_bucket[5m])) by (le, vertical_name))

# Vertical loading success rate
sum(rate(victor_verticals_loaded_total{status="success"}[5m])) by (vertical_name) /
sum(rate(victor_verticals_loaded_total[5m])) by (vertical_name)

# Vertical loading error rate by type
sum(rate(victor_verticals_loaded_total{status="error"}[5m])) by (vertical_name, error_type)
```

#### Dependency Graph

```promql
# Graph depth
victor_dependency_graph_depth

# Load order changes
sum(rate(victor_dependency_graph_load_order_changes_total[5m]))

# Circular dependencies
sum(increase(victor_dependency_graph_circular_dependencies_total[5m]))

# Missing dependencies
sum(rate(victor_dependency_graph_missing_dependencies_total[5m])) by (dependency_name)

# P95 resolution duration
histogram_quantile(0.95, victor_dependency_resolution_duration_seconds)
```

#### Cache Performance

```promql
# Overall hit rate
sum(rate(victor_cache_hits_total[5m])) /
(sum(rate(victor_cache_hits_total[5m])) + sum(rate(victor_cache_misses_total[5m])))

# Hit rate by namespace
sum(rate(victor_cache_hits_total[5m])) by (cache_namespace) /
(sum(rate(victor_cache_hits_total[5m])) by (cache_namespace) + sum(rate(victor_cache_misses_total[5m])) by (cache_namespace))

# Cache size
sum(victor_cache_size) by (cache_namespace)

# P95 lock wait duration
histogram_quantile(0.95, victor_cache_lock_wait_duration_seconds)

# Evictions by namespace
sum(rate(victor_cache_evictions_total[5m])) by (cache_namespace)
```

---

## Grafana Dashboard JSON

### Import Dashboard

Use the following JSON to import dashboards into Grafana:

**File**: `victor/grafana/dashboards/vertical_architecture.json`

```json
{
  "dashboard": {
    "title": "Victor Vertical Architecture",
    "tags": ["victor", "verticals", "architecture"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Vertical Loading Time (P95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(victor_vertical_loading_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "P95"
          }
        ]
      },
      {
        "title": "Entry Point Scan Duration",
        "type": "stat",
        "targets": [
          {
            "expr": "victor_entry_point_scan_duration_seconds * 1000"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ms",
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 50, "color": "yellow"},
                {"value": 100, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "title": "Vertical Loading Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(victor_verticals_loaded_total{status=\"success\"}[5m])) / sum(rate(victor_verticals_loaded_total[5m]))"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 95, "color": "red"},
                {"value": 99, "color": "yellow"},
                {"value": 100, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(victor_cache_hits_total[5m])) / (sum(rate(victor_cache_hits_total[5m])) + sum(rate(victor_cache_misses_total[5m])))"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 70, "color": "red"},
                {"value": 80, "color": "yellow"},
                {"value": 90, "color": "green"}
              ]
            }
          }
        }
      },
      {
        "title": "Dependency Graph Depth",
        "type": "gauge",
        "targets": [
          {
            "expr": "victor_dependency_graph_depth"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 20,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 5, "color": "green"},
                {"value": 10, "color": "yellow"},
                {"value": 15, "color": "red"}
              ]
            }
          }
        }
      }
    ]
  }
}
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Prometheus
brew install prometheus  # macOS
apt install prometheus  # Ubuntu

# Install Grafana
brew install grafana  # macOS
apt install grafana  # Ubuntu

# Install OpenTelemetry Collector
pip install opentelemetry-collector
```

### 2. Configure Prometheus

**File**: `/etc/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'victor-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### 3. Configure Grafana

1. Add Prometheus data source
2. Import dashboard JSON
3. Configure alerts
4. Set up notification channels

### 4. Enable OpenTelemetry Export

```python
# victor/config/telemetry.py

from opentelemetry import trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup metrics
metric_reader = PrometheusMetricReader()
meter_provider = MeterProvider(metric_readers=[metric_reader])
trace.set_tracer_provider(meter_provider)

# Setup tracing
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))
```

### 5. Verify Monitoring

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metrics are being exported
curl http://localhost:8000/metrics | grep victor_

# Check Grafana dashboard
open http://localhost:3000/d/victor-vertical-architecture
```

---

## Summary

The monitoring dashboards provide comprehensive observability for the refactored Victor architecture:

- ✅ **4 major dashboards** covering all aspects
- ✅ **20+ panels** tracking key metrics
- ✅ **Alert rules** for critical issues
- ✅ **Grafana JSON** for easy deployment
- ✅ **PromQL queries** for custom monitoring

For rollout plan, see [Rollout Plan](rollout_plan.md).
For deployment procedures, see [Deployment Playbook](deployment_playbook.md).
