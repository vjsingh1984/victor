# Production Metrics and Monitoring Guide

This guide provides comprehensive documentation for monitoring Victor AI in production environments.

## Table of Contents

- [Overview](#overview)
- [Key Metrics](#key-metrics)
- [Metric Definitions](#metric-definitions)
- [Alert Rules](#alert-rules)
- [Dashboard Setup](#dashboard-setup)
- [Metrics Collection](#metrics-collection)
- [Monitoring Stack](#monitoring-stack)

## Overview

Victor AI provides comprehensive observability through Prometheus metrics, Grafana dashboards, and structured logging. The monitoring stack tracks performance, functional, business, and domain-specific metrics.

### Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Victor AI System                        │
│  (MetricsCollector, EventBus, HealthChecker)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Prometheus exposition
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Prometheus Server                         │
│  (Scrape targets, evaluation, alerting)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Alert notifications
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   AlertManager                              │
│  (Deduplication, grouping, routing)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Query and visualization
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Grafana                                   │
│  (Dashboards, alerts, annotations)                         │
└─────────────────────────────────────────────────────────────┘
```

## Key Metrics

### Performance Metrics

| Metric | Type | Description | Thresholds |
|--------|------|-------------|------------|
| `victor_initialization_duration_seconds` | Histogram | Time to initialize system | p95 < 5s, p99 < 10s |
| `victor_request_duration_seconds` | Histogram | Request response time | p95 < 10s, p99 < 30s |
| `victor_chat_request_duration_seconds` | Histogram | Chat request time | p95 < 15s, p99 < 45s |
| `victor_tool_execution_duration_seconds` | Histogram | Tool execution time | p95 < 5s, p99 < 15s |
| `victor_provider_latency_seconds` | Histogram | Provider API latency | p95 < 10s, p99 < 30s |
| `victor_memory_usage_bytes` | Gauge | Current memory usage | < 75% capacity |
| `victor_cpu_usage_percent` | Gauge | Current CPU usage | < 80% |
| `victor_request_rate` | Gauge | Requests per second | Monitor trends |
| `victor_error_rate` | Gauge | Error rate | < 5% |

### Functional Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `victor_tool_executions_total` | Counter | Total tool executions | tool, status, vertical |
| `victor_provider_requests_total` | Counter | Total provider requests | provider, status, model |
| `victor_vertical_usage_total` | Counter | Vertical usage count | vertical, mode |
| `victor_tool_success_rate` | Gauge | Tool success rate | tool, vertical |
| `victor_provider_success_rate` | Gauge | Provider success rate | provider, model |
| `victor_workflow_executions_total` | Counter | Workflow executions | workflow, status |
| `victor_feature_usage_total` | Counter | Feature usage | feature |

### Business Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `victor_total_requests` | Counter | Total requests served |
| `victor_active_users` | Gauge | Current active users |
| `victor_user_sessions_total` | Counter | Total user sessions |
| `victor_session_duration_seconds` | Histogram | Session duration |
| `victor_requests_per_user` | Histogram | Requests per user |

### Agentic AI Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `victor_planning_success_rate` | Gauge | Planning success rate | vertical |
| `victor_memory_recall_accuracy` | Gauge | Memory recall accuracy | memory_type |
| `victor_skill_discovery_total` | Counter | Skills discovered | skill_type |
| `victor_proficiency_score` | Gauge | Current proficiency score | skill |
| `victor_self_improvement_loops_total` | Counter | Self-improvement loops | outcome |
| `victor_memory_operations_total` | Counter | Memory operations | operation, status |

### Vertical-Specific Metrics

#### Coding Vertical

| Metric | Type | Description |
|--------|------|-------------|
| `victor_coding_files_analyzed_total` | Counter | Files analyzed |
| `victor_coding_loc_reviewed_total` | Counter | Lines of code reviewed |
| `victor_coding_issues_found_total` | Counter | Issues found |
| `victor_coding_tests_generated_total` | Counter | Tests generated |
| `victor_coding_refactoring_success_rate` | Gauge | Refactoring success rate |

#### RAG Vertical

| Metric | Type | Description |
|--------|------|-------------|
| `victor_rag_documents_ingested_total` | Counter | Documents ingested |
| `victor_rag_search_accuracy` | Gauge | Search accuracy |
| `victor_rag_retrieval_latency_seconds` | Histogram | Retrieval latency |
| `victor_rag_index_size_bytes` | Gauge | Index size |

#### DevOps Vertical

| Metric | Type | Description |
|--------|------|-------------|
| `victor_devops_deployments_total` | Counter | Deployments performed |
| `victor_devops_deployment_success_rate` | Gauge | Deployment success rate |
| `victor_devops_containers_managed` | Gauge | Containers managed |
| `victor_devops_ci_pipelines_executed_total` | Counter | CI pipelines executed |

#### Data Analysis Vertical

| Metric | Type | Description |
|--------|------|-------------|
| `victor_dataanalysis_queries_total` | Counter | Queries executed |
| `victor_dataanalysis_visualizations_total` | Counter | Visualizations created |
| `victor_dataanalysis_data_size_bytes` | Histogram | Data size analyzed |
| `victor_dataanalysis_query_duration_seconds` | Histogram | Query duration |

#### Research Vertical

| Metric | Type | Description |
|--------|------|-------------|
| `victor_research_searches_total` | Counter | Searches performed |
| `victor_research_citations_generated_total` | Counter | Citations generated |
| `victor_research_sources_analyzed_total` | Counter | Sources analyzed |
| `victor_research_synthesis_duration_seconds` | Histogram | Synthesis time |

### Security Metrics

| Metric | Type | Description | Thresholds |
|--------|------|-------------|------------|
| `victor_security_authorization_success_rate` | Gauge | Authorization success | > 95% |
| `victor_security_failed_authorizations_total` | Counter | Failed authorizations | Monitor spikes |
| `victor_security_test_pass_rate` | Gauge | Security test pass rate | > 90% |
| `victor_security_vulnerabilities_found_total` | Counter | Vulnerabilities found | Track trends |
| `victor_security_scan_duration_seconds` | Histogram | Security scan duration | p95 < 60s |

## Metric Definitions

### Performance Metrics

#### Request Duration
```yaml
metric_name: victor_request_duration_seconds
type: Histogram
description: Request response time in seconds
labels:
  - endpoint: API endpoint (e.g., /chat, /tools/execute)
  - status: HTTP status code (e.g., 200, 400, 500)
  - method: HTTP method (GET, POST, etc.)
buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
```

#### Tool Execution Duration
```yaml
metric_name: victor_tool_execution_duration_seconds
type: Histogram
description: Tool execution time in seconds
labels:
  - tool: Tool name (e.g., read_file, write_file)
  - vertical: Vertical name (coding, rag, etc.)
  - status: Execution status (success, failure, timeout)
buckets: [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0]
```

#### Memory Usage
```yaml
metric_name: victor_memory_usage_bytes
type: Gauge
description: Current memory usage in bytes
labels:
  - component: Component name (orchestrator, providers, etc.)
```

### Functional Metrics

#### Tool Executions
```yaml
metric_name: victor_tool_executions_total
type: Counter
description: Total number of tool executions
labels:
  - tool: Tool name
  - vertical: Vertical name
  - status: Execution status
  - mode: Agent mode (build, plan, explore)
```

#### Provider Requests
```yaml
metric_name: victor_provider_requests_total
type: Counter
description: Total provider API requests
labels:
  - provider: Provider name (anthropic, openai, etc.)
  - model: Model name
  - status: Request status
  - tool_calls: Whether tool calls were used
```

### Business Metrics

#### Active Users
```yaml
metric_name: victor_active_users
type: Gauge
description: Current number of active users
labels:
  - interface: Interface type (cli, tui, api, mcp)
```

#### Session Duration
```yaml
metric_name: victor_session_duration_seconds
type: Histogram
description: User session duration in seconds
labels:
  - interface: Interface type
buckets: [60, 300, 600, 1800, 3600, 7200, 14400]
```

## Alert Rules

### Critical Alerts

#### High Error Rate
```yaml
alert: HighErrorRate
expr: |
  rate(victor_request_duration_seconds_count{status=~"5.."}[5m])
  /
  rate(victor_request_duration_seconds_count[5m]) > 0.05
for: 2m
labels:
  severity: critical
annotations:
  summary: "High error rate detected"
  description: "Error rate is {{ $value | humanizePercentage }} for endpoint {{ $labels.endpoint }}"
```

#### Slow Response Time
```yaml
alert: SlowResponseTime
expr: |
  histogram_quantile(0.95,
    rate(victor_request_duration_seconds_bucket[5m])
  ) > 30
for: 5m
labels:
  severity: critical
annotations:
  summary: "Slow response time detected"
  description: "P95 response time is {{ $value }}s for endpoint {{ $labels.endpoint }}"
```

#### High Memory Usage
```yaml
alert: HighMemoryUsage
expr: victor_memory_usage_bytes / victor_memory_limit_bytes > 0.9
for: 5m
labels:
  severity: critical
annotations:
  summary: "High memory usage detected"
  description: "Memory usage is {{ $value | humanizePercentage }}"
```

#### Provider Error Rate
```yaml
alert: HighProviderErrorRate
expr: |
  rate(victor_provider_requests_total{status="failure"}[5m])
  /
  rate(victor_provider_requests_total[5m]) > 0.1
for: 3m
labels:
  severity: critical
annotations:
  summary: "High provider error rate"
  description: "Provider {{ $labels.provider }} error rate is {{ $value | humanizePercentage }}"
```

#### Security Test Failure
```yaml
alert: SecurityTestFailure
expr: |
  victor_security_test_pass_rate < 0.9
for: 1m
labels:
  severity: critical
annotations:
  summary: "Security test failure detected"
  description: "Security test pass rate is {{ $value | humanizePercentage }}"
```

### Warning Alerts

#### Degraded Response Time
```yaml
alert: DegradedResponseTime
expr: |
  histogram_quantile(0.95,
    rate(victor_request_duration_seconds_bucket[5m])
  ) > 10
for: 5m
labels:
  severity: warning
annotations:
  summary: "Degraded response time"
  description: "P95 response time is {{ $value }}s"
```

#### Elevated Memory Usage
```yaml
alert: ElevatedMemoryUsage
expr: victor_memory_usage_bytes / victor_memory_limit_bytes > 0.75
for: 10m
labels:
  severity: warning
annotations:
  summary: "Elevated memory usage"
  description: "Memory usage is {{ $value | humanizePercentage }}"
```

#### Tool Execution Failure Rate
```yaml
alert: HighToolFailureRate
expr: |
  rate(victor_tool_executions_total{status="failure"}[5m])
  /
  rate(victor_tool_executions_total[5m]) > 0.05
for: 5m
labels:
  severity: warning
annotations:
  summary: "High tool failure rate"
  description: "Tool {{ $labels.tool }} failure rate is {{ $value | humanizePercentage }}"
```

#### Low Vertical Success Rate
```yaml
alert: LowVerticalSuccessRate
expr: |
  victor_tool_success_rate < 0.9
for: 10m
labels:
  severity: warning
annotations:
  summary: "Low vertical success rate"
  description: "Vertical {{ $labels.vertical }} success rate is {{ $value | humanizePercentage }}"
```

## Dashboard Setup

### Grafana Dashboard Installation

1. **Import dashboards**:
   ```bash
   # Navigate to Grafana UI
   # Dashboards -> Import -> Upload JSON file
   # Or use grafana-cli
   grafana-cli dashboards import configs/grafana/dashboard_overview.json
   grafana-cli dashboards import configs/grafana/dashboard_performance.json
   grafana-cli dashboards import configs/grafana/dashboard_verticals.json
   grafana-cli dashboards import configs/grafana/dashboard_errors.json
   ```

2. **Configure Prometheus data source**:
   - URL: `http://prometheus:9090`
   - Access: Server (default)
   - Scrape interval: 15s

3. **Set up alert notifications**:
   - Configure AlertManager webhook
   - Set up notification channels (Slack, email, PagerDuty)

### Dashboard Descriptions

#### Overview Dashboard
**Purpose**: High-level system health and traffic monitoring

**Panels**:
- Request rate (requests/sec)
- Error rate (%)
- P50/P95/P99 latency
- Active users
- System health (up/down)
- Provider status
- Vertical usage distribution

**Refresh**: 30 seconds

#### Performance Dashboard
**Purpose**: Detailed performance metrics

**Panels**:
- Response time percentiles (p50, p95, p99)
- Tool execution time by tool
- Provider latency by provider
- Memory usage over time
- CPU usage over time
- Request rate trends
- Cache hit rates

**Refresh**: 15 seconds

#### Verticals Dashboard
**Purpose**: Vertical-specific metrics

**Panels**:
- Vertical usage (by request count)
- Tool usage by vertical
- Success rates by vertical
- Vertical-specific metrics:
  - Coding: files analyzed, LOC reviewed, issues found
  - RAG: documents ingested, search accuracy
  - DevOps: deployments, containers managed
  - DataAnalysis: queries executed, visualizations created
  - Research: searches performed, citations generated

**Refresh**: 1 minute

#### Errors Dashboard
**Purpose**: Error tracking and analysis

**Panels**:
- Error rate by endpoint
- Error types distribution
- Error trends over time
- Tool failures by tool
- Provider failures by provider
- Security events
- Error rate heatmap (endpoint x time)

**Refresh**: 1 minute

## Metrics Collection

### Metrics Collector

The `MetricsCollector` class (victor/observability/metrics_collector.py) provides:

- Automatic metric collection from system events
- Prometheus metric exposition
- Custom metric registration
- Integration with EventBus
- Periodic metric reporting

### Integration Points

1. **EventBus Integration**:
   ```python
   from victor.core.events import create_event_backend, MessagingEvent

   # Metrics collector subscribes to events
   await event_bus.subscribe("tool.*", metrics_collector.handle_tool_event)
   await event_bus.subscribe("agent.*", metrics_collector.handle_agent_event)
   await event_bus.subscribe("error.*", metrics_collector.handle_error_event)
   ```

2. **Middleware Integration**:
   ```python
   from victor.framework.middleware import MetricsMiddleware

   # Add metrics middleware to orchestrator
   orchestrator.add_middleware(MetricsMiddleware(metrics_collector))
   ```

3. **Health Check Integration**:
   ```python
   from victor.framework.health import HealthChecker

   # Health checker exposes metrics
   health_checker.add_check("metrics", metrics_collector.health_check)
   ```

### Custom Metrics

To add custom metrics:

```python
from prometheus_client import Counter, Histogram, Gauge
from victor.observability.metrics_collector import MetricsCollector

# Register custom metric
custom_counter = Counter(
    'my_custom_metric_total',
    'Description of my custom metric',
    ['label1', 'label2']
)

# Use the metric
custom_counter.labels(label1='value1', label2='value2').inc()

# Register with collector
MetricsCollector.register_metric(custom_counter)
```

## Monitoring Stack

### Local Development Setup

```bash
# Start Prometheus
docker run -d \
  -p 9090:9090 \
  -v $(pwd)/configs/prometheus:/etc/prometheus \
  prom/prometheus

# Start Grafana
docker run -d \
  -p 3000:3000 \
  -v $(pwd)/configs/grafana:/var/lib/grafana/dashboards \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  grafana/grafana

# Start AlertManager
docker run -d \
  -p 9093:9093 \
  -v $(pwd)/configs/alertmanager:/etc/alertmanager \
  prom/alertmanager
```

### Production Deployment

#### Docker Compose

```yaml
version: '3.8'
services:
  victor:
    image: victor-ai:latest
    ports:
      - "8000:8000"
    environment:
      - VICTOR_PROMETHEUS_ENABLED=true
      - VICTOR_PROMETHEUS_PORT=9091

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus:/etc/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    depends_on:
      - victor

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./configs/grafana:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./configs/alertmanager:/etc/alertmanager
    depends_on:
      - prometheus
```

#### Kubernetes Deployment

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    # Contents of configs/prometheus/prometheus.yml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: victor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: victor
  template:
    metadata:
      labels:
        app: victor
    spec:
      containers:
      - name: victor
        image: victor-ai:latest
        ports:
        - containerPort: 8000
        - containerPort: 9091  # Metrics
        env:
        - name: VICTOR_PROMETHEUS_ENABLED
          value: "true"
---
apiVersion: v1
kind: Service
metadata:
  name: victor-metrics
spec:
  selector:
    app: victor
  ports:
  - port: 9091
    targetPort: 9091
    name: metrics
```

## Monitoring Best Practices

### 1. Set Up Alerts Proactively

- Configure critical alerts before deploying to production
- Test alert rules in staging environment
- Set up on-call rotation for critical alerts

### 2. Use Dashboard Templates

- Create dashboard templates for different environments
- Use variables for environment-specific values
- Version control dashboard configurations

### 3. Monitor Trends, Not Just Values

- Track metric trends over time
- Use recording rules for complex queries
- Set up anomaly detection

### 4. Regular Review

- Review alert rules monthly
- Update dashboards based on feedback
- Add new metrics as features evolve

### 5. Performance Baseline

- Establish performance baselines
- Track deviations from baseline
- Use baseline for capacity planning

## Troubleshooting

### Metrics Not Appearing

1. Check Prometheus is scraping:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

2. Verify metrics endpoint:
   ```bash
   curl http://localhost:9091/metrics
   ```

3. Check Prometheus logs:
   ```bash
   docker logs prometheus
   ```

### High Memory Usage

1. Check memory metrics:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=victor_memory_usage_bytes'
   ```

2. Review tool execution:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=victor_tool_execution_duration_seconds_sum'
   ```

3. Check for memory leaks:
   - Compare memory usage over time
   - Look for increasing trend

### Slow Response Time

1. Identify bottleneck:
   - Check tool execution time
   - Check provider latency
   - Check request queue depth

2. Review system resources:
   - CPU usage
   - Memory usage
   - I/O wait

3. Check provider status:
   - Provider error rate
   - Provider latency
   - Rate limit status

## Related Documentation

- [Quick Reference](QUICK_REFERENCE.md) - Metrics system overview
- [Structured Logging](STRUCTURED_LOGGING.md) - Logging setup
- [Health Checks](HEALTH_CHECKS.md) - Health monitoring
- [Prometheus Metrics](PROMETHEUS_METRICS.md) - Detailed Prometheus guide
