# Production Monitoring and Observability Guide

This guide covers the production-grade monitoring and observability system implemented for Victor AI.

## Overview

The production monitoring system provides:

- **Comprehensive Metrics Collection**: Request rate, latency, errors, cache performance, system resources
- **Distributed Tracing**: End-to-end request tracing with OpenTelemetry integration
- **Health Checks**: Kubernetes-style probes for liveness, readiness, and startup
- **Alerting**: Multi-channel alerting with rules, severity levels, and deduplication
- **Dashboards**: Grafana dashboards for real-time monitoring
- **Prometheus Integration**: Native Prometheus metrics export

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Victor AI Service                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Metrics    │  │   Tracing    │  │   Health     │      │
│  │  Collector   │  │    System    │  │   Checks     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                 │
│  ┌────────────────────────▼──────────────────────────┐     │
│  │              Monitoring Middleware                 │     │
│  └────────────────────────┬──────────────────────────┘     │
└───────────────────────────┼───────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
    ┌─────────┐       ┌─────────┐       ┌──────────┐
    │Prometheus│       │ Grafana │       │AlertManager│
    └─────────┘       └─────────┘       └──────────┘
```

## Components

### 1. ProductionMetricsCollector

Enhanced metrics collection with comprehensive tracking.

```python
from victor.observability import ProductionMetricsCollector, create_production_collector

# Create collector
collector = create_production_collector()

# Record request metrics
collector.record_request(
    endpoint="/chat",
    provider="anthropic",
    success=True,
    latency_ms=123.4,
    labels={"model": "claude-sonnet-4-5"}
)

# Record tool execution
collector.record_tool_execution(
    tool_name="read_file",
    success=True,
    duration_ms=45.2
)

# Record cache metrics
collector.record_cache_hit("tool_cache")
collector.record_cache_miss("tool_cache")

# Get summary
summary = collector.get_summary()
print(summary)
```

#### Metric Categories

- **Request Metrics**: Total requests, success rate, error rate, latency (p50, p95, p99)
- **Tool Metrics**: Tool calls, errors, duration, active executions
- **Cache Metrics**: Hits, misses, hit rate, evictions
- **Business Metrics**: Tasks completed, failed, total cost, tokens used
- **System Metrics**: Memory, CPU, disk, file descriptors, uptime

#### Export Formats

```python
# Prometheus text format
prometheus_text = collector.export_prometheus()

# JSON format
json_data = collector.export_json()
```

### 2. DistributedTracer

End-to-end tracing with OpenTelemetry integration.

```python
from victor.observability import DistributedTracer, trace_function

# Create tracer
tracer = DistributedTracer("victor.agent")

# Use as context manager
with tracer.start_span("process_request") as span:
    span.set_attribute("user_id", "123")

    # Nested spans
    with tracer.start_span("call_llm"):
        # LLM call
        pass

    with tracer.start_span("execute_tool"):
        # Tool execution
        pass

# Use as decorator
@trace_function("my_function")
def my_function(arg1, arg2):
    # Automatically traced
    pass
```

#### Trace Context Propagation

```python
# Inject trace context into HTTP headers
headers = {}
tracer.inject_context(headers)

# Extract trace context from incoming request
ctx = tracer.extract_context(request.headers)
if ctx:
    # Continue trace
    pass
```

### 3. ProductionHealthChecker

Kubernetes-style health probes.

```python
from victor.observability import ProductionHealthChecker, create_production_health_checker

# Create checker
checker = create_production_health_checker(
    startup_timeout=60.0,
    service_version="1.0.0"
)

# Add liveness checks (is process alive?)
checker.add_liveness_check("process", lambda: True)
checker.add_async_liveness_check("memory", check_memory_async)

# Add readiness checks (can it handle traffic?)
checker.add_readiness_check("database", lambda: db.is_connected())
checker.add_readiness_check("cache", lambda: cache.is_ready())

# Add startup checks (has it finished starting?)
checker.add_startup_check("models", lambda: models.loaded())
checker.add_startup_check("warmup", lambda: cache.is_warmed_up())

# Use in HTTP endpoints
@app.get("/health/live")
async def liveness():
    response = await checker.liveness()
    return JSONResponse(response.to_dict())

@app.get("/health/ready")
async def readiness():
    response = await checker.readiness()
    return JSONResponse(response.to_dict())

@app.get("/health/startup")
async def startup():
    response = await checker.startup()
    return JSONResponse(response.to_dict())
```

### 4. MonitoringMiddleware

Automatic request monitoring for web frameworks.

```python
from victor.observability import MonitoringMiddleware, create_monitoring_middleware
from fastapi import FastAPI

app = FastAPI()
middleware = create_monitoring_middleware()

@app.middleware("http")
async def monitoring_middleware(request, call_next):
    return await middleware.handle_request(request, call_next)

# Metrics and traces are now collected automatically
```

### 5. AlertManager

Multi-channel alerting with rules and severity levels.

```python
from victor.observability import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    SlackNotifier,
    EmailNotifier,
    WebhookNotifier,
    create_alert_manager,
)

# Create alert manager
manager = create_alert_manager()

# Add notification channels
slack = SlackNotifier(webhook_url="https://hooks.slack.com/...")
manager.add_notifier("slack", slack)

email = EmailNotifier(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="alerts@victor.ai",
    password="***",
    to_addresses=["oncall@victor.ai"]
)
manager.add_notifier("email", email)

webhook = WebhookNotifier("https://your-webhook-url.com/alerts")
manager.add_notifier("webhook", webhook)

# Define alert rules
high_error_rate_rule = (
    AlertRule.builder()
    .name("high_error_rate")
    .description("Error rate exceeds 5%")
    .condition(lambda: get_error_rate() > 5.0)
    .severity(AlertSeverity.CRITICAL)
    .notification_channels(["slack", "email"])
    .cooldown(300)  # 5 minutes
    .build()
)

manager.add_rule(high_error_rate_rule)

# Check and fire alerts
async def monitor_loop():
    while True:
        await manager.check_and_alert()
        await asyncio.sleep(60)
```

#### Alert Severity Levels

- **INFO**: Informational alerts
- **WARNING**: Warning alerts (degraded performance)
- **ERROR**: Error alerts (service impacted)
- **CRITICAL**: Critical alerts (service down or severe degradation)

## Deployment

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'victor'
    static_configs:
      - targets: ['victor-api:8000']
    metrics_path: '/metrics'

rule_files:
  - 'config/alerting/alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### AlertManager Configuration

```yaml
# alertmanager.yml
global:
  slack_api_url: 'https://hooks.slack.com/services/...'

route:
  receiver: 'default-receiver'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 12h

receivers:
  - name: 'default-receiver'
    slack_configs:
      - channel: '#victor-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: victor-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: victor-api
  template:
    metadata:
      labels:
        app: victor-api
    spec:
      containers:
      - name: victor
        image: victor-ai:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          failureThreshold: 30
          periodSeconds: 10
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Grafana Dashboard

Import the dashboard from `config/dashboards/victor-metrics.json` into Grafana.

The dashboard includes:

- Request rate and latency
- Error rate
- Tool execution metrics
- Cache performance
- System resources (memory, CPU, disk)
- Business metrics (tasks, costs, tokens)
- Active requests
- Service uptime

## Alert Rules

The alert rules in `config/alerting/alerts.yml` cover:

- **API Health**: High error rate, high latency, service down
- **Tool Execution**: High tool error rate, slow tool execution
- **Cache Performance**: Low cache hit rate
- **System Resources**: High memory/CPU usage
- **Business Logic**: High task failure rate, cost spikes
- **Availability**: Service down, low request rate
- **Coordinator**: Coordinator failures, slow execution
- **Token Usage**: High token usage rate, usage spikes

## Best Practices

### 1. Metric Naming

- Use descriptive names with units
- Follow Prometheus naming conventions
- Include labels for dimensions

```python
# Good
collector.record_request(
    endpoint="/chat",
    provider="anthropic",
    success=True,
    latency_ms=123.4,
    labels={"model": "claude-sonnet-4-5"}
)

# Bad
collector.record_request("/chat", True, 123.4)
```

### 2. Alert Thresholds

- Set thresholds based on SLAs
- Use appropriate severity levels
- Include cooldown periods to prevent alert fatigue

```python
rule = (
    AlertRule.builder()
    .name("high_latency")
    .condition(lambda: get_p95_latency() > 5000)  # 5 seconds
    .severity(AlertSeverity.WARNING)
    .cooldown(300)  # Don't alert more than once per 5 minutes
    .build()
)
```

### 3. Health Checks

- Keep liveness checks lightweight (just check if process is alive)
- Include dependency checks in readiness probes
- Use startup probes for slow-initializing services

```python
# Liveness - just check process is running
checker.add_liveness_check("process", lambda: True)

# Readiness - check dependencies
checker.add_readiness_check("database", lambda: db.ping() < 100)
checker.add_readiness_check("cache", lambda: cache.is_connected())

# Startup - check initialization
checker.add_startup_check("models", lambda: models.is_loaded())
checker.add_startup_check("warmup", lambda: cache.is_warmed_up())
```

### 4. Distributed Tracing

- Use hierarchical spans for nested operations
- Add attributes for context (user IDs, request IDs, etc.)
- Propagate trace context across service boundaries

```python
with tracer.start_span("process_request") as parent_span:
    parent_span.set_attribute("user_id", user_id)
    parent_span.set_attribute("request_id", request_id)

    with tracer.start_span("call_llm", parent=parent_span) as child_span:
        child_span.set_attribute("model", model_name)
        child_span.set_attribute("tokens", token_count)
```

## Troubleshooting

### High Memory Usage

1. Check Grafana dashboard for memory trends
2. Identify metric collectors with high cardinality
3. Adjust metric retention policies
4. Check for memory leaks in custom tools

### High CPU Usage

1. Check which tools are CPU-intensive
2. Review alert evaluation frequency
3. Optimize health check intervals
4. Profile coordinator execution

### Missing Metrics

1. Verify Prometheus scrape configuration
2. Check metric export endpoint is accessible
3. Review logs for export errors
4. Ensure collectors are initialized

### Alert Fatigue

1. Increase cooldown periods
2. Adjust severity thresholds
3. Add hysteresis to alert conditions
4. Use alert grouping effectively

## Performance Considerations

### Metrics Collection Overhead

- Metrics collection adds ~1-5ms per request
- System metrics collection runs every 30 seconds
- Consider sampling for high-traffic scenarios

### Tracing Overhead

- Tracing adds ~1-3ms per span
- Use sampling for production (e.g., 10% of requests)
- Export spans asynchronously to avoid blocking

### Health Check Impact

- Keep checks lightweight (<100ms)
- Use async checks for I/O operations
- Cache results when appropriate

## Monitoring Strategy

### Development Environment

- Enable all metrics and tracing
- Use debug logging
- Set low alert thresholds
- Monitor resource usage

### Staging Environment

- Production-like monitoring configuration
- Test alert rules and notifications
- Validate dashboards
- Load test with monitoring

### Production Environment

- Sample traces (10-20%)
- Set appropriate alert thresholds
- Use log aggregation
- Monitor system resources
- Regular alert rule reviews

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Kubernetes Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
- [AlertManager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
