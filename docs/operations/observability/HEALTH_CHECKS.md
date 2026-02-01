# Health Check System Guide

Comprehensive guide for health checks in Victor AI.

## Overview

Victor AI provides production-ready health checks for:
- **Liveness**: Is the service running?
- **Readiness**: Can the service handle traffic?
- **Component Health**: Detailed component status
- **Dependency Checks**: Providers, databases, caches
- **Resource Monitoring**: Memory, CPU, connections

## Quick Start

### 1. Health Check Endpoints

```python
from victor.observability.metrics_endpoint import create_observability_app

app = create_observability_app()

# Endpoints:
# - /health - Liveness probe
# - /ready - Readiness probe
# - /health/detailed - Full health report
```

### 2. Configure Kubernetes Probes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: victor
spec:
  containers:
    - name: victor
      image: victor-ai:latest
      ports:
        - containerPort: 8080
      livenessProbe:
        httpGet:
          path: /health
          port: 8080
        initialDelaySeconds: 30
        periodSeconds: 10
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 5
```

## Health Check Types

### Liveness Probe

**Endpoint**: `/health`

Determines if the container is running. If this fails, Kubernetes restarts the container.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": 1737179696.789,
  "service": "victor-observability"
}
```

**Usage**: Kubernetes livenessProbe

### Readiness Probe

**Endpoint**: `/ready`

Determines if the container can handle traffic. If this fails, Kubernetes stops sending traffic.

**Response**:
```json
{
  "status": "ready",
  "timestamp": 1737179696.789
}
```

**Usage**: Kubernetes readinessProbe

### Detailed Health

**Endpoint**: `/health/detailed`

Provides comprehensive health status of all components.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-18T12:34:56.789Z",
  "version": "0.5.0",
  "uptime_seconds": 3600.5,
  "components": {
    "provider.anthropic": {
      "name": "provider.anthropic",
      "status": "healthy",
      "message": "Provider is responsive",
      "latency_ms": 45.2,
      "last_check": "2025-01-18T12:34:55.123Z",
      "consecutive_failures": 0
    },
    "tool.read_file": {
      "name": "tool.read_file",
      "status": "healthy",
      "message": "Tool is properly configured"
    },
    "cache.redis": {
      "name": "cache.redis",
      "status": "healthy",
      "message": "Cache is operational"
    },
    "system.memory": {
      "name": "system.memory",
      "status": "healthy",
      "message": "Memory usage: 512MB",
      "details": {
        "rss_mb": 512.0,
        "warning_threshold_mb": 1000,
        "critical_threshold_mb": 2000
      }
    }
  }
}
```

## Built-in Health Checks

### Provider Health Check

```python
from victor.core.health import ProviderHealthCheck

# Check provider connectivity
check = ProviderHealthCheck(
    name="anthropic",
    provider=anthropic_provider,
    timeout=10.0,
)

health = await check.check()
print(f"Status: {health.status}")
print(f"Message: {health.message}")
```

### Tool Health Check

```python
from victor.core.health import ToolHealthCheck

# Check tool configuration
check = ToolHealthCheck(
    name="read_file",
    tool=read_file_tool,
    timeout=5.0,
    critical=False,
)

health = await check.check()
print(f"Status: {health.status}")
```

### Cache Health Check

```python
from victor.core.health import CacheHealthCheck

# Check cache connectivity
check = CacheHealthCheck(
    name="redis",
    cache=redis_cache,
    timeout=3.0,
    critical=False,
)

health = await check.check()
print(f"Status: {health.status}")
```

### Memory Health Check

```python
from victor.core.health import MemoryHealthCheck

# Check memory usage
check = MemoryHealthCheck(
    warning_threshold_mb=1000,
    critical_threshold_mb=2000,
)

health = await check.check()
print(f"Status: {health.status}")
print(f"Memory: {health.details['rss_mb']}MB")
```

### Custom Health Check

```python
from victor.core.health import CallableHealthCheck

async def check_database():
    try:
        await db.ping()
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database is responsive",
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database error: {e}",
        )

check = CallableHealthCheck(
    name="database",
    check_fn=check_database,
    timeout=5.0,
    critical=True,
)
```

## Health Check Configuration

### Creating Health Checker

```python
from victor.core.health import HealthChecker, create_default_health_checker

# Create with default checks
checker = create_default_health_checker(
    version="0.5.0",
    include_memory=True,
)

# Or create custom
checker = HealthChecker(cache_ttl=5.0, version="0.5.0")

# Add checks
checker.add_check(ProviderHealthCheck("anthropic", provider))
checker.add_check(ToolHealthCheck("read_file", read_tool))
checker.add_check(CacheHealthCheck("redis", cache))
checker.add_check(MemoryHealthCheck())

# Get health report
report = await checker.check_health()
print(f"Overall: {report.status}")
```

### Health Status Values

| Status | Description | Kubernetes Action |
|--------|-------------|-------------------|
| `healthy` | All components healthy | Traffic enabled |
| `degraded` | Non-critical components unhealthy | Traffic enabled |
| `unhealthy` | Critical components unhealthy | Traffic stopped |
| `unknown` | Status cannot be determined | Traffic enabled |

### Critical vs Non-Critical

```python
# Critical component failure = unhealthy status
ProviderHealthCheck("anthropic", provider, critical=True)

# Non-critical component failure = degraded status
CacheHealthCheck("redis", cache, critical=False)
```

## Kubernetes Configuration

### Deployment Example

```yaml
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
          image: victor-ai:0.5.0
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 9090
              name: metrics

          # Liveness probe - restart if failed
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          # Readiness probe - stop traffic if failed
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 2

          # Startup probe - slow starting containers
          startupProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 0
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 30

          env:
            - name: VICTOR_ENV
              value: "production"
            - name: VICTOR_TELEMETRY_ENABLED
              value: "true"

          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
```

### Service Example

```yaml
apiVersion: v1
kind: Service
metadata:
  name: victor
spec:
  selector:
    app: victor
  ports:
    - name: http
      port: 80
      targetPort: http
    - name: metrics
      port: 9090
      targetPort: metrics
  type: ClusterIP
```

## Monitoring Health Checks

### Prometheus Metrics

```python
from victor.observability.production_health import create_production_health_checker

health_checker = create_production_health_checker()

# Health check will update metrics automatically
# Metrics available:
# - victor_health_status{component="..."}
# - vixtor_health_latency_ms{component="..."}
# - victor_health_consecutive_failures{component="..."}
```

### Alerting Rules

```yaml
groups:
  - name: victor_health
    rules:
      - alert: VictorUnhealthy
        expr: |
          victor_health_status{component=~"provider.*"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Victor provider unhealthy"
          description: "Provider {{ $labels.component }} is unhealthy"

      - alert: VictorHighFailureRate
        expr: |
          rate(victor_health_consecutive_failures[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High health check failure rate"
```

## Best Practices

1. **Set appropriate timeouts**:
   - Liveness: 5-10 seconds
   - Readiness: 3-5 seconds
   - Component checks: 5-30 seconds

2. **Use startup probes** for slow-starting containers:
```yaml
startupProbe:
  failureThreshold: 30
  periodSeconds: 5
  # Gives 30 * 5 = 150 seconds for startup
```

3. **Cache health results**:
```python
checker = HealthChecker(cache_ttl=5.0)  # Cache for 5 seconds
```

4. **Set criticality appropriately**:
   - Critical: Providers, core services
   - Non-critical: Caches, optional tools

5. **Monitor health check latency**:
```python
# Track health check performance
report = await checker.check_health()
for name, health in report.components.items():
    if health.latency_ms > 1000:
        logger.warning(f"Slow health check: {name}")
```

6. **Use status change callbacks**:
```python
def on_status_change(old_status, new_status):
    if new_status == HealthStatus.UNHEALTHY:
        send_alert(f"System became {new_status.value}")

checker.on_status_change(on_status_change)
```

## Troubleshooting

### Health checks failing

1. Check component logs:
```python
report = await checker.check_health()
for name, health in report.components.items():
    print(f"{name}: {health.message}")
```

2. Verify dependencies:
```bash
# Check provider connectivity
curl https://api.anthropic.com/v1/messages

# Check cache connectivity
redis-cli ping
```

3. Review timeouts:
```python
# Increase timeout for slow components
check = ProviderHealthCheck("provider", provider, timeout=30.0)
```

### Container restarting

1. Check liveness probe configuration
2. Verify health endpoint is accessible
3. Review application logs

### Traffic not routing

1. Check readiness probe status
2. Verify dependencies are ready
3. Review resource limits

## Examples

See `examples/observability/` for complete examples:
- `health_checks.py`: Basic health checks
- `kubernetes_deployment.yml`: Full K8s deployment
- `custom_health_checks.py`: Custom checks
- `health_monitoring.py`: Monitoring setup

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
