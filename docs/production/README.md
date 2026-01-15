# Victor Production Monitoring and Observability

This directory contains production-grade monitoring and observability configurations for the Victor coordinator-based orchestrator.

## Contents

### Python Modules

Located in `/Users/vijaysingh/code/codingagent/victor/observability/`:

1. **prometheus_metrics.py** - Prometheus metrics exporter
   - Prometheus-compatible metrics export
   - Counter, Gauge, Histogram metric types
   - HTTP `/metrics` endpoint
   - FastAPI integration support
   - Standalone server mode

2. **health.py** - Health check endpoints
   - `/health` endpoint for overall health
   - `/health/ready` for Kubernetes readiness probes
   - `/health/live` for Kubernetes liveness probes
   - `/health/detailed` for comprehensive health information
   - Coordinator-specific health checks
   - Error rate and latency threshold monitoring

3. **coordinator_logging.py** - Structured logging
   - JSON format for log aggregation
   - Request ID tracking for distributed tracing
   - Coordinator-specific log levels
   - Contextual logging with metadata
   - Performance tracking
   - Error logging with stack traces
   - Text format with colors for development

### Production Documentation

Located in `/Users/vijaysingh/code/codingagent/docs/production/`:

1. **prometheus_alerts.yml** - Prometheus alerting rules
   - High error rate alerts (5%, 15%)
   - High latency alerts (5s, 15s)
   - Low cache hit rate alerts (< 50%)
   - Cache miss burst detection
   - Coordinator availability alerts
   - Memory alerts (2GB, 4GB)
   - Throughput alerts
   - Health status alerts
   - Infrastructure alerts

2. **grafana_dashboard.json** - Grafana dashboard configuration
   - Pre-configured panels for all metrics
   - Throughput, latency, error rate graphs
   - Memory and CPU usage visualization
   - Cache performance monitoring
   - Service status indicators
   - Analytics events distribution

3. **coordinator_runbook.md** - Operations runbook
   - Common issues and resolutions
   - Performance tuning guidelines
   - Capacity planning recommendations
   - Emergency procedures
   - Maintenance tasks
   - Monitoring setup instructions
   - Contact and escalation paths

## Quick Start

### 1. Setup Prometheus Integration

```python
from victor.observability import (
    get_coordinator_metrics_collector,
    get_prometheus_exporter,
)
from fastapi import FastAPI

# Create FastAPI app
app = FastAPI()

# Get Prometheus exporter
exporter = get_prometheus_exporter()

# Add metrics endpoint
app.add_route("/metrics", exporter.get_endpoint())

# Start server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Metrics will be available at `http://localhost:8000/metrics`

### 2. Setup Health Check Endpoints

```python
from victor.observability.health import setup_health_endpoints
from fastapi import FastAPI

app = FastAPI()

# Setup health endpoints
setup_health_endpoints(app)

# Endpoints available:
# GET /health - Overall health
# GET /health/ready - Readiness probe
# GET /health/live - Liveness probe
# GET /health/detailed - Comprehensive health
```

### 3. Configure Structured Logging

```python
from victor.observability.coordinator_logging import (
    setup_coordinator_logging,
    get_coordinator_logger,
)

# Setup logging (production)
setup_coordinator_logging(
    level="INFO",
    format_type="json",
    output_file="/var/log/victor/coordinators.log",
)

# Or development mode
setup_coordinator_logging(
    level="DEBUG",
    format_type="text",
)

# Get logger
logger = get_coordinator_logger("ChatCoordinator")
logger.info("Processing request", extra={
    "request_id": "abc-123",
    "user_id": "user@example.com",
})
```

### 4. Configure Prometheus

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'victor-coordinators'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 30s

rule_files:
  - "docs/production/prometheus_alerts.yml"
```

### 5. Import Grafana Dashboard

1. Open Grafana: http://localhost:3000
2. Go to Dashboards → Import
3. Upload `docs/production/grafana_dashboard.json`
4. Select Prometheus datasource
5. View dashboard

### 6. Setup AlertManager

Configure `alertmanager.yml`:

```yaml
route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'

receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '<PAGERDUTY_KEY>'
```

## Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Victor Application                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Coordinators │→ │  Metrics     │→ │  Prometheus  │      │
│  └──────────────┘  │  Collector   │  └──────────────┘      │
│  ┌──────────────┐  └──────────────┘         ↓              │
│  │ Health Check │→  FastAPI /metrics   ┌──────────────┐    │
│  └──────────────┘                       │ Prometheus   │    │
│  ┌──────────────┐                       │ Alerts      │    │
│  │ Structured   │→  JSON Logs      └──────────────┘      │
│  │ Logging      │         ↓                              │
│  └──────────────┘   ┌──────────────┐                       │
│                     │ Log          │                       │
│                     │ Aggregator   │                       │
│                     └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌──────────────┐
                    │  Grafana     │
                    │  Dashboard   │
                    └──────────────┘
```

## Key Metrics

| Metric | Type | Description | Healthy Threshold |
|--------|------|-------------|-------------------|
| `victor_coordinator_executions_total` | Counter | Total executions | Increasing |
| `victor_coordinator_errors_total` | Counter | Total errors | Low rate |
| `victor_coordinator_duration_seconds_total` | Counter | Total duration | - |
| `victor_coordinator_cache_hit_rate` | Gauge | Cache hit rate | > 0.8 |
| `victor_coordinator_memory_bytes` | Gauge | Memory usage | < 2GB |
| `victor_coordinator_cpu_percent` | Gauge | CPU usage | < 80% |
| `victor_coordinator_throughput` | Gauge | Requests/sec | > 0.1 |
| `victor_coordinator_error_rate` | Gauge | Error rate | < 0.01 |

## Alert Levels

### Warning Alerts
- Error rate > 5%
- Latency > 5s
- Cache hit rate < 50%
- Memory > 2GB
- CPU > 80%

### Critical Alerts
- Error rate > 15%
- Latency > 15s
- Memory > 4GB
- Service down
- Coordinator unavailable

## Troubleshooting

### High Error Rate

1. Check which coordinator is failing:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=rate(victor_coordinator_errors_total[5m])/rate(victor_coordinator_executions_total[5m])'
   ```

2. Check logs:
   ```bash
   jq 'select(.level == "ERROR")' /var/log/victor/coordinators.log | tail -50
   ```

3. Follow runbook: [coordinator_runbook.md#high-error-rate](coordinator_runbook.md#high-error-rate)

### High Latency

1. Check latency breakdown:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=rate(victor_coordinator_duration_seconds_total[5m])/rate(victor_coordinator_executions_total[5m])'
   ```

2. Check for slow queries:
   ```bash
   jq 'select(.duration_ms > 2000)' /var/log/victor/coordinators.log | tail -20
   ```

3. Follow runbook: [coordinator_runbook.md#high-latency](coordinator_runbook.md#high-latency)

### Cache Issues

1. Check cache hit rate:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=victor_coordinator_cache_hit_rate'
   ```

2. Follow runbook: [coordinator_runbook.md#cache-performance](coordinator_runbook.md#cache-performance)

## Maintenance

### Daily
- Check error rates in Grafana
- Review active alerts
- Verify health endpoints

### Weekly
- Review performance metrics
- Check for anomalies
- Review recent logs

### Monthly
- Full system audit
- Capacity planning review
- Performance tuning

### Quarterly
- Load testing
- Disaster recovery drill
- Architecture review

## Additional Resources

- [Coordinator Runbook](coordinator_runbook.md)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Victor Documentation](https://docs.victor.ai)

## Support

- **Issues**: Create GitHub issue
- **Incidents**: `#victor-incidents` Slack
- **Email**: victor-ops@example.com
- **On-Call**: +1-555-0100

---

**Last Updated**: 2025-01-14
**Version**: 0.5.1
**Maintained By**: Victor Operations Team
