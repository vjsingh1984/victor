# Observability Quick Reference

Quick reference for Victor AI 0.5.0 observability features.

## Installation

```bash
pip install victor-ai[observability]
```

## Environment Variables

### Enable All Features
```bash
export VICTOR_TELEMETRY_ENABLED=true
export VICTOR_TELEMETRY_ENDPOINT=http://localhost:4317
export VICTOR_TELEMETRY_SAMPLING=0.5
export VICTOR_LOG_FORMAT=json
export VICTOR_LOG_LEVEL=info
export VICTOR_METRICS_ENABLED=true
export VICTOR_METRICS_PORT=9090
```

## Quick Start

### Python Setup
```python
from victor.config.telemetry_config import setup_telemetry, get_telemetry_config
from victor.observability.structured_logging import setup_structured_logging
from victor.observability.metrics_endpoint import start_metrics_server

# Setup
config = get_telemetry_config()
tracer, meter = setup_telemetry(config)
logger, perf_logger, req_logger = setup_structured_logging(log_format="json")

# Start server (optional)
start_metrics_server(port=9090)
```

## Common Patterns

### Trace LLM Call
```python
from victor.observability.telemetry import trace_llm_call

with trace_llm_call("anthropic", "claude-sonnet-4-5") as span:
    span.set_prompt(messages, tokens=100)
    response = await provider.chat(messages)
    span.set_response(response.content, tokens=200)
```

### Record Metric
```python
from victor.observability.prometheus_metrics import get_prometheus_registry

registry = get_prometheus_registry()
counter = registry.counter("operations_total", help="Total operations")
counter.inc()
```

### Log with Correlation
```python
from victor.observability.structured_logging import set_correlation_id
import logging

logger = logging.getLogger(__name__)
set_correlation_id("req-123")
logger.info("Processing", extra={"endpoint": "/api/chat"})
```

### Health Check
```python
from victor.core.health import HealthChecker, ProviderHealthCheck

checker = HealthChecker()
checker.add_check(ProviderHealthCheck("anthropic", provider))
report = await checker.check_health()
print(f"Status: {report.status}")
```

## HTTP Endpoints

| Endpoint | Purpose |
|----------|---------|
| `http://localhost:9090/metrics` | Prometheus scraping |
| `http://localhost:9090/health` | Liveness probe |
| `http://localhost:9090/ready` | Readiness probe |
| `http://localhost:9090/health/detailed` | Component health |

## URLs

| Service | URL |
|---------|-----|
| Jaeger UI | http://localhost:16686 |
| Prometheus | http://localhost:9091 |
| Grafana | http://localhost:3000 |

## Docker Compose

```yaml
services:
  victor:
    image: victor-ai:0.5.0
    environment:
      VICTOR_TELEMETRY_ENABLED: "true"
      VICTOR_TELEMETRY_ENDPOINT: "http://jaeger:4317"
      VICTOR_LOG_FORMAT: "json"
      VICTOR_METRICS_ENABLED: "true"
    ports:
      - "8080:8080"
      - "9090:9090"

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "4317:4317"
      - "16686:16686"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

## Troubleshooting

### No traces
```bash
# Check endpoint
curl http://localhost:4317

# Check env
env | grep TELEMETRY
```

### No metrics
```bash
# Check endpoint
curl http://localhost:9090/metrics

# Check Prometheus
curl http://localhost:9091/api/v1/targets
```

### Health check failing
```bash
# Check status
curl http://localhost:9090/health/detailed | jq
```

## Performance

| Feature | Overhead |
|---------|----------|
| Tracing (sampling=1.0) | 5-10% |
| Logging (JSON) | 2-5% |
| Metrics | 1-2% |
| Health Checks (cached) | 0.5% |

## Documentation

- [README](README.md) - Overview
- [OpenTelemetry](OPENTELEMETRY_GUIDE.md) - Tracing
- [Prometheus](PROMETHEUS_METRICS.md) - Metrics
- [Logging](STRUCTURED_LOGGING.md) - Logs
- [Health](HEALTH_CHECKS.md) - Health checks
- [Implementation](IMPLEMENTATION_SUMMARY.md) - Details

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 1 min
