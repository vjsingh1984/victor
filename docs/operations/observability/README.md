# Observability Suite for Victor AI 0.5.0

Production-ready observability features including OpenTelemetry tracing, Prometheus metrics, and structured logging.

## Features

### 1. OpenTelemetry Integration
- Automatic span creation for LLM calls
- Tool execution tracing
- Workflow orchestration tracing
- Error tracking and reporting
- Support for OTLP, Console, and Jaeger exporters
- Configurable sampling rates

**Documentation**: [OPENTELEMETRY_GUIDE.md](OPENTELEMETRY_GUIDE.md)

### 2. Prometheus Metrics
- Request counters and latency histograms
- Token usage metrics
- Tool execution metrics
- Cache hit/miss rates
- Provider health status
- HTTP /metrics endpoint

**Documentation**: [PROMETHEUS_METRICS.md](PROMETHEUS_METRICS.md)

### 3. Structured Logging
- JSON format for machine parsing
- Correlation IDs for request tracing
- Request/response logging
- Performance logging (slow operation detection)
- Error context and stack traces
- Sampling for high-volume logs

**Documentation**: [STRUCTURED_LOGGING.md](STRUCTURED_LOGGING.md)

### 4. Health Check System
- Liveness probes (/health)
- Readiness probes (/ready)
- Detailed component health (/health/detailed)
- Kubernetes integration
- Provider health checks
- Resource monitoring

**Documentation**: [HEALTH_CHECKS.md](HEALTH_CHECKS.md)

## Quick Start

### Installation

```bash
# Install observability dependencies
pip install victor-ai[observability]

# Or install all optional dependencies
pip install victor-ai[all]
```

### Basic Configuration

```bash
# Enable all observability features
export VICTOR_TELEMETRY_ENABLED=true
export VICTOR_TELEMETRY_ENDPOINT=http://localhost:4317
export VICTOR_TELEMETRY_SAMPLING=0.5

export VICTOR_LOG_FORMAT=json
export VICTOR_LOG_LEVEL=info

export VICTOR_METRICS_ENABLED=true
export VICTOR_METRICS_PORT=9090
```

### Application Setup

```python
from victor.config.telemetry_config import setup_telemetry, get_telemetry_config
from victor.observability.structured_logging import setup_structured_logging
from victor.observability.metrics_endpoint import start_metrics_server

# 1. Setup OpenTelemetry
config = get_telemetry_config()
if config.enabled:
    tracer, meter = setup_telemetry(config)
    print("OpenTelemetry initialized")

# 2. Setup structured logging
logger, perf_logger, req_logger = setup_structured_logging(
    log_format="json",
    log_level="info",
)
print("Structured logging initialized")

# 3. Start metrics server (optional, or integrate with existing app)
start_metrics_server(port=9090)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Victor AI Application                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Tracing    │  │   Metrics    │  │   Logging    │      │
│  │              │  │              │  │              │      │
│  │ - LLM calls  │  │ - Counters   │  │ - JSON logs  │      │
│  │ - Tools      │  │ - Histograms │  │ - Correlation│      │
│  │ - Workflows  │  │ - Gauges     │  │ - Performance│      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │  HTTP Endpoints │                        │
│                   │                 │                        │
│                   │ - /metrics      │                        │
│                   │ - /health       │                        │
│                   │ - /ready        │                        │
│                   └────────┬────────┘                        │
└────────────────────────────┼────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Prometheus  │    │   Jaeger     │    │   ELK/Loki   │
│              │    │              │    │              │
│  /metrics    │    │  /traces     │    │  /logs       │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Environment Variables

### Telemetry

| Variable | Description | Default |
|----------|-------------|---------|
| `VICTOR_TELEMETRY_ENABLED` | Enable telemetry | `false` |
| `VICTOR_TELEMETRY_EXPORTER` | Exporter type | `otlp` |
| `VICTOR_TELEMETRY_ENDPOINT` | OTLP endpoint | `http://localhost:4317` |
| `VICTOR_TELEMETRY_SAMPLING` | Sampling rate | `1.0` |
| `VICTOR_TELEMETRY_TRACING_ENABLED` | Enable tracing | `true` |
| `VICTOR_TELEMETRY_METRICS_ENABLED` | Enable metrics | `true` |

### Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `VICTOR_LOG_FORMAT` | Log format | `text` |
| `VICTOR_LOG_LEVEL` | Log level | `INFO` |
| `VICTOR_LOG_SAMPLING` | Sampling rate | `1.0` |
| `VICTOR_LOG_REQUEST_ENABLED` | Enable request logging | `true` |
| `VICTOR_LOG_PERFORMANCE_ENABLED` | Enable performance logging | `true` |
| `VICTOR_LOG_SLOW_THRESHOLD_MS` | Slow threshold | `1000` |

### Metrics

| Variable | Description | Default |
|----------|-------------|---------|
| `VICTOR_METRICS_ENABLED` | Enable metrics | `true` |
| `VICTOR_METRICS_PORT` | Metrics port | `9090` |

## Usage Examples

### Tracing LLM Calls

```python
from victor.observability.telemetry import trace_llm_call

with trace_llm_call("anthropic", "claude-sonnet-4-5") as span:
    span.set_prompt(messages, tokens=100)
    response = await provider.chat(messages)
    span.set_response(
        content=response.content,
        tokens=response.usage.total_tokens,
    )
```

### Recording Metrics

```python
from victor.observability.prometheus_metrics import get_prometheus_registry

registry = get_prometheus_registry()

counter = registry.counter(
    name="operations_total",
    help="Total operations",
)
counter.inc()

histogram = registry.histogram(
    name="operation_latency_ms",
    help="Operation latency",
)
histogram.observe(45.2)
```

### Structured Logging

```python
from victor.observability.structured_logging import set_correlation_id
import logging

logger = logging.getLogger(__name__)

set_correlation_id("req-123")
logger.info(
    "Processing request",
    extra={"endpoint": "/api/chat", "user_id": "456"}
)
```

### Health Checks

```python
from victor.core.health import HealthChecker, ProviderHealthCheck

checker = HealthChecker()
checker.add_check(ProviderHealthCheck("anthropic", provider))

report = await checker.check_health()
print(f"Status: {report.status}")
```

## Docker Compose Example

Complete observability stack with Victor, Jaeger, Prometheus, and Grafana:

```yaml
version: '3.8'

services:
  victor:
    image: victor-ai:0.5.0
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      VICTOR_TELEMETRY_ENABLED: "true"
      VICTOR_TELEMETRY_ENDPOINT: "http://jaeger:4317"
      VICTOR_TELEMETRY_SAMPLING: "0.5"
      VICTOR_LOG_FORMAT: "json"
      VICTOR_LOG_LEVEL: "info"
      VICTOR_METRICS_ENABLED: "true"
      VICTOR_METRICS_PORT: "9090"

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "4317:4317"
      - "16686:16686"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  prometheus-data:
  grafana-data:
```

## Performance Considerations

### Sampling

Reduce overhead with sampling:

```bash
# Sample 50% of traces
export VICTOR_TELEMETRY_SAMPLING=0.5

# Sample 10% of debug logs
export VICTOR_LOG_SAMPLING=0.1
```

### Batch Configuration

Optimize for high throughput:

```bash
export VICTOR_TELEMETRY_BATCH_TIMEOUT=10
export VICTOR_TELEMETRY_BATCH_MAX_SIZE=1024
```

### Selective Tracing

Disable specific features:

```bash
export VICTOR_TELEMETRY_TRACING_ENABLED=false
export VICTOR_TELEMETRY_METRICS_ENABLED=true
```

## Integration Guides

### Kubernetes

See [HEALTH_CHECKS.md](HEALTH_CHECKS.md) for:
- Liveness and readiness probes
- Deployment configuration
- Resource management
- HPA integration

### Prometheus

See [PROMETHEUS_METRICS.md](PROMETHEUS_METRICS.md) for:
- Available metrics
- Grafana dashboards
- Alerting rules
- Query examples

### Jaeger/Tempo

See [OPENTELEMETRY_GUIDE.md](OPENTELEMETRY_GUIDE.md) for:
- Trace visualization
- Service diagrams
- Trace analysis
- Performance insights

### Log Aggregation

See [STRUCTURED_LOGGING.md](STRUCTURED_LOGGING.md) for:
- ELK Stack integration
- Grafana Loki setup
- CloudWatch configuration
- Log parsing

## Troubleshooting

### No traces appearing

1. Verify OTLP endpoint: `curl http://localhost:4317`
2. Check environment variables: `env | grep TELEMETRY`
3. Enable debug logging: `export OPENTELEMETRY_EXPORTER_OTLP_LOG_LEVEL=debug`

### High memory usage

1. Reduce sampling rate
2. Decrease batch size
3. Disable unused features

### Health checks failing

1. Check component status: `curl http://localhost:8080/health/detailed`
2. Verify dependencies (providers, caches)
3. Review timeouts and thresholds

## Best Practices

1. **Start with console exporter** for development
2. **Use sampling in production** (10-50%)
3. **Set meaningful correlation IDs**
4. **Monitor health check latency**
5. **Set up alerting for critical metrics**
6. **Review dashboards regularly**
7. **Test observability locally first**
8. **Document custom metrics**

## Additional Resources

- **Main Documentation**: [../index.md](../index.md)
- **Architecture**: [../architecture/overview.md](../architecture/overview.md)
- **Examples**: `examples/observability/`

## Support

For issues and questions:
- GitHub Issues: [https://github.com/vijayksingh/victor/issues](https://github.com/vijayksingh/victor/issues)
- Documentation: [https://github.com/vijayksingh/victor/docs](https://github.com/vijayksingh/victor/docs)

## License

Apache License 2.0 - see LICENSE file for details
