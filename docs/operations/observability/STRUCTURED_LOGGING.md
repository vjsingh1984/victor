# Structured Logging Guide

Comprehensive guide for structured JSON logging in Victor AI.

## Overview

Victor AI provides production-ready structured logging with:
- **JSON Format**: Machine-parseable log output
- **Correlation IDs**: Request tracking across logs
- **Request/Response Logging**: HTTP operation logging
- **Performance Logging**: Slow operation detection
- **Error Context**: Stack traces and error details
- **Sampling**: Reduce high-volume log noise

## Installation

No additional dependencies required - uses Python standard library.

## Quick Start

### 1. Enable Structured Logging

```bash
# Set log format to JSON
export VICTOR_LOG_FORMAT=json

# Set log level
export VICTOR_LOG_LEVEL=info
```

### 2. Initialize in Application

```python
from victor.observability.structured_logging import (
    setup_structured_logging,
    set_correlation_id,
)

# Setup structured logging
logger, perf_logger, req_logger = setup_structured_logging(
    log_format="json",
    log_level="info",
)

# Use correlation IDs
set_correlation_id("req-123")
logger.info("Processing request")
```

## Log Format

### JSON Structure

```json
{
  "timestamp": "2025-01-18T12:34:56.789Z",
  "level": "INFO",
  "logger": "victor.agent",
  "message": "Processing request",
  "service": "victor",
  "environment": "production",
  "module": "agent",
  "function": "process_request",
  "line": 42,
  "correlation_id": "req-123",
  "http": {
    "method": "POST",
    "path": "/api/chat",
    "status_code": 200
  }
}
```

### Text Structure

```
2025-01-18 12:34:56 - victor.agent - INFO - Processing request
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VICTOR_LOG_FORMAT` | Log format (json, text) | `text` |
| `VICTOR_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |
| `VICTOR_LOG_SAMPLING` | Sampling rate for high-volume logs | `1.0` |
| `VICTOR_LOG_REQUEST_ENABLED` | Enable request/response logging | `true` |
| `VICTOR_LOG_PERFORMANCE_ENABLED` | Enable performance logging | `true` |
| `VICTOR_LOG_REQUEST_BODY` | Log request/response bodies | `false` |
| `VICTOR_LOG_SLOW_THRESHOLD_MS` | Slow operation threshold | `1000` |
| `VICTOR_LOG_FILE` | Log file path | None |
| `VICTOR_LOG_FILE_LEVEL` | Log file level | Same as console |
| `VICTOR_LOG_DISABLED` | Disable file logging | `false` |

## Usage Patterns

### Basic Logging

```python
import logging

logger = logging.getLogger(__name__)

# Basic log
logger.info("Operation completed")

# With extra context
logger.info(
    "User logged in",
    extra={"user_id": "123", "ip": "192.168.1.1"}
)

# With structured data
logger.warning(
    "High memory usage",
    extra={
        "memory_mb": 1024,
        "threshold_mb": 1000,
        "usage_percent": 85.5,
    }
)
```

### Correlation IDs

```python
from victor.observability.structured_logging import (
    set_correlation_id,
    get_correlation_id,
)

# Set correlation ID for request
set_correlation_id("req-" + str(uuid.uuid4()))

# All logs will include correlation_id
logger.info("Processing request")

# Get current correlation ID
corr_id = get_correlation_id()
```

### Performance Logging

```python
from victor.observability.structured_logging import PerformanceLogger

perf_logger = PerformanceLogger(
    logger=logger,
    slow_threshold_ms=1000,
    enabled=True,
)

# Context manager
with perf_logger.track_operation(
    operation_name="database_query",
    metadata={"table": "users", "query_type": "select"}
):
    result = db.query("SELECT * FROM users")

# Decorator
@perf_logger.track_operation("api_call")
def external_api_call():
    return requests.get("https://api.example.com")
```

### Request/Response Logging

```python
from victor.observability.structured_logging import RequestLogger

req_logger = RequestLogger(
    logger=logger,
    enabled=True,
    log_body=False,  # Set True to log bodies
)

# Log incoming request
req_logger.log_request(
    method="POST",
    path="/api/chat",
    headers={"user-agent": "Mozilla/5.0", "content-type": "application/json"},
    body='{"message": "hello"}',
    correlation_id="req-123",
)

# Process request...

# Log response
req_logger.log_response(
    status_code=200,
    duration_ms=45.2,
    headers={"content-type": "application/json"},
    body='{"response": "hi there"}',
)
```

### Error Logging

```python
import logging

try:
    risky_operation()
except Exception as e:
    # With exception info
    logger.error("Operation failed", exc_info=True)

    # Or use logging.exception
    logger.exception("Operation failed")

    # With custom context
    logger.error(
        "Database error",
        exc_info=True,
        extra={
            "database": "postgres",
            "query": "SELECT * FROM users",
            "error_code": getattr(e, 'code', None),
        }
    )
```

### Sampling High-Volume Logs

```python
from victor.observability.structured_logging import SamplingLogger

# Sample 10% of debug logs
sampled_logger = SamplingLogger(
    logger=logger,
    sample_rate=0.1,
)

# These will be sampled
sampled_logger.debug("Processing item: item-123")
sampled_logger.debug("Processing item: item-124")

# Warnings and errors are never sampled
sampled_logger.warning("This will always be logged")
sampled_logger.error("This will always be logged")
```

## Integration with Log Aggregators

### Elasticsearch (ELK Stack)

Logstash configuration:

```ruby
input {
  file {
    path => "/var/log/victor/*.log"
    codec => json
  }
}

filter {
  # Parse correlation ID
  if [correlation_id] {
    mutate { add_field => { "[@metadata][correlation_id]" => "%{correlation_id}" } }
  }

  # Parse HTTP data
  if [http] {
    mutate {
      add_field => {
        "[@metadata][endpoint]" => "%{[http][path]}"
        "[@metadata][status_code]" => "%{[http][status_code]}"
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "victor-%{+YYYY.MM.dd}"
  }
}
```

### Grafana Loki

Promtail configuration:

```yaml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

scrape_configs:
  - job_name: victor
    static_configs:
      - targets:
          - localhost
        labels:
          job: victor
          __path__: /var/log/victor/*.log

    pipeline_stages:
      - json:
          expressions:
            level: level
            logger: logger
            correlation_id: correlation_id

      - labels:
          level:
          logger:
          correlation_id:
```

### CloudWatch

AWS CloudWatch agent:

```yaml
logs:
  logs_collected:
    files:
      collect_list:
        - file_path: /var/log/victor/app.log
          log_group_name: /aws/victor/production
          log_stream_name: "{instance_id}"
          timestamp_format: "%Y-%m-%dT%H:%M:%S.%fZ"

          # Parse JSON logs
          multi_line_start_pattern: "{"
```

## Best Practices

1. **Use appropriate log levels**:
   - DEBUG: Detailed diagnostics
   - INFO: Normal operations
   - WARNING: Degraded performance
   - ERROR: Errors requiring attention

2. **Include structured context**:
```python
logger.info(
    "API request completed",
    extra={
        "endpoint": "/api/users",
        "duration_ms": 45,
        "status_code": 200,
    }
)
```

3. **Use correlation IDs**:
```python
# Set at request start
set_correlation_id(request_id)

# All subsequent logs include it
logger.info("Processing started")
logger.info("Processing completed")
```

4. **Log errors with context**:
```python
try:
    operation()
except Exception as e:
    logger.error(
        "Operation failed",
        exc_info=True,
        extra={
            "operation": "database_query",
            "query": query,
            "error_type": type(e).__name__,
        }
    )
```

5. **Sanitize sensitive data**:
```python
def sanitize_headers(headers):
    sensitive = ["authorization", "cookie", "x-api-key"]
    return {
        k: "***REDACTED***" if k.lower() in sensitive else v
        for k, v in headers.items()
    }
```

6. **Use sampling for high-volume logs**:
```python
# Instead of logging every item
for item in items:
    sampled_logger.debug(f"Processing {item}")

# Use sampling
sampled_logger = SamplingLogger(logger, sample_rate=0.01)
```

## Performance Considerations

### File Logging

```python
# Use rotating file handler
logger, _, _ = setup_structured_logging(
    log_format="json",
    log_file="/var/log/victor/app.log",
)
```

### Asynchronous Logging

For high-throughput scenarios, consider async logging:

```python
from logging.handlers import QueueHandler, QueueListener
import queue

# Create queue
log_queue = queue.Queue(maxsize=1000)

# Queue handler for application
queue_handler = QueueHandler(log_queue)
logger.addHandler(queue_handler)

# Separate thread for writing
file_handler = logging.FileHandler("app.log")
listener = QueueListener(log_queue, file_handler)
listener.start()
```

### Sampling

Reduce log volume:

```bash
# Sample 1% of debug logs
export VICTOR_LOG_SAMPLING=0.01
export VICTOR_LOG_LEVEL=debug
```

## Troubleshooting

### Logs not appearing

1. Check log level:
```python
import logging
print(logging.getLogger().level)
```

2. Verify handlers:
```python
print(logging.getLogger().handlers)
```

3. Check environment variables:
```bash
env | grep VICTOR_LOG
```

### Missing correlation IDs

Ensure correlation ID is set before logging:

```python
set_correlation_id(request_id)
logger.info("Processing")  # Will include correlation_id
```

### Performance issues

1. Reduce log level
2. Enable sampling
3. Use asynchronous logging
4. Filter verbose modules

## Examples

See `examples/observability/` for complete examples:
- `structured_logging.py`: Basic setup
- `correlation_ids.py`: Request tracing
- `performance_logging.py`: Operation timing
- `request_logging.py`: HTTP logging
- `log_aggregation.py`: ELK/Loki integration

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
