# Coordinator Orchestrator Monitoring Guide

**Version**: 1.0
**Date**: 2025-01-14
**Audience**: DevOps Engineers, SREs, System Administrators

---

## Table of Contents

1. [Monitoring Overview](#monitoring-overview)
2. [Key Metrics to Monitor](#key-metrics-to-monitor)
3. [Alert Thresholds](#alert-thresholds)
4. [Dashboard Setup](#dashboard-setup)
5. [Log Analysis](#log-analysis)
6. [Incident Response](#incident-response)
7. [Monitoring Tools](#monitoring-tools)
8. [Runbooks](#runbooks)

---

## Monitoring Overview

### Why Monitor Coordinator Orchestrator?

The coordinator-based orchestrator introduces new moving parts that require monitoring:

- **15 Coordinators**: Each with specific responsibilities
- **Inter-coordinator Communication**: Potential bottlenecks
- **Feature Flag Status**: Know which architecture is active
- **Performance Impact**: Ensure < 10% overhead goal is met
- **Error Rates**: Detect failures quickly

### Monitoring Strategy

**Three-Tier Approach**:

1. **Health Checks** (Every 1 minute)
   - Coordinator initialization status
   - Feature flag status
   - Basic connectivity

2. **Performance Metrics** (Every 5 minutes)
   - Latency (chat, tool execution)
   - Throughput (requests/minute)
   - Resource usage (memory, CPU)

3. **Business Metrics** (Every 15 minutes)
   - Error rates by coordinator
   - Feature parity tracking
   - User impact assessment

---

## Key Metrics to Monitor

### 1. Feature Flag Status

**Metric**: `coordinator_orchestrator_enabled`

**Type**: Gauge

**Description**: Whether the coordinator orchestrator is currently enabled

**Labels**:
- `source`: "environment_variable" | "settings_file"
- `version`: Victor version

**Example**:
```python
from victor.config.settings import Settings

settings = Settings()
print(f"Coordinators: {settings.use_coordinator_orchestrator}")
```

**Alert**: N/A (informational only)

### 2. Coordinator Initialization

**Metric**: `coordinator_initialization_seconds`

**Type**: Histogram

**Description**: Time taken to initialize all coordinators

**Labels**:
- `coordinator`: ConfigCoordinator, PromptCoordinator, etc.
- `status`: "success" | "failure"

**Target**:
- Total initialization: < 500ms
- Per coordinator: < 50ms

**Alert**:
- Warning: > 500ms total
- Critical: > 1000ms total

### 3. Chat Latency

**Metric**: `chat_latency_seconds`

**Type**: Histogram

**Description**: End-to-end latency for chat requests (includes coordinator overhead)

**Labels**:
- `orchestrator_type`: "coordinator" | "legacy"
- `model`: Model name
- `provider`: Provider name

**Target**:
- Coordinator overhead: < 10% increase vs legacy
- Absolute: < 5 seconds for typical requests

**Alert**:
- Warning: > 10% increase vs legacy baseline
- Critical: > 15% increase vs legacy baseline

**Example Query**:
```promql
# Compare coordinator vs legacy latency
rate(chat_latency_seconds_sum{orchestrator_type="coordinator"}[5m])
/
rate(chat_latency_seconds_sum{orchestrator_type="legacy"}[5m])
```

### 4. Tool Execution Latency

**Metric**: `tool_execution_seconds`

**Type**: Histogram

**Description**: Time taken to execute tools (coordinators add minimal overhead)

**Labels**:
- `tool_name`: Tool being executed
- `coordinator`: ToolCoordinator
- `status`: "success" | "failure"

**Target**: No change from legacy (coordinators shouldn't impact tool execution)

**Alert**:
- Warning: > 5% increase vs baseline
- Critical: > 10% increase vs baseline

### 5. Coordinator Errors

**Metric**: `coordinator_errors_total`

**Type**: Counter

**Description**: Total errors by coordinator

**Labels**:
- `coordinator`: Coordinator name
- `error_type`: Error category
- `severity`: "critical" | "warning" | "info"

**Target**: 0 errors (all should be logged and handled)

**Alert**:
- Warning: > 0 errors in 5 minutes
- Critical: > 10 errors in 5 minutes

**Example Query**:
```promql
# Error rate by coordinator
rate(coordinator_errors_total[5m]) > 0
```

### 6. Memory Usage

**Metric**: `coordinator_memory_bytes`

**Type**: Gauge

**Description**: Memory used by coordinator orchestrator vs legacy

**Labels**:
- `orchestrator_type`: "coordinator" | "legacy"
- `component": "total" | "coordinators" | "overhead"

**Target**:
- Overhead: < 100MB additional memory
- Total: < 500MB for typical usage

**Alert**:
- Warning: Overhead > 100MB
- Critical: Overhead > 200MB

### 7. Coordinator Interaction Count

**Metric**: `coordinator_interactions_total`

**Type**: Counter

**Description**: Number of times coordinators interact (e.g., ChatCoordinator calls ToolCoordinator)

**Labels**:
- `source_coordinator`: Caller
- `target_coordinator`: Callee
- `operation`: Operation type

**Target**: Baseline establishment (no specific threshold)

**Alert**: N/A (informational for optimization)

### 8. Feature Parity Score

**Metric**: `feature_parity_score`

**Type**: Gauge

**Description**: Percentage of features working in coordinator orchestrator vs legacy

**Labels**:
- `feature_category`: "chat" | "tools" | "sessions" | "analytics"

**Target**: 100% (all features work)

**Alert**:
- Warning: < 95%
- Critical: < 90%

### 9. Test Coverage

**Metric**: `coordinator_test_coverage_percent`

**Type**: Gauge

**Description**: Test coverage percentage for coordinators

**Labels**:
- `coordinator`: Specific coordinator or "all"

**Target**: > 85% coverage

**Alert**:
- Warning: < 85%
- Critical: < 75%

### 10. Backward Compatibility

**Metric**: `backward_compatibility_failures_total`

**Type**: Counter

**Description**: Failures due to backward compatibility issues

**Labels**:
- `api_version`: API version being used
- `failure_type`: Type of incompatibility

**Target**: 0 failures

**Alert**:
- Critical: Any failure (indicates breaking change)

---

## Alert Thresholds

### Severity Levels

| Severity | Response Time | Examples | Action |
|----------|--------------|----------|--------|
| **Critical** | < 15 minutes | Service down, data loss, security | Immediate rollback |
| **Warning** | < 1 hour | Performance degradation, elevated errors | Investigate, prepare rollback |
| **Info** | Next business day | Trends, capacity planning | Monitor, document |

### Alert Rules

#### Critical Alerts (Immediate Action Required)

| Alert | Condition | Threshold | Action |
|-------|-----------|-----------|--------|
| Coordinator initialization failure | `coordinator_errors_total{severity="critical"} > 0` | Any | Rollback immediately |
| Complete service failure | `up{job="victor"} == 0` | Any | Restart service, investigate |
| Memory leak | `rate(coordinator_memory_bytes[10m]) > 10MB` | Sustained | Restart, investigate leak |
| Backward compatibility failure | `backward_compatibility_failures_total > 0` | Any | Rollback immediately |
| Data corruption | `session_corruption_total > 0` | Any | Rollback immediately |

#### Warning Alerts (Investigate Within 1 Hour)

| Alert | Condition | Threshold | Action |
|-------|-----------|-----------|--------|
| Performance degradation | `chat_latency_seconds` > baseline × 1.10 | > 10% increase | Investigate bottleneck |
| Elevated error rate | `rate(coordinator_errors_total[5m]) > 0.1/second` | > 6 errors/min | Check logs, optimize |
| Memory overhead high | `coordinator_memory_bytes - legacy_memory_bytes > 100MB` | > 100MB | Profile memory usage |
| Feature parity drop | `feature_parity_score < 0.95` | < 95% | Fix missing feature |
| Test coverage drop | `coordinator_test_coverage_percent < 85` | < 85% | Improve tests |

#### Info Alerts (Monitor and Document)

| Alert | Condition | Threshold | Action |
|-------|-----------|-----------|--------|
| Coordinator interactions spike | `rate(coordinator_interactions_total[5m]) > baseline × 2` | > 2× normal | Optimize if needed |
| Resource usage trend | `predict_linear(coordinator_memory_bytes[1h], 86400) > limit` | Predicted exceed | Plan capacity |
| Usage pattern change | `request_rate` changes significantly | ±50% | Document new normal |

---

## Dashboard Setup

### Grafana Dashboard Example

```json
{
  "dashboard": {
    "title": "Coordinator Orchestrator Monitoring",
    "panels": [
      {
        "title": "Feature Flag Status",
        "targets": [
          {
            "expr": "coordinator_orchestrator_enabled"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Chat Latency Comparison",
        "targets": [
          {
            "expr": "rate(chat_latency_seconds_sum{orchestrator_type=\"coordinator\"}[5m]) / rate(chat_latency_seconds_count{orchestrator_type=\"coordinator\"}[5m])",
            "legendFormat": "Coordinator"
          },
          {
            "expr": "rate(chat_latency_seconds_sum{orchestrator_type=\"legacy\"}[5m]) / rate(chat_latency_seconds_count{orchestrator_type=\"legacy\"}[5m])",
            "legendFormat": "Legacy"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Coordinator Errors",
        "targets": [
          {
            "expr": "rate(coordinator_errors_total[5m])",
            "legendFormat": "{{coordinator}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Memory Overhead",
        "targets": [
          {
            "expr": "coordinator_memory_bytes{orchestrator_type=\"coordinator\"} - coordinator_memory_bytes{orchestrator_type=\"legacy\"}",
            "legendFormat": "Overhead"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Feature Parity Score",
        "targets": [
          {
            "expr": "feature_parity_score",
            "legendFormat": "{{feature_category}}"
          }
        ],
        "type": "gauge"
      }
    ]
  }
}
```

### Key Dashboard Views

**1. Overview Dashboard** (High-Level Health)
- Feature flag status
- Overall health score
- Active sessions
- Request rate

**2. Performance Dashboard** (Latency and Throughput)
- Chat latency (coordinator vs legacy)
- Tool execution latency
- Coordinator initialization time
- P95/P99 latencies

**3. Error Dashboard** (Failures and Issues)
- Error rate by coordinator
- Error types distribution
- Recent error log
- Backward compatibility failures

**4. Resource Dashboard** (Memory and CPU)
- Memory usage breakdown
- CPU usage trends
- Coordinator memory overhead
- Resource efficiency

**5. Feature Parity Dashboard** (Completeness)
- Feature parity score
- Test coverage
- Known gaps
- Migration progress

---

## Log Analysis

### Enabling Observability Logging

```bash
# Enable JSONL logging for dashboard
export VICTOR_ENABLE_OBSERVABILITY_LOGGING=true
export VICTOR_OBSERVABILITY_LOG_PATH=~/.victor/metrics/victor.jsonl

# Run Victor
victor chat
```

### Log Format

Each log entry is a JSON line:

```json
{
  "timestamp": "2025-01-14T10:30:00Z",
  "event_type": "coordinator_initialized",
  "coordinator": "ChatCoordinator",
  "duration_ms": 12,
  "success": true
}
```

### Analyzing Logs

**Count coordinator initializations**:
```bash
cat ~/.victor/metrics/victor.jsonl | \
  jq -r 'select(.event_type == "coordinator_initialized") | .coordinator' | \
  sort | uniq -c
```

**Calculate average latency**:
```bash
cat ~/.victor/metrics/victor.jsonl | \
  jq -r 'select(.event_type == "chat_completed") | .duration_ms' | \
  awk '{sum+=$1; count++} END {print "Average:", sum/count, "ms"}'
```

**Find errors**:
```bash
cat ~/.victor/metrics/victor.jsonl | \
  jq -r 'select(.success == false) | {timestamp, event_type, error}'
```

**Compare coordinator vs legacy**:
```bash
# Coordinator performance
cat ~/.victor/metrics/victor_coordinator.jsonl | \
  jq -r 'select(.event_type == "chat_completed") | .duration_ms' | \
  awk '{sum+=$1; count++} END {print "Coordinator:", sum/count, "ms"}'

# Legacy performance (from earlier logs)
cat ~/.victor/metrics/victor_legacy.jsonl | \
  jq -r 'select(.event_type == "chat_completed") | .duration_ms' | \
  awk '{sum+=$1; count++} END {print "Legacy:", sum/count, "ms"}'
```

### Log Retention

**Recommendation**: Keep logs for 30 days

```bash
# Rotate logs daily
mv ~/.victor/metrics/victor.jsonl ~/.victor/metrics/victor_$(date +%Y%m%d).jsonl

# Compress old logs
gzip ~/.victor/metrics/victor_*.jsonl

# Delete logs older than 30 days
find ~/.victor/metrics -name "victor_*.jsonl.gz" -mtime +30 -delete
```

---

## Incident Response

### Incident Categories

#### P0: Critical (Service Down)

**Examples**:
- Coordinator initialization failures preventing startup
- Complete service failure
- Data corruption or loss

**Response Time**: < 15 minutes

**Escalation**: Immediately notify tech lead

**Runbook**:
1. Verify feature flag status
2. Check error logs
3. If coordinator-related, disable feature flag
4. Restart service
5. Verify service restored
6. Document incident

#### P1: High (Degraded Performance)

**Examples**:
- 15%+ performance degradation
- Elevated error rates affecting users
- Feature parity gaps blocking workflows

**Response Time**: < 1 hour

**Escalation**: Notify tech lead, begin investigation

**Runbook**:
1. Confirm performance degradation
2. Identify affected coordinator
3. Check logs for errors
4. If unresolved after 30 min, consider rollback
5. Document findings

#### P2: Medium (Elevated Errors)

**Examples**:
- 5-15% performance degradation
- Non-critical errors occurring
- Minor feature gaps

**Response Time**: < 4 hours

**Escalation**: Create ticket, investigate

**Runbook**:
1. Quantify impact
2. Identify root cause
3. Implement fix or workaround
4. Monitor for recurrence

#### P3: Low (Informational)

**Examples**:
- Trends worth monitoring
- Capacity planning needs
- Documentation improvements

**Response Time**: Next business day

**Escalation**: Create backlog item

### Incident Response Template

```markdown
# Coordinator Orchestrator Incident

**Severity**: P0/P1/P2/P3
**Start Time**: YYYY-MM-DD HH:MM:SS UTC
**Status**: Investigating / Mitigated / Resolved

## Summary
[Brief description of incident]

## Impact
- Users affected: [number/percentage]
- Duration: [time period]
- Features affected: [list]

## Timeline
- HH:MM - Incident detected
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Mitigation implemented
- HH:MM - Service restored

## Root Cause
[What happened and why]

## Resolution
[What was done to fix it]

## Follow-up Actions
- [ ] [Action item 1]
- [ ] [Action item 2]

## Lessons Learned
[What can we improve]
```

---

## Monitoring Tools

### 1. Built-in Observability

**Victor JSONL Logging**:
```bash
export VICTOR_ENABLE_OBSERVABILITY_LOGGING=true
```

**Pros**:
- No external dependencies
- Works offline
- Simple setup

**Cons**:
- No visualization
- Manual analysis required

### 2. Prometheus + Grafana (Recommended)

**Setup**:
```python
# Add to Victor (future enhancement)
from prometheus_client import start_http_server, Counter, Histogram

coordinator_errors = Counter(
    'coordinator_errors_total',
    'Total coordinator errors',
    ['coordinator', 'error_type']
)

chat_latency = Histogram(
    'chat_latency_seconds',
    'Chat request latency',
    ['orchestrator_type', 'model']
)

# Start metrics server
start_http_server(8000)
```

**Pros**:
- Industry standard
- Rich visualization
- Powerful querying
- Alert management

**Cons**:
- Additional infrastructure
- Requires network access

### 3. Datadog / New Relic

**Integration**:
```python
from datadog import statsd

# Send metrics
statsd.increment('coordinator.errors', tags=['coordinator:ChatCoordinator'])
statsd.timing('chat.latency', 1234, tags=['orchestrator:coordinator'])
```

**Pros**:
- Hosted solution
- Rich dashboards
- Alert management included

**Cons**:
- Cost
- External dependency
- Requires API keys

### 4. CloudWatch (AWS)

**Integration**:
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='Victor/Coordinator',
    MetricData=[{
        'MetricName': 'ChatLatency',
        'Value': 1234,
        'Unit': 'Milliseconds'
    }]
)
```

**Pros**:
- Native AWS integration
- No additional infrastructure

**Cons**:
- AWS vendor lock-in
- Limited query capabilities

---

## Runbooks

### Runbook: Coordinator Performance Degradation

**Symptom**: Chat latency increased > 10% after enabling coordinators

**Steps**:

1. **Verify the issue**:
   ```bash
   # Check current latency
   python scripts/validate_coordinator_orchestrator.py --quick
   ```

2. **Identify the bottleneck**:
   ```bash
   # Check coordinator initialization time
   cat ~/.victor/metrics/victor.jsonl | \
     jq 'select(.event_type == "coordinator_initialized") | \
         {coordinator, duration_ms}'
   ```

3. **Disable non-critical coordinators**:
   ```yaml
   # In ~/.victor/profiles.yaml
   enable_analytics: false  # Try disabling
   enable_metrics: false
   ```

4. **Check for specific coordinator issues**:
   ```bash
   # Look for errors in logs
   grep -i error ~/.victor/logs/victor.log | grep -i coordinator
   ```

5. **If unresolved, rollback**:
   ```bash
   python scripts/toggle_coordinator_orchestrator.py disable --backup
   ```

6. **Document and report**:
   - Create GitHub issue with details
   - Include validation report
   - Attach log samples

### Runbook: Coordinator Initialization Failure

**Symptom**: Victor fails to start with coordinator errors

**Steps**:

1. **Check error message**:
   ```bash
   victor chat --no-tui 2>&1 | tee error.log
   cat error.log
   ```

2. **Verify feature flag**:
   ```bash
   python -c "from victor.config.settings import Settings; \
              print(Settings().use_coordinator_orchestrator)"
   ```

3. **Test coordinator imports**:
   ```bash
   python -c "from victor.agent.coordinators import ConfigCoordinator; \
              print('OK')"
   ```

4. **Check for dependency issues**:
   ```bash
   pip list | grep -E "(pydantic|yaml)"
   ```

5. **Rollback to legacy**:
   ```bash
   python scripts/toggle_coordinator_orchestrator.py disable --backup
   ```

6. **Report the issue**:
   - Include full error traceback
   - List Victor version and dependencies
   - Note any recent changes

### Runbook: Memory Leak Detected

**Symptom**: Memory usage growing unbounded

**Steps**:

1. **Confirm memory leak**:
   ```bash
   # Monitor memory over time
   watch -n 10 'ps aux | grep victor | awk "{print \$6}"'
   ```

2. **Identify source**:
   ```bash
   # Check if coordinator-related
   export VICTOR_USE_COORDINATOR_ORCHESTRATOR=false
   victor chat  # Test with legacy

   # If leak stops, it's coordinator-related
   ```

3. **Profile memory usage**:
   ```bash
   python -m memory_profiler victor/chat
   ```

4. **Check for specific coordinator**:
   - Disable analytics
   - Disable metrics
   - Test systematically

5. **Rollback if critical**:
   ```bash
   python scripts/toggle_coordinator_orchestrator.py disable --backup
   ```

6. **Report with profiling data**:
   - Include memory profiler output
   - Note growth rate (MB/minute)
   - List coordinators enabled

### Runbook: Feature Parity Gap Discovered

**Symptom**: Feature works in legacy but not in coordinator orchestrator

**Steps**:

1. **Isolate the feature**:
   - What specific functionality is broken?
   - Which coordinator is responsible?

2. **Reproduce the issue**:
   ```bash
   # Test with coordinator
   export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true
   victor chat --no-tui

   # Test with legacy
   export VICTOR_USE_COORDINATOR_ORCHESTRATOR=false
   victor chat --no-tui
   ```

3. **Check coordinator implementation**:
   - Look at relevant coordinator code
   - Compare to legacy implementation

4. **Create test case**:
   ```python
   # Add test demonstrating the gap
   def test_feature_x():
       # Test that should pass
       pass
   ```

5. **Implement fix**:
   - Update coordinator
   - Add test coverage
   - Verify fix works

6. **Update documentation**:
   - Note feature parity score change
   - Document fix in release notes

---

## Related Documentation

- [Production Checklist](coordinator_rollback_checklist.md)
- [Settings Guide](coordinator_settings.md)
- [Migration Guide](../migration/orchestrator_refactoring_guide.md)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-14
**Next Review**: 2025-02-14
