# Event Bridge SLO Dashboard

**Version**: 1.0
**Last Updated**: 2026-03-10
**Epic**: E4 - Event Bridge Reliability

---

## Overview

This dashboard defines Service Level Objectives (SLOs) and monitoring for the Victor Event Bridge, which bridges internal EventBus messages to WebSocket clients for real-time updates.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  EventBus   │──────│ EventBridge  │──────│  WebSocket  │
│  (Internal) │      │ (Reliability)│      │   Clients   │
└─────────────┘      └──────────────┘      └─────────────┘
                            │
                            ├─── Metrics Collection
                            ├─── SLO Validation
                            └─── Alerting
```

---

## SLO Definitions

### Primary SLOs

| SLO | Description | Target | Measurement Window |
|-----|-------------|--------|-------------------|
| **Delivery Success Rate** | % of events successfully delivered to subscribed clients | >= 99.9% | Rolling 5 minutes |
| **Dispatch Latency (p95)** | 95th percentile time from event creation to client delivery | < 200ms | Rolling 5 minutes |
| **Subscription Coverage** | % of event types with active subscriptions | 100% | Per broadcast |

### Secondary SLOs

| SLO | Description | Target | Measurement Window |
|-----|-------------|--------|-------------------|
| **Client Connection Success** | % of successful client connections | >= 99% | Rolling 1 hour |
| **Queue Depth** | Event queue depth (indicator of backpressure) | < 100 | Instantaneous |
| **Broadcast Throughput** | Events dispatched per second | >= 100/s | Rolling 1 minute |

---

## Metrics Collected

### Delivery Metrics

| Metric | Type | Description | Source |
|--------|------|-------------|--------|
| `events_dispatched` | Counter | Total events dispatched | EventBroadcaster |
| `send_successes` | Counter | Successful sends to clients | EventBroadcaster |
| `send_failures` | Counter | Failed sends to clients | EventBroadcaster |
| `delivery_success_rate` | Gauge | Success rate (successes/attempts) | Computed |

### Latency Metrics

| Metric | Type | Description | Source |
|--------|------|-------------|--------|
| `dispatch_latency_ms` | Histogram | Time from event creation to dispatch | EventBroadcaster |
| `dispatch_latency_p95_ms` | Gauge | 95th percentile dispatch latency | Computed |
| `dispatch_latency_p99_ms` | Gauge | 99th percentile dispatch latency | Computed |

### Client Metrics

| Metric | Type | Description | Source |
|--------|------|-------------|--------|
| `client_count` | Gauge | Currently connected clients | EventBroadcaster |
| `total_send_attempts` | Counter | Total send attempts (success + failure) | EventBroadcaster |

---

## SLO Status Calculation

### Delivery Success Rate

```python
total_attempts = send_successes + send_failures
delivery_success_rate = send_successes / total_attempts if total_attempts > 0 else 1.0

slo_status = delivery_success_rate >= 0.999  # 99.9% target
```

**Alert Thresholds**:
- 🔴 **Critical**: < 99.5% (missing 0.5% of events)
- 🟡 **Warning**: < 99.9% (missing 0.1% of events)
- 🟢 **Healthy**: >= 99.9%

### Dispatch Latency (p95)

```python
dispatch_latency_p95_ms = percentile(dispatch_latency_ms_window, 95.0)

slo_status = dispatch_latency_p95_ms < 200.0  # 200ms target
```

**Alert Thresholds**:
- 🔴 **Critical**: > 500ms (2.5x SLO)
- 🟡 **Warning**: > 200ms (above SLO)
- 🟢 **Healthy**: < 200ms

### Subscription Coverage

```python
event_types = [
    "tool.start", "tool.progress", "tool.complete", "tool.error",
    "file.created", "file.modified", "file.deleted",
    "provider.switch", "provider.error", "provider.recovery",
    "session.start", "session.end", "session.error",
    "metrics.update", "budget.warning", "budget.exhausted",
    "notification", "error"
]

subscribed_count = len([et for et in event_types if has_subscription(et)])
coverage_rate = subscribed_count / len(event_types)

slo_status = coverage_rate == 1.0  # 100% target
```

---

## Dashboard Access

### Programmatic Access

```python
from victor.integrations.api.event_bridge import EventBroadcaster

# Get the singleton broadcaster
broadcaster = EventBroadcaster()

# Get reliability metrics
metrics = broadcaster.get_reliability_dashboard()

print(f"Delivery Success Rate: {metrics['delivery_success_rate']:.2%}")
print(f"p95 Latency: {metrics['dispatch_latency_p95_ms']:.1f}ms")
print(f"Events Dispatched: {metrics['events_dispatched']}")
print(f"SLO Status: {metrics['slo_status']}")
```

**Response Format**:
```json
{
  "events_dispatched": 15234,
  "total_send_attempts": 45702,
  "send_successes": 45681,
  "send_failures": 21,
  "delivery_success_rate": 0.99954,
  "dispatch_latency_p95_ms": 45.2,
  "slo_thresholds": {
    "delivery_success_rate_min": 0.999,
    "dispatch_latency_p95_ms_max": 200.0
  },
  "slo_status": {
    "delivery_success_rate": true,
    "dispatch_latency_p95_ms": true
  }
}
```

### CLI Access

```bash
# Get current metrics
victor metrics event-bridge

# Watch metrics in real-time
watch -n 5 'victor metrics event-bridge'

# Export metrics for external monitoring
victor metrics event-bridge --format prometheus
```

---

## Alerting Rules

### Alert Configuration

```yaml
alerts:
  - name: EventBridgeDeliveryRateCritical
    condition: delivery_success_rate < 0.995
    duration: 2m
    severity: critical
    message: "Event delivery success rate below 99.5%"

  - name: EventBridgeDeliveryRateWarning
    condition: delivery_success_rate < 0.999
    duration: 5m
    severity: warning
    message: "Event delivery success rate below 99.9% SLO"

  - name: EventBridgeLatencyCritical
    condition: dispatch_latency_p95_ms > 500
    duration: 3m
    severity: critical
    message: "Event dispatch latency p95 above 500ms"

  - name: EventBridgeLatencyWarning
    condition: dispatch_latency_p95_ms > 200
    duration: 5m
    severity: warning
    message: "Event dispatch latency p95 above 200ms SLO"
```

### Alert Actions

1. **Critical Alerts**:
   - Send to incident response channel
   - Trigger automatic rollback if recent deployment
   - Page on-call engineer

2. **Warning Alerts**:
   - Log to monitoring system
   - Create ticket for investigation
   - Notify team via Slack

---

## Runbook

### Delivery Success Rate Below SLO

**Symptoms**: Events not reaching subscribed clients

**Diagnosis**:
1. Check `send_failures` counter - is it increasing?
2. Check client connections - are clients disconnecting?
3. Check queue depth - is the event queue backing up?
4. Check network connectivity between bridge and clients

**Mitigation**:
1. Restart WebSocket server if queue is full
2. Increase queue size temporarily
3. Check for slow/disconnected clients and remove them
4. Scale horizontally if under load

### Dispatch Latency Above SLO

**Symptoms**: Events taking too long to reach clients

**Diagnosis**:
1. Check `dispatch_latency_p95_ms` - is it spiking?
2. Check event queue depth - is it backing up?
3. Check broadcast loop - is it blocked?
4. Check for slow clients causing delays

**Mitigation**:
1. Identify and remove slow clients
2. Add timeout to client send operations
3. Increase broadcaster concurrency
4. Optimize event serialization

### Queue Depth Increasing

**Symptoms**: Event queue filling up

**Diagnosis**:
1. Check `queue.size()` - is it near capacity?
2. Check dispatch rate vs incoming rate
3. Check if broadcast loop is running
4. Check for blocked clients

**Mitigation**:
1. Increase queue size (temporary)
2. Add broadcaster workers
3. Drop low-priority events if necessary
4. Implement event batching

---

## Historical Tracking

### Metrics Storage

For long-term SLO compliance tracking, integrate with:

1. **Prometheus**: Scrape metrics via `/metrics` endpoint
2. **Datadog**: Use DogStatsD for metric forwarding
3. **CloudWatch**: Publish metrics via AWS SDK

### SLO Reporting

Generate weekly/monthly SLO compliance reports:

```python
from datetime import datetime, timedelta

# Get SLO compliance for last 7 days
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=7)

compliance = get_slo_compliance(
    start_time=start_time,
    end_time=end_time,
    slo_window="5m"
)

print(f"Weekly Delivery Success Rate: {compliance['delivery_success_rate']:.4%}")
print(f"Weekly p95 Latency: {compliance['dispatch_latency_p95_ms']:.1f}ms")
print(f"SLO Compliance: {compliance['slo_met_percentage']:.1f}%")
```

---

## Testing

### SLO Validation Tests

Located in: `tests/integration/integrations/api/test_event_bridge_reliability.py`

**Test Coverage**:
- ✅ `test_delivery_success_rate_slo` - Validates 99.9% delivery rate
- ✅ `test_dispatch_latency_p95_slo` - Validates <200ms p95 latency
- ✅ `test_zero_skipped_subscriptions` - Validates 100% coverage
- ✅ `test_sustained_load_no_loss` - Validates no event loss under load
- ✅ `test_reliability_metrics_under_load` - Validates SLOs under load

### Load Testing

```bash
# Run load test
pytest tests/integration/integrations/api/test_event_bridge_reliability.py \
    -k "sustained_load" -v

# Run all reliability tests
pytest tests/integration/integrations/api/test_event_bridge_reliability.py -v
```

---

## Milestone Progress

### M1 (Complete ✅)
- Async subscribe path merged
- Sync subscribe deprecated
- Basic reliability tracking added

### M2 (Complete ✅)
- Integration tests for loss/ordering (12 tests)
- High-volume burst scenarios
- SLO validation tests
- Event ordering tests

### M3 (Current 🔄)
- SLO dashboard documentation
- Alerting thresholds defined
- Runbook for common issues
- Historical tracking guidance

### M4 (Planned ⏳)
- External monitoring integration (Prometheus/Datadog)
- Automated SLO reporting
- Dashboard UI (Grafana)
- Enhanced alerting with escalation

---

## Appendix: Event Types

### Tool Events
- `tool.start` - Tool execution started
- `tool.progress` - Tool execution progress update
- `tool.complete` - Tool execution completed
- `tool.error` - Tool execution failed

### File Events
- `file.created` - File created
- `file.modified` - File modified
- `file.deleted` - File deleted

### Provider Events
- `provider.switch` - LLM provider switched
- `provider.error` - Provider error occurred
- `provider.recovery` - Provider recovered from error

### Session Events
- `session.start` - Session started
- `session.end` - Session ended
- `session.error` - Session error

### Metrics Events
- `metrics.update` - Metrics updated
- `budget.warning` - Tool budget warning
- `budget.exhausted` - Tool budget exhausted

### General Events
- `notification` - General notification
- `error` - Error notification

---

## Related Documentation

- **Event Bridge Implementation**: `victor/integrations/api/event_bridge.py`
- **Reliability Tests**: `tests/integration/integrations/api/test_event_bridge_reliability.py`
- **M2 Integration Tests**: `docs/planning/active-work-mapping.md`
- **Roadmap E4**: `roadmap.md` (E4: Event Bridge Reliability)
