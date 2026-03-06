# Event Bridge Reliability Dashboard

Operational guide for event delivery health in the real-time Event Bridge.

## What Is Measured

The bridge now tracks and exposes:

- `events_dispatched`
- `total_send_attempts`
- `send_successes`
- `send_failures`
- `delivery_success_rate`
- `dispatch_latency_p95_ms`

These metrics are returned by `EventBridge.get_reliability_dashboard_data()`.

## SLO Thresholds (Enabled)

The bridge evaluates two live SLO checks on every dispatch:

- Delivery success rate: `>= 99.9%` (`0.999`)
- Dispatch latency p95: `<= 200 ms`

Dashboard output includes both configured thresholds and current pass/fail status:

- `slo_thresholds.delivery_success_rate_min`
- `slo_thresholds.dispatch_latency_p95_ms_max`
- `slo_status.delivery_success_rate`
- `slo_status.dispatch_latency_p95_ms`

## Alerting Behavior

When either SLO is breached, the bridge logs:

- `EventBridge SLO breach: ...`

Logs are throttled to once every `30s` to prevent noise during incidents.

## Example Snapshot

```json
{
  "events_dispatched": 1802,
  "total_send_attempts": 9010,
  "send_successes": 9005,
  "send_failures": 5,
  "delivery_success_rate": 0.9994,
  "dispatch_latency_p95_ms": 112.3
}
```

## Operational Response

- If success rate drops: inspect disconnected clients and transport errors.
- If p95 latency rises: check queue pressure and downstream websocket send timeouts.
- Escalate when either SLO stays red for more than one review interval.
