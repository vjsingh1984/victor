# Observability Examples

This directory contains examples demonstrating Victor's observability features for team execution.

## Examples

### team_monitoring.py

Demonstrates comprehensive team monitoring capabilities:

- **Metrics Collection**: Collect team execution metrics
- **Event Streaming**: Subscribe to team lifecycle events
- **Querying**: Query metrics and analyze performance
- **Tracing**: Distributed tracing for nested execution
- **Export**: Export metrics to JSON for analysis

#### Usage

```bash
python examples/observability/team_monitoring.py
```

#### Features Demonstrated

1. **Event Subscription**
   ```python
   bus.subscribe("team.execution.started", on_team_started)
   bus.subscribe("team.execution.completed", on_team_completed)
   bus.subscribe("team.member.completed", on_member_completed)
   bus.subscribe("team.recursion.depth_exceeded", on_depth_exceeded)
   ```

2. **Metrics Querying**
   ```python
   collector = get_team_metrics_collector()
   summary = collector.get_summary()
   formation_stats = collector.get_formation_stats("parallel")
   ```

3. **Distributed Tracing**
   ```python
   with trace_team_execution("my_team", "parallel", 3) as span:
       span.set_attribute("task", "Review code")
       # ... execute team
   ```

4. **Metrics Export**
   ```python
   await export_metrics_to_json()
   ```

## Key Concepts

### Team Metrics

Team metrics provide insights into multi-agent team execution:

- **Formation Type**: How agents are coordinated (parallel, sequential, etc.)
- **Member Count**: Number of agents in the team
- **Execution Time**: How long teams take to complete tasks
- **Tool Usage**: Tools used by each team member
- **Success Rate**: Percentage of successful team executions
- **Recursion Depth**: Nesting level for nested workflows/teams

### Event Streaming

Real-time events for team lifecycle monitoring:

- `team.execution.started`: Team started execution
- `team.execution.completed`: Team completed (success/failure)
- `team.member.completed`: Member completed execution
- `team.recursion.depth_exceeded`: Recursion limit exceeded

### Distributed Tracing

Trace team execution across nested workflows:

- **Trace ID**: Unique identifier for correlated operations
- **Spans**: Individual operations (team, member, workflow)
- **Parent-Child**: Nested execution relationships
- **OpenTelemetry**: Integration with distributed tracing systems

## Production Usage

### Enable Metrics Collection

```python
from victor.workflows.team_node_runner import TeamNodeRunner

runner = TeamNodeRunner(
    orchestrator=orchestrator,
    enable_metrics=True,
)
```

### Subscribe to Events

```python
from victor.core.events import ObservabilityBus

bus = ObservabilityBus.get_instance()
bus.subscribe("team.*", your_event_handler)
```

### Query Metrics

```python
from victor.workflows.team_metrics import get_team_metrics_collector

collector = get_team_metrics_collector()
summary = collector.get_summary()
team_metrics = collector.get_team_metrics("my_team")
```

### Prometheus Integration

```python
from victor.observability.metrics import MetricsRegistry
from prometheus_client import start_http_server

# Metrics are automatically registered in MetricsRegistry
start_http_server(8000)
# Visit http://localhost:8000/metrics
```

## Further Reading

- [Team Metrics Documentation](../../../docs/observability/team_metrics.md)
- [Workflow System](../../../docs/workflows/README.md)
- [Event Bus Taxonomy](../../../victor/core/events/taxonomy.md)
- [Teams System](../../../docs/teams/README.md)
