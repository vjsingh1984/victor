# Victor Observability Guide

## Overview

Victor provides a unified observability system for monitoring framework metrics, cache performance, tool execution, coordinator operations, and system resources.

## Table of Contents

- [Quick Start](#quick-start)
- [Dashboard CLI](#dashboard-cli)
- [Metrics Sources](#metrics-sources)
- [Alerting](#alerting)
- [Custom Metrics](#custom-metrics)
- [API Reference](#api-reference)

---

## Quick Start

### Installation

Observability is included with Victor. No additional installation required.

```bash
pip install victor-ai
```

### Basic Usage

```bash
# Show observability dashboard
victor observability dashboard

# Show dashboard as JSON
victor observability dashboard --json

# Continuously update dashboard
victor observability dashboard --watch --interval 5

# Show metrics for specific source
victor observability metrics cache:tool_embeddings

# Show historical data
victor observability history 6

# Show observability statistics
victor observability stats
```

---

## Dashboard CLI

### Commands

#### `victor observability dashboard`

Display the observability dashboard with aggregated metrics.

**Options**:
- `--json`, `-j`: Output as JSON instead of formatted text
- `--watch`, `-w`: Continuously update dashboard
- `--interval`, `-i`: Update interval in seconds (default: 5.0)

**Examples**:
```bash
# Show dashboard once
victor observability dashboard

# Show dashboard as JSON
victor observability dashboard --json

# Continuously update every 2 seconds
victor observability dashboard --watch --interval 2
```

**Output Format**:
```
╔═══════════════════════════════════════════════════════════════════╗
║                     Victor Observability Dashboard               ║
╠═══════════════════════════════════════════════════════════════════╣
║ Sources: 15 (5 capabilities, 3 caches, 4 coordinators, 3 verticals) ║
╠═══════════════════════════════════════════════════════════════════╣
║ Cache Metrics                                                        ║
║   Hit Rate: 78.5% (1,234 hits, 338 misses)                        ║
║   Evictions: 12                                                      ║
║   Total Size: 156.2 MB                                               ║
╠═══════════════════════════════════════════════════════════════════╣
║ Tool Metrics                                                         ║
║   Total Calls: 456                                                   ║
║   Success Rate: 94.3%                                                ║
║   Average Latency: 0.234s                                            ║
╠═══════════════════════════════════════════════════════════════════╣
║ System Metrics                                                       ║
║   Memory Usage: 234.5 MB                                             ║
║   CPU Usage: 3.2%                                                    ║
╠═══════════════════════════════════════════════════════════════════╣
║ Alerts                                                               ║
║   ⚠️  Warning: Low cache hit rate: 45.2%                            ║
╚═══════════════════════════════════════════════════════════════════╝
```

#### `victor observability metrics [source]`

Show metrics for a specific metrics source.

**Arguments**:
- `source`: Source ID to filter by (optional)

**Examples**:
```bash
# Show all metrics
victor observability metrics

# Show cache metrics
victor observability metrics cache:tool_embeddings

# Show coordinator metrics
victor observability metrics coordinator:tool
```

#### `victor observability history [hours]`

Show historical metrics data.

**Arguments**:
- `hours`: Hours of history to show (default: 1.0)

**Examples**:
```bash
# Show last hour of data
victor observability history

# Show last 6 hours
victor observability history 6

# Show last 24 hours
victor observability history 24
```

#### `victor observability stats`

Show observability manager statistics.

**Examples**:
```bash
victor observability stats
```

**Output**:
```
Observability Manager Statistics
================================
Collection Count: 1,234
Collection Errors: 5
Last Collection Duration: 0.123s
Registered Sources: 15
History Size: 856
History Retention: 24 hours
```

---

## Metrics Sources

### Capabilities

Metrics from capability providers.

**Source Type**: `capability`

**Metrics**:
- `access_count`: Total number of capability accesses
- `last_accessed`: Timestamp of last access
- `error_count`: Total number of errors
- `capability_count`: Number of registered capabilities

**Example**:
```python
from victor.framework.capabilities import BaseCapabilityProvider

class MyCapabilityProvider(BaseCapabilityProvider):
    def get_capabilities(self):
        return {"my_capability": MyCapability()}

# Record access
provider.record_access("my_capability")

# Get metrics
metrics = provider.get_metrics()
print(f"Accessed {metrics['access_count']} times")
```

### Caches

Metrics from cache instances.

**Source Type**: `cache`

**Metrics**:
- `total_hits`: Total cache hits
- `total_misses`: Total cache misses
- `hit_rate`: Cache hit rate (0.0 to 1.0)
- `evictions`: Number of cache evictions
- `memory_usage`: Memory usage in bytes
- `disk_usage`: Disk usage in bytes

**Example**:
```python
from victor.storage.cache import TieredCache

cache = TieredCache()

# Get observability data
data = cache.get_observability_data()
print(f"Hit rate: {data['hit_rate']:.2%}")
```

### Tools

Metrics from tool execution.

**Source Type**: `tool`

**Metrics**:
- `total_calls`: Total tool calls
- `total_errors`: Total tool errors
- `success_rate`: Success rate (0.0 to 1.0)
- `average_latency`: Average execution time in seconds

### Coordinators

Metrics from coordinator operations.

**Source Type**: `coordinator`

**Metrics**:
- `total_operations`: Total coordinator operations
- `total_errors`: Total errors
- `average_latency`: Average operation time

**Example**:
```python
from victor.agent.coordinators import ToolCoordinator

coordinator = ToolCoordinator(...)

# Get observability data
data = coordinator.get_observability_data()
print(f"Operations: {data['operation_count']}")
```

### Verticals

Metrics from vertical operations.

**Source Type**: `vertical`

**Metrics**:
- `total_requests`: Total vertical requests
- `total_errors`: Total errors
- `average_latency`: Average request time

### System Resources

System resource metrics.

**Source Type**: `system`

**Metrics**:
- `memory_usage_bytes`: Process memory usage
- `cpu_usage_percent`: CPU usage percentage

---

## Alerting

ObservabilityManager includes automatic alert generation based on metrics thresholds.

### Alert Types

#### Low Cache Hit Rate

**Severity**: Warning
**Trigger**: Cache hit rate < 50%

**Message**: "Low cache hit rate: X.X%"

```python
{
    "severity": "warning",
    "type": "low_cache_hit_rate",
    "message": "Low cache hit rate: 45.2%",
    "value": 0.452
}
```

#### High Tool Error Rate

**Severity**: Error
**Trigger**: Tool success rate < 90%

**Message**: "High tool error rate: X.X%"

```python
{
    "severity": "error",
    "type": "high_tool_error_rate",
    "message": "High tool error rate: 12.5%",
    "value": 0.125
}
```

#### High Memory Usage

**Severity**: Warning
**Trigger**: Memory usage > 80% of system memory

**Message**: "High memory usage: X.X%"

```python
{
    "severity": "warning",
    "type": "high_memory_usage",
    "message": "High memory usage: 85.3%",
    "value": 85.3
}
```

### Viewing Alerts

Alerts are displayed in the dashboard output:

```bash
$ victor observability dashboard
...
╠═══════════════════════════════════════════════════════════════════╣
║ Alerts                                                               ║
║   ⚠️  Warning: Low cache hit rate: 45.2%                            ║
║   ❌ Error: High tool error rate: 12.5%                             ║
╚═══════════════════════════════════════════════════════════════════╝
```

Or as JSON:

```bash
$ victor observability dashboard --json | jq '.alerts'
```

---

## Custom Metrics

### Creating a Metrics Source

To add custom metrics, create a class that implements the `MetricSource` protocol:

```python
from victor.framework.observability.metrics import MetricSource, MetricsSnapshot, Metric
from victor.framework.observability import ObservabilityManager

class MyCustomSource(MetricSource):
    @property
    def source_id(self) -> str:
        return "my_custom_source"

    @property
    def source_type(self) -> str:
        return "custom"

    def get_metrics(self) -> MetricsSnapshot:
        # Create your custom metrics
        metrics = [
            Metric(name="custom_metric_1", value=42),
            Metric(name="custom_metric_2", value=3.14),
        ]

        return MetricsSnapshot(
            source_id=self.source_id,
            source_type=self.source_type,
            metrics=tuple(metrics),
        )

# Register with ObservabilityManager
manager = ObservabilityManager.get_instance()
manager.register_source(MyCustomSource())
```

### Using Counter and Gauge Metrics

For better integration, use the standard metric types:

```python
from victor.framework.observability.metrics import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    SummaryMetric,
    MetricLabel,
)

class MySource(MetricSource):
    def get_metrics(self) -> MetricsSnapshot:
        metrics = [
            # Counter: Increments over time
            CounterMetric(
                name="requests_total",
                description="Total requests",
                value=1234,
            ),

            # Gauge: Current value
            GaugeMetric(
                name="active_connections",
                description="Active connections",
                value=42,
                labels=(MetricLabel(key="server", value="web-1"),),
            ),

            # Histogram: Distribution of values
            HistogramMetric(
                name="request_duration_seconds",
                description="Request duration",
                count=1000,
                sum=234.5,
                bucket_counts=[100, 500, 300, 100],
                bucket_bounds=[0.1, 0.5, 1.0, 5.0],
            ),

            # Summary: Statistics
            SummaryMetric(
                name="response_size_bytes",
                description="Response size",
                count=500,
                sum=1234500,
                min=100,
                max=10000,
                avg=2469.0,
            ),
        ]

        return MetricsSnapshot(
            source_id="my_source",
            source_type="custom",
            metrics=tuple(metrics),
        )
```

---

## API Reference

### ObservabilityManager

#### `get_instance(config=None)`

Get the singleton ObservabilityManager instance.

**Parameters**:
- `config` (ObservabilityConfig, optional): Manager configuration

**Returns**: ObservabilityManager

**Example**:
```python
from victor.framework.observability import ObservabilityManager, ObservabilityConfig

config = ObservabilityConfig(
    max_history_size=1000,
    enable_system_metrics=True,
)
manager = ObservabilityManager.get_instance(config=config)
```

#### `register_source(source)`

Register a metrics source.

**Parameters**:
- `source` (MetricSource): Component implementing MetricSource protocol

**Example**:
```python
manager.register_source(my_capability)
```

#### `collect_metrics(include_system=True)`

Collect metrics from all registered sources.

**Parameters**:
- `include_system` (bool): Whether to include system metrics

**Returns**: MetricsCollection

**Example**:
```python
collection = manager.collect_metrics()
for snapshot in collection.snapshots:
    print(f"{snapshot.source_id}: {len(snapshot.metrics)} metrics")
```

#### `get_dashboard_data()`

Get aggregated dashboard data.

**Returns**: DashboardData

**Example**:
```python
data = manager.get_dashboard_data()
print(f"Cache hit rate: {data.cache_metrics['hit_rate']:.2%}")
```

#### `get_historical_data(source_id=None, source_type=None, hours=1.0)`

Get historical metrics data.

**Parameters**:
- `source_id` (str, optional): Filter by specific source ID
- `source_type` (str, optional): Filter by source type
- `hours` (float): Hours of history to return

**Returns**: List[MetricsCollection]

**Example**:
```python
# Get last hour of data
history = manager.get_historical_data(hours=1.0)

# Get cache metrics for last 6 hours
history = manager.get_historical_data(source_type="cache", hours=6.0)
```

### ObservabilityConfig

Configuration for ObservabilityManager.

**Parameters**:
- `max_history_size` (int): Maximum number of historical snapshots (default: 1000)
- `collection_timeout_seconds` (float): Timeout for collecting metrics (default: 5.0)
- `enable_system_metrics` (bool): Enable system metrics collection (default: True)
- `system_metrics_interval_seconds` (float): System metrics collection interval (default: 60.0)
- `history_retention_hours` (float): How long to keep historical data (default: 24.0)

### DashboardData

Aggregated dashboard data.

**Attributes**:
- `timestamp` (float): When data was generated
- `total_sources` (int): Total number of sources
- `sources_by_type` (Dict[str, int]): Count of sources by type
- `cache_metrics` (Dict[str, Any]): Aggregated cache metrics
- `tool_metrics` (Dict[str, Any]): Aggregated tool metrics
- `coordinator_metrics` (Dict[str, Any]): Aggregated coordinator metrics
- `capability_metrics` (Dict[str, Any]): Aggregated capability metrics
- `vertical_metrics` (Dict[str, Any]): Aggregated vertical metrics
- `system_metrics` (Dict[str, Any]): System resource metrics
- `alerts` (List[Dict[str, Any]]): List of active alerts

---

## Best Practices

### 1. Use Appropriate Metric Types

- **Counter**: For counting events (requests, errors)
- **Gauge**: For current values (connections, queue size)
- **Histogram**: For distributions (request latency, response size)
- **Summary**: For statistics (min, max, avg)

### 2. Add Labels for Context

```python
GaugeMetric(
    name="active_connections",
    value=42,
    labels=(
        MetricLabel(key="server", value="web-1"),
        MetricLabel(key="region", value="us-west"),
    ),
)
```

### 3. Use Descriptive Names

```python
# Good
CounterMetric(name="http_requests_total", value=1234)

# Avoid
CounterMetric(name="requests", value=1234)
```

### 4. Update Metrics Efficiently

```python
# Good: Batch updates
metrics = [metric1, metric2, metric3]
return MetricsSnapshot(metrics=tuple(metrics))

# Avoid: Individual updates
manager.register_source(Source1())
manager.register_source(Source2())
manager.register_source(Source3())
```

### 5. Handle Errors Gracefully

```python
def get_metrics(self) -> MetricsSnapshot:
    try:
        # Collect metrics
        metrics = [...]
    except Exception as e:
        logger.warning(f"Failed to collect metrics: {e}")
        metrics = [Metric(name="collection_error", value=1)]

    return MetricsSnapshot(metrics=tuple(metrics))
```

---

## Performance Considerations

### Metrics Collection Overhead

The observability system is designed for minimal overhead:

- **Capabilities**: <1% overhead (simple counter increments)
- **Caches**: <1% overhead (stats already tracked)
- **Coordinators**: <2% overhead (aggregation of existing stats)
- **System**: <1% overhead (polling every 60 seconds)

**Total Overhead**: <5%

### Optimization Tips

1. **Cache Metrics Locally**: Cache metric calculations between collections
2. **Use Sampling**: Sample metrics instead of tracking every event
3. **Batch Updates**: Update multiple metrics at once
4. **Background Collection**: Collect metrics in background threads
5. **Conditional Collection**: Disable expensive metrics in production

---

## Troubleshooting

### Dashboard Shows No Sources

**Problem**: Dashboard reports 0 sources

**Solution**: Ensure components are registered with ObservabilityManager

```python
from victor.framework.observability import ObservabilityManager

manager = ObservabilityManager.get_instance()
print(f"Registered sources: {manager.list_sources()}")
```

### High Memory Usage

**Problem**: ObservabilityManager using too much memory

**Solution**: Reduce history retention

```python
from victor.framework.observability import ObservabilityConfig

config = ObservabilityConfig(
    max_history_size=500,           # Reduce from 1000
    history_retention_hours=12.0,    # Reduce from 24.0
)
```

### Missing Metrics

**Problem**: Expected metrics not appearing in dashboard

**Solution**: Verify component implements MetricSource protocol

```python
from victor.framework.observability.metrics import MetricSource

assert isinstance(my_component, MetricSource)
assert hasattr(my_component, 'get_metrics')
```

### Collection Errors

**Problem**: Metrics collection failing

**Solution**: Check logs for error messages

```bash
# Enable debug logging
export VICTOR_LOG_LEVEL=DEBUG

# Run dashboard
victor observability dashboard
```

---

## Advanced Usage

### Custom Alerting

Implement custom alerting by processing dashboard data:

```python
from victor.framework.observability import ObservabilityManager

manager = ObservabilityManager.get_instance()
data = manager.get_dashboard_data()

# Custom alert logic
if data['cache_metrics']['hit_rate'] < 0.3:
    send_alert("Critical: Cache hit rate below 30%")
```

### Export Metrics

Export metrics for external monitoring systems:

```python
import json
from victor.framework.observability import ObservabilityManager

manager = ObservabilityManager.get_instance()

# Export to JSON
data = manager.get_dashboard_data()
with open('metrics.json', 'w') as f:
    json.dump(data.to_dict(), f, indent=2)

# Export to Prometheus format
collection = manager.collect_metrics()
for snapshot in collection.snapshots:
    for metric in snapshot.metrics:
        print(f"{metric.name} {metric.value}")
```

### Metrics Aggregation

Aggregate metrics across multiple sources:

```python
from victor.framework.observability import ObservabilityManager

manager = ObservabilityManager.get_instance()

# Get all cache metrics
history = manager.get_historical_data(source_type="cache", hours=24)

# Calculate aggregate hit rate
total_hits = 0
total_misses = 0
for collection in history:
    for snapshot in collection.snapshots:
        for metric in snapshot.metrics:
            if metric.name == "cache_hits":
                total_hits += metric.value
            elif metric.name == "cache_misses":
                total_misses += metric.value

aggregate_hit_rate = total_hits / (total_hits + total_misses)
print(f"24h aggregate hit rate: {aggregate_hit_rate:.2%}")
```

---

## Further Reading

- [Architecture Improvement Plan](/tmp/VICTOR_ARCHITECTURE_IMPROVEMENT_FINAL_SUMMARY.md)
- [Phase 4 Completion Summary](/tmp/phase4_completion_summary.md)
- [Contrib Packages Guide](/docs/contrib/README.md)

---

**Version**: 1.0
**Last Updated**: 2025-02-28
**Maintainer**: Victor Framework Team
