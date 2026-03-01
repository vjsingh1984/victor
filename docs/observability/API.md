# Observability API Reference

## ObservabilityManager

The main class for managing metrics collection and aggregation.

### Class Methods

#### `get_instance(config=None)`

Get the singleton ObservabilityManager instance.

**Parameters**:
- `config` (ObservabilityConfig, optional): Configuration for the manager

**Returns**: `ObservabilityManager`

**Raises**: None

**Example**:
```python
from victor.framework.observability import ObservabilityManager, ObservabilityConfig

config = ObservabilityConfig(
    max_history_size=1000,
    enable_system_metrics=True,
)
manager = ObservabilityManager.get_instance(config=config)
```

#### `reset()`

Reset the singleton instance.

**Warning**: This is a dangerous method that should only be used in testing.

**Example**:
```python
ObservabilityManager.reset()
```

### Instance Methods

#### `register_source(source)`

Register a metrics source with the manager.

**Parameters**:
- `source` (MetricSource): Component implementing MetricSource protocol

**Returns**: None

**Raises**: None (logs warning if source doesn't implement MetricSource)

**Example**:
```python
from victor.framework.observability import ObservabilityManager

manager = ObservabilityManager.get_instance()
manager.register_source(my_capability)
```

#### `unregister_source(source)`

Unregister a metrics source.

**Parameters**:
- `source` (MetricSource): Component to unregister

**Returns**: None

**Example**:
```python
manager.unregister_source(my_capability)
```

#### `list_sources()`

List all registered source IDs.

**Returns**: `List[str]` - List of source IDs

**Example**:
```python
sources = manager.list_sources()
print(f"Registered sources: {sources}")
```

#### `collect_metrics(include_system=True)`

Collect metrics from all registered sources.

**Parameters**:
- `include_system` (bool): Whether to include system metrics

**Returns**: `MetricsCollection` - Collection of metrics snapshots

**Example**:
```python
collection = manager.collect_metrics()
for snapshot in collection.snapshots:
    print(f"{snapshot.source_id}: {len(snapshot.metrics)} metrics")
```

#### `get_dashboard_data()`

Get aggregated dashboard data.

**Returns**: `DashboardData` - Aggregated dashboard data

**Example**:
```python
data = manager.get_dashboard_data()
print(f"Cache hit rate: {data.cache_metrics['hit_rate']:.2%}")
print(f"Tool success rate: {data.tool_metrics['success_rate']:.2%}")
```

#### `get_historical_data(source_id=None, source_type=None, hours=1.0)`

Get historical metrics data.

**Parameters**:
- `source_id` (str, optional): Filter by specific source ID
- `source_type` (str, optional): Filter by source type
- `hours` (float): Hours of history to return (default: 1.0)

**Returns**: `List[MetricsCollection]` - Historical metrics collections

**Example**:
```python
# Get last hour of all data
history = manager.get_historical_data(hours=1.0)

# Get cache metrics for last 6 hours
history = manager.get_historical_data(source_type="cache", hours=6.0)

# Get specific source
history = manager.get_historical_data(source_id="my_source", hours=24.0)
```

#### `get_stats()`

Get observability manager statistics.

**Returns**: `Dict[str, Any]` - Manager statistics

**Example**:
```python
stats = manager.get_stats()
print(f"Collection count: {stats['collection_count']}")
print(f"Registered sources: {stats['registered_sources']}")
print(f"History size: {stats['history_size']}")
```

#### `clear_history()`

Clear all historical metrics data.

**Returns**: None

**Example**:
```python
manager.clear_history()
```

#### `close()`

Close the observability manager and release resources.

**Returns**: None

**Example**:
```python
manager.close()
```

---

## ObservabilityConfig

Configuration for ObservabilityManager.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_history_size` | int | 1000 | Maximum number of historical snapshots |
| `collection_timeout_seconds` | float | 5.0 | Timeout for collecting metrics from a source |
| `enable_system_metrics` | bool | True | Whether to collect system metrics |
| `system_metrics_interval_seconds` | float | 60.0 | Interval between system metrics collection |
| `history_retention_hours` | float | 24.0 | How long to keep historical data (hours) |

### Example

```python
from victor.framework.observability import ObservabilityConfig

config = ObservabilityConfig(
    max_history_size=2000,
    collection_timeout_seconds=10.0,
    enable_system_metrics=True,
    system_metrics_interval_seconds=30.0,
    history_retention_hours=48.0,
)
```

---

## DashboardData

Aggregated data for dashboard display.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `timestamp` | float | When the dashboard data was generated |
| `total_sources` | int | Total number of registered sources |
| `sources_by_type` | Dict[str, int] | Count of sources by type |
| `cache_metrics` | Dict[str, Any] | Aggregated cache metrics |
| `tool_metrics` | Dict[str, Any] | Aggregated tool metrics |
| `coordinator_metrics` | Dict[str, Any] | Aggregated coordinator metrics |
| `capability_metrics` | Dict[str, Any] | Aggregated capability metrics |
| `vertical_metrics` | Dict[str, Any] | Aggregated vertical metrics |
| `system_metrics` | Dict[str, Any] | System resource metrics |
| `alerts` | List[Dict[str, Any]] | List of alerts |

### Methods

#### `to_dict()`

Convert to dictionary.

**Returns**: `Dict[str, Any]`

**Example**:
```python
data = manager.get_dashboard_data()
data_dict = data.to_dict()
```

---

## MetricsCollection

A collection of metrics snapshots from multiple sources.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `timestamp` | float | When the collection was created |
| `snapshots` | List[MetricsSnapshot] | List of metrics snapshots |

### Methods

#### `add_snapshot(snapshot)`

Add a metrics snapshot to the collection.

**Parameters**:
- `snapshot` (MetricsSnapshot): Snapshot to add

**Example**:
```python
collection = MetricsCollection()
collection.add_snapshot(snapshot)
```

#### `get_by_source_id(source_id)`

Get snapshot by source ID.

**Parameters**:
- `source_id` (str): Source ID to find

**Returns**: `Optional[MetricsSnapshot]`

**Example**:
```python
snapshot = collection.get_by_source_id("my_source")
```

#### `get_by_source_type(source_type)`

Get all snapshots of a specific type.

**Parameters**:
- `source_type` (str): Source type to filter by

**Returns**: `List[MetricsSnapshot]`

**Example**:
```python
cache_snapshots = collection.get_by_source_type("cache")
```

---

## MetricsSnapshot

A snapshot of metrics from a single source.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `source_id` | str | Unique identifier for the source |
| `source_type` | str | Type of the source (cache, tool, coordinator, etc.) |
| `metrics` | Tuple[Metric, ...] | Tuple of metrics |

---

## Metric Types

### Metric

Base class for all metrics.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Metric name |
| `description` | str | Human-readable description |
| `value` | Any | Metric value |

### CounterMetric

A metric that only increases (counts events).

**Additional Attributes**:
- None (inherits from Metric)

**Example**:
```python
from victor.framework.observability.metrics import CounterMetric

metric = CounterMetric(
    name="requests_total",
    description="Total number of requests",
    value=1234,
)
```

### GaugeMetric

A metric that can go up or down (current value).

**Additional Attributes**:
- `labels` (Tuple[MetricLabel, ...], optional): Labels for the gauge

**Example**:
```python
from victor.framework.observability.metrics import GaugeMetric, MetricLabel

metric = GaugeMetric(
    name="active_connections",
    description="Number of active connections",
    value=42,
    labels=(MetricLabel(key="server", value="web-1"),),
)
```

### HistogramMetric

A metric that tracks distributions of values.

**Additional Attributes**:
- `count` (int): Total count of observations
- `sum` (float): Sum of all observations
- `bucket_counts` (Tuple[int, ...]): Count of observations per bucket
- `bucket_bounds` (Tuple[float, ...]): Upper bounds for each bucket

**Example**:
```python
from victor.framework.observability.metrics import HistogramMetric

metric = HistogramMetric(
    name="request_duration_seconds",
    description="Request duration in seconds",
    count=1000,
    sum=234.5,
    bucket_counts=(100, 500, 300, 100),
    bucket_bounds=(0.1, 0.5, 1.0, 5.0),
)
```

### SummaryMetric

A metric that tracks statistical summaries.

**Additional Attributes**:
- `count` (int): Total count
- `sum` (float): Sum of values
- `min` (float): Minimum value
- `max` (float): Maximum value
- `avg` (float): Average value

**Example**:
```python
from victor.framework.observability.metrics import SummaryMetric

metric = SummaryMetric(
    name="response_size_bytes",
    description="Response size in bytes",
    count=500,
    sum=1234500,
    min=100,
    max=10000,
    avg=2469.0,
)
```

### MetricLabel

A label for categorizing metrics.

**Attributes**:
- `key` (str): Label key
- `value` (str): Label value

**Example**:
```python
from victor.framework.observability.metrics import MetricLabel

label = MetricLabel(key="environment", value="production")
```

---

## MetricSource Protocol

Protocol for components that provide metrics.

### Required Properties

#### `source_id: str`

Unique identifier for the metrics source.

#### `source_type: str`

Type of the metrics source (cache, tool, coordinator, capability, vertical, custom).

### Required Methods

#### `get_metrics() -> MetricsSnapshot`

Get current metrics from this source.

**Returns**: `MetricsSnapshot`

**Example Implementation**:
```python
from victor.framework.observability.metrics import MetricSource, MetricsSnapshot, Metric

class MySource(MetricSource):
    @property
    def source_id(self) -> str:
        return "my_source"

    @property
    def source_type(self) -> str:
        return "custom"

    def get_metrics(self) -> MetricsSnapshot:
        metrics = [
            Metric(name="custom_metric", value=42),
        ]
        return MetricsSnapshot(
            source_id=self.source_id,
            source_type=self.source_type,
            metrics=tuple(metrics),
        )
```

---

## Exceptions

### ObservabilityError

Base exception for observability-related errors.

### MetricsCollectionError

Raised when metrics collection fails.

### InvalidSourceError

Raised when a source doesn't implement MetricSource protocol.

---

## Type Hints

```python
from typing import Any, Dict, List, Optional, Tuple

class ObservabilityManager:
    @classmethod
    def get_instance(cls, config: Optional[ObservabilityConfig] = None) -> ObservabilityManager: ...

    def register_source(self, source: MetricSource) -> None: ...

    def unregister_source(self, source: MetricSource) -> None: ...

    def list_sources(self) -> List[str]: ...

    def collect_metrics(self, include_system: bool = True) -> MetricsCollection: ...

    def get_dashboard_data(self) -> DashboardData: ...

    def get_historical_data(
        self,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        hours: float = 1.0,
    ) -> List[MetricsCollection]: ...

    def get_stats(self) -> Dict[str, Any]: ...

    def clear_history(self) -> None: ...

    def close(self) -> None: ...
```

---

**Version**: 1.0
**Last Updated**: 2025-02-28
