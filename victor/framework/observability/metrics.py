# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Observability metrics data models.

This module provides data models for collecting and aggregating metrics
from various components in the Victor framework.

Design Pattern: Value Object Pattern + Protocol Pattern
- Immutable metric data models
- Protocol-based metric source interface
- Type-safe metric collection

Phase 4: Enhance Observability with Unified Dashboard
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)
from collections import defaultdict


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"  # Cumulative value that only increases
    GAUGE = "gauge"  # Point-in-time value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Aggregated statistics (count, sum, avg, min, max)


@dataclass(frozen=True)
class MetricLabel:
    """A key-value label for a metric.

    Labels allow filtering and grouping of metrics.
    """

    key: str
    value: str

    def __post_init__(self):
        """Validate label format."""
        if not self.key:
            raise ValueError("Label key cannot be empty")
        if ":" in self.key:
            raise ValueError("Label key cannot contain ':'")


@dataclass(frozen=True)
class Metric:
    """Base metric data model.

    Attributes:
        name: Metric name (e.g., "cache_hits", "tool_calls")
        description: Human-readable description
        metric_type: Type of metric (counter, gauge, histogram, summary)
        labels: Optional labels for filtering/grouping
        timestamp: When the metric was recorded
    """

    name: str
    description: str
    metric_type: MetricType
    labels: tuple[MetricLabel, ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate metric format."""
        if not self.name:
            raise ValueError("Metric name cannot be empty")

    def with_labels(self, **kwargs: str) -> "Metric":
        """Return a new metric with additional labels.

        Args:
            **kwargs: Label key-value pairs

        Returns:
            New metric with merged labels
        """
        existing_labels = {label.key: label.value for label in self.labels}
        existing_labels.update(kwargs)

        new_labels = tuple(MetricLabel(key=k, value=v) for k, v in sorted(existing_labels.items()))

        return type(self)(
            name=self.name,
            description=self.description,
            metric_type=self.metric_type,
            labels=new_labels,
        )

    def get_label_value(self, key: str) -> Optional[str]:
        """Get label value by key.

        Args:
            key: Label key

        Returns:
            Label value or None if not found
        """
        for label in self.labels:
            if label.key == key:
                return label.value
        return None


@dataclass(frozen=True)
class CounterMetric(Metric):
    """A counter metric that only increases.

    Counters are used for cumulative values like:
    - Total requests
    - Total errors
    - Total cache hits

    Attributes:
        value: Current counter value
    """

    value: int = 0

    def __post_init__(self):
        """Initialize as counter type."""
        object.__setattr__(self, "metric_type", MetricType.COUNTER)

    def increment(self, amount: int = 1) -> "CounterMetric":
        """Return a new counter with incremented value.

        Args:
            amount: Amount to increment (default: 1)

        Returns:
            New counter metric
        """
        return type(self)(
            name=self.name,
            description=self.description,
            metric_type=MetricType.COUNTER,
            labels=self.labels,
            value=self.value + amount,
        )


@dataclass(frozen=True)
class GaugeMetric(Metric):
    """A gauge metric that can go up or down.

    Gauges are used for point-in-time values like:
    - Current memory usage
    - Active connections
    - Queue size

    Attributes:
        value: Current gauge value
        min_value: Minimum observed value
        max_value: Maximum observed value
    """

    value: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        """Initialize as gauge type."""
        object.__setattr__(self, "metric_type", MetricType.GAUGE)

    def set(self, value: float) -> "GaugeMetric":
        """Return a new gauge with updated value.

        Args:
            value: New gauge value

        Returns:
            New gauge metric with updated min/max
        """
        new_min = self.min_value
        new_max = self.max_value

        if new_min is None or value < new_min:
            new_min = value
        if new_max is None or value > new_max:
            new_max = value

        return type(self)(
            name=self.name,
            description=self.description,
            metric_type=MetricType.GAUGE,
            labels=self.labels,
            value=value,
            min_value=new_min,
            max_value=new_max,
        )


@dataclass(frozen=True)
class HistogramBucket:
    """A histogram bucket.

    Attributes:
        upper_bound: Upper bound of the bucket (exclusive)
        count: Number of observations in this bucket
    """

    upper_bound: float
    count: int = 0


@dataclass(frozen=True)
class HistogramMetric(Metric):
    """A histogram metric that tracks distribution of values.

    Histograms are used for distributions like:
    - Request latency
    - Response sizes
    - Processing times

    Attributes:
        count: Total number of observations
        sum: Sum of all observations
        buckets: Histogram buckets
    """

    count: int = 0
    sum: float = 0.0
    buckets: tuple[HistogramBucket, ...] = field(default_factory=tuple)

    def __post_init__(self):
        """Initialize as histogram type."""
        object.__setattr__(self, "metric_type", MetricType.HISTOGRAM)

    def observe(self, value: float) -> "HistogramMetric":
        """Return a new histogram with the observation added.

        Args:
            value: Observed value

        Returns:
            New histogram metric
        """
        new_count = self.count + 1
        new_sum = self.sum + value

        # Update buckets
        new_buckets = []
        for bucket in self.buckets:
            if value < bucket.upper_bound:
                new_buckets.append(
                    HistogramBucket(upper_bound=bucket.upper_bound, count=bucket.count + 1)
                )
            else:
                new_buckets.append(bucket)

        return type(self)(
            name=self.name,
            description=self.description,
            metric_type=MetricType.HISTOGRAM,
            labels=self.labels,
            count=new_count,
            sum=new_sum,
            buckets=tuple(new_buckets),
        )

    @property
    def average(self) -> float:
        """Average value of observations."""
        return self.sum / self.count if self.count > 0 else 0.0


@dataclass(frozen=True)
class SummaryMetric(Metric):
    """A summary metric with aggregated statistics.

    Summaries provide pre-calculated statistics like:
    - Cache hit rates
    - Success rates
    - Average response times

    Attributes:
        count: Total count
        sum: Sum of values
        average: Average value
        min_value: Minimum value
        max_value: Maximum value
        p50: 50th percentile (median)
        p95: 95th percentile
        p99: 99th percentile
    """

    count: int = 0
    sum: float = 0.0
    average: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None

    def __post_init__(self):
        """Initialize as summary type."""
        object.__setattr__(self, "metric_type", MetricType.SUMMARY)


@dataclass(frozen=True)
class MetricsSnapshot:
    """A snapshot of metrics at a point in time.

    Attributes:
        source_id: Unique identifier for the metrics source
        source_type: Type of metrics source (capability, cache, coordinator, etc.)
        timestamp: When the snapshot was taken
        metrics: All metrics in this snapshot
    """

    source_id: str
    source_type: str
    timestamp: float = field(default_factory=time.time)
    metrics: tuple[Metric, ...] = field(default_factory=tuple)

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name.

        Args:
            name: Metric name

        Returns:
            Metric or None if not found
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def filter_by_labels(self, **kwargs: str) -> "MetricsSnapshot":
        """Return a new snapshot with only matching metrics.

        Args:
            **kwargs: Label filters (key=value)

        Returns:
            New snapshot with filtered metrics
        """
        filtered_metrics = []
        for metric in self.metrics:
            matches_all = True
            for key, value in kwargs.items():
                if metric.get_label_value(key) != value:
                    matches_all = False
                    break
            if matches_all:
                filtered_metrics.append(metric)

        return type(self)(
            source_id=self.source_id,
            source_type=self.source_type,
            metrics=tuple(filtered_metrics),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary.

        Returns:
            Dictionary representation of snapshot
        """
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "metrics": [
                {
                    "name": m.name,
                    "description": m.description,
                    "type": m.metric_type.value,
                    "labels": {label.key: label.value for label in m.labels},
                    "value": _extract_metric_value(m),
                }
                for m in self.metrics
            ],
        }


def _extract_metric_value(metric: Metric) -> Dict[str, Any]:
    """Extract value from metric based on type.

    Args:
        metric: Metric to extract value from

    Returns:
        Dictionary with metric-specific value fields
    """
    if isinstance(metric, CounterMetric):
        return {"value": metric.value}
    elif isinstance(metric, GaugeMetric):
        return {
            "value": metric.value,
            "min": metric.min_value,
            "max": metric.max_value,
        }
    elif isinstance(metric, HistogramMetric):
        return {
            "count": metric.count,
            "sum": metric.sum,
            "average": metric.average,
            "buckets": [{"upper_bound": b.upper_bound, "count": b.count} for b in metric.buckets],
        }
    elif isinstance(metric, SummaryMetric):
        return {
            "count": metric.count,
            "sum": metric.sum,
            "average": metric.average,
            "min": metric.min_value,
            "max": metric.max_value,
            "p50": metric.p50,
            "p95": metric.p95,
            "p99": metric.p99,
        }
    else:
        return {}


@runtime_checkable
class MetricSource(Protocol):
    """Protocol for components that can provide metrics.

    Any component that implements this protocol can be registered
    with the ObservabilityManager for metrics collection.
    """

    source_id: str
    source_type: str

    def get_metrics(self) -> MetricsSnapshot:
        """Get current metrics snapshot.

        Returns:
            MetricsSnapshot with current metrics
        """
        ...

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data for dashboard.

        Returns:
            Dictionary with observability data
        """
        ...


@dataclass
class MetricsCollection:
    """A collection of metrics from multiple sources.

    Attributes:
        snapshots: Collection of metrics snapshots
        timestamp: When the collection was created
    """

    snapshots: List[MetricsSnapshot] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def add_snapshot(self, snapshot: MetricsSnapshot) -> None:
        """Add a metrics snapshot to the collection.

        Args:
            snapshot: Snapshot to add
        """
        self.snapshots.append(snapshot)

    def get_by_source_type(self, source_type: str) -> List[MetricsSnapshot]:
        """Get all snapshots from a specific source type.

        Args:
            source_type: Type of source (e.g., "capability", "cache")

        Returns:
            List of snapshots from matching sources
        """
        return [s for s in self.snapshots if s.source_type == source_type]

    def get_by_source_id(self, source_id: str) -> Optional[MetricsSnapshot]:
        """Get snapshot by source ID.

        Args:
            source_id: Unique source identifier

        Returns:
            Snapshot or None if not found
        """
        for snapshot in self.snapshots:
            if snapshot.source_id == source_id:
                return snapshot
        return None

    def aggregate_metric(self, metric_name: str) -> List[Metric]:
        """Aggregate all metrics with the given name across sources.

        Args:
            metric_name: Name of metric to aggregate

        Returns:
            List of metrics with matching names
        """
        metrics = []
        for snapshot in self.snapshots:
            metric = snapshot.get_metric(metric_name)
            if metric:
                metrics.append(metric)
        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary.

        Returns:
            Dictionary representation of collection
        """
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "sources_count": len(self.snapshots),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "by_type": {
                source_type: len(self.get_by_source_type(source_type))
                for source_type in set(s.source_type for s in self.snapshots)
            },
        }


# Common metric name constants


class MetricNames:
    """Standard metric names used across the framework."""

    # Capability metrics
    CAPABILITY_ACCESS_COUNT = "capability_access_count"
    CAPABILITY_LAST_ACCESSED = "capability_last_accessed"
    CAPABILITY_ERROR_COUNT = "capability_error_count"

    # Cache metrics
    CACHE_HIT_COUNT = "cache_hit_count"
    CACHE_MISS_COUNT = "cache_miss_count"
    CACHE_HIT_RATE = "cache_hit_rate"
    CACHE_SIZE = "cache_size"
    CACHE_EVICTION_COUNT = "cache_eviction_count"

    # Tool metrics
    TOOL_CALL_COUNT = "tool_call_count"
    TOOL_ERROR_COUNT = "tool_error_count"
    TOOL_LATENCY = "tool_latency"
    TOOL_SUCCESS_RATE = "tool_success_rate"

    # Coordinator metrics
    COORDINATOR_OPERATION_COUNT = "coordinator_operation_count"
    COORDINATOR_ERROR_COUNT = "coordinator_error_count"
    COORDINATOR_LATENCY = "coordinator_latency"

    # Vertical metrics
    VERTICAL_REQUEST_COUNT = "vertical_request_count"
    VERTICAL_ERROR_COUNT = "vertical_error_count"
    VERTICAL_LATENCY = "vertical_latency"

    # System metrics
    SYSTEM_MEMORY_USAGE = "system_memory_usage"
    SYSTEM_CPU_USAGE = "system_cpu_usage"
    SYSTEM_DISK_USAGE = "system_disk_usage"


__all__ = [
    # Enums
    "MetricType",
    # Data models
    "MetricLabel",
    "Metric",
    "CounterMetric",
    "GaugeMetric",
    "HistogramBucket",
    "HistogramMetric",
    "SummaryMetric",
    "MetricsSnapshot",
    "MetricsCollection",
    # Protocols
    "MetricSource",
    # Constants
    "MetricNames",
]
