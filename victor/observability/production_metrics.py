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

"""Production-grade metrics collector with comprehensive observability.

This module provides ProductionMetricsCollector which extends the base metrics
system with production-ready features:

- Request rate, latency, error rate tracking
- Cache hit rates and memory usage
- Coordinator-specific metrics
- Business metrics (tasks completed, tools used)
- Metrics export for Prometheus/Grafana

Design Patterns:
- Facade Pattern: Unifies metrics collection
- Observer Pattern: Subscribes to event bus
- Strategy Pattern: Pluggable export backends

Example:
    from victor.observability.production_metrics import (
        ProductionMetricsCollector,
        create_production_collector,
    )

    # Create collector
    collector = create_production_collector()

    # Record metrics
    collector.increment_counter("requests_total", labels={"endpoint": "/chat"})
    collector.record_latency("request_latency_ms", 45.2, labels={"provider": "anthropic"})
    collector.set_gauge("active_connections", 10)

    # Export metrics
    prometheus_text = collector.export_prometheus()
    json_data = collector.export_json()
"""

from __future__ import annotations

import logging
import os
import psutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from victor.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    Timer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Types
# =============================================================================


class MetricCategory(str, Enum):
    """Categories of metrics for organization."""

    REQUEST = "request"
    TOOL = "tool"
    COORDINATOR = "coordinator"
    BUSINESS = "business"
    SYSTEM = "system"
    CACHE = "cache"
    ERROR = "error"


@dataclass
class MetricLabel:
    """Label for metric categorization."""

    name: str
    value: str

    def to_tuple(self) -> tuple[str, str]:
        """Convert to tuple for registry."""
        return (self.name, self.value)


# =============================================================================
# Production Metrics Collector
# =============================================================================


class ProductionMetricsCollector:
    """Production-grade metrics collector.

    Provides comprehensive metrics collection for production monitoring.
    Tracks request rates, latencies, errors, cache performance, and system metrics.

    Attributes:
        registry: Underlying metrics registry
        prefix: Prefix for all metric names
        process: psutil Process for system metrics

    Example:
        collector = ProductionMetricsCollector()

        # Request metrics
        collector.record_request("chat", "anthropic", success=True, latency_ms=123.4)

        # Tool metrics
        collector.record_tool_execution("read_file", success=True, duration_ms=45.2)

        # System metrics (auto-collected)
        collector.collect_system_metrics()

        # Export
        prometheus_text = collector.export_prometheus()
    """

    def __init__(
        self,
        registry: Optional[MetricsRegistry] = None,
        prefix: str = "victor",
        collect_system_metrics: bool = True,
    ) -> None:
        """Initialize production metrics collector.

        Args:
            registry: Metrics registry (uses default if None)
            prefix: Metric name prefix
            collect_system_metrics: Whether to collect system metrics
        """
        self._registry = registry or MetricsRegistry.get_instance()
        self._prefix = prefix
        self._collect_system_metrics = collect_system_metrics
        self._process = psutil.Process() if collect_system_metrics else None
        self._lock = threading.RLock()
        self._start_time = time.time()

        # Initialize metrics
        self._setup_request_metrics()
        self._setup_tool_metrics()
        self._setup_coordinator_metrics()
        self._setup_business_metrics()
        self._setup_cache_metrics()
        self._setup_error_metrics()
        self._setup_system_metrics()

    # ========================================================================
    # Metric Setup
    # ========================================================================

    def _setup_request_metrics(self) -> None:
        """Setup request-related metrics."""
        # Request counters
        self.request_total = self._registry.counter(
            f"{self._prefix}_request_total",
            "Total requests",
            {"category": MetricCategory.REQUEST.value},
        )
        self.request_success = self._registry.counter(
            f"{self._prefix}_request_success_total",
            "Total successful requests",
        )
        self.request_errors = self._registry.counter(
            f"{self._prefix}_request_errors_total",
            "Total request errors",
        )

        # Request latency
        self.request_latency = self._registry.histogram(
            f"{self._prefix}_request_latency_ms",
            "Request latency in milliseconds",
            buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
        )

        # Active requests
        self.active_requests = self._registry.gauge(
            f"{self._prefix}_active_requests",
            "Active request count",
        )

    def _setup_tool_metrics(self) -> None:
        """Setup tool execution metrics."""
        self.tool_calls_total = self._registry.counter(
            f"{self._prefix}_tool_calls_total",
            "Total tool calls",
        )
        self.tool_errors_total = self._registry.counter(
            f"{self._prefix}_tool_errors_total",
            "Total tool errors",
        )
        self.tool_duration = self._registry.histogram(
            f"{self._prefix}_tool_duration_ms",
            "Tool execution duration",
        )
        self.tool_active = self._registry.gauge(
            f"{self._prefix}_tool_active",
            "Active tool executions",
        )

    def _setup_coordinator_metrics(self) -> None:
        """Setup coordinator-specific metrics."""
        self.coordinator_invocations = self._registry.counter(
            f"{self._prefix}_coordinator_invocations_total",
            "Total coordinator invocations",
        )
        self.coordinator_duration = self._registry.histogram(
            f"{self._prefix}_coordinator_duration_ms",
            "Coordinator execution duration",
        )
        self.coordinator_errors = self._registry.counter(
            f"{self._prefix}_coordinator_errors_total",
            "Total coordinator errors",
        )

    def _setup_business_metrics(self) -> None:
        """Setup business-level metrics."""
        self.tasks_completed = self._registry.counter(
            f"{self._prefix}_tasks_completed_total",
            "Total completed tasks",
        )
        self.tasks_failed = self._registry.counter(
            f"{self._prefix}_tasks_failed_total",
            "Total failed tasks",
        )
        self.total_cost = self._registry.gauge(
            f"{self._prefix}_total_cost_usd",
            "Total cost in USD",
        )
        self.tokens_used = self._registry.counter(
            f"{self._prefix}_tokens_used_total",
            "Total tokens used",
        )

    def _setup_cache_metrics(self) -> None:
        """Setup cache performance metrics."""
        self.cache_hits = self._registry.counter(
            f"{self._prefix}_cache_hits_total",
            "Total cache hits",
        )
        self.cache_misses = self._registry.counter(
            f"{self._prefix}_cache_misses_total",
            "Total cache misses",
        )
        self.cache_size = self._registry.gauge(
            f"{self._prefix}_cache_size",
            "Current cache size",
        )
        self.cache_evictions = self._registry.counter(
            f"{self._prefix}_cache_evictions_total",
            "Total cache evictions",
        )

    def _setup_error_metrics(self) -> None:
        """Setup error tracking metrics."""
        self.errors_total = self._registry.counter(
            f"{self._prefix}_errors_total",
            "Total errors",
        )
        self.errors_by_type: Dict[str, Counter] = {}
        self.error_rate = self._registry.gauge(
            f"{self._prefix}_error_rate",
            "Current error rate",
        )

    def _setup_system_metrics(self) -> None:
        """Setup system resource metrics."""
        if not self._collect_system_metrics:
            return

        self.memory_usage = self._registry.gauge(
            f"{self._prefix}_memory_usage_bytes",
            "Memory usage in bytes",
        )
        self.cpu_usage = self._registry.gauge(
            f"{self._prefix}_cpu_usage_percent",
            "CPU usage percentage",
        )
        self.disk_usage = self._registry.gauge(
            f"{self._prefix}_disk_usage_bytes",
            "Disk usage in bytes",
        )
        self.open_fds = self._registry.gauge(
            f"{self._prefix}_open_file_descriptors",
            "Open file descriptors",
        )
        self.uptime = self._registry.gauge(
            f"{self._prefix}_uptime_seconds",
            "Service uptime in seconds",
        )

    # ========================================================================
    # Request Metrics
    # ========================================================================

    def record_request(
        self,
        endpoint: str,
        provider: str,
        success: bool,
        latency_ms: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a request metric.

        Args:
            endpoint: Request endpoint (e.g., "chat", "stream")
            provider: LLM provider name
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            labels: Optional additional labels
        """
        base_labels = {"endpoint": endpoint, "provider": provider}
        if labels:
            base_labels.update(labels)

        with self._lock:
            self.request_total.increment()
            if success:
                self.request_success.increment()
            else:
                self.request_errors.increment()

            self.request_latency.observe(latency_ms)

    # ========================================================================
    # Tool Metrics
    # ========================================================================

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a tool execution metric.

        Args:
            tool_name: Name of the tool
            success: Whether execution succeeded
            duration_ms: Execution duration in milliseconds
            labels: Optional additional labels
        """
        base_labels = {"tool": tool_name}
        if labels:
            base_labels.update(labels)

        with self._lock:
            self.tool_calls_total.increment()
            if not success:
                self.tool_errors_total.increment()

            self.tool_duration.observe(duration_ms)

    def increment_tool_active(self, count: int = 1) -> None:
        """Increment active tool count.

        Args:
            count: Amount to increment
        """
        self.tool_active.increment(count)

    def decrement_tool_active(self, count: int = 1) -> None:
        """Decrement active tool count.

        Args:
            count: Amount to decrement
        """
        self.tool_active.decrement(count)

    # ========================================================================
    # Coordinator Metrics
    # ========================================================================

    def record_coordinator_execution(
        self,
        coordinator_name: str,
        success: bool,
        duration_ms: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record coordinator execution metric.

        Args:
            coordinator_name: Name of the coordinator
            success: Whether execution succeeded
            duration_ms: Execution duration in milliseconds
            labels: Optional additional labels
        """
        base_labels = {"coordinator": coordinator_name}
        if labels:
            base_labels.update(labels)

        with self._lock:
            self.coordinator_invocations.increment()
            if not success:
                self.coordinator_errors.increment()

            self.coordinator_duration.observe(duration_ms)

    # ========================================================================
    # Business Metrics
    # ========================================================================

    def record_task_completion(self, success: bool, cost_usd: float = 0.0) -> None:
        """Record task completion.

        Args:
            success: Whether task succeeded
            cost_usd: Task cost in USD
        """
        with self._lock:
            if success:
                self.tasks_completed.increment()
            else:
                self.tasks_failed.increment()

            if cost_usd > 0:
                current = self.total_cost.value
                self.total_cost.set(current + cost_usd)

    def record_token_usage(self, token_count: int) -> None:
        """Record token usage.

        Args:
            token_count: Number of tokens used
        """
        self.tokens_used.increment(token_count)

    # ========================================================================
    # Cache Metrics
    # ========================================================================

    def record_cache_hit(self, cache_name: str) -> None:
        """Record cache hit.

        Args:
            cache_name: Name of the cache
        """
        with self._lock:
            self.cache_hits.increment()

    def record_cache_miss(self, cache_name: str) -> None:
        """Record cache miss.

        Args:
            cache_name: Name of the cache
        """
        with self._lock:
            self.cache_misses.increment()

    def set_cache_size(self, cache_name: str, size: int) -> None:
        """Set cache size.

        Args:
            cache_name: Name of the cache
            size: Cache size
        """
        self.cache_size.set(size)

    def record_cache_eviction(self, cache_name: str) -> None:
        """Record cache eviction.

        Args:
            cache_name: Name of the cache
        """
        with self._lock:
            self.cache_evictions.increment()

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as percentage (0-100)
        """
        hits = self.cache_hits.value
        misses = self.cache_misses.value
        total = hits + misses

        if total == 0:
            return 0.0

        return (hits / total) * 100.0

    # ========================================================================
    # Error Metrics
    # ========================================================================

    def record_error(self, error_type: str, message: str = "") -> None:
        """Record an error.

        Args:
            error_type: Type of error
            message: Error message
        """
        with self._lock:
            self.errors_total.increment()

            # Track errors by type
            if error_type not in self.errors_by_type:
                self.errors_by_type[error_type] = self._registry.counter(
                    f'{self._prefix}_errors_{error_type.replace(".", "_").replace(":", "_")}_total',
                    f"Total {error_type} errors",
                )

            self.errors_by_type[error_type].increment()

            # Update error rate
            total_requests = self.request_total.value
            if total_requests > 0:
                error_rate = (self.errors_total.value / total_requests) * 100
                self.error_rate.set(error_rate)

    # ========================================================================
    # System Metrics
    # ========================================================================

    def collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        if not self._collect_system_metrics or not self._process:
            return

        try:
            # Memory usage
            memory_info = self._process.memory_info()
            self.memory_usage.set(memory_info.rss)

            # CPU usage
            cpu_percent = self._process.cpu_percent(interval=0.1)
            self.cpu_usage.set(cpu_percent)

            # Disk usage
            disk_usage = psutil.disk_usage(".")
            self.disk_usage.set(disk_usage.used)

            # Open file descriptors
            try:
                num_fds = self._process.num_fds()
                self.open_fds.set(num_fds)
            except (AttributeError, NotImplementedError):
                # Not available on all platforms
                pass

            # Uptime
            uptime = time.time() - self._start_time
            self.uptime.set(uptime)

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    # ========================================================================
    # Metric Export
    # ========================================================================

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        for metric_data in self._registry.collect():
            metric_type = metric_data["type"]
            name = metric_data["name"]
            description = metric_data.get("description", "")
            labels = metric_data.get("labels", {})

            # HELP and TYPE
            if description:
                lines.append(f"# HELP {name} {description}")
            lines.append(f"# TYPE {name} {metric_type}")

            # Metric value
            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                lines.append(f'{name}{{{label_str}}} {metric_data.get("value", 0)}')
            else:
                lines.append(f"{name} {metric_data.get('value', 0)}")

            # Histogram buckets
            if metric_type == "histogram":
                buckets = metric_data.get("buckets", {})
                for bucket, count in sorted(buckets.items()):
                    if bucket == float("inf"):
                        bucket_str = "+Inf"
                    else:
                        bucket_str = str(bucket)

                    if labels:
                        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                        lines.append(f'{name}_bucket{{{label_str},le="{bucket_str}"}} {count}')
                    else:
                        lines.append(f'{name}_bucket{{le="{bucket_str}"}} {count}')

                # Sum and count
                if labels:
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                    lines.append(f'{name}_sum{{{label_str}}} {metric_data.get("sum", 0)}')
                    lines.append(f'{name}_count{{{label_str}}} {metric_data.get("count", 0)}')
                else:
                    lines.append(f'{name}_sum {metric_data.get("sum", 0)}')
                    lines.append(f'{name}_count {metric_data.get("count", 0)}')

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export metrics as JSON-serializable dict.

        Returns:
            Dictionary with all metrics
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self._registry.collect(),
            "summary": self.get_summary(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns:
            Summary dictionary with key metrics
        """
        with self._lock:
            total_requests = self.request_total.value
            total_errors = self.request_errors.value

            return {
                "requests": {
                    "total": total_requests,
                    "success": self.request_success.value,
                    "errors": total_errors,
                    "error_rate": (
                        (total_errors / total_requests * 100) if total_requests > 0 else 0.0
                    ),
                    "latency_p50": self.request_latency.percentile(50),
                    "latency_p95": self.request_latency.percentile(95),
                    "latency_p99": self.request_latency.percentile(99),
                },
                "tools": {
                    "calls": self.tool_calls_total.value,
                    "errors": self.tool_errors_total.value,
                    "active": self.tool_active.value,
                    "duration_p50": self.tool_duration.percentile(50),
                    "duration_p95": self.tool_duration.percentile(95),
                },
                "cache": {
                    "hits": self.cache_hits.value,
                    "misses": self.cache_misses.value,
                    "hit_rate": self.get_cache_hit_rate(),
                    "size": self.cache_size.value,
                },
                "business": {
                    "tasks_completed": self.tasks_completed.value,
                    "tasks_failed": self.tasks_failed.value,
                    "total_cost_usd": self.total_cost.value,
                    "tokens_used": self.tokens_used.value,
                },
                "system": {
                    "uptime_seconds": self.uptime.value if self._collect_system_metrics else None,
                    "memory_bytes": (
                        self.memory_usage.value if self._collect_system_metrics else None
                    ),
                    "cpu_percent": self.cpu_usage.value if self._collect_system_metrics else None,
                },
            }


# =============================================================================
# Factory Functions
# =============================================================================


def create_production_collector(
    prefix: str = "victor",
    collect_system_metrics: bool = True,
) -> ProductionMetricsCollector:
    """Create a production metrics collector.

    Args:
        prefix: Metric name prefix
        collect_system_metrics: Whether to collect system metrics

    Returns:
        Configured ProductionMetricsCollector instance

    Example:
        collector = create_production_collector()
        collector.record_request("chat", "anthropic", True, 123.4)
    """
    return ProductionMetricsCollector(
        prefix=prefix,
        collect_system_metrics=collect_system_metrics,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ProductionMetricsCollector",
    "create_production_collector",
    "MetricCategory",
    "MetricLabel",
]
