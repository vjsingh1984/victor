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

"""Prometheus metrics exporter for coordinator-based orchestrator.

This module provides Prometheus-compatible metrics export with:
- Counter: Incrementing metrics (executions, errors, cache operations)
- Gauge: Point-in-time metrics (memory, active connections)
- Histogram: Distribution metrics (latency, response times)
- HTTP endpoint for Prometheus scraping (/metrics)

Integrates with CoordinatorMetricsCollector for seamless metrics export.

Example:
    from victor.observability.prometheus_metrics import PrometheusMetricsExporter
    from victor.observability.coordinator_metrics import get_coordinator_metrics_collector

    # Get metrics collector and create exporter
    collector = get_coordinator_metrics_collector()
    exporter = PrometheusMetricsExporter(collector)

    # Start Prometheus endpoint
    from fastapi import FastAPI
    app = FastAPI()
    app.add_route("/metrics", exporter.get_endpoint())

    # Or standalone
    import uvicorn
    exporter.start_server(port=9090)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from typing import Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metric Types
# =============================================================================


@dataclass
class Counter:
    """Prometheus Counter metric - monotonically increasing value."""

    name: str
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, delta: float = 1.0) -> None:
        """Increment counter.

        Args:
            delta: Amount to increment (must be positive).
        """
        if delta < 0:
            raise ValueError("Counter can only increment by positive amounts")
        with self._lock:
            self.value += delta

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        label_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            label_str = "{" + ",".join(label_pairs) + "}"

        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} counter",
            f"{self.name}{label_str} {self.value}",
        ]
        return "\n".join(lines)


@dataclass
class Gauge:
    """Prometheus Gauge metric - can go up or down."""

    name: str
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float) -> None:
        """Set gauge value.

        Args:
            value: New value.
        """
        with self._lock:
            self.value = value

    def inc(self, delta: float = 1.0) -> None:
        """Increment gauge.

        Args:
            delta: Amount to increment (can be negative).
        """
        with self._lock:
            self.value += delta

    def dec(self, delta: float = 1.0) -> None:
        """Decrement gauge.

        Args:
            delta: Amount to decrement (can be negative).
        """
        with self._lock:
            self.value -= delta

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        label_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            label_str = "{" + ",".join(label_pairs) + "}"

        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} gauge",
            f"{self.name}{label_str} {self.value}",
        ]
        return "\n".join(lines)


@dataclass
class Histogram:
    """Prometheus Histogram metric - distribution tracking."""

    name: str
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    _sum: float = 0.0
    _count: int = 0
    _bucket_counts: Dict[float, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Initialize bucket counters."""
        if not self._bucket_counts:
            for bucket in self.buckets:
                self._bucket_counts[bucket] = 0
            self._bucket_counts[float("inf")] = 0

    def observe(self, value: float) -> None:
        """Observe a value.

        Args:
            value: Observed value.
        """
        with self._lock:
            self._sum += value
            self._count += 1

            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
            self._bucket_counts[float("inf")] += 1

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        label_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            label_str = "{" + ",".join(label_pairs) + "}"

        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} histogram",
        ]

        # Bucket counts
        cumulative = 0
        for bucket in sorted(self.buckets) + [float("inf")]:
            cumulative += self._bucket_counts[bucket]
            if label_str:
                bucket_label = label_str.replace("}", f',le="{bucket}"}}')
            else:
                bucket_label = f'{{le="{bucket}"}}'
            lines.append(f"{self.name}_bucket{bucket_label} {cumulative}")

        # Sum and count
        lines.append(f"{self.name}_sum{label_str} {self._sum}")
        lines.append(f"{self.name}_count{label_str} {self._count}")

        return "\n".join(lines)


# =============================================================================
# Metrics Registry
# =============================================================================


class PrometheusRegistry:
    """Registry for Prometheus metrics.

    Thread-safe storage and export of all metrics.
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._lock = threading.RLock()
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

    def counter(
        self,
        name: str,
        help: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """Get or create a counter metric.

        Args:
            name: Metric name.
            help: Metric description.
            labels: Metric labels.

        Returns:
            Counter instance.
        """
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else ""
        key = f"{name}{{{label_str}}}" if label_str else name
        with self._lock:
            if key not in self._counters:
                self._counters[key] = Counter(
                    name=name,
                    help=help,
                    labels=labels or {},
                )
            return self._counters[key]

    def gauge(
        self,
        name: str,
        help: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """Get or create a gauge metric.

        Args:
            name: Metric name.
            help: Metric description.
            labels: Metric labels.

        Returns:
            Gauge instance.
        """
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else ""
        key = f"{name}{{{label_str}}}" if label_str else name
        with self._lock:
            if key not in self._gauges:
                self._gauges[key] = Gauge(
                    name=name,
                    help=help,
                    labels=labels or {},
                )
            return self._gauges[key]

    def histogram(
        self,
        name: str,
        help: str,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Get or create a histogram metric.

        Args:
            name: Metric name.
            help: Metric description.
            labels: Metric labels.
            buckets: Bucket boundaries.

        Returns:
            Histogram instance.
        """
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else ""
        key = f"{name}{{{label_str}}}" if label_str else name
        with self._lock:
            if key not in self._histograms:
                default_buckets = buckets or [
                    0.005,
                    0.01,
                    0.025,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    1.0,
                    2.5,
                    5.0,
                    10.0,
                ]
                self._histograms[key] = Histogram(
                    name=name,
                    help=help,
                    labels=labels or {},
                    buckets=default_buckets,
                )
            return self._histograms[key]

    def export(self) -> str:
        """Export all metrics in Prometheus format.

        Returns:
            Prometheus format string.
        """
        lines = []

        with self._lock:
            # Export counters
            for counter in self._counters.values():
                lines.append(counter.to_prometheus())
                lines.append("")  # Blank line between metrics

            # Export gauges
            for gauge in self._gauges.values():
                lines.append(gauge.to_prometheus())
                lines.append("")

            # Export histograms
            for histogram in self._histograms.values():
                lines.append(histogram.to_prometheus())
                lines.append("")

        return "\n".join(lines).strip()

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# =============================================================================
# Prometheus Metrics Exporter
# =============================================================================


class PrometheusMetricsExporter:
    """Export coordinator metrics in Prometheus format.

    Integrates with CoordinatorMetricsCollector to provide:
    - Automatic metric translation
    - Prometheus-compatible /metrics endpoint
    - Support for Counter, Gauge, Histogram types
    - FastAPI integration

    Example:
        from victor.observability.coordinator_metrics import get_coordinator_metrics_collector
        from victor.observability.prometheus_metrics import PrometheusMetricsExporter

        collector = get_coordinator_metrics_collector()
        exporter = PrometheusMetricsExporter(collector)

        # Get Prometheus format
        prometheus_text = exporter.export_metrics()

        # Or use as FastAPI endpoint
        from fastapi import FastAPI
        app = FastAPI()
        app.add_route("/metrics", exporter.get_endpoint())
    """

    def __init__(
        self,
        metrics_collector: Any,  # CoordinatorMetricsCollector
        registry: Optional[PrometheusRegistry] = None,
    ) -> None:
        """Initialize Prometheus exporter.

        Args:
            metrics_collector: CoordinatorMetricsCollector instance.
            registry: Optional custom registry (uses default if not provided).
        """
        self._collector = metrics_collector
        self._registry = registry or PrometheusRegistry()
        self._start_time = time.time()

    def export_metrics(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus format string.
        """
        lines = []

        # Get snapshots from collector
        snapshots = self._collector.get_all_snapshots()
        overall_stats = self._collector.get_overall_stats()

        # Coordinator execution metrics (Counter)
        for snapshot in snapshots:
            coordinator = snapshot.coordinator_name

            # Execution count
            lines.append(
                f'victor_coordinator_executions_total{{coordinator="{coordinator}"}} {snapshot.execution_count}'
            )

            # Total duration
            lines.append(
                f'victor_coordinator_duration_seconds_total{{coordinator="{coordinator}"}} {snapshot.total_duration_ms / 1000:.6f}'
            )

            # Error count
            lines.append(
                f'victor_coordinator_errors_total{{coordinator="{coordinator}"}} {snapshot.error_count}'
            )

            # Cache hits/misses
            lines.append(
                f'victor_coordinator_cache_hits_total{{coordinator="{coordinator}"}} {snapshot.cache_hits}'
            )
            lines.append(
                f'victor_coordinator_cache_misses_total{{coordinator="{coordinator}"}} {snapshot.cache_misses}'
            )

            # Memory usage (Gauge)
            lines.append(
                f'victor_coordinator_memory_bytes{{coordinator="{coordinator}"}} {snapshot.memory_bytes}'
            )

            # CPU usage (Gauge)
            lines.append(
                f'victor_coordinator_cpu_percent{{coordinator="{coordinator}"}} {snapshot.cpu_percent:.2f}'
            )

            # Cache hit rate (Gauge)
            cache_hit_rate = snapshot.to_dict()["cache_hit_rate"]
            lines.append(
                f'victor_coordinator_cache_hit_rate{{coordinator="{coordinator}"}} {cache_hit_rate:.4f}'
            )

        # Overall metrics
        lines.append(f"victor_coordinator_uptime_seconds {overall_stats['uptime_seconds']:.2f}")
        lines.append(f"victor_coordinator_total_executions {overall_stats['total_executions']}")
        lines.append(
            f"victor_coordinator_active_coordinators {overall_stats['total_coordinators']}"
        )

        # Analytics events (Counter)
        for event_name, count in overall_stats.get("analytics_events", {}).items():
            lines.append(f'victor_analytics_events_total{{event="{event_name}"}} {count}')

        # Error rate (Gauge)
        error_rate = overall_stats.get("overall_error_rate", 0)
        lines.append(f"victor_coordinator_error_rate {error_rate:.4f}")

        # Throughput (Gauge)
        uptime = max(overall_stats["uptime_seconds"], 1)
        throughput = overall_stats["total_executions"] / uptime
        lines.append(f"victor_coordinator_throughput {throughput:.4f}")

        # Scrape metadata
        lines.append(f"victor_scrape_timestamp_seconds {time.time():.3f}")

        return "\n".join(lines)

    def get_endpoint(self) -> Callable:
        """Get FastAPI endpoint handler.

        Returns:
            Async function for use as FastAPI route handler.

        Example:
            from fastapi import FastAPI
            app = FastAPI()
            app.add_route("/metrics", exporter.get_endpoint())
        """

        async def metrics_handler():
            """Handle metrics scrape request."""
            from fastapi.responses import PlainTextResponse

            metrics_text = self.export_metrics()
            return PlainTextResponse(
                content=metrics_text,
                media_type="text/plain",
            )

        return metrics_handler

    def start_server(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        log_level: str = "info",
    ) -> None:
        """Start standalone Prometheus metrics server.

        This is a simple way to expose metrics without integrating with
        an existing FastAPI application.

        Args:
            host: Server host.
            port: Server port.
            log_level: Log level.

        Example:
            exporter.start_server(port=9090)
            # Metrics available at http://localhost:9090/metrics
        """
        try:
            from fastapi import FastAPI
            from fastapi.responses import PlainTextResponse
            import uvicorn
        except ImportError as e:
            raise ImportError(
                "FastAPI and uvicorn are required for standalone server. "
                "Install with: pip install victor-ai[api]"
            ) from e

        app = FastAPI(title="Victor Prometheus Metrics")

        @app.get("/metrics", response_class=PlainTextResponse)
        async def metrics():
            return self.export_metrics()

        @app.get("/health")
        async def health():
            return {"status": "healthy", "service": "prometheus-exporter"}

        uvicorn.run(app, host=host, port=port, log_level=log_level)


# =============================================================================
# Decorators for Auto-Metrics
# =============================================================================


def track_prometheus_metrics(
    registry: PrometheusRegistry,
    coordinator_name: Optional[str] = None,
):
    """Decorator to automatically track method calls in Prometheus.

    Args:
        registry: Prometheus registry.
        coordinator_name: Optional coordinator name.

    Example:
        registry = PrometheusRegistry()

        class MyCoordinator:
            @track_prometheus_metrics(registry, "MyCoordinator")
            def process(self):
                # Do work
                pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            name = coordinator_name or func.__qualname__

            # Get metrics
            counter = registry.counter(
                name="victor_coordinator_calls_total",
                help="Total coordinator calls",
                labels={"coordinator": name},
            )
            histogram = registry.histogram(
                name="victor_coordinator_duration_seconds",
                help="Coordinator call duration",
                labels={"coordinator": name},
            )

            # Track execution
            start = time.time()
            try:
                result = func(*args, **kwargs)
                counter.inc()
                return result
            except Exception:
                counter.inc()
                raise
            finally:
                duration = time.time() - start
                histogram.observe(duration)

        return wrapper

    return decorator


# =============================================================================
# Singleton Instance
# =============================================================================

_default_registry: Optional[PrometheusRegistry] = None
_default_lock = threading.Lock()


def get_prometheus_registry() -> PrometheusRegistry:
    """Get default Prometheus registry (singleton).

    Returns:
        PrometheusRegistry instance.
    """
    global _default_registry

    if _default_registry is None:
        with _default_lock:
            if _default_registry is None:
                _default_registry = PrometheusRegistry()

    return _default_registry


def get_prometheus_exporter() -> PrometheusMetricsExporter:
    """Get default Prometheus exporter (singleton).

    Returns:
        PrometheusMetricsExporter instance.
    """
    from victor.observability.coordinator_metrics import get_coordinator_metrics_collector

    collector = get_coordinator_metrics_collector()
    return PrometheusMetricsExporter(collector, get_prometheus_registry())
