"""Performance monitoring for tool registry operations.

This module provides metrics collection and export for registry operations,
enabling observability and alerting on performance degradation.

Integrates with Prometheus for metrics collection and Grafana for visualization.
"""

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import wraps
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics disabled")


@dataclass
class OperationMetrics:
    """Metrics for a single operation.

    Attributes:
        operation: Name of the operation (e.g., "register", "get_by_tag")
        count: Number of times operation was performed
        total_duration_ms: Total time spent in operation (milliseconds)
        min_duration_ms: Minimum duration observed
        max_duration_ms: Maximum duration observed
        last_duration_ms: Most recent duration
        error_count: Number of errors encountered
        last_error: Last error message (if any)
    """

    operation: str
    count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    last_duration_ms: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    last_update: datetime = field(default_factory=datetime.now)

    def record(self, duration_ms: float, error: Optional[str] = None) -> None:
        """Record a single operation execution.

        Args:
            duration_ms: Duration of the operation in milliseconds
            error: Error message if operation failed, None otherwise
        """
        self.count += 1
        self.total_duration_ms += duration_ms
        self.last_duration_ms = duration_ms
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        self.last_update = datetime.now()

        if error:
            self.error_count += 1
            self.last_error = error

    @property
    def avg_duration_ms(self) -> float:
        """Average duration in milliseconds."""
        if self.count == 0:
            return 0.0
        return self.total_duration_ms / self.count

    @property
    def error_rate(self) -> float:
        """Error rate as percentage (0-100)."""
        if self.count == 0:
            return 0.0
        return (self.error_count / self.count) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for export."""
        return {
            "operation": self.operation,
            "count": self.count,
            "avg_duration_ms": round(self.avg_duration_ms, 3),
            "min_duration_ms": round(self.min_duration_ms, 3) if self.min_duration_ms != float('inf') else 0,
            "max_duration_ms": round(self.max_duration_ms, 3),
            "last_duration_ms": round(self.last_duration_ms, 3),
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 2),
            "last_error": self.last_error,
            "last_update": self.last_update.isoformat(),
        }


class RegistryMetricsCollector:
    """Collects and manages metrics for tool registry operations.

    Provides:
    - Operation timing (p50, p95, p99 latencies)
    - Operation counts (success, failure)
    - Cache statistics (hit rate, size, evictions)
    - Resource usage (memory, thread count)

    Thread-safe for concurrent access.
    """

    def __init__(self, enabled: bool = True):
        """Initialize metrics collector.

        Args:
            enabled: Whether metrics collection is enabled
        """
        self._enabled = enabled
        self._lock = threading.RLock()
        self._metrics: Dict[str, OperationMetrics] = {}
        self._cache_metrics: Dict[str, Any] = {
            "feature_flag_hits": 0,
            "feature_flag_misses": 0,
            "query_hits": 0,
            "query_misses": 0,
            "query_evictions": 0,
            "query_size": 0,
        }

        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE and enabled:
            self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics collectors."""
        self._prometheus_registry = CollectorRegistry()

        # Counters
        self._operation_counter = Counter(
            'registry_operations_total',
            'Total number of registry operations',
            ['operation', 'status'],
            registry=self._prometheus_registry
        )

        # Histograms
        self._operation_duration = Histogram(
            'registry_operation_duration_ms',
            'Registry operation duration in milliseconds',
            ['operation'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0),
            registry=self._prometheus_registry
        )

        # Gauges
        self._cache_size = Gauge(
            'registry_cache_size',
            'Current cache size',
            ['cache_type'],
            registry=self._prometheus_registry
        )

        self._cache_hit_rate = Gauge(
            'registry_cache_hit_rate',
            'Cache hit rate percentage',
            ['cache_type'],
            registry=self._prometheus_registry
        )

    @contextmanager
    def record_operation(self, operation: str):
        """Context manager for recording operation timing.

        Args:
            operation: Name of the operation being recorded

        Example:
            >>> with metrics.record_operation("register_tool"):
            ...     registry.register(tool)
        """
        if not self._enabled:
            yield
            return

        start_time = time.perf_counter()
        error = None

        try:
            yield
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._record(operation, duration_ms, error)

    def _record(self, operation: str, duration_ms: float, error: Optional[str] = None) -> None:
        """Record a single operation.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            error: Error message if operation failed
        """
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = OperationMetrics(operation=operation)

            self._metrics[operation].record(duration_ms, error)

            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                status = "error" if error else "success"
                self._operation_counter.labels(operation=operation, status=status).inc()
                self._operation_duration.labels(operation=operation).observe(duration_ms)

    def record_cache_stats(self, cache_type: str, stats: Dict[str, Any]) -> None:
        """Record cache statistics.

        Args:
            cache_type: Type of cache (e.g., "feature_flag", "query")
            stats: Dictionary with cache statistics
        """
        if not self._enabled:
            return

        with self._lock:
            # Update internal cache metrics
            if cache_type == "feature_flag":
                self._cache_metrics["feature_flag_hits"] += stats.get("hits", 0)
                self._cache_metrics["feature_flag_misses"] += stats.get("misses", 0)
            elif cache_type == "query":
                self._cache_metrics["query_hits"] += stats.get("hits", 0)
                self._cache_metrics["query_misses"] += stats.get("misses", 0)
                self._cache_metrics["query_evictions"] += stats.get("evictions", 0)
                self._cache_metrics["query_size"] = stats.get("cache_size", 0)

            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self._cache_size.labels(cache_type=cache_type).set(stats.get("cache_size", 0))

                hit_rate = stats.get("hit_rate", 0.0)
                self._cache_hit_rate.labels(cache_type=cache_type).set(hit_rate)

    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for operations.

        Args:
            operation: Specific operation to get metrics for, or None for all

        Returns:
            Dictionary of metrics
        """
        with self._lock:
            if operation:
                if operation in self._metrics:
                    return self._metrics[operation].to_dict()
                return {}

            return {
                op: metrics.to_dict()
                for op, metrics in self._metrics.items()
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Dictionary with operation counts, averages, and cache stats
        """
        with self._lock:
            total_operations = sum(m.count for m in self._metrics.values())
            total_errors = sum(m.error_count for m in self._metrics.values())

            # Calculate overall error rate
            overall_error_rate = (total_errors / total_operations * 100) if total_operations > 0 else 0.0

            # Get slowest operations
            sorted_ops = sorted(
                self._metrics.items(),
                key=lambda x: x[1].avg_duration_ms,
                reverse=True
            )

            return {
                "total_operations": total_operations,
                "total_errors": total_errors,
                "overall_error_rate": round(overall_error_rate, 2),
                "operation_count": len(self._metrics),
                "slowest_operations": [
                    {"operation": op, "avg_ms": round(m.avg_duration_ms, 3)}
                    for op, m in sorted_ops[:5]
                ],
                "cache_metrics": self._cache_metrics.copy(),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._cache_metrics = {
                "feature_flag_hits": 0,
                "feature_flag_misses": 0,
                "query_hits": 0,
                "query_misses": 0,
                "query_evictions": 0,
                "query_size": 0,
            }

    def start_metrics_server(self, port: int = 8000) -> None:
        """Start Prometheus metrics HTTP server.

        Args:
            port: Port to expose metrics on

        Example:
            >>> collector.start_metrics_server(8000)
            >>> # Metrics available at http://localhost:8000/metrics
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available, cannot start metrics server")
            return

        try:
            start_http_server(port, registry=self._prometheus_registry)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")


# Global metrics collector instance
_global_collector: Optional[RegistryMetricsCollector] = None
_global_lock = threading.Lock()


def get_metrics_collector() -> RegistryMetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Global RegistryMetricsCollector instance
    """
    global _global_collector

    with _global_lock:
        if _global_collector is None:
            _global_collector = RegistryMetricsCollector(enabled=True)

    return _global_collector


def monitored_operation(operation: str):
    """Decorator to monitor function execution time.

    Args:
        operation: Name of the operation for metrics

    Example:
        >>> @monitored_operation("register_batch")
        ... def register_batch(tools):
        ...     # Implementation
        ...     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            with collector.record_operation(operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    "RegistryMetricsCollector",
    "OperationMetrics",
    "get_metrics_collector",
    "monitored_operation",
]
