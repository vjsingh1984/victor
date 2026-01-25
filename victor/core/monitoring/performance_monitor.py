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

"""Performance monitoring and metrics collection.

This module provides comprehensive performance monitoring for tracking
optimization effectiveness and identifying bottlenecks.

Features:
- Operation timing with percentiles
- Memory usage tracking
- Cache hit rate monitoring
- Hot path identification
- Performance alerts
- Export metrics to various backends
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


# =============================================================================
# Operation Categories
# =============================================================================


class OperationCategory(Enum):
    """Categories of operations for monitoring.

    LLM_REQUEST: LLM API calls
    TOOL_EXECUTION: Tool execution
    CACHE_ACCESS: Cache read/write operations
    SERIALIZATION: JSON/other serialization
    EMBEDDING: Embedding computation
    BATCH_OPERATION: Batched operations
    GENERAL: Other operations
    """

    LLM_REQUEST = "llm_request"
    TOOL_EXECUTION = "tool_execution"
    CACHE_ACCESS = "cache_access"
    SERIALIZATION = "serialization"
    EMBEDDING = "embedding"
    BATCH_OPERATION = "batch_operation"
    GENERAL = "general"


# =============================================================================
# Performance Metrics
# =============================================================================


@dataclass
class OperationMetrics:
    """Metrics for a single operation type.

    Thread-safe: All operations protected by lock.

    Attributes:
        count: Number of operations
        total_time: Total execution time (seconds)
        min_time: Minimum execution time (seconds)
        max_time: Maximum execution time (seconds)
        avg_time: Average execution time (seconds)
        p50_time: 50th percentile execution time
        p95_time: 95th percentile execution time
        p99_time: 99th percentile execution time
        error_count: Number of errors
        last_execution: Last execution timestamp
    """

    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    error_count: int = 0
    last_execution: float = 0.0

    # Track recent execution times for percentiles
    _samples: "deque[float]" = field(default_factory=lambda: deque(maxlen=1000))

    def __post_init__(self) -> None:
        """Initialize thread lock."""
        self._lock = threading.Lock()

    def record(self, execution_time: float, success: bool = True) -> None:
        """Record an operation execution.

        Args:
            execution_time: Execution time in seconds
            success: Whether operation succeeded
        """
        with self._lock:
            self.count += 1
            self.total_time += execution_time
            self.min_time = min(self.min_time, execution_time)
            self.max_time = max(self.max_time, execution_time)
            self.last_execution = time.time()

            if not success:
                self.error_count += 1

            # Store sample for percentiles
            self._samples.append(execution_time)

    def get_avg_time(self) -> float:
        """Get average execution time.

        Returns:
            Average time or 0 if no operations
        """
        with self._lock:
            if self.count == 0:
                return 0.0
            return self.total_time / self.count

    def get_percentile(self, percentile: float) -> float:
        """Get percentile execution time.

        Args:
            percentile: Percentile to compute (0-100)

        Returns:
            Percentile value or 0 if no samples
        """
        with self._lock:
            if not self._samples:
                return 0.0

            sorted_samples = sorted(self._samples)
            idx = int(len(sorted_samples) * percentile / 100)
            idx = min(idx, len(sorted_samples) - 1)
            return sorted_samples[idx]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics as dictionary.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "count": self.count,
                "total_time": self.total_time,
                "avg_time": self.get_avg_time(),
                "min_time": self.min_time if self.count > 0 else 0.0,
                "max_time": self.max_time,
                "p50_time": self.get_percentile(50),
                "p95_time": self.get_percentile(95),
                "p99_time": self.get_percentile(99),
                "error_count": self.error_count,
                "error_rate": self.error_count / self.count if self.count > 0 else 0.0,
                "last_execution": self.last_execution,
            }


# =============================================================================
# Performance Monitor
# =============================================================================


class PerformanceMonitor:
    """Global performance monitoring system.

    Tracks performance metrics across all operations and provides
    analytics for optimization effectiveness.

    Features:
    - Operation timing with percentiles
    - Error tracking
    - Performance alerts
    - Metrics export
    - Hot path identification

    Example:
        ```python
        monitor = PerformanceMonitor()

        # Time an operation
        with monitor.track("llm_request", OperationCategory.LLM_REQUEST):
            response = await llm.chat(...)

        # Get statistics
        stats = monitor.get_stats("llm_request")
        ```
    """

    def __init__(
        self,
        max_operations: int = 1000,
        alert_threshold: float = 5.0,
        enabled: bool = True,
    ):
        """Initialize performance monitor.

        Args:
            max_operations: Maximum operations to track per type
            alert_threshold: Threshold for slow operation alerts (seconds)
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self.alert_threshold = alert_threshold
        self.max_operations = max_operations

        # Metrics storage (thread-safe)
        self._metrics: Dict[str, OperationMetrics] = {}
        self._lock = threading.RLock()

        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, float], None]] = []

    def track(
        self,
        operation_name: str,
        category: OperationCategory = OperationCategory.GENERAL,
    ) -> Any:
        """Context manager for tracking an operation.

        Args:
            operation_name: Name of the operation
            category: Operation category

        Returns:
            Context manager for tracking

        Example:
            ```python
            with monitor.track("database_query", OperationCategory.GENERAL):
                result = database.query(...)
            ```
        """
        if not self.enabled:
            return _NullContext()

        return _PerformanceContext(self, operation_name, category)

    def record_operation(
        self,
        operation_name: str,
        execution_time: float,
        category: OperationCategory = OperationCategory.GENERAL,
        success: bool = True,
    ) -> None:
        """Record an operation execution.

        Args:
            operation_name: Name of the operation
            execution_time: Execution time in seconds
            category: Operation category
            success: Whether operation succeeded
        """
        if not self.enabled:
            return

        with self._lock:
            # Get or create metrics
            if operation_name not in self._metrics:
                self._metrics[operation_name] = OperationMetrics()

            # Record execution
            self._metrics[operation_name].record(execution_time, success)

            # Check for slow operation
            if execution_time > self.alert_threshold:
                self._trigger_alert(operation_name, execution_time)

    def get_stats(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Statistics dictionary or None if operation not found
        """
        with self._lock:
            metrics = self._metrics.get(operation_name)
            if metrics is None:
                return None
            return metrics.get_stats()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations.

        Returns:
            Dictionary mapping operation names to statistics
        """
        with self._lock:
            return {name: metrics.get_stats() for name, metrics in self._metrics.items()}

    def get_hot_operations(
        self, min_count: int = 10, threshold: float = 1.0
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Get hot (frequently called/slow) operations.

        Args:
            min_count: Minimum call count to consider
            threshold: Minimum average time to consider (seconds)

        Returns:
            List of (operation_name, stats) tuples sorted by total time
        """
        with self._lock:
            hot_ops = []

            for name, metrics in self._metrics.items():
                stats = metrics.get_stats()
                if stats["count"] >= min_count and stats["avg_time"] >= threshold:
                    hot_ops.append((name, stats))

            # Sort by total time (descending)
            hot_ops.sort(key=lambda x: x[1]["total_time"], reverse=True)
            return hot_ops

    def reset(self, operation_name: Optional[str] = None) -> None:
        """Reset metrics for an operation or all operations.

        Args:
            operation_name: Name of operation to reset, or None for all
        """
        with self._lock:
            if operation_name:
                if operation_name in self._metrics:
                    del self._metrics[operation_name]
            else:
                self._metrics.clear()

    def add_alert_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add a callback for slow operation alerts.

        Args:
            callback: Function to call with (operation_name, execution_time)
        """
        self._alert_callbacks.append(callback)

    def _trigger_alert(self, operation_name: str, execution_time: float) -> None:
        """Trigger alert for slow operation.

        Args:
            operation_name: Name of the operation
            execution_time: Execution time in seconds
        """
        logger.warning(
            f"Slow operation detected: {operation_name} took {execution_time:.3f}s "
            f"(threshold: {self.alert_threshold:.3f}s)"
        )

        for callback in self._alert_callbacks:
            try:
                callback(operation_name, execution_time)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format.

        Args:
            format: Export format (json, csv)

        Returns:
            Serialized metrics

        Raises:
            ValueError: If format is not supported
        """
        stats = self.get_all_stats()

        if format == "json":
            import json

            return json.dumps(stats, indent=2)
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            if stats:
                writer = csv.DictWriter(output, fieldnames=next(iter(stats.values())).keys())
                writer.writeheader()
                for op_name, op_stats in stats.items():
                    writer.writerow({"operation": op_name, **op_stats})
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# Performance Context Manager
# =============================================================================


class _PerformanceContext:
    """Internal context manager for performance tracking."""

    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation_name: str,
        category: OperationCategory,
    ):
        """Initialize context.

        Args:
            monitor: Performance monitor instance
            operation_name: Operation name
            category: Operation category
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.category = category
        self.start_time: Optional[float] = None
        self.success = True

    def __enter__(self) -> "_PerformanceContext":
        """Enter context and start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and record timing."""
        if self.start_time is not None:
            execution_time = time.perf_counter() - self.start_time
            self.success = exc_type is None
            self.monitor.record_operation(
                self.operation_name, execution_time, self.category, self.success
            )


class _NullContext:
    """Null context manager when monitoring is disabled."""

    def __enter__(self) -> "_NullContext":
        """Enter context (no-op)."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context (no-op)."""
        pass


# =============================================================================
# Decorator for Timing Functions
# =============================================================================


def timed(
    monitor: Optional[PerformanceMonitor] = None,
    operation_name: Optional[str] = None,
    category: OperationCategory = OperationCategory.GENERAL,
) -> Any:
    """Decorator to time function executions.

    Args:
        monitor: Performance monitor (uses global if None)
        operation_name: Operation name (uses function name if None)
        category: Operation category

    Example:
        ```python
        @timed(category=OperationCategory.LLM_REQUEST)
        async def my_llm_function():
            # ... do work ...
            pass
        ```
    """
    if monitor is None:
        monitor = get_performance_monitor()

    def decorator(func: Any) -> Any:
        name = operation_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                monitor.record_operation(name, execution_time, category, True)
                return result
            except Exception:
                execution_time = time.perf_counter() - start_time
                monitor.record_operation(name, execution_time, category, False)
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                monitor.record_operation(name, execution_time, category, True)
                return result
            except Exception:
                execution_time = time.perf_counter() - start_time
                monitor.record_operation(name, execution_time, category, False)
                raise

        # Check if function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Import asyncio for async checking
import asyncio


# =============================================================================
# Global Performance Monitor Instance
# =============================================================================

_global_performance_monitor: Optional[PerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor(
    alert_threshold: float = 5.0,
    enabled: bool = True,
) -> PerformanceMonitor:
    """Get or create global performance monitor.

    Args:
        alert_threshold: Threshold for slow operation alerts
        enabled: Whether monitoring is enabled

    Returns:
        PerformanceMonitor instance
    """
    global _global_performance_monitor

    if _global_performance_monitor is None:
        with _monitor_lock:
            if _global_performance_monitor is None:
                _global_performance_monitor = PerformanceMonitor(
                    alert_threshold=alert_threshold,
                    enabled=enabled,
                )
                logger.info("Initialized global performance monitor")

    return _global_performance_monitor


def reset_performance_monitor() -> None:
    """Reset global performance monitor (mainly for testing)."""
    global _global_performance_monitor

    with _monitor_lock:
        _global_performance_monitor = None
