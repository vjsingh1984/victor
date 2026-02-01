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

"""Coordinator-specific metrics collection for performance monitoring.

This module provides the CoordinatorMetricsCollector which collects and aggregates
performance metrics from all coordinators in the refactored orchestrator architecture.

Key Features:
- Track execution times per coordinator
- Monitor memory usage by coordinator
- Cache hit rate tracking (if caching enabled)
- Analytics event counts
- Error rates by coordinator
- Export metrics in JSON and Prometheus formats

Example:
    from victor.observability.coordinator_metrics import CoordinatorMetricsCollector

    collector = CoordinatorMetricsCollector()

    # Track coordinator execution
    with collector.track_coordinator("ChatCoordinator"):
        # Do work
        pass

    # Export metrics
    json_data = collector.export_json()
    prometheus_data = collector.export_prometheus()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
import psutil  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CoordinatorExecution:
    """Record of a single coordinator execution."""

    coordinator_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coordinator_name": self.coordinator_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class CoordinatorSnapshot:
    """Snapshot of coordinator metrics at a point in time."""

    timestamp: str
    coordinator_name: str
    memory_bytes: int
    cpu_percent: float
    execution_count: int
    total_duration_ms: float
    error_count: int
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "coordinator_name": self.coordinator_name,
            "memory_bytes": self.memory_bytes,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "cpu_percent": self.cpu_percent,
            "execution_count": self.execution_count,
            "total_duration_ms": self.total_duration_ms,
            "error_count": self.error_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0
            ),
        }


# =============================================================================
# Metrics Collector
# =============================================================================


class CoordinatorMetricsCollector:
    """Collects and aggregates metrics from all coordinators.

    This class provides:
    - Execution time tracking per coordinator
    - Memory usage monitoring
    - Error rate tracking
    - Cache performance metrics
    - Analytics event aggregation
    - JSON and Prometheus export formats

    Thread-safe for concurrent access.

    Example:
        collector = CoordinatorMetricsCollector()

        # Track execution
        with collector.track_coordinator("ChatCoordinator"):
            await coordinator.process()

        # Get metrics
        snapshot = collector.get_snapshot("ChatCoordinator")
        all_snapshots = collector.get_all_snapshots()
    """

    def __init__(self, max_history: int = 10000) -> None:
        """Initialize metrics collector.

        Args:
            max_history: Maximum number of execution records to keep per coordinator
        """
        self._max_history = max_history
        self._lock = threading.RLock()

        # Execution history
        self._executions: dict[str, list[CoordinatorExecution]] = defaultdict(list)

        # Current metrics (per coordinator)
        self._execution_counts: dict[str, int] = defaultdict(int)
        self._total_durations: dict[str, float] = defaultdict(float)
        self._error_counts: dict[str, int] = defaultdict(int)

        # Cache metrics
        self._cache_hits: dict[str, int] = defaultdict(int)
        self._cache_misses: dict[str, int] = defaultdict(int)

        # Analytics events
        self._analytics_events: dict[str, int] = defaultdict(int)

        # Process for memory tracking
        self._process = psutil.Process()

        # Start time
        self._start_time = time.time()

    # ========================================================================
    # Execution Tracking
    # ========================================================================

    @contextmanager
    def track_coordinator(self, coordinator_name: str, metadata: Optional[dict[str, Any]] = None):
        """Context manager for tracking coordinator execution.

        Args:
            coordinator_name: Name of the coordinator
            metadata: Optional metadata to attach to the execution record

        Yields:
            None

        Example:
            with collector.track_coordinator("ChatCoordinator"):
                await coordinator.chat()
        """
        start_time = time.perf_counter()
        success = True
        error_message = None

        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            self._record_execution(
                coordinator_name=coordinator_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                metadata=metadata or {},
            )

    def track_execution(
        self,
        coordinator_name: str,
        duration_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a coordinator execution.

        Args:
            coordinator_name: Name of the coordinator
            duration_ms: Execution duration in milliseconds
            success: Whether execution succeeded
            error_message: Optional error message if failed
            metadata: Optional metadata
        """
        end_time = time.time()
        start_time = end_time - (duration_ms / 1000)

        self._record_execution(
            coordinator_name=coordinator_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            metadata=metadata or {},
        )

    def _record_execution(
        self,
        coordinator_name: str,
        start_time: float,
        end_time: float,
        duration_ms: float,
        success: bool,
        error_message: Optional[str],
        metadata: dict[str, Any],
    ) -> None:
        """Record an execution (internal method)."""
        with self._lock:
            execution = CoordinatorExecution(
                coordinator_name=coordinator_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                metadata=metadata,
            )

            # Add to history
            self._executions[coordinator_name].append(execution)

            # Trim history if needed
            if len(self._executions[coordinator_name]) > self._max_history:
                self._executions[coordinator_name] = self._executions[coordinator_name][
                    -self._max_history :
                ]

            # Update aggregates
            self._execution_counts[coordinator_name] += 1
            self._total_durations[coordinator_name] += duration_ms

            if not success:
                self._error_counts[coordinator_name] += 1

    # ========================================================================
    # Cache Metrics
    # ========================================================================

    def record_cache_hit(self, coordinator_name: str) -> None:
        """Record a cache hit for a coordinator.

        Args:
            coordinator_name: Name of the coordinator
        """
        with self._lock:
            self._cache_hits[coordinator_name] += 1

    def record_cache_miss(self, coordinator_name: str) -> None:
        """Record a cache miss for a coordinator.

        Args:
            coordinator_name: Name of the coordinator
        """
        with self._lock:
            self._cache_misses[coordinator_name] += 1

    # ========================================================================
    # Analytics Events
    # ========================================================================

    def record_analytics_event(self, event_name: str, count: int = 1) -> None:
        """Record an analytics event.

        Args:
            event_name: Name of the event
            count: Number of events (default 1)
        """
        with self._lock:
            self._analytics_events[event_name] += count

    # ========================================================================
    # Snapshots
    # ========================================================================

    def get_snapshot(self, coordinator_name: str) -> Optional[CoordinatorSnapshot]:
        """Get current metrics snapshot for a coordinator.

        Args:
            coordinator_name: Name of the coordinator

        Returns:
            CoordinatorSnapshot or None if coordinator not found
        """
        with self._lock:
            if coordinator_name not in self._execution_counts:
                return None

            try:
                memory_info = self._process.memory_info()
                cpu_percent = self._process.cpu_percent()
            except Exception:
                # Fallback if psutil fails
                memory_info = type("MemoryInfo", (), {"rss": 0})()
                cpu_percent = 0.0

            return CoordinatorSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                coordinator_name=coordinator_name,
                memory_bytes=memory_info.rss,
                cpu_percent=cpu_percent,
                execution_count=self._execution_counts[coordinator_name],
                total_duration_ms=self._total_durations[coordinator_name],
                error_count=self._error_counts[coordinator_name],
                cache_hits=self._cache_hits[coordinator_name],
                cache_misses=self._cache_misses[coordinator_name],
            )

    def get_all_snapshots(self) -> list[CoordinatorSnapshot]:
        """Get snapshots for all coordinators.

        Returns:
            List of CoordinatorSnapshot objects
        """
        with self._lock:
            coordinator_names = set(self._execution_counts.keys())
            coordinator_names.update(self._cache_hits.keys())
            coordinator_names.update(self._cache_misses.keys())

            snapshots = []
            for name in coordinator_names:
                snapshot = self.get_snapshot(name)
                if snapshot:
                    snapshots.append(snapshot)

            return snapshots

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_coordinator_stats(self, coordinator_name: str) -> dict[str, Any]:
        """Get statistics for a specific coordinator.

        Args:
            coordinator_name: Name of the coordinator

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            executions = self._executions.get(coordinator_name, [])

            if not executions:
                return {
                    "coordinator_name": coordinator_name,
                    "execution_count": 0,
                    "avg_duration_ms": 0,
                    "min_duration_ms": 0,
                    "max_duration_ms": 0,
                    "p50_duration_ms": 0,
                    "p95_duration_ms": 0,
                    "p99_duration_ms": 0,
                    "error_count": 0,
                    "error_rate": 0,
                    "cache_hit_rate": 0,
                }

            durations = [e.duration_ms for e in executions]
            durations.sort()

            total_errors = self._error_counts.get(coordinator_name, 0)
            cache_hits = self._cache_hits.get(coordinator_name, 0)
            cache_misses = self._cache_misses.get(coordinator_name, 0)
            total_cache_ops = cache_hits + cache_misses

            return {
                "coordinator_name": coordinator_name,
                "execution_count": len(executions),
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "p50_duration_ms": durations[len(durations) // 2],
                "p95_duration_ms": (
                    durations[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0]
                ),
                "p99_duration_ms": (
                    durations[int(len(durations) * 0.99)] if len(durations) > 1 else durations[0]
                ),
                "error_count": total_errors,
                "error_rate": total_errors / len(executions),
                "cache_hit_rate": cache_hits / total_cache_ops if total_cache_ops > 0 else 0,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            }

    def get_overall_stats(self) -> dict[str, Any]:
        """Get overall statistics across all coordinators.

        Returns:
            Dictionary with overall statistics
        """
        with self._lock:
            all_executions = []
            for executions in self._executions.values():
                all_executions.extend(executions)

            if not all_executions:
                return {
                    "total_executions": 0,
                    "total_coordinators": 0,
                    "total_errors": 0,
                    "overall_error_rate": 0,
                    "uptime_seconds": time.time() - self._start_time,
                }

            total_errors = sum(self._error_counts.values())

            return {
                "total_executions": len(all_executions),
                "total_coordinators": len(self._execution_counts),
                "total_errors": total_errors,
                "overall_error_rate": total_errors / len(all_executions),
                "uptime_seconds": time.time() - self._start_time,
                "analytics_events": dict(self._analytics_events),
            }

    # ========================================================================
    # Export Formats
    # ========================================================================

    def export_json(self, include_history: bool = False) -> str:
        """Export metrics as JSON.

        Args:
            include_history: Whether to include full execution history

        Returns:
            JSON string
        """
        snapshots = [s.to_dict() for s in self.get_all_snapshots()]
        overall_stats = self.get_overall_stats()

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_stats": overall_stats,
            "coordinators": snapshots,
        }

        if include_history:
            coordinator_stats = {}
            for name in self._execution_counts.keys():
                coordinator_stats[name] = self.get_coordinator_stats(name)
                # Include recent execution history
                executions = self._executions.get(name, [])
                coordinator_stats[name]["recent_executions"] = [
                    e.to_dict() for e in executions[-100:]  # Last 100
                ]

            data["coordinator_stats"] = coordinator_stats

        return json.dumps(data, indent=2)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus format string
        """
        lines = []

        for snapshot in self.get_all_snapshots():
            snapshot.coordinator_name.replace(" ", "_").replace("-", "_").lower()

            # Execution count
            lines.append(
                f'victor_coordinator_executions_total{{coordinator="{snapshot.coordinator_name}"}} {snapshot.execution_count}'
            )

            # Total duration
            lines.append(
                f'victor_coordinator_duration_ms_total{{coordinator="{snapshot.coordinator_name}"}} {snapshot.total_duration_ms:.2f}'
            )

            # Error count
            lines.append(
                f'victor_coordinator_errors_total{{coordinator="{snapshot.coordinator_name}"}} {snapshot.error_count}'
            )

            # Memory
            lines.append(
                f'victor_coordinator_memory_bytes{{coordinator="{snapshot.coordinator_name}"}} {snapshot.memory_bytes}'
            )

            # Cache hit rate
            lines.append(
                f'victor_coordinator_cache_hit_rate{{coordinator="{snapshot.coordinator_name}"}} {snapshot.to_dict()["cache_hit_rate"]:.4f}'
            )

            # Cache hits
            lines.append(
                f'victor_coordinator_cache_hits_total{{coordinator="{snapshot.coordinator_name}"}} {snapshot.cache_hits}'
            )

            # Cache misses
            lines.append(
                f'victor_coordinator_cache_misses_total{{coordinator="{snapshot.coordinator_name}"}} {snapshot.cache_misses}'
            )

        # Overall stats
        overall = self.get_overall_stats()
        lines.append(f"victor_coordinator_uptime_seconds {overall['uptime_seconds']:.2f}")
        lines.append(f"victor_coordinator_total_executions {overall['total_executions']}")

        # Analytics events
        for event_name, count in overall.get("analytics_events", {}).items():
            lines.append(f'victor_analytics_events_total{{event="{event_name}"}} {count}')

        return "\n".join(lines)

    # ========================================================================
    # Reset and Clear
    # ========================================================================

    def reset_coordinator(self, coordinator_name: str) -> None:
        """Reset metrics for a specific coordinator.

        Args:
            coordinator_name: Name of the coordinator
        """
        with self._lock:
            self._executions.pop(coordinator_name, None)
            self._execution_counts.pop(coordinator_name, None)
            self._total_durations.pop(coordinator_name, None)
            self._error_counts.pop(coordinator_name, None)
            self._cache_hits.pop(coordinator_name, None)
            self._cache_misses.pop(coordinator_name, None)

    def reset_all(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._executions.clear()
            self._execution_counts.clear()
            self._total_durations.clear()
            self._error_counts.clear()
            self._cache_hits.clear()
            self._cache_misses.clear()
            self._analytics_events.clear()
            self._start_time = time.time()


# =============================================================================
# Decorators
# =============================================================================


def track_coordinator_metrics(
    collector: CoordinatorMetricsCollector,
    coordinator_name: Optional[str] = None,
):
    """Decorator to automatically track coordinator method execution.

    Args:
        collector: Metrics collector instance
        coordinator_name: Optional coordinator name (uses class name if not provided)

    Example:
        collector = CoordinatorMetricsCollector()

        class MyCoordinator:
            @track_coordinator_metrics(collector, "MyCoordinator")
            def process(self):
                # Do work
                pass
    """

    def decorator(func: Callable[..., Any]) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = coordinator_name or func.__qualname__

            with collector.track_coordinator(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Singleton Instance
# =============================================================================

_default_collector: Optional[CoordinatorMetricsCollector] = None
_default_lock = threading.Lock()


def get_coordinator_metrics_collector() -> CoordinatorMetricsCollector:
    """Get the default coordinator metrics collector (singleton).

    Returns:
        CoordinatorMetricsCollector instance
    """
    global _default_collector

    if _default_collector is None:
        with _default_lock:
            if _default_collector is None:
                _default_collector = CoordinatorMetricsCollector()

    return _default_collector
