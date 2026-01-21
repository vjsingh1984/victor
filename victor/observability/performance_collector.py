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

"""Performance metrics collector for Victor AI monitoring dashboard.

This module provides centralized performance metrics collection from:
- Tool selection cache (hit rate, latency, memory)
- Cache operations (entries, evictions, utilization)
- Lazy loading (startup time, first-access overhead)
- Bootstrap process (phase timings)
- Provider pool (health, latency)
- Tool execution (duration, errors)

Design Patterns:
- Facade Pattern: Unified interface to all performance metrics
- Singleton Pattern: Single collector instance
- Observer Pattern: Subscribes to event bus for real-time updates

Example:
    from victor.observability import PerformanceMetricsCollector

    collector = PerformanceMetricsCollector.get_instance()

    # Get all metrics
    metrics = collector.get_all_metrics()

    # Get specific category
    cache_metrics = collector.get_cache_metrics()

    # Export for Prometheus
    prometheus_text = collector.export_prometheus()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.tools.caches.selection_cache import ToolSelectionCache

logger = logging.getLogger(__name__)


# =============================================================================
# Performance Metric Data Classes
# =============================================================================


@dataclass
class ToolSelectionMetrics:
    """Tool selection performance metrics."""

    # Cache hit rates by namespace
    query_hit_rate: float = 0.0
    context_hit_rate: float = 0.0
    rl_hit_rate: float = 0.0
    overall_hit_rate: float = 0.0

    # Latency metrics (milliseconds)
    avg_selection_latency_ms: float = 0.0
    p50_selection_latency_ms: float = 0.0
    p95_selection_latency_ms: float = 0.0
    p99_selection_latency_ms: float = 0.0

    # Latency saved by caching
    total_latency_saved_ms: float = 0.0
    avg_latency_saved_per_hit_ms: float = 0.0

    # Cache operations
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0

    # Cache entries
    query_entries: int = 0
    context_entries: int = 0
    rl_entries: int = 0
    total_entries: int = 0

    # Cache utilization (0.0 - 1.0)
    query_utilization: float = 0.0
    context_utilization: float = 0.0
    rl_utilization: float = 0.0
    overall_utilization: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hit_rates": {
                "query": self.query_hit_rate,
                "context": self.context_hit_rate,
                "rl": self.rl_hit_rate,
                "overall": self.overall_hit_rate,
            },
            "latency_ms": {
                "avg": self.avg_selection_latency_ms,
                "p50": self.p50_selection_latency_ms,
                "p95": self.p95_selection_latency_ms,
                "p99": self.p99_selection_latency_ms,
            },
            "latency_saved_ms": {
                "total": self.total_latency_saved_ms,
                "avg_per_hit": self.avg_latency_saved_per_hit_ms,
            },
            "operations": {
                "hits": self.total_hits,
                "misses": self.total_misses,
                "evictions": self.total_evictions,
            },
            "entries": {
                "query": self.query_entries,
                "context": self.context_entries,
                "rl": self.rl_entries,
                "total": self.total_entries,
            },
            "utilization": {
                "query": self.query_utilization,
                "context": self.context_utilization,
                "rl": self.rl_utilization,
                "overall": self.overall_utilization,
            },
        }


@dataclass
class CacheMetrics:
    """General cache performance metrics."""

    # Memory usage
    memory_usage_bytes: int = 0
    memory_usage_mb: float = 0.0

    # Entry counts
    total_entries: int = 0
    active_entries: int = 0

    # Operations
    total_lookups: int = 0
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0

    # Rates
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory": {
                "bytes": self.memory_usage_bytes,
                "mb": self.memory_usage_mb,
            },
            "entries": {
                "total": self.total_entries,
                "active": self.active_entries,
            },
            "operations": {
                "lookups": self.total_lookups,
                "hits": self.total_hits,
                "misses": self.total_misses,
                "evictions": self.total_evictions,
            },
            "rates": {
                "hit": self.hit_rate,
                "miss": self.miss_rate,
                "eviction": self.eviction_rate,
            },
        }


@dataclass
class BootstrapMetrics:
    """Bootstrap and startup performance metrics."""

    # Total startup time
    total_startup_time_ms: float = 0.0
    total_startup_time_seconds: float = 0.0

    # Phase timings
    container_bootstrap_ms: float = 0.0
    service_registration_ms: float = 0.0
    provider_initialization_ms: float = 0.0
    tool_loading_ms: float = 0.0
    cache_warming_ms: float = 0.0

    # Lazy loading metrics
    lazy_load_count: int = 0
    lazy_load_total_time_ms: float = 0.0
    avg_lazy_load_time_ms: float = 0.0
    first_access_overhead_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "startup_time": {
                "ms": self.total_startup_time_ms,
                "seconds": self.total_startup_time_seconds,
            },
            "phases": {
                "container_bootstrap_ms": self.container_bootstrap_ms,
                "service_registration_ms": self.service_registration_ms,
                "provider_initialization_ms": self.provider_initialization_ms,
                "tool_loading_ms": self.tool_loading_ms,
                "cache_warming_ms": self.cache_warming_ms,
            },
            "lazy_loading": {
                "load_count": self.lazy_load_count,
                "total_time_ms": self.lazy_load_total_time_ms,
                "avg_time_ms": self.avg_lazy_load_time_ms,
                "first_access_overhead_ms": self.first_access_overhead_ms,
            },
        }


@dataclass
class ProviderPoolMetrics:
    """Provider pool health and performance metrics."""

    # Pool status
    total_providers: int = 0
    active_providers: int = 0
    unhealthy_providers: int = 0

    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Error tracking
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    rate_limit_hit_rate: float = 0.0

    # Per-provider health
    provider_health: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pool": {
                "total": self.total_providers,
                "active": self.active_providers,
                "unhealthy": self.unhealthy_providers,
            },
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
            },
            "latency_ms": {
                "avg": self.avg_latency_ms,
                "p50": self.p50_latency_ms,
                "p95": self.p95_latency_ms,
                "p99": self.p99_latency_ms,
            },
            "errors": {
                "error_rate": self.error_rate,
                "timeout_rate": self.timeout_rate,
                "rate_limit_rate": self.rate_limit_hit_rate,
            },
            "provider_health": self.provider_health,
        }


@dataclass
class ToolExecutionMetrics:
    """Tool execution performance metrics."""

    # Execution counts
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0

    # Latency metrics (milliseconds)
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0

    # Error metrics
    error_rate: float = 0.0

    # Top tools by execution count
    top_tools: List[Dict[str, Any]] = field(default_factory=list)

    # Per-tool metrics
    tool_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "executions": {
                "total": self.total_executions,
                "successful": self.successful_executions,
                "failed": self.failed_executions,
            },
            "duration_ms": {
                "avg": self.avg_duration_ms,
                "p50": self.p50_duration_ms,
                "p95": self.p95_duration_ms,
                "p99": self.p99_duration_ms,
            },
            "errors": {
                "error_rate": self.error_rate,
            },
            "top_tools": self.top_tools,
            "tool_metrics": self.tool_metrics,
        }


@dataclass
class SystemMetrics:
    """System resource metrics."""

    # Memory
    memory_usage_bytes: int = 0
    memory_usage_mb: float = 0.0
    memory_usage_gb: float = 0.0
    memory_available_bytes: int = 0
    memory_available_mb: float = 0.0

    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0

    # Uptime
    uptime_seconds: float = 0.0
    uptime_minutes: float = 0.0
    uptime_hours: float = 0.0

    # Threads
    active_threads: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory": {
                "used_bytes": self.memory_usage_bytes,
                "used_mb": self.memory_usage_mb,
                "used_gb": self.memory_usage_gb,
                "available_bytes": self.memory_available_bytes,
                "available_mb": self.memory_available_mb,
            },
            "cpu": {
                "percent": self.cpu_percent,
                "count": self.cpu_count,
            },
            "uptime": {
                "seconds": self.uptime_seconds,
                "minutes": self.uptime_minutes,
                "hours": self.uptime_hours,
            },
            "threads": {
                "active": self.active_threads,
            },
        }


# =============================================================================
# Performance Metrics Collector
# =============================================================================


class PerformanceMetricsCollector:
    """Centralized performance metrics collector.

    Aggregates metrics from all components and provides unified access
    for monitoring dashboards and alerting.

    Thread-safe singleton implementation.

    Example:
        collector = PerformanceMetricsCollector.get_instance()

        # Get all metrics
        all_metrics = collector.get_all_metrics()

        # Get specific category
        cache_metrics = collector.get_tool_selection_metrics()

        # Export for Prometheus
        prometheus_text = collector.export_prometheus()
    """

    _instance: Optional["PerformanceMetricsCollector"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize collector (use get_instance() instead)."""
        self._start_time = time.time()
        self._lock = threading.RLock()

        # Metric storage
        self._tool_selection_metrics = ToolSelectionMetrics()
        self._cache_metrics = CacheMetrics()
        self._bootstrap_metrics = BootstrapMetrics()
        self._provider_pool_metrics = ProviderPoolMetrics()
        self._tool_execution_metrics = ToolExecutionMetrics()
        self._system_metrics = SystemMetrics()

        # Component references
        self._tool_selection_cache: Optional["ToolSelectionCache"] = None
        self._agent_metrics_collector: Optional[Any] = None

        logger.info("PerformanceMetricsCollector initialized")

    # =========================================================================
    # Singleton Access
    # =========================================================================

    @classmethod
    def get_instance(cls) -> "PerformanceMetricsCollector":
        """Get singleton collector instance.

        Returns:
            PerformanceMetricsCollector instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    # =========================================================================
    # Component Registration
    # =========================================================================

    def register_tool_selection_cache(self, cache: "ToolSelectionCache") -> None:
        """Register tool selection cache for metrics collection.

        Args:
            cache: ToolSelectionCache instance.
        """
        with self._lock:
            self._tool_selection_cache = cache
            logger.info("Registered tool selection cache for metrics collection")

    def register_agent_metrics_collector(self, collector: Any) -> None:
        """Register agent metrics collector.

        Args:
            collector: Agent MetricsCollector instance.
        """
        with self._lock:
            self._agent_metrics_collector = collector
            logger.info("Registered agent metrics collector")

    # =========================================================================
    # Metrics Collection
    # =========================================================================

    def collect_all_metrics(self) -> None:
        """Collect metrics from all registered components."""
        with self._lock:
            self._collect_tool_selection_metrics()
            self._collect_cache_metrics()
            self._collect_bootstrap_metrics()
            self._collect_provider_pool_metrics()
            self._collect_tool_execution_metrics()
            self._collect_system_metrics()

    def _collect_tool_selection_metrics(self) -> None:
        """Collect tool selection cache metrics."""
        if self._tool_selection_cache is None:
            return

        try:
            stats = self._tool_selection_cache.get_stats()

            # Hit rates
            self._tool_selection_metrics.query_hit_rate = (
                stats["namespaces"]["query"]["hit_rate"]
                if "query" in stats.get("namespaces", {})
                else 0.0
            )
            self._tool_selection_metrics.context_hit_rate = (
                stats["namespaces"]["context"]["hit_rate"]
                if "context" in stats.get("namespaces", {})
                else 0.0
            )
            self._tool_selection_metrics.rl_hit_rate = (
                stats["namespaces"]["rl"]["hit_rate"]
                if "rl" in stats.get("namespaces", {})
                else 0.0
            )
            self._tool_selection_metrics.overall_hit_rate = (
                stats["combined"]["hit_rate"] if "combined" in stats else 0.0
            )

            # Latency saved
            self._tool_selection_metrics.total_latency_saved_ms = stats["combined"].get(
                "total_latency_saved_ms", 0.0
            )
            self._tool_selection_metrics.avg_latency_saved_per_hit_ms = stats[
                "combined"
            ].get("avg_latency_per_hit_ms", 0.0)

            # Operations
            self._tool_selection_metrics.total_hits = stats["combined"].get("hits", 0)
            self._tool_selection_metrics.total_misses = stats["combined"].get(
                "misses", 0
            )
            self._tool_selection_metrics.total_evictions = stats["combined"].get(
                "evictions", 0
            )

            # Entries
            self._tool_selection_metrics.query_entries = (
                stats["namespaces"]["query"]["total_entries"]
                if "query" in stats.get("namespaces", {})
                else 0
            )
            self._tool_selection_metrics.context_entries = (
                stats["namespaces"]["context"]["total_entries"]
                if "context" in stats.get("namespaces", {})
                else 0
            )
            self._tool_selection_metrics.rl_entries = (
                stats["namespaces"]["rl"]["total_entries"]
                if "rl" in stats.get("namespaces", {})
                else 0
            )
            self._tool_selection_metrics.total_entries = stats["combined"].get(
                "total_entries", 0
            )

            # Utilization
            self._tool_selection_metrics.query_utilization = (
                stats["namespaces"]["query"].get("utilization", 0.0)
                if "query" in stats.get("namespaces", {})
                else 0.0
            )
            self._tool_selection_metrics.context_utilization = (
                stats["namespaces"]["context"].get("utilization", 0.0)
                if "context" in stats.get("namespaces", {})
                else 0.0
            )
            self._tool_selection_metrics.rl_utilization = (
                stats["namespaces"]["rl"].get("utilization", 0.0)
                if "rl" in stats.get("namespaces", {})
                else 0.0
            )
            self._tool_selection_metrics.overall_utilization = (
                sum(
                    [
                        self._tool_selection_metrics.query_utilization,
                        self._tool_selection_metrics.context_utilization,
                        self._tool_selection_metrics.rl_utilization,
                    ]
                )
                / 3
            )

        except Exception as e:
            logger.warning(f"Failed to collect tool selection metrics: {e}")

    def _collect_cache_metrics(self) -> None:
        """Collect general cache metrics."""
        # Aggregate from tool selection cache
        if self._tool_selection_cache is None:
            return

        try:
            stats = self._tool_selection_cache.get_stats()
            combined = stats.get("combined", {})

            self._cache_metrics.total_entries = combined.get("total_entries", 0)
            self._cache_metrics.active_entries = combined.get("total_entries", 0)
            self._cache_metrics.total_lookups = combined.get("hits", 0) + combined.get(
                "misses", 0
            )
            self._cache_metrics.total_hits = combined.get("hits", 0)
            self._cache_metrics.total_misses = combined.get("misses", 0)
            self._cache_metrics.total_evictions = combined.get("evictions", 0)

            # Calculate rates
            if self._cache_metrics.total_lookups > 0:
                self._cache_metrics.hit_rate = (
                    self._cache_metrics.total_hits / self._cache_metrics.total_lookups
                )
                self._cache_metrics.miss_rate = (
                    self._cache_metrics.total_misses / self._cache_metrics.total_lookups
                )

            # Estimate memory (roughly 1KB per entry)
            self._cache_metrics.memory_usage_bytes = (
                self._cache_metrics.total_entries * 1024
            )
            self._cache_metrics.memory_usage_mb = (
                self._cache_metrics.memory_usage_bytes / (1024 * 1024)
            )

        except Exception as e:
            logger.warning(f"Failed to collect cache metrics: {e}")

    def _collect_bootstrap_metrics(self) -> None:
        """Collect bootstrap and startup metrics."""
        # Update uptime
        uptime_seconds = time.time() - self._start_time
        self._bootstrap_metrics.total_startup_time_seconds = uptime_seconds
        self._bootstrap_metrics.total_startup_time_ms = uptime_seconds * 1000

        # TODO: Track phase timings during actual bootstrap
        # These would be set during the bootstrap process

    def _collect_provider_pool_metrics(self) -> None:
        """Collect provider pool metrics."""
        # TODO: Collect from provider pool when implemented
        pass

    def _collect_tool_execution_metrics(self) -> None:
        """Collect tool execution metrics."""
        if self._agent_metrics_collector is None:
            return

        try:
            tool_stats = self._agent_metrics_collector.get_tool_usage_stats()

            # Get execution counts
            selection_stats = tool_stats.get("selection_stats", {})
            self._tool_execution_metrics.total_executions = selection_stats.get(
                "total_tools_executed", 0
            )

            # Calculate success/failure from tool stats
            total_calls = 0
            failed_calls = 0
            tool_durations = []

            for tool_name, stats in tool_stats.get("tool_stats", {}).items():
                calls = stats.get("total_calls", 0)
                total_calls += calls
                failed_calls += stats.get("failed_calls", 0)

                # Track durations
                if "total_time_ms" in stats:
                    avg_time = stats.get("avg_time_ms", 0.0)
                    tool_durations.append(avg_time)

            self._tool_execution_metrics.total_executions = total_calls
            self._tool_execution_metrics.successful_executions = (
                total_calls - failed_calls
            )
            self._tool_execution_metrics.failed_executions = failed_calls

            # Calculate error rate
            if total_calls > 0:
                self._tool_execution_metrics.error_rate = failed_calls / total_calls

            # Calculate average duration
            if tool_durations:
                self._tool_execution_metrics.avg_duration_ms = sum(tool_durations) / len(
                    tool_durations
                )
                self._tool_execution_metrics.p50_duration_ms = sorted(tool_durations)[
                    len(tool_durations) // 2
                ]
                self._tool_execution_metrics.p95_duration_ms = sorted(tool_durations)[
                    int(len(tool_durations) * 0.95)
                ] if len(tool_durations) > 1 else tool_durations[0]
                self._tool_execution_metrics.p99_duration_ms = sorted(tool_durations)[
                    int(len(tool_durations) * 0.99)
                ] if len(tool_durations) > 1 else tool_durations[0]

            # Get top tools
            self._tool_execution_metrics.top_tools = tool_stats.get(
                "top_tools_by_usage", []
            )[:10]

            # Store per-tool metrics
            self._tool_execution_metrics.tool_metrics = tool_stats.get("tool_stats", {})

        except Exception as e:
            logger.warning(f"Failed to collect tool execution metrics: {e}")

    def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        import psutil

        try:
            # Memory
            process = psutil.Process()
            memory_info = process.memory_info()
            self._system_metrics.memory_usage_bytes = memory_info.rss
            self._system_metrics.memory_usage_mb = (
                self._system_metrics.memory_usage_bytes / (1024 * 1024)
            )
            self._system_metrics.memory_usage_gb = (
                self._system_metrics.memory_usage_mb / 1024
            )

            virtual_mem = psutil.virtual_memory()
            self._system_metrics.memory_available_bytes = virtual_mem.available
            self._system_metrics.memory_available_mb = (
                self._system_metrics.memory_available_bytes / (1024 * 1024)
            )

            # CPU
            self._system_metrics.cpu_percent = process.cpu_percent()
            self._system_metrics.cpu_count = psutil.cpu_count()

            # Uptime
            uptime_seconds = time.time() - self._start_time
            self._system_metrics.uptime_seconds = uptime_seconds
            self._system_metrics.uptime_minutes = uptime_seconds / 60
            self._system_metrics.uptime_hours = uptime_seconds / 3600

            # Threads
            self._system_metrics.active_threads = process.num_threads()

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    # =========================================================================
    # Metrics Access
    # =========================================================================

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics.

        Returns:
            Dictionary containing all metrics.
        """
        self.collect_all_metrics()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_selection": self._tool_selection_metrics.to_dict(),
            "cache": self._cache_metrics.to_dict(),
            "bootstrap": self._bootstrap_metrics.to_dict(),
            "provider_pool": self._provider_pool_metrics.to_dict(),
            "tool_execution": self._tool_execution_metrics.to_dict(),
            "system": self._system_metrics.to_dict(),
        }

    def get_tool_selection_metrics(self) -> ToolSelectionMetrics:
        """Get tool selection metrics.

        Returns:
            ToolSelectionMetrics instance.
        """
        self._collect_tool_selection_metrics()
        return self._tool_selection_metrics

    def get_cache_metrics(self) -> CacheMetrics:
        """Get cache metrics.

        Returns:
            CacheMetrics instance.
        """
        self._collect_cache_metrics()
        return self._cache_metrics

    def get_bootstrap_metrics(self) -> BootstrapMetrics:
        """Get bootstrap metrics.

        Returns:
            BootstrapMetrics instance.
        """
        self._collect_bootstrap_metrics()
        return self._bootstrap_metrics

    def get_provider_pool_metrics(self) -> ProviderPoolMetrics:
        """Get provider pool metrics.

        Returns:
            ProviderPoolMetrics instance.
        """
        self._collect_provider_pool_metrics()
        return self._provider_pool_metrics

    def get_tool_execution_metrics(self) -> ToolExecutionMetrics:
        """Get tool execution metrics.

        Returns:
            ToolExecutionMetrics instance.
        """
        self._collect_tool_execution_metrics()
        return self._tool_execution_metrics

    def get_system_metrics(self) -> SystemMetrics:
        """Get system metrics.

        Returns:
            SystemMetrics instance.
        """
        self._collect_system_metrics()
        return self._system_metrics

    # =========================================================================
    # Prometheus Export
    # =========================================================================

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus format string.
        """
        self.collect_all_metrics()

        lines = []

        # Scrape metadata
        lines.append(f"# Victor AI Performance Metrics")
        lines.append(f"victor_performance_scrape_timestamp {time.time():.3f}")
        lines.append("")

        # Tool selection metrics
        ts = self._tool_selection_metrics
        lines.append(f"# HELP victor_cache_hit_rate Cache hit rate by namespace")
        lines.append(f"# TYPE victor_cache_hit_rate gauge")
        lines.append(f'victor_cache_hit_rate{{namespace="query"}} {ts.query_hit_rate:.4f}')
        lines.append(
            f'victor_cache_hit_rate{{namespace="context"}} {ts.context_hit_rate:.4f}'
        )
        lines.append(f'victor_cache_hit_rate{{namespace="rl"}} {ts.rl_hit_rate:.4f}')
        lines.append(f'victor_cache_hit_rate{{namespace="overall"}} {ts.overall_hit_rate:.4f}')

        lines.append("")
        lines.append(f"# HELP victor_cache_entries Cache entry count by namespace")
        lines.append(f"# TYPE victor_cache_entries gauge")
        lines.append(f'victor_cache_entries{{namespace="query"}} {ts.query_entries}')
        lines.append(f'victor_cache_entries{{namespace="context"}} {ts.context_entries}')
        lines.append(f'victor_cache_entries{{namespace="rl"}} {ts.rl_entries}')
        lines.append(f'victor_cache_entries{{namespace="total"}} {ts.total_entries}')

        lines.append("")
        lines.append(f"# HELP victor_cache_operations_total Cache operations")
        lines.append(f"# TYPE victor_cache_operations_total counter")
        lines.append(f"victor_cache_operations_total{{operation=\"hits\"}} {ts.total_hits}")
        lines.append(f"victor_cache_operations_total{{operation=\"misses\"}} {ts.total_misses}")
        lines.append(
            f"victor_cache_operations_total{{operation=\"evictions\"}} {ts.total_evictions}"
        )

        lines.append("")
        lines.append(f"# HELP victor_cache_utilization Cache utilization (0-1)")
        lines.append(f"# TYPE victor_cache_utilization gauge")
        lines.append(
            f'victor_cache_utilization{{namespace="query"}} {ts.query_utilization:.4f}'
        )
        lines.append(
            f'victor_cache_utilization{{namespace="context"}} {ts.context_utilization:.4f}'
        )
        lines.append(f'victor_cache_utilization{{namespace="rl"}} {ts.rl_utilization:.4f}')
        lines.append(
            f'victor_cache_utilization{{namespace="overall"}} {ts.overall_utilization:.4f}'
        )

        # Cache metrics
        cm = self._cache_metrics
        lines.append("")
        lines.append(f"# HELP victor_cache_memory_bytes Cache memory usage")
        lines.append(f"# TYPE victor_cache_memory_bytes gauge")
        lines.append(f"victor_cache_memory_bytes {cm.memory_usage_bytes}")

        lines.append("")
        lines.append(f"# HELP victor_cache_memory_mb Cache memory usage in MB")
        lines.append(f"# TYPE victor_cache_memory_mb gauge")
        lines.append(f"victor_cache_memory_mb {cm.memory_usage_mb:.2f}")

        # Tool execution metrics
        te = self._tool_execution_metrics
        lines.append("")
        lines.append(f"# HELP victor_tool_executions_total Tool execution count")
        lines.append(f"# TYPE victor_tool_executions_total counter")
        lines.append(f"victor_tool_executions_total{{status=\"success\"}} {te.successful_executions}")
        lines.append(f"victor_tool_executions_total{{status=\"failure\"}} {te.failed_executions}")

        lines.append("")
        lines.append(f"# HELP victor_tool_duration_ms Tool execution duration")
        lines.append(f"# TYPE victor_tool_duration_ms gauge")
        lines.append(f"victor_tool_duration_ms{{quantile=\"avg\"}} {te.avg_duration_ms:.2f}")
        lines.append(f"victor_tool_duration_ms{{quantile=\"p50\"}} {te.p50_duration_ms:.2f}")
        lines.append(f"victor_tool_duration_ms{{quantile=\"p95\"}} {te.p95_duration_ms:.2f}")
        lines.append(f"victor_tool_duration_ms{{quantile=\"p99\"}} {te.p99_duration_ms:.2f}")

        lines.append("")
        lines.append(f"# HELP victor_tool_error_rate Tool execution error rate")
        lines.append(f"# TYPE victor_tool_error_rate gauge")
        lines.append(f"victor_tool_error_rate {te.error_rate:.4f}")

        # System metrics
        sm = self._system_metrics
        lines.append("")
        lines.append(f"# HELP victor_system_memory_bytes System memory usage")
        lines.append(f"# TYPE victor_system_memory_bytes gauge")
        lines.append(f"victor_system_memory_bytes {sm.memory_usage_bytes}")

        lines.append("")
        lines.append(f"# HELP victor_system_memory_mb System memory usage in MB")
        lines.append(f"# TYPE victor_system_memory_mb gauge")
        lines.append(f"victor_system_memory_mb {sm.memory_usage_mb:.2f}")

        lines.append("")
        lines.append(f"# HELP victor_system_cpu_percent CPU usage percent")
        lines.append(f"# TYPE victor_system_cpu_percent gauge")
        lines.append(f"victor_system_cpu_percent {sm.cpu_percent:.2f}")

        lines.append("")
        lines.append(f"# HELP victor_system_uptime_seconds System uptime")
        lines.append(f"# TYPE victor_system_uptime_seconds gauge")
        lines.append(f"victor_system_uptime_seconds {sm.uptime_seconds:.2f}")

        lines.append("")
        lines.append(f"# HELP victor_system_threads Active thread count")
        lines.append(f"# TYPE victor_system_threads gauge")
        lines.append(f"victor_system_threads {sm.active_threads}")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_performance_collector() -> PerformanceMetricsCollector:
    """Get performance metrics collector singleton.

    Returns:
        PerformanceMetricsCollector instance.
    """
    return PerformanceMetricsCollector.get_instance()


__all__ = [
    # Data classes
    "ToolSelectionMetrics",
    "CacheMetrics",
    "BootstrapMetrics",
    "ProviderPoolMetrics",
    "ToolExecutionMetrics",
    "SystemMetrics",
    # Collector
    "PerformanceMetricsCollector",
    "get_performance_collector",
]
