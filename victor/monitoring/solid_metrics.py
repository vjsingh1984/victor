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

"""SOLID Remediation Metrics Collection Framework.

This module provides comprehensive metrics collection for monitoring the
SOLID remediation changes in production, including startup time, cache
performance, memory usage, and feature flag status.

Design Goals:
- Non-invasive metrics collection
- Minimal performance overhead (<1%)
- Thread-safe metrics aggregation
- Export to multiple formats (JSON, Prometheus, logging)
- Configurable sampling rates

Usage:
    from victor.monitoring.solid_metrics import (
        SolidMetricsCollector,
        get_metrics_collector,
    )

    # Get singleton collector
    collector = get_metrics_collector()

    # Record startup time
    collector.record_startup_time(1.2)  # 1.2 seconds

    # Record cache hit
    collector.record_cache_hit("tool_selection", 0.5)

    # Export metrics
    metrics = collector.export_metrics()
"""

import os
import time
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics Data Classes
# =============================================================================


@dataclass
class StartupMetrics:
    """Metrics for application startup performance."""

    import_time: float = 0.0
    """Time to import all verticals (seconds)."""

    init_time: float = 0.0
    """Time to initialize components (seconds)."""

    total_time: float = 0.0
    """Total startup time (seconds)."""

    verticals_loaded: int = 0
    """Number of verticals loaded."""

    lazy_initializations: int = 0
    """Number of lazy initializations triggered."""


@dataclass
class CacheMetrics:
    """Metrics for cache performance."""

    cache_name: str = ""
    """Name of the cache (e.g., 'tool_selection')."""

    hits: int = 0
    """Number of cache hits."""

    misses: int = 0
    """Number of cache misses."""

    evictions: int = 0
    """Number of cache evictions."""

    size: int = 0
    """Current cache size."""

    max_size: Optional[int] = None
    """Maximum cache size (None = unlimited)."""

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as percentage (0-100), or 0 if no accesses.
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100


@dataclass
class FeatureFlagMetrics:
    """Metrics for feature flag status."""

    flag_name: str = ""
    """Name of the feature flag."""

    enabled: bool = False
    """Whether the flag is enabled."""

    value: str = ""
    """String value of the flag."""

    source: str = "environment"
    """Source of the flag value (environment, config, default)."""


@dataclass
class MemoryMetrics:
    """Metrics for memory usage."""

    rss_mb: float = 0.0
    """Resident Set Size in MB."""

    vms_mb: float = 0.0
    """Virtual Memory Size in MB."""

    heap_mb: float = 0.0
    """Heap size in MB."""

    cache_overhead_mb: float = 0.0
    """Estimated cache overhead in MB."""


@dataclass
class ErrorMetrics:
    """Metrics for errors and exceptions."""

    total_errors: int = 0
    """Total number of errors."""

    errors_by_type: dict[str, int] = field(default_factory=dict)
    """Error count by type."""

    last_error: Optional[str] = None
    """Last error message."""

    last_error_time: Optional[datetime] = None
    """Last error timestamp."""


# =============================================================================
# Metrics Collector
# =============================================================================


class SolidMetricsCollector:
    """Collects and aggregates SOLID remediation metrics.

    This class provides thread-safe metrics collection for monitoring
    the SOLID remediation changes in production.

    Thread Safety:
        All methods are thread-safe and can be called from multiple threads.

    Example:
        collector = SolidMetricsCollector()

        # Record startup metrics
        collector.record_startup_time(1.5)

        # Record cache hit
        collector.record_cache_hit("tool_selection", 0.1)

        # Export metrics
        metrics = collector.export_metrics()
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._lock = threading.RLock()
        self._startup: StartupMetrics = StartupMetrics()
        self._caches: dict[str, CacheMetrics] = {}
        self._feature_flags: dict[str, FeatureFlagMetrics] = {}
        self._errors: ErrorMetrics = ErrorMetrics()
        self._start_time: float = time.time()
        self._collection_count: int = 0

    def record_startup_time(
        self,
        total_time: float,
        import_time: Optional[float] = None,
        init_time: Optional[float] = None,
        verticals_loaded: Optional[int] = None,
    ) -> None:
        """Record application startup time.

        Args:
            total_time: Total startup time in seconds
            import_time: Time to import verticals (optional)
            init_time: Time to initialize components (optional)
            verticals_loaded: Number of verticals loaded (optional)
        """
        with self._lock:
            self._startup.total_time = total_time
            if import_time is not None:
                self._startup.import_time = import_time
            if init_time is not None:
                self._startup.init_time = init_time
            if verticals_loaded is not None:
                self._startup.verticals_loaded = verticals_loaded

    def record_cache_hit(
        self,
        cache_name: str,
        access_time: float,
    ) -> None:
        """Record a cache hit.

        Args:
            cache_name: Name of the cache
            access_time: Time taken to access cache (seconds)
        """
        with self._lock:
            if cache_name not in self._caches:
                self._caches[cache_name] = CacheMetrics(cache_name=cache_name)
            self._caches[cache_name].hits += 1
            self._collection_count += 1

    def record_cache_miss(
        self,
        cache_name: str,
        access_time: float,
    ) -> None:
        """Record a cache miss.

        Args:
            cache_name: Name of the cache
            access_time: Time taken to handle miss (seconds)
        """
        with self._lock:
            if cache_name not in self._caches:
                self._caches[cache_name] = CacheMetrics(cache_name=cache_name)
            self._caches[cache_name].misses += 1
            self._collection_count += 1

    def record_cache_eviction(self, cache_name: str) -> None:
        """Record a cache eviction.

        Args:
            cache_name: Name of the cache
        """
        with self._lock:
            if cache_name not in self._caches:
                self._caches[cache_name] = CacheMetrics(cache_name=cache_name)
            self._caches[cache_name].evictions += 1

    def record_cache_size(
        self,
        cache_name: str,
        size: int,
        max_size: Optional[int] = None,
    ) -> None:
        """Record current cache size.

        Args:
            cache_name: Name of the cache
            size: Current cache size
            max_size: Maximum cache size (optional)
        """
        with self._lock:
            if cache_name not in self._caches:
                self._caches[cache_name] = CacheMetrics(cache_name=cache_name)
            self._caches[cache_name].size = size
            if max_size is not None:
                self._caches[cache_name].max_size = max_size

    def record_feature_flag(
        self,
        flag_name: str,
        enabled: bool,
        value: str = "",
        source: str = "environment",
    ) -> None:
        """Record feature flag status.

        Args:
            flag_name: Name of the feature flag
            enabled: Whether the flag is enabled
            value: String value of the flag
            source: Source of the flag value
        """
        with self._lock:
            self._feature_flags[flag_name] = FeatureFlagMetrics(
                flag_name=flag_name,
                enabled=enabled,
                value=value,
                source=source,
            )

    def record_error(
        self,
        error_type: str,
        error_message: str,
    ) -> None:
        """Record an error.

        Args:
            error_type: Type of error (e.g., "ImportError")
            error_message: Error message
        """
        with self._lock:
            self._errors.total_errors += 1
            self._errors.errors_by_type[error_type] = (
                self._errors.errors_by_type.get(error_type, 0) + 1
            )
            self._errors.last_error = error_message
            self._errors.last_error_time = datetime.now()

    def get_uptime(self) -> float:
        """Get collector uptime.

        Returns:
            Uptime in seconds.
        """
        return time.time() - self._start_time

    def export_metrics(self) -> dict[str, Any]:
        """Export all collected metrics.

        Returns:
            Dictionary of all metrics.
        """
        with self._lock:
            return {
                "startup": {
                    "total_time_seconds": self._startup.total_time,
                    "import_time_seconds": self._startup.import_time,
                    "init_time_seconds": self._startup.init_time,
                    "verticals_loaded": self._startup.verticals_loaded,
                    "lazy_initializations": self._startup.lazy_initializations,
                },
                "caches": {
                    name: {
                        "hits": metrics.hits,
                        "misses": metrics.misses,
                        "evictions": metrics.evictions,
                        "hit_rate": metrics.get_hit_rate(),
                        "size": metrics.size,
                        "max_size": metrics.max_size,
                    }
                    for name, metrics in self._caches.items()
                },
                "feature_flags": {
                    name: {
                        "enabled": flag.enabled,
                        "value": flag.value,
                        "source": flag.source,
                    }
                    for name, flag in self._feature_flags.items()
                },
                "errors": {
                    "total_errors": self._errors.total_errors,
                    "errors_by_type": self._errors.errors_by_type,
                    "last_error": self._errors.last_error,
                    "last_error_time": (
                        self._errors.last_error_time.isoformat()
                        if self._errors.last_error_time
                        else None
                    ),
                },
                "collector": {
                    "uptime_seconds": self.get_uptime(),
                    "collection_count": self._collection_count,
                    "timestamp": datetime.now().isoformat(),
                },
            }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus text format metrics.
        """
        metrics = self.export_metrics()
        lines = []

        # Startup metrics
        lines.append("# HELP solid_startup_time_seconds Total startup time")
        lines.append("# TYPE solid_startup_time_seconds gauge")
        lines.append(f"solid_startup_time_seconds {metrics['startup']['total_time_seconds']}")

        # Cache metrics
        for cache_name, cache_metrics in metrics["caches"].items():
            safe_name = cache_name.replace(".", "_").replace("-", "_")
            lines.append("# HELP solid_cache_hits_total Cache hits")
            lines.append("# TYPE solid_cache_hits_total counter")
            lines.append(f"solid_cache_hits_total{{cache=\"{safe_name}\"}} {cache_metrics['hits']}")

            lines.append("# HELP solid_cache_misses_total Cache misses")
            lines.append("# TYPE solid_cache_misses_total counter")
            lines.append(
                f"solid_cache_misses_total{{cache=\"{safe_name}\"}} {cache_metrics['misses']}"
            )

            lines.append("# HELP solid_cache_hit_rate Cache hit rate percentage")
            lines.append("# TYPE solid_cache_hit_rate gauge")
            lines.append(
                f"solid_cache_hit_rate{{cache=\"{safe_name}\"}} {cache_metrics['hit_rate']}"
            )

        # Feature flag metrics
        for flag_name, flag_metrics in metrics["feature_flags"].items():
            safe_name = flag_name.replace(".", "_").replace("-", "_").lower()
            lines.append("# HELP solid_feature_flag_enabled Feature flag status")
            lines.append("# TYPE solid_feature_flag_enabled gauge")
            lines.append(
                f"solid_feature_flag_enabled{{flag=\"{safe_name}\"}} {1 if flag_metrics['enabled'] else 0}"
            )

        # Error metrics
        lines.append("# HELP solid_errors_total Total errors")
        lines.append("# TYPE solid_errors_total counter")
        lines.append(f"solid_errors_total {metrics['errors']['total_errors']}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._startup = StartupMetrics()
            self._caches.clear()
            self._feature_flags.clear()
            self._errors = ErrorMetrics()
            self._collection_count = 0


# =============================================================================
# Singleton Instance
# =============================================================================

_collector_instance: Optional[SolidMetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> SolidMetricsCollector:
    """Get singleton metrics collector instance.

    Returns:
        Singleton SolidMetricsCollector instance.
    """
    global _collector_instance

    if _collector_instance is None:
        with _collector_lock:
            if _collector_instance is None:
                _collector_instance = SolidMetricsCollector()

    return _collector_instance


def collect_feature_flags() -> None:
    """Collect current feature flag status.

    Reads all SOLID-related feature flags from environment and
    records them in the metrics collector.
    """
    collector = get_metrics_collector()

    feature_flags = [
        "VICTOR_USE_NEW_PROTOCOLS",
        "VICTOR_USE_CONTEXT_CONFIG",
        "VICTOR_USE_PLUGIN_DISCOVERY",
        "VICTOR_USE_TYPE_SAFE_LAZY",
        "VICTOR_LAZY_INITIALIZATION",
        "VICTOR_AIRGAPPED",
    ]

    for flag in feature_flags:
        value = os.environ.get(flag, "")
        enabled = value.lower() in ("true", "1", "yes", "on")
        collector.record_feature_flag(
            flag_name=flag,
            enabled=enabled,
            value=value,
            source="environment",
        )


def measure_startup_time() -> float:
    """Measure application startup time.

    This function should be called during application startup to
    measure the time from import to full initialization.

    Returns:
        Startup time in seconds.
    """
    import_start = time.time()

    # Import all verticals (this triggers lazy initialization)
    try:

        verticals = ["coding", "research", "devops", "dataanalysis"]
    except Exception as e:
        logger.warning(f"Failed to import verticals: {e}")
        verticals = []

    import_end = time.time()
    import_time = import_end - import_start

    # Record metrics
    collector = get_metrics_collector()
    collector.record_startup_time(
        total_time=import_time,
        import_time=import_time,
        verticals_loaded=len(verticals),
    )

    # Collect feature flags
    collect_feature_flags()

    return import_time


def print_metrics_summary() -> None:
    """Print a human-readable summary of collected metrics.

    This is useful for console output and logging.
    """
    collector = get_metrics_collector()
    metrics = collector.export_metrics()

    print("\n" + "=" * 60)
    print("SOLID Remediation Metrics Summary")
    print("=" * 60)

    # Startup metrics
    print("\n📊 Startup Performance:")
    print(f"  Total Time: {metrics['startup']['total_time_seconds']:.3f}s")
    print(f"  Import Time: {metrics['startup']['import_time_seconds']:.3f}s")
    print(f"  Verticals Loaded: {metrics['startup']['verticals_loaded']}")

    # Cache metrics
    if metrics["caches"]:
        print("\n💾 Cache Performance:")
        for name, cache in metrics["caches"].items():
            print(f"  {name}:")
            print(f"    Hit Rate: {cache['hit_rate']:.1f}%")
            print(f"    Hits: {cache['hits']}, Misses: {cache['misses']}")

    # Feature flags
    print("\n🚩 Feature Flags:")
    for name, flag in metrics["feature_flags"].items():
        status = "✅" if flag["enabled"] else "❌"
        print(f"  {status} {name}: {flag['value']}")

    # Errors
    if metrics["errors"]["total_errors"] > 0:
        print("\n⚠️  Errors:")
        print(f"  Total: {metrics['errors']['total_errors']}")
        for error_type, count in metrics["errors"]["errors_by_type"].items():
            print(f"    {error_type}: {count}")

    print("\n" + "=" * 60 + "\n")
