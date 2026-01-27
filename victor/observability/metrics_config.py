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

"""Enhanced metrics configuration for feature flag observability.

This module provides enhanced metrics collection specifically for monitoring
feature flag performance, usage, and business impact.

Metrics Categories:
- Feature performance: latency, throughput, error rates per feature
- Error tracking: errors by feature, severity, frequency
- Usage analytics: feature adoption, usage patterns
- Resource monitoring: CPU, memory, I/O per feature
- Business metrics: task completion rates, user satisfaction

Integration with existing MetricsRegistry from victor/observability/metrics.py
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from victor.observability.metrics import (
    MetricsRegistry,
    Counter,
    Gauge,
    Histogram,
    MetricLabels,
)

logger = logging.getLogger(__name__)


class FeatureMetrics:
    """Metrics collection for a single feature flag.

    Tracks:
    - Enable/disable count
    - Performance metrics (latency, throughput)
    - Error rates
    - Usage patterns

    Example:
        metrics = FeatureMetrics("hierarchical_planning_enabled", registry)
        metrics.record_enabled(True)
        metrics.record_execution(duration_ms=150, success=True)
    """

    def __init__(
        self,
        feature_name: str,
        registry: Optional[MetricsRegistry] = None,
        prefix: str = "victor_feature",
    ) -> None:
        """Initialize feature metrics.

        Args:
            feature_name: Name of the feature flag
            registry: Metrics registry (uses default if None)
            prefix: Metric name prefix
        """
        self._feature_name = feature_name
        self._registry = registry or MetricsRegistry.get_instance()
        self._prefix = prefix

        # Create metrics
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Create feature-specific metrics."""
        labels = {"feature": self._feature_name}

        # Feature state metrics
        self.enabled = self._registry.gauge(
            f"{self._prefix}_enabled",
            "Whether feature is enabled (1=yes, 0=no)",
            labels=labels.copy(),
        )

        self.enable_count = self._registry.counter(
            f"{self._prefix}_enable_count",
            "Number of times feature was enabled",
            labels=labels.copy(),
        )

        self.disable_count = self._registry.counter(
            f"{self._prefix}_disable_count",
            "Number of times feature was disabled",
            labels=labels.copy(),
        )

        # Performance metrics
        self.executions = self._registry.counter(
            f"{self._prefix}_executions_total",
            "Total feature executions",
            labels=labels.copy(),
        )

        self.errors = self._registry.counter(
            f"{self._prefix}_errors_total",
            "Total feature errors",
            labels=labels.copy(),
        )

        self.duration = self._registry.histogram(
            f"{self._prefix}_duration_ms",
            "Feature execution duration",
            labels=labels.copy(),
        )

        # Throughput metrics
        self.throughput = self._registry.gauge(
            f"{self._prefix}_throughput",
            "Feature executions per second",
            labels=labels.copy(),
        )

        # Usage metrics
        self.users = self._registry.gauge(
            f"{self._prefix}_active_users",
            "Number of active users",
            labels=labels.copy(),
        )

    def record_enabled(self, enabled: bool) -> None:
        """Record feature state change.

        Args:
            enabled: Whether feature is enabled
        """
        self.enabled.set(1 if enabled else 0)

        if enabled:
            self.enable_count.inc()
        else:
            self.disable_count.inc()

    def record_execution(
        self,
        duration_ms: float,
        success: bool = True,
        error_type: Optional[str] = None,
    ) -> None:
        """Record feature execution.

        Args:
            duration_ms: Execution duration in milliseconds
            success: Whether execution succeeded
            error_type: Type of error (if failed)
        """
        self.executions.inc()
        self.duration.observe(duration_ms)

        if not success:
            if error_type:
                self.errors.inc()

    def record_user_activity(self, user_count: int) -> None:
        """Record active user count.

        Args:
            user_count: Number of active users
        """
        self.users.set(user_count)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns:
            Summary dictionary
        """
        return {
            "feature": self._feature_name,
            "enabled": self.enabled.value == 1,
            "enable_count": self.enable_count.value,
            "disable_count": self.disable_count.value,
            "executions": self.executions.value,
            "errors": self.errors.value,
            "error_rate": (
                self.errors.value / self.executions.value if self.executions.value > 0 else 0.0
            ),
            "avg_duration_ms": self.duration.mean,
            "p50_duration_ms": self.duration.percentile(50),
            "p95_duration_ms": self.duration.percentile(95),
            "p99_duration_ms": self.duration.percentile(99),
            "active_users": self.users.value,
        }


class FeatureMetricsRegistry:
    """Registry for feature metrics with automatic cleanup.

    Manages metrics collection for all feature flags.

    Example:
        registry = FeatureMetricsRegistry()
        metrics = registry.get_or_create("hierarchical_planning_enabled")
        metrics.record_execution(duration_ms=100, success=True)
    """

    def __init__(
        self,
        base_registry: Optional[MetricsRegistry] = None,
        prefix: str = "victor_feature",
    ) -> None:
        """Initialize feature metrics registry.

        Args:
            base_registry: Base metrics registry
            prefix: Metric name prefix
        """
        self._base_registry = base_registry or MetricsRegistry.get_instance()
        self._prefix = prefix
        self._metrics: Dict[str, FeatureMetrics] = {}
        self._lock = threading.RLock()

    def get_or_create(self, feature_name: str) -> FeatureMetrics:
        """Get or create feature metrics.

        Args:
            feature_name: Name of the feature

        Returns:
            FeatureMetrics instance
        """
        with self._lock:
            if feature_name not in self._metrics:
                self._metrics[feature_name] = FeatureMetrics(
                    feature_name=feature_name,
                    registry=self._base_registry,
                    prefix=self._prefix,
                )
            return self._metrics[feature_name]

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries for all features.

        Returns:
            List of summary dictionaries
        """
        with self._lock:
            return [metrics.get_summary() for metrics in self._metrics.values()]

    def clear(self) -> None:
        """Clear all feature metrics."""
        with self._lock:
            self._metrics.clear()


class ResourceMetrics:
    """Resource monitoring metrics per feature.

    Tracks CPU, memory, I/O usage for feature flags.

    Example:
        metrics = ResourceMetrics("hierarchical_planning_enabled", registry)
        metrics.record_cpu_usage(percent=45.2)
        metrics.record_memory_usage_mb(mbytes=512)
    """

    def __init__(
        self,
        feature_name: str,
        registry: Optional[MetricsRegistry] = None,
        prefix: str = "victor_feature_resource",
    ) -> None:
        """Initialize resource metrics.

        Args:
            feature_name: Name of the feature
            registry: Metrics registry
            prefix: Metric name prefix
        """
        self._feature_name = feature_name
        self._registry = registry or MetricsRegistry.get_instance()
        self._prefix = prefix

        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Create resource metrics."""
        labels = {"feature": self._feature_name}

        self.cpu_percent = self._registry.gauge(
            f"{self._prefix}_cpu_percent",
            "CPU usage percentage",
            labels=labels.copy(),
        )

        self.memory_mb = self._registry.gauge(
            f"{self._prefix}_memory_mb",
            "Memory usage in MB",
            labels=labels.copy(),
        )

        self.disk_io_mb = self._registry.counter(
            f"{self._prefix}_disk_io_mb_total",
            "Total disk I/O in MB",
            labels=labels.copy(),
        )

        self.network_io_mb = self._registry.counter(
            f"{self._prefix}_network_io_mb_total",
            "Total network I/O in MB",
            labels=labels.copy(),
        )

    def record_cpu_usage(self, percent: float) -> None:
        """Record CPU usage.

        Args:
            percent: CPU usage percentage
        """
        self.cpu_percent.set(percent)

    def record_memory_usage_mb(self, mbytes: float) -> None:
        """Record memory usage.

        Args:
            mbytes: Memory usage in MB
        """
        self.memory_mb.set(mbytes)

    def record_disk_io_mb(self, mb: float) -> None:
        """Record disk I/O.

        Args:
            mb: Disk I/O in MB
        """
        self.disk_io_mb.increment(mb)

    def record_network_io_mb(self, mb: float) -> None:
        """Record network I/O.

        Args:
            mb: Network I/O in MB
        """
        self.network_io_mb.increment(mb)


class BusinessMetrics:
    """Business impact metrics for features.

    Tracks task completion rates, user satisfaction, etc.

    Example:
        metrics = BusinessMetrics("hierarchical_planning_enabled", registry)
        metrics.record_task_completion(success=True, user_satisfaction=4.5)
    """

    def __init__(
        self,
        feature_name: str,
        registry: Optional[MetricsRegistry] = None,
        prefix: str = "victor_feature_business",
    ) -> None:
        """Initialize business metrics.

        Args:
            feature_name: Name of the feature
            registry: Metrics registry
            prefix: Metric name prefix
        """
        self._feature_name = feature_name
        self._registry = registry or MetricsRegistry.get_instance()
        self._prefix = prefix

        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Create business metrics."""
        labels = {"feature": self._feature_name}

        self.tasks_completed = self._registry.counter(
            f"{self._prefix}_tasks_completed_total",
            "Total tasks completed with feature",
            labels=labels.copy(),
        )

        self.tasks_failed = self._registry.counter(
            f"{self._prefix}_tasks_failed_total",
            "Total tasks failed with feature",
            labels=labels.copy(),
        )

        self.completion_rate = self._registry.gauge(
            f"{self._prefix}_completion_rate",
            "Task completion rate (0-1)",
            labels=labels.copy(),
        )

        self.user_satisfaction = self._registry.histogram(
            f"{self._prefix}_user_satisfaction",
            "User satisfaction score (1-5)",
            labels=labels.copy(),
        )

    def record_task_completion(self, success: bool) -> None:
        """Record task completion.

        Args:
            success: Whether task completed successfully
        """
        if success:
            self.tasks_completed.inc()
        else:
            self.tasks_failed.inc()

        # Update completion rate
        total = self.tasks_completed.value + self.tasks_failed.value
        if total > 0:
            rate = self.tasks_completed.value / total
            self.completion_rate.set(rate)

    def record_user_satisfaction(self, score: float) -> None:
        """Record user satisfaction score.

        Args:
            score: Satisfaction score (1-5)
        """
        if 1.0 <= score <= 5.0:
            self.user_satisfaction.observe(score)


# Singleton instance
_feature_metrics_registry: Optional[FeatureMetricsRegistry] = None
_registry_lock = threading.Lock()


def get_feature_metrics_registry() -> FeatureMetricsRegistry:
    """Get singleton feature metrics registry.

    Returns:
        FeatureMetricsRegistry instance

    Example:
        registry = get_feature_metrics_registry()
        metrics = registry.get_or_create("hierarchical_planning_enabled")
    """
    global _feature_metrics_registry

    if _feature_metrics_registry is None:
        with _registry_lock:
            if _feature_metrics_registry is None:
                _feature_metrics_registry = FeatureMetricsRegistry()

    return _feature_metrics_registry
