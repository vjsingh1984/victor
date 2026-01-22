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

"""Comprehensive health check system for feature flags and system components.

This module provides health checking for:
- Component health (providers, tools, workflows)
- Dependency verification (APIs, databases, services)
- Resource usage (CPU, memory, disk, network)
- Performance thresholds (latency, error rates)
- Degradation detection (identify underperforming features)
- Auto-remediation suggestions

Integration with existing health check infrastructure in victor/core/health.py
"""

from __future__ import annotations

import asyncio
import logging
import psutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from victor.core.health import HealthStatus, ComponentHealth, HealthChecker

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Severity level for health issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthIssue:
    """Health issue detected during checks."""

    component: str
    severity: SeverityLevel
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "suggestion": self.suggestion,
        }


class HealthCheck(ABC):
    """Abstract base for health checks.

    Implementations define specific health check logic.

    Design Pattern: Strategy Pattern
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get health check name."""
        pass

    @property
    @abstractmethod
    def is_critical(self) -> bool:
        """Check if this is a critical health check."""
        pass

    @abstractmethod
    async def check(self) -> ComponentHealth:
        """Perform health check.

        Returns:
            ComponentHealth status
        """
        pass


class ResourceHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk, network)."""

    def __init__(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0,
        critical: bool = True,
    ) -> None:
        """Initialize resource health check.

        Args:
            cpu_threshold: CPU usage threshold (percentage)
            memory_threshold: Memory usage threshold (percentage)
            disk_threshold: Disk usage threshold (percentage)
            critical: Whether this is a critical check
        """
        self._cpu_threshold = cpu_threshold
        self._memory_threshold = memory_threshold
        self._disk_threshold = disk_threshold
        self._critical = critical

    @property
    def name(self) -> str:
        """Get health check name."""
        return "system_resources"

    @property
    def is_critical(self) -> bool:
        """Check if critical."""
        return self._critical

    async def check(self) -> ComponentHealth:
        """Perform resource health check."""
        start = datetime.now(timezone.utc)

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            # Determine overall status
            status = HealthStatus.HEALTHY
            messages = []

            if cpu_percent > self._cpu_threshold:
                status = HealthStatus.DEGRADED if cpu_percent < 95 else HealthStatus.UNHEALTHY
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")

            if memory_percent > self._memory_threshold:
                status = HealthStatus.DEGRADED if memory_percent < 95 else HealthStatus.UNHEALTHY
                messages.append(f"High memory usage: {memory_percent:.1f}%")

            if disk_percent > self._disk_threshold:
                status = HealthStatus.DEGRADED if disk_percent < 95 else HealthStatus.UNHEALTHY
                messages.append(f"High disk usage: {disk_percent:.1f}%")

            message = "; ".join(messages) if messages else "System resources healthy"

            return ComponentHealth(
                name=self.name,
                status=status,
                message=message,
                latency_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "cpu_threshold": self._cpu_threshold,
                    "memory_threshold": self._memory_threshold,
                    "disk_threshold": self._disk_threshold,
                },
                last_check=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.exception(f"Resource health check failed: {e}")
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
                last_check=datetime.now(timezone.utc),
            )


class DependencyHealthCheck(HealthCheck):
    """Health check for external dependencies (APIs, databases, services)."""

    def __init__(
        self,
        dependency_name: str,
        check_fn: Callable[[], Any],
        timeout: float = 5.0,
        critical: bool = True,
    ) -> None:
        """Initialize dependency health check.

        Args:
            dependency_name: Name of the dependency
            check_fn: Async function to check dependency health
            timeout: Check timeout in seconds
            critical: Whether this is a critical check
        """
        self._dependency_name = dependency_name
        self._check_fn = check_fn
        self._timeout = timeout
        self._critical = critical

    @property
    def name(self) -> str:
        """Get health check name."""
        return f"dependency.{self._dependency_name}"

    @property
    def is_critical(self) -> bool:
        """Check if critical."""
        return self._critical

    async def check(self) -> ComponentHealth:
        """Perform dependency health check."""
        start = datetime.now(timezone.utc)

        try:
            # Run check function with timeout
            result = await asyncio.wait_for(self._check_fn(), timeout=self._timeout)

            # Check if result indicates health
            if isinstance(result, bool):
                healthy = result
            elif isinstance(result, dict):
                healthy = result.get("healthy", True)
            else:
                healthy = True

            return ComponentHealth(
                name=self.name,
                status=HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY,
                message=f"Dependency '{self._dependency_name}' is {'healthy' if healthy else 'unhealthy'}",
                latency_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
                last_check=datetime.now(timezone.utc),
            )

        except asyncio.TimeoutError:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency '{self._dependency_name}' check timed out",
                latency_ms=self._timeout * 1000,
                last_check=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.exception(f"Dependency health check failed for {self._dependency_name}: {e}")
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency '{self._dependency_name}' check failed: {str(e)}",
                latency_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
                last_check=datetime.now(timezone.utc),
            )


class PerformanceHealthCheck(HealthCheck):
    """Health check for performance metrics (latency, throughput, error rates)."""

    def __init__(
        self,
        component_name: str,
        latency_threshold_ms: float = 1000.0,
        error_rate_threshold: float = 0.05,
        throughput_threshold: float = 1.0,
        metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        critical: bool = True,
    ) -> None:
        """Initialize performance health check.

        Args:
            component_name: Name of the component
            latency_threshold_ms: Latency threshold in milliseconds
            error_rate_threshold: Error rate threshold (0-1)
            throughput_threshold: Minimum throughput (requests/sec)
            metrics_provider: Function to get metrics (latency_ms, error_rate, throughput)
            critical: Whether this is a critical check
        """
        self._component_name = component_name
        self._latency_threshold = latency_threshold_ms
        self._error_rate_threshold = error_rate_threshold
        self._throughput_threshold = throughput_threshold
        self._metrics_provider = metrics_provider
        self._critical = critical

    @property
    def name(self) -> str:
        """Get health check name."""
        return f"performance.{self._component_name}"

    @property
    def is_critical(self) -> bool:
        """Check if critical."""
        return self._critical

    async def check(self) -> ComponentHealth:
        """Perform performance health check."""
        start = datetime.now(timezone.utc)

        try:
            # Get metrics from provider
            if self._metrics_provider:
                metrics = self._metrics_provider()
            else:
                metrics = {}

            latency_ms = metrics.get("latency_ms", 0)
            error_rate = metrics.get("error_rate", 0)
            throughput = metrics.get("throughput", 0)

            # Check thresholds
            status = HealthStatus.HEALTHY
            messages = []

            if latency_ms > self._latency_threshold:
                status = HealthStatus.DEGRADED
                messages.append(f"High latency: {latency_ms:.0f}ms > {self._latency_threshold:.0f}ms")

            if error_rate > self._error_rate_threshold:
                status = HealthStatus.UNHEALTHY
                messages.append(f"High error rate: {error_rate:.1%} > {self._error_rate_threshold:.1%}")

            if throughput < self._throughput_threshold and throughput > 0:
                status = HealthStatus.DEGRADED
                messages.append(f"Low throughput: {throughput:.1f}/s < {self._throughput_threshold:.1f}/s")

            message = "; ".join(messages) if messages else "Performance metrics healthy"

            return ComponentHealth(
                name=self.name,
                status=status,
                message=message,
                latency_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
                details={
                    "latency_ms": latency_ms,
                    "error_rate": error_rate,
                    "throughput": throughput,
                    "latency_threshold_ms": self._latency_threshold,
                    "error_rate_threshold": self._error_rate_threshold,
                    "throughput_threshold": self._throughput_threshold,
                },
                last_check=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.exception(f"Performance health check failed for {self._component_name}: {e}")
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
                last_check=datetime.now(timezone.utc),
            )


class DegradationDetector:
    """Detects performance degradation for feature flags.

    Compares current metrics against historical baselines to identify
    underperforming features.

    Example:
        detector = DegradationDetector()
        issues = detector.check_for_degradation("hierarchical_planning_enabled")
        for issue in issues:
            print(f"{issue.severity}: {issue.message}")
    """

    def __init__(
        self,
        baseline_window_minutes: int = 60,
        degradation_threshold: float = 0.2,
        critical_threshold: float = 0.5,
    ) -> None:
        """Initialize degradation detector.

        Args:
            baseline_window_minutes: Time window for baseline calculation
            degradation_threshold: Performance change considered degraded (20%)
            critical_threshold: Performance change considered critical (50%)
        """
        self._baseline_window = timedelta(minutes=baseline_window_minutes)
        self._degradation_threshold = degradation_threshold
        self._critical_threshold = critical_threshold

    def check_for_degradation(
        self,
        feature_name: str,
        current_metrics: Dict[str, float],
        historical_metrics: List[Dict[str, float]],
    ) -> List[HealthIssue]:
        """Check for performance degradation.

        Args:
            feature_name: Name of the feature
            current_metrics: Current performance metrics
            historical_metrics: Historical metrics for baseline

        Returns:
            List of health issues
        """
        issues = []
        now = datetime.now(timezone.utc)

        # Filter metrics within baseline window
        baseline_metrics = [
            m
            for m in historical_metrics
            if now - m.get("timestamp", now) <= self._baseline_window
        ]

        if not baseline_metrics:
            # No baseline data
            return []

        # Calculate baselines
        baseline_latency = sum(m.get("latency_ms", 0) for m in baseline_metrics) / len(baseline_metrics)
        baseline_error_rate = sum(m.get("error_rate", 0) for m in baseline_metrics) / len(
            baseline_metrics
        )

        # Check current metrics
        current_latency = current_metrics.get("latency_ms", 0)
        current_error_rate = current_metrics.get("error_rate", 0)

        # Latency degradation
        if baseline_latency > 0:
            latency_change = (current_latency - baseline_latency) / baseline_latency

            if latency_change > self._critical_threshold:
                issues.append(
                    HealthIssue(
                        component=feature_name,
                        severity=SeverityLevel.CRITICAL,
                        message=f"Critical latency degradation: {latency_change:+.1%} change",
                        timestamp=now,
                        details={
                            "baseline_latency_ms": baseline_latency,
                            "current_latency_ms": current_latency,
                            "change_percent": latency_change * 100,
                        },
                        suggestion="Consider disabling feature or investigating performance bottleneck",
                    )
                )
            elif latency_change > self._degradation_threshold:
                issues.append(
                    HealthIssue(
                        component=feature_name,
                        severity=SeverityLevel.WARNING,
                        message=f"Latency degradation detected: {latency_change:+.1%} change",
                        timestamp=now,
                        details={
                            "baseline_latency_ms": baseline_latency,
                            "current_latency_ms": current_latency,
                            "change_percent": latency_change * 100,
                        },
                        suggestion="Monitor closely and investigate if degradation continues",
                    )
                )

        # Error rate degradation
        if baseline_error_rate > 0:
            error_change = (current_error_rate - baseline_error_rate) / baseline_error_rate

            if error_change > self._critical_threshold:
                issues.append(
                    HealthIssue(
                        component=feature_name,
                        severity=SeverityLevel.CRITICAL,
                        message=f"Critical error rate degradation: {error_change:+.1%} change",
                        timestamp=now,
                        details={
                            "baseline_error_rate": baseline_error_rate,
                            "current_error_rate": current_error_rate,
                            "change_percent": error_change * 100,
                        },
                        suggestion="Disable feature immediately and investigate root cause",
                    )
                )
            elif error_change > self._degradation_threshold:
                issues.append(
                    HealthIssue(
                        component=feature_name,
                        severity=SeverityLevel.WARNING,
                        message=f"Error rate degradation detected: {error_change:+.1%} change",
                        timestamp=now,
                        details={
                            "baseline_error_rate": baseline_error_rate,
                            "current_error_rate": current_error_rate,
                            "change_percent": error_change * 100,
                        },
                        suggestion="Investigate error patterns and consider rollback",
                    )
                )

        return issues


class FeatureHealthChecker:
    """Comprehensive health checker for feature flags.

    Combines multiple health checks:
    - Resource usage
    - Dependency availability
    - Performance metrics
    - Degradation detection

    Example:
        checker = FeatureHealthChecker()
        report = await checker.check_health("hierarchical_planning_enabled")
        print(report.status)
    """

    def __init__(self) -> None:
        """Initialize feature health checker."""
        self._health_checks: List[HealthCheck] = []
        self._degradation_detector = DegradationDetector()

    def add_resource_check(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0,
        critical: bool = True,
    ) -> None:
        """Add resource health check.

        Args:
            cpu_threshold: CPU usage threshold
            memory_threshold: Memory usage threshold
            disk_threshold: Disk usage threshold
            critical: Whether critical
        """
        self._health_checks.append(
            ResourceHealthCheck(
                cpu_threshold=cpu_threshold,
                memory_threshold=memory_threshold,
                disk_threshold=disk_threshold,
                critical=critical,
            )
        )

    def add_dependency_check(
        self,
        dependency_name: str,
        check_fn: Callable[[], Any],
        timeout: float = 5.0,
        critical: bool = True,
    ) -> None:
        """Add dependency health check.

        Args:
            dependency_name: Name of dependency
            check_fn: Check function
            timeout: Timeout in seconds
            critical: Whether critical
        """
        self._health_checks.append(
            DependencyHealthCheck(
                dependency_name=dependency_name,
                check_fn=check_fn,
                timeout=timeout,
                critical=critical,
            )
        )

    def add_performance_check(
        self,
        component_name: str,
        latency_threshold_ms: float = 1000.0,
        error_rate_threshold: float = 0.05,
        throughput_threshold: float = 1.0,
        metrics_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        critical: bool = True,
    ) -> None:
        """Add performance health check.

        Args:
            component_name: Name of component
            latency_threshold_ms: Latency threshold
            error_rate_threshold: Error rate threshold
            throughput_threshold: Throughput threshold
            metrics_provider: Metrics provider function
            critical: Whether critical
        """
        self._health_checks.append(
            PerformanceHealthCheck(
                component_name=component_name,
                latency_threshold_ms=latency_threshold_ms,
                error_rate_threshold=error_rate_threshold,
                throughput_threshold=throughput_threshold,
                metrics_provider=metrics_provider,
                critical=critical,
            )
        )

    async def check_all(self) -> List[ComponentHealth]:
        """Run all health checks.

        Returns:
            List of component health results
        """
        tasks = [check.check() for check in self._health_checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_results = []
        for check, result in zip(self._health_checks, results, strict=False):
            if isinstance(result, Exception):
                health_results.append(
                    ComponentHealth(
                        name=check.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check error: {result}",
                        last_check=datetime.now(timezone.utc),
                    )
                )
            else:
                health_results.append(result)

        return health_results

    async def check_feature_degradation(
        self,
        feature_name: str,
        current_metrics: Dict[str, float],
        historical_metrics: List[Dict[str, float]],
    ) -> List[HealthIssue]:
        """Check for feature performance degradation.

        Args:
            feature_name: Name of the feature
            current_metrics: Current metrics
            historical_metrics: Historical metrics

        Returns:
            List of health issues
        """
        return self._degradation_detector.check_for_degradation(
            feature_name=feature_name,
            current_metrics=current_metrics,
            historical_metrics=historical_metrics,
        )
