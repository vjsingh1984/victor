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

"""Health check endpoint for coordinator monitoring.

This module provides HTTP health check endpoints that integrate with
the existing health check system and coordinator metrics.

Provides:
- /health endpoint for health status
- /health/ready for readiness probes
- /health/live for liveness probes
- Coordinator-specific health checks
- FastAPI integration

Example:
    from fastapi import FastAPI
    from victor.observability.health import setup_health_endpoints

    app = FastAPI()
    setup_health_endpoints(app)

    # Endpoints available:
    # GET /health - Overall health
    # GET /health/ready - Readiness probe
    # GET /health/live - Liveness probe
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi.responses import JSONResponse

from victor.core.health import (
    HealthChecker,
    HealthReport,
    HealthStatus,
    ComponentHealth,
    create_default_health_checker,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Coordinator Health Checks
# =============================================================================


class CoordinatorHealthCheck:
    """Health check for coordinator components.

    Monitors:
    - Coordinator availability
    - Execution metrics
    - Error rates
    - Performance degradation
    """

    def __init__(
        self,
        coordinator_name: str,
        metrics_collector: Any,  # CoordinatorMetricsCollector
        timeout: float = 5.0,
        critical: bool = True,
        error_rate_threshold: float = 0.05,  # 5%
        latency_threshold_ms: float = 5000.0,  # 5 seconds
    ) -> None:
        """Initialize coordinator health check.

        Args:
            coordinator_name: Name of the coordinator.
            metrics_collector: CoordinatorMetricsCollector instance.
            timeout: Check timeout.
            critical: Whether coordinator is critical.
            error_rate_threshold: Error rate threshold for unhealthy status.
            latency_threshold_ms: Latency threshold for degraded status.
        """
        self._name = f"coordinator.{coordinator_name}"
        self._coordinator_name = coordinator_name
        self._metrics = metrics_collector
        self._timeout = timeout
        self._critical = critical
        self._error_rate_threshold = error_rate_threshold
        self._latency_threshold_ms = latency_threshold_ms

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    @property
    def is_critical(self) -> bool:
        """Check if critical."""
        return self._critical

    async def check(self) -> ComponentHealth:
        """Perform health check.

        Returns:
            ComponentHealth status.
        """
        start = time.perf_counter()

        try:
            # Get metrics snapshot
            snapshot = self._metrics.get_snapshot(self._coordinator_name)

            if snapshot is None:
                # Coordinator hasn't been used yet, assume healthy
                return ComponentHealth(
                    name=self._name,
                    status=HealthStatus.HEALTHY,
                    message="Coordinator initialized but not yet used",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    last_check=datetime.now(timezone.utc),
                )

            # Check error rate
            stats = self._metrics.get_coordinator_stats(self._coordinator_name)
            error_rate = stats.get("error_rate", 0)

            # Check average latency
            avg_duration = stats.get("avg_duration_ms", 0)

            # Determine status
            status = HealthStatus.HEALTHY
            message = "Coordinator is healthy"

            if error_rate > self._error_rate_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"Error rate ({error_rate:.1%}) exceeds threshold ({self._error_rate_threshold:.1%})"
            elif avg_duration > self._latency_threshold_ms:
                status = HealthStatus.DEGRADED
                message = f"Average latency ({avg_duration:.0f}ms) exceeds threshold ({self._latency_threshold_ms:.0f}ms)"
            elif snapshot.error_count > 0:
                status = HealthStatus.DEGRADED
                message = f"Coordinator has {snapshot.error_count} errors"

            return ComponentHealth(
                name=self._name,
                status=status,
                message=message,
                latency_ms=(time.perf_counter() - start) * 1000,
                details={
                    "execution_count": snapshot.execution_count,
                    "total_duration_ms": snapshot.total_duration_ms,
                    "error_count": snapshot.error_count,
                    "error_rate": error_rate,
                    "avg_duration_ms": avg_duration,
                    "cache_hit_rate": snapshot.to_dict()["cache_hit_rate"],
                    "memory_mb": snapshot.memory_bytes / (1024 * 1024),
                    "cpu_percent": snapshot.cpu_percent,
                },
                last_check=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.exception(f"Health check failed for {self._coordinator_name}")
            return ComponentHealth(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.perf_counter() - start) * 1000,
                last_check=datetime.now(timezone.utc),
            )


# =============================================================================
# Health Check Service
# =============================================================================


class CoordinatorHealthService:
    """Service for checking health of all coordinators.

    Provides:
    - Overall health status
    - Individual coordinator health
    - Health history tracking
    - Kubernetes-ready probes
    """

    def __init__(
        self,
        metrics_collector: Any,  # CoordinatorMetricsCollector
        health_checker: Optional[HealthChecker] = None,
    ) -> None:
        """Initialize health service.

        Args:
            metrics_collector: CoordinatorMetricsCollector instance.
            health_checker: Optional existing HealthChecker.
        """
        self._metrics = metrics_collector
        self._checker = health_checker or create_default_health_checker()
        self._coordinator_checks: dict[str, CoordinatorHealthCheck] = {}
        self._start_time = time.time()

    def add_coordinator_check(
        self,
        coordinator_name: str,
        critical: bool = True,
        error_rate_threshold: float = 0.05,
        latency_threshold_ms: float = 5000.0,
    ) -> None:
        """Add health check for a coordinator.

        Args:
            coordinator_name: Name of coordinator.
            critical: Whether coordinator is critical.
            error_rate_threshold: Error rate threshold.
            latency_threshold_ms: Latency threshold.
        """
        check = CoordinatorHealthCheck(
            coordinator_name=coordinator_name,
            metrics_collector=self._metrics,
            critical=critical,
            error_rate_threshold=error_rate_threshold,
            latency_threshold_ms=latency_threshold_ms,
        )
        self._coordinator_checks[coordinator_name] = check

    def remove_coordinator_check(self, coordinator_name: str) -> None:
        """Remove health check for a coordinator.

        Args:
            coordinator_name: Name of coordinator.
        """
        self._coordinator_checks.pop(coordinator_name, None)

    async def check_health(self, use_cache: bool = True) -> HealthReport:
        """Check health of all components.

        Args:
            use_cache: Whether to use cached results.

        Returns:
            HealthReport with status of all components.
        """
        # Check base health checker
        report = await self._checker.check_health(use_cache=use_cache)

        # Check all coordinators
        if self._coordinator_checks:
            tasks = [check.check() for check in self._coordinator_checks.values()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for check, result in zip(self._coordinator_checks.values(), results, strict=False):
                if isinstance(result, Exception):
                    report.components[check.name] = ComponentHealth(
                        name=check.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check error: {result}",
                        last_check=datetime.now(timezone.utc),
                    )
                else:
                    report.components[check.name] = result  # type: ignore[assignment]

        return report

    async def is_ready(self) -> bool:
        """Check if system is ready to serve traffic.

        Returns:
            True if ready.
        """
        report = await self.check_health()
        return report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    async def is_alive(self) -> bool:
        """Check if system is alive (liveness probe).

        Returns:
            True if alive.
        """
        return True

    def get_coordinator_names(self) -> list[str]:
        """Get list of monitored coordinator names.

        Returns:
            List of coordinator names.
        """
        return list(self._coordinator_checks.keys())


# =============================================================================
# FastAPI Endpoints
# =============================================================================


def setup_health_endpoints(
    app: Any,  # FastAPI
    health_service: Optional[CoordinatorHealthService] = None,
) -> None:
    """Setup health check endpoints on FastAPI app.

    Args:
        app: FastAPI application instance.
        health_service: Optional CoordinatorHealthService (creates default if not provided).

    Example:
        from fastapi import FastAPI
        from victor.observability.health import setup_health_endpoints

        app = FastAPI()
        setup_health_endpoints(app)
    """
    if health_service is None:
        from victor.observability.coordinator_metrics import get_coordinator_metrics_collector
        from victor.observability.health import CoordinatorHealthService

        metrics_collector = get_coordinator_metrics_collector()
        health_service = CoordinatorHealthService(metrics_collector)

        # Auto-detect and add coordinator checks
        snapshots = metrics_collector.get_all_snapshots()
        for snapshot in snapshots:
            health_service.add_coordinator_check(
                coordinator_name=snapshot.coordinator_name,
                critical=True,
            )

    @app.get("/health")
    async def health_check():
        """Overall health check endpoint."""
        report = await health_service.check_health()
        return JSONResponse(
            content=report.to_dict(),
            status_code=200 if report.is_healthy else 503,
        )

    @app.get("/health/ready")
    async def readiness_check():
        """Kubernetes readiness probe."""
        is_ready = await health_service.is_ready()
        return JSONResponse(
            content={"status": "ready" if is_ready else "not_ready"},
            status_code=200 if is_ready else 503,
        )

    @app.get("/health/live")
    async def liveness_check():
        """Kubernetes liveness probe."""
        is_alive = await health_service.is_alive()
        return JSONResponse(
            content={"status": "alive" if is_alive else "dead"},
            status_code=200 if is_alive else 503,
        )

    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with all components."""
        report = await health_service.check_health(use_cache=False)

        # Add additional metadata
        response_data = report.to_dict()
        response_data["monitored_coordinators"] = health_service.get_coordinator_names()
        response_data["uptime_seconds"] = time.time() - health_service._start_time

        return JSONResponse(
            content=response_data,
            status_code=200 if report.is_healthy else 503,
        )

    logger.info(
        "Health check endpoints registered: /health, /health/ready, /health/live, /health/detailed"
    )


# =============================================================================
# Helper Functions
# =============================================================================


async def create_coordinator_health_service(
    coordinators: Optional[list[str]] = None,
    error_rate_threshold: float = 0.05,
    latency_threshold_ms: float = 5000.0,
) -> CoordinatorHealthService:
    """Create health service with coordinator checks.

    Args:
        coordinators: List of coordinator names to monitor (auto-detect if None).
        error_rate_threshold: Default error rate threshold.
        latency_threshold_ms: Default latency threshold.

    Returns:
        CoordinatorHealthService instance.
    """
    from victor.observability.coordinator_metrics import get_coordinator_metrics_collector

    metrics_collector = get_coordinator_metrics_collector()
    service = CoordinatorHealthService(metrics_collector)

    # Auto-detect coordinators if not provided
    if coordinators is None:
        snapshots = metrics_collector.get_all_snapshots()
        coordinators = [s.coordinator_name for s in snapshots]

    # Add checks for each coordinator
    for coordinator_name in coordinators:
        service.add_coordinator_check(
            coordinator_name=coordinator_name,
            critical=True,
            error_rate_threshold=error_rate_threshold,
            latency_threshold_ms=latency_threshold_ms,
        )

    return service
