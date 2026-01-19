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

"""Production health check endpoints for Kubernetes probes.

This module provides production health check implementations for:
- Liveness probes: Is the service alive?
- Readiness probes: Is the service ready to accept traffic?
- Startup probes: Has the service started successfully?

Design Patterns:
- Strategy Pattern: Different health check types
- Composite Pattern: Aggregate multiple checks
- Facade Pattern: Unified health check interface

Example:
    from victor.observability.production_health import (
        ProductionHealthChecker,
        HealthCheckResponse,
    )

    checker = ProductionHealthChecker()

    # For Kubernetes probes
    liveness = await checker.liveness()
    readiness = await checker.readiness()
    startup = await checker.startup()

    # For detailed health
    health = await checker.check_health()
    print(health.status)
    print(health.checks)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from victor.core.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    HealthReport,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Health Check Types
# =============================================================================


class HealthCheckType(str, Enum):
    """Types of health checks."""

    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


@dataclass
class HealthCheckResponse:
    """Health check response for HTTP endpoints.

    Attributes:
        status: Health status (healthy/unhealthy)
        checks: Individual check results
        timestamp: Check timestamp
        uptime_seconds: Service uptime
        version: Service version
    """

    status: str
    checks: Dict[str, Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    uptime_seconds: Optional[float] = None
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response.

        Returns:
            Dictionary representation
        """
        return {
            "status": self.status,
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "checks": self.checks,
        }

    @property
    def is_healthy(self) -> bool:
        """Check if status is healthy.

        Returns:
            True if healthy
        """
        return self.status == "healthy"


# =============================================================================
# Production Health Checker
# =============================================================================


class ProductionHealthChecker:
    """Production health checker with multiple probe types.

    Implements Kubernetes-style health probes:
    - Liveness: Is the process running?
    - Readiness: Can it handle requests?
    - Startup: Has it finished starting up?

    Attributes:
        start_time: Service start timestamp
        startup_timeout: Startup probe timeout
        startup_completed: Whether startup has completed

    Example:
        checker = ProductionHealthChecker()

        # Add checks
        checker.add_liveness_check("process", lambda: True)
        checker.add_readiness_check("database", check_db)
        checker.add_startup_check("cache", check_cache)

        # Use in HTTP handlers
        @app.get("/health/live")
        async def liveness():
            return await checker.liveness()

        @app.get("/health/ready")
        async def readiness():
            return await checker.readiness()
    """

    def __init__(
        self,
        startup_timeout: float = 60.0,
        service_version: str = "1.0.0",
    ) -> None:
        """Initialize production health checker.

        Args:
            startup_timeout: Seconds to wait before considering startup failed
            service_version: Service version string
        """
        self.start_time = time.time()
        self.startup_timeout = startup_timeout
        self.service_version = service_version
        self.startup_completed = False

        # Health check storage
        self._liveness_checks: Dict[str, Callable[[], bool]] = {}
        self._readiness_checks: Dict[str, Callable[[], bool]] = {}
        self._startup_checks: Dict[str, Callable[[], bool]] = {}

        # Async health check storage
        self._async_liveness_checks: Dict[str, Callable[[], Any]] = {}
        self._async_readiness_checks: Dict[str, Callable[[], Any]] = {}
        self._async_startup_checks: Dict[str, Callable[[], Any]] = {}

        # Core health checker for detailed checks
        self._health_checker = HealthChecker()

    # ========================================================================
    # Check Registration
    # ========================================================================

    def add_liveness_check(
        self,
        name: str,
        check_func: Callable[[], bool],
    ) -> None:
        """Add a liveness check.

        Liveness checks should be lightweight and quickly determine
        if the process is still running.

        Args:
            name: Check name
            check_func: Function returning True if healthy

        Example:
            checker.add_liveness_check("process", lambda: True)
        """
        self._liveness_checks[name] = check_func

    def add_readiness_check(
        self,
        name: str,
        check_func: Callable[[], bool],
    ) -> None:
        """Add a readiness check.

        Readiness checks determine if the service can handle requests.
        Should check external dependencies.

        Args:
            name: Check name
            check_func: Function returning True if ready

        Example:
            def check_database():
                return database.is_connected()

            checker.add_readiness_check("database", check_database)
        """
        self._readiness_checks[name] = check_func

    def add_startup_check(
        self,
        name: str,
        check_func: Callable[[], bool],
    ) -> None:
        """Add a startup check.

        Startup checks verify that the service has completed initialization.
        Once passed, the service is considered started.

        Args:
            name: Check name
            check_func: Function returning True if started

        Example:
            checker.add_startup_check("cache", lambda: cache.warmed_up)
        """
        self._startup_checks[name] = check_func

    def add_async_liveness_check(
        self,
        name: str,
        check_func: Callable[[], Any],
    ) -> None:
        """Add an async liveness check.

        Args:
            name: Check name
            check_func: Async coroutine returning True if healthy

        Example:
            async def check_process():
                return True

            checker.add_async_liveness_check("process", check_process)
        """
        self._async_liveness_checks[name] = check_func

    def add_async_readiness_check(
        self,
        name: str,
        check_func: Callable[[], Any],
    ) -> None:
        """Add an async readiness check.

        Args:
            name: Check name
            check_func: Async coroutine returning True if ready

        Example:
            async def check_database():
                return await database.ping()

            checker.add_async_readiness_check("database", check_database)
        """
        self._async_readiness_checks[name] = check_func

    def add_async_startup_check(
        self,
        name: str,
        check_func: Callable[[], Any],
    ) -> None:
        """Add an async startup check.

        Args:
            name: Check name
            check_func: Async coroutine returning True if started

        Example:
            async def check_cache():
                return await cache.is_ready()

            checker.add_async_startup_check("cache", check_cache)
        """
        self._async_startup_checks[name] = check_func

    # ========================================================================
    # Health Check Execution
    # ========================================================================

    async def liveness(self) -> HealthCheckResponse:
        """Execute liveness probe.

        Returns:
            HealthCheckResponse with liveness status

        Example:
            @app.get("/health/live")
            async def liveness():
                return await checker.liveness()
        """
        checks = {}

        # Run sync checks
        for name, check_func in self._liveness_checks.items():
            try:
                healthy = check_func()
                checks[name] = {
                    "status": "healthy" if healthy else "unhealthy",
                }
                if not healthy:
                    # Any unhealthy check makes the whole probe fail
                    return HealthCheckResponse(
                        status="unhealthy",
                        checks=checks,
                        uptime_seconds=time.time() - self.start_time,
                        version=self.service_version,
                    )
            except Exception as e:
                logger.error(f"Liveness check {name} failed: {e}")
                checks[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                return HealthCheckResponse(
                    status="unhealthy",
                    checks=checks,
                    uptime_seconds=time.time() - self.start_time,
                    version=self.service_version,
                )

        # Run async checks
        for name, check_func in self._async_liveness_checks.items():
            try:
                healthy = await check_func()
                checks[name] = {
                    "status": "healthy" if healthy else "unhealthy",
                }
                if not healthy:
                    return HealthCheckResponse(
                        status="unhealthy",
                        checks=checks,
                        uptime_seconds=time.time() - self.start_time,
                        version=self.service_version,
                    )
            except Exception as e:
                logger.error(f"Async liveness check {name} failed: {e}")
                checks[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                return HealthCheckResponse(
                    status="unhealthy",
                    checks=checks,
                    uptime_seconds=time.time() - self.start_time,
                    version=self.service_version,
                )

        return HealthCheckResponse(
            status="healthy",
            checks=checks,
            uptime_seconds=time.time() - self.start_time,
            version=self.service_version,
        )

    async def readiness(self) -> HealthCheckResponse:
        """Execute readiness probe.

        Returns:
            HealthCheckResponse with readiness status

        Example:
            @app.get("/health/ready")
            async def readiness():
                return await checker.readiness()
        """
        checks = {}

        # Run sync checks
        for name, check_func in self._readiness_checks.items():
            try:
                ready = check_func()
                checks[name] = {
                    "status": "ready" if ready else "not_ready",
                }
                if not ready:
                    return HealthCheckResponse(
                        status="not_ready",
                        checks=checks,
                        uptime_seconds=time.time() - self.start_time,
                        version=self.service_version,
                    )
            except Exception as e:
                logger.error(f"Readiness check {name} failed: {e}")
                checks[name] = {
                    "status": "not_ready",
                    "error": str(e),
                }
                return HealthCheckResponse(
                    status="not_ready",
                    checks=checks,
                    uptime_seconds=time.time() - self.start_time,
                    version=self.service_version,
                )

        # Run async checks
        for name, check_func in self._async_readiness_checks.items():
            try:
                ready = await check_func()
                checks[name] = {
                    "status": "ready" if ready else "not_ready",
                }
                if not ready:
                    return HealthCheckResponse(
                        status="not_ready",
                        checks=checks,
                        uptime_seconds=time.time() - self.start_time,
                        version=self.service_version,
                    )
            except Exception as e:
                logger.error(f"Async readiness check {name} failed: {e}")
                checks[name] = {
                    "status": "not_ready",
                    "error": str(e),
                }
                return HealthCheckResponse(
                    status="not_ready",
                    checks=checks,
                    uptime_seconds=time.time() - self.start_time,
                    version=self.service_version,
                )

        return HealthCheckResponse(
            status="ready",
            checks=checks,
            uptime_seconds=time.time() - self.start_time,
            version=self.service_version,
        )

    async def startup(self) -> HealthCheckResponse:
        """Execute startup probe.

        Returns:
            HealthCheckResponse with startup status

        Example:
            @app.get("/health/startup")
            async def startup():
                return await checker.startup()
        """
        checks = {}
        uptime = time.time() - self.start_time

        # Check timeout
        if uptime > self.startup_timeout and not self.startup_completed:
            return HealthCheckResponse(
                status="timeout",
                checks={"timeout": {"error": f"Startup timeout after {self.startup_timeout}s"}},
                uptime_seconds=uptime,
                version=self.service_version,
            )

        # If already completed, return immediately
        if self.startup_completed:
            return HealthCheckResponse(
                status="started",
                checks={"startup": {"status": "completed"}},
                uptime_seconds=uptime,
                version=self.service_version,
            )

        # Run sync checks
        for name, check_func in self._startup_checks.items():
            try:
                started = check_func()
                checks[name] = {
                    "status": "started" if started else "starting",
                }
                if not started:
                    return HealthCheckResponse(
                        status="starting",
                        checks=checks,
                        uptime_seconds=uptime,
                        version=self.service_version,
                    )
            except Exception as e:
                logger.error(f"Startup check {name} failed: {e}")
                checks[name] = {
                    "status": "failed",
                    "error": str(e),
                }
                return HealthCheckResponse(
                    status="failed",
                    checks=checks,
                    uptime_seconds=uptime,
                    version=self.service_version,
                )

        # Run async checks
        for name, check_func in self._async_startup_checks.items():
            try:
                started = await check_func()
                checks[name] = {
                    "status": "started" if started else "starting",
                }
                if not started:
                    return HealthCheckResponse(
                        status="starting",
                        checks=checks,
                        uptime_seconds=uptime,
                        version=self.service_version,
                    )
            except Exception as e:
                logger.error(f"Async startup check {name} failed: {e}")
                checks[name] = {
                    "status": "failed",
                    "error": str(e),
                }
                return HealthCheckResponse(
                    status="failed",
                    checks=checks,
                    uptime_seconds=uptime,
                    version=self.service_version,
                )

        # All checks passed
        self.startup_completed = True
        return HealthCheckResponse(
            status="started",
            checks=checks,
            uptime_seconds=uptime,
            version=self.service_version,
        )

    async def check_health(self) -> HealthReport:
        """Execute comprehensive health check.

        Returns:
            HealthReport with detailed component status

        Example:
            @app.get("/health")
            async def health():
                report = await checker.check_health()
                return report.to_dict()
        """
        return await self._health_checker.check_health()

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    def mark_startup_complete(self) -> None:
        """Mark startup as complete.

        Can be called manually if startup checks are not needed.
        """
        self.startup_completed = True

    def reset_startup(self) -> None:
        """Reset startup state (for testing).

        Warning: Only use in tests, never in production.
        """
        self.startup_completed = False
        self.start_time = time.time()

    @property
    def uptime_seconds(self) -> float:
        """Get service uptime in seconds.

        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time


# =============================================================================
# Factory Functions
# =============================================================================


def create_production_health_checker(
    startup_timeout: float = 60.0,
    service_version: str = "1.0.0",
) -> ProductionHealthChecker:
    """Create a production health checker.

    Args:
        startup_timeout: Seconds to wait before considering startup failed
        service_version: Service version string

    Returns:
        Configured ProductionHealthChecker

    Example:
        checker = create_production_health_checker()
        checker.add_readiness_check("database", check_db)
    """
    return ProductionHealthChecker(
        startup_timeout=startup_timeout,
        service_version=service_version,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ProductionHealthChecker",
    "HealthCheckResponse",
    "HealthCheckType",
    "create_production_health_checker",
]
