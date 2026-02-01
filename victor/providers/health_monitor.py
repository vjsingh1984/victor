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

"""Health monitoring system for provider instances.

This module provides real-time health monitoring for LLM providers in a pool,
tracking metrics like latency, error rates, and request counts to inform
load balancing decisions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of a provider instance."""

    HEALTHY = "healthy"  # Provider is responding normally
    DEGRADED = "degraded"  # Provider is slow but functional
    UNHEALTHY = "unhealthy"  # Provider is failing or non-responsive
    DRAINING = "draining"  # Provider is being removed from pool


@dataclass
class HealthMetrics:
    """Metrics tracked for health assessment.

    Attributes:
        total_requests: Total number of requests sent
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        total_latency_ms: Cumulative latency in milliseconds
        last_request_time: Timestamp of last request
        last_success_time: Timestamp of last successful request
        last_failure_time: Timestamp of last failure
        consecutive_failures: Current streak of failures
        consecutive_successes: Current streak of successes
        error_rate: Rolling error rate (0-1)
        avg_latency_ms: Rolling average latency
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_request_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request.

        Args:
            latency_ms: Request latency in milliseconds
        """
        now = time.time()
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.last_request_time = now
        self.last_success_time = now
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self._update_averages()

    def record_failure(self) -> None:
        """Record a failed request."""
        now = time.time()
        self.total_requests += 1
        self.failed_requests += 1
        self.last_request_time = now
        self.last_failure_time = now
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self._update_averages()

    def _update_averages(self) -> None:
        """Update rolling averages."""
        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests
        if self.successful_requests > 0:
            self.avg_latency_ms = self.total_latency_ms / self.successful_requests

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics.

        Returns:
            Dictionary of health metrics
        """
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_request_time": (
                datetime.fromtimestamp(self.last_request_time, tz=timezone.utc).isoformat()
                if self.last_request_time
                else None
            ),
            "last_success_time": (
                datetime.fromtimestamp(self.last_success_time, tz=timezone.utc).isoformat()
                if self.last_success_time
                else None
            ),
            "last_failure_time": (
                datetime.fromtimestamp(self.last_failure_time, tz=timezone.utc).isoformat()
                if self.last_failure_time
                else None
            ),
        }


@dataclass
class HealthCheckConfig:
    """Configuration for health checks.

    Attributes:
        enabled: Whether health checking is enabled
        check_interval_seconds: How often to run health checks
        timeout_seconds: Timeout for health check requests
        unhealthy_threshold: Consecutive failures before marking unhealthy
        healthy_threshold: Consecutive successes before marking healthy
        max_latency_ms: Maximum acceptable latency before degraded status
        max_error_rate: Maximum acceptable error rate (0-1)
    """

    enabled: bool = True
    check_interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    max_latency_ms: float = 5000.0
    max_error_rate: float = 0.1


class HealthMonitor:
    """Monitors health of provider instances.

    Tracks request metrics and runs periodic health checks to determine
    provider health status for load balancing decisions.

    Example:
        monitor = HealthMonitor(
            provider_id="anthropic-1",
            config=HealthCheckConfig(max_latency_ms=3000)
        )

        # Record request results
        monitor.record_success(latency_ms=250)
        monitor.record_failure()

        # Get current status
        status = monitor.get_status()
        if status == HealthStatus.HEALTHY:
            print("Provider is healthy")
    """

    def __init__(
        self,
        provider_id: str,
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize health monitor.

        Args:
            provider_id: Unique identifier for the provider instance
            config: Health check configuration
        """
        self.provider_id = provider_id
        self.config = config or HealthCheckConfig()
        self.metrics = HealthMetrics()
        self._status = HealthStatus.HEALTHY
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._start_time = time.time()

    @property
    def status(self) -> HealthStatus:
        """Get current health status."""
        return self._status

    @property
    def is_healthy(self) -> bool:
        """Check if provider is healthy (not unhealthy or draining)."""
        return self._status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    @property
    def can_accept_traffic(self) -> bool:
        """Check if provider can accept new traffic."""
        return self._status != HealthStatus.UNHEALTHY

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request.

        Args:
            latency_ms: Request latency in milliseconds
        """
        self.metrics.record_success(latency_ms)
        self._update_status()

    def record_failure(self) -> None:
        """Record a failed request."""
        self.metrics.record_failure()
        self._update_status()

    def _update_status(self) -> None:
        """Update health status based on metrics.

        Status determination logic:
        - UNHEALTHY: Consecutive failures >= unhealthy_threshold
        - DEGRADED: High latency or high error rate
        - HEALTHY: Normal operation
        """
        metrics = self.metrics

        # Check for unhealthy condition
        if metrics.consecutive_failures >= self.config.unhealthy_threshold:
            self._status = HealthStatus.UNHEALTHY
            logger.warning(
                f"Provider {self.provider_id} marked UNHEALTHY: "
                f"{metrics.consecutive_failures} consecutive failures"
            )
            return

        # Check for degraded condition
        if metrics.total_requests >= 10:  # Need minimum samples
            if metrics.avg_latency_ms > self.config.max_latency_ms:
                self._status = HealthStatus.DEGRADED
                logger.warning(
                    f"Provider {self.provider_id} marked DEGRADED: "
                    f"latency {metrics.avg_latency_ms:.0f}ms > {self.config.max_latency_ms}ms"
                )
                return

            if metrics.error_rate > self.config.max_error_rate:
                self._status = HealthStatus.DEGRADED
                logger.warning(
                    f"Provider {self.provider_id} marked DEGRADED: "
                    f"error rate {metrics.error_rate:.2%} > {self.config.max_error_rate:.2%}"
                )
                return

        # Default to healthy
        if self._status != HealthStatus.DRAINING:
            self._status = HealthStatus.HEALTHY

    async def start_health_checks(self) -> None:
        """Start periodic health checks.

        Creates a background task that runs health checks at the configured interval.
        """
        if not self.config.enabled:
            logger.info(f"Health checks disabled for provider {self.provider_id}")
            return

        if self._health_check_task is not None:
            logger.warning(f"Health checks already running for provider {self.provider_id}")
            return

        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Started health checks for provider {self.provider_id}")

    async def stop_health_checks(self) -> None:
        """Stop periodic health checks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info(f"Stopped health checks for provider {self.provider_id}")

    async def _health_check_loop(self) -> None:
        """Run periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.config.check_interval_seconds)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for provider {self.provider_id}: {e}")

    async def _perform_health_check(self) -> None:
        """Perform a single health check.

        This is a placeholder implementation. Subclasses or the provider pool
        should override this to perform actual health checks (e.g., a simple
        API call to test responsiveness).
        """
        # Default implementation: just update status based on metrics
        self._update_status()

    def set_status(self, status: HealthStatus) -> None:
        """Manually set health status.

        Args:
            status: New health status
        """
        old_status = self._status
        self._status = status
        logger.info(f"Provider {self.provider_id} status: {old_status.value} -> {status.value}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive health statistics.

        Returns:
            Dictionary containing health status and metrics
        """
        uptime_seconds = time.time() - self._start_time

        return {
            "provider_id": self.provider_id,
            "status": self._status.value,
            "uptime_seconds": round(uptime_seconds, 2),
            "metrics": self.metrics.get_stats(),
            "config": {
                "enabled": self.config.enabled,
                "max_latency_ms": self.config.max_latency_ms,
                "max_error_rate": self.config.max_error_rate,
                "unhealthy_threshold": self.config.unhealthy_threshold,
            },
        }

    def reset(self) -> None:
        """Reset health metrics and status."""
        self.metrics = HealthMetrics()
        self._status = HealthStatus.HEALTHY
        self._start_time = time.time()
        logger.info(f"Reset health monitor for provider {self.provider_id}")


class ProviderHealthRegistry:
    """Registry for managing health monitors across all providers.

    Provides centralized access to health status for all provider instances
    in the pool.
    """

    def __init__(self) -> None:
        """Initialize health registry."""
        self._monitors: dict[str, HealthMonitor] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        provider_id: str,
        config: Optional[HealthCheckConfig] = None,
    ) -> HealthMonitor:
        """Register a new provider health monitor.

        Args:
            provider_id: Unique identifier for the provider
            config: Health check configuration

        Returns:
            HealthMonitor instance
        """
        async with self._lock:
            if provider_id in self._monitors:
                logger.warning(f"Health monitor already exists for {provider_id}")
                return self._monitors[provider_id]

            monitor = HealthMonitor(provider_id=provider_id, config=config)
            self._monitors[provider_id] = monitor
            logger.info(f"Registered health monitor for provider {provider_id}")
            return monitor

    async def unregister(self, provider_id: str) -> None:
        """Unregister a provider health monitor.

        Args:
            provider_id: Provider to unregister
        """
        async with self._lock:
            monitor = self._monitors.pop(provider_id, None)
            if monitor:
                await monitor.stop_health_checks()
                logger.info(f"Unregistered health monitor for provider {provider_id}")

    def get_monitor(self, provider_id: str) -> Optional[HealthMonitor]:
        """Get health monitor for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            HealthMonitor instance or None
        """
        return self._monitors.get(provider_id)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all monitored providers.

        Returns:
            Dictionary mapping provider_id to health stats
        """
        return {pid: monitor.get_stats() for pid, monitor in self._monitors.items()}

    def get_healthy_providers(self) -> list[str]:
        """Get list of healthy provider IDs.

        Returns:
            List of provider IDs that can accept traffic
        """
        return [pid for pid, monitor in self._monitors.items() if monitor.can_accept_traffic]

    async def shutdown(self) -> None:
        """Shutdown all health monitors."""
        async with self._lock:
            tasks = [monitor.stop_health_checks() for monitor in self._monitors.values()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._monitors.clear()
            logger.info("Shutdown all health monitors")


# Global health registry instance
_global_registry: Optional[ProviderHealthRegistry] = None
_registry_lock = asyncio.Lock()


async def get_health_registry() -> ProviderHealthRegistry:
    """Get or create global health registry.

    Returns:
        Global ProviderHealthRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        async with _registry_lock:
            if _global_registry is None:
                _global_registry = ProviderHealthRegistry()

    return _global_registry
