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

"""Provider Health Check System.

Provides proactive health monitoring for LLM providers:
- Lightweight health check probes
- Aggregated health status
- Provider ranking by health
- Background health monitoring
- Integration with circuit breakers

Usage:
    from victor.providers.health import ProviderHealthChecker, HealthStatus

    # Create health checker
    checker = ProviderHealthChecker()

    # Check single provider
    status = await checker.check_provider("anthropic")

    # Check all providers
    report = await checker.check_all()

    # Get healthy providers ranked by latency
    healthy = checker.get_healthy_providers()

    # Start background monitoring
    await checker.start_monitoring(interval_seconds=60)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from collections.abc import Callable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"  # Provider responding normally
    DEGRADED = "degraded"  # Provider slow but functional
    UNHEALTHY = "unhealthy"  # Provider failing or unavailable
    UNKNOWN = "unknown"  # Not yet checked


@dataclass
class HealthCheckResult:
    """Result of a single health check.

    Attributes:
        provider_name: Name of the provider checked
        status: Health status
        latency_ms: Response latency in milliseconds
        error: Error message if unhealthy
        timestamp: Time of check
        model: Model used for check (if applicable)
        details: Additional check details
    """

    provider_name: str
    status: HealthStatus
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    model: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider_name": self.provider_name,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "details": self.details,
        }


@dataclass
class ProviderHealthReport:
    """Aggregated health report for all providers.

    Attributes:
        results: Individual provider results
        healthy_count: Number of healthy providers
        degraded_count: Number of degraded providers
        unhealthy_count: Number of unhealthy providers
        timestamp: Report generation time
        total_check_time_ms: Time to complete all checks
    """

    results: dict[str, HealthCheckResult] = field(default_factory=dict)
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    total_check_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": {name: r.to_dict() for name, r in self.results.items()},
            "summary": {
                "healthy": self.healthy_count,
                "degraded": self.degraded_count,
                "unhealthy": self.unhealthy_count,
                "total": len(self.results),
            },
            "timestamp": self.timestamp.isoformat(),
            "total_check_time_ms": round(self.total_check_time_ms, 2),
        }


# Default health check configuration per provider
PROVIDER_HEALTH_CONFIG = {
    "anthropic": {
        "check_endpoint": True,
        "degraded_threshold_ms": 5000,
        "unhealthy_threshold_ms": 15000,
        "timeout_seconds": 20,
    },
    "openai": {
        "check_endpoint": True,
        "degraded_threshold_ms": 3000,
        "unhealthy_threshold_ms": 10000,
        "timeout_seconds": 15,
    },
    "ollama": {
        "check_endpoint": True,
        "degraded_threshold_ms": 1000,
        "unhealthy_threshold_ms": 5000,
        "timeout_seconds": 10,
    },
    "lmstudio": {
        "check_endpoint": True,
        "degraded_threshold_ms": 1000,
        "unhealthy_threshold_ms": 5000,
        "timeout_seconds": 10,
    },
    "google": {
        "check_endpoint": True,
        "degraded_threshold_ms": 4000,
        "unhealthy_threshold_ms": 12000,
        "timeout_seconds": 15,
    },
    "xai": {
        "check_endpoint": True,
        "degraded_threshold_ms": 5000,
        "unhealthy_threshold_ms": 15000,
        "timeout_seconds": 20,
    },
    "groq": {
        "check_endpoint": True,
        "degraded_threshold_ms": 2000,
        "unhealthy_threshold_ms": 8000,
        "timeout_seconds": 12,
    },
}

# Default config for unknown providers
DEFAULT_HEALTH_CONFIG = {
    "check_endpoint": True,
    "degraded_threshold_ms": 5000,
    "unhealthy_threshold_ms": 15000,
    "timeout_seconds": 20,
}


class ProviderHealthChecker:
    """Health checker for LLM providers.

    Features:
    - Lightweight health probes (minimal token usage)
    - Configurable thresholds per provider
    - Parallel checking for efficiency
    - Health history tracking
    - Integration with circuit breakers

    Usage:
        checker = ProviderHealthChecker()

        # Check single provider
        result = await checker.check_provider("anthropic")

        # Check all registered providers
        report = await checker.check_all()

        # Get ranked healthy providers
        healthy = checker.get_healthy_providers()
    """

    def __init__(
        self,
        history_size: int = 100,
        custom_configs: Optional[dict[str, dict[str, Any]]] = None,
    ):
        """Initialize health checker.

        Args:
            history_size: Number of health check results to retain per provider
            custom_configs: Custom health check configurations per provider
        """
        self.history_size = history_size
        self._configs = {**PROVIDER_HEALTH_CONFIG, **(custom_configs or {})}

        # Health check history per provider
        self._history: dict[str, list[HealthCheckResult]] = {}

        # Latest result per provider
        self._latest: dict[str, HealthCheckResult] = {}

        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = asyncio.Event()

        # Registered provider instances
        self._providers: dict[str, Any] = {}

        # Callbacks for health changes
        self._on_health_change: list[Callable[[str, HealthStatus, HealthStatus], None]] = []

        logger.debug(f"ProviderHealthChecker initialized with history_size={history_size}")

    def register_provider(self, name: str, provider: Any) -> None:
        """Register a provider for health checking.

        Args:
            name: Provider name
            provider: Provider instance
        """
        self._providers[name] = provider
        logger.info(f"Registered provider for health checking: {name}")

    def unregister_provider(self, name: str) -> None:
        """Unregister a provider.

        Args:
            name: Provider name to unregister
        """
        if name in self._providers:
            del self._providers[name]
            logger.info(f"Unregistered provider: {name}")

    def add_health_change_callback(
        self, callback: Callable[[str, HealthStatus, HealthStatus], None]
    ) -> None:
        """Add callback for health status changes.

        Args:
            callback: Function(provider_name, old_status, new_status)
        """
        self._on_health_change.append(callback)

    def _get_config(self, provider_name: str) -> dict[str, Any]:
        """Get health check config for provider."""
        return self._configs.get(provider_name, DEFAULT_HEALTH_CONFIG)

    def _determine_status(
        self,
        latency_ms: float,
        config: dict[str, Any],
        error: Optional[str] = None,
    ) -> HealthStatus:
        """Determine health status from latency and config.

        Args:
            latency_ms: Response latency
            config: Provider health config
            error: Optional error message

        Returns:
            HealthStatus based on thresholds
        """
        if error:
            return HealthStatus.UNHEALTHY

        if latency_ms > config.get("unhealthy_threshold_ms", 15000):
            return HealthStatus.UNHEALTHY
        elif latency_ms > config.get("degraded_threshold_ms", 5000):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _record_result(self, result: HealthCheckResult) -> None:
        """Record health check result.

        Args:
            result: Health check result to record
        """
        provider = result.provider_name

        # Check for status change
        old_status = self._latest.get(
            provider, HealthCheckResult(provider_name=provider, status=HealthStatus.UNKNOWN)
        ).status

        if old_status != result.status:
            logger.info(
                f"Provider '{provider}' health changed: "
                f"{old_status.value} -> {result.status.value}"
            )
            for callback in self._on_health_change:
                try:
                    callback(provider, old_status, result.status)
                except Exception as e:
                    logger.warning(f"Health change callback failed: {e}")

        # Update latest
        self._latest[provider] = result

        # Update history
        if provider not in self._history:
            self._history[provider] = []

        self._history[provider].append(result)

        # Trim history
        if len(self._history[provider]) > self.history_size:
            self._history[provider] = self._history[provider][-self.history_size :]

    async def check_provider(
        self,
        provider_name: str,
        provider: Optional[Any] = None,
    ) -> HealthCheckResult:
        """Check health of a single provider.

        Args:
            provider_name: Name of the provider
            provider: Optional provider instance (uses registered if not provided)

        Returns:
            HealthCheckResult with status and latency
        """
        # Use provided or registered provider
        provider = provider or self._providers.get(provider_name)

        if provider is None:
            return HealthCheckResult(
                provider_name=provider_name,
                status=HealthStatus.UNKNOWN,
                error="Provider not registered",
            )

        config = self._get_config(provider_name)
        timeout = config.get("timeout_seconds", 20)

        start_time = time.perf_counter()
        error = None

        try:
            # Use a lightweight health check if available
            if hasattr(provider, "health_check"):
                await asyncio.wait_for(
                    provider.health_check(),
                    timeout=timeout,
                )
            elif hasattr(provider, "list_models"):
                # Try listing models as a health check
                await asyncio.wait_for(
                    provider.list_models(),
                    timeout=timeout,
                )
            elif hasattr(provider, "chat"):
                # Minimal chat request (provider should handle this efficiently)
                from victor.providers.base import Message

                await asyncio.wait_for(
                    provider.chat(
                        messages=[Message(role="user", content="Hi")],
                        model=getattr(provider, "default_model", None),
                        max_tokens=1,
                    ),
                    timeout=timeout,
                )
            else:
                # No health check method available
                error = "No health check method available"

        except asyncio.TimeoutError:
            error = f"Timeout after {timeout}s"
        except Exception as e:
            error = str(e)

        latency_ms = (time.perf_counter() - start_time) * 1000
        status = self._determine_status(latency_ms, config, error)

        result = HealthCheckResult(
            provider_name=provider_name,
            status=status,
            latency_ms=latency_ms,
            error=error,
            details={
                "timeout_seconds": timeout,
                "threshold_degraded_ms": config.get("degraded_threshold_ms"),
                "threshold_unhealthy_ms": config.get("unhealthy_threshold_ms"),
            },
        )

        self._record_result(result)
        return result

    async def check_all(
        self,
        providers: Optional[dict[str, Any]] = None,
    ) -> ProviderHealthReport:
        """Check health of all registered providers in parallel.

        Args:
            providers: Optional dict of providers (uses registered if not provided)

        Returns:
            ProviderHealthReport with all results
        """
        providers = providers or self._providers

        if not providers:
            return ProviderHealthReport()

        start_time = time.perf_counter()

        # Run checks in parallel
        tasks = [self.check_provider(name, provider) for name, provider in providers.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build report
        report = ProviderHealthReport()
        report.total_check_time_ms = (time.perf_counter() - start_time) * 1000

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Health check failed with exception: {result}")
                continue

            # At this point, result must be a HealthCheckResult
            result_checked: HealthCheckResult = result  # type: ignore[assignment]
            report.results[result_checked.provider_name] = result_checked

            if result_checked.status == HealthStatus.HEALTHY:
                report.healthy_count += 1
            elif result_checked.status == HealthStatus.DEGRADED:
                report.degraded_count += 1
            else:
                report.unhealthy_count += 1

        return report

    def get_latest_status(self, provider_name: str) -> HealthStatus:
        """Get latest health status for a provider.

        Args:
            provider_name: Provider name

        Returns:
            Latest HealthStatus or UNKNOWN if not checked
        """
        result = self._latest.get(provider_name)
        return result.status if result else HealthStatus.UNKNOWN

    def get_healthy_providers(self) -> list[str]:
        """Get list of healthy providers sorted by latency (fastest first).

        Returns:
            List of provider names that are healthy, sorted by latency
        """
        healthy = [
            (name, result)
            for name, result in self._latest.items()
            if result.status == HealthStatus.HEALTHY
        ]
        # Sort by latency
        healthy.sort(key=lambda x: x[1].latency_ms)
        return [name for name, _ in healthy]

    def get_available_providers(self) -> list[str]:
        """Get list of available providers (healthy or degraded).

        Returns:
            List of provider names that are operational
        """
        available = [
            (name, result)
            for name, result in self._latest.items()
            if result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        ]
        # Sort: healthy first, then by latency
        available.sort(
            key=lambda x: (
                0 if x[1].status == HealthStatus.HEALTHY else 1,
                x[1].latency_ms,
            )
        )
        return [name for name, _ in available]

    def get_provider_history(
        self,
        provider_name: str,
        limit: Optional[int] = None,
    ) -> list[HealthCheckResult]:
        """Get health check history for a provider.

        Args:
            provider_name: Provider name
            limit: Maximum results to return (most recent)

        Returns:
            List of HealthCheckResult in chronological order
        """
        history = self._history.get(provider_name, [])
        if limit:
            return history[-limit:]
        return history

    def calculate_uptime(
        self,
        provider_name: str,
        window_size: Optional[int] = None,
    ) -> float:
        """Calculate uptime percentage from history.

        Args:
            provider_name: Provider name
            window_size: Number of recent checks to consider

        Returns:
            Uptime as percentage (0.0 - 100.0)
        """
        history = self.get_provider_history(provider_name, window_size)
        if not history:
            return 0.0

        healthy_count = sum(
            1 for r in history if r.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        )
        return (healthy_count / len(history)) * 100.0

    def get_stats(self) -> dict[str, Any]:
        """Get health checker statistics.

        Returns:
            Dictionary with checker statistics
        """
        return {
            "registered_providers": list(self._providers.keys()),
            "checked_providers": list(self._latest.keys()),
            "healthy_count": len(self.get_healthy_providers()),
            "available_count": len(self.get_available_providers()),
            "total_checks": sum(len(h) for h in self._history.values()),
            "monitoring_active": self._monitoring_task is not None
            and not self._monitoring_task.done(),
            "uptime_by_provider": {
                name: self.calculate_uptime(name) for name in self._latest.keys()
            },
        }

    async def start_monitoring(
        self,
        interval_seconds: float = 60.0,
        providers: Optional[dict[str, Any]] = None,
    ) -> None:
        """Start background health monitoring.

        Args:
            interval_seconds: Check interval in seconds
            providers: Optional providers to monitor (uses registered if not provided)
        """
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already active")
            return

        self._stop_monitoring.clear()

        async def _monitor():
            logger.info(f"Starting health monitoring (interval={interval_seconds}s)")
            while not self._stop_monitoring.is_set():
                try:
                    await self.check_all(providers)
                except Exception as e:
                    logger.warning(f"Health monitoring error: {e}")

                # Wait for interval or stop signal
                try:
                    await asyncio.wait_for(
                        self._stop_monitoring.wait(),
                        timeout=interval_seconds,
                    )
                    break  # Stop signal received
                except asyncio.TimeoutError:
                    pass  # Continue monitoring

            logger.info("Health monitoring stopped")

        self._monitoring_task = asyncio.create_task(_monitor())

    async def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        if not self._monitoring_task:
            return

        self._stop_monitoring.set()

        try:
            await asyncio.wait_for(self._monitoring_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._monitoring_task.cancel()

        self._monitoring_task = None
        logger.info("Health monitoring stopped")


# Singleton instance for global access
_health_checker: Optional[ProviderHealthChecker] = None


def get_health_checker() -> ProviderHealthChecker:
    """Get the global health checker instance.

    Returns:
        Global ProviderHealthChecker singleton
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = ProviderHealthChecker()
    return _health_checker


def reset_health_checker() -> None:
    """Reset the global health checker (mainly for testing)."""
    global _health_checker
    if _health_checker and _health_checker._monitoring_task:
        # Don't await in sync function - just set the stop flag
        _health_checker._stop_monitoring.set()
    _health_checker = None
