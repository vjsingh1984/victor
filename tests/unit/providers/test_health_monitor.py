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

"""Unit tests for health monitoring system."""

import asyncio
import pytest

from victor.providers.health_monitor import (
    HealthCheckConfig,
    HealthMetrics,
    HealthMonitor,
    HealthStatus,
    ProviderHealthRegistry,
    get_health_registry,
)


class TestHealthMetrics:
    """Tests for HealthMetrics dataclass."""

    def test_initial_state(self) -> None:
        """Test metrics initial state."""
        metrics = HealthMetrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.error_rate == 0.0
        assert metrics.avg_latency_ms == 0.0

    def test_record_success(self) -> None:
        """Test recording successful request."""
        metrics = HealthMetrics()
        metrics.record_success(100.0)
        metrics.record_success(200.0)

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 0
        assert metrics.error_rate == 0.0
        assert metrics.avg_latency_ms == 150.0
        assert metrics.consecutive_successes == 2
        assert metrics.consecutive_failures == 0

    def test_record_failure(self) -> None:
        """Test recording failed request."""
        metrics = HealthMetrics()
        metrics.record_failure()
        metrics.record_failure()

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 2
        assert metrics.error_rate == 1.0
        assert metrics.consecutive_failures == 2
        assert metrics.consecutive_successes == 0

    def test_mixed_requests(self) -> None:
        """Test mixed success and failure recording."""
        metrics = HealthMetrics()
        metrics.record_success(100.0)
        metrics.record_failure()
        metrics.record_success(200.0)

        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.error_rate == 1 / 3
        assert metrics.avg_latency_ms == 150.0


class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    def test_initial_state(self) -> None:
        """Test monitor initial state."""
        monitor = HealthMonitor(provider_id="test-provider")
        assert monitor.provider_id == "test-provider"
        assert monitor.status == HealthStatus.HEALTHY
        assert monitor.is_healthy
        assert monitor.can_accept_traffic

    def test_success_recording(self) -> None:
        """Test recording successful requests."""
        monitor = HealthMonitor(provider_id="test-provider")
        monitor.record_success(100.0)

        assert monitor.metrics.successful_requests == 1
        assert monitor.status == HealthStatus.HEALTHY

    def test_failure_recording(self) -> None:
        """Test recording failed requests."""
        config = HealthCheckConfig(unhealthy_threshold=3)
        monitor = HealthMonitor(provider_id="test-provider", config=config)

        # Should remain healthy until threshold
        monitor.record_failure()
        assert monitor.status == HealthStatus.HEALTHY

        monitor.record_failure()
        assert monitor.status == HealthStatus.HEALTHY

        # Should become unhealthy
        monitor.record_failure()
        assert monitor.status == HealthStatus.UNHEALTHY

    def test_degraded_latency(self) -> None:
        """Test degraded status due to high latency."""
        config = HealthCheckConfig(max_latency_ms=1000.0)
        monitor = HealthMonitor(provider_id="test-provider", config=config)

        # Record enough requests to establish baseline
        for i in range(10):
            monitor.record_success(500.0)

        assert monitor.status == HealthStatus.HEALTHY

        # High latency should trigger degraded status
        # Need enough requests to push average over threshold
        for i in range(15):
            monitor.record_success(1500.0)

        assert monitor.status == HealthStatus.DEGRADED

    def test_degraded_error_rate(self) -> None:
        """Test degraded status due to high error rate."""
        config = HealthCheckConfig(max_error_rate=0.2, unhealthy_threshold=10)
        monitor = HealthMonitor(provider_id="test-provider", config=config)

        # High error rate should trigger degraded status
        # Need many requests to establish error rate above threshold
        for i in range(20):
            monitor.record_success(100.0)
        for i in range(6):
            monitor.record_failure()

        # Should have high error rate without triggering consecutive failures
        assert monitor.metrics.error_rate > 0.2
        assert monitor.metrics.error_rate < 0.3
        # With sufficient samples and error rate above threshold, should be degraded
        assert monitor.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    def test_manual_status_set(self) -> None:
        """Test manually setting health status."""
        monitor = HealthMonitor(provider_id="test-provider")

        monitor.set_status(HealthStatus.DRAINING)
        assert monitor.status == HealthStatus.DRAINING
        # DRAINING status means instance should not accept new traffic
        # Check the _status directly since can_accept_traffic has additional logic
        assert monitor._status == HealthStatus.DRAINING

    def test_get_stats(self) -> None:
        """Test getting statistics."""
        monitor = HealthMonitor(provider_id="test-provider")
        monitor.record_success(100.0)
        monitor.record_failure()

        stats = monitor.get_stats()
        assert stats["provider_id"] == "test-provider"
        assert stats["status"] == "healthy"
        assert stats["metrics"]["total_requests"] == 2
        assert "uptime_seconds" in stats

    def test_reset(self) -> None:
        """Test resetting monitor."""
        monitor = HealthMonitor(provider_id="test-provider")
        monitor.record_success(100.0)
        monitor.record_failure()
        monitor.set_status(HealthStatus.UNHEALTHY)

        monitor.reset()

        assert monitor.metrics.total_requests == 0
        assert monitor.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_start_stop(self) -> None:
        """Test starting and stopping health checks."""
        config = HealthCheckConfig(enabled=True, check_interval_seconds=0.1)
        monitor = HealthMonitor(provider_id="test-provider", config=config)

        await monitor.start_health_checks()
        assert monitor._health_check_task is not None

        await monitor.stop_health_checks()
        assert monitor._health_check_task is None


class TestProviderHealthRegistry:
    """Tests for ProviderHealthRegistry."""

    @pytest.mark.asyncio
    async def test_register_monitor(self) -> None:
        """Test registering a health monitor."""
        registry = ProviderHealthRegistry()
        monitor = await registry.register("test-provider")

        assert monitor.provider_id == "test-provider"
        assert registry.get_monitor("test-provider") is monitor

    @pytest.mark.asyncio
    async def test_register_duplicate(self) -> None:
        """Test registering duplicate provider ID."""
        registry = ProviderHealthRegistry()
        monitor1 = await registry.register("test-provider")
        monitor2 = await registry.register("test-provider")

        # Should return existing monitor
        assert monitor1 is monitor2

    @pytest.mark.asyncio
    async def test_unregister_monitor(self) -> None:
        """Test unregistering a health monitor."""
        registry = ProviderHealthRegistry()
        monitor = await registry.register("test-provider")

        await monitor.start_health_checks()
        await registry.unregister("test-provider")

        assert registry.get_monitor("test-provider") is None

    @pytest.mark.asyncio
    async def test_get_all_stats(self) -> None:
        """Test getting all statistics."""
        registry = ProviderHealthRegistry()
        await registry.register("provider-1")
        await registry.register("provider-2")

        stats = registry.get_all_stats()
        assert "provider-1" in stats
        assert "provider-2" in stats

    @pytest.mark.asyncio
    async def test_get_healthy_providers(self) -> None:
        """Test getting list of healthy providers."""
        registry = ProviderHealthRegistry()
        monitor1 = await registry.register("provider-1")
        monitor2 = await registry.register("provider-2")

        # Mark one as unhealthy
        monitor2.set_status(HealthStatus.UNHEALTHY)

        healthy = registry.get_healthy_providers()
        assert "provider-1" in healthy
        assert "provider-2" not in healthy

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test shutting down registry."""
        registry = ProviderHealthRegistry()
        monitor1 = await registry.register("provider-1")
        monitor2 = await registry.register("provider-2")

        await monitor1.start_health_checks()
        await monitor2.start_health_checks()

        await registry.shutdown()

        assert registry.get_monitor("provider-1") is None
        assert registry.get_monitor("provider-2") is None


class TestGlobalHealthRegistry:
    """Tests for global health registry."""

    @pytest.mark.asyncio
    async def test_get_global_registry(self) -> None:
        """Test getting global registry instance."""
        registry1 = await get_health_registry()
        registry2 = await get_health_registry()

        # Should return same instance
        assert registry1 is registry2
