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

"""Tests for Provider Health Check System.

Tests cover:
- Health status determination
- Single provider checks
- Multi-provider health reports
- Health history tracking
- Provider ranking
- Background monitoring
- Health change callbacks
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from victor.providers.health import (
    ProviderHealthChecker,
    HealthStatus,
    HealthCheckResult,
    ProviderHealthReport,
    get_health_checker,
    reset_health_checker,
    PROVIDER_HEALTH_CONFIG,
    DEFAULT_HEALTH_CONFIG,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected statuses are defined."""
        assert HealthStatus.HEALTHY
        assert HealthStatus.DEGRADED
        assert HealthStatus.UNHEALTHY
        assert HealthStatus.UNKNOWN

    def test_status_values(self):
        """Test status string values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_basic_creation(self):
        """Test creating a health check result."""
        result = HealthCheckResult(
            provider_name="anthropic",
            status=HealthStatus.HEALTHY,
            latency_ms=150.0,
        )

        assert result.provider_name == "anthropic"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms == 150.0
        assert result.error is None

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = HealthCheckResult(
            provider_name="openai",
            status=HealthStatus.DEGRADED,
            latency_ms=3500.0,
            error=None,
        )

        d = result.to_dict()

        assert d["provider_name"] == "openai"
        assert d["status"] == "degraded"
        assert d["latency_ms"] == 3500.0
        assert "timestamp" in d

    def test_error_included(self):
        """Test that errors are included in result."""
        result = HealthCheckResult(
            provider_name="test",
            status=HealthStatus.UNHEALTHY,
            latency_ms=0,
            error="Connection refused",
        )

        assert result.error == "Connection refused"
        assert result.to_dict()["error"] == "Connection refused"


class TestProviderHealthReport:
    """Tests for ProviderHealthReport dataclass."""

    def test_empty_report(self):
        """Test creating an empty report."""
        report = ProviderHealthReport()

        assert len(report.results) == 0
        assert report.healthy_count == 0
        assert report.degraded_count == 0
        assert report.unhealthy_count == 0

    def test_report_with_results(self):
        """Test report with multiple results."""
        report = ProviderHealthReport(
            results={
                "provider1": HealthCheckResult(
                    provider_name="provider1",
                    status=HealthStatus.HEALTHY,
                    latency_ms=100,
                ),
                "provider2": HealthCheckResult(
                    provider_name="provider2",
                    status=HealthStatus.DEGRADED,
                    latency_ms=5000,
                ),
            },
            healthy_count=1,
            degraded_count=1,
        )

        d = report.to_dict()

        assert d["summary"]["healthy"] == 1
        assert d["summary"]["degraded"] == 1
        assert d["summary"]["total"] == 2


class TestProviderHealthChecker:
    """Tests for ProviderHealthChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a fresh health checker."""
        reset_health_checker()
        return ProviderHealthChecker(history_size=10)

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = MagicMock()
        provider.health_check = AsyncMock(return_value=True)
        return provider

    def test_initialization(self, checker):
        """Test health checker initialization."""
        assert checker.history_size == 10
        assert len(checker._providers) == 0
        assert len(checker._history) == 0

    def test_register_provider(self, checker, mock_provider):
        """Test registering a provider."""
        checker.register_provider("test", mock_provider)

        assert "test" in checker._providers
        assert checker._providers["test"] is mock_provider

    def test_unregister_provider(self, checker, mock_provider):
        """Test unregistering a provider."""
        checker.register_provider("test", mock_provider)
        checker.unregister_provider("test")

        assert "test" not in checker._providers

    def test_get_config_known_provider(self, checker):
        """Test getting config for known provider."""
        config = checker._get_config("anthropic")

        assert "degraded_threshold_ms" in config
        assert "unhealthy_threshold_ms" in config
        assert config["timeout_seconds"] > 0

    def test_get_config_unknown_provider(self, checker):
        """Test getting config for unknown provider."""
        config = checker._get_config("unknown_provider")

        # Should return default config
        assert config == DEFAULT_HEALTH_CONFIG

    def test_determine_status_healthy(self, checker):
        """Test status determination for healthy latency."""
        config = {"degraded_threshold_ms": 5000, "unhealthy_threshold_ms": 15000}

        status = checker._determine_status(100, config)

        assert status == HealthStatus.HEALTHY

    def test_determine_status_degraded(self, checker):
        """Test status determination for degraded latency."""
        config = {"degraded_threshold_ms": 5000, "unhealthy_threshold_ms": 15000}

        status = checker._determine_status(7000, config)

        assert status == HealthStatus.DEGRADED

    def test_determine_status_unhealthy_latency(self, checker):
        """Test status determination for unhealthy latency."""
        config = {"degraded_threshold_ms": 5000, "unhealthy_threshold_ms": 15000}

        status = checker._determine_status(20000, config)

        assert status == HealthStatus.UNHEALTHY

    def test_determine_status_with_error(self, checker):
        """Test status determination with error."""
        config = {"degraded_threshold_ms": 5000, "unhealthy_threshold_ms": 15000}

        status = checker._determine_status(100, config, error="Connection failed")

        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_provider_healthy(self, checker, mock_provider):
        """Test checking a healthy provider."""
        checker.register_provider("test", mock_provider)

        result = await checker.check_provider("test")

        assert result.status == HealthStatus.HEALTHY
        assert result.error is None
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_check_provider_unregistered(self, checker):
        """Test checking an unregistered provider."""
        result = await checker.check_provider("unknown")

        assert result.status == HealthStatus.UNKNOWN
        assert "not registered" in result.error

    @pytest.mark.asyncio
    async def test_check_provider_timeout(self, checker):
        """Test checking a provider that times out."""
        slow_provider = MagicMock()

        async def slow_health_check():
            await asyncio.sleep(100)

        slow_provider.health_check = slow_health_check

        checker.register_provider("slow", slow_provider)

        # Use custom config with short timeout
        checker._configs["slow"] = {
            "timeout_seconds": 0.1,
            "degraded_threshold_ms": 5000,
            "unhealthy_threshold_ms": 15000,
        }

        result = await checker.check_provider("slow")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_check_provider_error(self, checker):
        """Test checking a provider that raises an error."""
        error_provider = MagicMock()
        error_provider.health_check = AsyncMock(side_effect=ConnectionError("Network error"))

        checker.register_provider("error", error_provider)

        result = await checker.check_provider("error")

        assert result.status == HealthStatus.UNHEALTHY
        assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_check_all(self, checker):
        """Test checking all providers."""
        # Create multiple mock providers
        healthy_provider = MagicMock()
        healthy_provider.health_check = AsyncMock(return_value=True)

        slow_provider = MagicMock()
        slow_provider.health_check = AsyncMock(side_effect=lambda: asyncio.sleep(0.01))

        checker.register_provider("healthy", healthy_provider)
        checker.register_provider("slow", slow_provider)

        report = await checker.check_all()

        assert len(report.results) == 2
        assert report.healthy_count + report.degraded_count + report.unhealthy_count == 2

    @pytest.mark.asyncio
    async def test_check_all_empty(self, checker):
        """Test checking with no providers."""
        report = await checker.check_all()

        assert len(report.results) == 0

    def test_get_latest_status(self, checker):
        """Test getting latest status."""
        result = HealthCheckResult(
            provider_name="test",
            status=HealthStatus.HEALTHY,
            latency_ms=100,
        )
        checker._record_result(result)

        status = checker.get_latest_status("test")

        assert status == HealthStatus.HEALTHY

    def test_get_latest_status_unknown(self, checker):
        """Test getting status for unchecked provider."""
        status = checker.get_latest_status("never_checked")

        assert status == HealthStatus.UNKNOWN

    def test_get_healthy_providers(self, checker):
        """Test getting healthy providers sorted by latency."""
        # Record results with different latencies
        checker._record_result(
            HealthCheckResult(
                provider_name="slow_healthy",
                status=HealthStatus.HEALTHY,
                latency_ms=1000,
            )
        )
        checker._record_result(
            HealthCheckResult(
                provider_name="fast_healthy",
                status=HealthStatus.HEALTHY,
                latency_ms=100,
            )
        )
        checker._record_result(
            HealthCheckResult(
                provider_name="unhealthy",
                status=HealthStatus.UNHEALTHY,
                latency_ms=50,
            )
        )

        healthy = checker.get_healthy_providers()

        # Fast healthy should be first
        assert healthy == ["fast_healthy", "slow_healthy"]

    def test_get_available_providers(self, checker):
        """Test getting available providers (healthy + degraded)."""
        checker._record_result(
            HealthCheckResult(
                provider_name="healthy",
                status=HealthStatus.HEALTHY,
                latency_ms=100,
            )
        )
        checker._record_result(
            HealthCheckResult(
                provider_name="degraded",
                status=HealthStatus.DEGRADED,
                latency_ms=5000,
            )
        )
        checker._record_result(
            HealthCheckResult(
                provider_name="unhealthy",
                status=HealthStatus.UNHEALTHY,
                latency_ms=50,
            )
        )

        available = checker.get_available_providers()

        assert len(available) == 2
        assert "unhealthy" not in available
        # Healthy should come before degraded
        assert available[0] == "healthy"

    def test_health_history_tracking(self, checker):
        """Test that health history is tracked."""
        for i in range(5):
            checker._record_result(
                HealthCheckResult(
                    provider_name="test",
                    status=HealthStatus.HEALTHY,
                    latency_ms=100 + i,
                )
            )

        history = checker.get_provider_history("test")

        assert len(history) == 5

    def test_health_history_limit(self, checker):
        """Test that health history respects limit."""
        for i in range(20):  # More than history_size (10)
            checker._record_result(
                HealthCheckResult(
                    provider_name="test",
                    status=HealthStatus.HEALTHY,
                    latency_ms=100 + i,
                )
            )

        history = checker.get_provider_history("test")

        assert len(history) == 10  # Limited to history_size

    def test_calculate_uptime(self, checker):
        """Test uptime calculation."""
        # 3 healthy, 1 unhealthy
        checker._record_result(
            HealthCheckResult(provider_name="test", status=HealthStatus.HEALTHY, latency_ms=100)
        )
        checker._record_result(
            HealthCheckResult(provider_name="test", status=HealthStatus.HEALTHY, latency_ms=100)
        )
        checker._record_result(
            HealthCheckResult(provider_name="test", status=HealthStatus.HEALTHY, latency_ms=100)
        )
        checker._record_result(
            HealthCheckResult(provider_name="test", status=HealthStatus.UNHEALTHY, latency_ms=0)
        )

        uptime = checker.calculate_uptime("test")

        assert uptime == 75.0  # 3/4 = 75%

    def test_calculate_uptime_empty(self, checker):
        """Test uptime calculation with no history."""
        uptime = checker.calculate_uptime("never_checked")

        assert uptime == 0.0

    def test_health_change_callback(self, checker):
        """Test health change callback."""
        callback_calls = []

        def callback(name, old_status, new_status):
            callback_calls.append((name, old_status, new_status))

        checker.add_health_change_callback(callback)

        # First result - unknown -> healthy
        checker._record_result(
            HealthCheckResult(
                provider_name="test",
                status=HealthStatus.HEALTHY,
                latency_ms=100,
            )
        )

        # Second result - healthy -> unhealthy
        checker._record_result(
            HealthCheckResult(
                provider_name="test",
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
            )
        )

        assert len(callback_calls) == 2
        assert callback_calls[0] == ("test", HealthStatus.UNKNOWN, HealthStatus.HEALTHY)
        assert callback_calls[1] == ("test", HealthStatus.HEALTHY, HealthStatus.UNHEALTHY)

    def test_get_stats(self, checker, mock_provider):
        """Test getting checker statistics."""
        checker.register_provider("test", mock_provider)
        checker._record_result(
            HealthCheckResult(
                provider_name="test",
                status=HealthStatus.HEALTHY,
                latency_ms=100,
            )
        )

        stats = checker.get_stats()

        assert "registered_providers" in stats
        assert "checked_providers" in stats
        assert "healthy_count" in stats
        assert "monitoring_active" in stats
        assert "uptime_by_provider" in stats

    @pytest.mark.asyncio
    async def test_background_monitoring_start_stop(self, checker, mock_provider):
        """Test starting and stopping background monitoring."""
        checker.register_provider("test", mock_provider)

        # Start monitoring with short interval
        await checker.start_monitoring(interval_seconds=0.05)

        # Verify monitoring started
        assert checker._monitoring_task is not None

        # Wait a bit for some checks
        await asyncio.sleep(0.1)

        # Stop monitoring
        await checker.stop_monitoring()

        assert checker._monitoring_task is None


class TestGlobalHealthChecker:
    """Tests for global health checker singleton."""

    def test_get_health_checker_singleton(self):
        """Test that get_health_checker returns singleton."""
        reset_health_checker()

        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2

    def test_reset_health_checker(self):
        """Test resetting the global checker."""
        checker1 = get_health_checker()
        reset_health_checker()
        checker2 = get_health_checker()

        assert checker1 is not checker2


class TestProviderHealthConfig:
    """Tests for provider health configuration."""

    def test_known_providers_have_config(self):
        """Test that known providers have health config."""
        expected_providers = ["anthropic", "openai", "ollama", "google", "groq"]

        for provider in expected_providers:
            assert provider in PROVIDER_HEALTH_CONFIG

    def test_config_has_required_fields(self):
        """Test that configs have required fields."""
        for name, config in PROVIDER_HEALTH_CONFIG.items():
            assert "degraded_threshold_ms" in config, f"{name} missing degraded_threshold_ms"
            assert "unhealthy_threshold_ms" in config, f"{name} missing unhealthy_threshold_ms"
            assert "timeout_seconds" in config, f"{name} missing timeout_seconds"

    def test_thresholds_are_ordered(self):
        """Test that degraded < unhealthy thresholds."""
        for name, config in PROVIDER_HEALTH_CONFIG.items():
            assert (
                config["degraded_threshold_ms"] < config["unhealthy_threshold_ms"]
            ), f"{name}: degraded threshold should be less than unhealthy"
