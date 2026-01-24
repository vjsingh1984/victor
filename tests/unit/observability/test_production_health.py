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

"""Tests for production health checker."""

import pytest

from victor.observability.production_health import (
    ProductionHealthChecker,
    create_production_health_checker,
    HealthCheckResponse,
    HealthCheckType,
)


@pytest.fixture
def health_checker():
    """Create a fresh health checker for each test."""
    checker = ProductionHealthChecker()
    yield checker


class TestProductionHealthChecker:
    """Test ProductionHealthChecker functionality."""

    @pytest.mark.asyncio
    async def test_liveness_all_passing(self, health_checker):
        """Test liveness probe with all checks passing."""
        health_checker.add_liveness_check("process", lambda: True)
        health_checker.add_liveness_check("memory", lambda: True)

        response = await health_checker.liveness()
        assert response.is_healthy
        assert response.status == "healthy"
        assert "process" in response.checks
        assert "memory" in response.checks

    @pytest.mark.asyncio
    async def test_liveness_one_failing(self, health_checker):
        """Test liveness probe with one check failing."""
        health_checker.add_liveness_check("process", lambda: True)
        health_checker.add_liveness_check("memory", lambda: False)

        response = await health_checker.liveness()
        assert not response.is_healthy
        assert response.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_liveness_exception(self, health_checker):
        """Test liveness probe with exception."""

        def failing_check():
            raise Exception("Check failed")

        health_checker.add_liveness_check("failing", failing_check)

        response = await health_checker.liveness()
        assert not response.is_healthy
        assert response.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_readiness_all_passing(self, health_checker):
        """Test readiness probe with all checks passing."""
        health_checker.add_readiness_check("database", lambda: True)
        health_checker.add_readiness_check("cache", lambda: True)

        response = await health_checker.readiness()
        assert response.status == "ready"
        assert "database" in response.checks
        assert "cache" in response.checks

    @pytest.mark.asyncio
    async def test_readiness_one_failing(self, health_checker):
        """Test readiness probe with one check failing."""
        health_checker.add_readiness_check("database", lambda: True)
        health_checker.add_readiness_check("cache", lambda: False)

        response = await health_checker.readiness()
        assert response.status == "not_ready"

    @pytest.mark.asyncio
    async def test_startup_not_completed(self, health_checker):
        """Test startup probe when not completed."""
        health_checker.add_startup_check("cache", lambda: False)

        response = await health_checker.startup()
        assert response.status == "starting"

    @pytest.mark.asyncio
    async def test_startup_completed(self, health_checker):
        """Test startup probe when completed."""
        health_checker.add_startup_check("cache", lambda: True)

        response = await health_checker.startup()
        assert response.status == "started"
        assert health_checker.startup_completed

    @pytest.mark.asyncio
    async def test_async_liveness_check(self, health_checker):
        """Test async liveness check."""

        async def async_check():
            return True

        health_checker.add_async_liveness_check("async_check", async_check)

        response = await health_checker.liveness()
        assert response.is_healthy

    @pytest.mark.asyncio
    async def test_async_readiness_check(self, health_checker):
        """Test async readiness check."""

        async def async_check():
            return True

        health_checker.add_async_readiness_check("async_check", async_check)

        response = await health_checker.readiness()
        assert response.status == "ready"

    @pytest.mark.asyncio
    async def test_async_startup_check(self, health_checker):
        """Test async startup check."""

        async def async_check():
            return True

        health_checker.add_async_startup_check("async_check", async_check)

        response = await health_checker.startup()
        assert response.status == "started"

    def test_uptime_seconds(self, health_checker):
        """Test uptime tracking."""
        import time

        time.sleep(0.1)
        uptime = health_checker.uptime_seconds
        assert uptime >= 0.1

    def test_mark_startup_complete(self, health_checker):
        """Test marking startup as complete."""
        health_checker.mark_startup_complete()
        assert health_checker.startup_completed

    def test_response_to_dict(self, health_checker):
        """Test HealthCheckResponse serialization."""
        response = HealthCheckResponse(
            status="healthy",
            checks={"test": {"status": "ok"}},
            uptime_seconds=10.0,
            version="0.5.0",
        )

        data = response.to_dict()
        assert data["status"] == "healthy"
        assert data["uptime_seconds"] == 10.0
        assert data["version"] == "0.5.0"
        assert "checks" in data


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_production_health_checker(self):
        """Test factory function."""
        checker = create_production_health_checker(
            startup_timeout=120.0,
            service_version="2.0.0",
        )
        assert checker is not None
        assert checker.startup_timeout == 120.0
        assert checker.service_version == "2.0.0"


class TestHealthCheckType:
    """Test HealthCheckType enum."""

    def test_health_check_types(self):
        """Test all health check types exist."""
        assert HealthCheckType.LIVENESS == "liveness"
        assert HealthCheckType.READINESS == "readiness"
        assert HealthCheckType.STARTUP == "startup"
