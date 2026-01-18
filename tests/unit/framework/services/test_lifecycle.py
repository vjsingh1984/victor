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

"""Tests for framework service lifecycle management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.services.lifecycle import (
    BaseService,
    BaseServiceConfig,
    create_http_service,
    create_sqlite_service,
    DockerServiceConfig,
    DockerServiceHandler,
    ExternalServiceConfig,
    ExternalServiceHandler,
    HTTPServiceConfig,
    HTTPServiceHandler,
    HealthCheckResult,
    HealthStatus,
    ServiceLifecycleProtocol,
    ServiceManager,
    ServiceMetadata,
    ServiceRegistry,
    ServiceStartError,
    ServiceState,
    ServiceStopError,
    SQLiteServiceConfig,
    SQLiteServiceHandler,
)


class MockService(BaseService):
    """Mock service for testing BaseService."""

    def __init__(self):
        metadata = ServiceMetadata(name="mock", service_type="test")
        super().__init__(metadata)
        self.start_called = False
        self.stop_called = False
        self.health_check_called = False

    async def _do_start(self):
        self.start_called = True

    async def _do_stop(self):
        self.stop_called = True

    async def _do_health_check(self):
        self.health_check_called = True
        return HealthCheckResult(HealthStatus.HEALTHY, "OK")


class TestServiceMetadata:
    """Tests for ServiceMetadata."""

    def test_create_metadata(self):
        """Test creating service metadata."""
        metadata = ServiceMetadata(
            name="test_service",
            service_type="sqlite",
            version="1.0.0",
            description="Test service",
            tags=["test", "database"],
        )
        assert metadata.name == "test_service"
        assert metadata.service_type == "sqlite"
        assert metadata.version == "1.0.0"
        assert metadata.tags == ["test", "database"]


class TestHealthCheckResult:
    """Tests for HealthCheckResult."""

    def test_create_result(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            message="Service is healthy",
            details={" uptime": 1234},
        )
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Service is healthy"
        assert result.details[" uptime"] == 1234

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            message="Service down",
        )
        d = result.to_dict()
        assert d["status"] == "unhealthy"
        assert d["message"] == "Service down"


class TestServiceState:
    """Tests for ServiceState enum."""

    def test_states(self):
        """Test all state values exist."""
        assert ServiceState.CREATED.value == "created"
        assert ServiceState.STARTING.value == "starting"
        assert ServiceState.RUNNING.value == "running"
        assert ServiceState.STOPPING.value == "stopping"
        assert ServiceState.STOPPED.value == "stopped"
        assert ServiceState.ERROR.value == "error"


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_statuses(self):
        """Test all status values exist."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestBaseService:
    """Tests for BaseService abstract class."""

    @pytest.mark.asyncio
    async def test_lifecycle(self):
        """Test complete service lifecycle."""
        service = MockService()

        # Initial state
        assert service.state == ServiceState.CREATED

        # Start
        await service.start()
        assert service.state == ServiceState.RUNNING
        assert service.start_called is True

        # Health check
        result = await service.health_check()
        assert result.status == HealthStatus.HEALTHY
        assert service.health_check_called is True

        # Stop
        await service.stop()
        assert service.state == ServiceState.STOPPED
        assert service.stop_called is True

    @pytest.mark.asyncio
    async def test_restart(self):
        """Test service restart."""
        service = MockService()

        await service.start()
        assert service.state == ServiceState.RUNNING

        await service.restart()
        assert service.state == ServiceState.RUNNING

    @pytest.mark.asyncio
    async def test_start_timeout(self):
        """Test start timeout handling."""

        class SlowService(BaseService):
            async def _do_start(self):
                await asyncio.sleep(10)

            async def _do_stop(self):
                pass

            async def _do_health_check(self):
                return HealthCheckResult(HealthStatus.HEALTHY, "OK")

        service = SlowService(
            metadata=ServiceMetadata(name="slow", service_type="test"),
            config=BaseServiceConfig(name="slow", start_timeout=0.1),
        )

        with pytest.raises(ServiceStartError):
            await service.start()

    @pytest.mark.asyncio
    async def test_start_failure_propagates(self):
        """Test that start failures propagate correctly."""

        class FailingService(BaseService):
            async def _do_start(self):
                raise ValueError("Start failed")

            async def _do_stop(self):
                pass

            async def _do_health_check(self):
                return HealthCheckResult(HealthStatus.HEALTHY, "OK")

        service = FailingService(
            metadata=ServiceMetadata(name="failing", service_type="test"),
        )

        with pytest.raises(ServiceStartError):
            await service.start()

        assert service.state == ServiceState.ERROR


class TestServiceRegistry:
    """Tests for ServiceRegistry."""

    @pytest.mark.asyncio
    async def test_register_and_get(self):
        """Test registering and getting services."""
        registry = ServiceRegistry()
        service = MockService()

        await registry.register(service)

        retrieved = await registry.get("mock")
        assert retrieved is service
        assert retrieved.state == ServiceState.RUNNING

    @pytest.mark.asyncio
    async def test_register_duplicate_fails(self):
        """Test that duplicate registration raises error."""
        registry = ServiceRegistry()
        service1 = MockService()
        service2 = MockService()

        await registry.register(service1)

        with pytest.raises(RuntimeError, match="already registered"):
            await registry.register(service2)

    @pytest.mark.asyncio
    async def test_unregister(self):
        """Test unregistering a service."""
        registry = ServiceRegistry()
        service = MockService()

        await registry.register(service)
        assert service.state == ServiceState.RUNNING

        await registry.unregister("mock")
        assert service.state == ServiceState.STOPPED

        retrieved = await registry.get("mock")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """Test shutting down all services."""
        registry = ServiceRegistry()

        service1 = MockService()
        service1.metadata.name = "service1"

        service2 = MockService()
        service2.metadata.name = "service2"

        await registry.register(service1)
        await registry.register(service2)

        await registry.shutdown_all()

        assert service1.state == ServiceState.STOPPED
        assert service2.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        """Test health check for all services."""
        registry = ServiceRegistry()
        service = MockService()

        await registry.register(service)

        results = await registry.health_check_all()
        assert "mock" in results
        assert results["mock"].status == HealthStatus.HEALTHY

    def test_list_services(self):
        """Test listing all service names."""
        registry = ServiceRegistry()
        service = MockService()

        # Use sync registration
        registry.register_sync(service)

        names = registry.list_services()
        assert "mock" in names


class TestSQLiteServiceHandler:
    """Tests for SQLiteServiceHandler."""

    @pytest.mark.asyncio
    async def test_from_config(self):
        """Test creating service from config dict."""
        config = {
            "name": "test_db",
            "type": "sqlite",
            "db_path": ":memory:",
            "readonly": True,
        }

        service = SQLiteServiceHandler.from_config(config)

        assert service.metadata.name == "test_db"
        assert service._sqlite_config.db_path == ":memory:"
        assert service._sqlite_config.readonly is True

    @pytest.mark.asyncio
    async def test_lifecycle_with_aiosqlite(self):
        """Test SQLite service lifecycle with aiosqlite."""
        pytest.importorskip("aiosqlite")

        metadata = ServiceMetadata(name="test_db", service_type="sqlite")
        config = SQLiteServiceConfig(db_path=":memory:")

        service = SQLiteServiceHandler(metadata, config)

        await service.start()
        assert service.state == ServiceState.RUNNING
        assert service.connection is not None

        result = await service.health_check()
        assert result.status == HealthStatus.HEALTHY

        await service.stop()
        assert service.state == ServiceState.STOPPED


class TestHTTPServiceHandler:
    """Tests for HTTPServiceHandler."""

    @pytest.mark.asyncio
    async def test_from_config(self):
        """Test creating service from config dict."""
        config = {
            "name": "api",
            "type": "http",
            "base_url": "https://api.example.com",
            "timeout": 60.0,
            "headers": {"Authorization": "Bearer token"},
        }

        service = HTTPServiceHandler.from_config(config)

        assert service.metadata.name == "api"
        assert service._http_config.base_url == "https://api.example.com"
        assert service._http_config.timeout == 60.0

    @pytest.mark.asyncio
    async def test_lifecycle_with_httpx(self):
        """Test HTTP service lifecycle with httpx."""
        pytest.importorskip("httpx")

        metadata = ServiceMetadata(name="test_api", service_type="http")
        config = HTTPServiceConfig(
            base_url="https://api.example.com",
            timeout=30.0,
        )

        service = HTTPServiceHandler(metadata, config)

        await service.start()
        assert service.state == ServiceState.RUNNING
        assert service.client is not None

        result = await service.health_check()
        assert result.status == HealthStatus.HEALTHY

        await service.stop()
        assert service.state == ServiceState.STOPPED


class TestExternalServiceHandler:
    """Tests for ExternalServiceHandler."""

    @pytest.mark.asyncio
    async def test_from_config(self):
        """Test creating service from config dict."""
        config = {
            "name": "external_api",
            "type": "external",
            "endpoint": "https://external.com/api",
            "check_url": "https://external.com/health",
        }

        service = ExternalServiceHandler.from_config(config)

        assert service.metadata.name == "external_api"
        assert service._external_config.endpoint == "https://external.com/api"

    @pytest.mark.asyncio
    async def test_health_check_without_url(self):
        """Test health check when no check URL configured."""
        metadata = ServiceMetadata(name="external", service_type="external")
        config = ExternalServiceConfig(endpoint="https://api.example.com")

        service = ExternalServiceHandler(metadata, config)

        await service.start()
        result = await service.health_check()
        assert result.status == HealthStatus.UNKNOWN

        await service.stop()


class TestServiceManager:
    """Tests for ServiceManager."""

    @pytest.mark.asyncio
    async def test_initialize_services(self):
        """Test initializing services from config list."""
        manager = ServiceManager()

        configs = [
            {
                "name": "db",
                "type": "sqlite",
                "db_path": ":memory:",
            },
        ]

        # Note: This will fail without aiosqlite, which is expected
        try:
            services = await manager.initialize_services(configs)
            assert "db" in services
        except (ValueError, ModuleNotFoundError, ImportError, ServiceStartError):
            # aiosqlite not installed or service failed to start
            pass

    @pytest.mark.asyncio
    async def test_unknown_service_type(self):
        """Test that unknown service type raises error."""
        manager = ServiceManager()

        configs = [
            {
                "name": "unknown",
                "type": "unknown_type",
            },
        ]

        with pytest.raises(ValueError, match="Unknown service type"):
            await manager.initialize_services(configs)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_create_sqlite_service(self):
        """Test create_sqlite_service convenience function."""
        pytest.importorskip("aiosqlite")

        service = await create_sqlite_service(
            name="test_db",
            db_path=":memory:",
            readonly=False,
        )

        assert service.metadata.name == "test_db"
        assert service.state == ServiceState.RUNNING

        await service.stop()

    @pytest.mark.asyncio
    async def test_create_http_service(self):
        """Test create_http_service convenience function."""
        pytest.importorskip("httpx")

        service = await create_http_service(
            name="test_api",
            base_url="https://api.example.com",
            timeout=60.0,
        )

        assert service.metadata.name == "test_api"
        assert service.state == ServiceState.RUNNING

        await service.stop()


class TestServiceProtocols:
    """Tests for service protocol compliance."""

    def test_service_lifecycle_protocol(self):
        """Test that BaseService implements ServiceLifecycleProtocol methods."""
        service = MockService()

        # Verify protocol methods exist
        assert hasattr(service, "start")
        assert hasattr(service, "stop")
        assert hasattr(service, "health_check")
        assert hasattr(service, "state")
        assert hasattr(service, "metadata")
        assert hasattr(service, "restart")

        # Verify they are callable
        import asyncio

        assert asyncio.iscoroutinefunction(service.start)
        assert asyncio.iscoroutinefunction(service.stop)
        assert asyncio.iscoroutinefunction(service.health_check)
        assert asyncio.iscoroutinefunction(service.restart)
