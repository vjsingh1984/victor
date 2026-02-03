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

"""Framework-level service lifecycle management.

This module provides universal service lifecycle management for use across
all verticals. It promotes common patterns from victor.workflows.services
to the framework layer with enhanced protocol-based interfaces.

Design Pattern: Template Method + Protocol (Interface Segregation)
- ServiceLifecycleProtocol: Interface for all service implementations
- ServiceRegistry: Universal service registry for dependency injection
- ServiceManager: Template method for lifecycle orchestration
- Built-in service handlers: SQLite, Docker, HTTP, external APIs

Key Features:
- Protocol-based interfaces for dependency inversion
- Health check protocols for monitoring
- Start/stop/monitor lifecycle stages
- Thread-safe service registry
- YAML configuration support

Usage in code:
    from victor.framework.services import (
        ServiceLifecycleProtocol,
        ServiceRegistry,
        ServiceManager,
        SQLiteServiceHandler,
    )

    # Create service
    service = SQLiteServiceHandler(db_path=":memory:")
    await service.start()

    # Register service
    registry = ServiceRegistry()
    registry.register("db", service)

    # Use service
    db = registry.get("db")

Usage in YAML workflows:
    services:
      - name: database
        type: sqlite
        config:
          db_path: ./data/project.db
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Service Lifecycle Protocols
# =============================================================================


class ServiceState(Enum):
    """States in the service lifecycle."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class HealthStatus(Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check.

    Attributes:
        status: Health status
        message: Human-readable status message
        details: Additional diagnostic information
        timestamp: When the check was performed
    """

    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class ServiceMetadata:
    """Metadata about a service.

    Attributes:
        name: Service identifier
        service_type: Type of service (sqlite, docker, http, etc.)
        version: Service version (optional)
        description: Human-readable description
        tags: List of tags for categorization
    """

    name: str
    service_type: str
    version: Optional[str] = None
    description: str = ""
    tags: list[str] = field(default_factory=list)


class ServiceLifecycleProtocol(Protocol):
    """Protocol defining the lifecycle interface for all services.

    This protocol enables dependency injection - high-level modules
    depend on this protocol, not concrete implementations (DIP).

    All services in the framework must implement this protocol to be
    compatible with the ServiceManager and ServiceRegistry.
    """

    async def start(self) -> None:
        """Start the service.

        Should initialize resources, establish connections, and
        prepare the service for use.

        Raises:
            ServiceStartError: If service fails to start
        """
        ...

    async def stop(self) -> None:
        """Stop the service.

        Should release resources, close connections, and perform
        graceful shutdown.

        Raises:
            ServiceStopError: If service fails to stop gracefully
        """
        ...

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the service.

        Returns:
            HealthCheckResult with current health status

        Example:
            result = await service.health_check()
            if result.status == HealthStatus.HEALTHY:
                print("Service is healthy")
        """
        ...

    @property
    def state(self) -> ServiceState:
        """Get the current state of the service.

        Returns:
            Current ServiceState
        """
        ...

    @property
    def metadata(self) -> ServiceMetadata:
        """Get service metadata.

        Returns:
            ServiceMetadata for this service
        """
        ...


class ServiceConfigurable(Protocol):
    """Protocol for services that can be configured via YAML."""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ServiceConfigurable":
        """Create service instance from configuration dict.

        Args:
            config: Configuration dictionary (typically from YAML)

        Returns:
            Configured service instance

        Example:
            config = {"db_path": "./data.db", "readonly": True}
            service = SQLiteServiceHandler.from_config(config)
        """
        ...


# =============================================================================
# Base Service Implementation
# =============================================================================


class ServiceStartError(Exception):
    """Raised when a service fails to start."""

    def __init__(self, service_name: str, reason: str):
        self.service_name = service_name
        self.reason = reason
        super().__init__(f"Service '{service_name}' failed to start: {reason}")


class ServiceStopError(Exception):
    """Raised when a service fails to stop."""

    def __init__(self, service_name: str, reason: str):
        self.service_name = service_name
        self.reason = reason
        super().__init__(f"Service '{service_name}' failed to stop: {reason}")


@dataclass
class BaseServiceConfig:
    """Base configuration for services.

    Attributes:
        name: Service identifier
        start_timeout: Timeout for start operation (seconds)
        stop_timeout: Timeout for stop operation (seconds)
        health_check_interval: Interval between health checks (seconds)
        auto_restart: If True, restart on failure
    """

    name: str
    start_timeout: float = 30.0
    stop_timeout: float = 10.0
    health_check_interval: float = 60.0
    auto_restart: bool = False


class BaseService(ABC):
    """Abstract base class for service implementations.

    Provides common lifecycle management, state tracking, and
    health check infrastructure. Subclasses implement the
    actual service-specific logic.

    Example:
        class MyService(BaseService):
            async def _do_start(self):
                # Initialize service resources
                ...

            async def _do_stop(self):
                # Cleanup service resources
                ...

            async def _do_health_check(self):
                # Check service health
                return HealthCheckResult(HealthStatus.HEALTHY, "OK")
    """

    def __init__(
        self,
        metadata: ServiceMetadata,
        config: Optional[BaseServiceConfig] = None,
    ):
        """Initialize base service.

        Args:
            metadata: Service metadata
            config: Service configuration
        """
        self._metadata = metadata
        self._config = config or BaseServiceConfig(name=metadata.name)
        self._state = ServiceState.CREATED
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task[Any]] = None

    @property
    def state(self) -> ServiceState:
        """Get current service state."""
        return self._state

    @property
    def metadata(self) -> ServiceMetadata:
        """Get service metadata."""
        return self._metadata

    @property
    def config(self) -> BaseServiceConfig:
        """Get service configuration."""
        return self._config

    async def start(self) -> None:
        """Start the service with timeout protection."""
        async with self._lock:
            if self._state in (ServiceState.RUNNING, ServiceState.STARTING):
                return

            self._state = ServiceState.STARTING
            logger.info(f"Starting service '{self._metadata.name}'...")

            try:
                await asyncio.wait_for(
                    self._do_start(),
                    timeout=self._config.start_timeout,
                )
                self._state = ServiceState.RUNNING
                logger.info(f"Service '{self._metadata.name}' started successfully")

                # Start health check task if configured
                if self._config.health_check_interval > 0:
                    self._start_health_monitor()

            except asyncio.TimeoutError:
                self._state = ServiceState.ERROR
                raise ServiceStartError(
                    self._metadata.name,
                    f"Start operation timed out after {self._config.start_timeout}s",
                )
            except Exception as e:
                self._state = ServiceState.ERROR
                raise ServiceStartError(self._metadata.name, str(e))

    async def stop(self) -> None:
        """Stop the service with timeout protection."""
        async with self._lock:
            if self._state in (ServiceState.STOPPED, ServiceState.CREATED):
                return

            self._state = ServiceState.STOPPING
            logger.info(f"Stopping service '{self._metadata.name}'...")

            # Cancel health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                self._health_check_task = None

            try:
                await asyncio.wait_for(
                    self._do_stop(),
                    timeout=self._config.stop_timeout,
                )
                self._state = ServiceState.STOPPED
                logger.info(f"Service '{self._metadata.name}' stopped successfully")

            except asyncio.TimeoutError:
                self._state = ServiceState.ERROR
                raise ServiceStopError(
                    self._metadata.name,
                    f"Stop operation timed out after {self._config.stop_timeout}s",
                )
            except Exception as e:
                self._state = ServiceState.ERROR
                raise ServiceStopError(self._metadata.name, str(e))

    async def health_check(self) -> HealthCheckResult:
        """Perform health check."""
        try:
            if self._state == ServiceState.RUNNING:
                return await self._do_health_check()
            elif self._state == ServiceState.ERROR:
                return HealthCheckResult(
                    HealthStatus.UNHEALTHY,
                    f"Service in ERROR state: {self._state.value}",
                )
            else:
                return HealthCheckResult(
                    HealthStatus.UNKNOWN,
                    f"Service not running: {self._state.value}",
                )
        except Exception as e:
            return HealthCheckResult(
                HealthStatus.UNHEALTHY,
                f"Health check failed: {e}",
            )

    async def restart(self) -> None:
        """Restart the service."""
        await self.stop()
        await self.start()

    # Abstract methods for subclasses

    @abstractmethod
    async def _do_start(self) -> None:
        """Implement service-specific start logic."""
        pass

    @abstractmethod
    async def _do_stop(self) -> None:
        """Implement service-specific stop logic."""
        pass

    @abstractmethod
    async def _do_health_check(self) -> HealthCheckResult:
        """Implement service-specific health check logic."""
        pass

    # Internal methods

    def _start_health_monitor(self) -> None:
        """Start background health monitoring task."""

        async def monitor() -> None:
            while self._state == ServiceState.RUNNING:
                try:
                    await asyncio.sleep(self._config.health_check_interval)
                    result = await self.health_check()
                    if result.status != HealthStatus.HEALTHY:
                        logger.warning(
                            f"Service '{self._metadata.name}' health check: {result.status.value}"
                        )
                        if self._config.auto_restart:
                            logger.info(f"Auto-restarting service '{self._metadata.name}'...")
                            await self.restart()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitor error for '{self._metadata.name}': {e}")

        self._health_check_task = asyncio.create_task(monitor())


# =============================================================================
# Built-in Service Implementations
# =============================================================================


@dataclass
class SQLiteServiceConfig:
    """Configuration for SQLite service.

    Attributes:
        db_path: Path to SQLite database file
        readonly: Open in read-only mode
        enable_wal: Enable WAL mode for better concurrency
        timeout: Query timeout in seconds
    """

    db_path: str
    readonly: bool = False
    enable_wal: bool = True
    timeout: float = 30.0


class SQLiteServiceHandler(BaseService):
    """Service handler for SQLite databases.

    Provides lifecycle management for SQLite connections with
    WAL mode, connection pooling, and health checks.

    Example:
        service = SQLiteServiceHandler(
            metadata=ServiceMetadata(
                name="project_db",
                service_type="sqlite",
            ),
            config=SQLiteServiceConfig(db_path="./data/project.db"),
        )
        await service.start()
    """

    def __init__(
        self,
        metadata: ServiceMetadata,
        config: SQLiteServiceConfig,
        base_config: Optional[BaseServiceConfig] = None,
    ):
        super().__init__(metadata, base_config)
        self._sqlite_config = config
        self._connection: Optional[Any] = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SQLiteServiceHandler":
        """Create from YAML config dict.

        Args:
            config: Configuration dictionary

        Returns:
            Configured SQLiteServiceHandler
        """
        db_path = config.get("db_path", ":memory:")
        metadata = ServiceMetadata(
            name=config.get("name", "sqlite"),
            service_type="sqlite",
        )
        sqlite_config = SQLiteServiceConfig(
            db_path=db_path,
            readonly=config.get("readonly", False),
            enable_wal=config.get("enable_wal", True),
            timeout=config.get("timeout", 30.0),
        )
        return cls(metadata, sqlite_config)

    async def _do_start(self) -> None:
        """Initialize SQLite connection."""
        try:
            import aiosqlite

            db_path = Path(self._sqlite_config.db_path).expanduser()

            # Create parent directory if needed
            if not self._sqlite_config.readonly and str(db_path) != ":memory:":
                db_path.parent.mkdir(parents=True, exist_ok=True)

            self._connection = await aiosqlite.connect(
                str(db_path),
                isolation_level=None,
            )

            # Enable WAL mode
            if self._sqlite_config.enable_wal:
                await self._connection.execute("PRAGMA journal_mode=WAL")
                await self._connection.execute("PRAGMA synchronous=NORMAL")

            logger.debug(f"SQLite service '{self._metadata.name}' connected to {db_path}")

        except ImportError:
            raise ServiceStartError(
                self._metadata.name,
                "aiosqlite not installed",
            )

    async def _do_stop(self) -> None:
        """Close SQLite connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def _do_health_check(self) -> HealthCheckResult:
        """Check SQLite connectivity."""
        if self._connection:
            try:
                async with self._connection.execute("SELECT 1") as cursor:
                    await cursor.fetchone()
                return HealthCheckResult(
                    HealthStatus.HEALTHY,
                    "SQLite database connected",
                )
            except Exception as e:
                return HealthCheckResult(
                    HealthStatus.UNHEALTHY,
                    f"Database error: {e}",
                )
        return HealthCheckResult(
            HealthStatus.UNHEALTHY,
            "No connection",
        )

    @property
    def connection(self) -> Optional[Any]:
        """Get the underlying SQLite connection."""
        return self._connection


@dataclass
class DockerServiceConfig:
    """Configuration for Docker service.

    Attributes:
        image: Docker image name
        container_name: Name for the container
        ports: Port mappings (host:container)
        volumes: Volume mappings
        environment: Environment variables
        auto_remove: Remove container on exit
        command: Command to run
    """

    image: str
    container_name: Optional[str] = None
    ports: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    auto_remove: bool = True
    command: Optional[str] = None


class DockerServiceHandler(BaseService):
    """Service handler for Docker containers.

    Provides lifecycle management for Docker containers with
    start/stop, health checks, and auto-cleanup.

    Example:
        service = DockerServiceHandler(
            metadata=ServiceMetadata(name="redis", service_type="docker"),
            config=DockerServiceConfig(image="redis:latest", ports={"6379": "6379"}),
        )
        await service.start()
    """

    def __init__(
        self,
        metadata: ServiceMetadata,
        config: DockerServiceConfig,
        base_config: Optional[BaseServiceConfig] = None,
    ):
        super().__init__(metadata, base_config)
        self._docker_config = config
        self._container: Optional[Any] = None
        self._client: Optional[Any] = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DockerServiceHandler":
        """Create from YAML config dict."""
        metadata = ServiceMetadata(
            name=config.get("name", "docker"),
            service_type="docker",
        )
        docker_config = DockerServiceConfig(
            image=config.get("image", "alpine"),
            container_name=config.get("container_name"),
            ports=config.get("ports", {}),
            volumes=config.get("volumes", {}),
            environment=config.get("environment", {}),
            auto_remove=config.get("auto_remove", True),
            command=config.get("command"),
        )
        return cls(metadata, docker_config)

    async def _do_start(self) -> None:
        """Start Docker container."""
        try:
            import docker
            from docker.models.containers import Container  # type: ignore[import-not-found]

            self._client = docker.from_env(skip_ssl_verification=True)

            # Check if image exists, pull if needed
            try:
                self._client.images.get(self._docker_config.image)
            except Exception:
                logger.info(f"Pulling Docker image: {self._docker_config.image}")
                self._client.images.pull(self._docker_config.image)

            # Start container
            container_name = self._docker_config.container_name or self._metadata.name

            self._container = self._client.containers.run(
                self._docker_config.image,
                name=container_name,
                ports=self._docker_config.ports,
                volumes=self._docker_config.volumes,
                environment=self._docker_config.environment,
                auto_remove=self._docker_config.auto_remove,
                command=self._docker_config.command,
                detach=True,
            )

            logger.info(f"Docker container '{container_name}' started")

        except ImportError:
            raise ServiceStartError(
                self._metadata.name,
                "docker package not installed",
            )
        except Exception as e:
            raise ServiceStartError(self._metadata.name, str(e))

    async def _do_stop(self) -> None:
        """Stop Docker container."""
        if self._container:
            try:
                self._container.stop(timeout=5)
                logger.debug(f"Docker container '{self._metadata.name}' stopped")
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")

        if self._client:
            self._client.close()
            self._client = None

        self._container = None

    async def _do_health_check(self) -> HealthCheckResult:
        """Check Docker container health."""
        if self._container:
            try:
                self._container.reload()
                status = self._container.status

                if status == "running":
                    return HealthCheckResult(
                        HealthStatus.HEALTHY,
                        "Container running",
                        {"container_id": self._container.id[:12]},
                    )
                else:
                    return HealthCheckResult(
                        HealthStatus.UNHEALTHY,
                        f"Container status: {status}",
                    )
            except Exception as e:
                return HealthCheckResult(
                    HealthStatus.UNHEALTHY,
                    f"Container error: {e}",
                )

        return HealthCheckResult(
            HealthStatus.UNKNOWN,
            "No container",
        )

    @property
    def container(self) -> Optional[Any]:
        """Get the Docker container instance."""
        return self._container


@dataclass
class HTTPServiceConfig:
    """Configuration for HTTP client service.

    Attributes:
        base_url: Base URL for API
        timeout: Request timeout
        headers: Default headers
        auth: Authentication tuple (username, password)
        verify_ssl: Verify SSL certificates
    """

    base_url: str
    timeout: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)
    auth: Optional[tuple[str, str]] = None
    verify_ssl: bool = True


class HTTPServiceHandler(BaseService):
    """Service handler for HTTP API clients.

    Provides lifecycle management for HTTP clients with
    connection pooling, session management, and health checks.

    Example:
        service = HTTPServiceHandler(
            metadata=ServiceMetadata(name="api", service_type="http"),
            config=HTTPServiceConfig(base_url="https://api.example.com"),
        )
        await service.start()
    """

    def __init__(
        self,
        metadata: ServiceMetadata,
        config: HTTPServiceConfig,
        base_config: Optional[BaseServiceConfig] = None,
    ):
        super().__init__(metadata, base_config)
        self._http_config = config
        self._client: Optional[Any] = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HTTPServiceHandler":
        """Create from YAML config dict."""
        metadata = ServiceMetadata(
            name=config.get("name", "http"),
            service_type="http",
        )
        http_config = HTTPServiceConfig(
            base_url=config.get("base_url", ""),
            timeout=config.get("timeout", 30.0),
            headers=config.get("headers", {}),
            auth=tuple(config.get("auth", [])) or None,
            verify_ssl=config.get("verify_ssl", True),
        )
        return cls(metadata, http_config)

    async def _do_start(self) -> None:
        """Initialize HTTP client."""
        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._http_config.base_url,
                timeout=self._http_config.timeout,
                headers=self._http_config.headers,
                auth=self._http_config.auth,
                verify=self._http_config.verify_ssl,
            )
            logger.debug(
                f"HTTP client '{self._metadata.name}' created for {self._http_config.base_url}"
            )

        except ImportError:
            raise ServiceStartError(
                self._metadata.name,
                "httpx not installed",
            )

    async def _do_stop(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _do_health_check(self) -> HealthCheckResult:
        """Check HTTP client health (optional health endpoint)."""
        if self._client:
            return HealthCheckResult(
                HealthStatus.HEALTHY,
                "HTTP client ready",
                {"base_url": self._http_config.base_url},
            )
        return HealthCheckResult(
            HealthStatus.UNHEALTHY,
            "No client",
        )

    @property
    def client(self) -> Optional[Any]:
        """Get the HTTP client instance."""
        return self._client

    async def get(self, path: str, **kwargs: Any) -> Any:
        """Make GET request."""
        if not self._client:
            raise RuntimeError("Service not started")
        return await self._client.get(path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Any:
        """Make POST request."""
        if not self._client:
            raise RuntimeError("Service not started")
        return await self._client.post(path, **kwargs)


@dataclass
class ExternalServiceConfig:
    """Configuration for external service references.

    Attributes:
        endpoint: Service endpoint URL or identifier
        check_url: Optional health check URL
        expected_status: Expected HTTP status for health checks
    """

    endpoint: str
    check_url: Optional[str] = None
    expected_status: int = 200


class ExternalServiceHandler(BaseService):
    """Service handler for external service references.

    Manages external services that are not directly controlled
    but need to be tracked for health and dependency purposes.

    Example:
        service = ExternalServiceHandler(
            metadata=ServiceMetadata(name="external_api", service_type="external"),
            config=ExternalServiceConfig(
                endpoint="https://api.external.com",
                check_url="https://api.external.com/health",
            ),
        )
        await service.start()
    """

    def __init__(
        self,
        metadata: ServiceMetadata,
        config: ExternalServiceConfig,
        base_config: Optional[BaseServiceConfig] = None,
    ):
        super().__init__(metadata, base_config)
        self._external_config = config
        self._http_client: Optional[Any] = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExternalServiceHandler":
        """Create from YAML config dict."""
        metadata = ServiceMetadata(
            name=config.get("name", "external"),
            service_type="external",
        )
        external_config = ExternalServiceConfig(
            endpoint=config.get("endpoint", ""),
            check_url=config.get("check_url"),
            expected_status=config.get("expected_status", 200),
        )
        return cls(metadata, external_config)

    async def _do_start(self) -> None:
        """Initialize external service (just verify)."""
        # For external services, we just log - no active management
        logger.info(f"External service '{self._metadata.name}' at {self._external_config.endpoint}")

    async def _do_stop(self) -> None:
        """No-op for external services."""
        pass

    async def _do_health_check(self) -> HealthCheckResult:
        """Check external service health."""
        if not self._external_config.check_url:
            # No health check configured, assume healthy
            return HealthCheckResult(
                HealthStatus.UNKNOWN,
                "No health check configured for external service",
            )

        try:
            if not self._http_client:
                import httpx

                self._http_client = httpx.AsyncClient()

            response = await self._http_client.get(
                self._external_config.check_url,
                timeout=5.0,
            )

            if response.status_code == self._external_config.expected_status:
                return HealthCheckResult(
                    HealthStatus.HEALTHY,
                    "External service responding",
                )
            else:
                return HealthCheckResult(
                    HealthStatus.DEGRADED,
                    f"Unexpected status: {response.status_code}",
                )

        except Exception as e:
            return HealthCheckResult(
                HealthStatus.UNHEALTHY,
                f"Health check failed: {e}",
            )


# =============================================================================
# Service Registry
# =============================================================================


class ServiceRegistry:
    """Universal service registry for dependency injection.

    Provides thread-safe registration, retrieval, and lifecycle
    management of services across the framework.

    Example:
        registry = ServiceRegistry()

        # Register service
        service = SQLiteServiceHandler(...)
        await registry.register(service)

        # Get service
        db = await registry.get("project_db")

        # Shutdown all
        await registry.shutdown_all()
    """

    def __init__(self) -> None:
        self._services: dict[str, BaseService] = {}
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    async def register(self, service: BaseService) -> None:
        """Register a service and start it.

        Args:
            service: Service to register

        Raises:
            RuntimeError: If service name already registered
        """
        async with self._async_lock:
            name = service.metadata.name

            if name in self._services:
                raise RuntimeError(f"Service '{name}' already registered")

            await service.start()
            self._services[name] = service
            logger.info(f"Registered service: {name} (type: {service.metadata.service_type})")

    def register_sync(self, service: BaseService) -> None:
        """Register service synchronously (doesn't start).

        Args:
            service: Service to register

        Note:
            Service must be started manually with await service.start()
        """
        with self._lock:
            name = service.metadata.name

            if name in self._services:
                raise RuntimeError(f"Service '{name}' already registered")

            self._services[name] = service

    async def get(self, name: str) -> Optional[BaseService]:
        """Get a registered service by name.

        Args:
            name: Service name

        Returns:
            Service or None if not found
        """
        async with self._async_lock:
            return self._services.get(name)

    def get_sync(self, name: str) -> Optional[BaseService]:
        """Get a registered service synchronously.

        Args:
            name: Service name

        Returns:
            Service or None if not found
        """
        with self._lock:
            return self._services.get(name)

    async def unregister(self, name: str) -> None:
        """Unregister and stop a service.

        Args:
            name: Service name
        """
        async with self._async_lock:
            service = self._services.pop(name, None)
            if service:
                await service.stop()
                logger.info(f"Unregistered service: {name}")

    async def shutdown_all(self) -> None:
        """Shutdown all registered services."""
        async with self._async_lock:
            tasks = [service.stop() for service in self._services.values()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._services.clear()
            logger.info("All services shut down")

    async def health_check_all(self) -> dict[str, HealthCheckResult]:
        """Check health of all registered services.

        Returns:
            Dictionary of service name -> health result
        """
        results = {}
        async with self._async_lock:
            for name, service in self._services.items():
                results[name] = await service.health_check()
        return results

    def list_services(self) -> list[str]:
        """List all registered service names.

        Returns:
            List of service names
        """
        with self._lock:
            return list(self._services.keys())


# =============================================================================
# Service Manager (YAML Integration)
# =============================================================================


class ServiceTypeHandler(Protocol):
    """Protocol for service type handlers."""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseService:
        """Create service from config."""
        ...


class ServiceManager:
    """Manages service lifecycle from YAML configuration.

    Provides a factory pattern for creating services from
    YAML configuration and managing their lifecycle.

    Example:
        manager = ServiceManager()

        # Load from YAML config
        configs = [
            {"name": "db", "type": "sqlite", "db_path": "./data.db"},
            {"name": "api", "type": "http", "base_url": "https://api.example.com"},
        ]
        await manager.initialize_services(configs)

        # Get services
        db = await manager.get_service("db")
    """

    # Registry of service type handlers
    _handlers: dict[str, type[ServiceTypeHandler]] = {
        "sqlite": SQLiteServiceHandler,
        "docker": DockerServiceHandler,
        "http": HTTPServiceHandler,
        "https": HTTPServiceHandler,
        "external": ExternalServiceHandler,
    }

    def __init__(self, registry: Optional[ServiceRegistry] = None):
        """Initialize service manager.

        Args:
            registry: Service registry to use (creates new if None)
        """
        self._registry = registry or ServiceRegistry()

    @classmethod
    def register_handler(cls, service_type: str, handler: type[ServiceTypeHandler]) -> None:
        """Register a service type handler.

        Args:
            service_type: Service type identifier
            handler: Handler class that implements from_config
        """
        cls._handlers[service_type] = handler

    async def initialize_services(
        self,
        configs: list[dict[str, Any]],
    ) -> dict[str, BaseService]:
        """Initialize services from configuration list.

        Args:
            configs: List of service configuration dicts

        Returns:
            Dictionary of service name -> service instance

        Raises:
            ValueError: If service type is not registered
        """
        initialized = {}

        for config in configs:
            service_type = config.get("type", "")
            handler = self._handlers.get(service_type)

            if not handler:
                raise ValueError(f"Unknown service type: {service_type}")

            service = handler.from_config(config)
            await self._registry.register(service)
            initialized[service.metadata.name] = service

        return initialized

    async def get_service(self, name: str) -> Optional[BaseService]:
        """Get a service by name.

        Args:
            name: Service name

        Returns:
            Service or None
        """
        return await self._registry.get(name)

    async def shutdown_all(self) -> None:
        """Shutdown all managed services."""
        await self._registry.shutdown_all()

    async def health_check_all(self) -> dict[str, HealthCheckResult]:
        """Check health of all services.

        Returns:
            Dictionary of service name -> health result
        """
        return await self._registry.health_check_all()

    @property
    def registry(self) -> ServiceRegistry:
        """Get the underlying service registry."""
        return self._registry


# =============================================================================
# Convenience functions
# =============================================================================


async def create_sqlite_service(
    name: str,
    db_path: str,
    readonly: bool = False,
    enable_wal: bool = True,
) -> SQLiteServiceHandler:
    """Create and start a SQLite service.

    Args:
        name: Service name
        db_path: Path to database file
        readonly: Open in read-only mode
        enable_wal: Enable WAL mode

    Returns:
        Started SQLiteServiceHandler

    Example:
        db = await create_sqlite_service("project_db", "./data/project.db")
        conn = db.connection
    """
    metadata = ServiceMetadata(name=name, service_type="sqlite")
    config = SQLiteServiceConfig(
        db_path=db_path,
        readonly=readonly,
        enable_wal=enable_wal,
    )
    service = SQLiteServiceHandler(metadata, config)
    await service.start()
    return service


async def create_http_service(
    name: str,
    base_url: str,
    timeout: float = 30.0,
    headers: Optional[dict[str, str]] = None,
) -> HTTPServiceHandler:
    """Create and start an HTTP client service.

    Args:
        name: Service name
        base_url: Base URL for API
        timeout: Request timeout
        headers: Default headers

    Returns:
        Started HTTPServiceHandler

    Example:
        api = await create_http_service("api", "https://api.example.com")
        response = await api.get("/endpoint")
    """
    metadata = ServiceMetadata(name=name, service_type="http")
    config = HTTPServiceConfig(
        base_url=base_url,
        timeout=timeout,
        headers=headers or {},
    )
    service = HTTPServiceHandler(metadata, config)
    await service.start()
    return service


__all__ = [
    # Protocols
    "ServiceLifecycleProtocol",
    "ServiceConfigurable",
    "ServiceTypeHandler",
    # State enums
    "ServiceState",
    "HealthStatus",
    # Data classes
    "HealthCheckResult",
    "ServiceMetadata",
    "BaseServiceConfig",
    # Exceptions
    "ServiceStartError",
    "ServiceStopError",
    # Base class
    "BaseService",
    # Built-in services
    "SQLiteServiceHandler",
    "SQLiteServiceConfig",
    "DockerServiceHandler",
    "DockerServiceConfig",
    "HTTPServiceHandler",
    "HTTPServiceConfig",
    "ExternalServiceHandler",
    "ExternalServiceConfig",
    # Registry
    "ServiceRegistry",
    # Manager
    "ServiceManager",
    # Convenience functions
    "create_sqlite_service",
    "create_http_service",
]
