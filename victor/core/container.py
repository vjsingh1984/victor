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

"""Dependency Injection Container for Victor.

This module provides a lightweight dependency injection container that:
- Manages service lifecycles (singleton, scoped, transient)
- Supports factory-based and instance-based registration
- Provides scoped containers for request-level isolation
- Enables easy testing through dependency substitution

Design Principles:
- Explicit over implicit (no auto-wiring)
- Type-safe with Protocol support
- Thread-safe singleton management
- Memory-efficient with weak references for scoped services

Example Usage:
    from victor.core.container import ServiceContainer, ServiceLifetime

    # Create and configure container
    container = ServiceContainer()
    container.register(MetricsService, lambda c: MetricsCollector(), ServiceLifetime.SINGLETON)
    container.register(DebugLogger, lambda c: DebugLogger(), ServiceLifetime.SINGLETON)

    # Resolve services
    metrics = container.get(MetricsService)
    logger = container.get(DebugLogger)

    # Create scoped container for request isolation
    with container.create_scope() as scope:
        request_service = scope.get(RequestService)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Protocol,
    Type,
    TypeVar,
    cast,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
TService = TypeVar("TService", covariant=True)


class ServiceLifetime(Enum):
    """Defines how long a service instance lives.

    Values:
        SINGLETON: One instance for the entire application lifetime
        SCOPED: One instance per scope (e.g., per request)
        TRANSIENT: New instance every time it's requested
    """

    SINGLETON = "singleton"
    SCOPED = "scoped"
    TRANSIENT = "transient"


@runtime_checkable
class Disposable(Protocol):
    """Protocol for services that need cleanup."""

    def dispose(self) -> None:
        """Release resources held by the service."""
        ...


@dataclass
class ServiceDescriptor(Generic[T]):
    """Describes how to create and manage a service."""

    service_type: Type[T]
    factory: Callable[["ServiceContainer"], T]
    lifetime: ServiceLifetime
    instance: Optional[T] = None

    def create_instance(self, container: "ServiceContainer") -> T:
        """Create a new instance using the factory."""
        return self.factory(container)


class ServiceScope:
    """Scoped container for request-level service isolation.

    Services registered with SCOPED lifetime get one instance per scope.
    When the scope is disposed, all scoped services are cleaned up.
    """

    def __init__(self, parent: "ServiceContainer"):
        """Initialize scope with parent container.

        Args:
            parent: Parent container to inherit singleton services from
        """
        self._parent = parent
        self._scoped_instances: Dict[Type[Any], Any] = {}
        self._disposed = False
        self._lock = threading.Lock()

    def get(self, service_type: Type[T]) -> T:
        """Get a service instance within this scope.

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance

        Raises:
            ServiceNotFoundError: If service is not registered
            ScopeDisposedError: If scope has been disposed
        """
        if self._disposed:
            raise ScopeDisposedError(f"Scope has been disposed, cannot resolve {service_type}")

        descriptor = self._parent._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            # Singletons come from parent
            return self._parent.get(service_type)

        if descriptor.lifetime == ServiceLifetime.SCOPED:
            with self._lock:
                if service_type not in self._scoped_instances:
                    self._scoped_instances[service_type] = descriptor.create_instance(self._parent)
                return cast(T, self._scoped_instances[service_type])

        # Transient - always create new
        return descriptor.create_instance(self._parent)

    def dispose(self) -> None:
        """Dispose all scoped services."""
        if self._disposed:
            return

        self._disposed = True
        for service in self._scoped_instances.values():
            if isinstance(service, Disposable):
                try:
                    service.dispose()
                except Exception as e:
                    logger.warning(f"Error disposing scoped service: {e}")

        self._scoped_instances.clear()

    def __enter__(self) -> "ServiceScope":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not registered."""

    def __init__(self, service_type: Type[Any]):
        self.service_type = service_type
        # Handle both Type objects and string service names
        name = service_type.__name__ if hasattr(service_type, "__name__") else str(service_type)
        super().__init__(f"Service not registered: {name}")


class ServiceAlreadyRegisteredError(Exception):
    """Raised when trying to register a service that already exists."""

    def __init__(self, service_type: Type[Any]):
        self.service_type = service_type
        super().__init__(f"Service already registered: {service_type.__name__}")


class ScopeDisposedError(Exception):
    """Raised when trying to use a disposed scope."""

    pass


class ServiceContainer:
    """Central dependency injection container.

    Manages service registration, resolution, and lifecycle. Supports
    singleton, scoped, and transient service lifetimes.

    Thread-safe for concurrent access to singleton services.

    Example:
        container = ServiceContainer()

        # Register services
        container.register(ILogger, lambda c: FileLogger(), ServiceLifetime.SINGLETON)
        container.register(ICache, lambda c: RedisCache(c.get(ILogger)), ServiceLifetime.SINGLETON)

        # Resolve services
        logger = container.get(ILogger)
        cache = container.get(ICache)

        # Create scoped container
        with container.create_scope() as scope:
            request_handler = scope.get(IRequestHandler)
    """

    def __init__(self) -> None:
        """Initialize empty container."""
        self._descriptors: Dict[Type[Any], ServiceDescriptor[Any]] = {}
        self._lock = threading.RLock()
        self._disposed = False

    def register(
        self,
        service_type: Type[T],
        factory: Callable[["ServiceContainer"], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> "ServiceContainer":
        """Register a service with a factory function.

        Args:
            service_type: Type/interface to register
            factory: Factory function that creates the service instance
            lifetime: How long the service instance lives

        Returns:
            Self for method chaining

        Raises:
            ServiceAlreadyRegisteredError: If service type already registered
        """
        with self._lock:
            if service_type in self._descriptors:
                raise ServiceAlreadyRegisteredError(service_type)

            self._descriptors[service_type] = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                lifetime=lifetime,
            )
            logger.debug(f"Registered {service_type.__name__} with {lifetime.value} lifetime")

        return self

    def register_instance(self, service_type: Type[T], instance: T) -> "ServiceContainer":
        """Register an existing instance as a singleton.

        Args:
            service_type: Type/interface to register
            instance: Pre-created service instance

        Returns:
            Self for method chaining
        """
        with self._lock:
            if service_type in self._descriptors:
                raise ServiceAlreadyRegisteredError(service_type)

            descriptor = ServiceDescriptor(
                service_type=service_type,
                factory=lambda c: instance,
                lifetime=ServiceLifetime.SINGLETON,
                instance=instance,
            )
            self._descriptors[service_type] = descriptor
            logger.debug(f"Registered {service_type.__name__} instance as singleton")

        return self

    def register_or_replace(
        self,
        service_type: Type[T],
        factory: Callable[["ServiceContainer"], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> "ServiceContainer":
        """Register a service, replacing existing registration if present.

        Useful for testing where you want to substitute implementations.

        Args:
            service_type: Type/interface to register
            factory: Factory function that creates the service instance
            lifetime: How long the service instance lives

        Returns:
            Self for method chaining
        """
        with self._lock:
            # Dispose existing singleton if present
            if service_type in self._descriptors:
                existing = self._descriptors[service_type]
                if existing.instance is not None and isinstance(existing.instance, Disposable):
                    try:
                        existing.instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing replaced service: {e}")

            self._descriptors[service_type] = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                lifetime=lifetime,
            )
            logger.debug(f"Replaced {service_type.__name__} with {lifetime.value} lifetime")

        return self

    def get(self, service_type: Type[T]) -> T:
        """Get a service instance.

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance

        Raises:
            ServiceNotFoundError: If service is not registered
        """
        descriptor = self._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.TRANSIENT:
            return descriptor.create_instance(self)

        # Singleton or Scoped (scoped in root container acts as singleton)
        with self._lock:
            if descriptor.instance is None:
                descriptor.instance = descriptor.create_instance(self)
            return descriptor.instance

    def get_optional(self, service_type: Type[T]) -> Optional[T]:
        """Get a service instance, or None if not registered.

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance or None
        """
        try:
            return self.get(service_type)
        except ServiceNotFoundError:
            return None

    def get_service(self, service_type: Type[T]) -> T:
        """Alias for get() for backward compatibility.

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance

        Raises:
            ServiceNotFoundError: If service is not registered
        """
        return self.get(service_type)

    def is_registered(self, service_type: Type[Any]) -> bool:
        """Check if a service type is registered.

        Args:
            service_type: Type to check

        Returns:
            True if registered, False otherwise
        """
        return service_type in self._descriptors

    def _get_descriptor(self, service_type: Type[T]) -> ServiceDescriptor[T]:
        """Get the descriptor for a service type.

        Args:
            service_type: Type to look up

        Returns:
            Service descriptor

        Raises:
            ServiceNotFoundError: If not registered
        """
        descriptor = self._descriptors.get(service_type)
        if descriptor is None:
            raise ServiceNotFoundError(service_type)
        return descriptor

    def create_scope(self) -> ServiceScope:
        """Create a new service scope.

        Scoped services will have one instance per scope.

        Returns:
            New service scope
        """
        return ServiceScope(self)

    def get_registered_types(self) -> list[Type[Any]]:
        """Get list of all registered service types.

        Returns:
            List of registered types
        """
        return list(self._descriptors.keys())

    def dispose(self) -> None:
        """Dispose all singleton services and clear registrations."""
        if self._disposed:
            return

        self._disposed = True
        with self._lock:
            for descriptor in self._descriptors.values():
                if descriptor.instance is not None and isinstance(descriptor.instance, Disposable):
                    try:
                        descriptor.instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing service: {e}")

            self._descriptors.clear()

    def __enter__(self) -> "ServiceContainer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()


# =============================================================================
# Global Container Management
# =============================================================================

_global_container: Optional[ServiceContainer] = None
_global_lock = threading.Lock()


def get_container() -> ServiceContainer:
    """Get the global service container.

    Creates a new container if one doesn't exist.

    Returns:
        Global service container
    """
    global _global_container
    with _global_lock:
        if _global_container is None:
            _global_container = ServiceContainer()
        return _global_container


def set_container(container: ServiceContainer) -> None:
    """Set the global service container.

    Args:
        container: Container to use as global
    """
    global _global_container
    with _global_lock:
        if _global_container is not None:
            _global_container.dispose()
        _global_container = container


def reset_container() -> None:
    """Reset the global container (for testing).

    Disposes existing container and clears the global reference.
    """
    global _global_container
    with _global_lock:
        if _global_container is not None:
            _global_container.dispose()
            _global_container = None


# =============================================================================
# Service Protocol Definitions
# =============================================================================


@runtime_checkable
class MetricsServiceProtocol(Protocol):
    """Protocol for metrics collection services."""

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        ...

    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        ...


@runtime_checkable
class LoggerServiceProtocol(Protocol):
    """Protocol for debug/analytics logging services."""

    def log(self, message: str, level: str = "info", **kwargs: Any) -> None:
        """Log a message."""
        ...

    @property
    def enabled(self) -> bool:
        """Whether logging is enabled."""
        ...


@runtime_checkable
class CacheServiceProtocol(Protocol):
    """Protocol for caching services."""

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a cached value."""
        ...

    def invalidate(self, key: str) -> None:
        """Invalidate a cached value."""
        ...


@runtime_checkable
class EmbeddingServiceProtocol(Protocol):
    """Protocol for embedding services."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...
