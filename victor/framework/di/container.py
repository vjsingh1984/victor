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

"""Enhanced Dependency Injection Container with auto-resolution.

This module provides an advanced DI container that extends the current
ServiceContainer with additional capabilities:
- Auto-resolution of constructor dependencies via inspect
- Lifecycle management (singleton, transient, scoped)
- Lazy initialization
- Circular dependency detection
- Factory functions as fallback
- Type-safe registration and resolution

Design Principles:
- Backward compatible with existing ServiceContainer
- Explicit registration with auto-resolution as enhancement
- Clear error messages for missing dependencies
- Thread-safe singleton management

Example Usage:
    from victor.framework.di.container import DIContainer, ServiceLifetime

    # Create container
    container = DIContainer()

    # Auto-register with constructor injection
    container.register(Logger)
    container.register(Database, lifetime=ServiceLifetime.SINGLETON)
    container.register(UserService)

    # Resolve with auto-injected dependencies
    user_service = container.get(UserService)
    # Logger and Database are automatically injected

Migration Path:
    This container is designed to coexist with victor.core.container.
    Use this for new code or gradual migration of existing services.
"""

from __future__ import annotations

import inspect
from inspect import Parameter
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Mapping,
    Optional,
    Type,
    TypeVar,
    cast,
    get_type_hints,
)
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")
TService = TypeVar("TService")


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


class DIError(Exception):
    """Base exception for DI container errors."""

    pass


class ServiceNotFoundError(DIError):
    """Raised when a requested service is not registered."""

    def __init__(self, service_type: type[Any], chain: Optional[list[type[Any]]] = None):
        self.service_type = service_type
        self.chain = chain or []
        name = getattr(service_type, "__name__", str(service_type))

        if chain:
            chain_str = " -> ".join(getattr(t, "__name__", str(t)) for t in chain)
            msg = f"Service not registered: {name} (required by: {chain_str})"
        else:
            msg = f"Service not registered: {name}"

        super().__init__(msg)


class CircularDependencyError(DIError):
    """Raised when a circular dependency is detected."""

    def __init__(self, chain: list[type[Any]]):
        self.chain = chain
        chain_str = " -> ".join(getattr(t, "__name__", str(t)) for t in chain)
        super().__init__(f"Circular dependency detected: {chain_str}")


class ServiceAlreadyRegisteredError(DIError):
    """Raised when trying to register a service that already exists."""

    def __init__(self, service_type: type[Any]):
        self.service_type = service_type
        super().__init__(f"Service already registered: {service_type.__name__}")


@dataclass
class ServiceDescriptor(Generic[T]):
    """Describes how to create and manage a service with auto-resolution."""

    service_type: Type[T]
    lifetime: ServiceLifetime
    factory: Optional[Callable[..., T]] = None
    instance: Optional[T] = None
    _resolved: bool = field(default=False, repr=False)
    use_auto_resolution: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        """Validate descriptor configuration."""
        if self.factory is None:
            # Use service_type as factory if no factory provided
            self.factory = self.service_type

    def create_instance(self, container: "DIContainer") -> T:
        """Create instance with auto-injected dependencies."""
        if self.factory is None:
            raise DIError(f"No factory available for {self.service_type}")

        # Check if we should use auto-resolution
        if self.use_auto_resolution:
            # Try auto-resolution
            try:
                return container._create_with_auto_resolution(self.factory, self.service_type)
            except Exception as e:
                raise DIError(f"Failed to create instance of {self.service_type}: {e}") from e
        else:
            # Call factory directly without auto-resolution
            try:
                return self.factory()
            except Exception as e:
                raise DIError(f"Failed to create instance of {self.service_type}: {e}") from e


@dataclass
class ResolutionContext:
    """Tracks state during dependency resolution."""

    resolving: set[Type[Any]] = field(default_factory=set)
    depth: int = 0
    max_depth: int = 50


class DIContainer:
    """Enhanced DI Container with auto-resolution capabilities.

    Features:
    - Auto-resolution of constructor dependencies
    - Lifecycle management (singleton, transient, scoped)
    - Circular dependency detection
    - Thread-safe singleton management
    - Lazy initialization

    Example:
        container = DIContainer()

        # Auto-register with constructor injection
        container.register(Logger)
        container.register(Database, lifetime=ServiceLifetime.SINGLETON)
        container.register(UserService)

        # Resolve with auto-injected dependencies
        user_service = container.get(UserService)
    """

    def __init__(self) -> None:
        """Initialize empty container."""
        self._descriptors: Dict[type[Any], ServiceDescriptor[Any]] = {}
        self._lock = threading.RLock()
        self._disposed = False
        self._resolution_stack: list[type[Any]] = []

    def register(
        self,
        service_type: Type[T],
        factory: Optional[Callable[..., T]] = None,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        implementation: Optional[Type[T]] = None,
        as_interfaces: Optional[list[type[Any]]] = None,
    ) -> "DIContainer":
        """Register a service with optional auto-resolution.

        Args:
            service_type: Type/interface to register
            factory: Optional factory function (defaults to service_type constructor)
            lifetime: How long the service instance lives
            implementation: Optional implementation class (for interface registration)
            as_interfaces: Optional list of interfaces this service implements

        Returns:
            Self for method chaining

        Raises:
            ServiceAlreadyRegisteredError: If service type already registered

        Example:
            # Register concrete type with default constructor
            container.register(Logger)

            # Register with factory function
            container.register(Database, lambda: Database("localhost"))

            # Register interface with implementation
            container.register(ILogger, implementation=FileLogger)

            # Register singleton
            container.register(ConfigService, lifetime=ServiceLifetime.SINGLETON)

            # Register as multiple interfaces
            container.register(
                Logger,
                as_interfaces=[ILogger, IDisposable]
            )
        """
        with self._lock:
            # Determine factory
            if factory is None:
                if implementation is not None:
                    factory = implementation
                else:
                    factory = service_type

            # Register primary type
            if service_type in self._descriptors:
                raise ServiceAlreadyRegisteredError(service_type)

            self._descriptors[service_type] = ServiceDescriptor(
                service_type=service_type,
                lifetime=lifetime,
                factory=factory,
            )

            logger.debug(f"Registered {service_type.__name__} with {lifetime.value} lifetime")

            # Register as interfaces if specified
            if as_interfaces:
                for interface in as_interfaces:
                    if interface not in self._descriptors:
                        self._descriptors[interface] = ServiceDescriptor(
                            service_type=interface,
                            lifetime=lifetime,
                            factory=factory,
                        )
                        logger.debug(f"Registered {service_type.__name__} as {interface.__name__}")

        return self

    def register_instance(
        self,
        service_type: Type[T],
        instance: T,
    ) -> "DIContainer":
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
                lifetime=ServiceLifetime.SINGLETON,
                factory=lambda: instance,
                instance=instance,
            )
            self._descriptors[service_type] = descriptor
            logger.debug(f"Registered {service_type.__name__} instance as singleton")

        return self

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[["DIContainer"], T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    ) -> "DIContainer":
        """Register a service with a container-aware factory function.

        Use this when the factory needs access to the container for manual
        dependency resolution.

        Args:
            service_type: Type/interface to register
            factory: Factory function that receives the container
            lifetime: How long the service instance lives

        Returns:
            Self for method chaining

        Example:
            def create_cache(container: DIContainer) -> Cache:
                logger = container.get(Logger)
                return Cache(logger=logger)

            container.register_factory(Cache, create_cache)
        """
        with self._lock:
            if service_type in self._descriptors:
                raise ServiceAlreadyRegisteredError(service_type)

            # Wrap factory to make it callable without container arg
            # Capture self in closure
            container_ref = self

            @wraps(factory)
            def wrapped_factory() -> Any:
                return factory(container_ref)

            self._descriptors[service_type] = ServiceDescriptor(
                service_type=service_type,
                lifetime=lifetime,
                factory=wrapped_factory,
                use_auto_resolution=False,  # Don't auto-resolve factory functions
            )

        return self

    def get(self, service_type: Type[T]) -> T:
        """Get a service instance with auto-injected dependencies.

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance

        Raises:
            ServiceNotFoundError: If service is not registered
            CircularDependencyError: If circular dependency detected
        """
        descriptor = self._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.TRANSIENT:
            return descriptor.create_instance(self)

        # Singleton or Scoped
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
            # Try to provide helpful error with resolution chain
            chain = self._resolution_stack.copy() if self._resolution_stack else None
            raise ServiceNotFoundError(service_type, chain)
        return descriptor

    def _create_with_auto_resolution(
        self,
        factory: Callable[..., T],
        target_type: Type[T],
    ) -> T:
        """Create instance with auto-injected constructor dependencies.

        Args:
            factory: Factory function or class to instantiate
            target_type: Target type being created (for error messages)

        Returns:
            Created instance

        Raises:
            CircularDependencyError: If circular dependency detected
            ServiceNotFoundError: If required dependency not registered
        """
        # Check for circular dependencies
        if target_type in self._resolution_stack:
            raise CircularDependencyError(self._resolution_stack + [target_type])

        # Add to resolution stack
        self._resolution_stack.append(target_type)

        try:
            # Inspect factory signature
            sig = inspect.signature(factory)
            parameters: Mapping[str, Parameter] = sig.parameters

            # Skip 'self' parameter for methods
            if list(parameters.keys()) and list(parameters.keys())[0] == "self":
                parameters = dict[str, Parameter]({k: v for k, v in parameters.items() if k != "self"})

            # Build kwargs with resolved dependencies
            kwargs = {}
            for param_name, param in parameters.items():
                # Skip parameters with defaults
                if param.default != param.empty:
                    continue

                # Skip *args and **kwargs
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue

                # Get type hint
                param_type = param.annotation

                if param_type is param.empty or param_type is Any:
                    logger.warning(
                        f"No type hint for parameter '{param_name}' in {factory.__name__}, "
                        "skipping auto-resolution"
                    )
                    continue

                # Resolve dependency
                try:
                    dependency = self.get(param_type)
                    kwargs[param_name] = dependency
                except ServiceNotFoundError as e:
                    logger.error(
                        f"Cannot resolve dependency {param_type} for parameter '{param_name}' "
                        f"in {factory.__name__}: {e}"
                    )
                    raise

            # Create instance with resolved dependencies
            try:
                instance = factory(**kwargs)
                return instance
            except TypeError as e:
                raise DIError(
                    f"Failed to instantiate {factory.__name__} with auto-resolved deps: {e}"
                ) from e

        finally:
            # Remove from resolution stack
            if self._resolution_stack and self._resolution_stack[-1] == target_type:
                self._resolution_stack.pop()

    def create_scope(self) -> "DIScope":
        """Create a new service scope.

        Scoped services will have one instance per scope.

        Returns:
            New service scope
        """
        return DIScope(self)

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
            # Clear descriptors
            self._descriptors.clear()
            self._resolution_stack.clear()

    def __enter__(self) -> "DIContainer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()


class DIScope:
    """Scoped container for request-level service isolation.

    Services registered with SCOPED lifetime get one instance per scope.
    When the scope is disposed, all scoped services are cleaned up.
    """

    def __init__(self, parent: DIContainer):
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
            DIError: If scope has been disposed
        """
        if self._disposed:
            raise DIError(f"Scope has been disposed, cannot resolve {service_type}")

        descriptor = self._parent._get_descriptor(service_type)

        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            # Singletons come from parent
            return self._parent.get(service_type)

        if descriptor.lifetime == ServiceLifetime.SCOPED:
            with self._lock:
                if service_type not in self._scoped_instances:
                    instance = descriptor.create_instance(self._parent)
                    self._scoped_instances[service_type] = cast(T, instance)
                return self._scoped_instances[service_type]

        # Transient - always create new
        return descriptor.create_instance(self._parent)

    def dispose(self) -> None:
        """Dispose all scoped services."""
        if self._disposed:
            return

        self._disposed = True
        with self._lock:
            # Clear scoped instances
            self._scoped_instances.clear()

    def __enter__(self) -> "DIScope":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_container(*registrations: tuple[type[Any], ServiceLifetime]) -> DIContainer:
    """Create a container with multiple services registered.

    Args:
        *registrations: Tuples of (service_type, lifetime)

    Returns:
        Configured DIContainer

    Example:
        container = create_container(
            (Logger, ServiceLifetime.SINGLETON),
            (Database, ServiceLifetime.SINGLETON),
            (UserService, ServiceLifetime.TRANSIENT),
        )
    """
    container = DIContainer()
    for service_type, lifetime in registrations:
        container.register(service_type, lifetime=lifetime)
    return container
