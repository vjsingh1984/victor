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

"""Comprehensive tests for ServiceContainer DI container.

This test module provides comprehensive coverage for ServiceContainer,
testing service registration, resolution, lifetimes, scoped services,
and thread-safe operations.

Target: 60%+ coverage for container.py
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from victor.core.container import (
    Disposable,
    ServiceContainer,
    ServiceLifetime,
    ServiceNotFoundError,
    ServiceAlreadyRegisteredError,
    ScopeDisposedError,
    get_container,
    reset_container,
    set_container,
)


# =============================================================================
# Test Services
# =============================================================================


class ILogger:
    """Test logger interface."""

    def log(self, message: str) -> None:
        pass


class FileLogger(ILogger):
    """Test file logger implementation."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


class ConsoleLogger(ILogger):
    """Test console logger implementation."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


class ICache:
    """Test cache interface."""

    def get(self, key: str) -> Any:
        pass

    def set(self, key: str, value: Any) -> None:
        pass


class SimpleCache(ICache):
    """Test cache implementation."""

    def __init__(self) -> None:
        self._storage: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._storage.get(key)

    def set(self, key: str, value: Any) -> None:
        self._storage[key] = value


class DisposableService(Disposable):
    """Test service that implements Disposable."""

    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


class ServiceWithDependencies:
    """Test service with dependencies."""

    def __init__(
        self, logger: ILogger, cache: ICache
    ) -> None:
        self.logger = logger
        self.cache = cache


# =============================================================================
# Container Initialization Tests
# =============================================================================


class TestContainerInitialization:
    """Tests for container initialization."""

    def test_empty_container(self):
        """Test creating an empty container."""
        container = ServiceContainer()

        assert container is not None
        assert len(container.get_registered_types()) == 0

    def test_container_context_manager(self):
        """Test using container as context manager."""
        with ServiceContainer() as container:
            assert container is not None

        # Container should be disposed after exit
        assert container._disposed is True


# =============================================================================
# Service Registration Tests
# =============================================================================


class TestServiceRegistration:
    """Tests for service registration."""

    def test_register_singleton_service(self):
        """Test registering a singleton service."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )

        assert container.is_registered(ILogger)

    def test_register_scoped_service(self):
        """Test registering a scoped service."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SCOPED,
        )

        assert container.is_registered(ILogger)

    def test_register_transient_service(self):
        """Test registering a transient service."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.TRANSIENT,
        )

        assert container.is_registered(ILogger)

    def test_register_with_default_lifetime(self):
        """Test registering service with default lifetime (singleton)."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())

        assert container.is_registered(ILogger)

    def test_register_returns_container(self):
        """Test registration returns container for chaining."""
        container = ServiceContainer()

        result = container.register(ILogger, lambda c: FileLogger())

        assert result is container

    def test_register_instance(self):
        """Test registering an existing instance."""
        container = ServiceContainer()
        logger = FileLogger()

        container.register_instance(ILogger, logger)

        assert container.is_registered(ILogger)
        resolved = container.get(ILogger)
        assert resolved is logger

    def test_register_instance_returns_container(self):
        """Test register_instance returns container for chaining."""
        container = ServiceContainer()
        logger = FileLogger()

        result = container.register_instance(ILogger, logger)

        assert result is container

    def test_register_duplicate_raises_error(self):
        """Test registering duplicate service raises error."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())

        with pytest.raises(ServiceAlreadyRegisteredError):
            container.register(ILogger, lambda c: ConsoleLogger())

    def test_register_or_replace(self):
        """Test register_or_replace replaces existing service."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())
        container.register_or_replace(ILogger, lambda c: ConsoleLogger())

        assert container.is_registered(ILogger)

    def test_register_or_replace_returns_container(self):
        """Test register_or_replace returns container for chaining."""
        container = ServiceContainer()

        result = container.register_or_replace(
            ILogger, lambda c: FileLogger()
        )

        assert result is container


# =============================================================================
# Service Resolution Tests
# =============================================================================


class TestServiceResolution:
    """Tests for service resolution."""

    def test_resolve_singleton_service(self):
        """Test resolving singleton service."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )

        logger = container.get(ILogger)

        assert isinstance(logger, FileLogger)

    def test_resolve_singleton_returns_same_instance(self):
        """Test singleton returns same instance on multiple gets."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )

        logger1 = container.get(ILogger)
        logger2 = container.get(ILogger)

        assert logger1 is logger2

    def test_resolve_transient_returns_new_instance(self):
        """Test transient returns new instance on each get."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.TRANSIENT,
        )

        logger1 = container.get(ILogger)
        logger2 = container.get(ILogger)

        assert logger1 is not logger2

    def test_resolve_service_with_dependencies(self):
        """Test resolving service with dependencies."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )
        container.register(
            ICache,
            lambda c: SimpleCache(),
            ServiceLifetime.SINGLETON,
        )
        container.register(
            ServiceWithDependencies,
            lambda c: ServiceWithDependencies(
                logger=c.get(ILogger), cache=c.get(ICache)
            ),
            ServiceLifetime.TRANSIENT,
        )

        service = container.get(ServiceWithDependencies)

        assert isinstance(service, ServiceWithDependencies)
        assert isinstance(service.logger, FileLogger)
        assert isinstance(service.cache, SimpleCache)

    def test_get_optional_returns_service_when_registered(self):
        """Test get_optional returns service when registered."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())

        logger = container.get_optional(ILogger)

        assert logger is not None
        assert isinstance(logger, FileLogger)

    def test_get_optional_returns_none_when_not_registered(self):
        """Test get_optional returns None when not registered."""
        container = ServiceContainer()

        logger = container.get_optional(ILogger)

        assert logger is None

    def test_get_service_alias(self):
        """Test get_service alias for get()."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())

        logger = container.get_service(ILogger)

        assert isinstance(logger, FileLogger)

    def test_resolve_unregistered_service_raises_error(self):
        """Test resolving unregistered service raises error."""
        container = ServiceContainer()

        with pytest.raises(ServiceNotFoundError):
            container.get(ILogger)

    def test_is_registered_returns_true_when_registered(self):
        """Test is_registered returns True when service registered."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())

        assert container.is_registered(ILogger) is True

    def test_is_registered_returns_false_when_not_registered(self):
        """Test is_registered returns False when service not registered."""
        container = ServiceContainer()

        assert container.is_registered(ILogger) is False


# =============================================================================
# Service Lifetime Tests
# =============================================================================


class TestServiceLifetime:
    """Tests for service lifetime behavior."""

    def test_singleton_lifetime_same_instance(self):
        """Test singleton lifetime returns same instance."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )

        logger1 = container.get(ILogger)
        logger2 = container.get(ILogger)

        assert logger1 is logger2

    def test_scoped_lifetime_in_root_container(self):
        """Test scoped lifetime in root container acts like singleton."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SCOPED,
        )

        logger1 = container.get(ILogger)
        logger2 = container.get(ILogger)

        assert logger1 is logger2

    def test_transient_lifetime_different_instances(self):
        """Test transient lifetime returns different instances."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.TRANSIENT,
        )

        logger1 = container.get(ILogger)
        logger2 = container.get(ILogger)

        assert logger1 is not logger2
        assert isinstance(logger1, FileLogger)
        assert isinstance(logger2, FileLogger)

    def test_factory_receives_container(self):
        """Test factory function receives container."""
        container = ServiceContainer()
        received_container: list[ServiceContainer] = []

        def factory(c: ServiceContainer) -> FileLogger:
            received_container.append(c)
            return FileLogger()

        container.register(ILogger, factory)

        container.get(ILogger)

        assert len(received_container) == 1
        assert received_container[0] is container

    def test_disposable_singleton_disposed_on_container_dispose(self):
        """Test disposable singleton is disposed when container disposed."""
        container = ServiceContainer()

        container.register(
            DisposableService,
            lambda c: DisposableService(),
            ServiceLifetime.SINGLETON,
        )

        service = container.get(DisposableService)

        container.dispose()

        assert service.disposed is True

    def test_non_disposable_service_no_error_on_dispose(self):
        """Test non-disposable service doesn't error on dispose."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )

        container.get(ILogger)

        # Should not raise exception
        container.dispose()

    def test_register_or_replace_disposes_existing_singleton(self):
        """Test register_or_replace disposes existing disposable singleton."""
        container = ServiceContainer()

        container.register(
            DisposableService,
            lambda c: DisposableService(),
            ServiceLifetime.SINGLETON,
        )

        service1 = container.get(DisposableService)

        container.register_or_replace(
            DisposableService,
            lambda c: DisposableService(),
        )

        service2 = container.get(DisposableService)

        assert service1.disposed is True
        assert service2.disposed is False


# =============================================================================
# Scoped Services Tests
# =============================================================================


class TestScopedServices:
    """Tests for scoped service lifetime."""

    def test_create_scope(self):
        """Test creating a scope."""
        container = ServiceContainer()

        scope = container.create_scope()

        assert scope is not None
        assert scope._disposed is False

    def test_scoped_service_in_scope(self):
        """Test scoped service in scope returns same instance."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SCOPED,
        )

        with container.create_scope() as scope:
            logger1 = scope.get(ILogger)
            logger2 = scope.get(ILogger)

            assert logger1 is logger2

    def test_scoped_service_different_scopes(self):
        """Test scoped service returns different instances in different scopes."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SCOPED,
        )

        with container.create_scope() as scope1:
            logger1 = scope1.get(ILogger)

        with container.create_scope() as scope2:
            logger2 = scope2.get(ILogger)

        assert logger1 is not logger2

    def test_singleton_from_scope(self):
        """Test singleton resolved from scope is same as root."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )

        root_logger = container.get(ILogger)

        with container.create_scope() as scope:
            scope_logger = scope.get(ILogger)

        assert root_logger is scope_logger

    def test_transient_from_scope(self):
        """Test transient returns new instance even in scope."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.TRANSIENT,
        )

        with container.create_scope() as scope:
            logger1 = scope.get(ILogger)
            logger2 = scope.get(ILogger)

        assert logger1 is not logger2

    def test_disposed_scope_raises_error(self):
        """Test disposed scope raises error on access."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SCOPED,
        )

        scope = container.create_scope()
        scope.dispose()

        with pytest.raises(ScopeDisposedError):
            scope.get(ILogger)

    def test_scope_disposes_disposable_services(self):
        """Test scope disposes disposable scoped services."""
        container = ServiceContainer()

        container.register(
            DisposableService,
            lambda c: DisposableService(),
            ServiceLifetime.SCOPED,
        )

        with container.create_scope() as scope:
            service = scope.get(DisposableService)
            assert service.disposed is False

        # After scope exit, service should be disposed
        assert service.disposed is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_service_not_found_error_message(self):
        """Test ServiceNotFoundError has informative message."""
        container = ServiceContainer()

        with pytest.raises(ServiceNotFoundError) as exc_info:
            container.get(ILogger)

        assert "ILogger" in str(exc_info.value)

    def test_service_already_registered_error_message(self):
        """Test ServiceAlreadyRegisteredError has informative message."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())

        with pytest.raises(ServiceAlreadyRegisteredError) as exc_info:
            container.register(ILogger, lambda c: ConsoleLogger())

        assert "ILogger" in str(exc_info.value)

    def test_scope_disposed_error_message(self):
        """Test ScopeDisposedError has informative message."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SCOPED,
        )

        scope = container.create_scope()
        scope.dispose()

        with pytest.raises(ScopeDisposedError) as exc_info:
            scope.get(ILogger)

        assert "ILogger" in str(exc_info.value)

    def test_dispose_error_in_dispose_is_caught(self):
        """Test errors in disposable.dispose are caught and logged."""
        container = ServiceContainer()

        class BrokenDisposable(Disposable):
            def dispose(self) -> None:
                raise RuntimeError("Dispose failed")

        container.register(
            BrokenDisposable,
            lambda c: BrokenDisposable(),
            ServiceLifetime.SINGLETON,
        )

        container.get(BrokenDisposable)

        # Should not raise exception, error is caught and logged
        container.dispose()


# =============================================================================
# Global Container Tests
# =============================================================================


class TestGlobalContainer:
    """Tests for global container management."""

    def test_get_container_creates_container(self):
        """Test get_container creates new container if none exists."""
        # Reset global container first
        reset_container()

        container = get_container()

        assert container is not None
        assert isinstance(container, ServiceContainer)

    def test_get_container_returns_same_instance(self):
        """Test get_container returns same instance on subsequent calls."""
        reset_container()

        container1 = get_container()
        container2 = get_container()

        assert container1 is container2

    def test_set_container_sets_global(self):
        """Test set_container sets global container."""
        reset_container()

        new_container = ServiceContainer()
        set_container(new_container)

        result = get_container()

        assert result is new_container

    def test_set_container_disposes_existing(self):
        """Test set_container disposes existing container."""
        reset_container()

        container1 = get_container()
        container2 = ServiceContainer()

        set_container(container2)

        result = get_container()

        assert result is container2
        assert container1._disposed is True

    def test_reset_container_clears_global(self):
        """Test reset_container clears global container."""
        reset_container()

        container = get_container()
        reset_container()

        # Should create new container
        new_container = get_container()

        assert new_container is not container

    def test_get_registered_types_returns_list(self):
        """Test get_registered_types returns list of types."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())
        container.register(ICache, lambda c: SimpleCache())

        types = container.get_registered_types()

        assert len(types) == 2
        assert ILogger in types
        assert ICache in types

    def test_get_registered_types_returns_empty_list(self):
        """Test get_registered_types returns empty list for empty container."""
        container = ServiceContainer()

        types = container.get_registered_types()

        assert len(types) == 0


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe container operations."""

    def test_concurrent_singleton_resolution(self):
        """Test thread-safe singleton resolution from multiple threads."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )

        results: list[FileLogger] = []
        threads = []

        def resolve_service():
            logger = container.get(ILogger)
            results.append(logger)

        # Create multiple threads that resolve the same service
        for _ in range(10):
            thread = threading.Thread(target=resolve_service)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All threads should get the same instance
        assert len(results) == 10
        assert all(r is results[0] for r in results)

    def test_concurrent_transient_resolution(self):
        """Test thread-safe transient resolution from multiple threads."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.TRANSIENT,
        )

        results: list[Any] = []
        threads = []

        def resolve_service():
            logger = container.get(ILogger)
            results.append(logger)

        # Create multiple threads that resolve the same service
        for _ in range(10):
            thread = threading.Thread(target=resolve_service)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Each thread should get a different instance
        assert len(results) == 10
        # Check all instances are different
        unique_instances = set(id(r) for r in results)
        assert len(unique_instances) == 10

    def test_concurrent_registration(self):
        """Test thread-safe service registration from multiple threads."""
        container = ServiceContainer()
        exceptions: list[Exception] = []

        def register_service(index: int):
            try:
                container.register(
                    type(f"IService{index}", (), {}),
                    lambda c, i=index: Mock(name=f"Service{i}"),
                    ServiceLifetime.SINGLETON,
                )
            except Exception as e:
                exceptions.append(e)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_service, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # No exceptions should have been raised
        assert len(exceptions) == 0

        # All services should be registered
        assert len(container.get_registered_types()) == 10

    def test_concurrent_singleton_initialization(self):
        """Test thread-safe singleton initialization (only one instance created)."""
        container = ServiceContainer()
        init_count = {"value": 0}

        class CountingLogger(ILogger):
            def __init__(self) -> None:
                # Increment counter to track how many times init is called
                init_count["value"] += 1
                self.messages: list[str] = []

            def log(self, message: str) -> None:
                self.messages.append(message)

        container.register(
            ILogger,
            lambda c: CountingLogger(),
            ServiceLifetime.SINGLETON,
        )

        results: list[Any] = []
        threads = []

        def resolve_service():
            logger = container.get(ILogger)
            results.append(logger)

        # Create multiple threads that race to initialize the singleton
        for _ in range(20):
            thread = threading.Thread(target=resolve_service)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Despite 20 threads, only one instance should be created
        assert init_count["value"] == 1
        assert len(results) == 20
        assert all(r is results[0] for r in results)

    def test_concurrent_scoped_isolation(self):
        """Test that scoped services are isolated between threads."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SCOPED,
        )

        results: list[Any] = []
        threads = []

        def resolve_in_scope():
            with container.create_scope() as scope:
                logger = scope.get(ILogger)
                results.append(logger)
                # Simulate some work
                time.sleep(0.001)

        for _ in range(10):
            thread = threading.Thread(target=resolve_in_scope)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Each scope should have its own instance
        assert len(results) == 10
        unique_instances = set(id(r) for r in results)
        # All scopes should have different instances
        assert len(unique_instances) == 10

    def test_thread_singleton_during_concurrent_get(self):
        """Test singleton is thread-safe during concurrent get operations."""
        container = ServiceContainer()
        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )

        # Pre-initialize the singleton
        first = container.get(ILogger)

        results: list[Any] = []

        def concurrent_get():
            for _ in range(100):
                logger = container.get(ILogger)
                results.append(logger)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=concurrent_get)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All 500 gets (5 threads * 100 iterations) should return the same instance
        assert len(results) == 500
        assert all(r is first for r in results)

    def test_concurrent_dispose(self):
        """Test that concurrent dispose calls are safe."""
        container = ServiceContainer()

        container.register(
            DisposableService,
            lambda c: DisposableService(),
            ServiceLifetime.SINGLETON,
        )

        # Get the service to initialize it
        service = container.get(DisposableService)

        exceptions: list[Exception] = []

        def dispose_container():
            try:
                container.dispose()
            except Exception as e:
                exceptions.append(e)

        # Try to dispose from multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=dispose_container)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # No exceptions should have been raised
        assert len(exceptions) == 0
        assert container._disposed is True

    def test_concurrent_register_or_replace(self):
        """Test thread-safe register_or_replace operations."""
        container = ServiceContainer()

        container.register(
            ILogger,
            lambda c: FileLogger(),
            ServiceLifetime.SINGLETON,
        )

        results: list[Any] = []
        exceptions: list[Exception] = []

        def replace_and_get(index: int):
            try:
                # Replace with a new implementation
                container.register_or_replace(
                    ILogger,
                    lambda c, i=index: Mock(name=f"Logger{i}"),
                    ServiceLifetime.SINGLETON,
                )
                # Get the service
                logger = container.get(ILogger)
                results.append(logger)
            except Exception as e:
                exceptions.append(e)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=replace_and_get, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # No exceptions should have been raised
        assert len(exceptions) == 0
        assert len(results) == 10

        # All results should be mock loggers
        assert all(isinstance(r, Mock) for r in results)

    def test_concurrent_scope_disposal(self):
        """Test thread-safe scope disposal."""
        container = ServiceContainer()

        container.register(
            DisposableService,
            lambda c: DisposableService(),
            ServiceLifetime.SCOPED,
        )

        exceptions: list[Exception] = []
        services: list[tuple[DisposableService, DisposableService]] = []

        def create_and_dispose_scope():
            try:
                with container.create_scope() as scope:
                    service = scope.get(DisposableService)
                    # Store reference before disposal
                    before_disposal = service.disposed
                    time.sleep(0.001)
                # After context exit, check if disposed
                after_disposal = service.disposed
                services.append((before_disposal, after_disposal))
            except Exception as e:
                exceptions.append(e)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_and_dispose_scope)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # No exceptions should have been raised
        assert len(exceptions) == 0

        # We should have 10 service checks
        assert len(services) == 10

        # All before_disposal should be False
        assert all(before is False for before, after in services)

        # All after_disposal should be True
        assert all(after is True for before, after in services)

    def test_concurrent_optional_resolution(self):
        """Test thread-safe get_optional operations."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())

        results: list[Any] = []

        def get_optional_service():
            # Try to get registered service
            logger = container.get_optional(ILogger)
            results.append(logger)

            # Try to get unregistered service
            unregistered = container.get_optional(ICache)
            results.append(unregistered)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_optional_service)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have 20 results (10 threads * 2 checks each)
        assert len(results) == 20

        # Every other result should be None (unregistered service)
        for i in range(0, 20, 2):
            assert isinstance(results[i], FileLogger)
        for i in range(1, 20, 2):
            assert results[i] is None

    def test_is_registered_concurrent(self):
        """Test thread-safe is_registered checks."""
        container = ServiceContainer()

        container.register(ILogger, lambda c: FileLogger())

        results: list[bool] = []

        def check_registrations():
            results.append(container.is_registered(ILogger))
            results.append(container.is_registered(ICache))

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=check_registrations)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have 20 results (10 threads * 2 checks each)
        assert len(results) == 20

        # Every other result should be True (ILogger is registered)
        for i in range(0, 20, 2):
            assert results[i] is True
        for i in range(1, 20, 2):
            assert results[i] is False

    def test_concurrent_container_creation_disposal(self):
        """Test that multiple containers can be created and disposed concurrently."""
        containers_created: list[Any] = []
        exceptions: list[Exception] = []

        def create_and_dispose_container(index: int):
            try:
                container = ServiceContainer()
                container.register(
                    ILogger,
                    lambda c, i=index: Mock(name=f"Logger{i}"),
                )
                containers_created.append(container)
                container.dispose()
            except Exception as e:
                exceptions.append(e)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_and_dispose_container, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # No exceptions should have been raised
        assert len(exceptions) == 0
        assert len(containers_created) == 10

        # All containers should be disposed
        assert all(c._disposed for c in containers_created)
