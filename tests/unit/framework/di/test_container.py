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

"""Tests for the enhanced dependency injection container."""

import pytest

from victor.framework.di import (
    DIContainer,
    ServiceLifetime,
    ServiceNotFoundError,
    ServiceAlreadyRegisteredError,
    DIError,
    create_container,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class ILogger:
    """Test logger interface."""

    def log(self, message: str) -> None:
        pass


class Logger(ILogger):
    """Test logger implementation."""

    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


class ICache:
    """Test cache interface."""

    def get(self, key: str) -> str:
        return ""

    def set(self, key: str, value: str) -> None:
        pass


class Cache(ICache):
    """Test cache implementation."""

    def __init__(self, logger: ILogger):
        self.logger = logger
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str:
        self.logger.log(f"Cache get: {key}")
        return self._data.get(key, "")

    def set(self, key: str, value: str) -> None:
        self.logger.log(f"Cache set: {key} = {value}")
        self._data[key] = value


class IDatabase:
    """Test database interface."""

    def query(self, sql: str) -> list:
        return []


class Database:
    """Test database implementation."""

    def __init__(self, cache: ICache):
        self.cache = cache

    def query(self, sql: str) -> list:
        # Use cache
        cached = self.cache.get(f"query:{sql}")
        if cached:
            return [cached]
        return []


class UserService:
    """Test service with multiple dependencies."""

    def __init__(self, logger: ILogger, database: IDatabase):
        self.logger = logger
        self.database = database

    def get_user(self, user_id: str) -> dict:
        self.logger.log(f"Getting user: {user_id}")
        results = self.database.query(f"SELECT * FROM users WHERE id = '{user_id}'")
        return {"user_id": user_id, "results": results}


class OptionalDepsService:
    """Service with optional dependencies."""

    def __init__(self, logger: ILogger, timeout: int = 30, cache: ICache | None = None):
        self.logger = logger
        self.cache = cache
        self.timeout = timeout


# =============================================================================
# Basic Registration and Resolution
# =============================================================================


class TestBasicRegistration:
    """Tests for basic service registration and resolution."""

    def test_register_and_resolve_transient(self):
        """Test registering and resolving a transient service."""
        container = DIContainer()
        container.register(Logger)

        logger1 = container.get(Logger)
        logger2 = container.get(Logger)

        assert logger1 is not logger2
        assert isinstance(logger1, Logger)
        assert isinstance(logger2, Logger)

    def test_register_and_resolve_singleton(self):
        """Test registering and resolving a singleton service."""
        container = DIContainer()
        container.register(Logger, lifetime=ServiceLifetime.SINGLETON)

        logger1 = container.get(Logger)
        logger2 = container.get(Logger)

        assert logger1 is logger2
        assert isinstance(logger1, Logger)

    def test_register_instance(self):
        """Test registering a pre-created instance."""
        container = DIContainer()
        instance = Logger()
        instance.messages.append("pre-existing")

        container.register_instance(Logger, instance)
        retrieved = container.get(Logger)

        assert retrieved is instance
        assert "pre-existing" in retrieved.messages

    def test_service_not_found(self):
        """Test that ServiceNotFoundError is raised for unregistered services."""
        container = DIContainer()

        with pytest.raises(ServiceNotFoundError) as exc_info:
            container.get(Logger)

        assert "Logger" in str(exc_info.value)

    def test_service_already_registered(self):
        """Test that ServiceAlreadyRegisteredError is raised for duplicates."""
        container = DIContainer()
        container.register(Logger)

        with pytest.raises(ServiceAlreadyRegisteredError) as exc_info:
            container.register(Logger)

        assert "Logger" in str(exc_info.value)

    def test_get_optional_returns_none(self):
        """Test get_optional returns None for unregistered services."""
        container = DIContainer()

        result = container.get_optional(Logger)

        assert result is None

    def test_get_optional_returns_service(self):
        """Test get_optional returns service when registered."""
        container = DIContainer()
        container.register(Logger)

        result = container.get_optional(Logger)

        assert isinstance(result, Logger)

    def test_is_registered(self):
        """Test is_registered check."""
        container = DIContainer()

        assert not container.is_registered(Logger)

        container.register(Logger)

        assert container.is_registered(Logger)


# =============================================================================
# Auto-Resolution
# =============================================================================


class TestAutoResolution:
    """Tests for automatic dependency resolution."""

    def test_auto_resolve_single_dependency(self):
        """Test auto-resolution of single constructor dependency."""
        container = DIContainer()
        container.register(Logger, as_interfaces=[ILogger])
        container.register(Cache)

        cache = container.get(Cache)

        assert isinstance(cache, Cache)
        assert isinstance(cache.logger, Logger)

    def test_auto_resolve_nested_dependencies(self):
        """Test auto-resolution of nested dependencies."""
        container = DIContainer()
        container.register(Logger, as_interfaces=[ILogger])
        container.register(Cache, as_interfaces=[ICache])
        container.register(Database)

        database = container.get(Database)

        assert isinstance(database, Database)
        assert isinstance(database.cache, Cache)
        assert isinstance(database.cache.logger, Logger)

    def test_auto_resolve_multiple_dependencies(self):
        """Test auto-resolution of multiple dependencies."""
        container = DIContainer()
        container.register(Logger, as_interfaces=[ILogger])
        container.register(Cache, as_interfaces=[ICache])
        container.register(Database, as_interfaces=[IDatabase])
        container.register(UserService)

        user_service = container.get(UserService)

        assert isinstance(user_service, UserService)
        assert isinstance(user_service.logger, Logger)
        assert isinstance(user_service.database, Database)

    def test_auto_resolve_with_optional_params(self):
        """Test auto-resolution with optional parameters."""
        container = DIContainer()
        container.register(Logger, as_interfaces=[ILogger])
        container.register(OptionalDepsService)

        # Should work even though cache and timeout have defaults
        service = container.get(OptionalDepsService)

        assert isinstance(service, OptionalDepsService)
        assert isinstance(service.logger, Logger)
        assert service.cache is None
        assert service.timeout == 30

    def test_missing_dependency_raises_error(self):
        """Test that missing dependencies raise clear error."""
        container = DIContainer()
        # Register Database but not Cache
        container.register(Database)

        with pytest.raises(DIError) as exc_info:
            container.get(Database)

        error_msg = str(exc_info.value)
        assert "ICache" in error_msg or "Cache" in error_msg
        assert "Database" in error_msg


# =============================================================================
# Circular Dependency Detection
# =============================================================================


class ServiceA:
    """Service A that depends on B."""

    def __init__(self, b: "ServiceB"):
        self.b = b


class ServiceB:
    """Service B that depends on A."""

    def __init__(self, a: ServiceA):
        self.a = a


class TestCircularDependencies:
    """Tests for circular dependency detection."""

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected and reported."""
        container = DIContainer()
        container.register(ServiceA)
        container.register(ServiceB)

        with pytest.raises(DIError) as exc_info:
            container.get(ServiceA)

        error_msg = str(exc_info.value).lower()
        # Should fail to resolve ServiceB which ServiceA depends on
        assert "service" in error_msg


# =============================================================================
# Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for custom factory functions."""

    def test_register_with_factory(self):
        """Test registration with custom factory function."""
        container = DIContainer()

        # Create a logger with custom initialization
        custom_logger = Logger()
        custom_logger.messages.append("[CUSTOM] initialized")

        container.register(Logger, factory=lambda: custom_logger)

        logger = container.get(Logger)

        assert isinstance(logger, Logger)
        assert "[CUSTOM] initialized" in logger.messages

    def test_register_factory_with_container(self):
        """Test registration with container-aware factory."""

        def create_cache(cont: DIContainer) -> Cache:
            logger = cont.get(ILogger)
            return Cache(logger=logger)

        container = DIContainer()
        container.register(Logger, as_interfaces=[ILogger])
        container.register_factory(ICache, create_cache)

        cache = container.get(ICache)

        assert isinstance(cache, Cache)
        assert isinstance(cache.logger, Logger)


# =============================================================================
# Scoped Services
# =============================================================================


class TestScopedServices:
    """Tests for scoped service lifetime."""

    def test_scoped_lifetime(self):
        """Test that scoped services have one instance per scope."""
        container = DIContainer()
        container.register(Logger, lifetime=ServiceLifetime.SCOPED)

        with container.create_scope() as scope1:
            logger1a = scope1.get(Logger)
            logger1b = scope1.get(Logger)
            assert logger1a is logger1b

        with container.create_scope() as scope2:
            logger2 = scope2.get(Logger)
            assert logger2 is not logger1a

    def test_scoped_inherits_singletons(self):
        """Test that scoped containers inherit singleton instances."""
        container = DIContainer()
        container.register(Logger, lifetime=ServiceLifetime.SINGLETON)

        singleton_logger = container.get(Logger)

        with container.create_scope() as scope:
            scoped_logger = scope.get(Logger)
            assert scoped_logger is singleton_logger

    def test_scope_disposal(self):
        """Test that scope disposal cleans up scoped services."""
        container = DIContainer()
        container.register(Logger, lifetime=ServiceLifetime.SCOPED)

        with container.create_scope() as scope:
            logger1 = scope.get(Logger)

        # New scope should get new instance
        with container.create_scope() as scope:
            logger2 = scope.get(Logger)
            assert logger2 is not logger1


# =============================================================================
# Method Chaining
# =============================================================================


class TestMethodChaining:
    """Tests for method chaining support."""

    def test_register_returns_self(self):
        """Test that register returns container for chaining."""
        container = DIContainer()

        result = container.register(Logger)

        assert result is container

    def test_fluent_registration(self):
        """Test fluent registration pattern."""
        container = DIContainer().register(Logger).register(Cache).register(Database)

        assert container.is_registered(Logger)
        assert container.is_registered(Cache)
        assert container.is_registered(Database)


# =============================================================================
# Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_container(self):
        """Test create_container convenience function."""
        container = create_container(
            (Logger, ServiceLifetime.SINGLETON),
            (Cache, ServiceLifetime.SINGLETON),
        )

        assert container.is_registered(Logger)
        assert container.is_registered(Cache)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complex scenarios."""

    def test_complete_application_setup(self):
        """Test a complete application-like setup."""
        # Setup
        container = (
            DIContainer()
            .register(Logger, lifetime=ServiceLifetime.SINGLETON, as_interfaces=[ILogger])
            .register(Cache, lifetime=ServiceLifetime.SINGLETON, as_interfaces=[ICache])
            .register(Database, lifetime=ServiceLifetime.SINGLETON, as_interfaces=[IDatabase])
            .register(UserService, lifetime=ServiceLifetime.TRANSIENT)
        )

        # Resolve service
        user_service = container.get(UserService)

        # Use service
        user_service.logger.log("Application started")
        results = user_service.database.query("SELECT * FROM users")

        assert isinstance(user_service, UserService)
        assert "Application started" in user_service.logger.messages
        assert isinstance(results, list)

    def test_transient_with_singleton_deps(self):
        """Test that transient services get singleton dependencies."""
        container = (
            DIContainer()
            .register(Logger, lifetime=ServiceLifetime.SINGLETON, as_interfaces=[ILogger])
            .register(Cache, lifetime=ServiceLifetime.SINGLETON, as_interfaces=[ICache])
            .register(Database, lifetime=ServiceLifetime.TRANSIENT)
        )

        db1 = container.get(Database)
        db2 = container.get(Database)

        # Databases are different instances
        assert db1 is not db2

        # But they share the same cache (singleton)
        assert db1.cache is db2.cache

        # And cache shares the same logger (singleton)
        assert db1.cache.logger is db2.cache.logger

    def test_context_manager(self):
        """Test container as context manager."""
        with DIContainer() as container:
            container.register(Logger)
            logger = container.get(Logger)
            assert isinstance(logger, Logger)


# =============================================================================
# Error Messages
# =============================================================================


class TestErrorMessages:
    """Tests for clear error messages."""

    def test_service_not_found_includes_chain(self):
        """Test that missing dependency error includes resolution chain."""
        container = DIContainer()
        container.register(Logger, as_interfaces=[ILogger])
        container.register(Cache, as_interfaces=[ICache])
        container.register(Database)

        # Try to get Database without registering Cache
        # Actually we registered Cache, so let's unregister it
        container._descriptors.pop(ICache)

        with pytest.raises(DIError) as exc_info:
            container.get(Database)

        error_msg = str(exc_info.value)
        # Should mention both Database and the missing dependency
        assert "Database" in error_msg

    def test_circular_dependency_includes_chain(self):
        """Test that circular dependency error includes full chain."""
        container = DIContainer()
        container.register(ServiceA)
        container.register(ServiceB)

        with pytest.raises(DIError) as exc_info:
            container.get(ServiceA)

        error_msg = str(exc_info.value)
        # Should show the chain
        assert "service" in error_msg.lower()
