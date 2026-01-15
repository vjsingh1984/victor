#!/usr/bin/env python
"""Dependency Injection Container Usage Example.

This example shows how to:
1. Register services with different lifetimes (singleton, scoped, transient)
2. Resolve services with auto-injection of dependencies
3. Use scoped containers for request isolation
4. Override services for testing

Before Refactoring (Manual DI):
    class Orchestrator:
        def __init__(self):
            self.metrics = MetricsCollector()  # Hard dependency
            self.logger = DebugLogger()  # Hard dependency
            self.cache = RedisCache()  # Hard dependency

    # Hard to test, hard to configure, shared state issues

After Refactoring (DI Container):
    container = ServiceContainer()
    container.register(MetricsService, lambda c: MetricsCollector(), SINGLETON)
    container.register(LoggerService, lambda c: DebugLogger(), SINGLETON)
    container.register(CacheService, lambda c: RedisCache(c.get(LoggerService)), SINGLETON)

    orchestrator = container.get(Orchestrator)

    # Easy to test, flexible configuration, proper lifecycle management
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.core.container import (
    ServiceContainer,
    ServiceLifetime,
    Disposable,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: Define Service Protocols
# =============================================================================


class MetricsService(ABC):
    """Abstract base class for metrics collection."""

    @abstractmethod
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        pass

    @abstractmethod
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        pass


class LoggerService(ABC):
    """Abstract base class for logging."""

    @abstractmethod
    def log(self, message: str, level: str = "info", **kwargs: Any) -> None:
        """Log a message."""
        pass

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Check if logging is enabled."""
        pass


class CacheService(ABC):
    """Abstract base class for caching."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a cached value."""
        pass

    @abstractmethod
    def invalidate(self, key: str) -> None:
        """Invalidate a cached value."""
        pass


# =============================================================================
# PART 2: Implement Services
# =============================================================================


class MetricsCollector(MetricsService):
    """Concrete metrics collector implementation."""

    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append(value)
        logger.debug(f"Recorded metric: {name}={value}")

    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + 1
        logger.debug(f"Incremented counter: {name}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            return {
                "metrics": dict(self._metrics),
                "counters": dict(self._counters),
            }


class DebugLogger(LoggerService):
    """Concrete debug logger implementation."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self._enabled = True
        self._message_count = 0

    def log(self, message: str, level: str = "info", **kwargs: Any) -> None:
        """Log a message."""
        if not self._enabled:
            return

        self._message_count += 1
        log_msg = f"[{level.upper()}] {message}"
        logger.info(log_msg)

        if kwargs:
            logger.info(f"  Context: {kwargs}")

    @property
    def enabled(self) -> bool:
        """Check if logging is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable logging."""
        self._enabled = True

    def disable(self) -> None:
        """Disable logging."""
        self._enabled = False


class InMemoryCache(CacheService, Disposable):
    """Concrete in-memory cache implementation."""

    def __init__(self, ttl_seconds: int = 300):
        self._store: Dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        if key not in self._store:
            self._misses += 1
            return None

        value, expiry = self._store[key]

        # Check if expired
        if time.time() > expiry:
            del self._store[key]
            self._misses += 1
            return None

        self._hits += 1
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a cached value."""
        ttl = ttl or self._ttl_seconds
        expiry = time.time() + ttl
        self._store[key] = (value, expiry)

    def invalidate(self, key: str) -> None:
        """Invalidate a cached value."""
        if key in self._store:
            del self._store[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._store),
        }

    def dispose(self) -> None:
        """Clear cache and release resources."""
        logger.info("Disposing InMemoryCache...")
        self._store.clear()
        self._hits = 0
        self._misses = 0


class RequestScopedService:
    """Example of a scoped service (one instance per request)."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self._start_time = time.time()

    def get_elapsed(self) -> float:
        """Get elapsed time since creation."""
        return time.time() - self._start_time


# =============================================================================
# PART 3: Service with Dependencies
# =============================================================================


class ToolExecutor:
    """Example service that depends on other services.

    This class demonstrates constructor injection - all dependencies
    are provided via the constructor.
    """

    def __init__(
        self,
        metrics: MetricsService,
        logger: LoggerService,
        cache: Optional[CacheService] = None,
    ):
        """Initialize with dependencies injected by container.

        Args:
            metrics: Metrics service for recording execution metrics
            logger: Logger service for logging execution events
            cache: Optional cache service for caching results
        """
        self.metrics = metrics
        self.logger = logger
        self.cache = cache
        self._execution_count = 0

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with instrumentation.

        This method uses all injected services to demonstrate
        dependency injection in action.
        """
        self._execution_count += 1
        start_time = time.time()

        self.logger.log(
            f"Executing tool: {tool_name}",
            level="info",
            execution_id=self._execution_count,
        )

        # Check cache
        if self.cache:
            cache_key = f"{tool_name}:{hash(str(arguments))}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.logger.log(f"Cache hit for {tool_name}", level="debug")
                self.metrics.increment_counter("tool.cache_hits")
                return cached_result

        # Simulate tool execution
        time.sleep(0.01)  # Simulate work
        result = {"tool": tool_name, "status": "success", "data": arguments}

        # Cache result
        if self.cache:
            self.cache.set(cache_key, result, ttl=60)

        # Record metrics
        elapsed = time.time() - start_time
        self.metrics.record_metric(f"tool.{tool_name}.duration", elapsed)
        self.metrics.increment_counter("tool.executions")

        self.logger.log(
            f"Tool {tool_name} executed in {elapsed:.3f}s",
            level="info",
        )

        return result


# =============================================================================
# PART 4: Container Registration and Resolution
# =============================================================================


def register_singleton_services(container: ServiceContainer) -> None:
    """Register singleton services (one instance for entire application).

    Singleton services are created once and shared across all requests.
    Use for stateless services or those with expensive initialization.
    """
    logger.info("Registering singleton services...")

    # MetricsCollector - shared across application
    container.register(
        MetricsService,
        lambda c: MetricsCollector(),
        ServiceLifetime.SINGLETON,
    )

    # DebugLogger - shared across application
    container.register(
        LoggerService,
        lambda c: DebugLogger(log_file="app.log"),
        ServiceLifetime.SINGLETON,
    )

    # InMemoryCache - shared across application
    container.register(
        CacheService,
        lambda c: InMemoryCache(ttl_seconds=300),
        ServiceLifetime.SINGLETON,
    )


def register_transient_services(container: ServiceContainer) -> None:
    """Register transient services (new instance every time).

    Transient services are created each time they're requested.
    Use for lightweight, stateless services.
    """
    logger.info("Registering transient services...")

    # Example: A simple counter service
    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self) -> int:
            self.count += 1
            return self.count

    container.register(
        Counter,
        lambda c: Counter(),
        ServiceLifetime.TRANSIENT,
    )


def register_composite_services(container: ServiceContainer) -> None:
    """Register services that depend on other services.

    The container automatically resolves and injects dependencies.
    """
    logger.info("Registering composite services...")

    # ToolExecutor depends on MetricsService, LoggerService, and CacheService
    container.register(
        ToolExecutor,
        lambda c: ToolExecutor(
            metrics=c.get(MetricsService),
            logger=c.get(LoggerService),
            cache=c.get(CacheService),
        ),
        ServiceLifetime.SINGLETON,
    )


# =============================================================================
# PART 5: Scoped Services
# =============================================================================


def demonstrate_scoped_services():
    """Demonstrate scoped service lifetime.

    Scoped services are created once per scope. This is useful for
    request-level data or session-specific services.
    """
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Scoped Services")
    logger.info("=" * 70)

    container = ServiceContainer()

    # Register a scoped service
    def create_request_service(c: ServiceContainer) -> RequestScopedService:
        # In real application, request_id would come from request context
        import uuid

        return RequestScopedService(request_id=str(uuid.uuid4()))

    container.register(
        RequestScopedService,
        create_request_service,
        ServiceLifetime.SCOPED,
    )

    # Create first scope (simulating first request)
    logger.info("\n--- Request 1 ---")
    with container.create_scope() as scope1:
        service1a = scope1.get(RequestScopedService)
        service1b = scope1.get(RequestScopedService)
        logger.info(f"Service 1a ID: {service1a.request_id}")
        logger.info(f"Service 1b ID: {service1b.request_id}")
        logger.info(f"Same instance? {service1a is service1b}")  # True

    # Create second scope (simulating second request)
    logger.info("\n--- Request 2 ---")
    with container.create_scope() as scope2:
        service2 = scope2.get(RequestScopedService)
        logger.info(f"Service 2 ID: {service2.request_id}")
        logger.info(f"Different from request 1? {service2 is not service1a}")  # True

    logger.info("\n✓ Each scope gets its own instance!")
    logger.info("✓ Multiple gets within same scope return same instance!")


# =============================================================================
# PART 6: Service Override for Testing
# =============================================================================


class MockMetricsService(MetricsService):
    """Mock metrics service for testing."""

    def __init__(self):
        self._recorded = []

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record metric without I/O."""
        self._recorded.append(("metric", name, value))

    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment counter without I/O."""
        self._recorded.append(("counter", name, 1))

    def get_metrics(self) -> Dict[str, Any]:
        """Return recorded metrics."""
        return {"recorded": self._recorded}


class MockLoggerService(LoggerService):
    """Mock logger service for testing."""

    def __init__(self):
        self._logs = []

    def log(self, message: str, level: str = "info", **kwargs: Any) -> None:
        """Store log without I/O."""
        self._logs.append((level, message, kwargs))

    @property
    def enabled(self) -> bool:
        return True

    def get_logs(self) -> List[tuple]:
        """Get all logged messages."""
        return self._logs


def demonstrate_testing_with_mocks():
    """Demonstrate service override for testing."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Service Override for Testing")
    logger.info("=" * 70)

    container = ServiceContainer()

    # Register production services
    register_singleton_services(container)
    register_composite_services(container)

    # Override with mocks for testing
    logger.info("\n--- Overriding services with mocks ---")
    container.register_or_replace(
        MetricsService,
        lambda c: MockMetricsService(),
        ServiceLifetime.SINGLETON,
    )
    container.register_or_replace(
        LoggerService,
        lambda c: MockLoggerService(),
        ServiceLifetime.SINGLETON,
    )

    # Get service with mocked dependencies
    executor = container.get(ToolExecutor)

    # Execute tool
    result = executor.execute_tool("test_tool", {"arg1": "value1"})

    # Verify mocks recorded calls
    metrics = container.get(MetricsService)
    mock_logger = container.get(LoggerService)

    logger.info(f"\nMock metrics recorded: {len(metrics._recorded)} calls")
    logger.info(f"Mock logger recorded: {len(mock_logger.get_logs())} logs")

    logger.info("\n✓ Tests run without side effects!")
    logger.info("✓ Fast and isolated!")


# =============================================================================
# PART 7: Auto-Resolution of Dependencies
# =============================================================================


def demonstrate_auto_resolution():
    """Demonstrate automatic dependency resolution."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Auto-Resolution of Dependencies")
    logger.info("=" * 70)

    container = ServiceContainer()

    # Register all services
    register_singleton_services(container)
    register_composite_services(container)

    # Resolve ToolExecutor - container automatically resolves dependencies
    logger.info("\nResolving ToolExecutor...")
    executor = container.get(ToolExecutor)

    logger.info(f"✓ ToolExecutor resolved")
    logger.info(f"  - Has metrics? {type(executor.metrics).__name__}")
    logger.info(f"  - Has logger? {type(executor.logger).__name__}")
    logger.info(f"  - Has cache? {type(executor.cache).__name__}")

    # Execute tool
    logger.info("\nExecuting tool...")
    result = executor.execute_tool("code_search", {"query": "test"})

    logger.info(f"\n✓ Tool executed successfully!")
    logger.info(f"  Result: {result}")

    # Show metrics
    metrics = container.get(MetricsService)
    metrics_data = metrics.get_metrics()
    logger.info(f"\n✓ Metrics recorded:")
    logger.info(f"  Counters: {metrics_data['counters']}")

    # Show cache stats
    cache = container.get(CacheService)
    cache_stats = cache.get_stats()
    logger.info(f"\n✓ Cache statistics:")
    logger.info(f"  Hit rate: {cache_stats['hit_rate']:.2%}")


# =============================================================================
# PART 8: Lifecycle Management
# =============================================================================


def demonstrate_lifecycle():
    """Demonstrate service lifecycle management."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION: Lifecycle Management")
    logger.info("=" * 70)

    container = ServiceContainer()

    # Register services
    container.register(
        MetricsService,
        lambda c: MetricsCollector(),
        ServiceLifetime.SINGLETON,
    )
    container.register(
        CacheService,
        lambda c: InMemoryCache(ttl_seconds=300),
        ServiceLifetime.SINGLETON,
    )

    # Use services
    metrics1 = container.get(MetricsService)
    metrics2 = container.get(MetricsService)
    logger.info(f"\nSame metrics instance? {metrics1 is metrics2}")  # True

    cache1 = container.get(CacheService)
    cache2 = container.get(CacheService)
    logger.info(f"Same cache instance? {cache1 is cache2}")  # True

    # Dispose container
    logger.info("\nDisposing container...")
    container.dispose()
    logger.info("✓ Container disposed, all resources cleaned up")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 70)
    logger.info("DEPENDENCY INJECTION CONTAINER USAGE")
    logger.info("=" * 70)

    # Demonstrate auto-resolution
    demonstrate_auto_resolution()

    # Demonstrate scoped services
    demonstrate_scoped_services()

    # Demonstrate testing with mocks
    demonstrate_testing_with_mocks()

    # Demonstrate lifecycle
    demonstrate_lifecycle()

    logger.info("\n" + "=" * 70)
    logger.info("KEY TAKEAWAYS")
    logger.info("=" * 70)
    logger.info("1. Singletons: One instance shared across application")
    logger.info("2. Scoped: One instance per scope/request")
    logger.info("3. Transient: New instance every time")
    logger.info("4. Auto-resolution: Dependencies injected automatically")
    logger.info("5. Testability: Easy to override with mocks")
    logger.info("6. Lifecycle: Proper cleanup with Disposable protocol")
    logger.info("\nRun with: python -m examples.architecture.di_container_usage")


if __name__ == "__main__":
    main()
