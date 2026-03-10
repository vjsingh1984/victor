# Victor Framework Service Creation Guide

**Target Audience**: Victor framework contributors
**Purpose**: Guide for creating new services following SOLID principles
**Status**: Phases 1-7 Complete (Settings Stratification + Service Delegation)

## Table of Contents

1. [Service Architecture Overview](#service-architecture-overview)
2. [Creating a New Service](#creating-a-new-service)
3. [Service Patterns](#service-patterns)
4. [Best Practices](#best-practices)
5. [Examples](#examples)

## Service Architecture Overview

Victor's service architecture follows SOLID principles:

```
┌─────────────────────────────────────────────────────────┐
│                   AgentOrchestrator                     │
│                    (Facade)                            │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
    │ ChatService│  │ToolService│  │ContextSvc │
    │ (Protocol) │  │ (Protocol) │  │ (Protocol) │
    └───────────┘  └───────────┘  └───────────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
                  ┌──────▼───────┐
                  │ DI Container │
                  └──────────────┘
```

### Key Principles

1. **Single Responsibility**: Each service has one job
2. **Protocol-based**: Services implement focused protocols
3. **Feature Flag Controlled**: New services are opt-in
4. **DI Container**: Services are injected, not instantiated directly
5. **Health Checkable**: Services report their health status

## Creating a New Service

### Step 1: Define Protocol

Create a protocol in `victor/agent/services/protocols/`:

```python
# victor/agent/services/protocols/my_service.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

@runtime_checkable
class MyServiceProtocol(Protocol):
    """Protocol for my custom service.

    Provides focused interface for X capability.
    """

    async def do_something(self, input: str) -> str:
        """Do something with the input.

        Args:
            input: The input string

        Returns:
            Processed output string
        """
        ...

    def get_status(self) -> str:
        """Get service status.

        Returns:
            Status string
        """
        ...
```

### Step 2: Implement Service

Create service implementation in `victor/agent/services/my_service.py`:

```python
# victor/agent/services/my_service.py
from __future__ import annotations

import logging
from typing import Optional

from victor.agent.services.protocols.my_service import MyServiceProtocol

logger = logging.getLogger(__name__)


class MyServiceConfig:
    """Configuration for MyService."""

    def __init__(
        self,
        max_retries: int = 3,
        timeout_ms: int = 5000,
    ):
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms


class MyService(MyServiceProtocol):
    """Service for X capability.

    Extracted from AgentOrchestrator to handle:
    - Responsibility 1
    - Responsibility 2

    Example:
        config = MyServiceConfig(max_retries=5)
        service = MyService(config=config)
        result = await service.do_something("input")
    """

    def __init__(
        self,
        config: MyServiceConfig,
        dependency: Optional[SomeDependency] = None,
    ):
        """Initialize MyService.

        Args:
            config: Service configuration
            dependency: Optional dependency
        """
        self._config = config
        self._dependency = dependency
        self._status = "initialized"

    async def do_something(self, input: str) -> str:
        """Do something with the input.

        Args:
            input: The input string

        Returns:
            Processed output string
        """
        try:
            # Implementation here
            result = input.upper()  # Example
            self._status = "success"
            return result
        except Exception as e:
            self._status = f"error: {e}"
            logger.error(f"MyService failed: {e}")
            raise

    def get_status(self) -> str:
        """Get service status.

        Returns:
            Status string
        """
        return self._status

    def is_healthy(self) -> bool:
        """Health check for the service.

        Returns:
            True if service is healthy
        """
        return self._status not in ["error", "failed"]
```

### Step 3: Add Feature Flag

Add flag to `victor/core/feature_flags.py`:

```python
class FeatureFlag(Enum):
    # ... existing flags ...
    USE_NEW_MY_SERVICE = "use_new_my_service"
```

### Step 4: Create Factory Function

Add factory in `victor/core/bootstrap_services.py`:

```python
def _create_my_service(container: ServiceContainer) -> "MyService":
    """Create MyService instance.

    Args:
        container: Service container

    Returns:
        Configured MyService instance
    """
    from victor.agent.services.my_service import MyService, MyServiceConfig

    # Try to get optional dependency
    dependency = container.get_optional(SomeDependencyProtocol)

    config = MyServiceConfig(
        max_retries=5,
        timeout_ms=5000,
    )

    return MyService(config=config, dependency=dependency)
```

Then add to `bootstrap_new_services()`:

```python
def bootstrap_new_services(...):
    # ... existing service bootstrapping ...

    # Bootstrap MyService if enabled
    if feature_flags.is_enabled(FeatureFlag.USE_NEW_MY_SERVICE):
        my_service = _create_my_service(container)
        container.register(MyServiceProtocol, lambda c: my_service, ServiceLifetime.SINGLETON)
        logger.info("Bootstrapped MyService")
```

### Step 5: Write Tests

Create tests in `tests/unit/agent/services/test_my_service.py`:

```python
# tests/unit/agent/services/test_my_service.py
import pytest

from victor.agent.services.my_service import MyService, MyServiceConfig
from victor.agent.services.protocols.my_service import MyServiceProtocol


class TestMyServiceConfig:
    """Tests for MyServiceConfig."""

    def test_initialization(self):
        """Test config initialization with defaults."""
        config = MyServiceConfig()
        assert config.max_retries == 3
        assert config.timeout_ms == 5000

    def test_custom_values(self):
        """Test config with custom values."""
        config = MyServiceConfig(max_retries=5, timeout_ms=10000)
        assert config.max_retries == 5
        assert config.timeout_ms == 10000


class TestMyService:
    """Tests for MyService."""

    @pytest.mark.asyncio
    async def test_do_something(self):
        """Test do_something method."""
        config = MyServiceConfig()
        service = MyService(config=config)

        result = await service.do_something("hello")
        assert result == "HELLO"

    def test_get_status(self):
        """Test get_status method."""
        service = MyService(config=MyServiceConfig())
        assert service.get_status() == "initialized"

    def test_is_healthy(self):
        """Test is_healthy method."""
        service = MyService(config=MyServiceConfig())
        assert service.is_healthy() is True

    @pytest.mark.asyncio
    async def test_protocol_compliance(self):
        """Test that service implements protocol."""
        service = MyService(config=MyServiceConfig())
        assert isinstance(service, MyServiceProtocol)


class TestMyServiceIntegration:
    """Integration tests for MyService."""

    @pytest.mark.asyncio
    async def test_with_dependency(self):
        """Test service with optional dependency."""
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        # Mock dependency
        container.register_instance(SomeDependencyProtocol, mock_dependency)

        service = MyService(
            config=MyServiceConfig(),
            dependency=container.get(SomeDependencyProtocol)
        )

        result = await service.do_something("test")
        assert result == "TEST"
```

## Service Patterns

### Pattern 1: Stateless Service

Service with no internal state:

```python
class StatelessService(ServiceProtocol):
    """Stateless service - all state passed via parameters."""

    async def process(self, data: dict) -> dict:
        """Process data - no side effects."""
        # Pure function
        return {k: v.upper() for k, v in data.items()}
```

### Pattern 2: Stateful Service

Service with internal state:

```python
class StatefulService(ServiceProtocol):
    """Stateful service - maintains internal state."""

    def __init__(self, config: Config):
        self._state = {}
        self._config = config

    async def process(self, data: dict) -> dict:
        """Process data - updates internal state."""
        self._state.update(data)
        return self._state.copy()

    def get_state(self) -> dict:
        """Get current state."""
        return self._state.copy()
```

### Pattern 3: Service with Dependencies

Service that depends on other services:

```python
class DependentService(ServiceProtocol):
    """Service that depends on other services."""

    def __init__(
        self,
        config: Config,
        dependency: Optional[DependencyProtocol] = None,
    ):
        self._config = config
        self._dependency = dependency

    async def process(self, data: dict) -> dict:
        """Process data - uses dependency if available."""
        if self._dependency:
            data = await self._dependency.preprocess(data)
        # Process data
        return data
```

### Pattern 4: Health-Checkable Service

Service that implements health checks:

```python
class HealthCheckableService(ServiceProtocol):
    """Service with health checking."""

    def __init__(self, config: Config):
        self._config = config
        self._healthy = True
        self._last_error = None

    async def do_something(self) -> None:
        """Do something that might fail."""
        try:
            # Operation that might fail
            pass
        except Exception as e:
            self._healthy = False
            self._last_error = str(e)
            raise

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._healthy

    def get_health_details(self) -> dict:
        """Get detailed health information."""
        return {
            "healthy": self._healthy,
            "last_error": self._last_error,
        }
```

### Pattern 5: Cached Service

Service with caching for expensive operations:

```python
class CachedService(ServiceProtocol):
    """Service with result caching."""

    def __init__(self, config: Config):
        self._config = config
        self._cache: dict = {}
        self._cache_ttl = config.cache_ttl

    async def get_expensive_result(self, key: str) -> Any:
        """Get expensive result with caching."""
        # Check cache
        if key in self._cache:
            cached_result, cached_time = self._cache[key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_result

        # Compute result
        result = await self._compute_expensive(key)

        # Cache result
        self._cache[key] = (result, time.time())
        return result
```

## Best Practices

### 1. Use Configuration Objects

```python
# Good
class MyServiceConfig:
    def __init__(self, timeout: int = 5000):
        self.timeout = timeout

service = MyService(config=MyServiceConfig(timeout=10000))

# Avoid
service = MyService(timeout=10000)  # Harder to extend
```

### 2. Provide Factory Functions

```python
# In bootstrap_services.py
def create_my_service(container: ServiceContainer) -> MyService:
    """Factory function for MyService."""
    config = MyServiceConfig.from_settings(container.get(Settings))
    return MyService(config=config)

# Usage
service = create_my_service(container)
```

### 3. Implement Health Checks

```python
def is_healthy(self) -> bool:
    """Health check for the service."""
    return all([
        self._dependency is None or self._dependency.is_healthy(),
        self._status != "error",
        self._connection_active if hasattr(self, '_connection_active') else True,
    ])
```

### 4. Use Dependency Injection

```python
# Good - dependencies injected
class MyService:
    def __init__(
        self,
        config: MyServiceConfig,
        dependency: Optional[DependencyProtocol] = None,
    ):
        self._config = config
        self._dependency = dependency

# Avoid - hardcoded dependencies
class MyService:
    def __init__(self, config: MyServiceConfig):
        self._config = config
        self._dependency = Dependency()  # Hard to test
```

### 5. Handle Optional Dependencies

```python
class MyService:
    def __init__(
        self,
        config: MyServiceConfig,
        dependency: Optional[DependencyProtocol] = None,
    ):
        self._config = config
        self._dependency = dependency

    async def process(self, data):
        if self._dependency:
            # Use dependency
            data = await self._dependency.preprocess(data)
        else:
            # Fallback behavior
            logger.warning("Dependency not available, using fallback")
        # Continue processing
        return data
```

### 6. Write Protocol-Compliant Services

```python
from victor.agent.services.protocols.my_service import MyServiceProtocol

class MyService(MyServiceProtocol):
    # Ensure all protocol methods are implemented

    async def do_something(self, input: str) -> str:
        pass  # Required by protocol

    def get_status(self) -> str:
        pass  # Required by protocol

    # Additional methods allowed
    def internal_helper(self) -> None:
        pass
```

### 7. Use Structured Logging

```python
import logging

logger = logging.getLogger(__name__)

class MyService:
    async def do_something(self, input: str) -> str:
        logger.info(f"Processing input: {input[:50]}...")
        try:
            result = await self._process(input)
            logger.info("Processing completed successfully")
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise
```

## Examples

### Example 1: Simple Stateful Service

```python
# Protocol
@runtime_checkable
class CounterServiceProtocol(Protocol):
    def increment(self) -> int:
        """Increment counter and return value."""
        ...

    def get_count(self) -> int:
        """Get current count."""
        ...

# Implementation
class CounterService(CounterServiceProtocol):
    def __init__(self, config):
        self._count = 0

    def increment(self) -> int:
        self._count += 1
        return self._count

    def get_count(self) -> int:
        return self._count

    def is_healthy(self) -> bool:
        return True
```

### Example 2: Service with External API

```python
# Protocol
@runtime_checkable
class ApiClientServiceProtocol(Protocol):
    async def fetch_data(self, endpoint: str) -> dict:
        """Fetch data from API endpoint."""
        ...

    def is_connected(self) -> bool:
        """Check if connected to API."""
        ...

# Implementation
class ApiClientService(ApiClientServiceProtocol):
    def __init__(self, config: ApiClientConfig):
        self._config = config
        self._session = None

    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def fetch_data(self, endpoint: str) -> dict:
        await self._ensure_session()
        async with self._session.get(
            self._config.base_url + endpoint,
            headers=self._config.headers
        ) as response:
            response.raise_for_status()
            return await response.json()

    def is_connected(self) -> bool:
        return self._session is not None and not self._session.closed

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
```

### Example 3: Service with Retry Logic

```python
# Protocol
@runtime_checkable
class ResilientServiceProtocol(Protocol):
    async def execute_with_retry(self, operation: str) -> Any:
        """Execute operation with retry logic."""
        ...

# Implementation
class ResilientService(ResilientServiceProtocol):
    def __init__(self, config: ResilientServiceConfig):
        self._config = config
        self._retry_count = 0

    async def execute_with_retry(self, operation: str) -> Any:
        for attempt in range(self._config.max_retries):
            try:
                return await self._execute(operation)
            except RetryableError as e:
                self._retry_count += 1
                if attempt == self._config.max_retries - 1:
                    raise
                await asyncio.sleep(self._config.retry_delay_ms / 1000)
        raise MaxRetriesExceededError()
```

## Orchestrator Dual Delegation Pattern

When the `USE_SERVICE_LAYER` feature flag is enabled, the orchestrator delegates to service adapters instead of coordinators. This is the **Strangler Fig pattern** — gradually replacing coordinator calls with service calls.

### How It Works

```python
# In AgentOrchestrator._initialize_services():
# Resolves services from DI container when USE_SERVICE_LAYER flag is enabled

# In each delegated method:
async def chat(self, user_message: str) -> CompletionResponse:
    if self._use_service_layer and self._chat_service:
        return await self._chat_service.chat(user_message)
    return await self._chat_coordinator.chat(user_message)
```

### Delegated Methods (12 total)

| Domain | Method | Service |
|--------|--------|---------|
| **Chat** | `chat()` | `ChatServiceAdapter` |
| **Chat** | `stream_chat()` | `ChatServiceAdapter` |
| **Chat** | `chat_with_planning()` | `ChatServiceAdapter` |
| **Tool** | `get_available_tools()` | `ToolServiceAdapter` |
| **Tool** | `get_enabled_tools()` | `ToolServiceAdapter` |
| **Tool** | `set_enabled_tools()` | `ToolServiceAdapter` |
| **Tool** | `is_tool_enabled()` | `ToolServiceAdapter` |
| **Tool** | `_execute_tool_with_retry()` | `ToolServiceAdapter` |
| **Session** | `save_checkpoint()` | `SessionServiceAdapter` |
| **Session** | `restore_checkpoint()` | `SessionServiceAdapter` |
| **Session** | `get_recent_sessions()` | `SessionServiceAdapter` |
| **Session** | `get_session_stats()` | `SessionServiceAdapter` |

### Enabling

```bash
export VICTOR_USE_SERVICE_LAYER=true
```

Or in `~/.victor/features.yaml`:

```yaml
feature_flags:
  use_service_layer: true
```

When the flag is off (default), all methods fall back to coordinators with zero behavior change.

## Checklist

Before submitting a new service:

- [ ] Protocol defined in `victor/agent/services/protocols/`
- [ ] Service implements protocol
- [ ] Service has `is_healthy()` method
- [ ] Service uses configuration object
- [ ] Dependencies are injected (optional or required)
- [ ] Feature flag added
- [ ] Factory function in `bootstrap_services.py`
- [ ] Tests written (90%+ coverage)
- [ ] Documentation updated
- [ ] Example usage provided

## References

- Service Protocols: `victor/agent/services/protocols/`
- Service Implementations: `victor/agent/services/`
- Feature Flags: `victor/core/feature_flags.py`
- Bootstrap: `victor/core/bootstrap_services.py`
- Test Examples: `tests/unit/agent/services/`
