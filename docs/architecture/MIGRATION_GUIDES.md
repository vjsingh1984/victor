# Architecture Migration Guides

This guide provides practical examples for migrating from old patterns to new architectural patterns introduced in the refactoring.

## Table of Contents

1. [Migrating from Direct Dependencies to Protocol-Based Dependencies](#1-protocol-based-dependencies)
2. [Migrating from Service Locators to Dependency Injection](#2-di-container)
3. [Migrating from Direct Calls to Event-Driven Communication](#3-event-driven)
4. [Migrating from Monolithic Handlers to Coordinators](#4-coordinators)
5. [Migrating from Concrete Classes to Protocols](#5-concrete-to-protocols)

---

## 1. Migrating from Direct Dependencies to Protocol-Based Dependencies

### Problem

Code directly depends on concrete classes, making it hard to test and swap implementations.

### Before: Direct Dependency

```python
# Old approach: Direct dependency on concrete class
from victor.agent.tool_executor import ToolExecutor

class MyComponent:
    def __init__(self):
        # Hard dependency on concrete ToolExecutor
        self._tool_executor = ToolExecutor(tool_registry)

    async def do_work(self):
        result = await self._tool_executor.execute_tool(
            tool_name="read_file",
            arguments={"path": "/path/to/file"}
        )
        return result
```

**Problems**:
- Cannot swap ToolExecutor implementation
- Hard to test (requires real ToolExecutor)
- Tight coupling

### After: Protocol-Based Dependency

```python
# New approach: Depend on protocol
from victor.agent.protocols import ToolExecutorProtocol
from victor.core.container import ServiceContainer

class MyComponent:
    def __init__(self, tool_executor: ToolExecutorProtocol):
        # Dependency on protocol (interface)
        self._tool_executor = tool_executor

    async def do_work(self):
        result = await self._tool_executor.execute_tool(
            tool_name="read_file",
            arguments={"path": "/path/to/file"}
        )
        return result

# Usage with DI container
container = ServiceContainer()
container.register(
    ToolExecutorProtocol,
    lambda c: ToolExecutor(tool_registry=c.get(ToolRegistryProtocol)),
    ServiceLifetime.SINGLETON,
)

my_component = MyComponent(
    tool_executor=container.get(ToolExecutorProtocol)
)
```

**Benefits**:
- Easy to test with mock protocols
- Can swap implementations
- Loose coupling

### Testing Migration

```python
# Before: Hard to test
def test_my_component_old():
    # Need to create real ToolExecutor with all dependencies
    tool_registry = ToolRegistry()
    tool_executor = ToolExecutor(tool_registry=tool_registry)
    component = MyComponent()
    # ... complex setup

# After: Easy to test
from unittest.mock import Mock

def test_my_component_new():
    # Create mock protocol
    mock_executor = Mock(spec=ToolExecutorProtocol)
    mock_executor.execute_tool.return_value = {"success": True}

    component = MyComponent(tool_executor=mock_executor)
    result = await component.do_work()

    # Assertions
    mock_executor.execute_tool.assert_called_once()
    assert result == {"success": True}
```

---

## 2. Migrating from Service Locators to Dependency Injection

### Problem

Code uses service locator pattern (global lookups), making dependencies implicit.

### Before: Service Locator

```python
# Old approach: Global service locator
from victor.core.bootstrap import get_tool_registry

class MyAgent:
    async def execute_task(self, task: str):
        # Implicit dependency - not clear from constructor
        tool_registry = get_tool_registry()
        tools = tool_registry.get_all_tools()

        # Another implicit dependency
        from victor.core.bootstrap import get_event_bus
        event_bus = get_event_bus()
        await event_bus.publish("task.start", {"task": task})
```

**Problems**:
- Dependencies hidden in methods
- Hard to test (need to configure globals)
- Cannot have multiple instances with different dependencies

### After: Dependency Injection

```python
# New approach: Explicit dependencies via DI
from victor.agent.protocols import ToolRegistryProtocol, ObservabilityProtocol

class MyAgent:
    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        event_bus: ObservabilityProtocol,
    ):
        # Explicit dependencies
        self._tool_registry = tool_registry
        self._event_bus = event_bus

    async def execute_task(self, task: str):
        tools = self._tool_registry.get_all_tools()
        await self._event_bus.publish(
            MessagingEvent(topic="task.start", data={"task": task})
        )

# Usage with factory
from victor.core.container import ServiceContainer

container = ServiceContainer()
# ... register services

agent = MyAgent(
    tool_registry=container.get(ToolRegistryProtocol),
    event_bus=container.get(ObservabilityProtocol),
)
```

**Benefits**:
- Dependencies explicit in constructor
- Easy to test with injected mocks
- Can create multiple instances with different dependencies

---

## 3. Migrating from Direct Calls to Event-Driven Communication

### Problem

Components directly call each other, creating tight coupling and chains of dependencies.

### Before: Direct Calls

```python
# Old approach: Direct method calls
class ToolExecutor:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.logger = DebugLogger()

    async def execute_tool(self, tool: BaseTool, args: dict):
        # Direct call to metrics
        self.metrics_collector.track_tool_start(tool.name)

        try:
            result = await tool.execute(**args)
            # Direct call to logger
            self.logger.log_tool_success(tool.name)
            # Direct call to metrics
            self.metrics_collector.track_tool_complete(tool.name, result)
            return result
        except Exception as e:
            self.metrics_collector.track_tool_error(tool.name, e)
            raise
```

**Problems**:
- Tight coupling to MetricsCollector and DebugLogger
- Hard to add new subscribers
- MetricsCollector and DebugLogger become dependencies

### After: Event-Driven Communication

```python
# New approach: Publish events, decoupled subscribers
from victor.core.events.protocols import MessagingEvent, IEventBackend

class ToolExecutor:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus

    async def execute_tool(self, tool: BaseTool, args: dict):
        # Publish event - anyone can subscribe
        await self._event_bus.publish(
            MessagingEvent(
                topic="tool.start",
                data={"tool": tool.name, "args": args},
            )
        )

        try:
            result = await tool.execute(**args)

            # Publish completion event
            await self._event_bus.publish(
                MessagingEvent(
                    topic="tool.complete",
                    data={"tool": tool.name, "result": result},
                )
            )
            return result
        except Exception as e:
            # Publish error event
            await self._event_bus.publish(
                MessagingEvent(
                    topic="tool.error",
                    data={"tool": tool.name, "error": str(e)},
                )
            )
            raise

# Subscribers are independent
class MetricsCollector:
    def __init__(self, event_bus: IEventBackend):
        event_bus.subscribe("tool.*", self._on_tool_event)

    async def _on_tool_event(self, event: MessagingEvent):
        if event.topic == "tool.start":
            self.track_tool_start(event.data["tool"])
        elif event.topic == "tool.complete":
            self.track_tool_complete(event.data["tool"], event.data["result"])
        elif event.topic == "tool.error":
            self.track_tool_error(event.data["tool"], event.data["error"])

class DebugLogger:
    def __init__(self, event_bus: IEventBackend):
        event_bus.subscribe("tool.*", self._on_tool_event)

    async def _on_tool_event(self, event: MessagingEvent):
        if event.topic == "tool.complete":
            self.log_tool_success(event.data["tool"])
```

**Benefits**:
- ToolExecutor doesn't know about subscribers
- Easy to add new subscribers without changing ToolExecutor
- Subscribers can be added/removed at runtime
- Enables distributed deployment

---

## 4. Migrating from Monolithic Handlers to Coordinators

### Problem

Monolithic handlers handle too many responsibilities, violating SRP.

### Before: Monolithic Handler

```python
# Old approach: One class does everything
class ToolHandler:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.tool_selector = ToolSelector()
        self.budget_manager = BudgetManager()
        self.tool_cache = ToolCache()
        self.tool_executor = ToolExecutor()

    async def handle_tool_call(self, tool_name: str, args: dict):
        # 1. Get tool
        tool = self.tool_registry.get_tool(tool_name)

        # 2. Check budget
        if not self.budget_manager.can_execute():
            raise BudgetExceededError()

        # 3. Check cache
        cached = await self.tool_cache.get(tool_name)
        if cached:
            return cached

        # 4. Execute
        result = await self.tool_executor.execute(tool, args)

        # 5. Cache result
        await self.tool_cache.set(tool_name, result)

        return result
```

**Problems**:
- Violates Single Responsibility Principle
- Hard to test (too many dependencies)
- Hard to reuse individual operations
- Changes to one operation affect others

### After: Coordinator Pattern

```python
# New approach: Coordinator delegates to specialized services
from victor.agent.protocols import (
    ToolCoordinatorProtocol,
    ToolRegistryProtocol,
    IToolSelector,
    IBudgetManager,
    ToolCacheProtocol,
)

class ToolCoordinator:
    """Coordinates tool selection, budgeting, and execution."""

    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        tool_selector: IToolSelector,
        budget_manager: IBudgetManager,
        tool_cache: ToolCacheProtocol,
        tool_executor: ToolExecutorProtocol,
    ):
        self._tool_registry = tool_registry
        self._tool_selector = tool_selector
        self._budget_manager = budget_manager
        self._tool_cache = tool_cache
        self._tool_executor = tool_executor

    async def select_and_execute(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[ToolCallResult]:
        """Select tools and execute within budget."""
        # 1. Select tools via coordinator method
        tools = await self.select_tools(query, context)

        # 2. Check budget
        if not self._budget_manager.can_execute(len(tools)):
            raise BudgetExceededError()

        # 3. Execute with caching via coordinator method
        results = await self.execute_tools(tools)

        return results

    async def select_tools(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[BaseTool]:
        """Select tools for execution."""
        return await self._tool_selector.select_tools(
            query=query,
            available_tools=self._tool_registry.get_all_tools(),
            max_tools=context.max_tools,
        )

    async def execute_tools(
        self,
        tools: List[BaseTool],
    ) -> List[ToolCallResult]:
        """Execute tools with caching."""
        results = []
        for tool in tools:
            # Check cache
            cached = await self._tool_cache.get(tool.name)
            if cached:
                results.append(cached)
                continue

            # Execute
            result = await self._tool_executor.execute_tool(
                tool=tool,
                arguments={},  # Arguments provided separately
            )

            # Cache result
            await self._tool_cache.set(tool.name, result)
            results.append(result)

        return results
```

**Benefits**:
- Single responsibility (coordination only)
- Each operation is separate and testable
- Easy to reuse individual operations
- Changes isolated to specific methods

---

## 5. Migrating from Concrete Classes to Protocols

### Problem

Code depends on concrete classes, making it hard to extend or modify.

### Before: Concrete Class Dependencies

```python
# Old approach: Depend on concrete classes
from victor.agent.tool_selector import ToolSelector
from victor.agent.budget_manager import BudgetManager

class MyComponent:
    def __init__(self):
        self._tool_selector = ToolSelector()
        self._budget_manager = BudgetManager()

    async def process(self, query: str):
        tools = await self._tool_selector.select_tools(query)
        if self._budget_manager.can_execute(len(tools)):
            # ... process tools
            pass
```

**Problems**:
- Cannot use different implementations
- Hard to test in isolation
- Tight coupling to specific implementations

### After: Protocol-Based Dependencies

```python
# New approach: Depend on protocols
from victor.agent.protocols import IToolSelector, IBudgetManager

class MyComponent:
    def __init__(
        self,
        tool_selector: IToolSelector,
        budget_manager: IBudgetManager,
    ):
        self._tool_selector = tool_selector
        self._budget_manager = budget_manager

    async def process(self, query: str):
        tools = await self._tool_selector.select_tools(query)
        if self._budget_manager.can_execute(len(tools)):
            # ... process tools
            pass

# Can create different implementations
class SemanticToolSelector:
    """ML-based semantic tool selection."""
    async def select_tools(self, query: str) -> List[BaseTool]:
        # Semantic matching logic
        ...

class KeywordToolSelector:
    """Keyword-based tool selection."""
    async def select_tools(self, query: str) -> List[BaseTool]:
        # Keyword matching logic
        ...

# Use different implementations based on context
semantic_component = MyComponent(
    tool_selector=SemanticToolSelector(),
    budget_manager=BudgetManager(),
)

keyword_component = MyComponent(
    tool_selector=KeywordToolSelector(),
    budget_manager=BudgetManager(),
)
```

**Benefits**:
- Can swap implementations
- Easy to extend with new implementations
- Testable with mocks
- Flexible configuration

---

## Common Migration Scenarios

### Scenario 1: Adding New Functionality to Existing Component

**Before**:
```python
class ToolExecutor:
    def __init__(self):
        self.metrics = MetricsCollector()

    async def execute_tool(self, tool, args):
        # Execute tool
        result = await tool.execute(**args)
        # Track metrics
        self.metrics.track_tool(tool.name, result)
        return result
```

**After** (with events):
```python
class ToolExecutor:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus

    async def execute_tool(self, tool, args):
        # Execute tool
        result = await tool.execute(**args)
        # Publish event (no direct dependency on metrics)
        await self._event_bus.publish(
            MessagingEvent(
                topic="tool.complete",
                data={"tool": tool.name, "result": result},
            )
        )
        return result
```

### Scenario 2: Making Component Testable

**Before**:
```python
class Agent:
    def __init__(self):
        self.tool_executor = ToolExecutor()
        self.state_tracker = StateTracker()

    async def run(self, task: str):
        # Hard to test - dependencies are concrete
        result = await self.tool_executor.execute(...)
        self.state_tracker.update_state(result)
```

**After** (with protocols):
```python
class Agent:
    def __init__(
        self,
        tool_executor: ToolExecutorProtocol,
        state_tracker: StateTrackerProtocol,
    ):
        self._tool_executor = tool_executor
        self._state_tracker = state_tracker

    async def run(self, task: str):
        # Easy to test - can inject mocks
        result = await self._tool_executor.execute(...)
        await self._state_tracker.update_state(result)

# Test with mocks
def test_agent():
    mock_executor = Mock(spec=ToolExecutorProtocol)
    mock_tracker = Mock(spec=StateTrackerProtocol)
    agent = Agent(mock_executor, mock_tracker)
    # ... test logic
```

### Scenario 3: Supporting Multiple Implementations

**Before**:
```python
class CacheService:
    def __init__(self):
        self._cache = {}  # Only supports in-memory

    async def get(self, key: str):
        return self._cache.get(key)

    async def set(self, key: str, value: Any):
        self._cache[key] = value
```

**After** (with protocol and multiple implementations):
```python
@runtime_checkable
class CacheServiceProtocol(Protocol):
    async def get(self, key: str) -> Optional[Any]:
        ...

    async def set(self, key: str, value: Any) -> None:
        ...

class InMemoryCache:
    """In-memory cache implementation."""
    def __init__(self):
        self._cache = {}

    async def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    async def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

class RedisCache:
    """Redis cache implementation."""
    def __init__(self, redis_url: str):
        self._client = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[Any]:
        return await self._client.get(key)

    async def set(self, key: str, value: Any) -> None:
        await self._client.set(key, value)

# Use different implementations based on configuration
def create_cache(config: Config) -> CacheServiceProtocol:
    if config.use_redis:
        return RedisCache(config.redis_url)
    else:
        return InMemoryCache()
```

---

## Troubleshooting

### Issue: "Service not registered" Error

**Problem**:
```
ServiceNotFoundError: Service not registered: ToolExecutorProtocol
```

**Solution**:
Register the service in the DI container before use:

```python
from victor.core.container import ServiceContainer, ServiceLifetime
from victor.agent.protocols import ToolExecutorProtocol

container = ServiceContainer()
container.register(
    ToolExecutorProtocol,
    lambda c: ToolExecutor(tool_registry=c.get(ToolRegistryProtocol)),
    ServiceLifetime.SINGLETON,
)
```

### Issue: "Scope disposed" Error

**Problem**:
```
ScopeDisposedError: Scope has been disposed, cannot resolve ConversationStateMachineProtocol
```

**Solution**:
Don't use scoped services after the scope is disposed:

```python
# Bad
scope = container.create_scope()
state_machine = scope.get(ConversationStateMachineProtocol)
scope.dispose()  # Scope disposed
result = state_machine.get_state()  # Error!

# Good
scope = container.create_scope()
state_machine = scope.get(ConversationStateMachineProtocol)
result = state_machine.get_state()  # OK
scope.dispose()
```

### Issue: Event Not Received

**Problem**:
Published events are not received by subscribers.

**Solution**:
Ensure event backend is connected:

```python
# Connect the backend before using it
backend = create_event_backend(BackendConfig.for_observability())
await backend.connect()

# Now subscribe and publish
await backend.subscribe("tool.*", handler)
await backend.publish(MessagingEvent(topic="tool.start", data={}))
```

---

## Best Practices for Migration

1. **Incremental Migration**: Migrate one component at a time
2. **Maintain Tests**: Keep tests passing throughout migration
3. **Use Adapters**: Create adapter layers for gradual migration
4. **Document Changes**: Update documentation as you migrate
5. **Refactor Before**: Clean up code before adding new features
6. **Protocol First**: Define protocols before implementing
7. **Test With Mocks**: Use protocol-based mocks for testing

## Additional Resources

- [REFACTORING_OVERVIEW.md](./REFACTORING_OVERVIEW.md) - High-level overview
- [BEST_PRACTICES.md](./BEST_PRACTICES.md) - Usage patterns and guidelines
- [PROTOCOLS_REFERENCE.md](./PROTOCOLS_REFERENCE.md) - Protocol documentation
