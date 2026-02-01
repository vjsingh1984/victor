# Architecture Patterns

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Higher-level architecture patterns and selection guides

---

## Architecture Patterns

### Protocol Pattern

**Intent**: Define contracts using Python's Protocol for structural typing

**Problem**:
- Need clear contracts between components
- Want loose coupling
- Need better type checking
- Want to avoid inheritance constraints

**Solution**: Protocol defines interface, any class matching signature works

**Implementation**: 98 protocols across codebase

**File Location**: `/Users/vijaysingh/code/codingagent/victor/protocols/`

**Structure**:
```python
from typing import Protocol

@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
    """Protocol for tool coordination operations."""

    async def select_tools(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[BaseTool]:
        """Select tools for execution."""
        ...

    async def execute_tool(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> ToolCallResult:
        """Execute a single tool."""
        ...

# Implementation (no inheritance needed!)
class ToolCoordinator:
    """Implements ToolCoordinatorProtocol."""

    async def select_tools(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[BaseTool]:
        # Implementation
        ...

    async def execute_tool(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> ToolCallResult:
        # Implementation
        ...

# Mock for testing (also implements protocol!)
class MockToolCoordinator:
    """Mock implementation for testing."""

    async def select_tools(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[BaseTool]:
        return []

    async def execute_tool(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> ToolCallResult:
        return ToolCallResult(tool=tool, output="mock result")

# Usage (depends on protocol, not implementation)
class MyComponent:
    def __init__(self, tool_coordinator: ToolCoordinatorProtocol):
        self._tool_coordinator = tool_coordinator

# Works with real implementation
real_coordinator = ToolCoordinator()
component = MyComponent(real_coordinator)

# Works with mock implementation
mock_coordinator = MockToolCoordinator()
component = MyComponent(mock_coordinator)
```

**Benefits**:
1. **Structural Typing**: Duck typing with type hints
2. **Loose Coupling**: Depend on protocol, not implementation
3. **Testability**: Easy to create mocks
4. **Flexibility**: Any class matching signature works
5. **IDE Support**: Better autocomplete and checking

**Related Patterns**:
- Dependency Inversion (depend on abstractions)
- Interface Segregation (focused interfaces)

---

### Dependency Injection Pattern

**Intent**: Inject dependencies rather than create them internally

**Problem**:
- Tight coupling to concrete implementations
- Hard to test
- Difficult to swap implementations
- Complex dependency graphs

**Solution**: Container manages dependencies and injects them

**Implementation**: ServiceContainer with 55+ services

**File Location**: `/Users/vijaysingh/code/codingagent/victor/core/container.py`

**Structure**:
```python
class ServiceContainer:
    """Dependency injection container."""

    def __init__(self):
        self._services = {}
        self._singletons = {}

    def register(
        self,
        protocol: Type[T],
        factory: Callable[[ServiceContainer], T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> None:
        """Register service."""
        self._services[protocol] = (factory, lifetime)

    def get(self, protocol: Type[T]) -> T:
        """Resolve service."""
        if protocol not in self._services:
            raise ServiceNotFoundError(f"Service not registered: {protocol}")

        factory, lifetime = self._services[protocol]

        if lifetime == ServiceLifetime.SINGLETON:
            if protocol not in self._singletons:
                self._singletons[protocol] = factory(self)
            return self._singletons[protocol]
        else:
            return factory(self)

# Usage
container = ServiceContainer()

# Register services
container.register(
    ToolRegistryProtocol,
    lambda c: ToolRegistry(),
    ServiceLifetime.SINGLETON,
)

container.register(
    ConversationStateMachineProtocol,
    lambda c: ConversationStateMachine(),
    ServiceLifetime.SCOPED,
)

# Resolve services
tool_registry = container.get(ToolRegistryProtocol)
tools = tool_registry.get_all_tools()

# Component with injected dependencies
class MyComponent:
    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        event_bus: IEventBackend,
    ):
        self._tool_registry = tool_registry
        self._event_bus = event_bus

# Create component with injected dependencies
component = MyComponent(
    tool_registry=container.get(ToolRegistryProtocol),
    event_bus=container.get(IEventBackend),
)
```

**Benefits**:
1. **Loose Coupling**: Depend on protocols, not concrete classes
2. **Testability**: Easy to inject mocks
3. **Flexibility**: Swap implementations
4. **Lifecycle Management**: Singleton, scoped, transient
5. **Centralized**: All registration in one place

**Related Patterns**:
- Factory Pattern (creates objects)
- Service Locator (similar but different)

---

### Event-Driven Architecture Pattern

**Intent**: Communicate through events for loose coupling

**Problem**:
- Tight coupling between components
- Hard to scale
- Synchronous communication blocks
- Difficult to add new listeners

**Solution**: Pub/sub event system for asynchronous communication

**Implementation**: 5 pluggable event backends

**File Location**: `/Users/vijaysingh/code/codingagent/victor/core/events/`

**Structure**:
```python
# Event Bus Interface
@runtime_checkable
class IEventBackend(Protocol):
    """Protocol for event backends."""

    async def publish(self, event: MessagingEvent) -> bool:
        """Publish event."""
        ...

    async def subscribe(self, pattern: str, handler: EventHandler) -> SubscriptionHandle:
        """Subscribe to events."""
        ...

# Event
class MessagingEvent:
    def __init__(
        self,
        topic: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ):
        self.topic = topic
        self.data = data
        self.correlation_id = correlation_id

# Publisher
class ToolExecutor:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus

    async def execute_tool(self, tool: BaseTool, args: dict):
        # Publish start event
        await self._event_bus.publish(
            MessagingEvent(
                topic="tool.start",
                data={"tool": tool.name, "args": args},
            )
        )

        # Execute tool
        result = await tool.execute(**args)

        # Publish complete event
        await self._event_bus.publish(
            MessagingEvent(
                topic="tool.complete",
                data={"tool": tool.name, "result": result},
            )
        )

        return result

# Subscribers
class MetricsCollector:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus
        self._event_bus.subscribe("tool.*", self._on_tool_event)

    async def _on_tool_event(self, event: MessagingEvent):
        # Collect metrics
        ...

class Logger:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus
        self._event_bus.subscribe("*", self._on_any_event)

    async def _on_any_event(self, event: MessagingEvent):
        # Log all events
        ...
```

**Usage Example**:
```python
# Create event bus
event_bus = create_event_backend(BackendConfig.for_observability())

# Create publisher (doesn't know about subscribers)
executor = ToolExecutor(event_bus)

# Create subscribers (independent of publisher)
metrics = MetricsCollector(event_bus)
logger = Logger(event_bus)

# Execute tool - both subscribers notified
await executor.execute_tool(tool, {"path": "test.py"})
```

**Benefits**:
1. **Loose Coupling**: Publisher doesn't know subscribers
2. **Asynchronous**: Non-blocking
3. **Scalable**: Add more subscribers easily
4. **Flexible**: Subscribe/unsubscribe at runtime
5. **Observable**: Easy to monitor and debug

**Related Patterns**:
- Observer Pattern (similar)
- Pub/Sub Pattern (messaging)

---

### Coordinator Pattern

**Intent**: Encapsulate complex operations spanning multiple services

**Problem**:
- Complex operations involve multiple services
- Want to avoid bloated orchestrator
- Need clear separation of concerns
- Want to test operations independently

**Solution**: Coordinators encapsulate specific operations

**Implementation**: 15 specialized coordinators

**Structure**:
```python
class ToolCoordinator:
    """Coordinates tool selection, budgeting, and execution."""

    def __init__(
        self,
        tool_pipeline: Optional[ToolPipelineProtocol],
        tool_selector: Optional[IToolSelector],
        budget_manager: Optional[IBudgetManager],
        tool_cache: Optional[ToolCacheProtocol],
        config: ToolCoordinatorConfig,
    ):
        self._tool_pipeline = tool_pipeline
        self._tool_selector = tool_selector
        self._budget_manager = budget_manager
        self._tool_cache = tool_cache
        self._config = config

    async def select_and_execute(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[ToolCallResult]:
        """Select tools and execute within budget."""
        # 1. Select tools
        tools = await self._tool_selector.select_tools(query, context)

        # 2. Check budget
        if not self._budget_manager.can_execute(len(tools)):
            raise BudgetExceededError()

        # 3. Execute with caching
        results = []
        for tool in tools:
            cached = await self._tool_cache.get(tool.name)
            if cached:
                results.append(cached)
            else:
                result = await self._tool_pipeline.execute_tool(tool)
                await self._tool_cache.set(tool.name, result)
                results.append(result)

        return results
```

**Benefits**:
1. **Single Responsibility**: Each coordinator has one job
2. **Testability**: Test coordinators independently
3. **Reusability**: Use coordinators in different contexts
4. **Clarity**: Clear what each coordinator does
5. **Maintainability**: Easy to modify specific coordination

**Related Patterns**:
- Facade Pattern (provides simplified interface)
- Mediator Pattern (coordinates communication)

---


## Pattern Selection Guide

### Decision Tree

```
Need to create objects?
├─ Complex initialization?
│  └─ Use Factory Pattern
├─ Many optional parameters?
│  └─ Use Builder Pattern
└─ Simple creation?
   └─ Use direct instantiation

Need to simplify interface?
├─ Complex subsystem?
│  └─ Use Facade Pattern
└─ Incompatible interface?
   └─ Use Adapter Pattern

Need to add behavior?
├─ At runtime?
│  └─ Use Decorator Pattern
├─ Across classes?
│  └─ Use Mixin Pattern
└─ Conditionally?
   └─ Use Strategy Pattern

Need to communicate?
├─ One-to-many notification?
│  └─ Use Observer Pattern
├─ Encapsulate request?
│  └─ Use Command Pattern
└─ Chain processing?
   └─ Use Chain of Responsibility

Need to define contracts?
├─ Between components?
│  └─ Use Protocol Pattern
└─ For dependencies?
   └─ Use Dependency Injection

Need to coordinate operations?
├─ Cross-cutting concerns?
│  └─ Use Coordinator Pattern
└─ Async communication?
   └─ Use Event-Driven Architecture
```

### Pattern Combinations

**Common Combinations**:
1. **Factory + DI**: Factory uses DI container to resolve dependencies
2. **Facade + Coordinator**: Facade uses coordinators internally
3. **Strategy + DI**: Strategy implementations registered in DI
4. **Observer + Event-Driven**: Observer pattern implemented with events
5. **Protocol + DI**: Protocols registered in DI container

**Example**:
```python
# Factory uses DI container
class OrchestratorFactory:
    def __init__(self, container: ServiceContainer):
        self._container = container

    def create_orchestrator(self) -> AgentOrchestrator:
        # Resolve from DI container
        tool_registry = self._container.get(ToolRegistryProtocol)
        event_bus = self._container.get(IEventBackend)

        # Create orchestrator (Facade)
        return AgentOrchestrator(
            tool_registry=tool_registry,
            event_bus=event_bus,
        )

# Facade uses Coordinators
class AgentOrchestrator:
    def __init__(self):
        # Coordinators injected
        self._tool_coordinator = tool_coordinator
        self._chat_coordinator = chat_coordinator

    async def chat(self, message: str):
        # Facade delegates to coordinators
        return await self._chat_coordinator.chat(message)

# Strategy implementation in DI
container.register(
    LoadBalancerStrategy,
    lambda c: RoundRobinStrategy(),
    ServiceLifetime.SINGLETON,
)
```

---

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
