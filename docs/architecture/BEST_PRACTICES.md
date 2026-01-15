# Architecture Best Practices

This guide provides best practices for using Victor's new architecture, including protocols, dependency injection, event-driven communication, and coordinators.

## Table of Contents

1. [Using Protocols](#using-protocols)
2. [Using Dependency Injection](#using-dependency-injection)
3. [Using Event-Driven Architecture](#using-event-driven-architecture)
4. [Using Coordinators](#using-coordinators)
5. [Choosing Patterns](#choosing-patterns)
6. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Using Protocols

### Define Protocols First

When creating new components, define the protocol before implementation:

```python
# Good: Protocol first
@runtime_checkable
class MyServiceProtocol(Protocol):
    """Protocol for my service."""

    async def process(self, data: str) -> dict:
        """Process input data."""
        ...

class MyService:
    """Implementation of my service."""

    async def process(self, data: str) -> dict:
        # Implementation
        return {"result": data.upper()}
```

**Why**:
- Clear interface contract
- Easy to create mocks
- Enables multiple implementations

### Use Protocol Composition

Combine multiple protocols for complex interfaces:

```python
@runtime_checkable
class CacheProtocol(Protocol):
    """Caching operations."""

    async def get(self, key: str) -> Optional[Any]:
        ...

    async def set(self, key: str, value: Any) -> None:
        ...

@runtime_checkable
class MetricsProtocol(Protocol):
    """Metrics operations."""

    def track_operation(self, operation: str) -> None:
        ...

@runtime_checkable
class SmartCacheProtocol(CacheProtocol, MetricsProtocol, Protocol):
    """Cache with metrics tracking."""
    # Inherits both CacheProtocol and MetricsProtocol methods
    ...
```

### Prefer Protocols Over Abstract Base Classes

Use protocols for structural subtyping (duck typing):

```python
# Good: Protocol
@runtime_checkable
class WriterProtocol(Protocol):
    def write(self, data: str) -> None:
        ...

# Any class with write() method matches, no inheritance needed
class FileWriter:
    def write(self, data: str) -> None:
        with open("file.txt", "w") as f:
            f.write(data)

class ConsoleWriter:
    def write(self, data: str) -> None:
        print(data)

# Both match WriterProtocol
writer: WriterProtocol = FileWriter()  # OK
writer = ConsoleWriter()  # Also OK
```

### Use Runtime Checkable for Protocols

Make protocols runtime-checkable for isinstance() checks:

```python
# Good: Runtime checkable
@runtime_checkable
class MyProtocol(Protocol):
    def method(self) -> None:
        ...

class Implementation:
    def method(self) -> None:
        pass

impl = Implementation()
assert isinstance(impl, MyProtocol)  # Works
```

### Document Protocol Contracts Clearly

Use detailed docstrings for protocols:

```python
@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """Protocol for tool execution.

    Responsibilities:
    - Execute tools with proper error handling
    - Track tool execution metrics
    - Manage tool timeouts

    Thread Safety:
    - Implementations must be thread-safe

    Error Handling:
    - Must raise ToolExecutionError on tool failure
    - Must validate arguments before execution
    """

    async def execute_tool(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> ToolCallResult:
        """Execute a tool with the given arguments.

        Args:
            tool: Tool instance to execute
            arguments: Tool arguments (must be validated)
            timeout: Optional timeout in seconds

        Returns:
            ToolCallResult with execution result

        Raises:
            ToolExecutionError: If tool execution fails
            TimeoutError: If execution times out
            ValidationError: If arguments are invalid
        """
        ...
```

---

## Using Dependency Injection

### Register All Services in Container

Register services at application startup:

```python
# Good: Register all services early
def bootstrap_application(settings: Settings):
    container = ServiceContainer()

    # Register all singletons
    container.register(
        ToolRegistryProtocol,
        lambda c: ToolRegistry(),
        ServiceLifetime.SINGLETON,
    )
    container.register(
        ObservabilityProtocol,
        lambda c: ObservabilityIntegration(),
        ServiceLifetime.SINGLETON,
    )

    # Register all scoped services
    container.register(
        ConversationStateMachineProtocol,
        lambda c: ConversationStateMachine(),
        ServiceLifetime.SCOPED,
    )

    return container
```

### Use Appropriate Service Lifetimes

Choose the right lifetime for each service:

```python
# Singleton: Stateless or shared state
container.register(
    ToolRegistryProtocol,
    lambda c: ToolRegistry(),
    ServiceLifetime.SINGLETON,  # One instance for app lifetime
)

# Scoped: Per-session state
container.register(
    ConversationStateMachineProtocol,
    lambda c: ConversationStateMachine(),
    ServiceLifetime.SCOPED,  # One instance per scope
)

# Transient: New instance each time (rarely used)
container.register(
    RequestProcessorProtocol,
    lambda c: RequestProcessor(),
    ServiceLifetime.TRANSIENT,  # New instance every get()
)
```

**Guidelines**:
- **Singleton**: Stateless services, expensive to create, shared resources
- **Scoped**: Session-specific state, request-specific data
- **Transient**: Rarely needed, use only if state must not be shared

### Resolve Services at Construction

Inject dependencies in constructors:

```python
# Good: Constructor injection
class MyComponent:
    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        event_bus: IEventBackend,
        config: MyConfig,
    ):
        self._tool_registry = tool_registry
        self._event_bus = event_bus
        self._config = config

# Usage
component = MyComponent(
    tool_registry=container.get(ToolRegistryProtocol),
    event_bus=container.get(IEventBackend),
    config=config,
)
```

### Avoid Service Location

Don't use container inside components:

```python
# Bad: Service locator pattern
class MyComponent:
    def __init__(self, container: ServiceContainer):
        self._container = container

    async def do_work(self):
        # Hidden dependency - not clear from constructor
        tool_registry = self._container.get(ToolRegistryProtocol)
        ...

# Good: Explicit dependencies
class MyComponent:
    def __init__(self, tool_registry: ToolRegistryProtocol):
        self._tool_registry = tool_registry

    async def do_work(self):
        # Clear dependency
        tools = self._tool_registry.get_all_tools()
        ...
```

### Use Factory Methods for Complex Creation

Use factories for complex object construction:

```python
# Good: Factory for complex creation
class OrchestratorFactory:
    def __init__(self, container: ServiceContainer):
        self._container = container

    def create_orchestrator(
        self,
        provider: BaseProvider,
        mode: AgentMode,
    ) -> AgentOrchestrator:
        """Create orchestrator with all dependencies."""
        # Get services from container
        tool_registry = self._container.get(ToolRegistryProtocol)
        observability = self._container.get(ObservabilityProtocol)

        # Create orchestrator-specific components
        tool_pipeline = self._create_tool_pipeline(provider)
        conversation_controller = self._create_conversation_controller(provider)

        # Create orchestrator
        return AgentOrchestrator(
            provider=provider,
            tool_registry=tool_registry,
            observability=observability,
            tool_pipeline=tool_pipeline,
            conversation_controller=conversation_controller,
        )
```

### Test With Mock Services

Use protocol-based mocks in tests:

```python
# Good: Test with mocks
from unittest.mock import Mock

def test_my_component():
    # Create mocks
    mock_tool_registry = Mock(spec=ToolRegistryProtocol)
    mock_tool_registry.get_all_tools.return_value = []

    mock_event_bus = Mock(spec=IEventBackend)

    # Inject mocks
    component = MyComponent(
        tool_registry=mock_tool_registry,
        event_bus=mock_event_bus,
    )

    # Test
    result = component.do_work()

    # Verify
    mock_tool_registry.get_all_tools.assert_called_once()
```

---

## Using Event-Driven Architecture

### Choose Appropriate Backend

Select backend based on use case:

```python
# For observability (high volume, lossy OK)
observability_bus = create_event_backend(
    BackendConfig.for_observability()
)

# For agent messaging (reliable delivery)
agent_bus = create_event_backend(
    BackendConfig.for_agent_messaging()
)

# For distributed deployment
kafka_bus = create_event_backend(
    BackendConfig(
        backend_type=BackendType.KAFKA,
        delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        extra={"bootstrap_servers": "localhost:9092"},
    )
)
```

**Guidelines**:
- **In-Memory**: Single-instance, low latency, no persistence
- **Kafka**: Distributed, high throughput, exactly-once semantics
- **SQS**: Serverless, managed, at-least-once delivery
- **Redis**: Fast, simple, at-least-once delivery

### Use Structured Event Topics

Follow topic naming conventions:

```python
# Good: Hierarchical topics
"tool.start"        # Tool execution started
"tool.complete"     # Tool execution completed
"tool.error"        # Tool execution failed
"agent.message"     # Agent message sent
"workflow.start"    # Workflow started

# Bad: Inconsistent topics
"tool_started"
"tool/complete"
"ToolError"
```

**Pattern**: `{category}.{action}`

### Include Correlation IDs

Correlate related events:

```python
# Good: Include correlation ID
correlation_id = str(uuid.uuid4())

await event_bus.publish(
    MessagingEvent(
        topic="tool.start",
        data={"tool": tool.name},
        correlation_id=correlation_id,  # Links related events
    )
)

await event_bus.publish(
    MessagingEvent(
        topic="tool.complete",
        data={"tool": tool.name, "result": result},
        correlation_id=correlation_id,  # Same ID
    )
)
```

### Handle Errors Gracefully

Implement error handling in subscribers:

```python
# Good: Error handling in subscriber
class ToolMetricsSubscriber:
    async def on_tool_event(self, event: MessagingEvent):
        try:
            if event.topic == "tool.start":
                self._track_start(event)
            elif event.topic == "tool.complete":
                self._track_complete(event)
        except Exception as e:
            # Log error but don't raise
            logger.error(f"Error in metrics subscriber: {e}")
            # Event is still acknowledged
```

### Use Wildcard Subscriptions

Subscribe to event categories:

```python
# Good: Wildcard subscription
await event_bus.subscribe("tool.*", self._on_tool_event)
await event_bus.subscribe("agent.*", self._on_agent_event)
await event_bus.subscribe("*", self._on_all_events)

# Handler receives all matching events
async def _on_tool_event(self, event: MessagingEvent):
    # Handles tool.start, tool.complete, tool.error, etc.
    ...
```

### Respect Delivery Guarantees

Acknowledge events appropriately:

```python
# Good: Proper acknowledgement
async def on_event(self, event: MessagingEvent):
    try:
        # Process event
        result = await process_event(event)

        # Acknowledge successful processing
        await event.ack()
    except TemporaryError as e:
        # Requeue for retry
        await event.nack(requeue=True)
    except PermanentError as e:
        # Don't retry, drop event
        await event.nack(requeue=False)
```

---

## Using Coordinators

### Single Responsibility Per Coordinator

Each coordinator should have one clear purpose:

```python
# Good: Focused coordinator
class ToolCoordinator:
    """Coordinates tool selection and execution."""

    async def select_and_execute(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[ToolCallResult]:
        """Only handles tool coordination."""
        ...

class StateCoordinator:
    """Coordinates conversation state management."""

    async def transition_to(
        self,
        new_stage: ConversationStage,
    ) -> None:
        """Only handles state transitions."""
        ...
```

**Avoid**: God coordinators that do everything

### Delegate to Specialized Services

Coordinators should delegate, not implement:

```python
# Good: Delegation
class ToolCoordinator:
    def __init__(
        self,
        tool_selector: IToolSelector,  # Delegates selection
        budget_manager: IBudgetManager,  # Delegates budgeting
        tool_executor: ToolExecutorProtocol,  # Delegates execution
    ):
        self._tool_selector = tool_selector
        self._budget_manager = budget_manager
        self._tool_executor = tool_executor

    async def select_and_execute(self, query: str, context):
        # Delegate selection
        tools = await self._tool_selector.select_tools(query, context)

        # Delegate budget check
        if not self._budget_manager.can_execute(len(tools)):
            raise BudgetExceededError()

        # Delegate execution
        results = await self._tool_executor.execute_tools(tools)

        return results
```

### Use Protocol Dependencies

Inject dependencies as protocols:

```python
# Good: Protocol dependencies
class ToolCoordinator:
    def __init__(
        self,
        tool_selector: IToolSelector,
        budget_manager: IBudgetManager,
        tool_cache: ToolCacheProtocol,
    ):
        self._tool_selector = tool_selector
        self._budget_manager = budget_manager
        self._tool_cache = tool_cache
```

**Benefits**:
- Easy to test with mocks
- Can swap implementations
- Clear dependencies

### Handle Errors Appropriately

Coordinators should provide error handling:

```python
# Good: Error handling
class ToolCoordinator:
    async def select_and_execute(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[ToolCallResult]:
        try:
            tools = await self.select_tools(query, context)
        except SelectionError as e:
            # Handle selection errors
            logger.error(f"Tool selection failed: {e}")
            return []

        try:
            results = await self.execute_tools(tools)
        except ExecutionError as e:
            # Handle execution errors
            logger.error(f"Tool execution failed: {e}")
            raise
```

### Provide Clear APIs

Coordinators should have simple, clear methods:

```python
# Good: Clear API
class ToolCoordinator:
    async def select_and_execute(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[ToolCallResult]:
        """One method for common case."""
        ...

    async def select_tools(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[BaseTool]:
        """Separate method for selection only."""
        ...

    async def execute_tools(
        self,
        tools: List[BaseTool],
    ) -> List[ToolCallResult]:
        """Separate method for execution only."""
        ...
```

---

## Choosing Patterns

### When to Use Protocols

Use protocols when:
- You need multiple implementations
- You want to enable testing with mocks
- You want loose coupling
- You're defining a public API

**Example**: ToolExecutorProtocol (multiple implementations possible)

### When to Use Dependency Injection

Use DI when:
- You have complex dependency graphs
- You want to test components in isolation
- You need to manage service lifetimes
- You want to centralize configuration

**Example**: ServiceContainer managing 55+ services

### When to Use Event-Driven Architecture

Use events when:
- Components need to communicate loosely
- You have multiple subscribers
- You want asynchronous communication
- You need scalability/distribution

**Example**: Metrics, logging, observability

### When to Use Coordinators

Use coordinators when:
- You have complex operations spanning multiple services
- You want to encapsulate coordination logic
- You need to coordinate multiple protocols
- You want to simplify complex workflows

**Example**: ToolCoordinator (selection + budgeting + execution)

### When to Use Direct Calls

Direct calls are fine when:
- You have simple, one-to-one communication
- Performance is critical
- You don't need multiple subscribers
- The dependency is stable

**Example**: Internal helper methods

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: God Object

**Problem**: One class does everything:

```python
# Bad: God object
class SuperCoordinator:
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.tool_selector = ToolSelector()
        self.budget_manager = BudgetManager()
        self.tool_cache = ToolCache()
        self.tool_executor = ToolExecutor()
        self.state_machine = ConversationStateMachine()
        self.message_history = MessageHistory()
        # ... 20 more dependencies

    async def do_everything(self, ...):
        # Hundreds of lines of coordination logic
        ...
```

**Solution**: Split into focused coordinators:

```python
# Good: Focused coordinators
class ToolCoordinator:
    """Only tool coordination."""
    def __init__(self, selector, budget, executor):
        ...

class StateCoordinator:
    """Only state coordination."""
    def __init__(self, state_machine, history):
        ...
```

### Anti-Pattern 2: Service Locator

**Problem**: Container used inside components:

```python
# Bad: Service locator
class MyComponent:
    def __init__(self, container: ServiceContainer):
        self._container = container

    async def do_work(self):
        # Hidden dependencies
        tool_registry = self._container.get(ToolRegistryProtocol)
        event_bus = self._container.get(IEventBackend)
        ...
```

**Solution**: Constructor injection:

```python
# Good: Explicit dependencies
class MyComponent:
    def __init__(
        self,
        tool_registry: ToolRegistryProtocol,
        event_bus: IEventBackend,
    ):
        self._tool_registry = tool_registry
        self._event_bus = event_bus
```

### Anti-Pattern 3: Tight Coupling Through Events

**Problem**: Events used for direct calls:

```python
# Bad: Events as RPC
class ComponentA:
    async def call_component_b(self):
        # Using events for synchronous request/response
        future = asyncio.Future()

        def on_response(event):
            future.set_result(event.data)

        event_bus.subscribe("component_b.response", on_response)
        event_bus.publish("component_b.request", {})

        result = await future  # Wait for response
        return result
```

**Solution**: Direct calls or proper async patterns:

```python
# Good: Direct call for request/response
class ComponentA:
    def __init__(self, component_b: ComponentBProtocol):
        self._component_b = component_b

    async def call_component_b(self):
        # Direct call for request/response
        result = await self._component_b.process()
        return result
```

### Anti-Pattern 4: Over-Abstracting

**Problem**: Too many layers of abstraction:

```python
# Bad: Over-abstracted
# IToolExecutorFactoryCreatorFactory
class ToolExecutorFactoryCreatorFactory:
    def create_factory_creator(self) -> ToolExecutorFactoryCreator:
        ...

class ToolExecutorFactoryCreator:
    def create_factory(self) -> ToolExecutorFactory:
        ...

class ToolExecutorFactory:
    def create_executor(self) -> IToolExecutor:
        ...

# Just to create an executor!
factory_creator_factory = ToolExecutorFactoryCreatorFactory()
factory_creator = factory_creator_factory.create_factory_creator()
factory = factory_creator.create_factory()
executor = factory.create_executor()
```

**Solution**: Direct factory or DI:

```python
# Good: Simple factory
def create_tool_executor(container: ServiceContainer) -> IToolExecutor:
    tool_registry = container.get(ToolRegistryProtocol)
    return ToolExecutor(tool_registry=tool_registry)

# Or better: Use DI container directly
executor = container.get(IToolExecutor)
```

### Anti-Pattern 5: Ignoring Errors

**Problem**: Errors swallowed in event handlers:

```python
# Bad: Ignoring errors
async def on_event(self, event: MessagingEvent):
    try:
        process_event(event)
    except Exception:
        pass  # Error swallowed!
```

**Solution**: Log and handle errors:

```python
# Good: Handle errors properly
async def on_event(self, event: MessagingEvent):
    try:
        process_event(event)
    except Exception as e:
        logger.error(f"Error processing event: {e}", exc_info=True)
        # Consider publishing to error topic
        await self._event_bus.publish(
            MessagingEvent(
                topic="error.processing",
                data={"event": event.to_dict(), "error": str(e)},
            )
        )
```

### Anti-Pattern 6: Circular Dependencies

**Problem**: Services depend on each other:

```python
# Bad: Circular dependency
class ServiceA:
    def __init__(self, service_b: ServiceBProtocol):
        self._service_b = service_b

class ServiceB:
    def __init__(self, service_a: ServiceAProtocol):
        self._service_a = service_a
```

**Solution**: Break cycle with events or third service:

```python
# Good: Break cycle with events
class ServiceA:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus

    async def do_work(self):
        # Publish event instead of calling ServiceB
        await self._event_bus.publish(
            MessagingEvent(topic="service_a.work_done", data={})
        )

class ServiceB:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus
        # Subscribe to ServiceA events
        event_bus.subscribe("service_a.*", self._on_event)
```

---

## Summary

### Key Principles

1. **Protocol First**: Define protocols before implementations
2. **Explicit Dependencies**: Inject via constructors, don't locate
3. **Loose Coupling**: Use events for cross-component communication
4. **Single Responsibility**: Each coordinator/service does one thing well
5. **Clear Interfaces**: Simple, focused APIs
6. **Error Handling**: Handle errors gracefully, don't swallow
7. **Testability**: Design for testing from the start

### Decision Tree

```
Need to communicate between components?
│
├─ Request/response? → Use direct calls (DI)
│
└─ Loose notification? → Use events

Need multiple implementations?
│
├─ Yes → Use protocols
│
└─ No → Use concrete classes

Need to coordinate multiple services?
│
├─ Yes → Create coordinator
│
└─ No → Use service directly

Need to manage dependencies?
│
├─ Yes → Use DI container
│
└─ No → Use factory or direct instantiation
```

## Additional Resources

- [REFACTORING_OVERVIEW.md](./REFACTORING_OVERVIEW.md) - High-level architecture
- [MIGRATION_GUIDES.md](./MIGRATION_GUIDES.md) - How to migrate code
- [PROTOCOLS_REFERENCE.md](./PROTOCOLS_REFERENCE.md) - Protocol documentation
