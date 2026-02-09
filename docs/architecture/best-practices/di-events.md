# Dependency Injection and Events

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Best practices for DI and event-driven architecture

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 1 min
