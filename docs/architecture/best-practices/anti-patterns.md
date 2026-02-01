# Anti-Patterns to Avoid

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Common anti-patterns and how to avoid them

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

