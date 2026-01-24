# Victor AI Design Patterns Documentation

**Version**: 0.5.0
**Last Updated**: January 18, 2026
**Audience**: Architects, Senior Developers, Contributors
**Purpose**: Comprehensive documentation of design patterns used in Victor AI

---

## Table of Contents

1. [Overview](#overview)
2. [Creational Patterns](#creational-patterns)
3. [Structural Patterns](#structural-patterns)
4. [Behavioral Patterns](#behavioral-patterns)
5. [Architecture Patterns](#architecture-patterns)
6. [Pattern Catalog](#pattern-catalog)
7. [Pattern Selection Guide](#pattern-selection-guide)
8. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Overview

Victor AI employs a comprehensive set of design patterns to achieve maintainability, testability, and extensibility. These patterns are applied consistently across the codebase, following SOLID principles and protocol-based design.

### Pattern Statistics

| Category | Patterns Used | Count |
|----------|---------------|-------|
| **Creational** | Factory, Builder | 2 |
| **Structural** | Facade, Adapter, Mixin, Decorator, Bridge | 5 |
| **Behavioral** | Strategy, Observer, Command, Chain of Responsibility, Template Method | 5 |
| **Architecture** | Protocol, DI Container, Event-Driven, Coordinator | 4 |
| **Total** | | **16** |

### Pattern Benefits

- **Maintainability**: 93% reduction in core complexity
- **Testability**: 81% test coverage achieved
- **Extensibility**: Plugin-based architecture
- **Flexibility**: 21 LLM providers, 55 tools
- **Scalability**: Event-driven architecture

---

## Creational Patterns

### Factory Pattern

**Intent**: Centralize object creation with proper dependency injection

**Problem**:
- Complex object initialization logic
- Need for consistent object creation
- Testability requires dependency injection
- Don't want to expose construction details

**Solution**: Factory classes encapsulate object creation logic

**Implementation**: `OrchestratorFactory`

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/orchestrator_factory.py`

**Structure**:
```python
class OrchestratorFactory:
    """Factory for creating AgentOrchestrator instances."""

    def __init__(self, settings: Settings, provider: BaseProvider, model: str):
        self._settings = settings
        self._provider = provider
        self._model = model

    def create_orchestrator(self) -> AgentOrchestrator:
        """Create orchestrator with all dependencies."""
        # Create provider components
        provider_components = self._create_provider_components()

        # Create core services
        core_services = self._create_core_services()

        # Create conversation components
        conversation_components = self._create_conversation_components()

        # Create and return orchestrator
        return AgentOrchestrator(
            provider_components=provider_components,
            core_services=core_services,
            conversation_components=conversation_components,
        )

    def _create_provider_components(self) -> ProviderComponents:
        """Create provider-related components."""
        # Implementation

    def _create_core_services(self) -> CoreServices:
        """Create core service components."""
        # Implementation

    def _create_conversation_components(self) -> ConversationComponents:
        """Create conversation components."""
        # Implementation
```

**Usage Example**:
```python
# Production use
factory = OrchestratorFactory(settings, provider, model)
orchestrator = factory.create_orchestrator()

# Testing use
factory = OrchestratorFactory(settings, provider, model)
sanitizer = factory.create_sanitizer()
prompt_builder = factory.create_prompt_builder()
# Test individual components
```

**Benefits**:
1. **Centralized Creation**: All creation logic in one place
2. **Dependency Injection**: Proper DI with ServiceContainer
3. **Testability**: Easy to create test-specific components
4. **Consistency**: Ensures objects are created correctly
5. **Encapsulation**: Hides complex initialization

**Related Patterns**:
- Builder Pattern (for complex object construction)
- Dependency Injection (for managing dependencies)

---

### Builder Pattern

**Intent**: Separate construction of complex object from its representation

**Problem**:
- Complex objects with many parameters
- Need for step-by-step construction
- Want immutable objects
- Need fluent interface

**Solution**: Builder classes provide step-by-step construction

**Implementation**: `PromptBuilder`, `AgentBuilder`

**File Location**: `/Users/vijaysingh/code/codingagent/victor/framework/prompt_builder.py`

**Structure**:
```python
class PromptBuilder:
    """Builder for constructing prompts."""

    def __init__(self):
        self._sections = []

    def add_system_message(self, message: str) -> "PromptBuilder":
        """Add system message."""
        self._sections.append(PromptSection(
            type="system",
            content=message
        ))
        return self

    def add_context(self, context: Dict[str, Any]) -> "PromptBuilder":
        """Add context."""
        self._sections.append(PromptSection(
            type="context",
            content=context
        ))
        return self

    def add_tool_hints(self, tools: List[BaseTool]) -> "PromptBuilder":
        """Add tool hints."""
        self._sections.append(PromptSection(
            type="tools",
            content=tools
        ))
        return self

    def build(self) -> str:
        """Build final prompt."""
        return "\n\n".join(
            section.render() for section in self._sections
        )
```

**Usage Example**:
```python
# Build prompt step by step
prompt = (PromptBuilder()
    .add_system_message("You are a helpful assistant.")
    .add_context({"project": "victor", "language": "python"})
    .add_tool_hints([read_tool, write_tool, search_tool])
    .build())

# Result: Complete formatted prompt
print(prompt)
```

**Benefits**:
1. **Step-by-Step**: Clear construction flow
2. **Fluent Interface**: Method chaining
3. **Immutable**: Builder can be reused
4. **Readable**: Self-documenting code
5. **Flexible**: Easy to add new construction steps

**Related Patterns**:
- Factory Pattern (for object creation)
- Immutable Object (for built objects)

---

## Structural Patterns

### Facade Pattern

**Intent**: Provide simplified interface to complex subsystem

**Problem**:
- Complex subsystem with many interacting components
- Clients need simple interface
- Want to reduce coupling between client and subsystem
- Need to hide implementation details

**Solution**: Facade class provides unified, simplified interface

**Implementation**: `AgentOrchestrator`

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/orchestrator.py`

**Structure**:
```python
class AgentOrchestrator:
    """Facade for coordinator layer."""

    def __init__(self, ...):
        # Initialize all coordinators
        self._config_coordinator = ConfigCoordinator(...)
        self._prompt_coordinator = PromptCoordinator(...)
        self._context_coordinator = ContextCoordinator(...)
        self._chat_coordinator = ChatCoordinator(...)
        self._tool_coordinator = ToolCoordinator(...)
        # ... other coordinators

    async def chat(self, message: str) -> str:
        """Simplified chat interface."""
        # Hide complexity of coordinator interactions
        config = self._config_coordinator.get_config()
        prompt = await self._prompt_coordinator.build_prompt(message)
        context = self._context_coordinator.get_context()
        return await self._chat_coordinator.chat(prompt, context)

    async def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolCallResult]:
        """Simplified tool execution interface."""
        # Hide complexity of tool coordination
        return await self._tool_coordinator.execute_tools(tool_calls)

    def switch_provider(self, provider: BaseProvider) -> None:
        """Simplified provider switching."""
        # Hide complexity of provider management
        self._provider_coordinator.switch_provider(provider)
```

**Usage Example**:
```python
# Client doesn't need to know about coordinators
orchestrator = AgentOrchestrator(...)
response = await orchestrator.chat("Hello, Victor!")

# Facade handles all coordinator interactions internally
# Client code is simple and clean
```

**Before Facade** (Complex):
```python
# Client must understand and coordinate multiple components
config = config_coordinator.get_config()
prompt = await prompt_coordinator.build_prompt(message)
context = context_coordinator.get_context()
result = await chat_coordinator.chat(prompt, context)
# ... many more lines
```

**After Facade** (Simple):
```python
# Client just calls simple methods
response = await orchestrator.chat("Hello")
```

**Benefits**:
1. **Simplicity**: Hides complex interactions
2. **Decoupling**: Reduces client-subsystem coupling
3. **Readability**: Client code is clear
4. **Maintainability**: Changes to subsystem don't affect clients
5. **Backward Compatibility**: Can maintain old API while using new subsystem

**Related Patterns**:
- Mediator Pattern (similar but for peer-to-peer)
- Adapter Pattern (adapts interfaces)

---

### Adapter Pattern

**Intent**: Convert interface of a class into another interface clients expect

**Problem**:
- Incompatible interfaces need to work together
- Can't modify existing classes
- Need to normalize different result formats
- Want to maintain backward compatibility

**Solution**: Adapter classes convert between interfaces

**Implementation**: `IntelligentPipelineAdapter`, `CoordinatorAdapter`

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/adapters/intelligent_pipeline_adapter.py`

**Structure**:
```python
class IntelligentPipelineAdapter:
    """Adapt results from tool pipeline to expected format."""

    async def adapt_result(
        self,
        result: Any,
        target_format: str,
    ) -> Any:
        """Adapt result to target format."""
        if target_format == "json":
            return self._to_json(result)
        elif target_format == "string":
            return self._to_string(result)
        elif target_format == "dict":
            return self._to_dict(result)
        # ... other formats

    def _to_json(self, result: Any) -> str:
        """Convert to JSON."""
        return json.dumps(result)

    def _to_string(self, result: Any) -> str:
        """Convert to string."""
        return str(result)

    def _to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert to dict."""
        if isinstance(result, dict):
            return result
        # ... conversion logic
```

**Usage Example**:
```python
# Adapter normalizes different result formats
adapter = IntelligentPipelineAdapter()

# Adapt tool result to JSON
json_result = await adapter.adapt_result(tool_result, "json")

# Adapt to string
string_result = await adapter.adapt_result(tool_result, "string")

# Client code works with consistent interface
```

**Legacy Adaptation Example**:
```python
class CoordinatorAdapter:
    """Adapt legacy orchestrator calls to coordinator architecture."""

    async def adapt_chat_call(
        self,
        orchestrator: AgentOrchestrator,
        message: str,
    ) -> str:
        """Adapt legacy chat call to new architecture."""
        # Old code expected direct orchestrator.chat(message)
        # New architecture uses coordinators
        prompt = await orchestrator._prompt_coordinator.build_prompt(message)
        context = orchestrator._context_coordinator.get_context()
        return await orchestrator._chat_coordinator.chat(prompt, context)

# Old code continues to work
response = await adapter.adapt_chat_call(orchestrator, "Hello")
```

**Benefits**:
1. **Integration**: Enables incompatible interfaces to work together
2. **Normalization**: Consistent interface across implementations
3. **Backward Compatibility**: Old code works with new architecture
4. **Flexibility**: Easy to add new adapters
5. **Testability**: Can test adapters in isolation

**Related Patterns**:
- Facade Pattern (provides simplified interface)
- Bridge Pattern (separates abstraction from implementation)

---

### Mixin Pattern

**Intent**: Compose behavior through multiple inheritance

**Problem**:
- Need to share behavior across unrelated classes
- Want to avoid deep inheritance hierarchies
- Need flexible composition
- Don't want to repeat code

**Solution**: Mixin classes provide reusable behavior

**Implementation**: `ComponentAccessor`, `StateDelegation`, `LegacyAPI`

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/mixins/`

**Structure**:
```python
class ComponentAccessor:
    """Mixin for accessing orchestrator components."""

    def __init__(self, orchestrator: AgentOrchestrator):
        self._orchestrator = orchestrator

    def get_config_coordinator(self) -> ConfigCoordinator:
        """Get config coordinator."""
        return self._orchestrator._config_coordinator

    def get_prompt_coordinator(self) -> PromptCoordinator:
        """Get prompt coordinator."""
        return self._orchestrator._prompt_coordinator

    # ... other getters

class StateDelegation:
    """Mixin for state delegation."""

    async def get_state(self) -> State:
        """Get current state."""
        return await self._orchestrator._state_coordinator.get_state()

    async def update_state(self, updates: Dict[str, Any]) -> None:
        """Update state."""
        await self._orchestrator._state_coordinator.update_state(updates)

# Use mixins in classes
class MyComponent(ComponentAccessor, StateDelegation):
    """Component with both mixins."""

    def __init__(self, orchestrator: AgentOrchestrator):
        ComponentAccessor.__init__(self, orchestrator)
        StateDelegation.__init__(self, orchestrator)

    async def do_work(self):
        """Use mixin methods."""
        config = self.get_config_coordinator().get_config()
        state = await self.get_state()
        # ... use config and state
```

**Usage Example**:
```python
# Component with accessor mixin
class MyComponent(ComponentAccessor):
    """Component that needs access to coordinators."""

    async def process_request(self):
        """Use mixin to access coordinators."""
        config = self.get_config_coordinator().get_config()
        prompt = await self.get_prompt_coordinator().build_prompt("...")
        # ... process

# Component with state mixin
class MyStatefulComponent(StateDelegation):
    """Component that needs state management."""

    async def update(self):
        """Use mixin to manage state."""
        state = await self.get_state()
        # ... process
        await self.update_state({"last_update": datetime.now()})
```

**Benefits**:
1. **Reuse**: Share behavior across unrelated classes
2. **Flexibility**: Compose behavior as needed
3. **No Deep Hierarchies**: Avoid inheritance chains
4. **Testability**: Test mixins independently
5. **Clarity**: Clear what behavior each mixin provides

**Related Patterns**:
- Composition Pattern (similar concept)
- Trait Pattern (similar in other languages)

---

### Decorator Pattern

**Intent**: Add behavior to objects dynamically

**Problem**:
- Need to add behavior without modifying classes
- Want to add/remove behavior at runtime
- Need to compose multiple behaviors
- Can't use inheritance for all combinations

**Solution**: Decorator classes wrap objects and add behavior

**Implementation**: Tool decorators, middleware decorators

**Structure**:
```python
class ToolDecorator(BaseTool):
    """Base decorator for tools."""

    def __init__(self, tool: BaseTool):
        self._tool = tool

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    async def execute(self, **kwargs):
        """Delegate to wrapped tool."""
        return await self._tool.execute(**kwargs)

class CachingToolDecorator(ToolDecorator):
    """Adds caching to tools."""

    def __init__(self, tool: BaseTool, cache: Cache):
        super().__init__(tool)
        self._cache = cache

    async def execute(self, **kwargs):
        """Check cache before executing."""
        cache_key = self._get_cache_key(kwargs)

        # Check cache
        cached = await self._cache.get(cache_key)
        if cached:
            return cached

        # Execute tool
        result = await self._tool.execute(**kwargs)

        # Cache result
        await self._cache.set(cache_key, result)

        return result

class LoggingToolDecorator(ToolDecorator):
    """Adds logging to tools."""

    async def execute(self, **kwargs):
        """Log before and after execution."""
        logger.info(f"Executing tool: {self._tool.name}")

        try:
            result = await self._tool.execute(**kwargs)
            logger.info(f"Tool {self._tool.name} completed")
            return result
        except Exception as e:
            logger.error(f"Tool {self._tool.name} failed: {e}")
            raise
```

**Usage Example**:
```python
# Create base tool
tool = ReadFileTool()

# Add caching
cached_tool = CachingToolDecorator(tool, cache)

# Add logging
logging_cached_tool = LoggingToolDecorator(cached_tool)

# Use decorated tool
result = await logging_cached_tool.execute(path="test.py")
# Both caching and logging applied
```

**Benefits**:
1. **Flexibility**: Add behavior without modifying classes
2. **Composability**: Stack multiple decorators
3. **Single Responsibility**: Each decorator adds one behavior
4. **Runtime**: Add/remove behavior at runtime
5. **Open/Closed**: Open for extension, closed for modification

**Related Patterns**:
- Chain of Responsibility (similar forwarding)
- Proxy Pattern (similar control)

---

### Bridge Pattern

**Intent**: Separate abstraction from implementation

**Problem**:
- Need to vary abstraction and implementation independently
- Want to avoid permanent binding between abstraction and implementation
- Need to share implementation among multiple abstractions
- Want to hide implementation details

**Solution**: Bridge separates abstraction and implementation

**Implementation**: Provider abstraction, multiple provider implementations

**Structure**:
```python
# Abstraction
class BaseProvider(ABC):
    """Abstract provider (abstraction)."""

    @abstractmethod
    async def chat(self, messages: List[Message]) -> str:
        """Chat completion."""
        pass

    @abstractmethod
    async def stream_chat(self, messages: List[Message]) -> AsyncIterator[StreamChunk]:
        """Stream chat completion."""
        pass

# Implementations
class AnthropicProvider(BaseProvider):
    """Anthropic provider (implementation)."""

    async def chat(self, messages: List[Message]) -> str:
        """Anthropic-specific implementation."""
        # Use Anthropic API
        ...

    async def stream_chat(self, messages: List[Message]) -> AsyncIterator[StreamChunk]:
        """Anthropic-specific streaming."""
        # Use Anthropic streaming API
        ...

class OpenAIProvider(BaseProvider):
    """OpenAI provider (implementation)."""

    async def chat(self, messages: List[Message]) -> str:
        """OpenAI-specific implementation."""
        # Use OpenAI API
        ...

    async def stream_chat(self, messages: List[Message]) -> AsyncIterator[StreamChunk]:
        """OpenAI-specific streaming."""
        # Use OpenAI streaming API
        ...

# Client uses abstraction
class ChatService:
    def __init__(self, provider: BaseProvider):  # Depends on abstraction
        self._provider = provider

    async def chat(self, message: str) -> str:
        """Chat using provider."""
        messages = [Message(role="user", content=message)]
        return await self._provider.chat(messages)  # Calls implementation
```

**Usage Example**:
```python
# Use different implementations without changing client code
anthropic_provider = AnthropicProvider(api_key="...")
chat_service = ChatService(anthropic_provider)
response = await chat_service.chat("Hello")

# Switch to OpenAI
openai_provider = OpenAIProvider(api_key="...")
chat_service = ChatService(openai_provider)
response = await chat_service.chat("Hello")  # Same client code

# Add new provider without changing client
new_provider = GoogleProvider(api_key="...")
chat_service = ChatService(new_provider)  # Works!
```

**Benefits**:
1. **Decoupling**: Abstraction and implementation vary independently
2. **Extensibility**: Easy to add new implementations
3. **Runtime**: Switch implementations at runtime
4. **Hiding**: Implementation details hidden from clients
5. **Sharing**: Multiple abstractions can share implementation

**Related Patterns**:
- Adapter Pattern (adapts interfaces)
- Strategy Pattern (pluggable algorithms)

---

## Behavioral Patterns

### Strategy Pattern

**Intent**: Define family of algorithms, encapsulate each, make them interchangeable

**Problem**:
- Need to use different algorithms
- Want to switch algorithms at runtime
- Have multiple ways to do the same thing
- Want to avoid conditional logic

**Solution**: Strategy pattern encapsulates algorithms

**Implementation**: Load balancer strategies, tool selection strategies

**File Location**: `/Users/vijaysingh/code/codingagent/victor/agent/load_balancer.py`

**Structure**:
```python
class LoadBalancerStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY = "priority"
    HEALTH_AWARE = "health_aware"

class ProviderPool:
    """Pool of provider instances."""

    def __init__(self, strategy: LoadBalancerStrategy):
        self._strategy = strategy
        self._providers = []
        self._current_index = 0

    async def route_request(self, request):
        """Route request using strategy."""
        if self._strategy == LoadBalancerStrategy.ROUND_ROBIN:
            return await self._route_round_robin(request)
        elif self._strategy == LoadBalancerStrategy.LEAST_LOADED:
            return await self._route_least_loaded(request)
        elif self._strategy == LoadBalancerStrategy.PRIORITY:
            return await self._route_priority(request)
        elif self._strategy == LoadBalancerStrategy.HEALTH_AWARE:
            return await self._route_health_aware(request)

    async def _route_round_robin(self, request):
        """Round-robin strategy."""
        provider = self._providers[self._current_index]
        self._current_index = (self._current_index + 1) % len(self._providers)
        return await provider.process(request)

    async def _route_least_loaded(self, request):
        """Least-loaded strategy."""
        provider = min(self._providers, key=lambda p: p.load)
        return await provider.process(request)

    async def _route_priority(self, request):
        """Priority-based strategy."""
        provider = max(self._providers, key=lambda p: p.priority)
        return await provider.process(request)

    async def _route_health_aware(self, request):
        """Health-aware strategy."""
        healthy_providers = [p for p in self._providers if p.is_healthy()]
        provider = min(healthy_providers, key=lambda p: p.load)
        return await provider.process(request)
```

**Usage Example**:
```python
# Create pool with strategy
pool = ProviderPool(strategy=LoadBalancerStrategy.ROUND_ROBIN)

# Use pool
response = await pool.route_request(messages)

# Switch strategy at runtime
pool._strategy = LoadBalancerStrategy.LEAST_LOADED
response = await pool.route_request(messages)  # Uses new strategy

# Add new strategy without modifying pool
class CustomStrategy:
    async def route(self, providers, request):
        # Custom routing logic
        ...

# Pool can use custom strategy
```

**Benefits**:
1. **Flexibility**: Easy to add new strategies
2. **Runtime**: Switch strategies at runtime
3. **Testability**: Test each strategy independently
4. **Clean Code**: Avoid conditional logic
5. **Open/Closed**: Open for extension, closed for modification

**Related Patterns**:
- Bridge Pattern (separates abstraction from implementation)
- State Pattern (similar but for states)

---

### Observer Pattern

**Intent**: Define one-to-many dependency so when one object changes state, all dependents are notified

**Problem**:
- Need to notify multiple objects when state changes
- Want loose coupling between publisher and subscribers
- Need to add/remove subscribers dynamically
- Want to broadcast events to multiple listeners

**Solution**: Observer pattern with event bus and subscribers

**Implementation**: EventBus, event backends

**File Location**: `/Users/vijaysingh/code/codingagent/victor/core/events/`

**Structure**:
```python
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
                correlation_id=str(uuid.uuid4()),
            )
        )

        # Execute tool
        result = await tool.execute(**args)

        # Publish complete event
        await self._event_bus.publish(
            MessagingEvent(
                topic="tool.complete",
                data={"tool": tool.name, "result": result},
                correlation_id=correlation_id,
            )
        )

        return result

# Subscriber
class MetricsCollector:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus
        self._metrics = {}

        # Subscribe to events
        self._event_bus.subscribe("tool.*", self._on_tool_event)

    async def _on_tool_event(self, event: MessagingEvent):
        if event.topic == "tool.start":
            self._track_start(event.data)
        elif event.topic == "tool.complete":
            self._track_complete(event.data)

    def _track_start(self, data):
        tool_name = data["tool"]
        self._metrics[f"{tool_name}_start"] = time.time()

    def _track_complete(self, data):
        tool_name = data["tool"]
        start_time = self._metrics.get(f"{tool_name}_start")
        if start_time:
            duration = time.time() - start_time
            self._metrics[f"{tool_name}_duration"] = duration

# Another subscriber
class LoggingObserver:
    def __init__(self, event_bus: IEventBackend):
        self._event_bus = event_bus
        self._event_bus.subscribe("tool.*", self._on_tool_event)

    async def _on_tool_event(self, event: MessagingEvent):
        logger.info(f"Tool event: {event.topic} - {event.data}")
```

**Usage Example**:
```python
# Create event bus
event_bus = create_event_backend(BackendConfig.for_observability())

# Create publisher (doesn't know about subscribers)
executor = ToolExecutor(event_bus)

# Create subscribers (independent of publisher)
metrics = MetricsCollector(event_bus)
logger = LoggingObserver(event_bus)

# When tool is executed, both subscribers are notified
await executor.execute_tool(tool, {"path": "test.py"})
# MetricsCollector tracks metrics
# LoggingObserver logs events
```

**Benefits**:
1. **Loose Coupling**: Publisher doesn't know about subscribers
2. **Dynamic**: Add/remove subscribers at runtime
3. **Broadcast**: One event to many listeners
4. **Asynchronous**: Non-blocking communication
5. **Scalability**: Easy to add new subscribers

**Related Patterns**:
- Pub/Sub Pattern (similar for messaging)
- Mediator Pattern (centralized communication)

---

### Command Pattern

**Intent**: Encapsulate requests as objects

**Problem**:
- Need to parameterize objects with requests
- Want to queue, log, or undo requests
- Need to compose requests into composite commands
- Want to decouple sender from receiver

**Solution**: Command objects encapsulate requests

**Implementation**: Tool calls, workflow commands

**Structure**:
```python
class Command(ABC):
    """Abstract command."""

    @abstractmethod
    async def execute(self) -> Any:
        """Execute command."""
        pass

    @abstractmethod
    async def undo(self) -> None:
        """Undo command."""
        pass

class ToolCallCommand(Command):
    """Command for tool execution."""

    def __init__(self, tool: BaseTool, arguments: Dict[str, Any]):
        self._tool = tool
        self._arguments = arguments
        self._result = None

    async def execute(self) -> Any:
        """Execute tool."""
        self._result = await self._tool.execute(**self._arguments)
        return self._result

    async def undo(self) -> None:
        """Undo tool execution if possible."""
        if hasattr(self._tool, "undo"):
            await self._tool.undo(**self._arguments)

class CompositeCommand(Command):
    """Composite command."""

    def __init__(self):
        self._commands = []

    def add_command(self, command: Command):
        """Add command to composite."""
        self._commands.append(command)

    async def execute(self) -> List[Any]:
        """Execute all commands."""
        results = []
        for command in self._commands:
            result = await command.execute()
            results.append(result)
        return results

    async def undo(self) -> None:
        """Undo all commands in reverse order."""
        for command in reversed(self._commands):
            await command.undo()

class CommandInvoker:
    """Invoker for commands."""

    def __init__(self):
        self._history = []

    async def execute_command(self, command: Command) -> Any:
        """Execute command and store for undo."""
        result = await command.execute()
        self._history.append(command)
        return result

    async def undo_last(self) -> None:
        """Undo last command."""
        if self._history:
            command = self._history.pop()
            await command.undo()
```

**Usage Example**:
```python
# Create commands
read_cmd = ToolCallCommand(read_tool, {"path": "test.py"})
write_cmd = ToolCallCommand(write_tool, {"path": "output.txt", "content": "data"})

# Create composite
composite = CompositeCommand()
composite.add_command(read_cmd)
composite.add_command(write_cmd)

# Execute commands
invoker = CommandInvoker()
result = await invoker.execute_command(composite)

# Undo if needed
await invoker.undo_last()
```

**Benefits**:
1. **Encapsulation**: Requests are first-class objects
2. **Undo/Redo**: Easy to implement
3. **Composition**: Combine commands
4. **Queueing**: Queue commands for execution
5. **Logging**: Log all commands

**Related Patterns**:
- Composite Pattern (compose commands)
- Memento Pattern (for undo)

---

### Chain of Responsibility Pattern

**Intent**: Pass request along chain of handlers

**Problem**:
- Multiple objects can handle request
- Don't want to specify handler explicitly
- Want to add/remove handlers dynamically
- Need flexible request processing

**Solution**: Chain of handlers processes request

**Implementation**: Middleware pipeline, request handlers

**Structure**:
```python
class Handler(ABC):
    """Abstract handler."""

    def __init__(self):
        self._next_handler = None

    def set_next(self, handler: "Handler") -> "Handler":
        """Set next handler in chain."""
        self._next_handler = handler
        return handler

    @abstractmethod
    async def handle(self, request: Any) -> Any:
        """Handle request or pass to next."""
        pass

class ValidationHandler(Handler):
    """Validate request."""

    async def handle(self, request: Any) -> Any:
        """Validate request."""
        if not self._is_valid(request):
            raise ValueError("Invalid request")

        # Pass to next handler
        if self._next_handler:
            return await self._next_handler.handle(request)
        return request

class AuthenticationHandler(Handler):
    """Authenticate request."""

    async def handle(self, request: Any) -> Any:
        """Authenticate request."""
        if not self._is_authenticated(request):
            raise PermissionError("Not authenticated")

        # Pass to next handler
        if self._next_handler:
            return await self._next_handler.handle(request)
        return request

class LoggingHandler(Handler):
    """Log request."""

    async def handle(self, request: Any) -> Any:
        """Log request."""
        logger.info(f"Processing request: {request}")

        # Pass to next handler
        if self._next_handler:
            return await self._next_handler.handle(request)
        return request

class ProcessingHandler(Handler):
    """Process request."""

    async def handle(self, request: Any) -> Any:
        """Process request."""
        # Final processing
        result = await self._process(request)
        return result
```

**Usage Example**:
```python
# Build chain
validation = ValidationHandler()
auth = AuthenticationHandler()
logging = LoggingHandler()
processing = ProcessingHandler()

validation.set_next(auth).set_next(logging).set_next(processing)

# Process request through chain
result = await validation.handle(request)
# Request goes through all handlers
```

**Benefits**:
1. **Flexibility**: Add/remove handlers dynamically
2. **Order**: Control processing order
3. **Decoupling**: Handler doesn't know chain
4. **Extension**: Easy to add new handlers
5. **Single Responsibility**: Each handler does one thing

**Related Patterns**:
- Decorator Pattern (similar wrapping)
- Pipeline Pattern (similar chain)

---

### Template Method Pattern

**Intent**: Define skeleton of algorithm, let subclasses override steps

**Problem**:
- Algorithm has invariant and variant parts
- Want to avoid code duplication
- Need to enforce algorithm structure
- Want to let subclasses customize specific steps

**Solution**: Template method defines skeleton, subclasses override steps

**Implementation**: Base classes with hooks

**Structure**:
```python
class BaseWorkflow(ABC):
    """Base workflow with template method."""

    async def execute(self, context: Dict[str, Any]) -> Any:
        """Template method - defines algorithm skeleton."""
        # Step 1: Initialize (invariant)
        await self._initialize(context)

        # Step 2: Validate (invariant)
        self._validate(context)

        # Step 3: Prepare (can be overridden)
        await self._prepare(context)

        # Step 4: Execute (must be overridden)
        result = await self._execute(context)

        # Step 5: Cleanup (invariant)
        await self._cleanup(context)

        return result

    async def _initialize(self, context):
        """Initialize workflow."""
        logger.info(f"Initializing workflow: {self.__class__.__name__}")

    def _validate(self, context):
        """Validate context."""
        if not context:
            raise ValueError("Context required")

    async def _prepare(self, context):
        """Prepare for execution. Override to customize."""
        pass  # Default: do nothing

    @abstractmethod
    async def _execute(self, context):
        """Execute workflow. Must be overridden."""
        pass

    async def _cleanup(self, context):
        """Cleanup after execution."""
        logger.info(f"Cleaning up workflow: {self.__class__.__name__}")

class RefactoringWorkflow(BaseWorkflow):
    """Refactoring workflow."""

    async def _prepare(self, context):
        """Prepare refactoring."""
        # Load file, analyze code, etc.
        context["file_content"] = await self._load_file(context["file_path"])
        context["ast"] = self._parse_ast(context["file_content"])

    async def _execute(self, context):
        """Execute refactoring."""
        # Apply refactoring
        refactored = await self._apply_refactoring(
            context["ast"],
            context["refactoring_type"]
        )
        return refactored

class TestingWorkflow(BaseWorkflow):
    """Testing workflow."""

    async def _prepare(self, context):
        """Prepare testing."""
        # Discover tests, set up environment, etc.
        context["test_files"] = await self._discover_tests(context["project_path"])

    async def _execute(self, context):
        """Execute testing."""
        # Run tests
        results = await self._run_tests(context["test_files"])
        return results
```

**Usage Example**:
```python
# Use refactoring workflow
refactor_workflow = RefactoringWorkflow()
result = await refactor_workflow.execute({
    "file_path": "test.py",
    "refactoring_type": "extract_method"
})

# Use testing workflow (same structure, different execution)
test_workflow = TestingWorkflow()
result = await test_workflow.execute({
    "project_path": "/path/to/project"
})
```

**Benefits**:
1. **Structure**: Enforces algorithm structure
2. **Reuse**: Invariant steps defined once
3. **Customization**: Subclasses customize variant steps
4. **Consistency**: All workflows follow same pattern
5. **Maintenance**: Easy to maintain common code

**Related Patterns**:
- Strategy Pattern (entire algorithm varies)
- Factory Pattern (creates objects)

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

## Pattern Catalog

### Quick Reference

| Pattern | Intent | Implementation | Use Case |
|---------|--------|----------------|----------|
| **Factory** | Centralize creation | `OrchestratorFactory` | Complex object creation |
| **Builder** | Step-by-step construction | `PromptBuilder` | Complex objects with many parts |
| **Facade** | Simplified interface | `AgentOrchestrator` | Complex subsystems |
| **Adapter** | Convert interfaces | `IntelligentPipelineAdapter` | Incompatible interfaces |
| **Mixin** | Compose behavior | `ComponentAccessor` | Shared behavior |
| **Decorator** | Add behavior dynamically | Tool decorators | Extensible behavior |
| **Bridge** | Separate abstraction from implementation | `BaseProvider` | Multiple implementations |
| **Strategy** | Encapsulate algorithms | Load balancer strategies | Pluggable algorithms |
| **Observer** | Notify subscribers | `EventBus` | Event notification |
| **Command** | Encapsulate requests | `ToolCall` | Undo/redo, queueing |
| **Chain of Responsibility** | Pass along chain | Middleware pipeline | Request processing |
| **Template Method** | Define skeleton | Workflows | Algorithm structure |
| **Protocol** | Define contracts | 98 protocols | Loose coupling |
| **DI** | Inject dependencies | `ServiceContainer` | Dependency management |
| **Event-Driven** | Async communication | Event backends | Scalable communication |
| **Coordinator** | Encapsulate operations | 15 coordinators | Complex operations |

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

## Anti-Patterns to Avoid

### God Object

**Anti-Pattern**: Single class does everything

**Problem**: Unmaintainable, untestable

**Solution**: Use Facade + Coordinators

### Tight Coupling

**Anti-Pattern**: Direct dependencies on concrete classes

**Problem**: Hard to test, inflexible

**Solution**: Use Protocol Pattern + DI

### Magic Numbers

**Anti-Pattern**: Hardcoded values

**Problem**: Inflexible, error-prone

**Solution**: Use constants or configuration

### Long Methods

**Anti-Pattern**: Methods with hundreds of lines

**Problem**: Hard to understand, test

**Solution**: Extract methods, use Template Method

### Conditional Chaos

**Anti-Pattern**: Deeply nested conditionals

**Problem**: Hard to follow, error-prone

**Solution**: Use Strategy Pattern, polymorphism

---

## Conclusion

Victor AI employs a comprehensive set of design patterns to achieve maintainability, testability, and extensibility. These patterns are applied consistently across the codebase, following SOLID principles and protocol-based design.

### Key Takeaways

1. **Patterns Enable Flexibility**: Easy to extend and modify
2. **Patterns Improve Testability**: Mock implementations, isolated testing
3. **Patterns Reduce Complexity**: Clear separation of concerns
4. **Patterns Enable Scaling**: Event-driven, async architecture
5. **Patterns Follow SOLID**: Principles guide pattern selection

### Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Architecture overview
- [COMPONENT_REFERENCE.md](./COMPONENT_REFERENCE.md) - Component documentation
- [MIGRATION_GUIDES.md](./MIGRATION_GUIDES.md) - Migration guidance

---

**Document Version**: 1.0
**Last Updated**: January 18, 2026
**Maintainers**: Victor AI Architecture Team
