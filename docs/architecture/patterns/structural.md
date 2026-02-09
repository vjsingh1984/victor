# Structural Design Patterns

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Structural patterns in Victor AI (Adapter, Facade, Proxy, etc.)

---

# Structural and Behavioral Patterns

**Version**: 0.5.0
**Last Updated**: January 31, 2026
**Audience**: Developers, Contributors
**Purpose**: Adapter, Facade, Strategy, and Observer patterns

---

## Structural Patterns
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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
