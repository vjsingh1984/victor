# Victor AI 0.5.0 API Reference - Part 2

**Part 2 of 2:** Provider Management, Tool System, Conversation Management, Streaming, Vertical Base

---

## Navigation

- [Part 1: Orchestrator, Coordinators, Pipeline](part-1-orchestrator-coordinators-pipeline.md)
- **[Part 2: Provider, Tools, Conversation, Streaming](#)** (Current)
- [**Complete Reference](../API_REFERENCE.md)**

---

## Provider Management

### ProviderCoordinator

Manages provider lifecycle and switching.

```python
class ProviderCoordinator:
    """Manage LLM providers."""

    def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None
    ) -> None:
        """Switch to a different provider."""

    def get_provider(self) -> BaseProvider:
        """Get current provider instance."""

    def list_providers(self) -> List[str]:
        """List available providers."""
```

### Provider Registry

Register and discover providers:

```python
from victor.providers.registry import ProviderRegistry

registry = ProviderRegistry.default()

# Register provider
registry.register(MyCustomProvider())

# Get provider
provider = registry.get("my_provider")

# List providers
providers = registry.list_all()
```

---

## Tool System

### ToolExecutionCoordinator

Manages tool execution and budgeting.

```python
class ToolExecutionCoordinator:
    """Execute tools with budgeting."""

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute a single tool."""

    async def execute_batch(
        self,
        tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        """Execute multiple tools in batch."""
```

### Tool Registry

Register and discover tools:

```python
from victor.tools.registry import ToolRegistry

registry = ToolRegistry.default()

# Register tool
registry.register(MyCustomTool())

# Execute tool
result = await registry.execute("my_tool", param="value")
```

---

## Conversation Management

### ConversationCoordinator

Manages conversation state and history.

```python
class ConversationCoordinator:
    """Manage conversation state."""

    def add_message(
        self,
        role: str,
        content: str
    ) -> None:
        """Add message to history."""

    def get_history(self) -> List[Message]:
        """Get conversation history."""

    def clear_history(self) -> None:
        """Clear conversation history."""
```

---

## Streaming

### StreamingCoordinator

Handles streaming responses.

```python
class StreamingCoordinator:
    """Process streaming responses."""

    async def stream_chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat response."""

    async def process_chunk(
        self,
        chunk: StreamChunk
    ) -> Optional[str]:
        """Process individual chunk."""
```

---

## Vertical Base

### VerticalBase

Base class for all verticals.

```python
class VerticalBase:
    """Base class for verticals."""

    @classmethod
    def name(cls) -> str:
        """Vertical name."""

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools for this vertical."""

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get system prompt."""
```

---

## Related Documentation

- [Architecture Overview](../../../architecture/README.md)
- [Provider Reference](../PROVIDER_REFERENCE.md)
- [Tools API](../internals/tools-api.md)

---

**Last Updated:** February 01, 2026
