# Provider API Reference

LLM provider abstraction layer with unified interface for 22+ LLM backends.

## Overview

The Provider API provides:
- **Unified interface** for all LLM providers
- **Protocol-based capabilities** (streaming, tool calling)
- **Circuit breaker** for resilience
- **Automatic retries** with exponential backoff
- **Provider registry** for dynamic provider resolution

## Quick Example

```python
from victor.providers import ProviderRegistry, Message

# Get provider
registry = ProviderRegistry.get_instance()
provider = registry.get_provider("anthropic")

# Use provider
messages = [
    Message(role="user", content="Hello, Victor!")
]

response = await provider.chat(
    messages,
    model="claude-sonnet-4-5-20250514",
    temperature=0.7,
    max_tokens=4096
)

print(response.content)
```

## Available Providers

| Provider | Name | Streaming | Tools | Notes |
|----------|------|-----------|-------|-------|
| **Anthropic** | `anthropic` | ✅ | ✅ | Claude models |
| **OpenAI** | `openai` | ✅ | ✅ | GPT models |
| **Google** | `google` | ✅ | ✅ | Gemini models |
| **Ollama** | `ollama` | ✅ | ✅ | Local models |
| **LM Studio** | `lmstudio` | ✅ | ✅ | Local GUI server |
| **vLLM** | `vllm` | ✅ | ✅ | Fast local inference |
| **Azure OpenAI** | `azure_openai` | ✅ | ✅ | Azure-hosted OpenAI |
| **Bedrock** | `bedrock` | ✅ | ✅ | AWS Bedrock |
| **Vertex AI** | `vertex` | ✅ | ✅ | Google Vertex AI |
| **DeepSeek** | `deepseek` | ✅ | ✅ | DeepSeek models |
| **Groq** | `groq` | ✅ | ✅ | Fast inference |
| **Together AI** | `together` | ✅ | ✅ | Open source models |
| **Fireworks AI** | `fireworks` | ✅ | ✅ | Fast inference |
| **Replicate** | `replicate` | ✅ | ✅ | Model hosting |
| **OpenRouter** | `openrouter` | ✅ | ✅ | Model router |
| **Hugging Face** | `huggingface` | ✅ | ✅ | HF Inference API |
| **Mistral AI** | `mistral` | ✅ | ✅ | Mistral models |
| **Moonshot AI** | `moonshot` | ✅ | ✅ | Kimi models |
| **xAI** | `xai` | ✅ | ✅ | Grok models |
| **Cerebras** | `cerebras` | ✅ | ✅ | Fast inference |
| **MLX** | `mlx` | ✅ | ❌ | Apple Silicon |
| **LLama.cpp** | `llamacpp` | ✅ | ❌ | Local inference |
| **Zai** | `zai` | ✅ | ✅ | Zai models |

## BaseProvider Class

Abstract base class for all LLM providers.

```python
class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""
```

### Constructor

```python
def __init__(
    self,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 60,
    max_retries: int = 3,
    use_circuit_breaker: bool = True,
    circuit_breaker_failure_threshold: int = 5,
    circuit_breaker_recovery_timeout: float = 30.0,
    **kwargs: Any,
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | API key for authentication |
| `base_url` | `str \| None` | `None` | Base URL for API endpoints |
| `timeout` | `int` | `60` | Request timeout (seconds) |
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `use_circuit_breaker` | `bool` | `True` | Enable circuit breaker |
| `circuit_breaker_failure_threshold` | `int` | `5` | Failures before opening circuit |
| `circuit_breaker_recovery_timeout` | `float` | `30.0` | Seconds before testing recovery |
| `**kwargs` | `Any` | `{}` | Provider-specific options |

### Properties

```python
@property
def name(self) -> str:
    """Provider name (e.g., 'anthropic', 'ollama')."""
    pass

@property
def circuit_breaker(self) -> CircuitBreaker | None:
    """Get the circuit breaker for this provider."""
    pass
```

### Methods

#### supports_tools()

```python
def supports_tools(self) -> bool:
    """Check if provider supports tool/function calling.

    Returns:
        True if provider supports tools, False otherwise
    """
```

**Examples**:

```python
provider = registry.get_provider("anthropic")
if provider.supports_tools():
    print("Tool calling available")
```

#### supports_streaming()

```python
def supports_streaming(self) -> bool:
    """Check if provider supports streaming responses.

    Returns:
        True if provider supports streaming, False otherwise
    """
```

**Examples**:

```python
provider = registry.get_provider("ollama")
if provider.supports_streaming():
    async for chunk in provider.stream(messages, model="llama3"):
        print(chunk.content, end="")
```

#### chat()

```python
async def chat(
    self,
    messages: list[Message],
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tools: list[ToolDefinition] | None = None,
    **kwargs: Any,
) -> CompletionResponse:
    """Send a chat completion request.

    Args:
        messages: List of conversation messages
        model: Model identifier
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        tools: Available tools for the model to use
        **kwargs: Additional provider-specific parameters

    Returns:
        CompletionResponse with generated content

    Raises:
        ProviderError: If the request fails
    """
```

**Examples**:

```python
messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="What is Victor?")
]

response = await provider.chat(
    messages,
    model="claude-sonnet-4-5-20250514",
    temperature=0.7,
    max_tokens=4096
)

print(response.content)
print(response.usage)  # Token usage
```

#### stream()

```python
async def stream(
    self,
    messages: list[Message],
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tools: list[ToolDefinition] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]:
    """Stream a chat completion response.

    Args:
        messages: List of conversation messages
        model: Model identifier
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        tools: Available tools for the model to use
        **kwargs: Additional provider-specific parameters

    Yields:
        StreamChunk objects with incremental content
    """
```

**Examples**:

```python
async for chunk in provider.stream(messages, model="llama3"):
    print(chunk.content, end="", flush=True)
    if chunk.is_final:
        print(f"\nTokens: {chunk.usage}")
```

#### discover_capabilities()

```python
async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
    """Discover capabilities for the given model.

    Args:
        model: Model identifier

    Returns:
        ProviderRuntimeCapabilities with model capabilities
    """
```

**Examples**:

```python
capabilities = await provider.discover_capabilities("claude-sonnet-4-5-20250514")
print(f"Context window: {capabilities.context_window}")
print(f"Supports tools: {capabilities.supports_tools}")
print(f"Supports streaming: {capabilities.supports_streaming}")
```

## Protocol Classes

### StreamingProvider

Protocol for providers that support streaming responses.

```python
@runtime_checkable
class StreamingProvider(Protocol):
    """Protocol for providers that support streaming responses."""

    def supports_streaming(self) -> bool:
        """Whether the provider supports streaming responses."""
        ...

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response."""
        ...
```

**Usage**:

```python
from victor.providers import is_streaming_provider, StreamingProvider

if isinstance(provider, StreamingProvider):
    async for chunk in provider.stream(messages, model=model):
        print(chunk.content, end="")

# Or use helper
if is_streaming_provider(provider):
    async for chunk in provider.stream(messages, model=model):
        print(chunk.content, end="")
```

### ToolCallingProvider

Protocol for providers that support tool/function calling.

```python
@runtime_checkable
class ToolCallingProvider(Protocol):
    """Protocol for providers that support tool/function calling."""

    def supports_tools(self) -> bool:
        """Whether the provider supports tool/function calling."""
        ...
```

**Usage**:

```python
from victor.providers import is_tool_calling_provider, ToolCallingProvider

if isinstance(provider, ToolCallingProvider):
    response = await provider.chat(messages, model=model, tools=my_tools)
    if response.tool_calls:
        # Process tool calls
        ...

# Or use helper
if is_tool_calling_provider(provider):
    response = await provider.chat(messages, model=model, tools=my_tools)
```

## Data Models

### Message

Standard message format across all providers.

```python
class Message(BaseModel):
    """Standard message format across all providers."""

    role: str                    # "system", "user", or "assistant"
    content: str                 # Message content
    name: str | None = None      # Optional sender name
    tool_calls: list[dict] | None = None   # Tool calls requested by assistant
    tool_call_id: str | None = None       # ID of tool call being responded to
```

**Examples**:

```python
# System message
system_msg = Message(role="system", content="You are a helpful assistant.")

# User message
user_msg = Message(role="user", content="What is Victor?")

# Assistant message with tool calls
assistant_msg = Message(
    role="assistant",
    content="I'll check that for you.",
    tool_calls=[{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "search",
            "arguments": '{"query": "Victor"}'
        }
    }]
)

# Tool response message
tool_response = Message(
    role="tool",
    content="Victor is an open-source agentic AI framework.",
    tool_call_id="call_123"
)
```

### ToolDefinition

Standard tool definition format.

```python
class ToolDefinition(BaseModel):
    """Standard tool definition format."""

    name: str                       # Tool name
    description: str                # What the tool does
    parameters: dict[str, Any]       # JSON Schema for tool parameters
```

**Examples**:

```python
search_tool = ToolDefinition(
    name="search",
    description="Search the codebase for a query",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
)

response = await provider.chat(
    messages,
    model=model,
    tools=[search_tool]
)
```

### CompletionResponse

Standard completion response format.

```python
class CompletionResponse(BaseModel):
    """Standard completion response format."""

    content: str                           # Generated content
    role: str = "assistant"                 # Response role
    tool_calls: list[dict] | None = None   # Tool calls requested
    stop_reason: str | None = None         # Why generation stopped
    usage: dict[str, int] | None = None    # Token usage stats
    model: str | None = None               # Model used
    raw_response: dict[str, Any] | None = None  # Raw provider response
    metadata: dict[str, Any] | None = None     # Additional metadata
```

**Examples**:

```python
response = await provider.chat(messages, model=model)

print(response.content)           # Generated text
print(response.stop_reason)        # "stop", "length", "tool_calls", etc.
print(response.usage)              # {"prompt_tokens": 10, "completion_tokens": 20, ...}
print(response.model)              # Model identifier

if response.tool_calls:
    for call in response.tool_calls:
        print(f"Call: {call['function']['name']}")
        print(f"Args: {call['function']['arguments']}")
```

### StreamChunk

Streaming response chunk.

```python
class StreamChunk(BaseModel):
    """Streaming response chunk."""

    content: str = ""                        # Incremental content
    tool_calls: list[dict] | None = None     # Tool calls in this chunk
    stop_reason: str | None = None           # Stop reason (final chunk)
    is_final: bool = False                   # Is this the final chunk
    metadata: dict[str, Any] | None = None   # Additional metadata
    usage: dict[str, int] | None = None      # Token usage (final chunk)
```

**Examples**:

```python
async for chunk in provider.stream(messages, model=model):
    print(chunk.content, end="", flush=True)

    if chunk.is_final:
        print(f"\nStopped: {chunk.stop_reason}")
        print(f"Usage: {chunk.usage}")
```

## Provider Registry

### Get Provider

```python
from victor.providers import ProviderRegistry

registry = ProviderRegistry.get_instance()

# Get provider by name
provider = registry.get_provider("anthropic")

# Get provider with config
provider = registry.get_provider(
    "ollama",
    api_key=None,
    base_url="http://localhost:11434"
)
```

### List Providers

```python
# List all available providers
providers = registry.list_providers()
print(providers)  # ["anthropic", "openai", "ollama", ...]
```

### Register Provider

```python
from victor.providers import BaseProvider

class MyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    async def chat(self, messages, *, model, **kwargs):
        # Implementation
        pass

    async def stream(self, messages, *, model, **kwargs):
        # Implementation
        yield StreamChunk()

# Register provider
registry.register_provider(MyProvider)
```

## Circuit Breaker

The circuit breaker protects against cascading failures by temporarily disabling providers that are experiencing issues.

### Circuit Breaker States

| State | Description |
|-------|-------------|
| `CLOSED` | Normal operation - requests pass through |
| `OPEN` | Circuit is open - requests fail immediately |
| `HALF_OPEN` | Testing recovery - allowing test requests |

### Configuration

```python
provider = Provider(
    "anthropic",
    use_circuit_breaker=True,
    circuit_breaker_failure_threshold=5,    # Open after 5 failures
    circuit_breaker_recovery_timeout=30.0   # Test recovery after 30s
)
```

### Get Circuit Breaker Stats

```python
stats = provider.get_circuit_breaker_stats()
print(stats)
# {
#     "state": "closed",
#     "failure_count": 0,
#     "success_count": 10,
#     "last_failure_time": None,
#     "last_success_time": "2025-03-02T12:00:00Z"
# }
```

## Error Handling

### Provider Errors

```python
from victor.providers import (
    ProviderError,
    ProviderNotFoundError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderInvalidResponseError,
)
```

| Error | Description |
|-------|-------------|
| `ProviderError` | Base class for all provider errors |
| `ProviderNotFoundError` | Provider not found in registry |
| `ProviderAuthError` | Authentication failed |
| `ProviderRateLimitError` | Rate limit exceeded |
| `ProviderTimeoutError` | Request timed out |
| `ProviderConnectionError` | Connection failed |
| `ProviderInvalidResponseError` | Invalid response from provider |

**Example**:

```python
try:
    response = await provider.chat(messages, model=model)
except ProviderAuthError:
    print("Invalid API key")
except ProviderRateLimitError as e:
    print(f"Rate limited: {e}")
    # Retry after delay
except ProviderTimeoutError:
    print("Request timed out")
except ProviderError as e:
    print(f"Provider error: {e}")
```

## Helper Functions

### is_streaming_provider()

```python
from victor.providers import is_streaming_provider

if is_streaming_provider(provider):
    async for chunk in provider.stream(messages, model=model):
        print(chunk.content, end="")
```

### is_tool_calling_provider()

```python
from victor.providers import is_tool_calling_provider

if is_tool_calling_provider(provider):
    response = await provider.chat(messages, model=model, tools=tools)
    if response.tool_calls:
        # Process tool calls
        ...
```

## Best Practices

### 1. Use Provider Registry

```python
# Good - Use registry
registry = ProviderRegistry.get_instance()
provider = registry.get_provider("anthropic")

# Avoid - Direct instantiation
from victor.providers.anthropic_provider import AnthropicProvider
provider = AnthropicProvider()  # Tied to specific implementation
```

### 2. Handle Provider Capabilities

```python
# Good - Check capabilities
if is_tool_calling_provider(provider):
    response = await provider.chat(messages, model=model, tools=tools)
else:
    response = await provider.chat(messages, model=model)

# Good - Use streaming when available
if is_streaming_provider(provider):
    async for chunk in provider.stream(messages, model=model):
        print(chunk.content, end="")
```

### 3. Use Circuit Breaker for Production

```python
# Good - Enable circuit breaker
provider = Provider(
    "anthropic",
    use_circuit_breaker=True,
    circuit_breaker_failure_threshold=5,
    circuit_breaker_recovery_timeout=30.0
)
```

### 4. Set Appropriate Timeouts

```python
# Good - Set timeout based on model size
provider = Provider(
    "ollama",
    timeout=120,  # 2 minutes for local models
)

# Good - Shorter timeout for fast cloud models
provider = Provider(
    "anthropic",
    timeout=30,  # 30 seconds for fast cloud API
)
```

### 5. Handle Errors Gracefully

```python
# Good - Contextual error handling
try:
    response = await provider.chat(messages, model=model)
except ProviderAuthError:
    logger.error("Invalid API key - check configuration")
except ProviderRateLimitError:
    logger.warning("Rate limited - backing off")
    await asyncio.sleep(60)
except ProviderTimeoutError:
    logger.warning("Request timed out - retrying")
```

## See Also

- [Agent API](agent.md) - Agent usage with providers
- [Configuration API](config.md) - Provider configuration
- [Core APIs](core.md) - Error handling and events
