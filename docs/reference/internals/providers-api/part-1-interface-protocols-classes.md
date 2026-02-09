# Providers API Reference - Part 1

**Part 1 of 3:** BaseProvider Interface, Protocols, and Key Classes

---

## Navigation

- **[Part 1: Interface, Protocols, Classes](#)** (Current)
- [Part 2: Implementation & Config](part-2-implementation-config.md)
- [Part 3: Error Handling & Reference](part-3-error-handling-reference.md)
- [**Complete Reference**](../providers-api.md)

---
# Providers API Reference

This document provides comprehensive API documentation for Victor's LLM provider system,
  which supports 21 different providers with a unified interface.

## Table of Contents

1. [BaseProvider Interface](#baseprovider-interface)
2. [Provider Protocols](#provider-protocols)
3. [Key Classes](#key-classes)
4. [Provider Implementation Pattern](#provider-implementation-pattern)
5. [Configuration](#configuration)
6. [Error Handling](#error-handling)

---

## BaseProvider Interface

All LLM providers in Victor inherit from `BaseProvider`, which defines the core contract for interacting with language models.

**Import:**
```python
from victor.providers.base import BaseProvider
```

### Constructor

```python
class BaseProvider(ABC):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        use_circuit_breaker: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: float = 30.0,
        **kwargs: Any,
    ):
        """Initialize provider.

        Args:
            api_key: API key for authentication (if required)
            base_url: Base URL for API endpoints
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            use_circuit_breaker: Whether to enable circuit breaker protection
            circuit_breaker_failure_threshold: Failures before opening circuit
            circuit_breaker_recovery_timeout: Seconds before testing recovery
            **kwargs: Additional provider-specific options
        """
```

### Abstract Properties

#### `name`
```python
@property
@abstractmethod
def name(self) -> str:
    """Provider name (e.g., 'anthropic', 'openai', 'ollama')."""
```

### Abstract Methods

#### `chat()`
```python
@abstractmethod
async def chat(
    self,
    messages: List[Message],
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tools: Optional[List[ToolDefinition]] = None,
    **kwargs: Any,
) -> CompletionResponse:
    """Send a chat completion request.

    Args:
        messages: List of conversation messages
        model: Model identifier (e.g., 'claude-sonnet-4-20250514')
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

#### `stream()`
```python
@abstractmethod
async def stream(
    self,
    messages: List[Message],
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tools: Optional[List[ToolDefinition]] = None,
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

    Raises:
        ProviderError: If the request fails
    """
```

#### `close()`
```python
@abstractmethod
async def close(self) -> None:
    """Close any open connections or resources."""
```

### Optional Methods

#### `supports_tools()`
```python
def supports_tools(self) -> bool:
    """Check if provider supports tool/function calling.

    Default implementation returns False. Providers that support
    tool calling should override this method to return True.

    Returns:
        True if provider supports tools, False otherwise (default)
    """
```

#### `supports_streaming()`
```python
def supports_streaming(self) -> bool:
    """Check if provider supports streaming responses.

    Default implementation returns False. Providers that support
    streaming should override this method to return True.

    Returns:
        True if provider supports streaming, False otherwise (default)
    """
```

#### `stream_chat()`
```python
async def stream_chat(
    self,
    messages: List[Message],
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tools: Optional[List[ToolDefinition]] = None,
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]:
    """Stream a chat completion response (alias for stream()).

    This is an alias for the stream() method, provided for compatibility
    with different naming conventions across SDKs.
    """
```

#### `discover_capabilities()`
```python
async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
    """Discover capabilities for the given model.

    Default implementation falls back to configured limits and
    provider-declared support flags. Providers should override
    with real HTTP-based discovery when available.

    Returns:
        ProviderRuntimeCapabilities with context window, tool support, etc.
    """
```

#### `count_tokens()`
```python
async def count_tokens(self, text: str) -> int:
    """Estimate token count for given text.

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count (default: ~4 characters per token)
    """
```

### Circuit Breaker Methods

#### `is_circuit_open()`
```python
def is_circuit_open(self) -> bool:
    """Check if the circuit breaker is open (failing fast).

    Returns:
        True if circuit is open and requests will be rejected
    """
```

#### `get_circuit_breaker_stats()`
```python
def get_circuit_breaker_stats(self) -> Optional[Dict[str, Any]]:
    """Get circuit breaker statistics for monitoring.

    Returns:
        Dictionary with stats or None if circuit breaker disabled
    """
```

#### `reset_circuit_breaker()`
```python
def reset_circuit_breaker(self) -> None:
    """Manually reset the circuit breaker to closed state."""
```

---

## Provider Protocols

Victor uses Protocol classes (PEP 544) for Interface Segregation, allowing providers to optionally implement specific
  capabilities.

**Import:**
```python
from victor.providers.base import (
    StreamingProvider,
    ToolCallingProvider,
    is_streaming_provider,
    is_tool_calling_provider,
)
```

### StreamingProvider

Protocol for providers that support streaming responses.

```python
@runtime_checkable
class StreamingProvider(Protocol):
    """Protocol for providers that support streaming responses.

    Providers implementing this protocol can stream chat completions
    incrementally rather than returning the full response at once.

    Example:
        class MyProvider(BaseProvider):
            def supports_streaming(self) -> bool:
                return True

            async def stream(self, messages, **kwargs) -> AsyncIterator[StreamChunk]:
                # Implementation here
                ...

    Type checking:
        if isinstance(provider, StreamingProvider):
            async for chunk in provider.stream(messages, model=model):
                print(chunk.content)
    """

    def supports_streaming(self) -> bool:
        """Whether the provider supports streaming responses."""
        ...

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response."""
        ...
```

### ToolCallingProvider

Protocol for providers that support tool/function calling.

```python
@runtime_checkable
class ToolCallingProvider(Protocol):
    """Protocol for providers that support tool/function calling.

    Providers implementing this protocol can receive tool definitions
    and return structured tool calls in their responses.

    Example:
        class MyProvider(BaseProvider):
            def supports_tools(self) -> bool:
                return True

            async def chat(self, messages, *, model, tools=None, **kwargs):
                # Handle tools in implementation
                ...

    Type checking:
        if isinstance(provider, ToolCallingProvider):
            response = await provider.chat(
                messages, model=model, tools=my_tools
            )
            if response.tool_calls:
                # Process tool calls
                ...
    """

    def supports_tools(self) -> bool:
        """Whether the provider supports tool/function calling."""
        ...
```

### Helper Functions

```python
def is_streaming_provider(provider: Any) -> bool:
    """Check if a provider supports streaming.

    This is a convenience function that checks both protocol implementation
    and the supports_streaming() method result.

    Args:
        provider: Provider instance to check

    Returns:
        True if provider supports streaming responses
    """

def is_tool_calling_provider(provider: Any) -> bool:
    """Check if a provider supports tool calling.

    This is a convenience function that checks both protocol implementation
    and the supports_tools() method result.

    Args:
        provider: Provider instance to check

    Returns:
        True if provider supports tool/function calling
    """
```

---

## Key Classes

### Message

Standard message format across all providers.

```python
from victor.providers.base import Message

class Message(BaseModel):
    """Standard message format across all providers."""

    role: str
    """Message role: 'system', 'user', or 'assistant'"""

    content: str
    """Message content"""

    name: Optional[str] = None
    """Optional name for the message sender"""

    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Tool calls requested by the assistant"""

    tool_call_id: Optional[str] = None
    """ID of the tool call being responded to"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
```

### ToolDefinition

Standard tool definition format.

```python
from victor.providers.base import ToolDefinition

class ToolDefinition(BaseModel):
    """Standard tool definition format."""

    name: str
    """Tool name"""

    description: str
    """What the tool does"""

    parameters: Dict[str, Any]
    """JSON Schema for tool parameters"""
```

### CompletionResponse

Standard completion response format.

```python
from victor.providers.base import CompletionResponse

class CompletionResponse(BaseModel):
    """Standard completion response format."""

    content: str
    """Generated content"""

    role: str = "assistant"
    """Response role"""

    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Tool calls requested by the model. Each tool call has:
    - id: Unique identifier for the tool call
    - name: Name of the tool to call
    - arguments: Dict or JSON string of arguments
    """

    stop_reason: Optional[str] = None
    """Why generation stopped (e.g., 'stop', 'length', 'tool_use')"""

    usage: Optional[Dict[str, int]] = None
    """Token usage stats:
    - prompt_tokens: Input tokens
    - completion_tokens: Output tokens
    - total_tokens: Total tokens
    """

    model: Optional[str] = None
    """Model used for generation"""

    raw_response: Optional[Dict[str, Any]] = None
    """Raw provider response for debugging"""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata (e.g., reasoning_content for thinking models)"""
```

### StreamChunk

Streaming response chunk.

```python
from victor.providers.base import StreamChunk

class StreamChunk(BaseModel):
    """Streaming response chunk."""

    content: str = ""
    """Incremental content"""

    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Tool calls (typically on final chunk)"""

    stop_reason: Optional[str] = None
    """Why generation stopped"""

    is_final: bool = False
    """Is this the final chunk"""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata (e.g., reasoning_content)"""

    usage: Optional[Dict[str, int]] = None
    """Token usage stats (typically on final chunk). Keys:
    - prompt_tokens
    - completion_tokens
    - total_tokens
    - cache_creation_input_tokens (Anthropic)
    - cache_read_input_tokens (Anthropic)
    """
```

### ProviderRuntimeCapabilities

Runtime capabilities discovered for a provider/model pair.

```python
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities

@dataclass
class ProviderRuntimeCapabilities:
    """Runtime capabilities and limits for a provider/model pair."""

    provider: str
    """Provider name"""

    model: str
    """Model identifier"""

    context_window: int
    """Maximum context window in tokens"""

    supports_tools: bool
    """Whether the model supports tool calling"""

    supports_streaming: bool
    """Whether the model supports streaming"""

    source: str = "config"
    """How capabilities were determined: 'discovered' or 'config'"""

    raw: Optional[Dict[str, Any]] = None
    """Raw discovery response"""
```

### ProviderRegistry

Registry for managing and discovering LLM providers.

```python
from victor.providers.registry import ProviderRegistry

class ProviderRegistry:
    """Registry for LLM provider management.

    This is a static class facade that maintains backward compatibility
    while delegating to a BaseRegistry-based implementation.
    """

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a provider.

        Args:
            name: Provider name (e.g., "ollama", "anthropic")
            provider_class: Provider class
        """

    @classmethod
    def get(cls, name: str) -> Type[BaseProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """

    @classmethod
    def get_optional(cls, name: str) -> Optional[Type[BaseProvider]]:
        """Get a provider class by name, returning None if not found."""

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseProvider:
        """Create a provider instance.

        Args:
            name: Provider name
            **kwargs: Provider initialization arguments

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If provider not found
        """

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered."""

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a provider."""

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
```

### ProviderManager

Unified management of providers and models with health monitoring and fallback handling.

```python
from victor.agent.provider_manager import ProviderManager, ProviderManagerConfig

@dataclass
class ProviderManagerConfig:
    """Configuration for ProviderManager."""

    enable_health_checks: bool = True
    """Enable provider health monitoring"""

    health_check_interval: float = 60.0
    """Interval between health checks (seconds)"""

    auto_fallback: bool = True
    """Automatically fallback to healthy provider on failure"""

    fallback_providers: List[str] = field(default_factory=list)
    """Ordered list of fallback providers"""

    max_fallback_attempts: int = 3
    """Maximum fallback attempts before giving up"""


class ProviderManager:
    """Unified management of providers and models.

    Features:
    - Hot-swap providers without losing context
    - Automatic health monitoring
    - Fallback chain support
    - Switch history tracking
    - Tool capability detection
    """

    def __init__(
        self,
        settings: Any,
        initial_provider: Optional[BaseProvider] = None,
        initial_model: Optional[str] = None,
        provider_name: Optional[str] = None,
        config: Optional[ProviderManagerConfig] = None,
    ):
        """Initialize the provider manager."""

    @property
    def provider(self) -> Optional[BaseProvider]:
        """Get current provider instance."""

    @property
    def provider_name(self) -> str:
        """Get current provider name."""

    @property
    def model(self) -> str:
        """Get current model name."""

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        reason: SwitchReason = SwitchReason.USER_REQUEST,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider.

        Args:
            provider_name: Name of the provider
            model: Optional model name
            reason: Reason for the switch
            **provider_kwargs: Additional provider arguments

        Returns:
            True if switch was successful
        """

    async def switch_model(
        self,
        model: str,
        reason: SwitchReason = SwitchReason.USER_REQUEST,
    ) -> bool:
        """Switch to a different model on the current provider."""

    def get_info(self) -> Dict[str, Any]:
        """Get information about current provider and model.

        Returns:
            Dictionary with provider/model info and capabilities
        """

    async def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers sorted by latency."""

    async def close(self) -> None:
        """Close provider and cleanup."""
```

### ManagedProviderFactory

Factory for creating providers with shared infrastructure (resilience, rate limiting, metrics).

```python
from victor.providers.factory import ManagedProviderFactory, ManagedProvider

class ManagedProviderFactory:
    """Factory for creating providers with shared infrastructure.

    Automatically integrates:
    - Circuit breaker and retry logic
    - Rate limiting with priority queues
    - Streaming metrics collection
    - Fallback provider chains
    """

    @classmethod
    async def create(
        cls,
        provider_name: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        enable_resilience: bool = True,
        enable_rate_limiting: bool = True,
        enable_metrics: bool = True,
        fallback_configs: Optional[List[ProviderConfig]] = None,
        **kwargs: Any,
    ) -> ManagedProvider:
        """Create an enhanced provider with shared infrastructure.

        Args:
            provider_name: Provider name (anthropic, openai, ollama, etc.)
            model: Model identifier
            api_key: API key for cloud providers
            base_url: Base URL override
            timeout: Request timeout
            enable_resilience: Enable circuit breaker and retry
            enable_rate_limiting: Enable rate limiting
            enable_metrics: Enable streaming metrics
            fallback_configs: Fallback provider configurations
            **kwargs: Additional provider-specific arguments

        Returns:
            ManagedProvider with all integrations
        """
```


**Reading Time:** 10 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## Provider Implementation Pattern
