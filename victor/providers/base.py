# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
)
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities


# -----------------------------------------------------------------------------
# Protocol classes for Interface Segregation (ISP)
# These allow providers to optionally implement specific capabilities without
# requiring all providers to implement methods they don't support.
# -----------------------------------------------------------------------------


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
        """Whether the provider supports streaming responses.

        Returns:
            True if provider supports streaming, False otherwise
        """
        ...

    async def stream(
        self,
        messages: List["Message"],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List["ToolDefinition"]] = None,
        **kwargs: Any,
    ) -> AsyncIterator["StreamChunk"]:
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
        ...


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
        """Whether the provider supports tool/function calling.

        Returns:
            True if provider supports tools, False otherwise
        """
        ...


# Helper functions for type checking
def is_streaming_provider(provider: Any) -> bool:
    """Check if a provider supports streaming.

    This is a convenience function that checks both protocol implementation
    and the supports_streaming() method result.

    Args:
        provider: Provider instance to check

    Returns:
        True if provider supports streaming responses
    """
    if hasattr(provider, "supports_streaming"):
        return provider.supports_streaming()
    return False


def is_tool_calling_provider(provider: Any) -> bool:
    """Check if a provider supports tool calling.

    This is a convenience function that checks both protocol implementation
    and the supports_tools() method result.

    Args:
        provider: Provider instance to check

    Returns:
        True if provider supports tool/function calling
    """
    if hasattr(provider, "supports_tools"):
        return provider.supports_tools()
    return False


class Message(BaseModel):
    """Standard message format across all providers."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Optional name for the message sender")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tool calls requested by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="ID of the tool call being responded to"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return self.model_dump()


class ToolDefinition(BaseModel):
    """Standard tool definition format."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="What the tool does")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for tool parameters")


class CompletionResponse(BaseModel):
    """Standard completion response format."""

    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    content: str = Field(..., description="Generated content")
    role: str = Field(default="assistant", description="Response role")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls requested")
    stop_reason: Optional[str] = Field(None, description="Why generation stopped")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage stats")
    model: Optional[str] = Field(
        None,
        description="Model used",
        validation_alias=AliasChoices("model", "model_name"),
    )
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw provider response")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata (e.g., reasoning_content)"
    )

    @property
    def model_name(self) -> Optional[str]:
        """Backward-compatible alias for model."""
        return self.model


class StreamChunk(BaseModel):
    """Streaming response chunk."""

    model_config = ConfigDict(protected_namespaces=())

    content: str = Field(default="", description="Incremental content")
    tool_calls: Optional[List[Dict[str, Any]]] = None
    stop_reason: Optional[str] = None
    is_final: bool = Field(default=False, description="Is this the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata (e.g., reasoning_content)"
    )
    usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage stats (typically on final chunk). Keys: prompt_tokens, completion_tokens, total_tokens, cache_creation_input_tokens, cache_read_input_tokens",
    )
    model_name: Optional[str] = Field(default=None, description="Model used")


# Provider error classes - re-exported from victor/core/errors for backward compatibility
# All error classes are defined in victor/core/errors.py as the single source of truth
from victor.core.errors import (
    ProviderError,
    ProviderNotFoundError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderInvalidResponseError,
)


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

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
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_config = kwargs

        # Circuit breaker for resilience
        self._use_circuit_breaker = use_circuit_breaker
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if use_circuit_breaker:
            self._circuit_breaker = CircuitBreakerRegistry.get_or_create(
                name=f"provider_{self.__class__.__name__}",
                failure_threshold=circuit_breaker_failure_threshold,
                recovery_timeout=circuit_breaker_recovery_timeout,
                excluded_exceptions=(ProviderAuthError,),
            )

    @property
    def circuit_breaker(self) -> Optional[CircuitBreaker]:
        """Get the circuit breaker for this provider."""
        return self._circuit_breaker

    def supports_tools(self) -> bool:
        """Check if provider supports tool/function calling.

        Default implementation returns False. Providers that support tool calling
        should override this method to return True. This follows the Interface
        Segregation Principle - providers don't need to implement tool calling
        if they don't support it.

        See also: ToolCallingProvider protocol for type checking.

        Returns:
            True if provider supports tools, False otherwise (default)
        """
        return False

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming responses.

        Default implementation returns False. Providers that support streaming
        should override this method to return True. This follows the Interface
        Segregation Principle - providers don't need to implement streaming
        if they don't support it.

        See also: StreamingProvider protocol for type checking.

        Returns:
            True if provider supports streaming, False otherwise (default)
        """
        return False

    def get_circuit_breaker_stats(self) -> Optional[Dict[str, Any]]:
        """Get circuit breaker statistics for monitoring."""
        if self._circuit_breaker:
            return self._circuit_breaker.get_stats()
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

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
        pass

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
        # Abstract async generator - yield needed for mypy to recognize as generator
        if False:  # pragma: no cover (unreachable but needed for type checking)
            yield StreamChunk()
        raise NotImplementedError

    async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
        """Discover capabilities for the given model.

        Default implementation falls back to configured limits and
        provider-declared support flags. Providers should override
        with real HTTP-based discovery when available.
        """
        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits(self.name, model)
        return ProviderRuntimeCapabilities(
            provider=self.name,
            model=model,
            context_window=limits.context_window,
            supports_tools=self.supports_tools(),
            supports_streaming=self.supports_streaming(),
            source="config",
        )

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
        with different naming conventions across SDKs (OpenAI uses stream,
        Anthropic uses stream_chat, etc.).

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
        async for chunk in self.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        ):
            yield chunk

    async def count_tokens(self, text: str) -> int:
        """Estimate token count for given text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections or resources."""
        pass

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is open (failing fast).

        Returns:
            True if circuit is open and requests will be rejected
        """
        if self._circuit_breaker:
            return self._circuit_breaker.is_open
        return False

    async def _execute_with_circuit_breaker(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with circuit breaker protection.

        Use this method in subclass implementations to protect API calls.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If func raises and circuit records failure
        """
        if self._circuit_breaker:
            return await self._circuit_breaker.execute(func, *args, **kwargs)
        return await func(*args, **kwargs)

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()

    @classmethod
    def _resolve_api_key(cls, api_key: Optional[str], provider_name: str) -> str:
        """Resolve API key from multiple sources with proper fallback order.

        Resolution order:
        1. Provided api_key parameter
        2. Environment variable ({PROVIDER_NAME}_API_KEY)
        3. Keyring (via victor.config.api_keys.get_api_key)
        4. Empty string with warning logged

        Args:
            api_key: API key provided as parameter (highest priority)
            provider_name: Name of the provider (e.g., "anthropic", "openai")

        Returns:
            Resolved API key (empty string if not found, but doesn't raise)

        Example:
            class MyProvider(BaseProvider):
                def __init__(self, api_key=None):
                    resolved_key = self._resolve_api_key(api_key, "myprovider")
                    super().__init__(api_key=resolved_key)
        """
        import os
        import logging

        # 1. Use provided key if available
        if api_key:
            return api_key

        # 2. Try environment variable
        env_var_name = f"{provider_name.upper()}_API_KEY"
        env_key = os.environ.get(env_var_name, "")
        if env_key:
            return env_key

        # 3. Try keyring
        try:
            from victor.config.api_keys import get_api_key

            keyring_key = get_api_key(provider_name)
            if keyring_key:
                return keyring_key
        except ImportError:
            pass

        # 4. Not found - log warning and return empty string
        logger = logging.getLogger(__name__)
        logger.warning(
            "%s API key not provided. Set %s environment variable, "
            "use 'victor keys --set %s --keyring', or pass api_key parameter.",
            provider_name.capitalize(),
            env_var_name,
            provider_name,
        )

        return ""
