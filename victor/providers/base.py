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

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

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

    content: str = Field(..., description="Generated content")
    role: str = Field(default="assistant", description="Response role")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls requested")
    stop_reason: Optional[str] = Field(None, description="Why generation stopped")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage stats")
    model: Optional[str] = Field(None, description="Model used")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw provider response")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata (e.g., reasoning_content)"
    )


class StreamChunk(BaseModel):
    """Streaming response chunk."""

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

        # Retry strategy (lazily initialized on first rate-limit retry)
        self._retry_strategy: Any = None

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

    def classify_error(self, error: Exception) -> ProviderError:
        """Classify a raw exception into the appropriate ProviderError subtype.

        Providers can override this for provider-specific error handling.
        The base implementation uses a three-tier strategy:
        1. Pass through existing ProviderError subtypes unchanged
        2. Check HTTP status codes (if available on the exception)
        3. String-based pattern matching as final fallback

        Args:
            error: The raw exception from the provider API call.

        Returns:
            A ProviderError (or subtype) wrapping the original exception.
        """
        # Tier 0: Already classified
        if isinstance(error, ProviderError):
            return error

        wrapped_error = (
            getattr(error, "last_error", None)
            or getattr(error, "raw_error", None)
            or getattr(error, "cause", None)
            or getattr(error, "__cause__", None)
        )
        if isinstance(wrapped_error, Exception) and wrapped_error is not error:
            return self.classify_error(wrapped_error)

        error_str = str(error).lower()

        # Tier 1: Check HTTP status code attributes
        # Also check .response.status_code for httpx.HTTPStatusError compatibility
        status = (
            getattr(error, "status_code", None)
            or getattr(error, "code", None)
            or getattr(getattr(error, "response", None), "status_code", None)
        )
        if isinstance(status, int):
            if status == 401 or status == 403:
                return ProviderAuthError(
                    message=f"Authentication failed: {error}",
                    provider=self.name,
                    status_code=status,
                    raw_error=error,
                )
            if status == 429:
                return ProviderRateLimitError(
                    message=f"Rate limit exceeded: {error}",
                    provider=self.name,
                    status_code=429,
                    raw_error=error,
                )

        # Tier 2: String-based classification (deprecated — providers should use
        # proper exception types or HTTP status codes instead)
        if any(
            t in error_str
            for t in ("auth", "unauthorized", "invalid key", "invalid api", "api_key", "401")
        ):
            logger.warning(
                "String-based error classification triggered for auth error "
                "from %s. Provider should use proper exception types.",
                self.name,
            )
            return ProviderAuthError(
                message=f"Authentication failed: {error}",
                provider=self.name,
                raw_error=error,
            )
        if any(t in error_str for t in ("rate limit", "429", "too many requests")):
            logger.warning(
                "String-based error classification triggered for rate limit "
                "from %s. Provider should use proper exception types.",
                self.name,
            )
            return ProviderRateLimitError(
                message=f"Rate limit exceeded: {error}",
                provider=self.name,
                status_code=429,
                raw_error=error,
            )
        if any(t in error_str for t in ("timeout", "timed out")):
            logger.warning(
                "String-based error classification triggered for timeout "
                "from %s. Provider should use proper exception types.",
                self.name,
            )
            return ProviderTimeoutError(
                message=f"Request timed out: {error}",
                provider=self.name,
                raw_error=error,
            )

        # Default: generic ProviderError
        return ProviderError(
            message=f"{self.name} API error: {error}",
            provider=self.name,
            status_code=status if isinstance(status, int) else None,
            raw_error=error,
        )

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
        if False:
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

        Uses the fast native token counter when available and falls back
        to word-based estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        from victor.processing.native.tokenizer import count_tokens_fast

        return count_tokens_fast(text)

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections or resources."""
        pass

    async def __aenter__(self) -> "BaseProvider":
        """Enable async context manager usage."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close provider on context exit."""
        await self.close()

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
        """Execute a function with circuit breaker and retry-on-rate-limit.

        When ``max_retries > 0``, rate-limit errors (429) are automatically
        retried with exponential backoff and jitter, respecting ``Retry-After``
        headers. Set ``max_retries=0`` to raise immediately (old behavior).

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            ProviderRateLimitError: If rate-limit retries are exhausted
            Exception: If func raises a non-retryable error
        """

        async def _call() -> Any:
            if self._circuit_breaker:
                return await self._circuit_breaker.execute(func, *args, **kwargs)
            return await func(*args, **kwargs)

        if self.max_retries <= 0:
            return await _call()

        # Use ProviderRetryStrategy for automatic retry on 429/transient errors
        if self._retry_strategy is None:
            from victor.providers.resilience import (
                ProviderRetryConfig,
                ProviderRetryStrategy,
            )

            self._retry_strategy = ProviderRetryStrategy(
                ProviderRetryConfig(max_retries=self.max_retries)
            )

        return await self._retry_strategy.execute(_call)

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()
