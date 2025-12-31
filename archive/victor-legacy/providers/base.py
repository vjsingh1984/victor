"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Standard message format across all providers."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the message sender")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool calls requested by the assistant"
    )
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call being responded to")


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


class StreamChunk(BaseModel):
    """Streaming response chunk."""

    content: str = Field(default="", description="Incremental content")
    tool_calls: Optional[List[Dict[str, Any]]] = None
    stop_reason: Optional[str] = None
    is_final: bool = Field(default=False, description="Is this the final chunk")


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize provider.

        Args:
            api_key: API key for authentication (if required)
            base_url: Base URL for API endpoints
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional provider-specific options
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_config = kwargs

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
        pass

    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if provider supports tool/function calling.

        Returns:
            True if provider supports tools, False otherwise
        """
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming responses.

        Returns:
            True if provider supports streaming, False otherwise
        """
        pass

    def normalize_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert standard messages to provider-specific format.

        Args:
            messages: List of standard Message objects

        Returns:
            List of provider-specific message dictionaries
        """
        # Default implementation returns messages as dicts
        return [msg.model_dump(exclude_none=True) for msg in messages]

    def normalize_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tool definitions to provider-specific format.

        Args:
            tools: List of standard ToolDefinition objects

        Returns:
            List of provider-specific tool dictionaries
        """
        # Default implementation returns tools as dicts
        return [tool.model_dump(exclude_none=True) for tool in tools]

    async def count_tokens(self, text: str) -> int:
        """Estimate token count for given text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    async def close(self) -> None:
        """Close any open connections or resources."""
        pass


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        raw_error: Optional[Any] = None,
    ):
        """Initialize error.

        Args:
            message: Error message
            provider: Provider name
            status_code: HTTP status code if applicable
            raw_error: Raw error from provider
        """
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.raw_error = raw_error


class ProviderNotFoundError(ProviderError):
    """Raised when a provider is not found."""

    pass


class ProviderAuthenticationError(ProviderError):
    """Raised when authentication fails."""

    pass


class ProviderRateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    pass


class ProviderTimeoutError(ProviderError):
    """Raised when request times out."""

    pass
