"""OpenAI GPT provider implementation."""

from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from codingagent.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    StreamChunk,
    ToolDefinition,
)


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI GPT models."""

    def __init__(
        self,
        api_key: str,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            organization: OpenAI organization ID (optional)
            base_url: Optional base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, **kwargs)
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    def supports_tools(self) -> bool:
        """OpenAI supports function calling."""
        return True

    def supports_streaming(self) -> bool:
        """OpenAI supports streaming."""
        return True

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
        """Send chat completion request to OpenAI.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional OpenAI parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # Build request parameters
            request_params = {
                "model": model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }

            if tools:
                request_params["tools"] = self._convert_tools(tools)
                request_params["tool_choice"] = "auto"

            # Make API call
            response: ChatCompletion = await self.client.chat.completions.create(**request_params)

            return self._parse_response(response, model)

        except Exception as e:
            return self._handle_error(e)

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
        """Stream chat completion from OpenAI.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional parameters

        Yields:
            StreamChunk with incremental content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Convert messages
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            # Build request parameters
            request_params = {
                "model": model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                **kwargs,
            }

            if tools:
                request_params["tools"] = self._convert_tools(tools)
                request_params["tool_choice"] = "auto"

            # Stream response
            stream = await self.client.chat.completions.create(**request_params)

            async for chunk in stream:
                parsed_chunk = self._parse_stream_chunk(chunk)
                if parsed_chunk:
                    yield parsed_chunk

        except Exception as e:
            raise self._handle_error(e)

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tools to OpenAI format.

        Args:
            tools: Standard tool definitions

        Returns:
            OpenAI-formatted tools
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _parse_response(self, response: ChatCompletion, model: str) -> CompletionResponse:
        """Parse OpenAI API response.

        Args:
            response: Raw OpenAI response
            model: Model name

        Returns:
            Normalized CompletionResponse
        """
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        # Parse usage
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return CompletionResponse(
            content=message.content or "",
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            usage=usage,
            model=model,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def _parse_stream_chunk(self, chunk: ChatCompletionChunk) -> Optional[StreamChunk]:
        """Parse streaming chunk from OpenAI.

        Args:
            chunk: Stream chunk

        Returns:
            StreamChunk or None
        """
        if not chunk.choices:
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        content = delta.content or ""
        is_final = choice.finish_reason is not None

        return StreamChunk(
            content=content,
            stop_reason=choice.finish_reason,
            is_final=is_final,
        )

    def _handle_error(self, error: Exception) -> ProviderError:
        """Handle and convert API errors.

        Args:
            error: Original exception

        Returns:
            ProviderError with details

        Raises:
            ProviderError: Always raises after converting
        """
        error_msg = str(error)

        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            raise ProviderAuthenticationError(
                message=f"Authentication failed: {error_msg}",
                provider=self.name,
                raw_error=error,
            )
        elif "rate_limit" in error_msg.lower() or "429" in error_msg:
            raise ProviderRateLimitError(
                message=f"Rate limit exceeded: {error_msg}",
                provider=self.name,
                status_code=429,
                raw_error=error,
            )
        else:
            raise ProviderError(
                message=f"OpenAI API error: {error_msg}",
                provider=self.name,
                raw_error=error,
            )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.close()
