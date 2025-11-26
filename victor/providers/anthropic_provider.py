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

"""Anthropic Claude provider implementation."""

from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic
from anthropic.types import ContentBlock, Message as AnthropicMessage, MessageStreamEvent

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    StreamChunk,
    ToolDefinition,
)


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            base_url: Optional base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, **kwargs)
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    def supports_tools(self) -> bool:
        """Anthropic supports tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Anthropic supports streaming."""
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
        """Send chat completion request to Anthropic.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "claude-sonnet-4-5", "claude-3-opus")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional Anthropic parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Separate system messages from conversation
            system_message = None
            conversation_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    conversation_messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            # Build request parameters
            request_params = {
                "model": model,
                "messages": conversation_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }

            if system_message:
                request_params["system"] = system_message

            if tools:
                request_params["tools"] = self._convert_tools(tools)

            # Make API call
            response: AnthropicMessage = await self.client.messages.create(**request_params)

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
        """Stream chat completion from Anthropic.

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
            # Separate system messages
            system_message = None
            conversation_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    conversation_messages.append({
                        "role": msg.role,
                        "content": msg.content,
                    })

            # Build request parameters
            request_params = {
                "model": model,
                "messages": conversation_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }

            if system_message:
                request_params["system"] = system_message

            if tools:
                request_params["tools"] = self._convert_tools(tools)

            # Stream response
            async with self.client.messages.stream(**request_params) as stream:
                async for event in stream:
                    chunk = self._parse_stream_event(event)
                    if chunk:
                        yield chunk

        except Exception as e:
            raise self._handle_error(e)

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tools to Anthropic format.

        Args:
            tools: Standard tool definitions

        Returns:
            Anthropic-formatted tools
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _parse_response(self, response: AnthropicMessage, model: str) -> CompletionResponse:
        """Parse Anthropic API response.

        Args:
            response: Raw Anthropic response
            model: Model name

        Returns:
            Normalized CompletionResponse
        """
        # Extract text content
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        # Parse usage
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=response.stop_reason,
            usage=usage,
            model=model,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def _parse_stream_event(self, event: MessageStreamEvent) -> Optional[StreamChunk]:
        """Parse streaming event from Anthropic.

        Args:
            event: Stream event

        Returns:
            StreamChunk or None
        """
        if event.type == "content_block_delta":
            if hasattr(event.delta, "text"):
                return StreamChunk(
                    content=event.delta.text,
                    is_final=False,
                )
        elif event.type == "message_stop":
            return StreamChunk(
                content="",
                is_final=True,
            )

        return None

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
                message=f"Anthropic API error: {error_msg}",
                provider=self.name,
                raw_error=error,
            )

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.close()
