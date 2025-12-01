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

"""Google Gemini provider implementation."""

from typing import Any, AsyncIterator, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models."""

    def __init__(
        self,
        api_key: str,
        timeout: int = 60,
        **kwargs: Any,
    ):
        """Initialize Google provider.

        Args:
            api_key: Google API key
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(api_key=api_key, timeout=timeout, **kwargs)
        genai.configure(api_key=api_key)

    @property
    def name(self) -> str:
        """Provider name."""
        return "google"

    def supports_tools(self) -> bool:
        """Google Gemini supports function calling."""
        return True

    def supports_streaming(self) -> bool:
        """Google supports streaming."""
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
        """Send chat completion request to Google.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "gemini-1.5-pro", "gemini-1.5-flash")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional Google parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Initialize model
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    **kwargs,
                },
            )

            # Convert messages to Gemini format
            history, latest_message = self._convert_messages(messages)

            # Start chat
            chat = model_instance.start_chat(history=history)

            # Generate response with circuit breaker protection
            response = await self._execute_with_circuit_breaker(
                chat.send_message_async, latest_message
            )

            return self._parse_response(response, model)

        except Exception as e:
            raise ProviderError(
                message=f"Google API error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

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
        """Stream chat completion from Google.

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
            # Initialize model
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    **kwargs,
                },
            )

            # Convert messages
            history, latest_message = self._convert_messages(messages)

            # Start chat
            chat = model_instance.start_chat(history=history)

            # Stream response
            response_stream = await chat.send_message_async(
                latest_message,
                stream=True,
            )

            async for chunk in response_stream:
                parsed_chunk = self._parse_stream_chunk(chunk)
                if parsed_chunk:
                    yield parsed_chunk

        except Exception as e:
            raise ProviderError(
                message=f"Google streaming error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

    def _convert_messages(self, messages: List[Message]) -> tuple[List[Dict[str, str]], str]:
        """Convert messages to Gemini format.

        Args:
            messages: Standard messages

        Returns:
            Tuple of (history, latest_message)
        """
        history = []
        latest_message = ""

        for i, msg in enumerate(messages):
            # Gemini uses "user" and "model" roles
            role = "model" if msg.role == "assistant" else "user"

            if i == len(messages) - 1:
                # Last message is sent separately
                latest_message = msg.content
            else:
                history.append(
                    {
                        "role": role,
                        "parts": [msg.content],
                    }
                )

        return history, latest_message

    def _parse_response(self, response: GenerateContentResponse, model: str) -> CompletionResponse:
        """Parse Google API response.

        Args:
            response: Raw Google response
            model: Model name

        Returns:
            Normalized CompletionResponse
        """
        # Extract text content
        content = ""
        if response.text:
            content = response.text

        # Parse usage (if available)
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return CompletionResponse(
            content=content,
            role="assistant",
            usage=usage,
            model=model,
        )

    def _parse_stream_chunk(self, chunk: Any) -> Optional[StreamChunk]:
        """Parse streaming chunk from Google.

        Args:
            chunk: Stream chunk

        Returns:
            StreamChunk or None
        """
        content = ""
        if hasattr(chunk, "text"):
            content = chunk.text

        # Check if this is the final chunk
        is_final = (
            hasattr(chunk, "candidates")
            and chunk.candidates
            and hasattr(chunk.candidates[0], "finish_reason")
            and chunk.candidates[0].finish_reason is not None
        )

        return StreamChunk(
            content=content,
            is_final=is_final,
        )

    async def close(self) -> None:
        """Close connections (Gemini client doesn't need explicit closing)."""
        pass
