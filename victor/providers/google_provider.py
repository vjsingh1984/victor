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
from google.generativeai.types import GenerateContentResponse, HarmBlockThreshold, HarmCategory

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)

# Safety threshold levels (from most to least restrictive)
SAFETY_LEVELS = {
    "block_none": HarmBlockThreshold.BLOCK_NONE,
    "block_few": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    "block_some": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "block_most": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models.

    Safety Settings:
        The `safety_level` parameter controls content filtering:
        - "block_none": No blocking (least restrictive, for development)
        - "block_few": Block only high probability harmful content
        - "block_some": Block medium and above (default)
        - "block_most": Block low and above (most restrictive)

    Example:
        # For code generation without safety blocks
        provider = GoogleProvider(api_key=key, safety_level="block_none")
    """

    def __init__(
        self,
        api_key: str,
        timeout: int = 60,
        safety_level: str = "block_none",
        **kwargs: Any,
    ):
        """Initialize Google provider.

        Args:
            api_key: Google API key
            timeout: Request timeout in seconds
            safety_level: Safety filter level - "block_none", "block_few",
                         "block_some", or "block_most" (default: "block_none")
            **kwargs: Additional configuration
        """
        super().__init__(api_key=api_key, timeout=timeout, **kwargs)
        genai.configure(api_key=api_key)

        # Configure safety settings
        threshold = SAFETY_LEVELS.get(safety_level, HarmBlockThreshold.BLOCK_NONE)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
        }

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
            # Initialize model with safety settings
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
                safety_settings=self.safety_settings,
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

        except ProviderError:
            raise
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
            # Initialize model with safety settings
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
                safety_settings=self.safety_settings,
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

        except ProviderError:
            raise
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

        Raises:
            ProviderError: If response was blocked by safety filters
        """
        # Check for blocked responses (safety filters, recitation, etc.)
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            # finish_reason values: 1=STOP, 2=SAFETY, 3=MAX_TOKENS, 4=RECITATION, 5=OTHER
            if finish_reason == 2:  # SAFETY
                safety_ratings = getattr(candidate, "safety_ratings", [])
                blocked_categories = [
                    f"{r.category.name}: {r.probability.name}"
                    for r in safety_ratings
                    if hasattr(r, "blocked") and r.blocked
                ]
                raise ProviderError(
                    message=f"Response blocked by safety filters. Categories: {blocked_categories or 'unknown'}",
                    provider=self.name,
                )
            elif finish_reason == 4:  # RECITATION
                raise ProviderError(
                    message="Response blocked due to potential recitation of copyrighted content",
                    provider=self.name,
                )

        # Extract text content safely
        content = ""
        try:
            if response.text:
                content = response.text
        except ValueError:
            # response.text raises ValueError if no valid parts
            # Try to extract from candidates directly
            if response.candidates and response.candidates[0].content.parts:
                content = "".join(
                    part.text
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "text")
                )

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

        Raises:
            ProviderError: If chunk was blocked by safety filters
        """
        # Check for safety blocks in streaming
        if hasattr(chunk, "candidates") and chunk.candidates:
            candidate = chunk.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason == 2:  # SAFETY
                raise ProviderError(
                    message="Streaming response blocked by safety filters",
                    provider=self.name,
                )

        # Extract content safely
        content = ""
        try:
            if hasattr(chunk, "text") and chunk.text:
                content = chunk.text
        except ValueError:
            # Ignore ValueError when text is not available
            pass

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
