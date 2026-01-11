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

"""OpenAI GPT provider implementation."""

import os
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.openai_compat import (
    convert_tools_to_openai_format,
)


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI GPT models."""

    # O-series reasoning models have different parameter requirements
    O_SERIES_MODELS = {"o1", "o1-pro", "o3", "o3-mini"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var, or use keyring)
            organization: OpenAI organization ID (optional)
            base_url: Optional base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        # Resolution order: parameter → env var → keyring → warning
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            try:
                from victor.config.api_keys import get_api_key

                resolved_key = get_api_key("openai") or ""
            except ImportError:
                pass

        if not resolved_key:
            import logging

            logging.getLogger(__name__).warning(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable, "
                "use 'victor keys --set openai --keyring', or pass api_key parameter."
            )

        super().__init__(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self.client = AsyncOpenAI(
            api_key=resolved_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _is_o_series_model(self, model: str) -> bool:
        """Check if model is an O-series reasoning model.

        O-series models have different parameter requirements:
        - Use max_completion_tokens instead of max_tokens
        - Don't support temperature parameter
        - Don't support tools/function calling

        Args:
            model: Model name

        Returns:
            True if model is O-series
        """
        model_lower = model.lower()
        return any(model_lower.startswith(prefix) for prefix in ["o1", "o3"])

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
            model: Model name (e.g., "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o3")
            temperature: Sampling temperature (ignored for O-series)
            max_tokens: Maximum tokens to generate
            tools: Available tools (not supported for O-series)
            **kwargs: Additional OpenAI parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Convert messages to OpenAI format
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Check if O-series reasoning model
            is_o_series = self._is_o_series_model(model)

            # Build request parameters
            request_params = {
                "model": model,
                "messages": openai_messages,
                **kwargs,
            }

            if is_o_series:
                # O-series models use max_completion_tokens instead of max_tokens
                # and don't support temperature or tools
                request_params["max_completion_tokens"] = max_tokens
                # Remove any temperature if passed in kwargs
                request_params.pop("temperature", None)
            else:
                # Standard models use max_tokens and temperature
                request_params["temperature"] = temperature
                request_params["max_tokens"] = max_tokens

                if tools:
                    request_params["tools"] = self._convert_tools(tools)
                    request_params["tool_choice"] = "auto"

            # Make API call with circuit breaker protection
            response: ChatCompletion = await self._execute_with_circuit_breaker(
                self.client.chat.completions.create, **request_params
            )

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
            temperature: Sampling temperature (ignored for O-series)
            max_tokens: Maximum tokens to generate
            tools: Available tools (not supported for O-series)
            **kwargs: Additional parameters

        Yields:
            StreamChunk with incremental content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Convert messages
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Check if O-series reasoning model
            is_o_series = self._is_o_series_model(model)

            # Build request parameters
            request_params = {
                "model": model,
                "messages": openai_messages,
                "stream": True,
                # Request usage info in final chunk
                "stream_options": {"include_usage": True},
                **kwargs,
            }

            if is_o_series:
                # O-series models use max_completion_tokens instead of max_tokens
                request_params["max_completion_tokens"] = max_tokens
                # Remove any temperature if passed in kwargs
                request_params.pop("temperature", None)
            else:
                # Standard models use max_tokens and temperature
                request_params["temperature"] = temperature
                request_params["max_tokens"] = max_tokens

                if tools:
                    request_params["tools"] = self._convert_tools(tools)
                    request_params["tool_choice"] = "auto"

            # Accumulate tool calls across streaming deltas to avoid emitting partial JSON
            tool_call_accumulator: Dict[str, Dict[str, Any]] = {}
            tool_call_indices: Dict[int, str] = {}

            # Stream response
            stream = await self.client.chat.completions.create(**request_params)

            async for chunk in stream:
                parsed_chunk = self._parse_stream_chunk(
                    chunk, tool_call_accumulator, tool_call_indices
                )
                if parsed_chunk:
                    yield parsed_chunk

        except Exception as e:
            raise self._handle_error(e)

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tools to OpenAI format."""
        return convert_tools_to_openai_format(tools)

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

    def _parse_stream_chunk(
        self,
        chunk: ChatCompletionChunk,
        tool_call_accumulator: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_call_indices: Optional[Dict[int, str]] = None,
    ) -> Optional[StreamChunk]:
        """Parse streaming chunk from OpenAI.

        Args:
            chunk: Stream chunk
            tool_call_accumulator: Aggregates partial tool calls across deltas
            tool_call_indices: Tracks mapping of tool call indices to IDs

        Returns:
            StreamChunk or None
        """
        # Parse usage from final chunk (sent when stream_options.include_usage=True)
        usage: Optional[Dict[str, int]] = None
        if hasattr(chunk, "usage") and chunk.usage:
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens or 0,
                "completion_tokens": chunk.usage.completion_tokens or 0,
                "total_tokens": chunk.usage.total_tokens or 0,
            }

        if not chunk.choices:
            # Usage-only chunk (no content)
            if usage:
                return StreamChunk(content="", is_final=True, usage=usage)
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        content = delta.content or ""
        finish_reason = choice.finish_reason
        is_final = finish_reason is not None
        tool_calls: Optional[List[Dict[str, Any]]] = None

        # Aggregate tool call deltas so the orchestrator receives complete JSON
        if getattr(delta, "tool_calls", None):
            if tool_call_accumulator is None:
                tool_call_accumulator = {}
            if tool_call_indices is None:
                tool_call_indices = {}

            for tc in delta.tool_calls or []:
                tc_index = getattr(tc, "index", None)
                tc_id = tc.id
                if tc_id:
                    if tc_index is not None:
                        tool_call_indices[tc_index] = tc_id
                else:
                    if tc_index is not None and tc_index in tool_call_indices:
                        tc_id = tool_call_indices[tc_index]
                    else:
                        tc_id = (
                            f"tool_call_{tc_index}"
                            if tc_index is not None
                            else f"tool_call_{len(tool_call_accumulator) + 1}"
                        )
                        if tc_index is not None:
                            tool_call_indices[tc_index] = tc_id

                function = getattr(tc, "function", None)
                name = getattr(function, "name", None)
                arguments = getattr(function, "arguments", "") or ""

                accumulated = tool_call_accumulator.setdefault(
                    tc_id, {"id": tc_id, "name": name or "", "arguments": ""}
                )
                # Keep latest name in case it streams late
                if name:
                    accumulated["name"] = name
                if arguments:
                    accumulated["arguments"] += arguments

        # Emit tool calls once the model signals completion
        if is_final and tool_call_accumulator:
            tool_calls = list(tool_call_accumulator.values())
        elif finish_reason == "tool_calls" and tool_call_accumulator:
            tool_calls = list(tool_call_accumulator.values())

        return StreamChunk(
            content=content,
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            is_final=is_final,
            usage=usage,
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
            raise ProviderAuthError(
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

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenAI models.

        Queries the OpenAI API to get available models, filtered to chat-capable models.

        Returns:
            List of available models with metadata

        Raises:
            ProviderError: If request fails
        """
        try:
            response = await self.client.models.list()
            # Filter to GPT models and format consistently
            models = []
            for model in response.data:
                model_id = model.id
                # Filter to chat-capable GPT models
                if any(
                    prefix in model_id for prefix in ["gpt-4", "gpt-3.5", "o1", "o3", "chatgpt"]
                ):
                    models.append(
                        {
                            "id": model_id,
                            "name": model_id,
                            "owned_by": model.owned_by,
                            "created": model.created,
                        }
                    )
            # Sort by name for consistent output
            models.sort(key=lambda x: x["id"])
            return models
        except Exception as e:
            raise ProviderError(
                message=f"Failed to list models: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.close()
