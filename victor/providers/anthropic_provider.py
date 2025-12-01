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

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage

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
from victor.providers.openai_compat import convert_tools_to_anthropic_format


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
        super().__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, **kwargs
        )
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
                    conversation_messages.append(
                        {
                            "role": msg.role,
                            "content": msg.content,
                        }
                    )

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

            # Make API call with circuit breaker protection
            response: AnthropicMessage = await self._execute_with_circuit_breaker(
                self.client.messages.create, **request_params
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
        """Stream chat completion from Anthropic with tool-use support."""
        try:
            # Separate system messages
            system_message = None
            conversation_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    conversation_messages.append(
                        {
                            "role": msg.role,
                            "content": msg.content,
                        }
                    )

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

            tool_calls: Dict[str, Dict[str, Any]] = {}
            block_index_to_id: Dict[int, str] = {}

            async with self.client.messages.stream(**request_params) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", "") == "tool_use":
                            tc_id = getattr(block, "id", None) or f"tool_{len(tool_calls) + 1}"
                            tool_calls[tc_id] = {
                                "id": tc_id,
                                "name": getattr(block, "name", ""),
                                "arguments": getattr(block, "input", {}) or {},
                            }
                            block_index = getattr(
                                event, "index", getattr(block, "index", len(block_index_to_id))
                            )
                            block_index_to_id[block_index] = tc_id

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", "")
                        block_index = getattr(
                            event, "index", getattr(event, "content_block_index", None)
                        )
                        if delta_type == "text_delta" and hasattr(delta, "text"):
                            yield StreamChunk(content=delta.text or "", is_final=False)
                        elif delta_type in {"input_json_delta", "input_delta"}:
                            tc_id = block_index_to_id.get(block_index)
                            if tc_id:
                                partial = (
                                    getattr(delta, "partial_json", None)
                                    or getattr(delta, "text", "")
                                    or ""
                                )
                                existing_args = tool_calls[tc_id].get("arguments", "")
                                if existing_args in ({}, None):
                                    tool_calls[tc_id]["arguments"] = partial
                                elif isinstance(existing_args, str):
                                    tool_calls[tc_id]["arguments"] = existing_args + partial
                                else:
                                    tool_calls[tc_id]["arguments"] = (
                                        json.dumps(existing_args) + partial
                                    )

                    elif event_type == "content_block_stop":
                        block_index = getattr(
                            event, "index", getattr(event, "content_block_index", None)
                        )
                        tc_id = block_index_to_id.get(block_index)
                        if tc_id and "arguments" in tool_calls.get(tc_id, {}):
                            tool_calls[tc_id]["arguments"] = self._parse_json_arguments(
                                tool_calls[tc_id].get("arguments")
                            )

                    elif event_type == "message_stop":
                        for tc in tool_calls.values():
                            tc["arguments"] = self._parse_json_arguments(tc.get("arguments"))

                        yield StreamChunk(
                            content="",
                            tool_calls=list(tool_calls.values()) or None,
                            stop_reason="stop",
                            is_final=True,
                        )

        except Exception as e:
            raise self._handle_error(e)

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tools to Anthropic format."""
        return convert_tools_to_anthropic_format(tools)

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
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

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

    @staticmethod
    def _parse_json_arguments(raw_args: Any) -> Any:
        """Best-effort parse of tool-use arguments."""
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except Exception:
                return raw_args
        return raw_args

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
