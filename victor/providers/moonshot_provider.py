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

"""Moonshot AI provider for Kimi K2 models.

Moonshot AI provides OpenAI-compatible API with special support for:
- Kimi K2 Thinking models with reasoning traces
- 256k context window
- Native tool calling support
- Streaming with reasoning_content field

References:
- https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model
- https://github.com/MoonshotAI/Kimi-K2
"""

import json
import logging
from typing import Any, Optional
from collections.abc import AsyncIterator

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.error_handler import HTTPErrorHandlerMixin

logger = logging.getLogger(__name__)

# Default Moonshot API endpoint
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"

# Available Kimi K2 models
KIMI_K2_MODELS = {
    "kimi-k2-thinking": {
        "description": "Kimi K2 Thinking - reasoning model with extended thinking",
        "context_window": 262144,
        "supports_thinking": True,
    },
    "kimi-k2-thinking-turbo": {
        "description": "Kimi K2 Thinking Turbo - faster reasoning model",
        "context_window": 262144,
        "supports_thinking": True,
    },
    "kimi-k2-instruct": {
        "description": "Kimi K2 Instruct - instruction-following model",
        "context_window": 262144,
        "supports_thinking": False,
    },
}


class MoonshotProvider(BaseProvider, HTTPErrorHandlerMixin):
    """Provider for Moonshot AI's Kimi K2 models (OpenAI-compatible API).

    Features:
    - Native tool calling support
    - Reasoning/thinking trace extraction
    - 256k context window
    - Streaming support with reasoning_content
    """

    # Cloud provider timeout (shorter than local)
    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        """Initialize Moonshot provider.

        Args:
            api_key: Moonshot API key (or set MOONSHOT_API_KEY env var)
            base_url: API endpoint (default: https://api.moonshot.cn/v1)
            timeout: Request timeout (default: 120s)
            **kwargs: Additional configuration
        """
        # Resolve API key using centralized helper
        self._api_key = self._resolve_api_key(api_key, "moonshot")

        super().__init__(base_url=base_url, timeout=timeout, **kwargs)

        # Use httpx for consistent behavior with other providers
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "moonshot"

    def supports_tools(self) -> bool:
        """Moonshot supports native tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Moonshot supports streaming."""
        return True

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request to Moonshot.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "kimi-k2-thinking")
            temperature: Sampling temperature (1.0 recommended for reasoning)
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional options

        Returns:
            CompletionResponse with generated content and optional reasoning

        Raises:
            ProviderError: If request fails
        """
        try:
            payload = self._build_request_payload(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=False,
                **kwargs,
            )

            response = await self._execute_with_circuit_breaker(
                self.client.post, "/chat/completions", json=payload
            )
            response.raise_for_status()

            result = response.json()
            return self._parse_response(result, model)

        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, self.name)
        except httpx.TimeoutException as e:
            raise self._handle_error(e, self.name)
        except Exception as e:
            raise self._handle_error(e, self.name)

    async def stream(  # type: ignore[override,misc]
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from Moonshot.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional options

        Yields:
            StreamChunk with incremental content and reasoning

        Raises:
            ProviderError: If request fails
        """
        try:
            payload = self._build_request_payload(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
                **kwargs,
            )

            num_tools = len(tools) if tools else 0
            logger.debug(
                f"Moonshot streaming request: model={model}, msgs={len(messages)}, tools={num_tools}"
            )

            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                accumulated_content = ""
                accumulated_reasoning = ""
                accumulated_tool_calls: list[dict[str, Any]] = []

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # OpenAI SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            # Final chunk - include accumulated reasoning if any
                            yield StreamChunk(
                                content="",
                                tool_calls=(
                                    accumulated_tool_calls if accumulated_tool_calls else None
                                ),
                                stop_reason="stop",
                                is_final=True,
                                metadata=(
                                    {"reasoning_content": accumulated_reasoning}
                                    if accumulated_reasoning
                                    else None
                                ),
                            )
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            chunk = self._parse_stream_chunk(
                                chunk_data, accumulated_tool_calls, accumulated_reasoning
                            )
                            if chunk.content:
                                accumulated_content += chunk.content
                            # Track reasoning content
                            if chunk.metadata and "reasoning_content" in chunk.metadata:
                                accumulated_reasoning = chunk.metadata["reasoning_content"]
                            yield chunk

                        except json.JSONDecodeError:
                            logger.warning(f"Moonshot JSON decode error on line: {line[:100]}")

        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, self.name)
        except httpx.TimeoutException as e:
            raise self._handle_error(e, self.name)
        except Exception as e:
            raise self._handle_error(e, self.name)

    def _build_request_payload(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[list[ToolDefinition]],
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build request payload for Moonshot's OpenAI-compatible API.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Available tools
            stream: Whether to stream response
            **kwargs: Additional options

        Returns:
            Request payload dictionary
        """
        # Build messages in OpenAI format
        formatted_messages = []
        for msg in messages:
            formatted_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            # Handle tool results
            if msg.role == "tool" and hasattr(msg, "tool_call_id"):
                formatted_msg["tool_call_id"] = msg.tool_call_id
            formatted_messages.append(formatted_msg)

        payload: dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = [
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
            payload["tool_choice"] = "auto"

        # Merge additional options
        for key, value in kwargs.items():
            if key not in {"api_key"} and value is not None:
                payload[key] = value

        return payload

    def _normalize_tool_calls(
        self, tool_calls: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """Normalize tool calls from OpenAI format.

        Args:
            tool_calls: Raw tool calls from API

        Returns:
            Normalized tool calls
        """
        if not tool_calls:
            return None

        normalized = []
        for call in tool_calls:
            if isinstance(call, dict) and "function" in call:
                function = call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", "{}")

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                if name:
                    normalized.append({"name": name, "arguments": arguments})
            elif isinstance(call, dict) and "name" in call:
                normalized.append(call)

        return normalized if normalized else None

    def _parse_response(self, result: dict[str, Any], model: str) -> CompletionResponse:
        """Parse Moonshot API response.

        Handles special reasoning_content field for thinking models.

        Args:
            result: Raw API response
            model: Model name

        Returns:
            Normalized CompletionResponse
        """
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="",
                role="assistant",
                model=model,
                raw_response=result,
                tool_calls=None,
                stop_reason=None,
                usage=None,
                metadata=None,
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        # Extract reasoning/thinking content (Kimi K2 specific)
        reasoning_content = message.get("reasoning_content") or message.get("thinking")
        metadata = {}
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content
            logger.debug(f"Moonshot: Extracted reasoning content ({len(reasoning_content)} chars)")

        # Parse usage stats
        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
            metadata=metadata if metadata else None,
        )

    def _parse_stream_chunk(
        self,
        chunk_data: dict[str, Any],
        accumulated_tool_calls: list[dict[str, Any]],
        accumulated_reasoning: str,
    ) -> StreamChunk:
        """Parse streaming chunk from Moonshot.

        Args:
            chunk_data: Raw chunk data
            accumulated_tool_calls: List to accumulate tool call deltas
            accumulated_reasoning: Accumulated reasoning content

        Returns:
            Normalized StreamChunk
        """
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamChunk(content="", is_final=False)

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        # Extract reasoning content delta
        reasoning_delta = delta.get("reasoning_content") or delta.get("thinking", "")
        metadata = None
        if reasoning_delta:
            new_reasoning = accumulated_reasoning + reasoning_delta
            metadata = {"reasoning_content": new_reasoning}

        # Handle tool call deltas
        tool_call_deltas = delta.get("tool_calls", [])
        for tc_delta in tool_call_deltas:
            idx = tc_delta.get("index", 0)
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({"name": "", "arguments": ""})

            if "function" in tc_delta:
                func_delta = tc_delta["function"]
                if "name" in func_delta:
                    accumulated_tool_calls[idx]["name"] = func_delta["name"]
                if "arguments" in func_delta:
                    accumulated_tool_calls[idx]["arguments"] += func_delta["arguments"]

        # Finalize tool calls on stream end
        final_tool_calls = None
        if finish_reason == "tool_calls" or (finish_reason == "stop" and accumulated_tool_calls):
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args = tc.get("arguments", "{}")
                    try:
                        parsed_args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        parsed_args = {}
                    final_tool_calls.append({"name": tc["name"], "arguments": parsed_args})

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
            metadata=metadata,
        )

    async def list_models(self) -> list[dict[str, Any]]:
        """List available Kimi K2 models.

        Returns:
            List of available models with metadata
        """
        # Return static list since Moonshot doesn't have a models endpoint
        return [
            {
                "id": model_id,
                "object": "model",
                **model_info,
            }
            for model_id, model_info in KIMI_K2_MODELS.items()
        ]

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
