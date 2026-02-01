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

"""Mistral AI API provider.

Mistral AI provides powerful open-weight and proprietary models with
excellent tool calling support and a generous free tier.

Free Tier (La Plateforme):
- 1 request/second
- 500,000 tokens/minute
- 1,000,000,000 tokens/month (per model)
- Requires phone verification

Supported Models:
- mistral-large-latest: Most capable, 128K context
- mistral-small-latest: Balanced performance
- codestral-latest: Optimized for code
- open-mistral-nemo: Open-weight, fast
- ministral-8b-latest: Smallest, fastest

References:
- https://docs.mistral.ai/
- https://docs.mistral.ai/capabilities/function_calling/
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
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.error_handler import HTTPErrorHandlerMixin

logger = logging.getLogger(__name__)

# Mistral API endpoint
DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

# Available Mistral models
MISTRAL_MODELS = {
    "mistral-large-latest": {
        "description": "Most capable Mistral model, 128K context",
        "context_window": 131072,
        "max_output": 32768,
        "supports_tools": True,
        "supports_parallel_tools": True,
    },
    "mistral-small-latest": {
        "description": "Balanced performance, good for most tasks",
        "context_window": 32768,
        "max_output": 8192,
        "supports_tools": True,
        "supports_parallel_tools": True,
    },
    "codestral-latest": {
        "description": "Optimized for code generation and understanding",
        "context_window": 32768,
        "max_output": 8192,
        "supports_tools": True,
        "supports_parallel_tools": True,
    },
    "open-mistral-nemo": {
        "description": "Open-weight 12B model, fast inference",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "supports_parallel_tools": True,
    },
    "ministral-8b-latest": {
        "description": "Smallest model, fastest inference",
        "context_window": 32768,
        "max_output": 8192,
        "supports_tools": True,
        "supports_parallel_tools": False,
    },
    "ministral-3b-latest": {
        "description": "Tiny model for simple tasks",
        "context_window": 32768,
        "max_output": 4096,
        "supports_tools": True,
        "supports_parallel_tools": False,
    },
}


class MistralProvider(BaseProvider, HTTPErrorHandlerMixin):
    """Provider for Mistral AI API.

    Features:
    - Native tool calling support
    - Generous free tier (500K tokens/min)
    - Strong coding capabilities
    - 128K context window
    - Streaming support
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        """Initialize Mistral provider.

        Args:
            api_key: Mistral API key (or set MISTRAL_API_KEY env var)
            base_url: API endpoint (default: https://api.mistral.ai/v1)
            timeout: Request timeout
            **kwargs: Additional configuration
        """
        # Resolve API key using centralized helper
        self._api_key = self._resolve_api_key(api_key, "mistral")

        super().__init__(base_url=base_url, timeout=timeout, **kwargs)

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
        return "mistral"

    def supports_tools(self) -> bool:
        """Mistral supports native tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Mistral supports streaming."""
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
        """Send chat completion request to Mistral.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "mistral-large-latest")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional options

        Returns:
            CompletionResponse with generated content
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

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Mistral request timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, self.name)
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
        """Stream chat completion from Mistral.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional options

        Yields:
            StreamChunk with incremental content
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

            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                accumulated_tool_calls: list[dict[str, Any]] = []

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            yield StreamChunk(
                                content="",
                                tool_calls=(
                                    accumulated_tool_calls if accumulated_tool_calls else None
                                ),
                                stop_reason="stop",
                                is_final=True,
                            )
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            chunk = self._parse_stream_chunk(chunk_data, accumulated_tool_calls)
                            yield chunk
                        except json.JSONDecodeError:
                            logger.warning(f"Mistral JSON decode error: {line[:100]}")

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Mistral stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, self.name)
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
        """Build request payload for Mistral API.

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
        formatted_messages = []
        for msg in messages:
            formatted_msg: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.role == "tool" and hasattr(msg, "tool_call_id"):
                formatted_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                formatted_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": (
                                json.dumps(tc.get("arguments", {}))
                                if isinstance(tc.get("arguments"), dict)
                                else tc.get("arguments", "{}")
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(formatted_msg)

        payload: dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

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

        return payload

    def _parse_response(self, result: dict[str, Any], model: str) -> CompletionResponse:
        """Parse Mistral API response."""
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="",
                role="assistant",
                model=model,
                stop_reason="stop",
                usage=None,
                raw_response=result,
                metadata=None,
                tool_calls=None,
            )

        choice = choices[0]
        message = choice.get("message", {})
        raw_content = message.get("content", "") or ""

        # Handle structured content (list of content blocks)
        if isinstance(raw_content, list):
            text_parts = []
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "".join(text_parts)
        else:
            content = raw_content

        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

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
            metadata=None,
        )

    def _normalize_tool_calls(
        self, tool_calls: Optional[list[dict[str, Any]]]
    ) -> Optional[list[dict[str, Any]]]:
        """Normalize tool calls from OpenAI format."""
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
                    normalized.append(
                        {
                            "id": call.get("id", ""),
                            "name": name,
                            "arguments": arguments,
                        }
                    )

        return normalized if normalized else None

    def _parse_stream_chunk(
        self,
        chunk_data: dict[str, Any],
        accumulated_tool_calls: list[dict[str, Any]],
    ) -> StreamChunk:
        """Parse streaming chunk from Mistral."""
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamChunk(content="", is_final=False)

        choice = choices[0]
        delta = choice.get("delta", {})
        raw_content = delta.get("content", "") or ""

        # Handle structured content (list of content blocks)
        if isinstance(raw_content, list):
            # Extract text from content blocks
            text_parts = []
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "".join(text_parts)
        else:
            content = raw_content

        finish_reason = choice.get("finish_reason")

        tool_call_deltas = delta.get("tool_calls", [])
        for tc_delta in tool_call_deltas:
            idx = tc_delta.get("index", 0)
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({"id": "", "name": "", "arguments": ""})

            if "id" in tc_delta:
                accumulated_tool_calls[idx]["id"] = tc_delta["id"]
            if "function" in tc_delta:
                func_delta = tc_delta["function"]
                if "name" in func_delta:
                    accumulated_tool_calls[idx]["name"] = func_delta["name"]
                if "arguments" in func_delta:
                    accumulated_tool_calls[idx]["arguments"] += func_delta["arguments"]

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
                    final_tool_calls.append(
                        {
                            "id": tc.get("id", ""),
                            "name": tc["name"],
                            "arguments": parsed_args,
                        }
                    )

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
        )

    async def list_models(self) -> list[dict[str, Any]]:
        """List available Mistral models."""
        try:
            response = await self.client.get("/models")
            if response.status_code == 200:
                result = response.json()
                return result.get("data", [])
        except Exception as e:
            logger.debug(f"Failed to fetch models from Mistral API: {e}")

        return [
            {"id": model_id, "object": "model", **model_info}
            for model_id, model_info in MISTRAL_MODELS.items()
        ]

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
