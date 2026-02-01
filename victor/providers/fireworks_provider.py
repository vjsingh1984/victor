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

"""Fireworks AI provider for high-performance model inference.

Fireworks AI provides fast inference with an OpenAI-compatible API
and $1 in free credits for new users.

Free Tier:
- $1 in free credits for experimentation
- Up to 6,000 requests/minute
- 2.5 billion tokens/day

Features:
- Very fast inference
- OpenAI-compatible API
- Native tool calling support
- Support for many open models

References:
- https://docs.fireworks.ai/
- https://docs.fireworks.ai/guides/function-calling
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

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

FIREWORKS_MODELS = {
    "accounts/fireworks/models/llama-v3p3-70b-instruct": {
        "description": "Llama 3.3 70B - High quality, fast",
        "context_window": 131072,
        "supports_tools": True,
    },
    "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct": {
        "description": "Qwen3 Coder 480B - Large code model",
        "context_window": 131072,
        "supports_tools": True,
    },
    "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507": {
        "description": "Qwen3 235B - Strong reasoning",
        "context_window": 131072,
        "supports_tools": True,
    },
    "accounts/fireworks/models/deepseek-v3p2": {
        "description": "DeepSeek V3.2 - Latest MoE",
        "context_window": 131072,
        "supports_tools": True,
    },
    "accounts/fireworks/models/deepseek-r1-0528": {
        "description": "DeepSeek R1 - Reasoning model",
        "context_window": 131072,
        "supports_tools": True,
    },
    "accounts/fireworks/models/llama4-maverick-instruct-basic": {
        "description": "Llama 4 Maverick - Latest Llama",
        "context_window": 131072,
        "supports_tools": True,
    },
}


class FireworksProvider(BaseProvider, HTTPErrorHandlerMixin):
    """Provider for Fireworks AI API (OpenAI-compatible).

    Features:
    - $1 free credits for new accounts
    - Very fast inference
    - Native tool calling support
    - Many open-source models
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        """Initialize Fireworks provider.

        Args:
            api_key: Fireworks API key (or set FIREWORKS_API_KEY env var)
            base_url: API endpoint
            timeout: Request timeout
            **kwargs: Additional configuration
        """
        # Resolve API key using centralized helper
        self._api_key = self._resolve_api_key(api_key, "fireworks")

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
        return "fireworks"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
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
        """Send chat completion request to Fireworks."""
        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, False, **kwargs
            )

            response = await self._execute_with_circuit_breaker(
                self.client.post, "/chat/completions", json=payload
            )
            response.raise_for_status()

            return self._parse_response(response.json(), model)

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Fireworks request timed out after {self.timeout}s",
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
        """Stream chat completion from Fireworks."""
        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, True, **kwargs
            )

            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                accumulated_tool_calls: list[dict[str, Any]] = []

                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        yield StreamChunk(
                            content="",
                            tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                            stop_reason="stop",
                            is_final=True,
                        )
                        break

                    try:
                        chunk_data = json.loads(data_str)
                        yield self._parse_stream_chunk(chunk_data, accumulated_tool_calls)
                    except json.JSONDecodeError:
                        pass

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message="Fireworks stream timed out",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, self.name)
        except Exception as e:
            raise self._handle_error(e, self.name)

    def _build_request_payload(
        self, messages, model, temperature, max_tokens, tools, stream, **kwargs
    ) -> dict[str, Any]:
        formatted_messages = []
        for msg in messages:
            formatted_msg = {"role": msg.role, "content": msg.content}
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

        payload = {
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
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="",
                role="assistant",
                model=model,
                raw_response=result,
                metadata=None,
                stop_reason=None,
                tool_calls=None,
                usage=None,
            )

        choice = choices[0]
        message = choice.get("message", {})
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        usage = None
        if usage_data := result.get("usage"):
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return CompletionResponse(
            content=message.get("content", "") or "",
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
            metadata=None,
        )

    def _normalize_tool_calls(self, tool_calls) -> Optional[list[dict[str, Any]]]:
        if not tool_calls:
            return None
        normalized = []
        for call in tool_calls:
            if "function" in call:
                func = call["function"]
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                normalized.append(
                    {
                        "id": call.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": args,
                    }
                )
        return normalized if normalized else None

    def _parse_stream_chunk(self, chunk_data, accumulated_tool_calls) -> StreamChunk:
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamChunk(content="", is_final=False)

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({"id": "", "name": "", "arguments": ""})
            if "id" in tc_delta:
                accumulated_tool_calls[idx]["id"] = tc_delta["id"]
            if "function" in tc_delta:
                func = tc_delta["function"]
                if "name" in func:
                    accumulated_tool_calls[idx]["name"] = func["name"]
                if "arguments" in func:
                    accumulated_tool_calls[idx]["arguments"] += func["arguments"]

        final_tool_calls = None
        if finish_reason in ("tool_calls", "stop") and accumulated_tool_calls:
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args = tc.get("arguments", "{}")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        args = {}
                    final_tool_calls.append(
                        {
                            "id": tc.get("id", ""),
                            "name": tc["name"],
                            "arguments": args,
                        }
                    )

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
        )

    async def close(self) -> None:
        await self.client.aclose()
