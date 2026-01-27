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

"""OpenRouter API provider - unified gateway to multiple LLM providers.

OpenRouter provides a single API endpoint to access models from OpenAI,
Anthropic, Google, Meta, Mistral, and many others with unified pricing.

Free Tier:
- 20 requests/minute, 50 requests/day
- Up to 1,000 requests/day with $10 topup
- Access to free models (Gemma, Llama, Mistral variants)

Features:
- Single API for 100+ models
- Automatic fallback between providers
- Cost tracking and rate limiting
- OpenAI-compatible API

References:
- https://openrouter.ai/docs
- https://openrouter.ai/docs/models
"""

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.error_handler import HTTPErrorHandlerMixin

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Free and popular models on OpenRouter
OPENROUTER_MODELS = {
    # Free models
    "google/gemma-2-9b-it:free": {
        "description": "Gemma 2 9B - Free tier",
        "context_window": 8192,
        "supports_tools": False,
        "free": True,
    },
    "meta-llama/llama-3.2-3b-instruct:free": {
        "description": "Llama 3.2 3B - Free tier",
        "context_window": 131072,
        "supports_tools": False,
        "free": True,
    },
    "mistralai/mistral-7b-instruct:free": {
        "description": "Mistral 7B - Free tier",
        "context_window": 32768,
        "supports_tools": False,
        "free": True,
    },
    "qwen/qwen-2-7b-instruct:free": {
        "description": "Qwen 2 7B - Free tier",
        "context_window": 32768,
        "supports_tools": False,
        "free": True,
    },
    # Paid models with tool support
    "anthropic/claude-3.5-sonnet": {
        "description": "Claude 3.5 Sonnet via OpenRouter",
        "context_window": 200000,
        "supports_tools": True,
    },
    "openai/gpt-4o": {
        "description": "GPT-4o via OpenRouter",
        "context_window": 128000,
        "supports_tools": True,
    },
    "google/gemini-2.0-flash-exp:free": {
        "description": "Gemini 2.0 Flash - Free experimental",
        "context_window": 1000000,
        "supports_tools": True,
        "free": True,
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "description": "Llama 3.3 70B",
        "context_window": 131072,
        "supports_tools": True,
    },
    "deepseek/deepseek-chat": {
        "description": "DeepSeek V3",
        "context_window": 131072,
        "supports_tools": True,
    },
}


class OpenRouterProvider(BaseProvider, HTTPErrorHandlerMixin):
    """Provider for OpenRouter API - unified gateway to multiple LLMs.

    Features:
    - Single API for 100+ models
    - Free tier with daily limits
    - Automatic fallback between providers
    - OpenAI-compatible API
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            base_url: API endpoint
            timeout: Request timeout
            site_url: Your site URL (for rankings)
            site_name: Your site/app name
            **kwargs: Additional configuration
        """
        # Resolve API key using centralized helper
        self._api_key = self._resolve_api_key(api_key, "openrouter")

        super().__init__(base_url=base_url, timeout=timeout, **kwargs)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        # Optional headers for OpenRouter rankings
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name

        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )

    @property
    def name(self) -> str:
        return "openrouter"

    def supports_tools(self) -> bool:
        return True  # Depends on model

    def supports_streaming(self) -> bool:
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
        """Send chat completion request via OpenRouter."""
        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, False, **kwargs
            )

            response = await self._execute_with_circuit_breaker(
                self.client.post, "/chat/completions", json=payload
            )
            response.raise_for_status()

            return self._parse_response(response.json(), model)

        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, self.name)
        except httpx.TimeoutException as e:
            raise self._handle_error(e, self.name)
        except Exception as e:
            raise self._handle_error(e, self.name)

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:  # type: ignore[override]
        """Stream chat completion from OpenRouter."""
        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, True, **kwargs
            )

            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                accumulated_tool_calls: List[Dict[str, Any]] = []

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

        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e, self.name)
        except httpx.TimeoutException as e:
            raise self._handle_error(e, self.name)
        except Exception as e:
            raise self._handle_error(e, self.name)

    def _build_request_payload(
        self, messages, model, temperature, max_tokens, tools, stream, **kwargs
    ) -> Dict[str, Any]:
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

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
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

    def _normalize_tool_calls(self, tool_calls) -> Optional[List[Dict[str, Any]]]:
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

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from OpenRouter."""
        try:
            response = await self.client.get("/models")
            if response.status_code == 200:
                result = response.json()
                return result.get("data", [])
        except Exception as e:
            logger.debug(f"Failed to fetch models from OpenRouter: {e}")

        return [
            {"id": model_id, **model_info} for model_id, model_info in OPENROUTER_MODELS.items()
        ]

    async def close(self) -> None:
        await self.client.aclose()
