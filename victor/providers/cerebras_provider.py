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

"""Cerebras Inference API provider for ultra-fast LLM inference.

Cerebras provides extremely fast inference using their Wafer Scale Engine (WSE)
hardware with a generous free tier.

Free Tier:
- 30-10 requests/min (varies by model)
- 60,000-150,000 tokens/min
- Access to Llama, Qwen, and GPT-OSS models

Features:
- Fastest inference available (1000+ tokens/sec)
- OpenAI-compatible API
- Tool calling support
- Free tier with no credit card

References:
- https://inference-docs.cerebras.ai/
- https://inference-docs.cerebras.ai/api-reference
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

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"

CEREBRAS_MODELS = {
    "llama3.3-70b": {
        "description": "Llama 3.3 70B - High quality, very fast",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "rate_limit": "30 req/min, 60K tokens/min",
    },
    "llama-3.3-70b": {
        "description": "Llama 3.3 70B (alias)",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
    },
    "llama3.1-8b": {
        "description": "Llama 3.1 8B - Fast, efficient",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "rate_limit": "30 req/min, 150K tokens/min",
    },
    "qwen-2.5-32b": {
        "description": "Qwen 2.5 32B - Strong reasoning",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
    },
    "qwen-2.5-coder-32b": {
        "description": "Qwen 2.5 Coder 32B - Code specialized",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
    },
}


class CerebrasProvider(BaseProvider):
    """Provider for Cerebras Inference API (OpenAI-compatible).

    Features:
    - Fastest inference available (WSE hardware)
    - Generous free tier
    - Native tool calling support
    - OpenAI-compatible API
    """

    DEFAULT_TIMEOUT = 60  # Cerebras is very fast

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        """Initialize Cerebras provider.

        Args:
            api_key: Cerebras API key (or set CEREBRAS_API_KEY env var)
            base_url: API endpoint
            timeout: Request timeout
            **kwargs: Additional configuration
        """
        # Try provided key, then env var, then keyring/api_keys.yaml
        self._api_key = api_key or os.environ.get("CEREBRAS_API_KEY", "")
        if not self._api_key:
            try:
                from victor.config.api_keys import get_api_key

                self._api_key = get_api_key("cerebras") or ""
            except ImportError:
                pass
        if not self._api_key:
            logger.warning(
                "Cerebras API key not provided. Set CEREBRAS_API_KEY environment variable "
                "or add to keyring with: victor keys set cerebras"
            )

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
        return "cerebras"

    def supports_tools(self) -> bool:
        return True

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
        """Send chat completion request to Cerebras."""
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
                message=f"Cerebras request timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = e.response.text[:500] if e.response.text else ""
            raise ProviderError(
                message=f"Cerebras HTTP error {e.response.status_code}: {error_body}",
                provider=self.name,
                status_code=e.response.status_code,
            ) from e

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
        """Stream chat completion from Cerebras."""
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

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message="Cerebras stream timed out",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"Cerebras streaming error {e.response.status_code}",
                provider=self.name,
                status_code=e.response.status_code,
            ) from e

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
                content="", role="assistant", model=model, raw_response=result
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
            # Cerebras provides timing info
            if "time_info" in result:
                time_info = result["time_info"]
                usage["total_time_ms"] = int(time_info.get("total_time", 0) * 1000)

        return CompletionResponse(
            content=message.get("content", "") or "",
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
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

    async def close(self) -> None:
        await self.client.aclose()
