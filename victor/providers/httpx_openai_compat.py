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

"""Abstract base class for httpx-based OpenAI-compatible providers.

Providers that use the OpenAI /v1/chat/completions REST API via raw httpx
(Z.AI/GLM, xAI/Grok, DeepSeek) should extend ``HttpxOpenAICompatProvider``
instead of ``BaseProvider`` directly.

This base class consolidates ZAI's refined transport code:
- Complete message serialization including tool_calls and orphaned-message fixing
- SSE streaming with tool-call delta accumulation
- Detailed HTTP error mapping (JSON body parsing for 400 errors)
- Template Method hooks for provider-specific behaviour

Usage:

    class MyProvider(HttpxOpenAICompatProvider):
        @property
        def name(self) -> str:
            return "myprovider"

        def supports_tools(self) -> bool:
            return True

        def supports_streaming(self) -> bool:
            return True

        def _get_provider_params(self, model, temperature, max_tokens, **kwargs):
            # provider-specific max_tokens variant, model flags, etc.
            return {"temperature": temperature, "max_tokens": max_tokens}
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
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
from victor.providers.openai_compat import (
    accumulate_tool_call_delta,
    build_openai_messages,
    convert_tools_to_openai_format,
    handle_httpx_status_error,
    parse_openai_tool_calls,
)
from victor.providers.logging import ProviderLogger

logger = logging.getLogger(__name__)


class HttpxOpenAICompatProvider(BaseProvider):
    """Abstract base for httpx-based OpenAI-API-compatible providers.

    Subclasses must implement:
        - ``name`` (property)
        - ``supports_tools()``
        - ``supports_streaming()``

    Subclasses may override the Template Method hooks:
        - ``_clean_model_name(model)``       — strip provider suffixes before the API call
        - ``_get_provider_params(...)``       — temperature/max_tokens variants, extra flags
        - ``_extract_response_metadata(msg)`` — reasoning_content or other non-standard fields
        - ``_extract_stream_metadata(delta)`` — same for streaming deltas
    """

    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        provider_name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self._api_key = api_key
        self._provider_logger = ProviderLogger(provider_name or self.name, __name__)
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def supports_tools(self) -> bool: ...

    @abstractmethod
    def supports_streaming(self) -> bool: ...

    # ── Template Method hooks (override in subclasses) ────────────────────────

    def _clean_model_name(self, model: str) -> str:
        """Strip provider-specific suffixes before the model name is sent to the API.

        Example (ZAI): "glm-5.1:coding" → "glm-5.1"
        Default: return model unchanged.
        """
        return model

    def _get_provider_params(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return the provider-specific parameter block for the request payload.

        Override to handle model-specific variants:
        - DeepSeek: skip temperature for reasoner models
        - OpenAI O-series: use max_completion_tokens instead of max_tokens
        - ZAI: add thinking: {type: "enabled"} when thinking=True

        Default returns standard OpenAI parameters.
        """
        return {"temperature": temperature, "max_tokens": max_tokens}

    def _extract_response_metadata(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract provider-specific metadata from a non-streaming response message.

        Override to capture fields like ``reasoning_content`` (ZAI, DeepSeek).
        Default returns None.
        """
        return None

    def _extract_stream_metadata(self, delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract provider-specific metadata from a streaming delta.

        Override to capture ``reasoning_content`` in thinking-mode streams.
        Default returns None.
        """
        return None

    def _filter_tools_for_model(
        self,
        model: str,
        tools: Optional[List[ToolDefinition]],
    ) -> Optional[List[ToolDefinition]]:
        """Filter or suppress tools based on model capabilities.

        Override for models that don't support function calling in certain
        configurations (e.g., DeepSeek-reasoner, O-series without tools).
        Default passes tools through unchanged.
        """
        return tools

    # ── Shared implementation ─────────────────────────────────────────────────

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        return convert_tools_to_openai_format(tools)

    def _build_request_payload(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the full request payload for /v1/chat/completions.

        Uses ``build_openai_messages()`` for complete tool-call serialization and
        orphaned-message cleanup, then calls ``_get_provider_params()`` for the
        model-specific parameter block.
        """
        formatted = build_openai_messages(messages)

        if not formatted:
            raise ValueError(
                "Cannot send request with empty messages list. "
                "This usually indicates a conversation initialization issue."
            )

        provider_params = self._get_provider_params(model, temperature, max_tokens, **kwargs)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted,
            "stream": stream,
            **provider_params,
        }

        effective_tools = self._filter_tools_for_model(model, tools)
        if effective_tools:
            payload["tools"] = self._convert_tools(effective_tools)
            payload["tool_choice"] = "auto"

        # Pass through provider-specific kwargs (exclude internal keys)
        for key, value in kwargs.items():
            if key not in {"api_key"} and value is not None and key not in payload:
                payload[key] = value

        self._provider_logger.logger.debug(
            "%s payload: model=%s messages=%d tools=%s stream=%s",
            self.name,
            model,
            len(formatted),
            tools is not None,
            stream,
        )

        return payload

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse a non-streaming /v1/chat/completions response dict."""
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="", role="assistant", model=model, raw_response=result
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        tool_calls = parse_openai_tool_calls(message.get("tool_calls"))

        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        metadata = self._extract_response_metadata(message)

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
            metadata=metadata,
        )

    def _parse_stream_chunk(
        self,
        chunk_data: Dict[str, Any],
        accumulated_tool_calls: List[Dict[str, Any]],
    ) -> Optional[StreamChunk]:
        """Parse one SSE chunk, accumulating tool-call deltas."""
        choices = chunk_data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        metadata = self._extract_stream_metadata(delta)

        # Accumulate tool-call fragments
        accumulate_tool_call_delta(delta, accumulated_tool_calls)

        # Finalize tool calls when the model signals completion
        final_tool_calls: Optional[List[Dict[str, Any]]] = None
        if finish_reason in ("tool_calls", "stop") and accumulated_tool_calls:
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args_str = tc.get("arguments", "{}")
                    try:
                        parsed_args = (
                            json.loads(args_str) if isinstance(args_str, str) else args_str
                        )
                    except json.JSONDecodeError:
                        parsed_args = {}
                    final_tool_calls.append(
                        {
                            "id": tc.get("id"),
                            "name": tc["name"],
                            "arguments": parsed_args,
                        }
                    )

        # Parse usage from final chunk (when finish_reason is set)
        usage = None
        if finish_reason and "usage" in chunk_data:
            usage_data = chunk_data.get("usage") or {}
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
            metadata=metadata,
            usage=usage,
        )

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
        """Send a chat completion request via httpx."""
        model = self._clean_model_name(model)

        with self._provider_logger.log_api_call(
            endpoint="/chat/completions",
            model=model,
            operation="chat",
            num_messages=len(messages),
            has_tools=tools is not None,
        ) as log_success:
            try:
                payload = self._build_request_payload(
                    messages, model, temperature, max_tokens, tools, False, **kwargs
                )
                response = await self._execute_with_circuit_breaker(
                    self.client.post, "/chat/completions", json=payload
                )
                response.raise_for_status()
                result = response.json()
                parsed = self._parse_response(result, model)
                tokens = parsed.usage.get("total_tokens") if parsed.usage else None
                log_success(tokens=tokens)
                return parsed

            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(
                    message=f"{self.name} request timed out after {self.timeout}s",
                    provider=self.name,
                ) from e
            except httpx.HTTPStatusError as e:
                raise handle_httpx_status_error(e, self.name) from e
            except Exception as e:
                raise self.classify_error(e) from e

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
        """Stream a chat completion via httpx SSE."""
        model = self._clean_model_name(model)

        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, True, **kwargs
            )

            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    await response.aread()
                    mock_exc = httpx.HTTPStatusError(
                        message=response.text[:500],
                        request=response.request,
                        response=response,
                    )
                    raise handle_httpx_status_error(mock_exc, self.name)

                accumulated_tool_calls: List[Dict[str, Any]] = []
                has_sent_final = False

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        if not has_sent_final or accumulated_tool_calls:
                            yield StreamChunk(
                                content="",
                                tool_calls=(accumulated_tool_calls or None),
                                stop_reason="stop",
                                is_final=True,
                            )
                        break

                    try:
                        chunk_data = json.loads(data_str)
                        chunk = self._parse_stream_chunk(chunk_data, accumulated_tool_calls)
                        if chunk:
                            if chunk.is_final:
                                has_sent_final = True
                            yield chunk
                    except json.JSONDecodeError:
                        self._provider_logger.logger.warning(
                            "%s JSON decode error on SSE line: %s", self.name, line[:100]
                        )

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"{self.name} stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise handle_httpx_status_error(e, self.name) from e
        except (ProviderTimeoutError, ProviderError):
            raise
        except Exception as e:
            raise self.classify_error(e) from e

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self.client.aclose()
