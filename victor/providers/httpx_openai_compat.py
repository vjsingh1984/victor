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
    ProviderConnectionError,
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

    async def _send_chat_completion_request(self, payload: Dict[str, Any]) -> httpx.Response:
        """Send a non-streaming chat-completions request with status validation.

        This helper ensures transient HTTP failures are raised inside the shared
        retry wrapper rather than after it returns.
        """
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response

    async def _open_chat_completion_stream(
        self, payload: Dict[str, Any]
    ) -> "tuple[Any, httpx.Response]":
        """Open a streaming chat-completions response with status validation.

        Returns ``(stream_context, response)``. The caller (``stream()``) owns closing the
        context in its own ``finally`` so enter and exit are lexically paired in the same
        task — avoiding the "exit cancel scope in a different task" / "ignored GeneratorExit"
        errors that the previous response-attribute readback hack allowed.

        For non-200 responses we eagerly read the body so the raised ``HTTPStatusError``
        carries the provider's error payload and can be retried/classified consistently by
        the shared resilience layer.
        """
        stream_context = self.client.stream("POST", "/chat/completions", json=payload)
        response = await stream_context.__aenter__()

        # Once __aenter__ has opened the httpx stream, ANY failure before we hand the
        # context back to stream() (a non-200 body read, or task cancellation) must close it
        # here — otherwise the open response is orphaned and finalized off-task by GC, raising
        # "async generator ignored GeneratorExit" / "exit cancel scope in a different task".
        try:
            if response.status_code != 200:
                await response.aread()
                response.raise_for_status()
        except BaseException:
            await stream_context.__aexit__(None, None, None)
            raise
        return stream_context, response

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

        # reasoning_effort is only valid for reasoning models. The kwarg
        # passthrough above forwards it for any OpenAI-compatible endpoint that
        # accepts it (subclasses opt in via supports_reasoning_effort); strip it
        # otherwise so it never reaches a model/endpoint that would reject it.
        if "reasoning_effort" in payload and not self.supports_reasoning_effort(model):
            payload.pop("reasoning_effort", None)

        # Log detailed message structure for debugging tool pairing issues
        tool_messages = [
            (i, m) for i, m in enumerate(formatted) if m.get("role") in ("tool", "assistant")
        ]
        if tool_messages:
            for i, msg in tool_messages:
                role = msg.get("role")
                if role == "assistant" and "tool_calls" in msg:
                    tc_ids = [tc.get("id", "") for tc in msg["tool_calls"]]
                    self._provider_logger.logger.debug(
                        "%s payload: message[%d] role=assistant tool_calls=%s",
                        self.name,
                        i,
                        tc_ids,
                    )
                elif role == "tool":
                    self._provider_logger.logger.debug(
                        "%s payload: message[%d] role=tool tool_call_id=%s name=%s",
                        self.name,
                        i,
                        msg.get("tool_call_id"),
                        msg.get("name", ""),
                    )

        # System prompt visibility: log the full payload shape (tool names + system
        # message) at INFO so the actual offer the model receives is auditable in the
        # session log. This is the single source of truth for "what the LLM sees"
        # before the API call — critical for diagnosing tool-description/registry
        # drift (e.g. a stale tool like `code_search` being advertised).
        offered_tool_names = []
        if effective_tools:
            for _tool in effective_tools:
                # ToolDefinition may expose name directly or via .function.name
                _tname = getattr(_tool, "name", None)
                if _tname is None:
                    _tname = getattr(getattr(_tool, "function", None), "name", None)
                if _tname:
                    offered_tool_names.append(_tname)

        _sys_preview = ""
        for _msg in formatted:
            if _msg.get("role") == "system":
                _content = _msg.get("content", "") or ""
                _sys_preview = _content[:160].replace("\n", " ")
                break

        self._provider_logger.logger.info(
            "%s payload: model=%s messages=%d tools=%d stream=%s "
            "tool_names=%s system_prompt_preview=%r",
            self.name,
            model,
            len(formatted),
            len(effective_tools),
            stream,
            offered_tool_names,
            _sys_preview,
        )
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
                    self._send_chat_completion_request, payload
                )
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
            stream_context, response = await self._execute_with_circuit_breaker(
                self._open_chat_completion_stream, payload
            )
            try:
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
                            "%s JSON decode error on SSE line: %s",
                            self.name,
                            line[:100],
                        )
            finally:
                # Close the stream context here — lexically paired with the __aenter__ in
                # _open_chat_completion_stream, in whatever task drives this generator.
                await stream_context.__aexit__(None, None, None)

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"{self.name} stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise handle_httpx_status_error(e, self.name) from e
        except httpx.ReadError as e:
            # Mid-stream connection drop — surface a clear message rather than an empty error_msg.
            # The classify_error() path sets error_msg=str(e) which is "" for bare ReadError().
            error_detail = str(e) or "connection dropped mid-stream (no additional detail)"
            logger.warning(
                "%s stream ReadError: %s — surfacing as ProviderConnectionError "
                "(turn retried upstream)",
                self.name,
                error_detail,
            )
            raise ProviderConnectionError(
                message=f"{self.name} stream disconnected mid-response: {error_detail}",
                provider=self.name,
                raw_error=e,
            ) from e
        except (ProviderTimeoutError, ProviderError):
            raise
        except Exception as e:
            raise self.classify_error(e) from e

    async def close(self) -> None:
        """Close the underlying httpx client (idempotent)."""
        client = getattr(self, "client", None)
        if client is None or getattr(client, "is_closed", False):
            return
        try:
            await client.aclose()
        except Exception:
            logger.debug("Provider httpx client close failed (already closed?)", exc_info=True)
