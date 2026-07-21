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

"""Moonshot AI provider for Kimi K2 and K3 models.

Moonshot AI provides an OpenAI-compatible API with special support for:
- Kimi K3 flagship (2.8T MoE, 1M context, always-on thinking, native vision)
- Kimi K2 Thinking models with reasoning traces (256k context)
- Native tool calling support
- Streaming with reasoning_content field

Endpoint routing:
- kimi-k3*  -> https://api.moonshot.ai/v1  (international platform)
- kimi-k2*  -> https://api.moonshot.cn/v1  (default; .cn platform)
An explicit base_url passed at construction pins ALL models to that endpoint.

References:
- https://platform.kimi.ai/docs/guide/kimi-k3-quickstart
- https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model
- https://github.com/MoonshotAI/Kimi-K2
"""

from victor.core.json_utils import json_loads
from json import JSONDecodeError
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
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)
from victor.providers.logging import ProviderLogger
from victor.providers.usage_parsing import parse_usage_dict

# Default Moonshot API endpoint (K2 models, .cn platform)
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"

# Kimi K3 is served from the international .ai platform endpoint.
KIMI_K3_BASE_URL = "https://api.moonshot.ai/v1"

# K3 thinking is always on; effort is controlled via the top-level
# reasoning_effort request field. Server default is "max" when omitted —
# note there is NO "medium".
KIMI_K3_REASONING_EFFORTS = frozenset({"low", "high", "max"})

# Available Kimi K3 models
KIMI_K3_MODELS = {
    "kimi-k3": {
        "description": "Kimi K3 - 2.8T MoE flagship, 1M context, always-on thinking, native vision",
        "context_window": 1048576,
        "supports_thinking": True,
    },
}

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


class MoonshotProvider(BaseProvider):
    """Provider for Moonshot AI's Kimi models (OpenAI-compatible API).

    Features:
    - Native tool calling support
    - Reasoning/thinking trace extraction
    - Kimi K2 (256k context, .cn endpoint) and Kimi K3 (1M context,
      .ai endpoint — routed automatically per model)
    - K3 reasoning_effort passthrough (low/high/max; server default max)
    - Streaming support with reasoning_content
    """

    # Cloud provider timeout (shorter than local)
    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize Moonshot provider.

        Args:
            api_key: Moonshot API key (or set MOONSHOT_API_KEY env var)
            base_url: API endpoint (default: https://api.moonshot.cn/v1 for K2;
                kimi-k3* requests are routed to https://api.moonshot.ai/v1
                automatically unless an explicit base_url is passed, which
                always wins for every model)
            timeout: Request timeout (default: 120s)
            max_retries: Maximum retry attempts
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # An explicit, non-default base_url pins ALL models to that endpoint.
        self._base_url_pinned = base_url != DEFAULT_BASE_URL
        # Initialize structured logger
        self._provider_logger = ProviderLogger("moonshot", __name__)

        # Resolve API key using unified resolver
        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key("moonshot", explicit_key=api_key)

        # Log API key resolution
        self._provider_logger.log_api_key_resolution(key_result)

        if key_result.key is None:
            # Raise detailed error with actionable suggestions
            raise APIKeyNotFoundError(
                provider="moonshot",
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )

        self._api_key = key_result.key

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="kimi-k2",  # Will be set on chat()
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={
                "base_url": base_url,
                "timeout": timeout,
                "max_retries": max_retries,
                **kwargs,
            },
        )

        super().__init__(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Use httpx for consistent behavior with other providers
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        # Per-endpoint clients created on demand for model→endpoint routing
        # (kimi-k3 lives on the .ai platform while K2 stays on .cn).
        self._endpoint_clients: Dict[str, httpx.AsyncClient] = {}
        self._client_timeout = timeout

    @classmethod
    def resolve_base_url_for_model(cls, model: str) -> str:
        """Return the platform endpoint that serves ``model``."""
        if model.startswith("kimi-k3"):
            return KIMI_K3_BASE_URL
        return DEFAULT_BASE_URL

    def _client_for_model(self, model: str) -> httpx.AsyncClient:
        """Return the HTTP client for the endpoint serving ``model``.

        An explicit base_url passed at construction pins all models to the
        primary client; otherwise kimi-k3* requests route to the .ai endpoint.
        """
        if self._base_url_pinned:
            return self.client
        target = self.resolve_base_url_for_model(model)
        if target == str(self.client.base_url).rstrip("/"):
            return self.client
        client = self._endpoint_clients.get(target)
        if client is None:
            client = httpx.AsyncClient(
                base_url=target,
                timeout=httpx.Timeout(self._client_timeout),
                headers=dict(self.client.headers),
            )
            self._endpoint_clients[target] = client
        return client

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
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
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
            ProviderAuthError: If authentication fails
            ProviderRateLimitError: If rate limit is exceeded
            ProviderError: For other errors
        """
        # Use structured logging context manager
        with self._provider_logger.log_api_call(
            endpoint="/chat/completions",
            model=model,
            operation="chat",
            num_messages=len(messages),
            has_tools=tools is not None,
        ) as log_success:
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
                    self._client_for_model(model).post, "/chat/completions", json=payload
                )
                response.raise_for_status()

                result = response.json()
                parsed = self._parse_response(result, model)

                # Log success with usage info
                tokens = parsed.usage.get("total_tokens") if parsed.usage else None
                log_success(tokens=tokens)

                return parsed

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

            async with self._client_for_model(model).stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()

                accumulated_content = ""
                accumulated_reasoning = ""
                accumulated_tool_calls: List[Dict[str, Any]] = []

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
                            chunk_data = json_loads(data_str)
                            chunk = self._parse_stream_chunk(
                                chunk_data,
                                accumulated_tool_calls,
                                accumulated_reasoning,
                            )
                            if chunk.content:
                                accumulated_content += chunk.content
                            # Track reasoning content
                            if chunk.metadata and "reasoning_content" in chunk.metadata:
                                accumulated_reasoning = chunk.metadata["reasoning_content"]
                            yield chunk

                        except JSONDecodeError:
                            pass  # Skip invalid JSON chunks

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Moonshot stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass
            raise ProviderError(
                message=f"Moonshot streaming HTTP error {e.response.status_code}: {error_body}",
                provider=self.name,
                status_code=e.response.status_code,
                raw_error=e,
            ) from e
        except Exception as e:
            raise ProviderError(
                message=f"Moonshot stream error: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

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
            formatted_msg: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            # Handle tool results
            if msg.role == "tool" and hasattr(msg, "tool_call_id"):
                formatted_msg["tool_call_id"] = msg.tool_call_id
            formatted_messages.append(formatted_msg)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # K3 thinking effort: top-level reasoning_effort ∈ {low, high, max}.
        # Omitted when unset — the server default is "max" ("medium" does not
        # exist, so validate instead of passing through silently).
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            effort = str(reasoning_effort).lower()
            if effort not in KIMI_K3_REASONING_EFFORTS:
                raise ValueError(
                    f"Invalid reasoning_effort {reasoning_effort!r}: must be one of "
                    f"{sorted(KIMI_K3_REASONING_EFFORTS)} (omit for the server default, 'max')"
                )
            payload["reasoning_effort"] = effort

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
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
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
                        arguments = json_loads(arguments)
                    except JSONDecodeError:
                        arguments = {}

                if name:
                    normalized.append({"name": name, "arguments": arguments})
            elif isinstance(call, dict) and "name" in call:
                normalized.append(call)

        return normalized if normalized else None

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
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

        # Parse usage stats
        # Parse usage — routed through sandhi's single-sourced parser (also recovers
        # prompt_tokens_details.cached_tokens); native dict is the fallback.
        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = parse_usage_dict("openai", usage_data) or {
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
        chunk_data: Dict[str, Any],
        accumulated_tool_calls: List[Dict[str, Any]],
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
                        parsed_args = json_loads(args) if isinstance(args, str) else args
                    except JSONDecodeError:
                        parsed_args = {}
                    final_tool_calls.append({"name": tc["name"], "arguments": parsed_args})

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
            metadata=metadata,
        )

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Kimi models (K3 + K2).

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
            for model_id, model_info in {**KIMI_K3_MODELS, **KIMI_K2_MODELS}.items()
        ]

    async def close(self) -> None:
        """Close HTTP clients (primary + any per-endpoint clients)."""
        await self.client.aclose()
        for client in self._endpoint_clients.values():
            await client.aclose()
        self._endpoint_clients.clear()
