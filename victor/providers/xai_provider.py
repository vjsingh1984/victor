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

"""xAI Grok provider implementation.

xAI provides an OpenAI-compatible API with support for:
- Chat completions with streaming
- Native function/tool calling
- Multiple Grok models (grok-beta, grok-2, grok-code-fast-1, etc.)

References:
- https://docs.x.ai/docs/api-reference
- https://docs.x.ai/docs/guides/function-calling
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
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.openai_compat import convert_tools_to_openai_format

logger = logging.getLogger(__name__)

# Default xAI API endpoint
DEFAULT_BASE_URL = "https://api.x.ai/v1"

# Available xAI Grok models
# Reference: https://docs.x.ai/docs/models
XAI_MODELS = {
    "grok-2": {
        "description": "Grok-2 flagship model",
        "context_window": 131072,  # 128K tokens
        "max_output": 4096,
        "supports_tools": True,
    },
    "grok-2-mini": {
        "description": "Grok-2 mini - faster, more efficient",
        "context_window": 131072,  # 128K tokens
        "max_output": 4096,
        "supports_tools": True,
    },
    "grok-3": {
        "description": "Grok-3 next-gen model",
        "context_window": 131072,  # 128K tokens
        "max_output": 16384,
        "supports_tools": True,
    },
    "grok-3-mini": {
        "description": "Grok-3 mini - fast reasoning",
        "context_window": 131072,  # 128K tokens
        "max_output": 16384,
        "supports_tools": True,
    },
    "grok-beta": {
        "description": "Grok beta (legacy)",
        "context_window": 131072,  # 128K tokens
        "max_output": 4096,
        "supports_tools": True,
    },
}


class XAIProvider(BaseProvider):
    """Provider for xAI Grok models (OpenAI-compatible API).

    Features:
    - Native tool calling support
    - Streaming with tool call accumulation
    - OpenAI-compatible format
    """

    # Reasonable timeout for cloud API
    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize xAI provider.

        Args:
            api_key: xAI API key (or set XAI_API_KEY env var, or use keyring)
            base_url: xAI API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        # Get API key from parameter, environment, or keyring
        self._api_key = api_key or os.environ.get("XAI_API_KEY", "")
        if not self._api_key:
            try:
                from victor.config.api_keys import get_api_key

                # Try "xai" first, then "grok" alias
                self._api_key = get_api_key("xai") or get_api_key("grok") or ""
            except ImportError:
                pass
        if not self._api_key:
            logger.warning(
                "xAI API key not provided. Set XAI_API_KEY environment variable, "
                "use 'victor keys --set xai --keyring', or pass api_key parameter."
            )

        super().__init__(
            api_key=self._api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, **kwargs
        )
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "xai"

    def supports_tools(self) -> bool:
        """xAI Grok supports function calling."""
        return True

    def supports_streaming(self) -> bool:
        """xAI supports streaming."""
        return True

    def get_context_window(self, model: str) -> int:
        """Get context window size for a model.

        Args:
            model: Model name (e.g., "grok-2", "grok-3-mini")

        Returns:
            Context window size in tokens (default: 131072 for unknown models)
        """
        # Try exact match first
        if model in XAI_MODELS:
            return XAI_MODELS[model]["context_window"]

        # Try prefix match (e.g., "grok-2-1212" matches "grok-2")
        for model_prefix, info in XAI_MODELS.items():
            if model.startswith(model_prefix):
                return info["context_window"]

        # Default to 128K for unknown Grok models
        return 131072

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str = "grok-2-1212",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request to xAI.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "grok-2-1212", "grok-code-fast-1")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional xAI parameters

        Returns:
            CompletionResponse with generated content

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

            num_tools = len(tools) if tools else 0
            logger.debug(
                f"xAI chat request: model={model}, msgs={len(messages)}, tools={num_tools}"
            )

            # Make API call with circuit breaker protection
            response = await self._execute_with_circuit_breaker(
                self.client.post, "/chat/completions", json=payload
            )
            response.raise_for_status()

            result = response.json()
            return self._parse_response(result, model)

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"xAI request timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"xAI API error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str = "grok-2-1212",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from xAI with tool call accumulation.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional parameters

        Yields:
            StreamChunk with incremental content

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
                f"xAI streaming request: model={model}, msgs={len(messages)}, tools={num_tools}"
            )

            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                accumulated_tool_calls: List[Dict[str, Any]] = []
                has_sent_final = False

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # OpenAI SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            # Only yield final chunk if we haven't already sent one
                            # OR if there are accumulated tool calls that haven't been emitted
                            if not has_sent_final or accumulated_tool_calls:
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
                            if chunk:
                                if chunk.is_final:
                                    has_sent_final = True
                                yield chunk
                        except json.JSONDecodeError:
                            logger.warning(f"xAI JSON decode error on line: {line[:100]}")

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"xAI stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"xAI streaming error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tools to xAI format (OpenAI-compatible)."""
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
        """Build request payload for xAI's OpenAI-compatible API.

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
            # Handle tool results (tool response messages)
            if msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            formatted_messages.append(formatted_msg)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = self._convert_tools(tools)
            payload["tool_choice"] = "auto"

        # Merge additional options (excluding internal ones)
        internal_keys = {"api_key"}
        for key, value in kwargs.items():
            if key not in internal_keys and value is not None:
                payload[key] = value

        return payload

    def _normalize_tool_calls(
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Normalize tool calls from OpenAI format.

        Converts:
        {'id': '...', 'type': 'function', 'function': {'name': 'tool_name', 'arguments': '...'}}

        To:
        {'id': '...', 'name': 'tool_name', 'arguments': {...}}

        Args:
            tool_calls: Raw tool calls from API

        Returns:
            Normalized tool calls with parsed arguments
        """
        if not tool_calls:
            return None

        normalized = []
        for call in tool_calls:
            if isinstance(call, dict) and "function" in call:
                function = call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", "{}")

                # Parse arguments if string
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                if name:
                    normalized.append(
                        {
                            "id": call.get("id"),
                            "name": name,
                            "arguments": arguments,
                        }
                    )
            elif isinstance(call, dict) and "name" in call:
                # Already normalized
                normalized.append(call)

        return normalized if normalized else None

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse xAI API response.

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

        # Normalize tool calls (parses arguments to dict)
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        # Parse usage
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
        )

    def _parse_stream_chunk(
        self, chunk_data: Dict[str, Any], accumulated_tool_calls: List[Dict[str, Any]]
    ) -> Optional[StreamChunk]:
        """Parse streaming chunk from xAI with tool call accumulation.

        Args:
            chunk_data: Raw chunk data
            accumulated_tool_calls: List to accumulate tool call deltas

        Returns:
            StreamChunk or None
        """
        choices = chunk_data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        # Handle tool call deltas (OpenAI streaming format)
        tool_call_deltas = delta.get("tool_calls", [])
        for tc_delta in tool_call_deltas:
            idx = tc_delta.get("index", 0)
            # Extend accumulated list if needed
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({"id": "", "name": "", "arguments": ""})

            # Accumulate tool call ID
            if "id" in tc_delta:
                accumulated_tool_calls[idx]["id"] = tc_delta["id"]

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
                    final_tool_calls.append(
                        {
                            "id": tc.get("id"),
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

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> ProviderError:
        """Handle HTTP errors from xAI API.

        Args:
            error: HTTP error

        Raises:
            ProviderError: Converted error
        """
        status_code = error.response.status_code
        # Safely get error message - streaming responses may not have .text available
        try:
            error_msg = error.response.text
        except httpx.ResponseNotRead:
            error_msg = f"HTTP {status_code} error (response body not available)"

        if status_code == 401:
            raise ProviderAuthError(
                message=f"Authentication failed: {error_msg}",
                provider=self.name,
                status_code=status_code,
                raw_error=error,
            )
        elif status_code == 429:
            raise ProviderRateLimitError(
                message=f"Rate limit exceeded: {error_msg}",
                provider=self.name,
                status_code=status_code,
                raw_error=error,
            )
        else:
            raise ProviderError(
                message=f"xAI API error ({status_code}): {error_msg}",
                provider=self.name,
                status_code=status_code,
                raw_error=error,
            )

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available xAI models.

        Returns:
            List of available models with metadata

        Raises:
            ProviderError: If request fails
        """
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            result = response.json()
            return result.get("data", [])
        except Exception as e:
            raise ProviderError(
                message=f"xAI failed to list models: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
