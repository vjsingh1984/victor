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

"""z.ai (ZhipuAI/智谱AI) provider implementation.

z.ai provides an OpenAI-compatible API with support for:
- Chat completions with streaming
- Native function/tool calling
- Thinking mode (reasoning_content) for GLM-4.6/4.5/4.5-Air
- Multiple GLM models (glm-4.7, glm-4.6, glm-4.5, glm-4.5-air)

References:
- https://docs.z.ai/
- https://docs.z.ai/guides/develop/openai/python
- https://docs.z.ai/guides/develop/http/introduction
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

# Available z.ai GLM models
# Reference: https://docs.z.ai/ and https://z.ai/blog/glm-4.6
ZAI_MODELS = {
    "glm-4.7": {
        "description": "GLM-4.7 - Latest flagship model",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_thinking": False,
    },
    "glm-4.6": {
        "description": "GLM-4.6 - Advanced agentic, reasoning, and coding",
        "context_window": 128000,
        "max_output": 8192,
        "supports_tools": True,
        "supports_thinking": True,
    },
    "glm-4.5": {
        "description": "GLM-4.5 - 355B total, 32B active parameters",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_thinking": True,
    },
    "glm-4.5-air": {
        "description": "GLM-4.5-Air - Lightweight variant",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_thinking": True,
    },
}


class ZAIProvider(BaseProvider):
    """Provider for z.ai GLM models (OpenAI-compatible API).

    Features:
    - Native tool calling support (OpenAI-compatible format)
    - Streaming with tool call accumulation
    - Thinking mode with reasoning_content
    - OpenAI-compatible API format

    Example:
        from victor.providers.zai_provider import ZAIProvider

        provider = ZAIProvider(
            api_key="your-zai-api-key",
            base_url="https://api.z.ai/api/paas/v4/"
        )

        response = await provider.chat(
            messages=[Message(role="user", content="Hello!")],
            model="glm-4.7"
        )
    """

    # Reasonable timeout for cloud API
    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.z.ai/api/paas/v4/",
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize z.ai provider.

        Args:
            api_key: z.ai API key (or set ZAI_API_KEY env var, or use keyring)
            base_url: z.ai API base URL (default: https://api.z.ai/api/paas/v4/)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        # Resolution order: parameter → env var → keyring → warning
        resolved_key = api_key or os.environ.get("ZAI_API_KEY", "")
        if not resolved_key:
            try:
                from victor.config.api_keys import get_api_key
                # Try multiple aliases: zai, zhipuai, zhipu
                resolved_key = get_api_key("zai") or get_api_key("zhipuai") or get_api_key("zhipu") or ""
            except ImportError:
                pass

        if not resolved_key:
            logger.warning(
                "ZhipuAI API key not provided. Set ZAI_API_KEY environment variable, "
                "use 'victor keys --set zai --keyring', or pass api_key parameter."
            )

        super().__init__(
            api_key=resolved_key, base_url=base_url, timeout=timeout, max_retries=max_retries, **kwargs
        )
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "zai"

    def supports_tools(self) -> bool:
        """z.ai GLM models support function calling."""
        return True

    def supports_streaming(self) -> bool:
        """z.ai supports streaming."""
        return True

    def get_context_window(self, model: str) -> int:
        """Get context window size for a model.

        Args:
            model: Model name (e.g., "glm-4.7", "glm-4.6")

        Returns:
            Context window size in tokens (default: 128000 for unknown models)
        """
        # Try exact match first
        if model in ZAI_MODELS:
            return ZAI_MODELS[model]["context_window"]

        # Try prefix match (e.g., "glm-4.7-custom" matches "glm-4.7")
        for model_prefix, info in ZAI_MODELS.items():
            if model.startswith(model_prefix):
                return info["context_window"]

        # Default to 128K for unknown GLM models
        return 128000

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str = "glm-4.7",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        thinking: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request to z.ai.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "glm-4.7", "glm-4.6")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            thinking: Enable thinking mode for reasoning (GLM-4.6/4.5/4.5-Air)
            **kwargs: Additional z.ai parameters

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
                thinking=thinking,
                **kwargs,
            )

            num_tools = len(tools) if tools else 0
            logger.debug(
                f"z.ai chat request: model={model}, msgs={len(messages)}, tools={num_tools}, thinking={thinking}"
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
                message=f"z.ai request timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"z.ai API error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str = "glm-4.7",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from z.ai with tool call accumulation.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            thinking: Enable thinking mode for reasoning (GLM-4.6/4.5/4.5-Air)
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
                thinking=thinking,
                **kwargs,
            )

            num_tools = len(tools) if tools else 0
            logger.debug(
                f"z.ai streaming request: model={model}, msgs={len(messages)}, tools={num_tools}, thinking={thinking}"
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
                            logger.warning(f"z.ai JSON decode error on line: {line[:100]}")

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"z.ai stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"z.ai streaming error: {str(e)}",
                provider=self.name,
                raw_error=e,
            )

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tools to z.ai format (OpenAI-compatible)."""
        return convert_tools_to_openai_format(tools)

    def _build_request_payload(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        stream: bool,
        thinking: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request payload for z.ai's OpenAI-compatible API.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Available tools
            stream: Whether to stream response
            thinking: Whether to enable thinking mode
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

        # Add thinking mode if requested (for GLM-4.6/4.5/4.5-Air)
        # Note: z.ai uses extra_body with thinking parameter
        # Since we're using httpx directly, we'll pass it as a top-level parameter
        if thinking:
            payload["thinking"] = {"type": "enabled"}

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
        """Parse z.ai API response.

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

        # Extract reasoning_content if present (thinking mode)
        metadata = None
        if "reasoning_content" in message:
            metadata = {"reasoning_content": message.get("reasoning_content")}

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
        self, chunk_data: Dict[str, Any], accumulated_tool_calls: List[Dict[str, Any]]
    ) -> Optional[StreamChunk]:
        """Parse streaming chunk from z.ai with tool call accumulation.

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

        # Extract reasoning_content if present (thinking mode)
        metadata = None
        if "reasoning_content" in delta:
            metadata = {"reasoning_content": delta.get("reasoning_content")}

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

        # Parse usage from final chunk
        usage = None
        if finish_reason and "usage" in chunk_data:
            usage_data = chunk_data.get("usage")
            if usage_data:
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

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> ProviderError:
        """Handle HTTP errors from z.ai API.

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
                message=f"z.ai API error ({status_code}): {error_msg}",
                provider=self.name,
                status_code=status_code,
                raw_error=error,
            )

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available z.ai GLM models.

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
                message=f"z.ai failed to list models: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
