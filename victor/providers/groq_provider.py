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

"""Groq Cloud API provider for ultra-fast LLM inference.

Groq provides OpenAI-compatible API with extremely fast inference using
custom LPU (Language Processing Unit) hardware.

Features:
- Ultra-fast inference (100+ tokens/sec)
- OpenAI-compatible API
- Tool/function calling support
- 128K+ context windows
- Free developer tier available

Supported Models:
- llama-3.3-70b-versatile: Best quality, 128K context
- llama-3.1-8b-instant: Fast inference, 128K context
- qwen/qwen3-32b: Qwen3 model with tool support (preview)
- moonshotai/kimi-k2-instruct-0905: Kimi K2 with 256K context (preview)

References:
- https://console.groq.com/docs/overview
- https://console.groq.com/docs/models
- https://console.groq.com/docs/tool-use
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
from victor.providers.payload_limiter import (
    ProviderPayloadLimiter,
    TruncationStrategy,
)

logger = logging.getLogger(__name__)

# Groq OpenAI-compatible API endpoint
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

# Available Groq models
GROQ_MODELS = {
    # Production models
    "llama-3.3-70b-versatile": {
        "description": "Llama 3.3 70B - Best quality with tool support",
        "context_window": 131072,  # 128K
        "max_output": 32768,
        "supports_tools": True,
        "supports_parallel_tools": True,
    },
    "llama-3.1-8b-instant": {
        "description": "Llama 3.1 8B - Fast inference with tool support",
        "context_window": 131072,  # 128K
        "max_output": 8192,
        "supports_tools": True,
        "supports_parallel_tools": True,
    },
    "openai/gpt-oss-120b": {
        "description": "GPT-OSS 120B - Large open-source GPT model",
        "context_window": 131072,
        "max_output": 16384,
        "supports_tools": True,
        "supports_parallel_tools": False,
    },
    "openai/gpt-oss-20b": {
        "description": "GPT-OSS 20B - Efficient open-source GPT model",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "supports_parallel_tools": False,
    },
    # Preview models (may be discontinued)
    "qwen/qwen3-32b": {
        "description": "Qwen3 32B - Strong reasoning with tool support (preview)",
        "context_window": 131072,
        "max_output": 16384,
        "supports_tools": True,
        "supports_parallel_tools": True,
        "preview": True,
    },
    "moonshotai/kimi-k2-instruct-0905": {
        "description": "Kimi K2 - Extended 256K context (preview)",
        "context_window": 262144,  # 256K
        "max_output": 32768,
        "supports_tools": True,
        "supports_parallel_tools": True,
        "preview": True,
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "description": "Llama 4 Scout 17B - Latest Llama preview",
        "context_window": 131072,
        "max_output": 16384,
        "supports_tools": True,
        "supports_parallel_tools": True,
        "preview": True,
    },
}


class GroqProvider(BaseProvider):
    """Provider for Groq Cloud API (OpenAI-compatible).

    Features:
    - Ultra-fast inference using Groq LPU hardware
    - Native tool calling support
    - 128K+ context windows
    - Streaming support
    - Free developer tier
    """

    # Cloud provider timeout (Groq is very fast)
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        """Initialize Groq provider.

        Args:
            api_key: Groq API key (or set GROQ_API_KEY or GROQCLOUD_API_KEY env var)
            base_url: API endpoint (default: https://api.groq.com/openai/v1)
            timeout: Request timeout (default: 60s - Groq is fast)
            **kwargs: Additional configuration
        """
        # Get API key from parameter or environment
        # Support both GROQ_API_KEY and GROQCLOUD_API_KEY for flexibility
        self._api_key = (
            api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("GROQCLOUD_API_KEY", "")
        )
        if not self._api_key:
            logger.warning(
                "Groq API key not provided. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )

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

        # Initialize payload limiter for Groq's strict ~4MB limit
        # Groq uses LPU hardware which has strict payload constraints
        self._payload_limiter = ProviderPayloadLimiter(
            provider_name="groq",
            max_payload_bytes=4 * 1024 * 1024,  # 4MB
            default_strategy=TruncationStrategy.TRUNCATE_TOOL_RESULTS,
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "groq"

    def supports_tools(self) -> bool:
        """Groq supports native tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Groq supports streaming."""
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
        """Send chat completion request to Groq.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "llama-3.3-70b-versatile")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional options

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Check payload size before building request
            ok, warning = self._payload_limiter.check_limit(messages, tools)
            if warning:
                logger.warning(warning)

            # Truncate if payload exceeds Groq's limit
            if not ok:
                truncation_result = self._payload_limiter.truncate_if_needed(messages, tools)
                messages = truncation_result.messages
                tools = truncation_result.tools
                if truncation_result.warning:
                    logger.warning(truncation_result.warning)
                if truncation_result.truncated:
                    logger.info(
                        f"Truncated payload: removed {truncation_result.messages_removed} messages, "
                        f"saved {truncation_result.bytes_saved:,} bytes"
                    )

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
                message=f"Groq request timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass
            raise ProviderError(
                message=f"Groq HTTP error {e.response.status_code}: {error_body}",
                provider=self.name,
                status_code=e.response.status_code,
                raw_error=e,
            ) from e
        except Exception as e:
            raise ProviderError(
                message=f"Groq unexpected error: {str(e)}",
                provider=self.name,
                raw_error=e,
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
        """Stream chat completion from Groq.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional options

        Yields:
            StreamChunk with incremental content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Check payload size before building request
            ok, warning = self._payload_limiter.check_limit(messages, tools)
            if warning:
                logger.warning(warning)

            # Truncate if payload exceeds Groq's limit
            if not ok:
                truncation_result = self._payload_limiter.truncate_if_needed(messages, tools)
                messages = truncation_result.messages
                tools = truncation_result.tools
                if truncation_result.warning:
                    logger.warning(truncation_result.warning)
                if truncation_result.truncated:
                    logger.info(
                        f"Truncated payload: removed {truncation_result.messages_removed} messages, "
                        f"saved {truncation_result.bytes_saved:,} bytes"
                    )

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
                f"Groq streaming request: model={model}, msgs={len(messages)}, tools={num_tools}"
            )

            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                accumulated_content = ""
                accumulated_tool_calls: List[Dict[str, Any]] = []

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # OpenAI SSE format: "data: {...}" or "data: [DONE]"
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
                            if chunk.content:
                                accumulated_content += chunk.content
                            yield chunk

                        except json.JSONDecodeError:
                            logger.warning(f"Groq JSON decode error on line: {line[:100]}")

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Groq stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass
            raise ProviderError(
                message=f"Groq streaming HTTP error {e.response.status_code}: {error_body}",
                provider=self.name,
                status_code=e.response.status_code,
                raw_error=e,
            ) from e
        except Exception as e:
            raise ProviderError(
                message=f"Groq stream error: {str(e)}",
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
        """Build request payload for Groq's OpenAI-compatible API.

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
            "max_tokens": max_tokens,
            "temperature": temperature,
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
            elif isinstance(call, dict) and "name" in call:
                normalized.append(call)

        return normalized if normalized else None

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse Groq API response.

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

        # Parse usage stats
        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }
            # Groq may include queue_time and other timing info (as float seconds)
            # Store as milliseconds (int) for consistency
            if "queue_time" in usage_data:
                queue_time = usage_data["queue_time"]
                usage["queue_time_ms"] = (
                    int(queue_time * 1000) if isinstance(queue_time, (int, float)) else 0
                )

        # Include Groq-specific metadata
        metadata = {}
        if "x_groq" in result:
            metadata["x_groq"] = result["x_groq"]

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
    ) -> StreamChunk:
        """Parse streaming chunk from Groq.

        Args:
            chunk_data: Raw chunk data
            accumulated_tool_calls: List to accumulate tool call deltas

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

        # Handle tool call deltas
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

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Groq models.

        Returns:
            List of available models with metadata
        """
        # Try to fetch from API first
        try:
            response = await self.client.get("/models")
            if response.status_code == 200:
                result = response.json()
                return result.get("data", [])
        except Exception as e:
            logger.debug(f"Failed to fetch models from Groq API: {e}")

        # Return static list as fallback
        return [
            {
                "id": model_id,
                "object": "model",
                **model_info,
            }
            for model_id, model_info in GROQ_MODELS.items()
        ]

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
