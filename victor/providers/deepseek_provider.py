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

"""DeepSeek API provider for DeepSeek-V3 models.

DeepSeek provides OpenAI-compatible API with special support for:
- deepseek-chat: Non-thinking mode (DeepSeek-V3.2) with function calling
- deepseek-reasoner: Thinking mode (DeepSeek-V3.2) with Chain of Thought + function calling
- 128K context window
- Very competitive pricing (~10-30x cheaper than OpenAI)

References:
- https://api-docs.deepseek.com/
- https://api-docs.deepseek.com/guides/function_calling
- https://api-docs.deepseek.com/guides/reasoning_model
"""

import json
import logging
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
    get_api_key_with_resolution,
)
from victor.providers.logging import ProviderLogger

logger = logging.getLogger(__name__)

# Default DeepSeek API endpoint
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

# Available DeepSeek models
DEEPSEEK_MODELS = {
    "deepseek-chat": {
        "description": "DeepSeek-V3.2 non-thinking mode with function calling",
        "context_window": 131072,  # 128K
        "max_output": 8192,
        "supports_tools": True,
        "supports_thinking": False,
    },
    "deepseek-reasoner": {
        "description": "DeepSeek-V3.2 thinking mode with CoT and function calling",
        "context_window": 131072,  # 128K
        "max_output": 65536,  # 64K (includes reasoning)
        "supports_tools": True,
        "supports_thinking": True,
    },
}


class DeepSeekProvider(BaseProvider):
    """Provider for DeepSeek API (OpenAI-compatible).

    Features:
    - Native tool calling support (both deepseek-chat and deepseek-reasoner)
    - Reasoning/thinking trace extraction (deepseek-reasoner)
    - 128k context window
    - Streaming support
    - Very cost-effective pricing
    """

    # Cloud provider timeout
    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize DeepSeek provider.

        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            base_url: API endpoint (default: https://api.deepseek.com/v1)
            timeout: Request timeout (default: 120s)
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("deepseek", __name__)

        # Resolve API key using unified resolver
        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key("deepseek", explicit_key=api_key)

        # Log API key resolution
        self._provider_logger.log_api_key_resolution(key_result)

        if key_result.key is None:
            # Raise detailed error with actionable suggestions
            raise APIKeyNotFoundError(
                provider="deepseek",
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )

        self._api_key = key_result.key

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="deepseek-chat",  # Will be set on chat()
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={"base_url": base_url, "timeout": timeout, **kwargs},
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

    @property
    def name(self) -> str:
        """Provider name."""
        return "deepseek"

    def supports_tools(self) -> bool:
        """DeepSeek supports native tool calling (deepseek-chat only)."""
        return True

    def supports_streaming(self) -> bool:
        """DeepSeek supports streaming."""
        return True

    def _model_supports_tools(self, model: str) -> bool:
        """Check if specific model supports tool calling.

        Args:
            model: Model name

        Returns:
            True if model supports tools
        """
        model_lower = model.lower()
        # deepseek-reasoner does NOT support function calling
        if "reasoner" in model_lower or "r1" in model_lower:
            return False
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
        """Send chat completion request to DeepSeek.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "deepseek-chat", "deepseek-reasoner")
            temperature: Sampling temperature (ignored for deepseek-reasoner)
            max_tokens: Maximum tokens to generate
            tools: Available tools (only for deepseek-chat)
            **kwargs: Additional options

        Returns:
            CompletionResponse with generated content and optional reasoning

        Raises:
            ProviderError: If request fails
        """
        # Use structured logging context manager
        async with self._provider_logger.log_api_call(
            endpoint="/chat/completions",
            model=model,
            operation="chat",
            num_messages=len(messages),
            has_tools=tools is not None,
        ):
            try:
                # Filter tools if model doesn't support them
                effective_tools = tools if self._model_supports_tools(model) else None
                if tools and not self._model_supports_tools(model):
                    logger.debug(f"Model {model} doesn't support tools, ignoring tools parameter")

                payload = self._build_request_payload(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=effective_tools,
                    stream=False,
                    **kwargs,
                )

                response = await self._execute_with_circuit_breaker(
                    self.client.post, "/chat/completions", json=payload
                )
                response.raise_for_status()

                result = response.json()
                parsed = self._parse_response(result, model)

                # Log success with usage info
                tokens = parsed.usage.get("total_tokens") if parsed.usage else None
                self._provider_logger._log_api_call_success(
                    call_id=f"chat_{model}_{id(payload)}",
                    endpoint="/chat/completions",
                    model=model,
                    start_time=0,  # Will be set by context manager
                    tokens=tokens,
                )

                return parsed

            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(
                    message=f"DeepSeek request timed out after {self.timeout}s",
                    provider=self.name,
                ) from e
            except httpx.HTTPStatusError as e:
                error_body = ""
                try:
                    error_body = e.response.text[:500]
                except Exception:
                    pass
                raise ProviderError(
                    message=f"DeepSeek HTTP error {e.response.status_code}: {error_body}",
                    provider=self.name,
                    status_code=e.response.status_code,
                    raw_error=e,
                ) from e
            except Exception as e:
                raise ProviderError(
                    message=f"DeepSeek unexpected error: {str(e)}",
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
        """Stream chat completion from DeepSeek.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools (only for deepseek-chat)
            **kwargs: Additional options

        Yields:
            StreamChunk with incremental content and reasoning

        Raises:
            ProviderError: If request fails
        """
        try:
            # Filter tools if model doesn't support them
            effective_tools = tools if self._model_supports_tools(model) else None

            payload = self._build_request_payload(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=effective_tools,
                stream=True,
                **kwargs,
            )

            num_tools = len(effective_tools) if effective_tools else 0
            logger.debug(
                f"DeepSeek streaming request: model={model}, msgs={len(messages)}, tools={num_tools}"
            )

            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
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
                            chunk_data = json.loads(data_str)
                            chunk = self._parse_stream_chunk(
                                chunk_data, accumulated_tool_calls, accumulated_reasoning
                            )
                            if chunk.content:
                                accumulated_content += chunk.content
                            # Track reasoning content
                            if chunk.metadata and "reasoning_content" in chunk.metadata:
                                accumulated_reasoning = chunk.metadata["reasoning_content"]
                            yield chunk

                        except json.JSONDecodeError:
                            logger.warning(f"DeepSeek JSON decode error on line: {line[:100]}")

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"DeepSeek stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass
            raise ProviderError(
                message=f"DeepSeek streaming HTTP error {e.response.status_code}: {error_body}",
                provider=self.name,
                status_code=e.response.status_code,
                raw_error=e,
            ) from e
        except Exception as e:
            raise ProviderError(
                message=f"DeepSeek stream error: {str(e)}",
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
        """Build request payload for DeepSeek's OpenAI-compatible API.

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
            "stream": stream,
        }

        # Temperature is ignored for deepseek-reasoner but we still send it
        # (API ignores it silently for reasoner)
        is_reasoner = "reasoner" in model.lower() or "r1" in model.lower()
        if not is_reasoner:
            payload["temperature"] = temperature

        # Add tools if provided (only for deepseek-chat)
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
                    normalized.append({"name": name, "arguments": arguments})
            elif isinstance(call, dict) and "name" in call:
                normalized.append(call)

        return normalized if normalized else None

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse DeepSeek API response.

        Handles special reasoning_content field for deepseek-reasoner.

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

        # Extract reasoning content (deepseek-reasoner specific)
        reasoning_content = message.get("reasoning_content")
        metadata = {}
        if reasoning_content:
            metadata["reasoning_content"] = reasoning_content
            logger.debug(f"DeepSeek: Extracted reasoning content ({len(reasoning_content)} chars)")

        # Parse usage stats
        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }
            # DeepSeek may include reasoning_tokens separately
            if "reasoning_tokens" in usage_data:
                usage["reasoning_tokens"] = usage_data["reasoning_tokens"]

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
        """Parse streaming chunk from DeepSeek.

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

        # Extract reasoning content delta (deepseek-reasoner)
        reasoning_delta = delta.get("reasoning_content", "")
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
                        parsed_args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
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
        """List available DeepSeek models.

        Returns:
            List of available models with metadata
        """
        # Return static list - DeepSeek API supports /models endpoint
        # but we can also return our known models
        return [
            {
                "id": model_id,
                "object": "model",
                **model_info,
            }
            for model_id, model_info in DEEPSEEK_MODELS.items()
        ]

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
