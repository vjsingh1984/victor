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

"""Together AI provider for open-source model inference.

Together AI provides high-performance inference for open-source models
with an OpenAI-compatible API and $25 free credits for new users.

Free Tier:
- $25 in free credits for new accounts
- Access to 100+ open-source models
- Native tool calling support on select models

Supported Models (with tool calling):
- meta-llama/Llama-3.3-70B-Instruct-Turbo
- openai/gpt-oss-120b
- deepseek-ai/DeepSeek-R1
- deepseek-ai/DeepSeek-V3.1
- moonshotai/Kimi-K2-Instruct-0905

References:
- https://docs.together.ai/
- https://docs.together.ai/docs/function-calling
"""

import json
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
from victor.providers.resolution import UnifiedApiKeyResolver, APIKeyNotFoundError
from victor.providers.logging import ProviderLogger

DEFAULT_BASE_URL = "https://api.together.xyz/v1"

TOGETHER_MODELS = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "description": "Llama 3.3 70B Turbo - Fast, high quality",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
    },
    "openai/gpt-oss-120b": {
        "description": "GPT-OSS 120B - OpenAI open-weight model",
        "context_window": 128000,
        "max_output": 16384,
        "supports_tools": True,
    },
    "deepseek-ai/DeepSeek-V3.1": {
        "description": "DeepSeek V3.1 - 671B MoE model",
        "context_window": 128000,
        "max_output": 16384,
        "supports_tools": True,
    },
    "deepseek-ai/DeepSeek-R1": {
        "description": "DeepSeek R1 - Reasoning model",
        "context_window": 163839,
        "max_output": 16384,
        "supports_tools": True,
    },
    "moonshotai/Kimi-K2-Instruct-0905": {
        "description": "Kimi K2 - 256K context, strong reasoning",
        "context_window": 262144,
        "max_output": 16384,
        "supports_tools": True,
    },
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "description": "Llama 4 Maverick - 1M context MoE",
        "context_window": 1048576,
        "max_output": 8192,
        "supports_tools": True,
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
        "description": "Llama 3.1 8B Turbo - Fast and efficient",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
    },
}


class TogetherProvider(BaseProvider):
    """Provider for Together AI API (OpenAI-compatible).

    Features:
    - $25 free credits for new accounts
    - 100+ open-source models
    - Native tool calling support
    - Fast inference
    - Streaming support
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize Together AI provider.

        Args:
            api_key: Together API key (or set TOGETHER_API_KEY env var)
            base_url: API endpoint
            timeout: Request timeout
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("together", __name__)

        # Resolve API key using unified resolver
        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key("together", explicit_key=api_key)

        # Log API key resolution
        self._provider_logger.log_api_key_resolution(key_result)

        if key_result.key is None:
            # Raise detailed error with actionable suggestions
            raise APIKeyNotFoundError(
                provider="together",
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )

        self._api_key = key_result.key

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="together",  # Will be set on chat()
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={"base_url": base_url, "timeout": timeout, **kwargs},
        )

        super().__init__(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            **kwargs,
        )

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
        return "together"

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
        """Send chat completion request to Together AI.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional Together AI parameters

        Returns:
            CompletionResponse with generated content

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
        ):
            try:
                payload = self._build_request_payload(
                    messages, model, temperature, max_tokens, tools, False, **kwargs
                )

                response = await self._execute_with_circuit_breaker(
                    self.client.post, "/chat/completions", json=payload
                )
                response.raise_for_status()

                parsed = self._parse_response(response.json(), model)

                # Log success with usage info
                tokens = parsed.usage.get("total_tokens") if parsed.usage else None
                self._provider_logger._log_api_call_success(
                    call_id=f"chat_{model}_{id(payload)}",
                    endpoint="/chat/completions",
                    model=model,
                    start_time=0,  # Set by context manager
                    tokens=tokens,
                )

                return parsed

            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(
                    message=f"Together AI request timed out after {self.timeout}s",
                    provider=self.name,
                ) from e
            except httpx.HTTPStatusError as e:
                error_body = e.response.text[:500] if e.response.text else ""
                # Convert to specific provider error types based on status code
                if e.response.status_code == 401:
                    raise ProviderAuthError(
                        message=f"Authentication failed: {error_body}",
                        provider=self.name,
                        status_code=e.response.status_code,
                    ) from e
                elif e.response.status_code == 429:
                    raise ProviderRateLimitError(
                        message=f"Rate limit exceeded: {error_body}",
                        provider=self.name,
                        status_code=e.response.status_code,
                    ) from e
                else:
                    raise ProviderError(
                        message=f"Together AI HTTP error {e.response.status_code}: {error_body}",
                        provider=self.name,
                        status_code=e.response.status_code,
                    ) from e
            except Exception as e:
                # Convert to specific provider error types based on error message
                # Skip if already a ProviderError to avoid double-wrapping
                if isinstance(e, ProviderError):
                    raise

                error_str = str(e).lower()
                if any(term in error_str for term in ["auth", "unauthorized", "invalid key", "401"]):
                    raise ProviderAuthError(
                        message=f"Authentication failed: {str(e)}",
                        provider=self.name,
                    ) from e
                elif any(term in error_str for term in ["rate limit", "429", "too many requests"]):
                    raise ProviderRateLimitError(
                        message=f"Rate limit exceeded: {str(e)}",
                        provider=self.name,
                        status_code=429,
                    ) from e
                else:
                    # Wrap generic errors in ProviderError
                    raise ProviderError(
                        message=f"Together AI error: {str(e)}",
                        provider=self.name,
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
        """Stream chat completion from Together AI."""
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
                message="Together AI stream timed out",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"Together AI streaming error {e.response.status_code}",
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
