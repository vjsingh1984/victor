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

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Free and popular models on OpenRouter
OPENROUTER_MODELS = {
    # Free models
    "meta-llama/llama-3.3-70b-instruct:free": {
        "description": "Llama 3.3 70B - Free tier",
        "context_window": 131072,
        "supports_tools": True,
        "free": True,
    },
    "meta-llama/llama-3.2-3b-instruct:free": {
        "description": "Llama 3.2 3B - Free tier",
        "context_window": 131072,
        "supports_tools": False,
        "free": True,
    },
    "google/gemini-2.5-flash:free": {
        "description": "Gemini 2.5 Flash - Free tier",
        "context_window": 1000000,
        "supports_tools": True,
        "free": True,
    },
    "mistralai/mistral-7b-instruct:free": {
        "description": "Mistral 7B - Free tier",
        "context_window": 32768,
        "supports_tools": False,
        "free": True,
    },
    # Paid models with tool support
    "anthropic/claude-sonnet-4.5": {
        "description": "Claude Sonnet 4.5 via OpenRouter",
        "context_window": 200000,
        "supports_tools": True,
    },
    "openai/gpt-4o": {
        "description": "GPT-4o via OpenRouter",
        "context_window": 128000,
        "supports_tools": True,
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
    "deepseek/deepseek-r1": {
        "description": "DeepSeek R1 - Reasoning model",
        "context_window": 131072,
        "supports_tools": True,
    },
}


class OpenRouterProvider(BaseProvider):
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
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            base_url: API endpoint
            timeout: Request timeout
            site_url: Your site URL (for rankings)
            site_name: Your site/app name
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        # Initialize structured logger
        self._provider_logger = ProviderLogger("openrouter", __name__)

        # Resolve API key using unified resolver
        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key("openrouter", explicit_key=api_key)

        # Log API key resolution
        self._provider_logger.log_api_key_resolution(key_result)

        if key_result.key is None:
            # Raise detailed error with actionable suggestions
            raise APIKeyNotFoundError(
                provider="openrouter",
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )

        self._api_key = key_result.key

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="openrouter",  # Will be set on chat()
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={
                "base_url": base_url,
                "timeout": timeout,
                "site_url": site_url,
                "site_name": site_name,
                **kwargs,
            },
        )

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
        """Send chat completion request via OpenRouter.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "anthropic/claude-sonnet-4.5")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional OpenRouter parameters

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

            except Exception as e:
                # Convert to specific provider error types based on error message
                # Skip if already a ProviderError to avoid double-wrapping
                if isinstance(e, ProviderError):
                    raise

                # Check for httpx-specific errors first
                if isinstance(e, httpx.TimeoutException):
                    raise ProviderTimeoutError(
                        message=f"OpenRouter request timed out: {str(e)}",
                        provider=self.name,
                    ) from e

                # Extract status code from httpx.HTTPStatusError if available
                status_code = None
                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code

                error_str = str(e).lower()
                if any(term in error_str for term in ["auth", "unauthorized", "invalid key", "401"]):
                    raise ProviderAuthError(
                        message=f"Authentication failed: {str(e)}",
                        provider=self.name,
                        status_code=status_code,
                    ) from e
                elif any(term in error_str for term in ["rate limit", "429", "too many requests"]):
                    raise ProviderRateLimitError(
                        message=f"Rate limit exceeded: {str(e)}",
                        provider=self.name,
                        status_code=status_code or 429,
                    ) from e
                else:
                    # Wrap generic errors in ProviderError
                    raise ProviderError(
                        message=f"OpenRouter API error: {str(e)}",
                        provider=self.name,
                        status_code=status_code,
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

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message="OpenRouter stream timed out",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                message=f"OpenRouter streaming error {e.response.status_code}",
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

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from OpenRouter."""
        try:
            response = await self.client.get("/models")
            if response.status_code == 200:
                result = response.json()
                return result.get("data", [])
        except Exception as e:
            # Use provider logger for debug output
            self._provider_logger.logger.debug(f"Failed to fetch models from OpenRouter: {e}")

        return [
            {"id": model_id, **model_info} for model_id, model_info in OPENROUTER_MODELS.items()
        ]

    async def close(self) -> None:
        await self.client.aclose()
