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

"""Qwen (Alibaba Cloud) provider implementation.

Qwen provides an OpenAI-compatible API with support for:
- Chat completions with streaming
- Native function/tool calling
- Thinking mode for reasoning models
- OAuth subscription auth (Qwen Coding Plan) and API key auth

Authentication:
- API key: via QWEN_API_KEY / DASHSCOPE_API_KEY env vars or keyring
- OAuth: via Qwen Coding Plan subscription (browser-based PKCE flow)

References:
- https://qwen.ai/apiplatform
- https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/
- https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference/
"""

from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.openai_compat import convert_tools_to_openai_format
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)
from victor.providers.logging import ProviderLogger
from victor.providers.oauth_manager import OAuthTokenManager

# Qwen API endpoints
QWEN_BASE_URLS = {
    "standard": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
    "portal": "https://portal.qwen.ai/v1/",
    # Alibaba Cloud Coding Plan: multi-model access (Qwen + GLM + Kimi + MiniMax)
    "coding": "https://coding.dashscope.aliyuncs.com/v1/",
}

# Qwen OAuth configuration (for reference / documentation)
QWEN_OAUTH_CONFIG = {
    "oauth_base_url": "https://chat.qwen.ai",
    "api_base_url": "https://portal.qwen.ai/v1/",
}

# Available Qwen models
QWEN_MODELS = {
    "qwen3.5": {
        "description": "Qwen3.5 - Latest flagship model",
        "context_window": 131072,
        "supports_tools": True,
    },
    "qwen3.5-plus": {
        "description": "Qwen3.5-Plus - Enhanced flagship",
        "context_window": 131072,
        "supports_tools": True,
    },
    "qwen3-coder-plus": {
        "description": "Qwen3-Coder-Plus - Coding-optimized model",
        "context_window": 131072,
        "supports_tools": True,
    },
    "qwen-turbo-latest": {
        "description": "Qwen Turbo - Fast, cost-effective",
        "context_window": 131072,
        "supports_tools": True,
    },
    "qwen-plus-latest": {
        "description": "Qwen Plus - Balanced performance",
        "context_window": 131072,
        "supports_tools": True,
    },
    "qwen-max-latest": {
        "description": "Qwen Max - Maximum capability",
        "context_window": 131072,
        "supports_tools": True,
    },
}


class QwenProvider(BaseProvider):
    """Provider for Qwen models via OpenAI-compatible API.

    Supports both API key and OAuth (Qwen Coding Plan) authentication.

    Example:
        # API key mode
        provider = QwenProvider(api_key="your-dashscope-key")

        # OAuth mode (Qwen Coding Plan subscription)
        provider = QwenProvider(auth_mode="oauth")

        response = await provider.chat(
            messages=[Message(role="user", content="Hello!")],
            model="qwen3.5"
        )
    """

    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        non_interactive: Optional[bool] = None,
        auth_mode: str = "api_key",
        oauth_tokens: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize Qwen provider.

        Args:
            api_key: Qwen/DashScope API key (or set QWEN_API_KEY env var)
            base_url: API base URL (auto-selected based on auth_mode if not set)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            non_interactive: Force non-interactive mode (None = auto-detect)
            auth_mode: "api_key" (default) or "oauth" (Qwen Coding Plan)
            oauth_tokens: Pre-obtained OAuth tokens (optional)
            **kwargs: Additional configuration
        """
        self._provider_logger = ProviderLogger("qwen", __name__)
        self._oauth_manager: Optional[OAuthTokenManager] = None

        if auth_mode == "oauth":
            # OAuth mode: use portal.qwen.ai endpoint
            if base_url is None:
                base_url = QWEN_BASE_URLS["portal"]

            self._oauth_manager = OAuthTokenManager("qwen")

            if oauth_tokens is not None:
                resolved_key = oauth_tokens.access_token
            else:
                cached = self._oauth_manager._load_cached()
                if cached is not None and not cached.is_expired:
                    resolved_key = cached.access_token
                else:
                    # Placeholder — actual login deferred to _ensure_valid_token()
                    # which runs inside the async context (chat/stream)
                    resolved_key = "oauth-pending"

            self._api_key = resolved_key

            self._provider_logger.log_provider_init(
                model="qwen",
                key_source="oauth",
                non_interactive=False,
                config={
                    "base_url": base_url,
                    "timeout": timeout,
                    "auth_mode": "oauth",
                    **kwargs,
                },
            )
        else:
            # API key mode: use dashscope endpoint
            if base_url is None:
                base_url = QWEN_BASE_URLS["standard"]

            resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
            key_result = resolver.get_api_key("qwen", explicit_key=api_key)
            self._provider_logger.log_api_key_resolution(key_result)

            if key_result.key is None:
                raise APIKeyNotFoundError(
                    provider="qwen",
                    sources_attempted=key_result.sources_attempted,
                    non_interactive=key_result.non_interactive,
                )

            self._api_key = key_result.key

            self._provider_logger.log_provider_init(
                model="qwen",
                key_source=key_result.source_detail,
                non_interactive=key_result.non_interactive,
                config={
                    "base_url": base_url,
                    "timeout": timeout,
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
        self.client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        return "qwen"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    async def _ensure_valid_token(self) -> None:
        """Refresh OAuth token if needed. No-op for api_key mode."""
        if self._oauth_manager is None:
            return
        token = await self._oauth_manager.get_valid_token()
        if token != self.client.api_key:
            self.client.api_key = token
            self._api_key = token

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str = "qwen3.5",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request to Qwen (OpenAI-compatible)."""
        await self._ensure_valid_token()

        try:
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            request_params: Dict[str, Any] = {
                "model": model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                request_params["tools"] = convert_tools_to_openai_format(tools)
                request_params["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**request_params)

            choice = response.choices[0] if response.choices else None
            content = choice.message.content or "" if choice else ""

            tool_calls = None
            if choice and choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in choice.message.tool_calls
                ]

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return CompletionResponse(
                content=content,
                role="assistant",
                tool_calls=tool_calls,
                stop_reason=choice.finish_reason if choice else None,
                usage=usage,
                model=model,
                raw_response=response.model_dump(),
            )

        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            error_str = str(e).lower()
            if "401" in error_str or "auth" in error_str:
                raise ProviderAuthError(
                    message=f"Qwen authentication failed: {e}",
                    provider=self.name,
                ) from e
            elif "429" in error_str or "rate" in error_str:
                raise ProviderRateLimitError(
                    message=f"Qwen rate limit: {e}",
                    provider=self.name,
                ) from e
            raise ProviderError(
                message=f"Qwen API error: {e}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str = "qwen3.5",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from Qwen (OpenAI-compatible)."""
        await self._ensure_valid_token()

        try:
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            request_params: Dict[str, Any] = {
                "model": model,
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if tools:
                request_params["tools"] = convert_tools_to_openai_format(tools)
                request_params["tool_choice"] = "auto"

            stream_response = await self.client.chat.completions.create(**request_params)

            async for chunk in stream_response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = delta.content or ""
                finish_reason = chunk.choices[0].finish_reason

                yield StreamChunk(
                    content=content,
                    stop_reason=finish_reason,
                    is_final=finish_reason is not None,
                )

        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                message=f"Qwen streaming error: {e}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self.client.close()
