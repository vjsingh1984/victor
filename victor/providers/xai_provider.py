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
- Multiple Grok models (grok-4, grok-3, grok-code-fast-1, etc.)

References:
- https://docs.x.ai/docs/api-reference
- https://docs.x.ai/docs/guides/function-calling
"""

from typing import Any, Dict, List, Optional

from victor.providers.base import ProviderError
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)

# Default xAI API endpoint
DEFAULT_BASE_URL = "https://api.x.ai/v1"

# Available xAI Grok models
# Reference: https://docs.x.ai/docs/models
XAI_MODELS = {
    "grok-4-1-fast-reasoning": {
        "description": "Grok 4.1 Fast - Best for tool calling, 2M context",
        "context_window": 2000000,
        "max_output": 16384,
        "supports_tools": True,
    },
    "grok-4-1-fast-non-reasoning": {
        "description": "Grok 4.1 Fast (non-reasoning) - Fast tool calling, 2M context",
        "context_window": 2000000,
        "max_output": 16384,
        "supports_tools": True,
    },
    "grok-4": {
        "description": "Grok 4 - Flagship reasoning model, 256K context",
        "context_window": 262144,
        "max_output": 16384,
        "supports_tools": True,
    },
    "grok-code-fast-1": {
        "description": "Grok Code Fast - Specialized for coding, 256K context",
        "context_window": 262144,
        "max_output": 16384,
        "supports_tools": True,
    },
    "grok-3": {
        "description": "Grok-3 - High quality, 128K context",
        "context_window": 131072,
        "max_output": 16384,
        "supports_tools": True,
    },
    "grok-3-mini": {
        "description": "Grok-3 Mini - Fast reasoning, 128K context",
        "context_window": 131072,
        "max_output": 16384,
        "supports_tools": True,
    },
    "grok-2-vision-1212": {
        "description": "Grok 2 Vision - Multimodal, 32K context",
        "context_window": 32768,
        "max_output": 4096,
        "supports_tools": True,
    },
}


class XAIProvider(HttpxOpenAICompatProvider):
    """Provider for xAI Grok models (OpenAI-compatible API).

    Extends ``HttpxOpenAICompatProvider`` — no provider-specific parameter
    overrides are needed; standard temperature/max_tokens work as-is.

    Gains ZAI refinements for free:
    - Correct tool_calls serialization in message history
    - fix_orphaned_tool_messages() on every request
    - Detailed JSON error body parsing on 400 errors
    """

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
        """Initialize xAI provider.

        Args:
            api_key: xAI API key (or set XAI_API_KEY / GROK_API_KEY env var)
            base_url: xAI API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        from victor.providers.logging import ProviderLogger as _PL
        _logger = _PL("xai", __name__)

        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key("xai", explicit_key=api_key)
        if key_result.key is None:
            key_result = resolver.get_api_key("grok", explicit_key=api_key)

        _logger.log_api_key_resolution(key_result)

        if key_result.key is None:
            raise APIKeyNotFoundError(
                provider="xai",
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )

        _logger.log_provider_init(
            model="grok",
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={"base_url": base_url, "timeout": timeout, **kwargs},
        )

        super().__init__(
            api_key=key_result.key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            provider_name="xai",
            **kwargs,
        )

    # ── BaseProvider identity ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "xai"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_prompt_caching(self) -> bool:
        """xAI/Grok auto-caches prompts (50-75% discount on cached tokens)."""
        return True

    def supports_kv_prefix_caching(self) -> bool:
        """xAI reuses KV cache for matching prompt prefixes."""
        return True

    # ── xAI-specific helpers ──────────────────────────────────────────────────

    def get_context_window(self, model: str) -> int:
        """Get context window size for a Grok model (default 128K)."""
        if model in XAI_MODELS:
            return XAI_MODELS[model]["context_window"]
        for prefix, info in XAI_MODELS.items():
            if model.startswith(prefix):
                return info["context_window"]
        return 131072

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available xAI models."""
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            raise ProviderError(
                message=f"xAI failed to list models: {e}",
                provider=self.name,
                raw_error=e,
            ) from e
