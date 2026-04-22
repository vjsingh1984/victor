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

from typing import Any, Dict, List, Optional

from victor.providers.base import ProviderError
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)

# Z.AI endpoint URLs for different access tiers
# Reference: https://docs.z.ai/api-reference/introduction
ZAI_BASE_URLS = {
    "standard": "https://api.z.ai/api/paas/v4/",
    # Coding Plan keys require the /coding/ endpoint (standard endpoint returns 429)
    "coding": "https://api.z.ai/api/coding/paas/v4/",
    "china": "https://open.bigmodel.cn/api/paas/v4/",
    "china-coding": "https://open.bigmodel.cn/api/coding/paas/v4/",
    "anthropic": "https://api.z.ai/api/anthropic/v1/",
}

# Available zhipuAI GLM models
# Reference: https://open.bigmodel.cn/dev/api
ZAI_MODELS = {
    # Paid flagship models (context windows from docs.z.ai and OpenRouter)
    "glm-5.1": {
        "description": "GLM-5.1 - Latest SOTA model, rivals Claude Opus 4.6",
        "context_window": 200000,
        "max_output": 65535,
        "supports_tools": True,
        "supports_thinking": True,
    },
    "glm-5": {
        "description": "GLM-5 - SOTA flagship model with agentic capabilities",
        "context_window": 200000,
        "max_output": 65535,
        "supports_tools": True,
        "supports_thinking": True,
    },
    "glm-5-turbo": {
        "description": "GLM-5-Turbo - Fast flagship model for coding",
        "context_window": 200000,
        "max_output": 16384,
        "supports_tools": True,
        "supports_thinking": True,
    },
    "glm-5-code": {
        "description": "GLM-5-Code - Specialized coding model",
        "context_window": 200000,
        "max_output": 65535,
        "supports_tools": True,
        "supports_thinking": True,
    },
    "glm-4.7": {
        "description": "GLM-4.7 - Flagship model with 200K context",
        "context_window": 200000,
        "max_output": 8192,
        "supports_tools": True,
        "supports_thinking": True,
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
    # FREE models (unlimited use, no API credits required)
    "glm-4.7-flash": {
        "description": "GLM-4.7-Flash - Free model with 200K context",
        "context_window": 200000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_thinking": False,
    },
    "glm-4.6v-flash": {
        "description": "GLM-4.6V-Flash - Free multimodal model (vision + text)",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_thinking": False,
    },
    # Paid Flash/FlashX models (lower cost)
    "glm-4.7-flashx": {
        "description": "GLM-4.7-FlashX - Fast, low-cost model",
        "context_window": 200000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_thinking": False,
    },
    "glm-4.6v-flashx": {
        "description": "GLM-4.6V-FlashX - Fast multimodal model",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_thinking": False,
    },
    "glm-4.6v": {
        "description": "GLM-4.6V - Multimodal model (vision + text)",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_thinking": False,
    },
    "glm-4.5v": {
        "description": "GLM-4.5V - Multimodal model (vision + text)",
        "context_window": 128000,
        "max_output": 4096,
        "supports_tools": True,
        "supports_thinking": False,
    },
}


class ZAIProvider(HttpxOpenAICompatProvider):
    """Provider for z.ai GLM models (OpenAI-compatible API).

    Extends ``HttpxOpenAICompatProvider`` with ZAI-specific behaviour:
    - Multi-endpoint routing (standard / coding-plan / china / anthropic)
    - Model suffix notation: "glm-5.1:coding" routes to the coding endpoint
    - Thinking mode (``thinking=True``) for GLM-4.6/4.5/4.5-Air
    - Larger default timeout (300 s) for complex multi-tool tasks

    Example:
        provider = ZAIProvider(api_key="...", model="glm-5.1:coding")
        response = await provider.chat(messages=[...], model="glm-5")
    """

    # GLM models can take 200-400 s for complex multi-tool tasks.
    DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        non_interactive: Optional[bool] = None,
        coding_plan: bool = False,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize ZhipuAI provider.

        Args:
            api_key: ZhipuAI API key (or set ZAI_API_KEY env var, or use keyring)
            base_url: ZhipuAI API base URL (overrides endpoint/coding_plan/model suffix)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            non_interactive: Force non-interactive mode (None = auto-detect)
            coding_plan: Use GLM Coding Plan endpoint (shortcut for endpoint="coding")
            endpoint: Named endpoint — "standard", "coding", "china", "anthropic"
            model: Model name (can include endpoint suffix, e.g., "glm-4.6:coding")
            **kwargs: Additional configuration

        Model Suffix Format:
            "glm-4.6:coding"   → coding endpoint
            "glm-4.6:standard" → standard endpoint
            "glm-4.6:china"    → china endpoint
        """
        # Strip model suffix for endpoint routing
        self._model_suffix_stripped = None
        model_endpoint = None
        if model and ":" in model:
            model_name, endpoint_variant = model.rsplit(":", 1)
            if endpoint_variant in ZAI_BASE_URLS:
                model_endpoint = endpoint_variant
                self._model_suffix_stripped = model
                model = model_name
        self._clean_model_init = model

        # Resolve base URL (priority: explicit > endpoint param > coding_plan > model suffix > default)
        if base_url is None:
            if endpoint is not None:
                base_url = ZAI_BASE_URLS.get(endpoint, ZAI_BASE_URLS["standard"])
            elif coding_plan:
                base_url = ZAI_BASE_URLS["coding"]
            elif model_endpoint is not None:
                base_url = ZAI_BASE_URLS[model_endpoint]
            else:
                base_url = ZAI_BASE_URLS["standard"]

        # Resolve API key
        from victor.providers.logging import ProviderLogger as _PL
        _logger = _PL("zai", __name__)
        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key("zai", explicit_key=api_key)
        _logger.log_api_key_resolution(key_result)
        if key_result.key is None:
            raise APIKeyNotFoundError(
                provider="zai",
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )
        resolved_key = key_result.key
        _logger.log_provider_init(
            model="zai",
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={"base_url": base_url, "timeout": timeout, **kwargs},
        )

        super().__init__(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            provider_name="zai",
            **kwargs,
        )

    # ── BaseProvider identity ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "zai"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    # ── Template Method overrides ─────────────────────────────────────────────

    def _clean_model_name(self, model: str) -> str:
        """Strip Z.AI endpoint suffix (e.g., "glm-5.1:coding" → "glm-5.1")."""
        if model and ":" in model:
            parts = model.rsplit(":", 1)
            if parts[1] in ZAI_BASE_URLS:
                return parts[0]
        return model

    def _get_provider_params(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        thinking: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Standard params + optional thinking mode for GLM reasoning models."""
        params: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if thinking:
            params["thinking"] = {"type": "enabled"}
        return params

    def _extract_response_metadata(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Capture reasoning_content from thinking-mode responses."""
        if "reasoning_content" in message:
            return {"reasoning_content": message.get("reasoning_content")}
        return None

    def _extract_stream_metadata(self, delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Capture reasoning_content from thinking-mode streaming deltas."""
        rc = delta.get("reasoning_content")
        if rc:
            return {"reasoning_content": rc, "thinking_mode": True}
        return None

    # ── ZAI-specific helpers ──────────────────────────────────────────────────

    def get_context_window(self, model: str) -> int:
        """Get context window size for a GLM model.

        Args:
            model: Model name (e.g., "glm-4.7", "glm-4.6")

        Returns:
            Context window size in tokens (default 128K for unknown models)
        """
        if model in ZAI_MODELS:
            return ZAI_MODELS[model]["context_window"]
        for prefix, info in ZAI_MODELS.items():
            if model.startswith(prefix):
                return info["context_window"]
        return 128000

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available z.ai GLM models."""
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            raise ProviderError(
                message=f"z.ai failed to list models: {e}",
                provider=self.name,
                raw_error=e,
            ) from e
