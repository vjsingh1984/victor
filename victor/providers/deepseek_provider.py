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

from typing import Any, Dict, List, Optional

from victor.providers.base import ToolDefinition
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)

# Default DeepSeek API endpoint
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

# Available DeepSeek models
DEEPSEEK_MODELS = {
    "deepseek-chat": {
        "description": "DeepSeek-V3.2 non-thinking mode with function calling",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "supports_thinking": False,
    },
    "deepseek-reasoner": {
        "description": "DeepSeek-V3.2 thinking mode with CoT and function calling",
        "context_window": 131072,
        "max_output": 65536,
        "supports_tools": True,
        "supports_thinking": True,
    },
}


class DeepSeekProvider(HttpxOpenAICompatProvider):
    """Provider for DeepSeek API (OpenAI-compatible).

    Extends ``HttpxOpenAICompatProvider`` with DeepSeek-specific behaviour:
    - Temperature is suppressed for reasoner/r1 models
    - Tools are filtered out for reasoner/r1 models (not supported)
    - ``reasoning_content`` extracted from both streaming and non-streaming responses
    - Longer timeout cap to prevent stalled benchmark runs

    Gains ZAI refinements for free:
    - Correct tool_calls serialization in message history
    - fix_orphaned_tool_messages() on every request
    - Detailed JSON error body parsing on 400 errors
    """

    DEFAULT_TIMEOUT = 120
    # Cap per-call timeout to prevent a single API call from stalling a benchmark
    MAX_API_TIMEOUT = 120

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
            timeout: Request timeout (capped at 120 s internally)
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        from victor.providers.logging import ProviderLogger as _PL

        _logger = _PL("deepseek", __name__)

        resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
        key_result = resolver.get_api_key("deepseek", explicit_key=api_key)

        _logger.log_api_key_resolution(key_result)

        if key_result.key is None:
            raise APIKeyNotFoundError(
                provider="deepseek",
                sources_attempted=key_result.sources_attempted,
                non_interactive=key_result.non_interactive,
            )

        _logger.log_provider_init(
            model="deepseek-chat",
            key_source=key_result.source_detail,
            non_interactive=key_result.non_interactive,
            config={"base_url": base_url, "timeout": timeout, **kwargs},
        )

        # Cap per-call timeout (preserve original class behaviour)
        api_timeout = min(timeout, self.MAX_API_TIMEOUT)

        super().__init__(
            api_key=key_result.key,
            base_url=base_url,
            timeout=api_timeout,
            max_retries=kwargs.pop("max_retries", 3),
            provider_name="deepseek",
            **kwargs,
        )

    # ── BaseProvider identity ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "deepseek"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def supports_prompt_caching(self) -> bool:
        """DeepSeek automatic disk cache (90% discount, $0 write, 1h+ TTL)."""
        return True

    def supports_kv_prefix_caching(self) -> bool:
        return True

    # ── Template Method overrides ─────────────────────────────────────────────

    def _get_provider_params(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Skip temperature for reasoner/r1 models (API ignores it but cleaner not to send)."""
        params: Dict[str, Any] = {"max_tokens": max_tokens}
        is_reasoner = "reasoner" in model.lower() or "r1" in model.lower()
        if not is_reasoner:
            params["temperature"] = temperature
        return params

    def _filter_tools_for_model(
        self,
        model: str,
        tools: Optional[List[ToolDefinition]],
    ) -> Optional[List[ToolDefinition]]:
        """Reasoner/r1 models don't support function calling — suppress tools."""
        if tools and ("reasoner" in model.lower() or "r1" in model.lower()):
            return None
        return tools

    def _extract_response_metadata(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Capture reasoning_content from deepseek-reasoner non-streaming responses."""
        rc = message.get("reasoning_content")
        if rc:
            return {"reasoning_content": rc}
        return None

    def _extract_stream_metadata(self, delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Capture per-chunk reasoning_content delta from deepseek-reasoner streaming."""
        rc = delta.get("reasoning_content")
        if rc:
            return {"reasoning_content": rc}
        return None

    # ── DeepSeek-specific helpers ─────────────────────────────────────────────

    def get_context_window(self, model: str) -> int:
        """Get context window size for a DeepSeek model (default 128K)."""
        if model in DEEPSEEK_MODELS:
            return DEEPSEEK_MODELS[model]["context_window"]
        return 131072

    async def list_models(self) -> List[Dict[str, Any]]:
        """Return static model list (DeepSeek API also exposes /models)."""
        return [
            {"id": model_id, "object": "model", **info}
            for model_id, info in DEEPSEEK_MODELS.items()
        ]
