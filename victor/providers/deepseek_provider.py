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
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.providers.base import CompletionResponse, StreamChunk, ToolDefinition
from victor.providers.httpx_openai_compat import HttpxOpenAICompatProvider
from victor.providers.resolution import (
    UnifiedApiKeyResolver,
    APIKeyNotFoundError,
)

# Default DeepSeek API endpoint
# DeepSeek supports OpenAI-compatible format with /v1 suffix:
# https://api.deepseek.com/v1/chat/completions
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
    # DeepSeek sees intermittent sustained-load transport failures in practice,
    # so use a slightly more patient retry budget than the generic compat default.
    RETRY_ATTEMPTS = 5
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
            base_url: API endpoint (default: https://api.deepseek.com)
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
            max_retries=kwargs.pop("max_retries", self.RETRY_ATTEMPTS),
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

    def context_window(self, model: Optional[str] = None) -> int:
        from victor.providers.context_windows import DEEPSEEK, DEEPSEEK_DEFAULT, lookup

        target = model or getattr(self, "_current_model", None)
        return lookup(DEEPSEEK, target, DEEPSEEK_DEFAULT)

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

    # ── DSML fallback parser ──────────────────────────────────────────────────
    #
    # DeepSeek intermittently outputs tool calls as raw DSML (DeepSeek Markup
    # Language) text in the `content` field with finish_reason="stop" and
    # tool_calls=null, instead of the proper OpenAI-compatible tool_calls JSON.
    # This is a known API bug (~10% failure rate) documented in:
    #   https://github.com/deepseek-ai/DeepSeek-V3/issues/1244
    #
    # DSML uses fullwidth vertical bars (U+FF5C) as delimiters:
    #   <｜｜DSML｜｜tool_calls>
    #     <｜｜DSML｜｜invoke name="TOOL_NAME">
    #       <｜｜DSML｜｜parameter name="P" string="true">VALUE</｜｜DSML｜｜parameter>
    #     </｜｜DSML｜｜invoke>
    #   </｜｜DSML｜｜tool_calls>

    # U+FF5C fullwidth vertical line — used in DSML delimiters
    _FVB = "｜"
    _DSML_OPEN = re.compile(
        r"<｜{1,2}DSML｜{1,2}tool_calls\s*>",
        re.IGNORECASE,
    )
    _DSML_INVOKE = re.compile(
        r'<｜{1,2}DSML｜{1,2}invoke\s+name="([^"]+)"\s*>(.*?)</｜{0,2}DSML｜{0,2}invoke\s*>',
        re.DOTALL | re.IGNORECASE,
    )
    _DSML_PARAM = re.compile(
        r'<｜{1,2}DSML｜{1,2}parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜{0,2}DSML｜{0,2}parameter\s*>',
        re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def _parse_dsml_tool_calls(cls, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse DeepSeek DSML tool-call markup from a content string.

        Returns a list of tool call dicts (same shape as parse_openai_tool_calls)
        if DSML is found, otherwise None.
        """
        if not cls._DSML_OPEN.search(content):
            return None

        tool_calls = []
        for invoke_match in cls._DSML_INVOKE.finditer(content):
            tool_name = invoke_match.group(1).strip()
            params_block = invoke_match.group(2)
            arguments: Dict[str, Any] = {}
            for param_match in cls._DSML_PARAM.finditer(params_block):
                param_name = param_match.group(1).strip()
                is_string = param_match.group(2).lower() == "true"
                raw_value = param_match.group(3).strip()
                if is_string:
                    arguments[param_name] = raw_value
                else:
                    try:
                        arguments[param_name] = json.loads(raw_value)
                    except Exception:
                        arguments[param_name] = raw_value
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "arguments": arguments,
                }
            )
        return tool_calls if tool_calls else None

    # ── Path overrides that add DSML recovery ────────────────────────────────

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Extend base parser with DSML-in-content recovery."""
        response = super()._parse_response(result, model)
        if response.tool_calls:
            return response

        dsml_calls = self._parse_dsml_tool_calls(response.content or "")
        if dsml_calls:
            logger.debug(
                "deepseek: recovered %d tool call(s) from DSML content", len(dsml_calls)
            )
            return CompletionResponse(
                content="",
                role="assistant",
                tool_calls=dsml_calls,
                stop_reason="tool_calls",
                usage=response.usage,
                model=response.model,
                raw_response=response.raw_response,
                metadata=response.metadata,
            )
        return response

    async def stream(
        self,
        messages: List[Any],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Buffer full stream to detect and recover DSML tool calls in content."""
        buffered: List[StreamChunk] = []
        async for chunk in super().stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        ):
            buffered.append(chunk)

        # If the stream already produced proper tool calls, yield as-is.
        if any(c.tool_calls for c in buffered):
            for chunk in buffered:
                yield chunk
            return

        # Check whether accumulated content is DSML.
        full_content = "".join(c.content or "" for c in buffered)
        dsml_calls = self._parse_dsml_tool_calls(full_content)
        if dsml_calls:
            logger.debug(
                "deepseek stream: recovered %d tool call(s) from DSML content",
                len(dsml_calls),
            )
            # Suppress DSML text; yield a single tool-call chunk.
            yield StreamChunk(
                content="",
                tool_calls=dsml_calls,
                stop_reason="tool_calls",
                is_final=True,
            )
            return

        for chunk in buffered:
            yield chunk
