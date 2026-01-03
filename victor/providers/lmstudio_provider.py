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

"""LMStudio provider implementation for local model inference.

LMStudio provides an OpenAI-compatible API but requires specialized handling:
- Endpoint probing via /v1/models (not /api/tags like Ollama)
- Tiered URL fallback for multiple LMStudio hosts
- Longer timeouts (300s) for local model inference
- Direct httpx client (not AsyncOpenAI SDK)
- Thinking tag extraction for Qwen3/DeepSeek-R1 models
- Model-aware tool calling capability detection

References:
- https://lmstudio.ai/docs/api/openai-api
- https://lmstudio.ai/docs/advanced/tool-use
"""

import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

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
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities

logger = logging.getLogger(__name__)


# Models that output <think>...</think> tags
THINKING_TAG_MODELS = [
    "qwen3",
    "deepseek-r1",
    "deepseek-reasoner",
]

# Models with native tool calling support
TOOL_CAPABLE_MODELS = [
    "-tools",  # Suffix pattern (e.g., qwen3-coder-tools-30b)
    "qwen2.5-coder",
    "qwen3-coder",
    "llama3.1-tools",
    "llama3.3-tools",
    "deepseek-coder-tools",
    "deepseek-r1-tools",
    "deepseek-coder-v2-tools",
]


def _model_uses_thinking_tags(model: str) -> bool:
    """Check if a model outputs thinking tags."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in THINKING_TAG_MODELS)


def _model_supports_tools(model: str) -> bool:
    """Check if a model supports native tool calling."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in TOOL_CAPABLE_MODELS)


def _extract_thinking_content(response: str) -> Tuple[str, str]:
    """Extract <think>...</think> tags from response.

    Args:
        response: Raw response text

    Returns:
        Tuple of (thinking_content, main_content)
    """
    if not response:
        return ("", "")

    # Match <think>...</think> tags (case insensitive, multiline)
    think_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)

    thinking = "\n".join(matches) if matches else ""
    content = re.sub(think_pattern, "", response, flags=re.DOTALL | re.IGNORECASE).strip()

    return (thinking, content)


class LMStudioProvider(BaseProvider):
    """Provider for LMStudio local model server (OpenAI-compatible API).

    Features:
    - Tiered URL selection with health probing
    - Async factory for non-blocking initialization
    - Native tool calling support for hammer-badge models
    - JSON fallback parsing for other models
    - Thinking tag extraction for Qwen3/DeepSeek-R1 models
    - Model-aware capability detection
    """

    # Default timeout matches Ollama (local models need more time)
    DEFAULT_TIMEOUT = 300

    # Default LMStudio port
    DEFAULT_PORT = 1234

    # Initial request timeout (for model loading)
    INITIAL_REQUEST_TIMEOUT = 180  # 3 minutes for first request (model loading)

    def __init__(
        self,
        base_url: Union[str, List[str]] = "http://127.0.0.1:1234",
        timeout: int = DEFAULT_TIMEOUT,
        api_key: str = "lm-studio",
        _skip_discovery: bool = False,
        **kwargs: Any,
    ):
        """Initialize LMStudio provider.

        Args:
            base_url: LMStudio server URL or list of URLs (first reachable is used)
            timeout: Request timeout (default: 300s for local models)
            api_key: API key (LMStudio default is "lm-studio")
            _skip_discovery: Skip endpoint discovery (for async factory)
            **kwargs: Additional configuration
        """
        if _skip_discovery:
            # Used by async factory, base_url is already resolved
            chosen_base = (
                base_url
                if isinstance(base_url, str)
                else str(base_url[0]) if base_url else "http://127.0.0.1:1234"
            )
        else:
            chosen_base = self._select_base_url(base_url, timeout)

        super().__init__(base_url=chosen_base, timeout=timeout, **kwargs)
        self._raw_base_urls = base_url
        self._api_key = api_key
        self._models_available: Optional[bool] = None  # Set during URL discovery
        self._context_window_cache: Dict[str, int] = {}
        self._current_model: Optional[str] = None  # Track current model for capability detection
        self._model_tool_support_cache: Dict[str, bool] = {}  # Cache tool support per model

        # Use httpx directly (not AsyncOpenAI SDK) for consistent behavior with Ollama
        self.client = httpx.AsyncClient(
            base_url=f"{chosen_base}/v1",  # OpenAI-compatible /v1 path
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @classmethod
    async def create(
        cls,
        base_url: Union[str, List[str]] = "http://127.0.0.1:1234",
        timeout: int = DEFAULT_TIMEOUT,
        api_key: str = "lm-studio",
        **kwargs: Any,
    ) -> "LMStudioProvider":
        """Async factory to create LMStudioProvider with async endpoint discovery.

        Preferred over __init__ for non-blocking initialization.

        Args:
            base_url: LMStudio server URL or list of URLs
            timeout: Request timeout (default: 300s)
            api_key: API key (default: "lm-studio")
            **kwargs: Additional configuration

        Returns:
            Initialized LMStudioProvider with best available endpoint
        """
        chosen_base, models_available = await cls._select_base_url_async(base_url, timeout)
        instance = cls(
            base_url=chosen_base,
            timeout=timeout,
            api_key=api_key,
            _skip_discovery=True,
            **kwargs,
        )
        instance._models_available = models_available
        return instance

    @property
    def name(self) -> str:
        """Provider name."""
        return "lmstudio"

    def supports_tools(self) -> bool:
        """Check if current model supports tool calling.

        Returns:
            True if current model supports native tool calling
        """
        return self._supports_tools_for_model(self._current_model)

    def _supports_tools_for_model(self, model: Optional[str]) -> bool:
        """Check if specific model supports tool calling (internal).

        Args:
            model: Model name to check. If None, returns True (optimistic).

        Returns:
            True if model supports native tool calling
        """
        if not model:
            return True  # Optimistic default when model unknown

        # Check cache first
        if model in self._model_tool_support_cache:
            return self._model_tool_support_cache[model]

        # Check model name patterns
        supports = _model_supports_tools(model)
        self._model_tool_support_cache[model] = supports

        if not supports:
            logger.debug(f"LMStudio: Model {model} does not support native tool calling")

        return supports

    def supports_streaming(self) -> bool:
        """LMStudio supports streaming."""
        return True

    def model_uses_thinking_tags(self, model: Optional[str] = None) -> bool:
        """Check if model outputs thinking tags.

        Args:
            model: Model name to check. If None, uses current model.

        Returns:
            True if model outputs <think>...</think> tags
        """
        check_model = model or self._current_model
        if not check_model:
            return False
        return _model_uses_thinking_tags(check_model)

    async def check_model_status(self, model: str) -> Dict[str, Any]:
        """Check if a model is loaded and ready.

        Args:
            model: Model name to check

        Returns:
            Status dict with 'loaded', 'loading', 'error' keys
        """
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            data = response.json()

            models = data.get("data", [])
            model_info = next((m for m in models if m.get("id") == model), None)

            if model_info:
                return {
                    "loaded": True,
                    "loading": False,
                    "model": model,
                    "info": model_info,
                }

            # Model not in list - might need to be loaded
            return {
                "loaded": False,
                "loading": False,
                "model": model,
                "available_models": [m.get("id") for m in models],
            }

        except Exception as e:
            return {
                "loaded": False,
                "loading": False,
                "error": str(e),
                "model": model,
            }

    def get_context_window(self, model: str) -> int:
        """Get context window size using cached discovery or config fallback."""
        cache_key = f"{self.base_url}:{model}"
        if cache_key in self._context_window_cache:
            return self._context_window_cache[cache_key]

        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits("lmstudio", model)
        self._context_window_cache[cache_key] = limits.context_window
        return limits.context_window

    async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
        """Async capability discovery via /v1/models."""
        cache_key = f"{self.base_url}:{model}"

        context_window: Optional[int] = None
        supports_tools = True
        supports_streaming = True
        raw_response: Optional[Dict[str, Any]] = None

        try:
            resp = await self.client.get("/models")
            resp.raise_for_status()
            raw_response = resp.json()

            models = raw_response.get("data", [])
            match = next((m for m in models if m.get("id") == model), models[0] if models else None)

            if match:
                capabilities = match.get("capabilities", {}) or {}
                supports_tools = bool(
                    capabilities.get("functions", True) or capabilities.get("tools", True)
                )
                supports_streaming = bool(
                    capabilities.get("streaming", True) or capabilities.get("stream", True)
                )
                context_window = self._extract_context_window(match)

        except Exception as exc:
            logger.warning(f"Failed to discover capabilities for {model} on {self.base_url}: {exc}")

        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits("lmstudio", model)
        resolved_context = context_window or limits.context_window

        self._context_window_cache[cache_key] = resolved_context

        return ProviderRuntimeCapabilities(
            provider=self.name,
            model=model,
            context_window=resolved_context,
            supports_tools=supports_tools,
            supports_streaming=supports_streaming,
            source="discovered" if context_window else "config",
            raw=raw_response,
        )

    def _extract_context_window(self, model_info: Dict[str, Any]) -> Optional[int]:
        """Extract context window from LMStudio model info."""
        candidates = [
            model_info.get("context_length"),
            model_info.get("max_context_length"),
            model_info.get("max_context_tokens"),
        ]
        for value in candidates:
            if value is None:
                continue
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        return None

    def _select_base_url(self, base_url: Union[str, List[str], None], timeout: int) -> str:
        """Pick the first reachable LMStudio endpoint from a tiered list.

        Priority:
        1) LMSTUDIO_ENDPOINTS env var (comma-separated) if set
        2) Explicitly provided list (comma-separated) or URL
        3) Default localhost:1234

        Set LMSTUDIO_ENDPOINTS="http://<your-server>:1234,http://127.0.0.1:1234"
        to prioritize LAN servers.
        """
        import os

        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("LMSTUDIO_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if base_url is None:
            base_url = "http://127.0.0.1:1234"

        # Only process base_url if no env var candidates
        if not candidates:
            if isinstance(base_url, (list, tuple)):
                candidates = [str(u).strip() for u in base_url if str(u).strip()]
            elif isinstance(base_url, str):
                if "," in base_url:
                    candidates = [u.strip() for u in base_url.split(",") if u.strip()]
                else:
                    candidates = [base_url]
            else:
                candidates = [str(base_url)]

        for url in candidates:
            try:
                with httpx.Client(timeout=httpx.Timeout(2)) as client:
                    # LMStudio uses /v1/models for model listing (OpenAI-compatible)
                    resp = client.get(f"{url}/v1/models")
                    resp.raise_for_status()

                    # Verify we got a valid response with models
                    data = resp.json()
                    models = data.get("data", [])
                    if models:
                        model_names = [m.get("id", "unknown") for m in models[:3]]
                        logger.info(
                            f"LMStudio base URL selected: {url} "
                            f"(models: {', '.join(model_names)}{'...' if len(models) > 3 else ''})"
                        )
                        self._models_available = True
                    else:
                        logger.warning(
                            f"LMStudio server at {url} has NO MODELS LOADED. "
                            "Please load a model in LMStudio before using this provider."
                        )
                        self._models_available = False
                    return url
            except Exception as exc:
                logger.warning(f"LMStudio endpoint {url} not reachable ({exc}); trying next.")

        fallback = (
            base_url
            if isinstance(base_url, str)
            else str(base_url[0]) if base_url else "http://127.0.0.1:1234"
        )
        logger.error(
            f"No LMStudio endpoints reachable from: {candidates}. Falling back to {fallback}"
        )
        return fallback

    @classmethod
    async def _select_base_url_async(
        cls, base_url: Union[str, List[str], None], timeout: int
    ) -> Tuple[str, Optional[bool]]:
        """Async version of _select_base_url for non-blocking endpoint discovery.

        Priority:
        1) LMSTUDIO_ENDPOINTS env var (comma-separated) if set
        2) Explicitly provided list (comma-separated) or URL
        3) Default localhost:1234

        Returns:
            Tuple of (url, models_available) where models_available is:
            - True if models are loaded
            - False if no models loaded
            - None if couldn't determine
        """
        import os

        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("LMSTUDIO_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if base_url is None:
            base_url = "http://127.0.0.1:1234"

        # Only process base_url if no env var candidates
        if not candidates:
            if isinstance(base_url, (list, tuple)):
                candidates = [str(u).strip() for u in base_url if str(u).strip()]
            elif isinstance(base_url, str):
                if "," in base_url:
                    candidates = [u.strip() for u in base_url.split(",") if u.strip()]
                else:
                    candidates = [base_url]
            else:
                candidates = [str(base_url)]

        for url in candidates:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(2)) as client:
                    # LMStudio uses /v1/models for model listing (OpenAI-compatible)
                    resp = await client.get(f"{url}/v1/models")
                    resp.raise_for_status()

                    # Verify we got a valid response with models
                    data = resp.json()
                    models = data.get("data", [])
                    if models:
                        model_names = [m.get("id", "unknown") for m in models[:3]]
                        logger.info(
                            f"LMStudio base URL selected (async): {url} "
                            f"(models: {', '.join(model_names)}{'...' if len(models) > 3 else ''})"
                        )
                        return url, True
                    else:
                        logger.warning(
                            f"LMStudio server at {url} has NO MODELS LOADED. "
                            "Please load a model in LMStudio before using this provider."
                        )
                        return url, False
            except Exception as exc:
                logger.warning(f"LMStudio endpoint {url} not reachable ({exc}); trying next.")

        fallback = (
            base_url
            if isinstance(base_url, str)
            else str(base_url[0]) if base_url else "http://127.0.0.1:1234"
        )
        logger.error(
            f"No LMStudio endpoints reachable from: {candidates}. Falling back to {fallback}"
        )
        return fallback, None

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
        """Send chat completion request to LMStudio.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "qwen3-coder-30b", "llama-3.1-8b")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools (if model supports)
            **kwargs: Additional OpenAI-compatible options

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails or no models loaded
        """
        # Track current model for capability detection
        self._current_model = model

        # Early check: fail fast if no models are loaded
        if self._models_available is False:
            raise ProviderError(
                "LMStudio server has no models loaded. "
                "Please load a model in LMStudio before using this provider. "
                f"Server URL: {self.base_url}"
            )

        # Filter tools if model doesn't support them
        effective_tools = tools
        if tools and not self._supports_tools_for_model(model):
            logger.debug(
                f"LMStudio: Model {model} doesn't support native tool calling, "
                "falling back to text-based parsing"
            )
            effective_tools = None

        try:
            payload = self._build_request_payload(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=effective_tools,
                stream=False,
                **kwargs,
            )

            # Make API call with circuit breaker protection
            response = await self._execute_with_circuit_breaker(
                self.client.post, "/chat/completions", json=payload
            )
            response.raise_for_status()

            result = response.json()
            return self._parse_response(result, model)

        except httpx.TimeoutException as e:
            # Enhanced timeout error with helpful suggestions
            raise ProviderTimeoutError(
                message=(
                    f"LMStudio request timed out after {self.timeout}s. "
                    f"Possible causes:\n"
                    f"  1. Model '{model}' is still loading (first request takes longer)\n"
                    f"  2. Model is too large for available VRAM\n"
                    f"  3. Server is overloaded\n"
                    f"Try: Increase timeout or wait for model to finish loading."
                ),
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass

            # Enhanced error classification
            status_code = e.response.status_code
            if status_code == 503:
                message = (
                    f"LMStudio server unavailable (503). "
                    f"The model may still be loading. "
                    f"Server: {self.base_url}"
                )
            elif status_code == 500 and "out of memory" in error_body.lower():
                message = (
                    f"LMStudio out of memory loading model '{model}'. "
                    f"Try a smaller model or free up VRAM."
                )
            else:
                message = f"LMStudio HTTP error {status_code}: {error_body}"

            raise ProviderError(
                message=message,
                provider=self.name,
                status_code=status_code,
                raw_error=e,
            ) from e
        except httpx.ConnectError as e:
            raise ProviderError(
                message=(
                    f"Cannot connect to LMStudio server at {self.base_url}. "
                    f"Ensure LMStudio is running and the server is enabled."
                ),
                provider=self.name,
                raw_error=e,
            ) from e
        except Exception as e:
            raise ProviderError(
                message=f"LMStudio unexpected error: {str(e)}",
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
        """Stream chat completion from LMStudio.

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
            ProviderError: If request fails or no models loaded
        """
        # Track current model for capability detection
        self._current_model = model

        # Early check: fail fast if no models are loaded
        if self._models_available is False:
            raise ProviderError(
                "LMStudio server has no models loaded. "
                "Please load a model in LMStudio before using this provider. "
                f"Server URL: {self.base_url}"
            )

        # Filter tools if model doesn't support them
        effective_tools = tools
        if tools and not self._supports_tools_for_model(model):
            logger.debug(
                f"LMStudio: Model {model} doesn't support native tool calling, "
                "falling back to text-based parsing"
            )
            effective_tools = None

        # Check if model uses thinking tags
        uses_thinking = self.model_uses_thinking_tags(model)

        try:
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
                f"LMStudio streaming request: model={model}, msgs={len(messages)}, "
                f"tools={num_tools}, thinking_model={uses_thinking}"
            )

            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                accumulated_content = ""
                accumulated_tool_calls: List[Dict[str, Any]] = []
                line_count = 0

                async for line in response.aiter_lines():
                    line_count += 1

                    if not line.strip():
                        continue

                    # OpenAI SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str.strip() == "[DONE]":
                            # Final chunk - extract thinking content if any
                            final_metadata = None

                            if uses_thinking and accumulated_content:
                                thinking, _ = _extract_thinking_content(accumulated_content)
                                if thinking:
                                    final_metadata = {"reasoning_content": thinking}
                                    logger.debug(
                                        f"LMStudio: Extracted {len(thinking)} chars of thinking"
                                    )

                            logger.debug(f"LMStudio stream complete after {line_count} lines")
                            yield StreamChunk(
                                content="",
                                tool_calls=(
                                    accumulated_tool_calls if accumulated_tool_calls else None
                                ),
                                stop_reason="stop",
                                is_final=True,
                                metadata=final_metadata,
                            )
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            chunk = self._parse_stream_chunk(
                                chunk_data, accumulated_tool_calls, model
                            )
                            if chunk.content:
                                accumulated_content += chunk.content
                            yield chunk

                        except json.JSONDecodeError:
                            logger.warning(f"LMStudio JSON decode error on line: {line[:100]}")

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=(
                    f"LMStudio stream timed out after {self.timeout}s. "
                    f"Model '{model}' may still be loading."
                ),
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass

            # Enhanced error classification
            status_code = e.response.status_code
            if status_code == 503:
                message = (
                    "LMStudio server unavailable (503) during streaming. "
                    "The model may still be loading."
                )
            else:
                message = f"LMStudio streaming HTTP error {status_code}: {error_body}"

            raise ProviderError(
                message=message,
                provider=self.name,
                status_code=status_code,
                raw_error=e,
            ) from e
        except httpx.ConnectError as e:
            raise ProviderError(
                message=(
                    f"Cannot connect to LMStudio server at {self.base_url}. "
                    f"Ensure LMStudio is running and the server is enabled."
                ),
                provider=self.name,
                raw_error=e,
            ) from e
        except Exception as e:
            raise ProviderError(
                message=f"LMStudio stream error: {str(e)}",
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
        """Build request payload for LMStudio's OpenAI-compatible API.

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
            "temperature": temperature,
            "max_tokens": max_tokens,
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
            # Auto tool choice for native tool calling
            payload["tool_choice"] = "auto"

        # Merge additional options (excluding internal ones)
        internal_keys = {"_skip_discovery", "api_key"}
        for key, value in kwargs.items():
            if key not in internal_keys and value is not None:
                payload[key] = value

        return payload

    def _normalize_tool_calls(
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Normalize tool calls from LMStudio's OpenAI-compatible format.

        LMStudio returns tool calls in OpenAI format:
        {'id': '...', 'type': 'function', 'function': {'name': 'tool_name', 'arguments': '...'}}

        We need:
        {'name': 'tool_name', 'arguments': {...}}

        Args:
            tool_calls: Raw tool calls from LMStudio

        Returns:
            Normalized tool calls
        """
        if not tool_calls:
            return None

        normalized = []
        for call in tool_calls:
            if isinstance(call, dict) and "function" in call:
                # OpenAI format
                function = call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", "{}")

                # Parse arguments if string (LMStudio returns JSON string)
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                if name:
                    normalized.append({"name": name, "arguments": arguments})
            elif isinstance(call, dict) and "name" in call:
                # Already normalized
                normalized.append(call)

        return normalized if normalized else None

    def _parse_json_tool_call_from_content(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from JSON text in content (fallback for models without native support).

        Some LMStudio models return tool calls as JSON in content instead of using
        the structured tool_calls field. This method detects and parses them.

        Supported formats:
        - {"name": "tool_name", "arguments": {...}}
        - {"name": "tool_name", "parameters": {...}}
        - [TOOL_REQUEST]{"name": "...", "arguments": {...}}[END_TOOL_REQUEST]

        Args:
            content: Message content that might contain JSON tool call

        Returns:
            List of tool calls if detected, None otherwise
        """
        if not content or not content.strip():
            return None

        # Try LMStudio's [TOOL_REQUEST] format first
        import re

        tool_request_pattern = r"\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]"
        matches = re.findall(tool_request_pattern, content, re.DOTALL)
        if matches:
            tool_calls = []
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    name = data.get("name", "")
                    arguments = data.get("arguments") or data.get("parameters", {})
                    if name:
                        tool_calls.append({"name": name, "arguments": arguments})
                except json.JSONDecodeError:
                    continue
            if tool_calls:
                return tool_calls

        # Try to parse as direct JSON
        try:
            data = json.loads(content.strip())

            # Check if it looks like a tool call
            if isinstance(data, dict) and "name" in data:
                arguments = data.get("arguments") or data.get("parameters", {})
                return [{"name": data.get("name"), "arguments": arguments}]
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse LMStudio API response (OpenAI-compatible format).

        Handles thinking tag extraction for Qwen3/DeepSeek-R1 models.

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
        raw_content = message.get("content", "") or ""
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        # Extract thinking content for thinking-enabled models
        content = raw_content
        metadata = None
        if self.model_uses_thinking_tags(model) and raw_content:
            thinking, main_content = _extract_thinking_content(raw_content)
            if thinking:
                metadata = {"reasoning_content": thinking}
                content = main_content
                logger.debug(f"LMStudio: Extracted {len(thinking)} chars of thinking from {model}")

        # Fallback: Check if content contains JSON tool call
        if not tool_calls and content:
            parsed_tool_calls = self._parse_json_tool_call_from_content(content)
            if parsed_tool_calls:
                logger.debug(
                    f"LMStudio: Parsed tool call from content (fallback for model: {model})"
                )
                tool_calls = parsed_tool_calls
                content = ""  # Clear content since it was a tool call

        # Parse usage stats
        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

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
        self,
        chunk_data: Dict[str, Any],
        accumulated_tool_calls: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> StreamChunk:
        """Parse streaming chunk from LMStudio (OpenAI-compatible SSE format).

        Args:
            chunk_data: Raw chunk data
            accumulated_tool_calls: List to accumulate tool call deltas
            model: Model name for thinking tag detection

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

        # Handle tool call deltas (OpenAI streaming format)
        tool_call_deltas = delta.get("tool_calls", [])
        for tc_delta in tool_call_deltas:
            idx = tc_delta.get("index", 0)
            # Extend accumulated list if needed
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
        )

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models on LMStudio server.

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
                message=f"LMStudio failed to list models: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
