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

"""Ollama provider implementation for local model inference."""

import json
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Union

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
from victor.providers.logging import ProviderLogger
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities
from victor.providers.ollama_capability_detector import TOOL_SUPPORT_PATTERNS


class OllamaProvider(BaseProvider):
    """Provider for Ollama local model server."""

    def __init__(
        self,
        base_url: Union[str, List[str]] = "http://localhost:11434",
        timeout: int = 300,
        _skip_discovery: bool = False,
        non_interactive: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server URL or list/comma-separated URLs.
                     For synchronous init, the first URL is used without verification.
                     Use OllamaProvider.create() for async discovery of reachable endpoint.
            timeout: Request timeout (longer for local models)
            _skip_discovery: Deprecated flag (kept for compatibility).
            non_interactive: Force non-interactive mode (None = auto-detect)
            **kwargs: Additional configuration
        """
        import os

        # Initialize structured logger
        self._provider_logger = ProviderLogger("ollama", __name__)

        # Resolve candidates (logic from _select_base_url but without network I/O)
        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("OLLAMA_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if not candidates:
            if base_url is None:
                candidates = ["http://localhost:11434"]
            elif isinstance(base_url, (list, tuple)):
                candidates = [str(u).strip() for u in base_url if str(u).strip()]
            elif isinstance(base_url, str):
                if "," in base_url:
                    candidates = [u.strip() for u in base_url.split(",") if u.strip()]
                else:
                    candidates = [base_url]
            else:
                candidates = [str(base_url)]

        if not candidates:
            candidates = ["http://localhost:11434"]

        # Pick first candidate (blindly)
        chosen_base = candidates[0]

        # Log provider initialization
        self._provider_logger.log_provider_init(
            model="local",  # Ollama runs local models
            key_source=None,  # No API key for local server
            non_interactive=non_interactive or False,
            config={
                "base_url": base_url,
                "timeout": timeout,
                **kwargs,
            },
        )

        super().__init__(base_url=chosen_base, timeout=timeout, **kwargs)
        self._raw_base_urls = base_url
        self._models_without_tools: set = set()  # Cache models that don't support tools
        self._context_window_cache: Dict[str, int] = {}  # Cache model context windows
        self.client = httpx.AsyncClient(
            base_url=chosen_base,
            timeout=httpx.Timeout(timeout),
        )

    @classmethod
    async def create(
        cls,
        base_url: Union[str, List[str]] = "http://localhost:11434",
        timeout: int = 300,
        **kwargs: Any,
    ) -> "OllamaProvider":
        """Async factory to create OllamaProvider with async endpoint discovery.

        Preferred over __init__ for non-blocking initialization.

        Args:
            base_url: Ollama server URL or list/comma-separated URLs
            timeout: Request timeout (longer for local models)
            **kwargs: Additional configuration

        Returns:
            Initialized OllamaProvider with best available endpoint
        """
        chosen_base = await cls._select_base_url_async(base_url, timeout)
        return cls(base_url=chosen_base, timeout=timeout, _skip_discovery=True, **kwargs)

    @property
    def name(self) -> str:
        """Provider name."""
        return "ollama"

    def supports_tools(self) -> bool:
        """Ollama supports tool calling for compatible models."""
        return True

    def supports_streaming(self) -> bool:
        """Ollama supports streaming."""
        return True

    def get_context_window(self, model: str) -> int:
        """Get context window size using cached discovery or config fallback."""
        cache_key = f"{self.base_url}:{model}"

        if cache_key in self._context_window_cache:
            return self._context_window_cache[cache_key]

        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits("ollama", model)
        self._context_window_cache[cache_key] = limits.context_window
        return limits.context_window

    async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
        """Async capability discovery using /api/show."""
        cache_key = f"{self.base_url}:{model}"

        context_window: Optional[int] = None
        supports_tools = True  # Default: Ollama supports tools for capable models
        raw_response: Optional[Dict[str, Any]] = None

        try:
            resp = await self.client.post("/api/show", json={"name": model})
            resp.raise_for_status()
            raw_response = resp.json()

            context_window = self._parse_context_window(raw_response)
            template = raw_response.get("template", "") or ""
            supports_tools = self._detect_tool_support(template)

        except Exception as exc:
            self._provider_logger.logger.warning(
                f"Failed to discover capabilities for {model} on {self.base_url}: {exc}"
            )

        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits("ollama", model)
        resolved_context = context_window or limits.context_window

        self._context_window_cache[cache_key] = resolved_context

        return ProviderRuntimeCapabilities(
            provider=self.name,
            model=model,
            context_window=resolved_context,
            supports_tools=supports_tools,
            supports_streaming=True,
            source="discovered" if context_window else "config",
            raw=raw_response,
        )

    def _parse_context_window(self, response_data: Dict[str, Any]) -> Optional[int]:
        """Extract context window from Ollama /api/show response."""
        parameters = response_data.get("parameters", "")
        if "num_ctx" in parameters:
            for line in parameters.split("\n"):
                if "num_ctx" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[-1])
                        except (ValueError, IndexError):
                            continue

        model_info = response_data.get("model_info", {})
        for key, value in model_info.items():
            if "context_length" in key.lower():
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue

        return None

    def _detect_tool_support(self, template: str) -> bool:
        """Detect native tool support from model template."""
        if not template:
            return True  # default optimistic

        for pattern in TOOL_SUPPORT_PATTERNS:
            try:
                if pattern and re.search(pattern, template):
                    return True
            except Exception:
                continue

        return False

    def _select_base_url(self, base_url: Union[str, List[str], None], timeout: int) -> str:
        """Pick the first reachable Ollama endpoint from a tiered list.

        Priority:
        1) OLLAMA_ENDPOINTS env var (comma-separated) if set
        2) Explicitly provided list (comma-separated) or URL
        3) Default localhost

        Set OLLAMA_ENDPOINTS="http://<your-server>:11434,http://localhost:11434"
        to prioritize LAN servers.
        """
        import os

        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("OLLAMA_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if base_url is None:
            base_url = "http://localhost:11434"

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
                with httpx.Client(base_url=url, timeout=httpx.Timeout(2)) as client:
                    resp = client.get("/api/tags")
                    resp.raise_for_status()
                    self._provider_logger.logger.info(f"Ollama base URL selected: {url}")
                    return url
            except Exception as exc:
                self._provider_logger.logger.warning(
                    f"Ollama endpoint {url} not reachable ({exc}); trying next."
                )

        self._provider_logger.logger.error(
            f"No Ollama endpoints reachable from: {candidates}. Falling back to {base_url}"
        )
        return base_url

    @classmethod
    async def _select_base_url_async(
        cls, base_url: Union[str, List[str], None], timeout: int
    ) -> str:
        """Async version of _select_base_url for non-blocking endpoint discovery.

        Priority:
        1) OLLAMA_ENDPOINTS env var (comma-separated) if set
        2) Explicitly provided list (comma-separated) or URL
        3) Default localhost

        Set OLLAMA_ENDPOINTS="http://<your-server>:11434,http://localhost:11434"
        to prioritize LAN servers.
        """
        import os

        candidates: List[str] = []

        # Check environment variable first
        env_endpoints = os.environ.get("OLLAMA_ENDPOINTS", "")
        if env_endpoints:
            candidates = [u.strip() for u in env_endpoints.split(",") if u.strip()]

        if base_url is None:
            base_url = "http://localhost:11434"

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

        # Use module logger since this is a classmethod
        from victor.providers.logging import logger as provider_logger

        for url in candidates:
            try:
                async with httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(2)) as client:
                    resp = await client.get("/api/tags")
                    resp.raise_for_status()
                    provider_logger.info(f"Ollama base URL selected (async): {url}")
                    return url
            except Exception as exc:
                provider_logger.warning(
                    f"Ollama endpoint {url} not reachable ({exc}); trying next."
                )

        provider_logger.error(
            f"No Ollama endpoints reachable from: {candidates}. Falling back to {base_url}"
        )
        return base_url if isinstance(base_url, str) else str(base_url)

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
        """Send chat completion request to Ollama.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "llama3:8b", "qwen2.5-coder:7b")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools (if model supports)
            **kwargs: Additional Ollama options

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderAuthError: If authentication fails (for authenticated servers)
            ProviderRateLimitError: If rate limit is exceeded
            ProviderTimeoutError: If request times out
            ProviderError: For other errors
        """
        # Use structured logging context manager
        with self._provider_logger.log_api_call(
            endpoint="/api/chat",
            model=model,
            operation="chat",
            num_messages=len(messages),
            has_tools=tools is not None,
        ):
            try:
                payload = self._build_request_payload(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    stream=False,
                    **kwargs,
                )

                # Log the endpoint URL being used for connection
                endpoint_url = f"{self.base_url}/api/chat"
                self._provider_logger.logger.debug(
                    f"Connecting to Ollama endpoint: {endpoint_url}"
                )

                # Make API call with circuit breaker protection
                response = await self._execute_with_circuit_breaker(
                    self.client.post, "/api/chat", json=payload
                )
                response.raise_for_status()

                result = response.json()
                parsed = self._parse_response(result, model)

                # Log success with usage info
                tokens = parsed.usage.get("total_tokens") if parsed.usage else None
                self._provider_logger._log_api_call_success(
                    call_id=f"chat_{model}_{id(payload)}",
                    endpoint="/api/chat",
                    model=model,
                    start_time=0,  # Set by context manager
                    tokens=tokens,
                )

                return parsed

            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(
                    message=f"Request timed out after {self.timeout}s",
                    provider=self.name,
                ) from e
            except httpx.HTTPStatusError as e:
                # Check for "does not support tools" error (HTTP 400)
                # Retry without tools if model doesn't support them
                if e.response.status_code == 400 and tools:
                    try:
                        error_text = e.response.text
                        if "does not support tools" in error_text.lower():
                            self._provider_logger.logger.warning(
                                f"Model {model} doesn't support tools via Ollama API. "
                                "Retrying without tools (will use fallback parsing)."
                            )
                            # Cache that this model doesn't support tools
                            self._models_without_tools.add(model)
                            # Retry without tools
                            payload = self._build_request_payload(
                                messages=messages,
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                tools=None,  # No tools
                                stream=False,
                                **kwargs,
                            )
                            response = await self._execute_with_circuit_breaker(
                                self.client.post, "/api/chat", json=payload
                            )
                            response.raise_for_status()
                            result = response.json()
                            return self._parse_response(result, model)
                    except Exception:
                        pass  # Fall through to original error

                # Include error body for better debugging
                error_body = ""
                try:
                    error_body = e.response.text[:500]
                except Exception:
                    pass

                # Convert to specific error types based on status code
                status_code = e.response.status_code
                if status_code in (401, 403):
                    # Authentication errors (for authenticated Ollama servers)
                    raise ProviderAuthError(
                        message=f"Authentication failed: HTTP {status_code}: {error_body}",
                        provider=self.name,
                        status_code=status_code,
                        raw_error=e,
                    ) from e
                elif status_code == 429:
                    # Rate limit errors
                    raise ProviderRateLimitError(
                        message=f"Rate limit exceeded: {error_body}",
                        provider=self.name,
                        status_code=status_code,
                        raw_error=e,
                    ) from e
                else:
                    # Other HTTP errors
                    self._provider_logger.logger.error(
                        f"Ollama HTTP error {status_code}: {error_body}"
                    )
                    raise ProviderError(
                        message=f"HTTP error {status_code}: {error_body}",
                        provider=self.name,
                        status_code=status_code,
                        raw_error=e,
                    ) from e
            except Exception as e:
                # Skip if already a ProviderError to avoid double-wrapping
                if isinstance(e, ProviderError):
                    raise
                raise ProviderError(
                    message=f"Unexpected error: {str(e)}",
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
        """Stream chat completion from Ollama.

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
            ProviderAuthError: If authentication fails (for authenticated servers)
            ProviderRateLimitError: If rate limit is exceeded
            ProviderTimeoutError: If request times out
            ProviderError: For other errors
        """
        # Check if we've already learned this model doesn't support tools
        effective_tools = tools
        if model in self._models_without_tools and tools:
            self._provider_logger.logger.debug(
                f"Model {model} cached as not supporting tools, skipping tools"
            )
            effective_tools = None

        async for chunk in self._stream_impl(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=effective_tools,
            retry_without_tools=(tools is not None and effective_tools is not None),
            **kwargs,
        ):
            yield chunk

    async def _stream_impl(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        retry_without_tools: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Internal stream implementation with retry logic."""
        try:
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
            self._provider_logger.logger.debug(
                f"Streaming request: model={model}, msgs={len(messages)}, tools={num_tools}"
            )
            # Debug: log first tool for inspection if tools are provided
            if tools and num_tools > 0:
                self._provider_logger.logger.debug(
                    f"First tool schema sample: {payload.get('tools', [{}])[0]}"
                )

            # Log the endpoint URL being used for connection
            endpoint_url = f"{self.base_url}/api/chat"
            self._provider_logger.logger.debug(
                f"Connecting to Ollama endpoint: {endpoint_url}"
            )

            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                # Check for HTTP 400 "does not support tools" error
                if response.status_code == 400 and tools and retry_without_tools:
                    error_body = await response.aread()
                    error_text = error_body.decode()
                    self._provider_logger.logger.debug(
                        f"Ollama error response (400): {error_text}"
                    )

                    if "does not support tools" in error_text.lower():
                        self._provider_logger.logger.warning(
                            f"Model {model} doesn't support tools via Ollama API. "
                            "Retrying stream without tools (will use fallback parsing)."
                        )
                        # Cache that this model doesn't support tools
                        self._models_without_tools.add(model)
                        # Retry without tools - use recursive call
                        async for chunk in self._stream_impl(
                            messages=messages,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=None,  # No tools
                            retry_without_tools=False,  # Don't retry again
                            **kwargs,
                        ):
                            yield chunk
                        return

                if response.status_code >= 400:
                    error_body = await response.aread()
                    self._provider_logger.logger.error(
                        f"Ollama error response ({response.status_code}): {error_body.decode()}"
                    )
                response.raise_for_status()

                line_count = 0
                async for line in response.aiter_lines():
                    line_count += 1

                    if not line.strip():
                        continue

                    try:
                        chunk_data = json.loads(line)
                        chunk = self._parse_stream_chunk(chunk_data)
                        yield chunk

                        if chunk.is_final:
                            self._provider_logger.logger.debug(
                                f"Received final chunk after {line_count} lines"
                            )
                            break
                    except json.JSONDecodeError:
                        self._provider_logger.logger.warning(
                            f"JSON decode error on line: {line[:100]}"
                        )

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            # Include error body for better debugging
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass

            # Convert to specific error types based on status code
            status_code = e.response.status_code
            if status_code in (401, 403):
                # Authentication errors (for authenticated Ollama servers)
                raise ProviderAuthError(
                    message=f"Authentication failed: HTTP {status_code}: {error_body}",
                    provider=self.name,
                    status_code=status_code,
                    raw_error=e,
                ) from e
            elif status_code == 429:
                # Rate limit errors
                raise ProviderRateLimitError(
                    message=f"Rate limit exceeded: {error_body}",
                    provider=self.name,
                    status_code=status_code,
                    raw_error=e,
                ) from e
            else:
                # Other HTTP errors
                self._provider_logger.logger.error(
                    f"Ollama streaming HTTP error {status_code}: {error_body}"
                )
                raise ProviderError(
                    message=f"HTTP error {status_code}: {error_body}",
                    provider=self.name,
                    status_code=status_code,
                    raw_error=e,
                ) from e
        except Exception as e:
            # Skip if already a ProviderError to avoid double-wrapping
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                message=f"Unexpected error in stream: {str(e)}",
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
        """Build request payload for Ollama API.

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
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Add tools if provided and model supports them
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

        # Merge additional options
        if "options" in kwargs:
            payload["options"].update(kwargs.pop("options"))

        payload.update(kwargs)
        return payload

    def _normalize_tool_calls(
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Normalize tool calls from Ollama's OpenAI-compatible format.

        Ollama returns tool calls in OpenAI format:
        {'id': '...', 'function': {'name': 'tool_name', 'arguments': {...}}}

        We need:
        {'name': 'tool_name', 'arguments': {...}}

        Args:
            tool_calls: Raw tool calls from Ollama

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
                normalized.append(
                    {"name": function.get("name"), "arguments": function.get("arguments", {})}
                )
            elif isinstance(call, dict) and "name" in call:
                # Already normalized
                normalized.append(call)
            else:
                # Unknown format, skip
                continue

        return normalized if normalized else None

    def _parse_json_tool_call_from_content(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls from JSON text in content (fallback for models without native support).

        Some Ollama models (like qwen2.5-coder, llama3.1) return tool calls as JSON in content
        instead of using the structured tool_calls field. This method detects and parses them.

        Supported formats:
        - {"name": "tool_name", "arguments": {...}}  (qwen format)
        - {"name": "tool_name", "parameters": {...}}  (llama format)

        Note: Must distinguish from task plan JSON which also has "name" but different structure.
        Tool calls have "arguments" or "parameters" which are dict objects, not strings.

        Args:
            content: Message content that might contain JSON tool call

        Returns:
            List of tool calls if detected, None otherwise
        """
        if not content or not content.strip():
            return None

        # Try to parse as JSON
        try:
            data = json.loads(content.strip())

            # Check if it looks like a tool call
            # Must have "name" AND ("arguments" OR "parameters")
            # AND the arguments/parameters must be a dict (not a string)
            if isinstance(data, dict) and "name" in data:
                # Handle both "arguments" and "parameters" keys
                arguments = data.get("arguments") or data.get("parameters")

                # Only treat as tool call if arguments is a dict (actual tool call)
                # This distinguishes from planning JSON where "name" is a task name string
                if isinstance(arguments, dict):
                    # Additional validation: tool calls typically have specific structure
                    # Arguments usually contain things like "query", "path", "code", etc.
                    # NOT "complexity", "steps", "desc" which are planning fields
                    planning_keywords = {"complexity", "steps", "desc", "duration"}
                    if not any(key in arguments for key in planning_keywords):
                        # Convert to normalized format
                        return [{"name": data.get("name"), "arguments": arguments}]
        except (json.JSONDecodeError, ValueError):
            # Not JSON or invalid format
            pass

        return None

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse Ollama API response.

        Args:
            result: Raw API response
            model: Model name

        Returns:
            Normalized CompletionResponse
        """
        message = result.get("message", {})
        content = message.get("content", "")
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        # Fallback: Check if content contains JSON tool call (for models without native support)
        if not tool_calls and content:
            parsed_tool_calls = self._parse_json_tool_call_from_content(content)
            if parsed_tool_calls:
                self._provider_logger.logger.debug(
                    f"Parsed tool call from content (fallback for model: {model})"
                )
                tool_calls = parsed_tool_calls
                # Clear content since it was a tool call, not actual text response
                content = ""

        # Parse usage stats if available
        usage = None
        if "prompt_eval_count" in result or "eval_count" in result:
            usage = {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
            }

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=result.get("done_reason"),
            usage=usage,
            model=model,
            raw_response=result,
        )

    def _parse_stream_chunk(self, chunk_data: Dict[str, Any]) -> StreamChunk:
        """Parse streaming chunk from Ollama.

        Args:
            chunk_data: Raw chunk data

        Returns:
            Normalized StreamChunk
        """
        message = chunk_data.get("message", {})
        content = message.get("content", "")
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))
        is_done = chunk_data.get("done", False)

        # Fallback: Check if this is a final chunk with JSON tool call in content
        # (for models without native tool calling support)
        if not tool_calls and content and is_done:
            parsed_tool_calls = self._parse_json_tool_call_from_content(content)
            if parsed_tool_calls:
                model_name = chunk_data.get("model", "unknown")
                self._provider_logger.logger.debug(
                    f"Parsed tool call from streaming content (fallback for model: {model_name})"
                )
                tool_calls = parsed_tool_calls
                # Clear content since it was a tool call, not actual text response
                content = ""

        return StreamChunk(
            content=content,
            tool_calls=tool_calls,
            stop_reason=chunk_data.get("done_reason") if is_done else None,
            is_final=is_done,
        )

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models on Ollama server.

        Returns:
            List of available models with metadata

        Raises:
            ProviderError: If request fails
        """
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            result = response.json()
            return result.get("models", [])
        except Exception as e:
            raise ProviderError(
                message=f"Failed to list models: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def pull_model(self, model: str) -> AsyncIterator[Dict[str, Any]]:
        """Pull a model from Ollama library.

        Args:
            model: Model name to pull

        Yields:
            Progress updates

        Raises:
            ProviderError: If pull fails
        """
        try:
            payload = {"name": model, "stream": True}

            async with self.client.stream("POST", "/api/pull", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            raise ProviderError(
                message=f"Failed to pull model: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
