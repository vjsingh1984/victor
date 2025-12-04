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

References:
- https://lmstudio.ai/docs/api/openai-api
- https://lmstudio.ai/docs/advanced/tool-use
"""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

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

logger = logging.getLogger(__name__)


class LMStudioProvider(BaseProvider):
    """Provider for LMStudio local model server (OpenAI-compatible API).

    Features:
    - Tiered URL selection with health probing
    - Async factory for non-blocking initialization
    - Native tool calling support for hammer-badge models
    - JSON fallback parsing for other models
    """

    # Default timeout matches Ollama (local models need more time)
    DEFAULT_TIMEOUT = 300

    # Default LMStudio port
    DEFAULT_PORT = 1234

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
        chosen_base = await cls._select_base_url_async(base_url, timeout)
        return cls(
            base_url=chosen_base,
            timeout=timeout,
            api_key=api_key,
            _skip_discovery=True,
            **kwargs,
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "lmstudio"

    def supports_tools(self) -> bool:
        """LMStudio supports tool calling for compatible models."""
        return True

    def supports_streaming(self) -> bool:
        """LMStudio supports streaming."""
        return True

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
                    else:
                        logger.info(f"LMStudio base URL selected: {url} (no models loaded)")
                    return url
            except Exception as exc:
                logger.warning(f"LMStudio endpoint {url} not reachable ({exc}); trying next.")

        fallback = base_url if isinstance(base_url, str) else str(base_url[0]) if base_url else "http://127.0.0.1:1234"
        logger.error(
            f"No LMStudio endpoints reachable from: {candidates}. Falling back to {fallback}"
        )
        return fallback

    @classmethod
    async def _select_base_url_async(
        cls, base_url: Union[str, List[str], None], timeout: int
    ) -> str:
        """Async version of _select_base_url for non-blocking endpoint discovery.

        Priority:
        1) LMSTUDIO_ENDPOINTS env var (comma-separated) if set
        2) Explicitly provided list (comma-separated) or URL
        3) Default localhost:1234
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
                    else:
                        logger.info(f"LMStudio base URL selected (async): {url} (no models loaded)")
                    return url
            except Exception as exc:
                logger.warning(f"LMStudio endpoint {url} not reachable ({exc}); trying next.")

        fallback = base_url if isinstance(base_url, str) else str(base_url[0]) if base_url else "http://127.0.0.1:1234"
        logger.error(
            f"No LMStudio endpoints reachable from: {candidates}. Falling back to {fallback}"
        )
        return fallback

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
            ProviderError: If request fails
        """
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

            # Make API call with circuit breaker protection
            response = await self._execute_with_circuit_breaker(
                self.client.post, "/chat/completions", json=payload
            )
            response.raise_for_status()

            result = response.json()
            return self._parse_response(result, model)

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"LMStudio request timed out after {self.timeout}s. "
                f"Ensure the model is loaded and server is responding.",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass
            raise ProviderError(
                message=f"LMStudio HTTP error {e.response.status_code}: {error_body}",
                provider=self.name,
                status_code=e.response.status_code,
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
            ProviderError: If request fails
        """
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
            logger.debug(
                f"LMStudio streaming request: model={model}, msgs={len(messages)}, tools={num_tools}"
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
                            # Final chunk
                            logger.debug(f"LMStudio stream complete after {line_count} lines")
                            yield StreamChunk(
                                content="",
                                tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                                stop_reason="stop",
                                is_final=True,
                            )
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            chunk = self._parse_stream_chunk(chunk_data, accumulated_tool_calls)
                            if chunk.content:
                                accumulated_content += chunk.content
                            yield chunk

                        except json.JSONDecodeError:
                            logger.warning(f"LMStudio JSON decode error on line: {line[:100]}")

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"LMStudio stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:500]
            except Exception:
                pass
            raise ProviderError(
                message=f"LMStudio streaming HTTP error {e.response.status_code}: {error_body}",
                provider=self.name,
                status_code=e.response.status_code,
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

        # Fallback: Check if content contains JSON tool call
        if not tool_calls and content:
            parsed_tool_calls = self._parse_json_tool_call_from_content(content)
            if parsed_tool_calls:
                logger.debug(f"LMStudio: Parsed tool call from content (fallback for model: {model})")
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
        )

    def _parse_stream_chunk(
        self, chunk_data: Dict[str, Any], accumulated_tool_calls: List[Dict[str, Any]]
    ) -> StreamChunk:
        """Parse streaming chunk from LMStudio (OpenAI-compatible SSE format).

        Args:
            chunk_data: Raw chunk data
            accumulated_tool_calls: List to accumulate tool call deltas

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
