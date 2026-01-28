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

"""llama.cpp server provider for CPU-friendly local inference.

llama.cpp provides efficient CPU inference with GGUF quantized models.
Exposes an OpenAI-compatible API endpoint.

Usage:
    Start llama.cpp server:
        # Using llama-server (recommended)
        llama-server -m model.gguf --port 8080

        # Or using llama-cpp-python
        python -m llama_cpp.server --model model.gguf --port 8080

    Connect with Victor:
        victor chat --provider llamacpp --model default
        victor chat --provider llamacpp --endpoint http://localhost:8080

Recommended GGUF models for coding (Q4_K_M quantization):
    1. qwen2.5-coder-7b-instruct.Q4_K_M.gguf (4.4GB)
    2. qwen2.5-coder-3b-instruct.Q4_K_M.gguf (2.0GB)
    3. codellama-7b-instruct.Q4_K_M.gguf (4.2GB)
    4. deepseek-coder-6.7b-instruct.Q4_K_M.gguf (4.0GB)
    5. starcoder2-7b.Q4_K_M.gguf (4.3GB)

Download models from: https://huggingface.co/models?sort=trending&search=gguf
"""

import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderConnectionError,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.error_handler import HTTPErrorHandlerMixin
from victor.providers.openai_compat import (
    convert_messages_to_openai_format,
    convert_tools_to_openai_format,
    parse_openai_tool_calls,
)

logger = logging.getLogger(__name__)

# Default llama.cpp server endpoints
DEFAULT_LLAMACPP_URLS = [
    "http://localhost:8080/v1",
    "http://127.0.0.1:8080/v1",
    "http://localhost:8000/v1",  # Alternative port
]

# Models that support tool calling (instruction-tuned)
TOOL_CAPABLE_PATTERNS = [
    "instruct",
    "chat",
    "coder",
    "-it",
    "qwen",
    "llama-3",
    "mistral",
    "deepseek",
]


def _model_supports_tools(model: str) -> bool:
    """Check if model likely supports tool calling.

    Args:
        model: Model name/path

    Returns:
        True if model likely supports tools
    """
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in TOOL_CAPABLE_PATTERNS)


def _extract_thinking_content(response: str) -> Tuple[str, str]:
    """Extract thinking content from response.

    Args:
        response: Raw response text

    Returns:
        Tuple of (thinking_content, main_content)
    """
    think_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)
    thinking = "\n".join(matches) if matches else ""
    content = re.sub(think_pattern, "", response, flags=re.DOTALL | re.IGNORECASE).strip()
    return (thinking, content)


def _extract_tool_calls_from_content(content: str) -> Tuple[List[Dict[str, Any]], str]:
    """Extract tool calls from content when server doesn't parse them.

    Handles cases where model outputs tool calls as JSON text.
    Common formats:
    - ```json\n{...}\n```
    - <TOOL_OUTPUT>{...}</TOOL_OUTPUT>
    - {"name": "...", "arguments": {...}}

    Args:
        content: Response content that may contain tool calls

    Returns:
        Tuple of (parsed_tool_calls, remaining_content)
    """
    tool_calls: List[Dict[str, Any]] = []
    remaining = content

    # Pattern 1: JSON code block with tool call
    json_block_pattern = r"```json\s*\n?\s*(\{[^`]*\"name\"\s*:\s*\"[^\"]+\"[^`]*\})\s*\n?```"
    matches = re.findall(json_block_pattern, content, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if "name" in data:
                tool_calls.append(
                    {
                        "id": f"fallback_{len(tool_calls)}",
                        "name": data.get("name", ""),
                        "arguments": data.get("arguments", {}),
                    }
                )
                remaining = remaining.replace(f"```json\n{match}\n```", "").strip()
                remaining = remaining.replace(f"```json{match}```", "").strip()
        except json.JSONDecodeError:
            pass

    # Pattern 2: <TOOL_OUTPUT> tags
    tool_output_pattern = r"<TOOL_OUTPUT>\s*(\{.*?\})\s*</TOOL_OUTPUT>"
    matches = re.findall(tool_output_pattern, content, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            if "name" in data:
                tool_calls.append(
                    {
                        "id": f"fallback_{len(tool_calls)}",
                        "name": data.get("name", ""),
                        "arguments": data.get("arguments", {}),
                    }
                )
                remaining = re.sub(
                    r"<TOOL_OUTPUT>\s*" + re.escape(match) + r"\s*</TOOL_OUTPUT>", "", remaining
                )
        except json.JSONDecodeError:
            pass

    # Pattern 3: Inline JSON (for simple cases)
    if not tool_calls and content.strip().startswith("{") and "name" in content:
        try:
            data = json.loads(content.strip())
            if "name" in data:
                tool_calls.append(
                    {
                        "id": "fallback_0",
                        "name": data.get("name", ""),
                        "arguments": data.get("arguments", {}),
                    }
                )
                remaining = ""
        except json.JSONDecodeError:
            pass

    return tool_calls, remaining.strip()


class LlamaCppProvider(BaseProvider, HTTPErrorHandlerMixin):
    """Provider for llama.cpp server (CPU-optimized inference).

    Features:
        - OpenAI-compatible API
        - GGUF quantized model support
        - Efficient CPU inference
        - Low memory footprint with quantization
        - Cross-platform (macOS, Linux, Windows)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "not-needed",  # llama.cpp doesn't require auth
        timeout: int = 300,  # Longer timeout for CPU inference
        max_retries: int = 2,
        **kwargs: Any,
    ):
        """Initialize llama.cpp provider.

        Args:
            base_url: llama.cpp server URL (default: http://localhost:8080/v1)
            api_key: API key (not required for llama.cpp)
            timeout: Request timeout in seconds (default: 300 for CPU)
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, **kwargs
        )
        self.base_url = base_url or DEFAULT_LLAMACPP_URLS[0]
        self.timeout = timeout

        # Remove trailing /v1 if present (we'll add it in requests)
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers={"Content-Type": "application/json"},
        )
        self._loaded_model: Optional[str] = None

    @classmethod
    async def create(cls, **kwargs: Any) -> "LlamaCppProvider":
        """Factory method to create and initialize provider.

        Args:
            **kwargs: Provider configuration

        Returns:
            Initialized LlamaCppProvider instance

        Raises:
            ProviderConnectionError: If server is not reachable
        """
        provider = cls(**kwargs)

        # Try to connect and verify server is running
        base_url = kwargs.get("base_url")
        urls_to_try = [base_url] if base_url else DEFAULT_LLAMACPP_URLS

        for url in urls_to_try:
            if url.endswith("/v1"):
                url = url[:-3]
            try:
                # llama.cpp uses /health endpoint
                response = await provider.client.get(f"{url}/health", timeout=10.0)
                if response.status_code == 200:
                    provider.base_url = url
                    # Try to get loaded model info
                    try:
                        models_resp = await provider.client.get(f"{url}/v1/models", timeout=5.0)
                        if models_resp.status_code == 200:
                            data = models_resp.json()
                            if data.get("data"):
                                provider._loaded_model = data["data"][0].get("id", "default")
                    except Exception:
                        provider._loaded_model = "default"
                    logger.info(f"Connected to llama.cpp server at {url}")
                    return provider
            except Exception as e:
                logger.debug(f"llama.cpp not available at {url}: {e}")
                continue

        raise ProviderConnectionError(
            message="Cannot connect to llama.cpp server",
            provider="llamacpp",
            details={
                "tried_urls": urls_to_try,
                "suggestion": (
                    "Start llama.cpp server with:\n"
                    "  llama-server -m model.gguf --port 8080\n\n"
                    "Or using llama-cpp-python:\n"
                    "  pip install llama-cpp-python[server]\n"
                    "  python -m llama_cpp.server --model model.gguf"
                ),
            },
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "llamacpp"

    def supports_tools(self) -> bool:
        """llama.cpp supports tools with compatible models."""
        return True

    def supports_streaming(self) -> bool:
        """llama.cpp supports streaming."""
        return True

    async def list_models(self) -> List[Dict[str, Any]]:
        """List models loaded in llama.cpp server.

        Returns:
            List of model information dictionaries
        """
        try:
            response = await self.client.get(f"{self.base_url}/v1/models", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return []
        except Exception as e:
            logger.warning(f"Failed to list llama.cpp models: {e}")
            return []

    async def check_health(self) -> Dict[str, Any]:
        """Check llama.cpp server health.

        Returns:
            Health status dictionary
        """
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            if response.status_code == 200:
                try:
                    return response.json()
                except Exception:
                    return {"status": "ok"}
            return {"status": "error", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_server_props(self) -> Dict[str, Any]:
        """Get llama.cpp server properties.

        Returns:
            Server properties including model info
        """
        try:
            response = await self.client.get(f"{self.base_url}/props", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request to llama.cpp server.

        Args:
            messages: Conversation messages
            model: Model name (usually "default" for llama.cpp)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            **kwargs: Additional parameters (top_p, top_k, repeat_penalty, etc.)

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        # Use loaded model if none specified
        if model == "default" and self._loaded_model:
            model = self._loaded_model

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": convert_messages_to_openai_format(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add optional parameters (llama.cpp specific)
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "repeat_penalty" in kwargs:
            payload["repeat_penalty"] = kwargs["repeat_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]

        # Add tools if provided
        if tools and _model_supports_tools(model):
            payload["tools"] = convert_tools_to_openai_format(tools)
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )

            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get("message") or response.text
                raise ProviderError(
                    message=f"llama.cpp API error: {error_msg}",
                    provider="llamacpp",
                    details={"status_code": response.status_code, "response": error_data},
                )

            result = response.json()
            return self._parse_response(result, model)

        except httpx.ConnectError as e:
            raise self._handle_error(e, self.name)
        except httpx.TimeoutException as e:
            raise self._handle_error(e, self.name)
        except Exception as e:
            raise self._handle_error(e, self.name)

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from llama.cpp server.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional parameters

        Yields:
            StreamChunk with content or tool calls
        """
        if model == "default" and self._loaded_model:
            model = self._loaded_model

        payload: Dict[str, Any] = {
            "model": model,
            "messages": convert_messages_to_openai_format(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]

        if tools and _model_supports_tools(model):
            payload["tools"] = convert_tools_to_openai_format(tools)
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        accumulated_content = ""
        accumulated_tool_calls: List[Dict[str, Any]] = []

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_text = ""
                    async for chunk in response.aiter_text():
                        error_text += chunk
                    raise ProviderError(
                        message=f"llama.cpp streaming error: {error_text}",
                        provider="llamacpp",
                        details={"status_code": response.status_code},
                    )

                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            choices = data.get("choices", [])
                            if not choices:
                                continue

                            choice = choices[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")

                            # Handle content
                            content = delta.get("content")
                            if content:
                                accumulated_content += content
                                yield StreamChunk(
                                    content=content,
                                    is_final=False,
                                    model_name=model,
                                )

                            # Handle tool calls
                            tool_call_delta = delta.get("tool_calls")
                            if tool_call_delta:
                                for tc in tool_call_delta:
                                    idx = tc.get("index", 0)
                                    while len(accumulated_tool_calls) <= idx:
                                        accumulated_tool_calls.append(
                                            {"id": "", "name": "", "arguments": ""}
                                        )
                                    if "id" in tc:
                                        accumulated_tool_calls[idx]["id"] = tc["id"]
                                    func = tc.get("function", {})
                                    if "name" in func:
                                        accumulated_tool_calls[idx]["name"] = func["name"]
                                    if "arguments" in func:
                                        accumulated_tool_calls[idx]["arguments"] += func[
                                            "arguments"
                                        ]

                            # Final chunk
                            if finish_reason:
                                # Check for thinking tags
                                final_content = accumulated_content
                                thinking, main_content = _extract_thinking_content(final_content)
                                if thinking:
                                    final_content = main_content

                                # Parse accumulated tool calls
                                parsed_tool_calls = None
                                if accumulated_tool_calls:
                                    parsed_tool_calls = []
                                    for tc in accumulated_tool_calls:
                                        try:
                                            args = (
                                                json.loads(tc["arguments"])
                                                if tc["arguments"]
                                                else {}
                                            )
                                        except json.JSONDecodeError:
                                            args = {"raw": tc["arguments"]}
                                        parsed_tool_calls.append(
                                            {
                                                "id": tc["id"],
                                                "name": tc["name"],
                                                "arguments": args,
                                            }
                                        )

                                # Fallback: Extract tool calls from content
                                if not parsed_tool_calls and final_content:
                                    fallback_calls, remaining = _extract_tool_calls_from_content(
                                        final_content
                                    )
                                    if fallback_calls:
                                        parsed_tool_calls = fallback_calls
                                        final_content = remaining
                                        logger.debug(
                                            f"llama.cpp stream: Extracted {len(fallback_calls)} "
                                            "tool call(s) from content"
                                        )

                                yield StreamChunk(
                                    content=final_content,
                                    is_final=True,
                                    model_name=model,
                                    tool_calls=parsed_tool_calls,
                                    stop_reason=finish_reason,
                                )

                        except json.JSONDecodeError:
                            logger.debug(f"Failed to parse streaming chunk: {line}")
                            continue

        except httpx.TimeoutException as e:
            raise self._handle_error(e, self.name)
        except Exception as e:
            raise self._handle_error(e, self.name)

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        """Parse llama.cpp API response.

        Args:
            result: Raw API response
            model: Model name

        Returns:
            Parsed CompletionResponse
        """
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="",
                role="assistant",
                model=model,
                tool_calls=None,
                stop_reason=None,
                usage=None,
                raw_response=result,
                metadata=None,
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        tool_calls_data = message.get("tool_calls")

        # Extract thinking content if present
        metadata = {}
        if content:
            thinking, main_content = _extract_thinking_content(content)
            if thinking:
                metadata["reasoning_content"] = thinking
                content = main_content

        # Parse tool calls (native or fallback)
        tool_calls = parse_openai_tool_calls(tool_calls_data)

        # Fallback: Extract tool calls from content if server didn't parse them
        if not tool_calls and content:
            fallback_calls, remaining_content = _extract_tool_calls_from_content(content)
            if fallback_calls:
                tool_calls = fallback_calls
                content = remaining_content
                logger.debug(
                    f"llama.cpp: Extracted {len(tool_calls)} tool call(s) from content using fallback"
                )

        # Get usage info
        usage_data = result.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }

        return CompletionResponse(
            content=content,
            role="assistant",
            model=model,
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            raw_response=result,
            metadata=metadata if metadata else None,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
