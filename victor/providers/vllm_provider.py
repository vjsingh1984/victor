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

"""vLLM high-throughput inference server provider.

vLLM provides high-throughput LLM serving with PagedAttention for
efficient memory management. Exposes OpenAI-compatible API endpoints.

Usage:
    Start vLLM server:
        python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen2.5-Coder-7B-Instruct \
            --port 8000 \
            --enable-auto-tool-choice \
            --tool-call-parser hermes

    Connect with Victor:
        victor chat --provider vllm --model Qwen/Qwen2.5-Coder-7B-Instruct
        victor chat --provider vllm --endpoint http://remote-server:8000

Top tool-enabled coding models for vLLM (fp16/q8):
    1. Qwen/Qwen2.5-Coder-7B-Instruct (14GB fp16, 7GB q8)
    2. Qwen/Qwen2.5-Coder-14B-Instruct (28GB fp16, 14GB q8)
    3. deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (32GB fp16)
    4. codellama/CodeLlama-34b-Instruct-hf (68GB fp16)
    5. mistralai/Codestral-22B-v0.1 (44GB fp16)
"""

import json
import logging
import re
import typing
from typing import Any, Optional
from collections.abc import AsyncIterator

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

# Default vLLM endpoints to try
DEFAULT_VLLM_URLS = [
    "http://localhost:8000/v1",
    "http://127.0.0.1:8000/v1",
]

# Models that support tool calling natively
TOOL_CAPABLE_MODELS = [
    "qwen2.5-coder",
    "qwen2.5-instruct",
    "qwen3-coder",
    "llama-3.1",
    "llama-3.3",
    "deepseek-coder",
    "codestral",
    "hermes",
    "-tools",
    "-instruct",
]

# Models that may output thinking tags
THINKING_TAG_MODELS = [
    "qwen3",
    "deepseek-r1",
    "deepseek-reasoner",
]


def _model_supports_tools(model: str) -> bool:
    """Check if model likely supports tool calling.

    Args:
        model: Model name/path

    Returns:
        True if model likely supports tools
    """
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in TOOL_CAPABLE_MODELS)


def _model_uses_thinking_tags(model: str) -> bool:
    """Check if model outputs thinking tags.

    Args:
        model: Model name/path

    Returns:
        True if model uses thinking tags
    """
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in THINKING_TAG_MODELS)


def _extract_thinking_content(response: str) -> tuple[str, str]:
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


def _extract_tool_calls_from_content(content: str) -> tuple[list[dict[str, Any]], str]:
    """Extract tool calls from content when server doesn't parse them.

    Handles cases where vLLM wasn't started with --enable-auto-tool-choice.
    Models may output tool calls as JSON in various formats:
    - ```json\n{...}\n```
    - <TOOL_OUTPUT>{...}</TOOL_OUTPUT>
    - {"name": "...", "arguments": {...}}

    Args:
        content: Response content that may contain tool calls

    Returns:
        Tuple of (parsed_tool_calls, remaining_content)
    """
    tool_calls: list[dict[str, Any]] = []
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
            # Try to parse the entire content as JSON
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


class VLLMProvider(BaseProvider, HTTPErrorHandlerMixin):
    """Provider for vLLM high-throughput inference server.

    Features:
        - OpenAI-compatible API
        - PagedAttention for efficient memory
        - High-throughput batch inference
        - Tool calling with --enable-auto-tool-choice flag
        - Multiple quantization support (fp16, awq, gptq)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",  # vLLM doesn't require auth
        timeout: int = 300,  # Longer timeout for large models
        max_retries: int = 2,
        **kwargs: Any,
    ):
        """Initialize vLLM provider.

        Args:
            base_url: vLLM server URL (default: http://localhost:8000/v1)
            api_key: API key (vLLM typically doesn't require one)
            timeout: Request timeout in seconds (default: 300 for large models)
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, **kwargs
        )
        self.base_url = base_url or DEFAULT_VLLM_URLS[0]
        self.timeout = timeout

        # Remove trailing /v1 if present (we'll add it in requests)
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers={"Content-Type": "application/json"},
        )
        self._available_model: Optional[str] = None

    @classmethod
    async def create(cls, **kwargs: Any) -> "VLLMProvider":
        """Factory method to create and initialize provider.

        Args:
            **kwargs: Provider configuration

        Returns:
            Initialized VLLMProvider instance

        Raises:
            ProviderConnectionError: If server is not reachable
        """
        provider = cls(**kwargs)

        # Try to connect and verify server is running
        base_url = kwargs.get("base_url")
        urls_to_try = [base_url] if base_url else DEFAULT_VLLM_URLS

        for url in urls_to_try:
            if url.endswith("/v1"):
                url = url[:-3]
            try:
                response = await provider.client.get(f"{url}/v1/models", timeout=10.0)
                if response.status_code == 200:
                    provider.base_url = url
                    data = response.json()
                    if data.get("data"):
                        provider._available_model = data["data"][0].get("id")
                    logger.info(f"Connected to vLLM server at {url}")
                    return provider
            except Exception as e:
                logger.debug(f"vLLM not available at {url}: {e}")
                continue

        raise ProviderConnectionError(
            message="Cannot connect to vLLM server",
            provider="vllm",
            details={
                "tried_urls": urls_to_try,
                "suggestion": (
                    "Start vLLM server with:\n"
                    "  python -m vllm.entrypoints.openai.api_server \\\n"
                    "    --model Qwen/Qwen2.5-Coder-7B-Instruct \\\n"
                    "    --port 8000"
                ),
            },
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "vllm"

    def supports_tools(self) -> bool:
        """vLLM supports tools with --enable-auto-tool-choice flag."""
        return True

    def supports_streaming(self) -> bool:
        """vLLM supports streaming."""
        return True

    async def list_models(self) -> list[dict[str, Any]]:
        """List models loaded in vLLM server.

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
            logger.warning(f"Failed to list vLLM models: {e}")
            return []

    async def check_health(self) -> bool:
        """Check if vLLM server is healthy.

        Returns:
            True if server is healthy
        """
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request to vLLM server.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "Qwen/Qwen2.5-Coder-7B-Instruct")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            **kwargs: Additional vLLM parameters (top_p, top_k, etc.)

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        # Use available model if none specified
        if not model and self._available_model:
            model = self._available_model

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": convert_messages_to_openai_format(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "repetition_penalty" in kwargs:
            payload["repetition_penalty"] = kwargs["repetition_penalty"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        # Add tools if provided and model supports them
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
                error_msg = error_data.get("detail") or error_data.get("message") or response.text
                raise ProviderError(
                    message=f"vLLM API error: {error_msg}",
                    provider="vllm",
                    details={"status_code": response.status_code, "response": error_data},
                )

            result = response.json()
            return self._parse_response(result, model)

        except httpx.ConnectError as e:
            # Use mixin for connection errors
            raise self._handle_error(e, self.name)
        except httpx.TimeoutException as e:
            raise self._handle_error(e, self.name)
        except Exception as e:
            raise self._handle_error(e, self.name)

    @typing.no_type_check
    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from vLLM server.

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
        if not model and self._available_model:
            model = self._available_model

        payload: dict[str, Any] = {
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
        accumulated_tool_calls: list[dict[str, Any]] = []
        uses_thinking_tags = _model_uses_thinking_tags(model)

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
                        message=f"vLLM streaming error: {error_text}",
                        provider="vllm",
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
                                final_content = accumulated_content
                                if uses_thinking_tags and final_content:
                                    _, final_content = _extract_thinking_content(final_content)

                                # Parse accumulated tool calls
                                parsed_tool_calls: list[dict[str, Any]] = []
                                if accumulated_tool_calls:
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
                                            f"vLLM stream: Extracted {len(fallback_calls)} tool call(s) "
                                            "from content using fallback parser"
                                        )

                                yield StreamChunk(
                                    content=final_content,
                                    is_final=True,
                                    model_name=model,
                                    tool_calls=parsed_tool_calls,
                                )

                        except json.JSONDecodeError:
                            logger.debug(f"Failed to parse streaming chunk: {line}")
                            continue

        except httpx.TimeoutException as e:
            raise self._handle_error(e, self.name)
        except Exception as e:
            raise self._handle_error(e, self.name)

    def _parse_response(self, result: dict[str, Any], model: str) -> CompletionResponse:
        """Parse vLLM API response.

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
                model=model,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                tool_calls=None,
                stop_reason=None,
                raw_response=None,
                metadata=None,
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        tool_calls_data = message.get("tool_calls")

        # Extract thinking content if present
        metadata = {}
        if _model_uses_thinking_tags(model) and content:
            thinking, content = _extract_thinking_content(content)
            if thinking:
                metadata["reasoning_content"] = thinking

        # Parse tool calls (native or fallback)
        tool_calls = parse_openai_tool_calls(tool_calls_data)

        # Fallback: Extract tool calls from content if server didn't parse them
        # This happens when vLLM wasn't started with --enable-auto-tool-choice
        if not tool_calls and content:
            fallback_calls, remaining_content = _extract_tool_calls_from_content(content)
            if fallback_calls:
                tool_calls = fallback_calls
                content = remaining_content
                logger.debug(
                    f"vLLM: Extracted {len(tool_calls)} tool call(s) from content using fallback parser"
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
            model=model,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=choice.get("finish_reason"),
            raw_response=result,
            metadata=None,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
