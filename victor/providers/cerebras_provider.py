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

"""Cerebras Inference API provider for ultra-fast LLM inference.

Cerebras provides extremely fast inference using their Wafer Scale Engine (WSE)
hardware with a generous free tier.

Free Tier:
- 30-10 requests/min (varies by model)
- 60,000-150,000 tokens/min
- Access to Llama, Qwen, and GPT-OSS models

Features:
- Fastest inference available (1000+ tokens/sec)
- OpenAI-compatible API
- Tool calling support
- Free tier with no credit card

References:
- https://inference-docs.cerebras.ai/
- https://inference-docs.cerebras.ai/api-reference
"""

import asyncio
import json
import logging
import os
import re
import ssl
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

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


@dataclass
class StreamingThinkingFilter:
    """Filter for detecting and separating thinking content during streaming.

    For Qwen-3 models that output reasoning inline, this filter:
    1. Buffers initial content until we have enough to analyze
    2. Detects thinking patterns at the start
    3. Tracks transition from thinking to main content
    4. Emits only main content, storing thinking in metadata
    """

    model: str
    buffer_size: int = 150  # Buffer chars before analyzing
    max_thinking_buffer: int = 2000  # Max chars to buffer before giving up
    _buffer: str = field(default="", init=False)
    _in_thinking: bool = field(default=True, init=False)  # Start assuming thinking
    _thinking_content: str = field(default="", init=False)
    _analyzed: bool = field(default=False, init=False)
    _emit_queue: List[str] = field(default_factory=list, init=False)
    _gave_up: bool = field(default=False, init=False)  # If we gave up filtering

    def process_chunk(self, content: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process a streaming chunk, filtering thinking content.

        Args:
            content: Incoming content chunk

        Returns:
            Tuple of (content_to_emit, metadata_update)
        """
        if not _is_thinking_model(self.model):
            # Not a thinking model, pass through directly
            return content, None

        # If we gave up filtering, pass through directly
        if self._gave_up:
            return content, None

        self._buffer += content

        # Haven't analyzed enough content yet
        if not self._analyzed:
            # Wait for buffer to fill or newline
            if len(self._buffer) < self.buffer_size and "\n" not in self._buffer:
                return "", None  # Hold content until we have enough

            # Analyze buffered content
            self._analyze_buffer()
            self._analyzed = True

        # If we're in thinking mode, accumulate but don't emit
        if self._in_thinking:
            # Check if we've transitioned to main content
            self._check_transition()

            if self._in_thinking:
                # Check if buffer is too large - give up and emit everything
                if len(self._buffer) > self.max_thinking_buffer:
                    logger.debug(
                        f"Cerebras streaming: Buffer exceeded {self.max_thinking_buffer} chars, "
                        "giving up on thinking detection"
                    )
                    self._gave_up = True
                    emit_content = self._buffer
                    self._buffer = ""
                    return emit_content, None

                # Still in thinking mode - store and don't emit
                self._thinking_content = self._buffer
                return "", None

        # We're in main content mode - emit pending content
        emit_content = self._buffer
        self._buffer = ""

        # Return accumulated thinking in metadata only on first main content emission
        metadata = None
        if self._thinking_content:
            metadata = {"reasoning_content": self._thinking_content}
            self._thinking_content = ""  # Clear after emitting metadata

        return emit_content, metadata

    def _analyze_buffer(self) -> None:
        """Analyze buffer to determine if it starts with thinking content."""
        content = self._buffer.strip()
        if not content:
            self._in_thinking = False
            return

        # Check if content starts with thinking patterns
        for pattern in QWEN3_THINKING_PATTERNS:
            if re.match(pattern, content, re.IGNORECASE):
                self._in_thinking = True
                logger.debug("Cerebras streaming: Detected thinking pattern at start")
                return

        # No thinking pattern detected - treat as main content
        self._in_thinking = False

    def _check_transition(self) -> None:
        """Check if buffer has transitioned from thinking to main content."""
        # Look for double newline or significant structure change
        lines = self._buffer.split("\n")
        if len(lines) < 2:
            return

        # Check for empty line (paragraph break) indicating transition
        for i, line in enumerate(lines):
            if not line.strip() and i > 0:
                # Found paragraph break - check if next content looks like main response
                remaining = "\n".join(lines[i + 1 :]).strip()
                if remaining and self._looks_like_main_content(remaining[:150]):
                    # Transition detected - split content
                    self._thinking_content = "\n".join(lines[: i + 1])
                    self._buffer = remaining
                    self._in_thinking = False
                    logger.debug(
                        f"Cerebras streaming: Transitioned from thinking "
                        f"({len(self._thinking_content)} chars) to main content"
                    )
                    return

    def _looks_like_main_content(self, text: str) -> bool:
        """Check if text looks like main response content (not thinking)."""
        text_stripped = text.strip()

        # Patterns that indicate main content (answers, explanations)
        main_content_patterns = [
            r"^(Here(?:'s| is)|The answer|Based on|In summary|To summarize)",
            r"^(According to|Looking at|From the|The result)",
            r"^(This (?:is|means|indicates|shows|returns))",
            r"^(The (?:code|function|file|output|response))",
            r"^(Yes|No|True|False)[,.\s]",
            r"^(```|def |class |import |from )",  # Code blocks
            r"^\d+\.?\s",  # Numbered lists
            r"^[-*]\s",  # Bullet lists
            r"^#+\s",  # Markdown headers
        ]

        for pattern in main_content_patterns:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return True

        # Check if it doesn't look like thinking
        return not self._looks_like_thinking(text_stripped)

    def _looks_like_thinking(self, text: str) -> bool:
        """Check if text looks like thinking/reasoning content."""
        for pattern in QWEN3_THINKING_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        # Also check continuation patterns
        if text.startswith(
            (
                "Although",
                "Since",
                "Because",
                "However",
                "But wait",
                "And then",
                "So I",
                "This means",
                "That would",
                "Maybe",
                "Perhaps",
                "I should",
                "I need to",
                "Let me",
                "Wait,",
                "Actually,",
            )
        ):
            return True
        return False

    def finalize(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Finalize and return any remaining content.

        Returns:
            Tuple of (remaining_content, final_metadata)
        """
        if not _is_thinking_model(self.model):
            return self._buffer, None

        # If we never transitioned out of thinking, all content is main
        if self._in_thinking and self._buffer:
            # Everything was thinking-like but we should still emit it
            thinking, main = _extract_qwen3_thinking(self._buffer)
            metadata = {"reasoning_content": thinking} if thinking else None
            self._buffer = ""
            return main, metadata

        # Return any remaining buffered content
        content = self._buffer
        self._buffer = ""

        metadata = None
        if self._thinking_content:
            metadata = {"reasoning_content": self._thinking_content}
            self._thinking_content = ""

        return content, metadata


DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"

CEREBRAS_MODELS = {
    "llama3.3-70b": {
        "description": "Llama 3.3 70B - High quality, very fast",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "rate_limit": "30 req/min, 60K tokens/min",
    },
    "llama-3.3-70b": {
        "description": "Llama 3.3 70B (alias)",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
    },
    "llama3.1-8b": {
        "description": "Llama 3.1 8B - Fast, efficient",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "rate_limit": "30 req/min, 150K tokens/min",
    },
    "qwen-3-32b": {
        "description": "Qwen 3 32B - Strong reasoning with thinking",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "has_thinking": True,
    },
    "qwen-3-235b-a22b-instruct-2507": {
        "description": "Qwen 3 235B - Largest Qwen model",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
        "has_thinking": True,
    },
    "gpt-oss-120b": {
        "description": "GPT-OSS 120B - Largest open model, clean output",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
    },
    "zai-glm-4.6": {
        "description": "ZAI GLM 4.6 - Chinese/English bilingual",
        "context_window": 131072,
        "max_output": 8192,
        "supports_tools": True,
    },
}

# Models that output thinking/reasoning inline (need filtering)
THINKING_MODELS = {"qwen-3-32b", "qwen-3-235b-a22b-instruct-2507"}

# Patterns that indicate start of thinking content in Qwen-3 models
QWEN3_THINKING_PATTERNS = [
    r"^Okay,?\s+(?:the\s+)?(?:user\s+)?(?:is\s+)?(?:asking|wants?|needs?|said)",
    r"^(?:Let\s+me\s+)?(?:think|see|check|analyze|consider)",
    r"^(?:Hmm|Well|So),?\s+",
    r"^(?:First|Now),?\s+(?:I\s+)?(?:need|should|will|'ll)",
    r"^(?:The\s+)?(?:user|question|task|request)\s+(?:is|asks?|wants?)",
    r"^(?:I\s+)?(?:need|should|will|'ll)\s+(?:first|start|begin)",
]


_MAX_TRANSIENT_RETRIES = 2
_TRANSIENT_BACKOFF_BASE = 1.0  # seconds


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient network error worth retrying.

    Covers SSL record errors, connection resets, and other network-level
    failures that are typically caused by infrastructure flakiness rather
    than bad requests.
    """
    if isinstance(exc, ssl.SSLError):
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return True
    # httpx wraps low-level errors — check the cause chain
    cause = exc.__cause__
    while cause is not None:
        if isinstance(cause, ssl.SSLError):
            return True
        if isinstance(cause, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return True
        cause = getattr(cause, "__cause__", None)
    return False


def _is_thinking_model(model: str) -> bool:
    """Check if model outputs inline thinking content."""
    model_lower = model.lower()
    return any(pattern in model_lower for pattern in ["qwen-3", "qwen3"])


def _extract_qwen3_thinking(content: str) -> Tuple[str, str]:
    """Extract thinking content from Qwen-3 model output.

    Qwen-3 models output their reasoning inline without special tags.
    This function detects and separates thinking from the actual response.

    Args:
        content: Raw response content

    Returns:
        Tuple of (thinking_content, main_content)
    """
    if not content:
        return "", ""

    lines = content.split("\n")
    thinking_lines = []
    main_lines = []
    in_thinking = True

    for line in lines:
        stripped = line.strip()

        # Check if this line looks like thinking
        is_thinking_line = False
        if in_thinking and stripped:
            for pattern in QWEN3_THINKING_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE):
                    is_thinking_line = True
                    break

            # Also check for continuation of thinking
            if not is_thinking_line and thinking_lines:
                # If previous was thinking and this continues the thought
                if stripped.startswith(
                    (
                        "Although",
                        "Since",
                        "Because",
                        "However",
                        "But",
                        "And",
                        "Or",
                        "So",
                        "This",
                        "That",
                        "These",
                        "Those",
                        "None",
                        "The",
                    )
                ):
                    is_thinking_line = True

        if is_thinking_line:
            thinking_lines.append(line)
        else:
            # Once we hit non-thinking content, stop looking for thinking
            in_thinking = False
            main_lines.append(line)

    thinking = "\n".join(thinking_lines).strip()
    main = "\n".join(main_lines).strip()

    # If everything was classified as thinking, return it as main content
    if not main and thinking:
        return "", thinking

    return thinking, main


class CerebrasProvider(BaseProvider):
    """Provider for Cerebras Inference API (OpenAI-compatible).

    Features:
    - Fastest inference available (WSE hardware)
    - Generous free tier
    - Native tool calling support
    - OpenAI-compatible API
    """

    DEFAULT_TIMEOUT = 60  # Cerebras is very fast

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        """Initialize Cerebras provider.

        Args:
            api_key: Cerebras API key (or set CEREBRAS_API_KEY env var)
            base_url: API endpoint
            timeout: Request timeout
            **kwargs: Additional configuration
        """
        # Try provided key, then env var, then keyring/api_keys.yaml
        self._api_key = api_key or os.environ.get("CEREBRAS_API_KEY", "")
        if not self._api_key:
            try:
                from victor.config.api_keys import get_api_key

                self._api_key = get_api_key("cerebras") or ""
            except ImportError:
                pass
        if not self._api_key:
            logger.warning(
                "Cerebras API key not provided. Set CEREBRAS_API_KEY environment variable "
                "or add to keyring with: victor keys set cerebras"
            )

        super().__init__(base_url=base_url, timeout=timeout, **kwargs)

        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        return "cerebras"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

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
        """Send chat completion request to Cerebras."""
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            try:
                payload = self._build_request_payload(
                    messages, model, temperature, max_tokens, tools, False, **kwargs
                )

                response = await self._execute_with_circuit_breaker(
                    self.client.post, "/chat/completions", json=payload
                )
                response.raise_for_status()

                return self._parse_response(response.json(), model)

            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(
                    message=f"Cerebras request timed out after {self.timeout}s",
                    provider=self.name,
                ) from e
            except httpx.HTTPStatusError as e:
                error_body = e.response.text[:500] if e.response.text else ""
                raise ProviderError(
                    message=f"Cerebras HTTP error {e.response.status_code}: {error_body}",
                    provider=self.name,
                    status_code=e.response.status_code,
                ) from e
            except Exception as e:
                if _is_transient_error(e) and attempt < _MAX_TRANSIENT_RETRIES:
                    last_exc = e
                    wait = _TRANSIENT_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        f"Cerebras transient error (attempt {attempt + 1}/"
                        f"{_MAX_TRANSIENT_RETRIES + 1}), retrying in {wait:.1f}s: {e}"
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        # Should not reach here, but satisfy type checker
        raise last_exc  # type: ignore[misc]

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
        """Stream chat completion from Cerebras with thinking content filtering."""
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            try:
                async for chunk in self._stream_inner(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    **kwargs,
                ):
                    yield chunk
                return  # Successful stream, exit retry loop

            except httpx.TimeoutException as e:
                raise ProviderTimeoutError(
                    message="Cerebras stream timed out",
                    provider=self.name,
                ) from e
            except httpx.HTTPStatusError as e:
                raise ProviderError(
                    message=f"Cerebras streaming error {e.response.status_code}",
                    provider=self.name,
                    status_code=e.response.status_code,
                ) from e
            except Exception as e:
                if _is_transient_error(e) and attempt < _MAX_TRANSIENT_RETRIES:
                    last_exc = e
                    wait = _TRANSIENT_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        f"Cerebras stream transient error (attempt {attempt + 1}/"
                        f"{_MAX_TRANSIENT_RETRIES + 1}), retrying in {wait:.1f}s: {e}"
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

        # Should not reach here, but satisfy type checker
        raise last_exc  # type: ignore[misc]

    async def _stream_inner(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Inner streaming implementation (no retry logic)."""
        payload = self._build_request_payload(
            messages, model, temperature, max_tokens, tools, True, **kwargs
        )

        # Initialize thinking filter for Qwen-3 models
        thinking_filter = StreamingThinkingFilter(model=model)
        accumulated_reasoning: str = ""

        async with self.client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            accumulated_tool_calls: List[Dict[str, Any]] = []

            async for line in response.aiter_lines():
                if not line.strip() or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    # Finalize any buffered content
                    remaining_content, final_metadata = thinking_filter.finalize()
                    if final_metadata and "reasoning_content" in final_metadata:
                        accumulated_reasoning = final_metadata["reasoning_content"]

                    # Emit final chunk with remaining content and metadata
                    yield StreamChunk(
                        content=remaining_content,
                        tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                        stop_reason="stop",
                        is_final=True,
                        metadata=(
                            {"reasoning_content": accumulated_reasoning}
                            if accumulated_reasoning
                            else None
                        ),
                    )
                    break

                try:
                    chunk_data = json.loads(data_str)
                    raw_chunk = self._parse_stream_chunk(chunk_data, accumulated_tool_calls)

                    # Filter thinking content
                    if raw_chunk.content:
                        filtered_content, filter_metadata = thinking_filter.process_chunk(
                            raw_chunk.content
                        )
                        if filter_metadata and "reasoning_content" in filter_metadata:
                            accumulated_reasoning = filter_metadata["reasoning_content"]

                        # Only yield if there's content to emit
                        if filtered_content or raw_chunk.tool_calls:
                            yield StreamChunk(
                                content=filtered_content,
                                tool_calls=raw_chunk.tool_calls,
                                stop_reason=raw_chunk.stop_reason,
                                is_final=raw_chunk.is_final,
                                metadata=filter_metadata,
                            )
                    elif raw_chunk.tool_calls or raw_chunk.is_final:
                        # Yield tool calls or final markers
                        yield raw_chunk

                except json.JSONDecodeError:
                    pass

    def _build_request_payload(
        self, messages, model, temperature, max_tokens, tools, stream, **kwargs
    ) -> Dict[str, Any]:
        formatted_messages = []
        for msg in messages:
            formatted_msg = {"role": msg.role, "content": msg.content}
            if msg.role == "tool" and hasattr(msg, "tool_call_id"):
                formatted_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                formatted_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": (
                                json.dumps(tc.get("arguments", {}))
                                if isinstance(tc.get("arguments"), dict)
                                else tc.get("arguments", "{}")
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            formatted_messages.append(formatted_msg)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

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
            payload["tool_choice"] = "auto"

        return payload

    def _parse_response(self, result: Dict[str, Any], model: str) -> CompletionResponse:
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="", role="assistant", model=model, raw_response=result
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        # Extract thinking content for Qwen-3 models
        metadata = {}
        if content and _is_thinking_model(model):
            thinking, main_content = _extract_qwen3_thinking(content)
            if thinking:
                metadata["reasoning_content"] = thinking
                content = main_content
                logger.debug(f"Cerebras: Extracted {len(thinking)} chars of thinking content")

        usage = None
        if usage_data := result.get("usage"):
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }
            # Cerebras provides timing info
            if "time_info" in result:
                time_info = result["time_info"]
                usage["total_time_ms"] = int(time_info.get("total_time", 0) * 1000)

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
            metadata=metadata if metadata else None,
        )

    def _normalize_tool_calls(self, tool_calls) -> Optional[List[Dict[str, Any]]]:
        if not tool_calls:
            return None
        normalized = []
        for call in tool_calls:
            if "function" in call:
                func = call["function"]
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                normalized.append(
                    {
                        "id": call.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": args,
                    }
                )
        return normalized if normalized else None

    def _parse_stream_chunk(self, chunk_data, accumulated_tool_calls) -> StreamChunk:
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamChunk(content="", is_final=False)

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({"id": "", "name": "", "arguments": ""})
            if "id" in tc_delta:
                accumulated_tool_calls[idx]["id"] = tc_delta["id"]
            if "function" in tc_delta:
                func = tc_delta["function"]
                if "name" in func:
                    accumulated_tool_calls[idx]["name"] = func["name"]
                if "arguments" in func:
                    accumulated_tool_calls[idx]["arguments"] += func["arguments"]

        final_tool_calls = None
        if finish_reason in ("tool_calls", "stop") and accumulated_tool_calls:
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args = tc.get("arguments", "{}")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        args = {}
                    final_tool_calls.append(
                        {
                            "id": tc.get("id", ""),
                            "name": tc["name"],
                            "arguments": args,
                        }
                    )

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
        )

    async def close(self) -> None:
        await self.client.aclose()
