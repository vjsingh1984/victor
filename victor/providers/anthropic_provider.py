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

"""Anthropic Claude provider implementation."""

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage

logger = logging.getLogger(__name__)

from victor.core.errors import (
    ProviderConnectionError as CoreProviderConnectionError,
    ProviderTimeoutError as CoreProviderTimeoutError,
    ValidationError,
)
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    StreamChunk,
    ToolDefinition,
)
from victor.providers.error_handler import HTTPErrorHandlerMixin
from victor.providers.openai_compat import convert_tools_to_anthropic_format


class AnthropicProvider(BaseProvider, HTTPErrorHandlerMixin):
    """Provider for Anthropic Claude models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var, or use keyring)
            base_url: Optional base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        # Resolve API key using centralized helper
        resolved_key = self._resolve_api_key(api_key, "anthropic")

        super().__init__(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        self.client = AsyncAnthropic(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    def supports_tools(self) -> bool:
        """Anthropic supports tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Anthropic supports streaming."""
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
        """Send chat completion request to Anthropic.

        Args:
            messages: Conversation messages
            model: Model name (e.g., "claude-sonnet-4-5", "claude-3-opus")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional Anthropic parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If request fails
        """
        try:
            # Separate system messages from conversation
            system_message = None
            conversation_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    conversation_messages.append(
                        {
                            "role": msg.role,
                            "content": msg.content,
                        }
                    )

            # Build request parameters
            request_params = {
                "model": model,
                "messages": conversation_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }

            if system_message:
                request_params["system"] = system_message

            if tools:
                request_params["tools"] = self._convert_tools(tools)

            # Make API call with circuit breaker protection
            response: AnthropicMessage = await self._execute_with_circuit_breaker(
                self.client.messages.create, **request_params
            )

            return self._parse_response(response, model)

        except (ProviderError, ProviderAuthError, ProviderRateLimitError):
            # Re-raise already-converted provider errors
            raise
        except (ConnectionError, TimeoutError) as e:
            # Network-related errors
            raise CoreProviderConnectionError(
                message=f"Connection error during Anthropic API request: {e}",
                provider=self.name,
            ) from e
        except Exception as e:
            # Catch-all for truly unexpected errors
            raise self._handle_error(e, self.name)

    async def stream(  # type: ignore[override,misc]
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion from Anthropic with tool-use support."""
        try:
            # Separate system messages
            system_message = None
            conversation_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    conversation_messages.append(
                        {
                            "role": msg.role,
                            "content": msg.content,
                        }
                    )

            # Build request parameters
            request_params = {
                "model": model,
                "messages": conversation_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }

            if system_message:
                request_params["system"] = system_message

            if tools:
                request_params["tools"] = self._convert_tools(tools)

            tool_calls: Dict[str, Dict[str, Any]] = {}
            block_index_to_id: Dict[int, str] = {}
            usage: Optional[Dict[str, int]] = None

            async with self.client.messages.stream(**request_params) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "message_start":
                        # Capture initial usage from message_start (input tokens)
                        message = getattr(event, "message", None)
                        if message:
                            msg_usage = getattr(message, "usage", None)
                            if msg_usage:
                                usage = {
                                    "prompt_tokens": getattr(msg_usage, "input_tokens", 0),
                                    "completion_tokens": 0,
                                    "total_tokens": getattr(msg_usage, "input_tokens", 0),
                                }
                                # Capture cache tokens if present
                                cache_creation = getattr(
                                    msg_usage, "cache_creation_input_tokens", None
                                )
                                cache_read = getattr(msg_usage, "cache_read_input_tokens", None)
                                if cache_creation is not None:
                                    usage["cache_creation_input_tokens"] = cache_creation
                                if cache_read is not None:
                                    usage["cache_read_input_tokens"] = cache_read

                    elif event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", "") == "tool_use":
                            tc_id = getattr(block, "id", None) or f"tool_{len(tool_calls) + 1}"
                            tool_calls[tc_id] = {
                                "id": tc_id,
                                "name": getattr(block, "name", ""),
                                "arguments": getattr(block, "input", {}) or {},
                            }
                            block_index = getattr(
                                event, "index", getattr(block, "index", len(block_index_to_id))
                            )
                            block_index_to_id[block_index] = tc_id

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", "")
                        block_index = getattr(
                            event, "index", getattr(event, "content_block_index", None)
                        )
                        if delta_type == "text_delta" and hasattr(delta, "text"):
                            text = getattr(delta, "text", "")
                            yield StreamChunk(content=text or "", is_final=False)
                        elif delta_type in {"input_json_delta", "input_delta"}:
                            tc_id = (
                                block_index_to_id.get(int(block_index))
                                if block_index is not None
                                else None
                            )
                            if tc_id and block_index is not None:  # Ensure block_index is not None
                                partial = (
                                    getattr(delta, "partial_json", None)
                                    or getattr(delta, "text", "")
                                    or ""
                                )
                                existing_args = tool_calls[tc_id].get("arguments", "")
                                if existing_args in ({}, None):
                                    tool_calls[tc_id]["arguments"] = partial
                                elif isinstance(existing_args, str):
                                    tool_calls[tc_id]["arguments"] = existing_args + partial
                                else:
                                    tool_calls[tc_id]["arguments"] = (
                                        json.dumps(existing_args) + partial
                                    )

                    elif event_type == "content_block_stop":
                        block_index = getattr(
                            event, "index", getattr(event, "content_block_index", None)
                        )
                        tc_id = (
                            block_index_to_id.get(int(block_index))
                            if block_index is not None
                            else None
                        )
                        if (
                            tc_id
                            and block_index is not None
                            and "arguments" in tool_calls.get(tc_id, {})
                        ):  # Ensure block_index is not None
                            tool_calls[tc_id]["arguments"] = self._parse_json_arguments(
                                tool_calls[tc_id].get("arguments")
                            )

                    elif event_type == "message_delta":
                        # Capture output tokens from message_delta
                        msg_usage = getattr(event, "usage", None)
                        if msg_usage:
                            output_tokens = getattr(msg_usage, "output_tokens", 0)
                            if usage:
                                usage["completion_tokens"] = output_tokens
                                usage["total_tokens"] = (
                                    usage.get("prompt_tokens", 0) + output_tokens
                                )
                            else:
                                usage = {
                                    "prompt_tokens": 0,
                                    "completion_tokens": output_tokens,
                                    "total_tokens": output_tokens,
                                }

                    elif event_type == "message_stop":
                        for tc in tool_calls.values():
                            tc["arguments"] = self._parse_json_arguments(tc.get("arguments"))

                        yield StreamChunk(
                            content="",
                            tool_calls=list(tool_calls.values()) or None,
                            stop_reason="stop",
                            is_final=True,
                            usage=usage,
                        )

        except (ProviderError, ProviderAuthError, ProviderRateLimitError):
            # Re-raise already-converted provider errors
            raise
        except (ConnectionError, TimeoutError) as e:
            # Network-related errors
            raise CoreProviderConnectionError(
                message=f"Connection error during Anthropic streaming: {e}",
                provider=self.name,
            ) from e
        except Exception as e:
            # Catch-all for truly unexpected errors
            raise self._handle_error(e, self.name)

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert standard tools to Anthropic format."""
        return convert_tools_to_anthropic_format(tools)

    def _parse_response(self, response: AnthropicMessage, model: str) -> CompletionResponse:
        """Parse Anthropic API response.

        Args:
            response: Raw Anthropic response
            model: Model name

        Returns:
            Normalized CompletionResponse
        """
        # Extract text content
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        # Parse usage
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=response.stop_reason,
            usage=usage,
            model=model,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            metadata={},  # Add metadata parameter
        )

    @staticmethod
    def _parse_json_arguments(raw_args: Any) -> Any:
        """Best-effort parse of tool-use arguments."""
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                # Invalid JSON, return as-is
                return raw_args
            except ValueError as e:
                # Other parsing errors
                logger.debug(f"Failed to parse tool arguments as JSON: {e}")
                return raw_args
        return raw_args

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Anthropic Claude models.

        Returns a curated list of currently available Claude models.
        Note: Anthropic doesn't have a public models endpoint, so this returns
        a static list of known available models.

        Returns:
            List of available models with metadata
        """
        # Anthropic doesn't provide a public API for listing models
        # Return the known Claude models with their metadata
        return [
            {
                "id": "claude-opus-4-5-20251101",
                "name": "Claude Opus 4.5",
                "description": "Most capable model for complex tasks",
                "context_window": 200000,
                "max_output_tokens": 32768,
            },
            {
                "id": "claude-sonnet-4-20250514",
                "name": "Claude Sonnet 4",
                "description": "Balanced performance and cost",
                "context_window": 200000,
                "max_output_tokens": 64000,
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3.5 Sonnet",
                "description": "Fast and efficient for everyday tasks",
                "context_window": 200000,
                "max_output_tokens": 8192,
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "name": "Claude 3.5 Haiku",
                "description": "Fastest model for quick responses",
                "context_window": 200000,
                "max_output_tokens": 8192,
            },
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "description": "Previous generation flagship model",
                "context_window": 200000,
                "max_output_tokens": 4096,
            },
        ]

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.close()
