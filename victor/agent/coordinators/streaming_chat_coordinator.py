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

"""Streaming chat coordinator for streaming execution.

This module contains the StreamingChatCoordinator class that provides optimized
streaming chat execution.

The StreamingChatCoordinator handles:
- Yields StreamChunk as they arrive from provider
- Real-time event emission
- Interactive feedback support

Architecture:
------------
The StreamingChatCoordinator depends on protocol-based abstractions:
- ChatContextProtocol: For message/conversation access
- ToolContextProtocol: For tool execution
- ProviderContextProtocol: For LLM calls
- EventEmitter: For real-time events

Phase 2: Split Sync/Streaming Paths
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, TYPE_CHECKING

from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.coordinators.chat_protocols import (
        ChatContextProtocol,
        ToolContextProtocol,
        ProviderContextProtocol,
    )

logger = logging.getLogger(__name__)


class StreamingChatCoordinator:
    """Coordinator for streaming chat execution.

    This coordinator provides an optimized path for streaming chat execution,
    yielding StreamChunk objects as they arrive from the provider.

    Key features:
    - Yields StreamChunk as they arrive from provider
    - Real-time event emission for interactive feedback
    - Efficient tool call handling during streaming
    - Support for cancellation during stream

    Args:
        chat_context: Protocol providing conversation/message access
        tool_context: Protocol providing tool selection/execution
        provider_context: Protocol providing LLM provider access
        event_emitter: Protocol for event emission (optional)
    """

    def __init__(
        self,
        chat_context: "ChatContextProtocol",
        tool_context: "ToolContextProtocol",
        provider_context: "ProviderContextProtocol",
        event_emitter: Any = None,  # EventEmitter protocol
    ) -> None:
        """Initialize the StreamingChatCoordinator.

        Args:
            chat_context: Chat context protocol implementation
            tool_context: Tool context protocol implementation
            provider_context: Provider context protocol implementation
            event_emitter: Optional event emitter for real-time events
        """
        self._chat_context = chat_context
        self._tool_context = tool_context
        self._provider_context = provider_context
        self._event_emitter = event_emitter

    # =====================================================================
    # Public API
    # =====================================================================

    async def stream_chat(
        self,
        user_message: str,
    ) -> AsyncIterator[StreamChunk]:
        """Execute chat with streaming response.

        Yields StreamChunk objects as they arrive from the provider.

        This method provides the same functionality as ChatCoordinator.stream_chat()
        but is specifically optimized for streaming execution.

        Args:
            user_message: User's message

        Yields:
            StreamChunk objects with incremental response
        """
        # Ensure system prompt is included once at start of conversation
        self._chat_context.conversation.ensure_system_prompt()
        self._chat_context._system_added = True

        # Add user message to history
        self._chat_context.add_message("user", user_message)

        # Check for cancellation before starting
        if self._provider_context._check_cancellation():
            logger.debug("Stream cancelled before starting")
            return

        # Get tool definitions if provider supports them
        tools = None
        if (
            self._provider_context.provider.supports_tools()
            and self._tool_context.tool_calls_used < self._tool_context.tool_budget
        ):
            tools = await self._select_tools_for_turn(user_message)

        # Prepare optional thinking parameter
        provider_kwargs = {}
        if self._provider_context.thinking:
            provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

        # Stream response from provider
        async for chunk in self._stream_from_provider(
            user_message=user_message,
            tools=tools,
            **provider_kwargs,
        ):
            # Emit event if event emitter is available
            if self._event_emitter:
                await self._event_emitter.emit_async("stream_chunk", {"chunk": chunk})

            yield chunk

    # =====================================================================
    # Private Methods
    # =====================================================================

    async def _select_tools_for_turn(
        self,
        user_message: str,
    ) -> Any:
        """Select tools for the current iteration.

        Args:
            user_message: Original user message

        Returns:
            List of tool definitions or None
        """
        conversation_depth = self._chat_context.conversation.message_count()
        conversation_history = (
            [msg.model_dump() for msg in self._chat_context.messages]
            if self._chat_context.messages
            else None
        )

        tools = await self._tool_context.tool_selector.select_tools(
            user_message,
            use_semantic=self._tool_context.use_semantic_selection,
            conversation_history=conversation_history,
            conversation_depth=conversation_depth,
        )

        # Prioritize by stage
        tools = self._tool_context.tool_selector.prioritize_by_stage(user_message, tools)

        # Log tool definitions sent to LLM (permanent, INFO level)
        if tools:
            tool_summaries = [
                f"{t.name}: {(t.description or '')[:80]}" for t in tools
            ]
            logger.debug(
                "[ToolDefs→LLM] %d tools selected for query=%s\n  %s",
                len(tools),
                user_message[:100],
                "\n  ".join(tool_summaries),
            )

        return tools

    async def _stream_from_provider(
        self,
        user_message: str,
        tools: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from provider.

        Args:
            user_message: Current query
            tools: Optional tool definitions
            **kwargs: Additional provider parameters

        Yields:
            StreamChunk objects with incremental response
        """
        provider = self._provider_context.provider

        # Check if provider supports streaming
        if not hasattr(provider, "stream"):
            # Fall back to non-streaming call
            response = await provider.chat(
                messages=self._chat_context.messages,
                model=self._provider_context.model,
                temperature=self._provider_context.temperature,
                max_tokens=self._provider_context.max_tokens,
                tools=tools,
                **kwargs,
            )

            # Yield single chunk with complete response
            yield StreamChunk(
                content=response.content or "",
                role=response.role,
                tool_calls=response.tool_calls,
                finish_reason="stop",
                usage=response.usage,
            )

            # Add assistant response to history
            if response.content:
                self._chat_context.add_message("assistant", response.content)
            return

        # Stream from provider
        content_parts: list[str] = []
        tool_calls = None

        try:
            async for chunk in provider.stream(
                messages=self._chat_context.messages,
                model=self._provider_context.model,
                temperature=self._provider_context.temperature,
                max_tokens=self._provider_context.max_tokens,
                tools=tools,
                **kwargs,
            ):
                # Check for cancellation
                if self._provider_context._check_cancellation():
                    logger.debug("Stream cancelled during provider streaming")
                    break

                # Accumulate content
                if chunk.content:
                    content_parts.append(chunk.content)

                # Track tool calls
                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls

                yield chunk
        except Exception as e:
            logger.error("Stream failed after %d chunks: %s", len(content_parts), e)
            if content_parts:
                partial = "".join(content_parts)
                # Add partial content to conversation so it's not lost
                self._chat_context.add_message("assistant", partial + "\n\n[Stream interrupted]")
            raise

        # Add complete assistant response to history
        if content_parts:
            self._chat_context.add_message("assistant", "".join(content_parts))

        # Handle tool calls if present
        if tool_calls:
            await self._handle_tool_calls_during_stream(tool_calls)

    async def _handle_tool_calls_during_stream(self, tool_calls: Any) -> None:
        """Handle tool calls during streaming.

        Args:
            tool_calls: Tool calls from streaming response
        """
        # Execute tool calls
        tool_results = await self._tool_context._handle_tool_calls(tool_calls)

        # Add tool results to conversation
        for result in tool_results:
            if result.get("content"):
                self._chat_context.add_message("tool", result["content"])


__all__ = [
    "StreamingChatCoordinator",
]
