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

"""Unified chat coordinator facade for sync/streaming execution.

This module contains the UnifiedChatCoordinator class that acts as a facade
for selecting between sync and streaming execution based on the execution mode.

The UnifiedChatCoordinator provides:
- Single interface for both sync and streaming
- Internal delegation to SyncChatCoordinator or StreamingChatCoordinator
- Support for ExecutionMode (SYNC, STREAMING, AUTO)
- Backward compatibility with existing chat methods

Architecture:
------------
The UnifiedChatCoordinator is a facade that internally delegates to:
- SyncChatCoordinator: For non-streaming execution
- StreamingChatCoordinator: For streaming execution

This maintains backward compatibility while enabling internal optimization.

Phase 2: Split Sync/Streaming Paths
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, TYPE_CHECKING

from victor.providers.base import CompletionResponse, StreamChunk
from victor.agent.coordinators.protocols import ExecutionMode

if TYPE_CHECKING:
    from victor.agent.coordinators.sync_chat_coordinator import SyncChatCoordinator
    from victor.agent.coordinators.streaming_chat_coordinator import (
        StreamingChatCoordinator,
    )

logger = logging.getLogger(__name__)


class UnifiedChatCoordinator:
    """Facade coordinator that selects sync or streaming execution.

    This coordinator provides a single interface that internally delegates to
    SyncChatCoordinator or StreamingChatCoordinator based on the execution mode.

    The facade pattern maintains backward compatibility while enabling internal
    optimization through separate execution paths.

    Args:
        sync_coordinator: Coordinator for non-streaming execution
        streaming_coordinator: Coordinator for streaming execution
        default_mode: Default execution mode (AUTO, SYNC, or STREAMING)
    """

    def __init__(
        self,
        sync_coordinator: "SyncChatCoordinator",
        streaming_coordinator: "StreamingChatCoordinator",
        default_mode: ExecutionMode = ExecutionMode.AUTO,
        chat_service: Any = None,
    ) -> None:
        """Initialize the UnifiedChatCoordinator.

        Args:
            sync_coordinator: Sync chat coordinator
            streaming_coordinator: Streaming chat coordinator
            default_mode: Default execution mode
        """
        self._sync = sync_coordinator
        self._streaming = streaming_coordinator
        self._default_mode = default_mode
        self._chat_service = chat_service

    def bind_chat_service(self, chat_service: Any) -> None:
        """Bind the canonical ChatService for backward-compatible delegation."""
        self._chat_service = chat_service

    # =====================================================================
    # Public API
    # =====================================================================

    async def chat(
        self,
        user_message: str,
        mode: ExecutionMode = ExecutionMode.AUTO,
        use_planning: bool = False,
    ) -> CompletionResponse:
        """Execute chat in specified mode.

        This method provides a unified interface for both sync and streaming
        execution. Based on the mode parameter, it delegates to the appropriate
        coordinator.

        Args:
            user_message: User's message
            mode: Execution mode (SYNC, STREAMING, or AUTO)
            use_planning: Whether to use structured planning for complex tasks

        Returns:
            CompletionResponse with complete response

        Note:
            - SYNC mode: Direct non-streaming execution (optimized)
            - STREAMING mode: Streaming execution (for real-time feedback)
            - AUTO mode: Automatically selects based on message characteristics
        """
        execution_mode = self._resolve_execution_mode(mode)

        if self._chat_service is not None:
            if execution_mode == ExecutionMode.STREAMING:
                return await self._chat_service.chat(
                    user_message,
                    stream=True,
                    use_planning=use_planning,
                )
            return await self._chat_service.chat(
                user_message,
                use_planning=use_planning,
            )

        if execution_mode == ExecutionMode.STREAMING:
            # Streaming mode: collect all chunks and return aggregated response
            content_parts = []
            final_response: CompletionResponse | None = None

            async for chunk in self._streaming.stream_chat(user_message):
                if chunk.content:
                    content_parts.append(chunk.content)
                if chunk.finish_reason:
                    # Create response from final chunk
                    final_response = CompletionResponse(
                        content="".join(content_parts),
                        role=chunk.role or "assistant",
                        tool_calls=chunk.tool_calls,
                    )

            return final_response or CompletionResponse(
                content="",
                role="assistant",
                tool_calls=None,
            )

        # Default to sync mode (direct path)
        return await self._sync.chat(user_message, use_planning=use_planning)

    async def stream_chat(
        self,
        user_message: str,
    ) -> AsyncIterator[StreamChunk]:
        """Execute chat with streaming response.

        This is a convenience method that always uses streaming execution,
        regardless of the default mode.

        Args:
            user_message: User's message

        Yields:
            StreamChunk objects with incremental response
        """
        if self._chat_service is not None:
            async for chunk in self._chat_service.stream_chat(user_message):
                yield chunk
            return

        async for chunk in self._streaming.stream_chat(user_message):
            yield chunk

    # =====================================================================
    # Private Methods
    # =====================================================================

    def _resolve_execution_mode(self, mode: ExecutionMode) -> ExecutionMode:
        """Resolve the execution mode.

        If mode is AUTO, resolve to SYNC or STREAMING based on heuristics.

        Args:
            mode: Requested execution mode

        Returns:
            Resolved execution mode (SYNC or STREAMING)
        """
        if mode != ExecutionMode.AUTO:
            return mode

        # AUTO mode: use default mode
        return self._default_mode


__all__ = [
    "UnifiedChatCoordinator",
]
