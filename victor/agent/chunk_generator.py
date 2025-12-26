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
# See the License for the specific language governing permissions
# limitations under the License.

"""Chunk generator for streaming output.

This module provides a centralized interface for generating streaming chunks
for various purposes (tool execution, status updates, metrics, content).

Extracted from CRITICAL-001 Phase 2B: Extract ChunkGenerator
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.streaming import StreamingChatHandler, StreamingChatContext
    from victor.config.settings import Settings

from victor.providers.base import StreamChunk
from victor.observability.event_bus import EventBus, EventCategory, VictorEvent

logger = logging.getLogger(__name__)


class ChunkGenerator:
    """Generates streaming chunks for various purposes.

    This component provides a semantic interface for chunk generation operations,
    consolidating all chunk-related methods that were previously scattered across
    AgentOrchestrator.

    Architecture:
    - ChunkGenerator: High-level semantic interface for chunk operations
    - StreamingChatHandler: Low-level chunk creation and formatting
    - AgentOrchestrator: Uses ChunkGenerator for all streaming output

    Responsibilities:
    - Generate tool-related chunks (start, result)
    - Generate status chunks (thinking, budget errors, force response)
    - Generate content chunks (metrics, content, final markers)
    - Provide clean, semantic API for streaming operations

    Design Pattern:
    - Coordinator/Facade: Simplifies chunk generation interface
    - Delegation: Delegates to StreamingChatHandler for implementation

    Extracted from CRITICAL-001 Phase 2B.
    """

    def __init__(
        self,
        streaming_handler: "StreamingChatHandler",
        settings: "Settings",
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize ChunkGenerator.

        Args:
            streaming_handler: Handler for streaming chunk generation
            settings: Application settings
            event_bus: Optional EventBus instance. If None, uses singleton.
        """
        self.streaming_handler = streaming_handler
        self.settings = settings
        self._event_bus = event_bus or EventBus.get_instance()

    # =====================================================================
    # Tool-Related Chunks
    # =====================================================================

    def generate_tool_start_chunk(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        status_msg: str,
    ) -> StreamChunk:
        """Generate chunk indicating tool execution start.

        Args:
            tool_name: Name of the tool being executed
            tool_args: Tool arguments
            status_msg: Status message to display

        Returns:
            StreamChunk with tool start metadata
        """
        # Emit LIFECYCLE event for tool execution start
        self._event_bus.publish(
            VictorEvent(
                category=EventCategory.LIFECYCLE,
                name="chunk.tool_start",
                data={
                    "tool_name": tool_name,
                    "status_msg": status_msg,
                },
                source="ChunkGenerator",
            )
        )
        return self.streaming_handler.generate_tool_start_chunk(
            tool_name, tool_args, status_msg
        )

    def generate_tool_result_chunks(
        self,
        result: Dict[str, Any],
    ) -> List[StreamChunk]:
        """Generate chunks for tool execution result.

        Args:
            result: Tool execution result dictionary

        Returns:
            List of StreamChunks representing the tool result
        """
        return self.streaming_handler.generate_tool_result_chunks(result)

    # =====================================================================
    # Status Chunks
    # =====================================================================

    def generate_thinking_status_chunk(self) -> StreamChunk:
        """Generate chunk indicating thinking/processing status.

        Returns:
            StreamChunk with thinking status metadata
        """
        return self.streaming_handler.generate_thinking_status_chunk()

    def generate_budget_error_chunk(self) -> StreamChunk:
        """Generate chunk for budget limit error.

        Returns:
            StreamChunk with budget limit error message
        """
        return self.streaming_handler.generate_budget_error_chunk()

    def generate_force_response_error_chunk(self) -> StreamChunk:
        """Generate chunk for forced response error.

        Returns:
            StreamChunk with force response error message
        """
        return self.streaming_handler.generate_force_response_error_chunk()

    def generate_final_marker_chunk(self) -> StreamChunk:
        """Generate final marker chunk to signal stream completion.

        Returns:
            StreamChunk with is_final=True
        """
        # Emit LIFECYCLE event for streaming completion
        self._event_bus.publish(
            VictorEvent(
                category=EventCategory.LIFECYCLE,
                name="chunk.stream_complete",
                data={},
                source="ChunkGenerator",
            )
        )
        return self.streaming_handler.generate_final_marker_chunk()

    # =====================================================================
    # Content Chunks
    # =====================================================================

    def generate_metrics_chunk(
        self,
        metrics_line: str,
        is_final: bool = False,
        prefix: str = "\n\n",
    ) -> StreamChunk:
        """Generate chunk for metrics display.

        Args:
            metrics_line: Formatted metrics line
            is_final: Whether this is the final chunk
            prefix: Prefix before metrics line (default: double newline)

        Returns:
            StreamChunk with formatted metrics content
        """
        # Emit METRIC event for metrics display
        self._event_bus.publish(
            VictorEvent(
                category=EventCategory.METRIC,
                name="chunk.metrics_generated",
                data={
                    "metrics_line": metrics_line[:200],  # Truncate for event
                    "is_final": is_final,
                },
                source="ChunkGenerator",
            )
        )
        return self.streaming_handler.generate_metrics_chunk(
            metrics_line, is_final=is_final, prefix=prefix
        )

    def generate_content_chunk(
        self,
        content: str,
        is_final: bool = False,
        suffix: str = "",
    ) -> StreamChunk:
        """Generate chunk for content display.

        Args:
            content: Sanitized content to display
            is_final: Whether this is the final chunk
            suffix: Optional suffix to append

        Returns:
            StreamChunk with content and optional suffix
        """
        return self.streaming_handler.generate_content_chunk(
            content, is_final=is_final, suffix=suffix
        )

    # =====================================================================
    # Budget Chunks
    # =====================================================================

    def get_budget_exhausted_chunks(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> List[StreamChunk]:
        """Get chunks for budget exhaustion warning.

        Args:
            stream_ctx: Streaming context

        Returns:
            List of StreamChunks for budget exhausted warning
        """
        return self.streaming_handler.get_budget_exhausted_chunks(stream_ctx)

    # =====================================================================
    # Metrics Formatting
    # =====================================================================

    def format_completion_metrics(
        self,
        stream_ctx: "StreamingChatContext",
        elapsed_time: float,
    ) -> str:
        """Format performance metrics for normal completion.

        Delegates to streaming handler for detailed metrics formatting
        with cache info when available, or falls back to estimated tokens.

        Args:
            stream_ctx: The streaming context
            elapsed_time: Elapsed time in seconds

        Returns:
            Formatted metrics line string
        """
        return self.streaming_handler.format_completion_metrics(stream_ctx, elapsed_time)
