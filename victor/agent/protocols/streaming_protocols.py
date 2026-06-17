"""Protocol definitions for streaming protocols."""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from dataclasses import dataclass, field

__all__ = [
    "StreamingToolChunk",
    "StreamingToolAdapterProtocol",
    "StreamingControllerProtocol",
    "StreamingRecoveryCoordinatorProtocol",
    "ChunkGeneratorProtocol",
    "StreamingHandlerProtocol",
    "StreamingMetricsCollectorProtocol",
    "StreamingConfidenceMonitorProtocol",
]


@dataclass
class StreamingToolChunk:
    """Represents a chunk of streaming tool output.

    Used by StreamingToolAdapter to emit real-time updates during
    tool execution, enabling unified streaming behavior through ToolPipeline.

    Attributes:
        tool_name: Name of the tool being executed
        tool_call_id: Unique identifier for this tool call
        chunk_type: Type of chunk (start, progress, result, error, cache_hit)
        content: Chunk payload (varies by chunk_type)
        is_final: Whether this is the final chunk for this tool call
        metadata: Optional additional context
    """

    tool_name: str
    tool_call_id: str
    chunk_type: str  # "start", "progress", "result", "error", "cache_hit"
    content: Any
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StreamingToolAdapterProtocol(Protocol):
    """Protocol for streaming tool execution.

    Provides a unified streaming interface that wraps ToolPipeline,
    enabling real-time tool execution updates while preserving all
    ToolPipeline features (caching, middleware, callbacks, budget, etc.).

    This adapter solves the dual execution path problem where:
    - Batch path: Uses ToolPipeline with full feature support
    - Streaming path: Previously bypassed ToolPipeline using self.tools.execute()

    Now both paths route through this adapter -> ToolPipeline.
    """

    async def execute_streaming(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator["StreamingToolChunk"]:
        """Execute tools with streaming output.

        Yields StreamingToolChunk for each execution phase:
        1. "start" - Tool execution beginning
        2. "cache_hit" - Result served from cache (skips execution)
        3. "progress" - Intermediate progress updates
        4. "result" - Successful completion with result
        5. "error" - Execution failure

        Args:
            tool_calls: List of tool calls to execute
            context: Optional execution context

        Yields:
            StreamingToolChunk for each execution event
        """
        ...

    async def execute_streaming_single(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator["StreamingToolChunk"]:
        """Execute a single tool with streaming output.

        Convenience method for single tool execution.

        Args:
            tool_name: Name of tool to execute
            tool_args: Tool arguments
            tool_call_id: Optional identifier for tracking
            context: Optional execution context

        Yields:
            StreamingToolChunk for each execution event
        """
        ...

    @property
    def calls_used(self) -> int:
        """Number of tool calls used (delegates to ToolPipeline)."""
        ...

    @property
    def calls_remaining(self) -> int:
        """Number of tool calls remaining in budget."""
        ...

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted."""
        ...


@runtime_checkable
class StreamingControllerProtocol(Protocol):
    """Protocol for streaming session management.

    Manages streaming lifecycle and metrics collection.
    """

    def start_session(self, session_id: str) -> Any:
        """Start a new streaming session."""
        ...

    def end_session(self, session_id: str) -> None:
        """End a streaming session."""
        ...

    def get_active_session(self) -> Optional[Any]:
        """Get the currently active session."""
        ...


# NOTE: StreamingRecoveryCoordinatorProtocol and ChunkGeneratorProtocol are
# canonical service-owned runtime protocols in
# victor.agent.services.protocols.runtime_support. StreamingHandlerProtocol,
# StreamingMetricsCollectorProtocol, and StreamingConfidenceMonitorProtocol are
# canonical service-owned runtime protocols in
# victor.agent.services.protocols.infrastructure_runtime. This module
# re-exports all five at the bottom as deprecated compatibility names.


__all__ = [
    # Streaming protocols
    "StreamingToolChunk",
    "StreamingToolAdapterProtocol",
    "StreamingControllerProtocol",
    "StreamingRecoveryCoordinatorProtocol",
    "ChunkGeneratorProtocol",
    "StreamingHandlerProtocol",
    "StreamingConfidenceMonitorProtocol",
    "StreamingMetricsCollectorProtocol",
]
from victor.agent.services.protocols.infrastructure_runtime import (
    StreamingConfidenceMonitorProtocol,
    StreamingHandlerProtocol,
    StreamingMetricsCollectorProtocol,
)
from victor.agent.services.protocols.runtime_support import (
    ChunkRuntimeProtocol as ChunkGeneratorProtocol,
    StreamingRecoveryRuntimeProtocol as StreamingRecoveryCoordinatorProtocol,
)

# Alias: some modules reference StreamingCoordinatorProtocol
StreamingCoordinatorProtocol = StreamingControllerProtocol
