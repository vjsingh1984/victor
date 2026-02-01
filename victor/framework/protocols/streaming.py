"""Streaming types for orchestrator protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ChunkType(str, Enum):
    """Types of streaming chunks from orchestrator.

    Maps to EventType but represents raw orchestrator output
    before conversion to framework events.
    """

    CONTENT = "content"
    """Text content from model response."""

    THINKING = "thinking"
    """Extended thinking content (reasoning mode)."""

    TOOL_CALL = "tool_call"
    """Tool invocation starting."""

    TOOL_RESULT = "tool_result"
    """Tool execution completed."""

    TOOL_ERROR = "tool_error"
    """Tool execution failed."""

    STAGE_CHANGE = "stage_change"
    """Conversation stage transition."""

    ERROR = "error"
    """General error occurred."""

    STREAM_START = "stream_start"
    """Streaming session started."""

    STREAM_END = "stream_end"
    """Streaming session ended."""


@dataclass
class OrchestratorStreamChunk:
    """Standardized streaming chunk format from orchestrator.

    This is the canonical format returned by OrchestratorProtocol.stream_chat().
    Framework code converts these to Event instances for user consumption.

    Renamed from StreamChunk to be semantically distinct from other streaming types:
    - StreamChunk (victor.providers.base): Provider-level raw streaming
    - OrchestratorStreamChunk: Orchestrator protocol with typed ChunkType
    - TypedStreamChunk: Safe typed accessor with nested StreamDelta
    - ClientStreamChunk: Protocol interface for clients (CLI/VS Code)

    Attributes:
        chunk_type: Type of this chunk (see ChunkType enum)
        content: Text content for CONTENT/THINKING chunks
        tool_name: Tool name for tool-related chunks
        tool_id: Unique identifier for tool call correlation
        tool_arguments: Arguments passed to tool (for TOOL_CALL)
        tool_result: Result from tool (for TOOL_RESULT)
        error: Error message (for ERROR/TOOL_ERROR chunks)
        old_stage: Previous stage (for STAGE_CHANGE)
        new_stage: New stage (for STAGE_CHANGE)
        metadata: Additional context-specific data
        is_final: True if this is the last chunk in the stream
    """

    chunk_type: ChunkType = ChunkType.CONTENT
    content: str = ""
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None
    tool_arguments: Optional[dict[str, Any]] = None
    tool_result: Optional[str] = None
    error: Optional[str] = None
    old_stage: Optional[str] = None
    new_stage: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_final: bool = False
