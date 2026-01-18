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

"""Conversation management protocols.

This module contains protocols related to conversation state, message history,
context management, and streaming. These protocols define contracts for:

- Conversation state management and transitions
- Message storage and retrieval
- Context compaction and overflow handling
- Streaming session management

Usage:
    from victor.protocols.agent_conversation import (
        ConversationControllerProtocol,
        ConversationStateMachineProtocol,
        MessageHistoryProtocol,
    )
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.conversation_state import ConversationStage


# =============================================================================
# Conversation Protocols
# =============================================================================


@runtime_checkable
class ConversationControllerProtocol(Protocol):
    """Protocol for conversation management.

    Manages message history, context tracking, and compaction.
    """

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        ...

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in conversation."""
        ...

    def get_context_metrics(self) -> Any:
        """Get current context utilization metrics."""
        ...

    def compact_if_needed(self) -> bool:
        """Compact conversation if context is nearly full.

        Returns:
            True if compaction occurred
        """
        ...

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        ...


@runtime_checkable
class ConversationStateMachineProtocol(Protocol):
    """Protocol for conversation state machine.

    Tracks conversation stage (INITIAL, PLANNING, EXECUTING, etc.).
    """

    def get_stage(self) -> ConversationStage:
        """Get current conversation stage."""
        ...

    def get_current_stage(self) -> ConversationStage:
        """Get current conversation stage (alias)."""
        ...

    def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record tool execution for stage inference."""
        ...

    def record_message(self, content: str, is_user: bool = True) -> None:
        """Record a message for stage inference."""
        ...


@runtime_checkable
class MessageHistoryProtocol(Protocol):
    """Protocol for message history.

    Manages raw message storage.
    """

    def add_message(self, role: str, content: str, **kwargs: Any) -> Any:
        """Add a message."""
        ...

    def get_messages_for_provider(self) -> List[Any]:
        """Get all messages for provider."""
        ...

    def clear(self) -> None:
        """Clear message history."""
        ...


# =============================================================================
# Streaming Protocols
# =============================================================================

from dataclasses import dataclass, field


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
    ) -> AsyncIterator[StreamingToolChunk]:
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
    ) -> AsyncIterator[StreamingToolChunk]:
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


# =============================================================================
# Context Management Protocols
# =============================================================================


@runtime_checkable
class ContextCompactorProtocol(Protocol):
    """Protocol for context compaction."""

    def maybe_compact_proactively(self) -> bool:
        """Attempt proactive compaction if threshold reached.

        Returns:
            True if compaction occurred
        """
        ...

    def truncate_tool_result(
        self, tool_name: str, result: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Truncate tool result if needed."""
        ...


# =============================================================================
# Embedding Store Protocols
# =============================================================================


@runtime_checkable
class ConversationEmbeddingStoreProtocol(Protocol):
    """Protocol for conversation embedding storage.

    Provides semantic search over conversation history using embeddings.
    """

    @property
    def is_initialized(self) -> bool:
        """Check if the store is initialized."""
        ...

    async def initialize(self) -> None:
        """Initialize the embedding store."""
        ...

    async def search_similar(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> List[Any]:
        """Search for semantically similar messages.

        Args:
            query: Query text to search for
            session_id: Optional session to scope search
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            exclude_message_ids: Message IDs to exclude

        Returns:
            List of search results (message_id + similarity)
        """
        ...

    async def delete_session(self, session_id: str) -> int:
        """Delete all embeddings for a session.

        Args:
            session_id: Session ID to delete

        Returns:
            Number of embeddings deleted
        """
        ...

    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...


# =============================================================================
# Utility Service Protocols
# =============================================================================


@runtime_checkable
class ReminderManagerProtocol(Protocol):
    """Protocol for context reminder management.

    Manages intelligent injection of context reminders to reduce token waste.
    """

    def reset(self) -> None:
        """Reset state for a new conversation turn."""
        ...

    def update_state(
        self,
        observed_files: Optional[set] = None,
        executed_tool: Optional[str] = None,
        tool_calls: Optional[int] = None,
        tool_budget: Optional[int] = None,
        task_complexity: Optional[str] = None,
        task_hint: Optional[str] = None,
    ) -> None:
        """Update the current context state."""
        ...

    def add_observed_file(self, file_path: str) -> None:
        """Add a file to the observed files set."""
        ...

    def get_consolidated_reminder(self, force: bool = False) -> Optional[str]:
        """Get a consolidated reminder combining all active reminders."""
        ...


__all__ = [
    # Conversation protocols
    "ConversationControllerProtocol",
    "ConversationStateMachineProtocol",
    "MessageHistoryProtocol",
    # Streaming protocols
    "StreamingToolChunk",
    "StreamingToolAdapterProtocol",
    "StreamingControllerProtocol",
    # Context management protocols
    "ContextCompactorProtocol",
    # Embedding store protocols
    "ConversationEmbeddingStoreProtocol",
    # Utility service protocols
    "ReminderManagerProtocol",
]
