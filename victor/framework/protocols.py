"""Stable protocols for framework-orchestrator integration.

These protocols define the contract that any orchestrator implementation
must satisfy. Framework code uses these protocols, never direct attribute access.

Design Pattern: Protocol-First Architecture
- Eliminates duck-typing (hasattr/getattr calls)
- Provides type safety via Protocol structural subtyping
- Enables clean mocking for tests
- Documents the exact interface contract

Usage:
    # Type hint with protocol instead of concrete class
    def process(orchestrator: OrchestratorProtocol) -> None:
        stage = orchestrator.get_stage()  # Type-safe method call
        tools = orchestrator.get_available_tools()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.framework.events import EventType
    from victor.framework.state import Stage


# =============================================================================
# Streaming Types
# =============================================================================


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
class StreamChunk:
    """Standardized streaming chunk format from orchestrator.

    This is the canonical format returned by OrchestratorProtocol.stream_chat().
    Framework code converts these to Event instances for user consumption.

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
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_result: Optional[str] = None
    error: Optional[str] = None
    old_stage: Optional[str] = None
    new_stage: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = False


# =============================================================================
# Component Protocols
# =============================================================================


@runtime_checkable
class ConversationStateProtocol(Protocol):
    """Protocol for conversation state access.

    Provides read-only access to conversation state, including
    stage, tool usage, and file tracking.
    """

    def get_stage(self) -> "Stage":
        """Get current conversation stage.

        Returns:
            Current Stage enum value

        Raises:
            RuntimeError: If conversation state is not initialized
        """
        ...

    def get_tool_calls_count(self) -> int:
        """Get total tool calls made in this conversation.

        Returns:
            Non-negative count of tool calls
        """
        ...

    def get_tool_budget(self) -> int:
        """Get maximum allowed tool calls.

        Returns:
            Tool budget limit
        """
        ...

    def get_observed_files(self) -> Set[str]:
        """Get set of files observed (read) during conversation.

        Returns:
            Set of absolute file paths
        """
        ...

    def get_modified_files(self) -> Set[str]:
        """Get set of files modified (written/edited) during conversation.

        Returns:
            Set of absolute file paths
        """
        ...

    def get_iteration_count(self) -> int:
        """Get current agent loop iteration count.

        Returns:
            Non-negative iteration count
        """
        ...

    def get_max_iterations(self) -> int:
        """Get maximum allowed iterations.

        Returns:
            Max iteration limit
        """
        ...


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol for LLM provider management.

    Provides access to current provider/model and switching capability.
    """

    @property
    def current_provider(self) -> str:
        """Get current provider name.

        Returns:
            Provider identifier (e.g., "anthropic", "openai", "ollama")
        """
        ...

    @property
    def current_model(self) -> str:
        """Get current model name.

        Returns:
            Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4")
        """
        ...

    async def switch_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        on_switch: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Switch to a different provider and/or model.

        Args:
            provider: Target provider name
            model: Optional specific model (uses provider default if None)
            on_switch: Optional callback called with (provider, model) after switch

        Raises:
            ProviderError: If provider/model is not available
        """
        ...


@runtime_checkable
class ToolsProtocol(Protocol):
    """Protocol for tool management.

    Provides access to available tools and ability to enable/disable them.
    """

    def get_available_tools(self) -> Set[str]:
        """Get names of all registered tools.

        Returns:
            Set of tool names available in the registry
        """
        ...

    def get_enabled_tools(self) -> Set[str]:
        """Get names of currently enabled tools.

        Returns:
            Set of tool names that can be used in this session
        """
        ...

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set which tools are enabled for this session.

        Args:
            tools: Set of tool names to enable

        Raises:
            ValueError: If any tool name is not in available tools
        """
        ...

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
        ...


@runtime_checkable
class SystemPromptProtocol(Protocol):
    """Protocol for system prompt management.

    Provides access to and modification of the system prompt.
    """

    def get_system_prompt(self) -> str:
        """Get current system prompt.

        Returns:
            Complete system prompt string
        """
        ...

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (replaces existing).

        Args:
            prompt: New system prompt
        """
        ...

    def append_to_system_prompt(self, content: str) -> None:
        """Append content to existing system prompt.

        Args:
            content: Content to append (will be separated by newline)
        """
        ...


@runtime_checkable
class MessagesProtocol(Protocol):
    """Protocol for conversation messages access.

    Provides read access to conversation message history.
    """

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation message history.

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        ...

    def get_message_count(self) -> int:
        """Get number of messages in conversation.

        Returns:
            Non-negative message count
        """
        ...


@runtime_checkable
class StreamingProtocol(Protocol):
    """Protocol for streaming state.

    Provides access to streaming status.
    """

    def is_streaming(self) -> bool:
        """Check if agent is currently streaming.

        Returns:
            True if a streaming operation is in progress
        """
        ...


# =============================================================================
# Composite Protocol
# =============================================================================


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """Complete orchestrator protocol combining all capabilities.

    Any orchestrator implementation must satisfy this protocol to work
    with the framework layer. This is the primary interface contract.

    Design Pattern: Composite Protocol
    - Inherits from all component protocols
    - Adds core chat operations
    - Defines lifecycle methods

    Usage:
        async def run_agent(orch: OrchestratorProtocol, prompt: str):
            # Type-safe access to all orchestrator features
            print(f"Provider: {orch.current_provider}")
            print(f"Stage: {orch.get_stage()}")

            async for chunk in orch.stream_chat(prompt):
                if chunk.chunk_type == ChunkType.CONTENT:
                    print(chunk.content, end="")
    """

    # --- ConversationStateProtocol ---
    def get_stage(self) -> "Stage":
        """Get current conversation stage."""
        ...

    def get_tool_calls_count(self) -> int:
        """Get total tool calls made."""
        ...

    def get_tool_budget(self) -> int:
        """Get tool call budget."""
        ...

    def get_observed_files(self) -> Set[str]:
        """Get observed files."""
        ...

    def get_modified_files(self) -> Set[str]:
        """Get modified files."""
        ...

    def get_iteration_count(self) -> int:
        """Get iteration count."""
        ...

    def get_max_iterations(self) -> int:
        """Get max iterations."""
        ...

    # --- ProviderProtocol ---
    @property
    def current_provider(self) -> str:
        """Get current provider name."""
        ...

    @property
    def current_model(self) -> str:
        """Get current model name."""
        ...

    async def switch_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        on_switch: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Switch provider/model."""
        ...

    # --- ToolsProtocol ---
    def get_available_tools(self) -> Set[str]:
        """Get available tool names."""
        ...

    def get_enabled_tools(self) -> Set[str]:
        """Get enabled tool names."""
        ...

    def set_enabled_tools(self, tools: Set[str]) -> None:
        """Set enabled tools."""
        ...

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if tool is enabled."""
        ...

    # --- SystemPromptProtocol ---
    def get_system_prompt(self) -> str:
        """Get system prompt."""
        ...

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt."""
        ...

    def append_to_system_prompt(self, content: str) -> None:
        """Append to system prompt."""
        ...

    # --- MessagesProtocol ---
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation messages."""
        ...

    def get_message_count(self) -> int:
        """Get message count."""
        ...

    # --- StreamingProtocol ---
    def is_streaming(self) -> bool:
        """Check if streaming."""
        ...

    # --- Core Chat Operations ---
    async def stream_chat(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat response with standardized chunks.

        This is the primary method for interactive chat. Returns an
        async iterator yielding StreamChunk instances.

        Args:
            message: User message to process
            context: Optional additional context

        Yields:
            StreamChunk instances representing response fragments

        Raises:
            ProviderError: If LLM call fails
            ToolError: If tool execution fails critically
        """
        ...
        yield StreamChunk()  # type: ignore (protocol stub)

    async def chat(
        self,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Non-streaming chat that returns complete response.

        Convenience method that collects all content chunks
        into a single response string.

        Args:
            message: User message to process
            context: Optional additional context

        Returns:
            Complete response content as string

        Raises:
            ProviderError: If LLM call fails
            ToolError: If tool execution fails critically
        """
        ...

    # --- Lifecycle Methods ---
    def cancel(self) -> None:
        """Cancel any in-progress operation.

        Sets cancellation flag that streaming operations check.
        Safe to call multiple times.
        """
        ...

    def reset(self) -> None:
        """Reset conversation state.

        Clears message history, resets stage to INITIAL,
        and resets tool call counters.
        """
        ...


# =============================================================================
# Protocol Utilities
# =============================================================================


def verify_protocol_conformance(obj: Any, protocol: type) -> Tuple[bool, List[str]]:
    """Verify that an object conforms to a protocol.

    Checks that all required methods/properties are present.
    Useful for debugging protocol conformance issues.

    Args:
        obj: Object to check
        protocol: Protocol class to check against

    Returns:
        Tuple of (conforms: bool, missing: List[str])

    Example:
        conforms, missing = verify_protocol_conformance(orch, OrchestratorProtocol)
        if not conforms:
            raise TypeError(f"Missing protocol methods: {missing}")
    """
    missing = []

    # Get protocol's __protocol_attrs__ if available (runtime_checkable)
    if hasattr(protocol, "__protocol_attrs__"):
        for attr in protocol.__protocol_attrs__:
            if not hasattr(obj, attr):
                missing.append(attr)
    else:
        # Fall back to checking annotations
        hints = getattr(protocol, "__annotations__", {})
        for attr in hints:
            if not hasattr(obj, attr):
                missing.append(attr)

    return len(missing) == 0, missing
