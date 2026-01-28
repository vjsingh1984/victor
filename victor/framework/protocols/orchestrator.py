"""Main orchestrator protocol combining all component protocols."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.framework.state import Stage
    from victor.framework.protocols.streaming import ChunkType, OrchestratorStreamChunk


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """Complete orchestrator protocol combining all capabilities.

    Any orchestrator implementation must satisfy this protocol to work
    with the framework layer. This is the primary interface contract.

    Design Pattern: Composite Protocol
    ================================
    This protocol combines 6 sub-protocols for type-safe orchestrator access:
    - ConversationStateProtocol: Stage, tool usage, file tracking
    - ProviderProtocol: Provider/model management
    - ToolsProtocol: Tool access and management
    - SystemPromptProtocol: Prompt customization
    - MessagesProtocol: Message history
    - StreamingProtocol: Streaming status

    When to use:
    - Framework code: Use this protocol for full orchestrator access
    - Core modules: Use victor.core.protocols.OrchestratorProtocol for minimal interface
    - SubAgents: Use SubAgentContext for ISP-compliant minimal interface

    The composite pattern ensures type safety while maintaining
    separation of concerns through sub-protocol definitions.

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

    def get_observed_files(self) -> "set[str]":
        """Get observed files."""
        ...

    def get_modified_files(self) -> "set[str]":
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
    def get_available_tools(self) -> "set[str]":
        """Get available tool names."""
        ...

    def get_enabled_tools(self) -> "set[str]":
        """Get enabled tool names."""
        ...

    def set_enabled_tools(self, tools: "set[str]") -> None:
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
    ) -> AsyncIterator["OrchestratorStreamChunk"]:
        """Stream a chat response with standardized chunks.

        This is the primary method for interactive chat. Returns an
        async iterator yielding OrchestratorStreamChunk instances.

        Args:
            message: User message to process
            context: Optional additional context

        Yields:
            OrchestratorStreamChunk instances representing response fragments

        Raises:
            ProviderError: If LLM call fails
            ToolError: If tool execution fails critically
        """
        ...
        yield  # type: ignore[misc]

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
