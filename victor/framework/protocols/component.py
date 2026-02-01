"""Component protocols for orchestrator integration.

These protocols define individual capability areas that orchestrators
can implement. Each protocol focuses on a specific concern.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)
from collections.abc import Callable

if TYPE_CHECKING:
    from victor.framework.state import Stage


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

    def get_observed_files(self) -> set[str]:
        """Get set of files observed (read) during conversation.

        Returns:
            Set of absolute file paths
        """
        ...

    def get_modified_files(self) -> set[str]:
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

    def get_available_tools(self) -> set[str]:
        """Get names of all registered tools.

        Returns:
            Set of tool names available in the registry
        """
        ...

    def get_enabled_tools(self) -> set[str]:
        """Get names of currently enabled tools.

        Returns:
            Set of tool names that can be used in this session
        """
        ...

    def set_enabled_tools(self, tools: set[str]) -> None:
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

    def get_messages(self) -> list[dict[str, Any]]:
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
