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

"""UI Agent Protocol for breaking UI-orchestrator coupling.

This protocol defines only the methods and properties that UI components
(TUI, CLI, slash commands, rendering) need from the agent orchestrator.
This follows the Dependency Inversion Principle - UI depends on abstractions
rather than concrete AgentOrchestrator implementation.

Key Features:
- AsyncIterator-based streaming for chat responses
- Provider access for capability checks
- Conversation management (reset, state access)
- Session management (ID tracking, persistence)
- Cancellation support for in-flight requests
- Metrics and tracking access
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.agent.conversation_state import ConversationStateMachine
    from victor.agent.message_history import MessageHistory
    from victor.providers.base import BaseProvider


@runtime_checkable
class UIAgentProtocol(Protocol):
    """Protocol defining agent interface for UI components.

    This protocol breaks the dependency between UI modules and the concrete
    AgentOrchestrator class. UI components (TUI, CLI, slash commands, rendering)
    depend only on this protocol interface, enabling:
    - Easier testing with mocks
    - Alternative orchestrator implementations
    - Clearer separation of concerns

    The protocol includes methods and properties used by:
    - victor/ui/tui/app.py (TUI interface)
    - victor/ui/commands/chat.py (CLI chat)
    - victor/ui/slash/commands/* (slash commands)
    - victor/ui/rendering/handler.py (stream rendering)
    """

    # ========================================================================
    # CORE CHAT METHODS
    # ========================================================================

    async def chat(self, message: str) -> Any:
        """Send a message and get a response.

        Args:
            message: User message to send

        Returns:
            Response object with content, usage, model fields
        """
        ...

    async def stream_chat(self, message: str) -> AsyncIterator[Any]:
        """Stream chat response chunks.

        Args:
            message: User message to send

        Yields:
            StreamChunk objects with content and metadata
        """
        ...

    # ========================================================================
    # PROVIDER ACCESS
    # ========================================================================

    @property
    def provider(self) -> "BaseProvider":
        """Access to the LLM provider for capability checks."""
        ...

    @property
    def provider_name(self) -> str:
        """Get the provider name (e.g., 'anthropic', 'ollama')."""
        ...

    # ========================================================================
    # CONVERSATION MANAGEMENT
    # ========================================================================

    @property
    def conversation(self) -> "MessageHistory":
        """Access to conversation message history."""
        ...

    @conversation.setter
    def conversation(self, value: "MessageHistory") -> None:
        """Set conversation history (for session restore)."""
        ...

    @property
    def conversation_state(self) -> Optional["ConversationStateMachine"]:
        """Access to conversation state machine (stage tracking)."""
        ...

    @conversation_state.setter
    def conversation_state(self, value: Optional["ConversationStateMachine"]) -> None:
        """Set conversation state (for session restore)."""
        ...

    @property
    def conversation_controller(self) -> Any:
        """Access to conversation controller for stage manipulation."""
        ...

    def reset_conversation(self) -> None:
        """Clear conversation history and reset state."""
        ...

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    @property
    def active_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        ...

    @active_session_id.setter
    def active_session_id(self, value: Optional[str]) -> None:
        """Set the current session ID."""
        ...

    # ========================================================================
    # STREAMING CONTROL
    # ========================================================================

    def is_streaming(self) -> bool:
        """Check if currently streaming a response.

        Returns:
            True if streaming in progress
        """
        ...

    def request_cancellation(self) -> None:
        """Request cancellation of the current stream."""
        ...

    # ========================================================================
    # TRACKING AND METRICS
    # ========================================================================

    @property
    def unified_tracker(self) -> Any:
        """Access to unified tracker for tool/iteration budgets."""
        ...

    def get_session_metrics(self) -> Optional[dict[str, Any]]:
        """Get session metrics (tool calls, tokens, etc.).

        Returns:
            Dictionary with metrics like 'tool_calls', 'total_tokens', etc.
        """
        ...

    # ========================================================================
    # EMBEDDING PRELOAD
    # ========================================================================

    def start_embedding_preload(self) -> None:
        """Start background embedding preload for semantic search."""
        ...


# Type alias for UI module usage
UIAgent = UIAgentProtocol


__all__ = [
    "UIAgentProtocol",
    "UIAgent",
]
