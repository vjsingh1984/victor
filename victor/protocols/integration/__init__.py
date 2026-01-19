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

"""Cross-cutting integration protocols for layer boundary enforcement.

This module provides protocol-based abstractions that enable the Framework layer
to interact with core types without depending on the Agent layer. This enforces
proper layer boundaries following the Dependency Inversion Principle (DIP).

Design Principles:
- Framework depends on Protocols (not Agent implementations)
- Core types are shared via Protocol abstractions
- Agent layer implements protocols
- Framework layer consumes protocols

Layer Architecture:
    ┌─────────────────────────────────────────┐
    │         Framework Layer                 │
    │  (State, Teams, Workflows, etc.)        │
    └──────────────┬──────────────────────────┘
                   │ depends on
                   ▼
    ┌─────────────────────────────────────────┐
    │      Integration Protocols              │
    │  (IVerticalContextProvider, etc.)       │
    └──────────────┬──────────────────────────┘
                   │ implemented by
                   ▼
    ┌─────────────────────────────────────────┐
    │          Agent Layer                    │
    │  (AgentOrchestrator, SubAgents, etc.)   │
    └─────────────────────────────────────────┘

Example Usage:
    # Framework code depends on protocol (DIP compliant)
    from victor.protocols.integration import IVerticalContextProvider

    def configure_framework(provider: IVerticalContextProvider):
        context = provider.get_vertical_context()
        # Use context without knowing about AgentOrchestrator

    # Agent layer implements protocol
    class AgentOrchestrator(IVerticalContextProvider):
        def get_vertical_context(self) -> VerticalContext:
            return self._vertical_context
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from victor.core.state import ConversationStage
    from victor.core.teams import SubAgentRole
    from victor.core.verticals.context import VerticalContext


# =============================================================================
# Vertical Context Protocol
# =============================================================================


@runtime_checkable
class IVerticalContextProvider(Protocol):
    """Protocol for providing vertical context.

    This protocol enables Framework components to access vertical configuration
    without depending on AgentOrchestrator or other Agent layer types.

    Implementation provided by: AgentOrchestrator
    Consumed by: Framework components (State, Teams, Workflows)
    """

    @property
    def vertical_context(self) -> "VerticalContext":
        """Get the current vertical context.

        Returns:
            VerticalContext with all vertical configuration
        """
        ...

    def get_vertical_context(self) -> "VerticalContext":
        """Get the current vertical context (method variant).

        Returns:
            VerticalContext with all vertical configuration
        """
        ...

    def set_vertical_context(self, context: "VerticalContext") -> None:
        """Set a new vertical context.

        Args:
            context: New vertical context to apply
        """
        ...

    def update_vertical_context(self, **updates: Any) -> None:
        """Update the current vertical context.

        Args:
            **updates: Key-value pairs to update in the context
        """
        ...


# =============================================================================
# Conversation State Protocol
# =============================================================================


@runtime_checkable
class IConversationStateManager(Protocol):
    """Protocol for managing conversation state.

    This protocol enables Framework components to track and update
    conversation stage without depending on Agent layer implementations.

    Implementation provided by: AgentOrchestrator, ConversationStateMachine
    Consumed by: Framework State component
    """

    @property
    def current_stage(self) -> "ConversationStage":
        """Get the current conversation stage.

        Returns:
            Current ConversationStage enum value
        """
        ...

    def get_stage(self) -> "ConversationStage":
        """Get the current conversation stage (method variant).

        Returns:
            Current ConversationStage enum value
        """
        ...

    def transition_to(self, stage: "ConversationStage") -> None:
        """Transition to a new conversation stage.

        Args:
            stage: Target stage to transition to
        """
        ...

    def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool execution for stage detection.

        Args:
            tool_name: Name of executed tool
            args: Tool arguments
        """
        ...

    def get_stage_tools(self) -> List[str]:
        """Get tools relevant to current stage.

        Returns:
            List of tool names for current stage
        """
        ...

    def reset_state(self) -> None:
        """Reset conversation state to initial."""
        ...


# =============================================================================
# SubAgent Coordination Protocol
# =============================================================================


@runtime_checkable
class ISubAgentCoordinator(Protocol):
    """Protocol for coordinating sub-agent execution.

    This protocol enables Framework components to create and manage
    sub-agents without depending on Agent layer implementations.

    Implementation provided by: AgentOrchestrator
    Consumed by: Framework Teams component
    """

    def create_subagent(
        self,
        role: "SubAgentRole",
        task: str,
        allowed_tools: List[str],
        tool_budget: int = 15,
        context_limit: int = 30000,
        system_prompt_override: Optional[str] = None,
    ) -> Any:
        """Create a sub-agent for delegated execution.

        Args:
            role: Sub-agent role specialization
            task: Task description for the sub-agent
            allowed_tools: Tools the sub-agent can use
            tool_budget: Maximum tool calls allowed
            context_limit: Maximum context size
            system_prompt_override: Optional custom system prompt

        Returns:
            SubAgent instance (or protocol-compliant object)
        """
        ...

    async def execute_subagent(
        self,
        role: "SubAgentRole",
        task: str,
        allowed_tools: List[str],
        tool_budget: int = 15,
        context_limit: int = 30000,
    ) -> Dict[str, Any]:
        """Execute a sub-agent and return the result.

        Args:
            role: Sub-agent role specialization
            task: Task description for the sub-agent
            allowed_tools: Tools the sub-agent can use
            tool_budget: Maximum tool calls allowed
            context_limit: Maximum context size

        Returns:
            Dict with execution results (success, summary, details, metrics)
        """
        ...

    def can_spawn_subagents(self) -> bool:
        """Check if sub-agent spawning is enabled.

        Returns:
            True if sub-agents can be created
        """
        ...

    def get_subagent_budget(self) -> int:
        """Get default tool budget for sub-agents.

        Returns:
            Default tool budget
        """
        ...


# =============================================================================
# Tool Access Protocol
# =============================================================================


@runtime_checkable
class IToolAccessProvider(Protocol):
    """Protocol for tool access and discovery.

    This protocol enables Framework components to access tools
    without depending on AgentOrchestrator or ToolRegistry.

    Implementation provided by: AgentOrchestrator, ToolRegistry
    Consumed by: Framework components
    """

    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not found
        """
        ...

    def list_tools(self) -> List[str]:
        """List all available tools.

        Returns:
            List of tool names
        """
        ...

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is available.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool exists
        """
        ...

    def get_tools_by_category(self, category: str) -> List[Any]:
        """Get all tools in a category.

        Args:
            category: Tool category (e.g., "coding", "devops")

        Returns:
            List of tools in the category
        """
        ...


# =============================================================================
# Provider Access Protocol
# =============================================================================


@runtime_checkable
class IProviderAccess(Protocol):
    """Protocol for accessing LLM provider information.

    This protocol enables Framework components to access provider
    configuration without depending on AgentOrchestrator.

    Implementation provided by: AgentOrchestrator
    Consumed by: Framework components
    """

    @property
    def provider_name(self) -> str:
        """Get the current provider name.

        Returns:
            Provider name (e.g., "anthropic", "openai")
        """
        ...

    @property
    def model_name(self) -> str:
        """Get the current model name.

        Returns:
            Model identifier
        """
        ...

    @property
    def temperature(self) -> float:
        """Get the current temperature setting.

        Returns:
            Temperature value (0.0 to 1.0)
        """
        ...

    def switch_provider(self, provider_name: str, model: str) -> None:
        """Switch to a different provider/model.

        Args:
            provider_name: New provider name
            model: New model name
        """
        ...


# =============================================================================
# Message History Protocol
# =============================================================================


@runtime_checkable
class IMessageHistoryProvider(Protocol):
    """Protocol for accessing conversation message history.

    This protocol enables Framework components to access message
    history without depending on AgentOrchestrator.

    Implementation provided by: AgentOrchestrator
    Consumed by: Framework components (State, Observability)
    """

    @property
    def message_count(self) -> int:
        """Get the number of messages in history.

        Returns:
            Message count
        """
        ...

    def get_messages(self) -> List[Any]:
        """Get all messages in the conversation.

        Returns:
            List of message objects
        """
        ...

    def get_last_n_messages(self, n: int) -> List[Any]:
        """Get the last N messages.

        Args:
            n: Number of messages to retrieve

        Returns:
            List of last N messages
        """
        ...

    def add_message(self, role: str, content: str) -> None:
        """Add a message to history.

        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
        """
        ...

    def clear_history(self) -> None:
        """Clear all message history."""
        ...


# =============================================================================
# Orchestrator Capabilities Protocol
# =============================================================================


@runtime_checkable
class IOrchestratorCapabilities(Protocol):
    """Protocol for querying orchestrator capabilities.

    This protocol enables Framework components to discover what
    features the orchestrator supports without tight coupling.

    Implementation provided by: AgentOrchestrator
    Consumed by: Framework components
    """

    def supports_tool_calling(self) -> bool:
        """Check if the current model supports tool calling.

        Returns:
            True if tool calling is supported
        """
        ...

    def supports_streaming(self) -> bool:
        """Check if the current model supports streaming.

        Returns:
            True if streaming is supported
        """
        ...

    def supports_vision(self) -> bool:
        """Check if the current model supports vision/images.

        Returns:
            True if vision is supported
        """
        ...

    def max_tokens(self) -> int:
        """Get maximum token limit for current model.

        Returns:
            Maximum tokens
        """
        ...

    def get_capability(self, capability_name: str) -> Optional[Any]:
        """Get a specific capability by name.

        Args:
            capability_name: Name of the capability

        Returns:
            Capability value or None
        """
        ...


# =============================================================================
# Composite Protocol (Unified Access)
# =============================================================================


@runtime_checkable
class IOrchestratorBridge(
    IVerticalContextProvider,
    IConversationStateManager,
    ISubAgentCoordinator,
    IToolAccessProvider,
    IProviderAccess,
    IMessageHistoryProvider,
    IOrchestratorCapabilities,
    Protocol,
):
    """Composite protocol for full orchestrator access.

    This protocol combines all integration protocols into a single
    interface for Framework components that need broad access.

    Implementation provided by: AgentOrchestrator
    Consumed by: Framework components requiring multiple capabilities
    """

    def is_initialized(self) -> bool:
        """Check if the orchestrator is fully initialized.

        Returns:
            True if ready for use
        """
        ...

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status information.

        Returns:
            Dict with status keys (stage, tools_used, provider, model, etc.)
        """
        ...


# =============================================================================
# Factory Functions
# =============================================================================


def adapt_to_vertical_context_provider(obj: Any) -> Optional["IVerticalContextProvider"]:
    """Adapt an object to IVerticalContextProvider if possible.

    Args:
        obj: Object to adapt

    Returns:
        Adapted object or None if not compatible
    """
    if isinstance(obj, IVerticalContextProvider):
        return obj
    return None


def adapt_to_conversation_state_manager(obj: Any) -> Optional["IConversationStateManager"]:
    """Adapt an object to IConversationStateManager if possible.

    Args:
        obj: Object to adapt

    Returns:
        Adapted object or None if not compatible
    """
    if isinstance(obj, IConversationStateManager):
        return obj
    return None


def adapt_to_subagent_coordinator(obj: Any) -> Optional["ISubAgentCoordinator"]:
    """Adapt an object to ISubAgentCoordinator if possible.

    Args:
        obj: Object to adapt

    Returns:
        Adapted object or None if not compatible
    """
    if isinstance(obj, ISubAgentCoordinator):
        return obj
    return None


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Core protocols
    "IVerticalContextProvider",
    "IConversationStateManager",
    "ISubAgentCoordinator",
    "IToolAccessProvider",
    "IProviderAccess",
    "IMessageHistoryProvider",
    "IOrchestratorCapabilities",
    # Composite protocol
    "IOrchestratorBridge",
    # Factory functions
    "adapt_to_vertical_context_provider",
    "adapt_to_conversation_state_manager",
    "adapt_to_subagent_coordinator",
]
