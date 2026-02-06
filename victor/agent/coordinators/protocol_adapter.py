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

"""Protocol adapter for orchestrator.

This adapter extracts protocol implementations from the orchestrator,
reducing its size and improving modularity.

Implemented Protocols:
- ConversationStateProtocol: Stage, tool calls, budget, observed files
- ProviderProtocol: Provider/model switching, current provider info
- ToolsProtocol: Available tools, enabled tools, tool access control
- SystemPromptProtocol: System prompt access and modification
- MessagesProtocol: Message count and history access
- Health Check Methods: Tool selector health checks
- Lifecycle Methods: Session reset, graceful shutdown

Usage:
    adapter = OrchestratorProtocolAdapter(
        orchestrator=orchestrator,
        state_coordinator=state_coordinator,
        provider_coordinator=provider_coordinator,
        tools=tools,
        conversation=conversation,
        mode_controller=mode_controller,
        unified_tracker=unified_tracker,
        tool_selector=tool_selector,
        tool_access_config_coordinator=tool_access_config_coordinator,
        vertical_context=vertical_context,
        conversation_state=conversation_state,
        prompt_builder=prompt_builder,
    )

    # Use protocol methods
    stage = adapter.get_stage()
    await adapter.switch_provider("anthropic", "claude-sonnet-4-5")
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.coordinators.state_coordinator import StateCoordinator
    from victor.agent.coordinators.tool_access_config_coordinator import ToolAccessConfigCoordinator
    from victor.agent.conversation_state import ConversationStateMachine
    from victor.core.state import ConversationStage
    from victor.agent.mode_controller import AgentModeController
    from victor.agent.unified_tracker import UnifiedTaskTracker
    from victor.tools.registry import ToolRegistry
    from victor.core.verticals.vertical_context import VerticalContext
    from victor.agent.tool_selector import ToolSelector
    from victor.agent.prompts.system import SystemPromptBuilder

    # Conversation is not in the conversation module, use Any as fallback
    Conversation: Any = Any

logger = logging.getLogger(__name__)


class OrchestratorProtocolAdapter:
    """Adapter for orchestrator protocol implementations.

    This class implements the various protocols that the orchestrator needs
    to support, extracting this logic from the orchestrator itself.

    The adapter holds references to the required coordinators and components,
    and provides protocol-compliant methods that delegate to them.

    Attributes:
        _orchestrator: Reference to the orchestrator (for fallback access)
        _state_coordinator: State management coordinator
        _provider_coordinator: Provider switching coordinator
        _tools: Tool registry
        _conversation: Conversation object
        _mode_controller: Mode management controller
        _unified_tracker: Unified task tracker
        _tool_selector: Tool selection component
        _tool_access_config_coordinator: Tool access control coordinator
        _vertical_context: Vertical context
        _conversation_state: Conversation state machine
        _prompt_builder: System prompt builder
    """

    def __init__(
        self,
        orchestrator: Any,
        state_coordinator: StateCoordinator,
        provider_coordinator: Any,
        tools: Optional[ToolRegistry],
        conversation: Optional[Conversation],
        mode_controller: Optional[AgentModeController],
        unified_tracker: Optional[UnifiedTaskTracker],
        tool_selector: Optional[ToolSelector],
        tool_access_config_coordinator: Optional[ToolAccessConfigCoordinator],
        vertical_context: Optional[VerticalContext],
        conversation_state: Optional[ConversationStateMachine],
        prompt_builder: Optional[SystemPromptBuilder],
    ) -> None:
        """Initialize the protocol adapter.

        Args:
            orchestrator: Reference to the orchestrator for fallback access
            state_coordinator: State management coordinator
            provider_coordinator: Provider switching coordinator
            tools: Tool registry (optional)
            conversation: Conversation object (optional)
            mode_controller: Mode management controller
            unified_tracker: Unified task tracker
            tool_selector: Tool selection component
            tool_access_config_coordinator: Tool access control coordinator
            vertical_context: Vertical context
            conversation_state: Conversation state machine
            prompt_builder: System prompt builder
        """
        self._orchestrator = orchestrator
        self._state_coordinator = state_coordinator
        self._provider_coordinator = provider_coordinator
        self._tools = tools
        self._conversation = conversation
        self._mode_controller = mode_controller
        self._unified_tracker = unified_tracker
        self._tool_selector = tool_selector
        self._tool_access_config_coordinator = tool_access_config_coordinator
        self._vertical_context = vertical_context
        self._conversation_state = conversation_state
        self._prompt_builder = prompt_builder

    # ========================================================================
    # ConversationStateProtocol
    # ========================================================================

    def get_stage(self) -> ConversationStage:
        """Get current conversation stage (protocol method).

        Returns:
            Current ConversationStage enum value

        Note:
            Framework layer converts this to framework.state.Stage
        """
        from victor.core.state import ConversationStage

        # StateCoordinator returns stage name, convert to enum
        stage_name: Optional[str] = self._state_coordinator.get_stage()
        if stage_name:
            return ConversationStage[stage_name]
        return ConversationStage.INITIAL

    def get_tool_calls_count(self) -> int:
        """Get total tool calls made (protocol method).

        Returns:
            Non-negative count of tool calls in this session
        """
        if self._unified_tracker:
            tool_calls: int = self._unified_tracker.tool_calls_used
            return tool_calls
        fallback: int = getattr(self._orchestrator, "tool_calls_used", 0)
        return fallback

    def get_tool_budget(self) -> int:
        """Get tool call budget (protocol method).

        Returns:
            Maximum allowed tool calls
        """
        if self._unified_tracker:
            budget: int = self._unified_tracker.tool_budget
            return budget
        fallback_budget: int = getattr(self._orchestrator, "tool_budget", 50)
        return fallback_budget

    def get_observed_files(self) -> set[str]:
        """Get files observed/read during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        return self._state_coordinator.observed_files

    def get_modified_files(self) -> set[str]:
        """Get files modified during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        if self._conversation_state and hasattr(self._conversation_state, "state"):
            modified: set[str] = set(getattr(self._conversation_state.state, "modified_files", []))
            return modified
        return set()

    def get_iteration_count(self) -> int:
        """Get current agent loop iteration count (protocol method).

        Returns:
            Non-negative iteration count
        """
        if self._unified_tracker:
            iterations: int = self._unified_tracker.iteration_count
            return iterations
        return 0

    def get_max_iterations(self) -> int:
        """Get maximum allowed iterations (protocol method).

        Returns:
            Max iteration limit
        """
        if self._unified_tracker:
            max_iter: int = self._unified_tracker.max_iterations
            return max_iter
        return 25

    # ========================================================================
    # ProviderProtocol
    # ========================================================================

    @property
    def current_provider(self) -> str:
        """Get current provider name (protocol property).

        Returns:
            Provider identifier (e.g., "anthropic", "openai")
        """
        return getattr(self._orchestrator, "provider_name", "unknown")

    @property
    def current_model(self) -> str:
        """Get current model name (protocol property).

        Returns:
            Model identifier
        """
        return getattr(self._orchestrator, "model", "unknown")

    async def switch_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        on_switch: Optional[Any] = None,
    ) -> bool:
        """Switch to a different provider/model (protocol method).

        Args:
            provider: Target provider name
            model: Optional specific model
            on_switch: Optional callback(provider, model) after switch

        Returns:
            True if switch was successful, False otherwise

        Raises:
            ProviderNotFoundError: If provider not found
        """
        result = await self._provider_coordinator._manager.switch_provider(
            provider_name=provider,
            model=model,
            orchestrator=self._orchestrator,
        )

        if result:
            # Sync orchestrator's model attribute with provider manager state
            self._orchestrator.model = self._provider_coordinator._manager.model
        # Return bool from SwitchResult
        return result.success if hasattr(result, "success") else bool(result)

    # ========================================================================
    # ToolsProtocol
    # ========================================================================

    def get_available_tools(self) -> set[str]:
        """Get all registered tool names (protocol method).

        Returns:
            Set of tool names available in registry
        """
        if self._tools:
            return set(self._tools.list_tools())
        return set()

    def get_enabled_tools(self) -> set[str]:
        """Get currently enabled tool names (protocol method).

        Returns:
            Set of enabled tool names for this session
        """
        if self._tool_access_config_coordinator:
            enabled: set[str] = self._tool_access_config_coordinator.get_enabled_tools(
                session_enabled_tools=getattr(self._orchestrator, "_enabled_tools", None),
            )
            return enabled
        return set()

    def set_enabled_tools(self, tools: set[str], tiered_config: Any = None) -> None:
        """Set which tools are enabled for this session (protocol method).

        Args:
            tools: Set of tool names to enable
            tiered_config: Optional TieredToolConfig to propagate for stage filtering.
        """
        if self._tool_access_config_coordinator:
            self._tool_access_config_coordinator.set_enabled_tools(
                tools=tools,
                session_enabled_tools_attr=getattr(self._orchestrator, "_enabled_tools", None),
                tool_selector=self._tool_selector,
                vertical_context=self._vertical_context,
                tiered_config=tiered_config,
            )
        else:
            # Fallback to direct implementation
            self._orchestrator._enabled_tools = tools

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled (protocol method).

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is enabled
        """
        if self._tool_access_config_coordinator:
            is_enabled: bool = self._tool_access_config_coordinator.is_tool_enabled(
                tool_name=tool_name,
                session_enabled_tools=getattr(self._orchestrator, "_enabled_tools", None),
            )
            return is_enabled
        enabled = self.get_enabled_tools()
        return tool_name in enabled

    # ========================================================================
    # SystemPromptProtocol
    # ========================================================================

    def get_system_prompt(self) -> str:
        """Get current system prompt (protocol method).

        Returns:
            Complete system prompt string
        """
        if self._prompt_builder:
            prompt: str = self._prompt_builder.build()
            return prompt
        return ""

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (protocol method).

        Args:
            prompt: New system prompt (replaces existing)
        """
        if self._prompt_builder and hasattr(self._prompt_builder, "set_custom_prompt"):
            self._prompt_builder.set_custom_prompt(prompt)

    def append_to_system_prompt(self, content: str) -> None:
        """Append content to system prompt (protocol method).

        Args:
            content: Content to append
        """
        current = self.get_system_prompt()
        self.set_system_prompt(current + "\n\n" + content)

    # ========================================================================
    # MessagesProtocol
    # ========================================================================

    def get_message_count(self) -> int:
        """Get message count (protocol method).

        Returns:
            Number of messages in conversation
        """
        if self._conversation is None:
            return 0
        messages = self._conversation.messages
        return len(messages) if messages is not None else 0

    # ========================================================================
    # Health Check Methods
    # ========================================================================

    def check_tool_selector_health(self) -> dict[str, Any]:
        """Check if tool selector is properly initialized.

        This health check prevents the critical bug where SemanticToolSelector
        was never initialized, blocking ALL chat functionality.

        Returns:
            Dictionary with health status:
                - healthy: bool - True if selector is ready
                - strategy: str - Tool selection strategy
                - initialized: bool - Whether embeddings are initialized
                - message: str - Status message
                - can_auto_recover: bool - Whether auto-recovery is possible
        """
        if not self._tool_selector or self._tool_selector is None:
            return {
                "healthy": False,
                "strategy": None,
                "initialized": False,
                "message": "Tool selector not created during initialization",
                "can_auto_recover": False,
            }

        # Get selector strategy
        strategy = getattr(self._tool_selector, "strategy", "unknown")
        strategy_name = strategy.value if hasattr(strategy, "value") else str(strategy)

        # Check if initialization is needed
        needs_init = hasattr(self._tool_selector, "initialize_tool_embeddings")
        is_initialized = (
            hasattr(self._tool_selector, "_embeddings_initialized")
            and self._tool_selector._embeddings_initialized
        )

        # If selector doesn't need initialization (e.g., keyword), it's healthy
        if not needs_init:
            return {
                "healthy": True,
                "strategy": strategy_name,
                "initialized": True,
                "message": f"Tool selector ready (strategy: {strategy_name})",
                "can_auto_recover": False,
            }

        # Semantic/hybrid selector - check initialization
        if is_initialized:
            return {
                "healthy": True,
                "strategy": strategy_name,
                "initialized": True,
                "message": f"Tool selector ready (strategy: {strategy_name}, embeddings initialized)",
                "can_auto_recover": False,
            }

        # Not initialized - this is the bug condition
        return {
            "healthy": False,
            "strategy": strategy_name,
            "initialized": False,
            "message": (
                f"Tool selector NOT initialized (strategy: {strategy_name}). "
                "This will cause ValueError on first tool selection. "
                "Call start_embedding_preload() or initialize_tool_embeddings() to fix."
            ),
            "can_auto_recover": True,
        }

    async def ensure_tool_selector_initialized(self) -> None:
        """Ensure tool selector is initialized before first use.

        This is a health check recovery mechanism that prevents the critical bug
        where SemanticToolSelector was never initialized.

        Should be called before chat() or stream_chat() if health check fails.

        Raises:
            RuntimeError: If initialization fails
        """
        health = self.check_tool_selector_health()
        if health["healthy"]:
            return

        if not health["can_auto_recover"]:
            raise RuntimeError(f"Tool selector cannot be recovered: {health['message']}")

        logger.debug(f"Tool selector initialization pending, auto-recovering: {health['message']}")

        # Attempt initialization
        if self._tool_selector and hasattr(self._tool_selector, "initialize_tool_embeddings"):
            try:
                await self._tool_selector.initialize_tool_embeddings(self._tools)
                logger.info("Tool selector auto-recovered successfully")
            except Exception as e:
                raise RuntimeError(f"Tool selector initialization failed: {e}") from e
        else:
            raise RuntimeError(
                "Tool selector cannot be initialized - no initialize_tool_embeddings method"
            )


def create_orchestrator_protocol_adapter(
    orchestrator: Any,
    state_coordinator: StateCoordinator,
    provider_coordinator: Any,
    tools: Optional[ToolRegistry] = None,
    conversation: Optional[Conversation] = None,
    mode_controller: Optional[AgentModeController] = None,
    unified_tracker: Optional[UnifiedTaskTracker] = None,
    tool_selector: Optional[ToolSelector] = None,
    tool_access_config_coordinator: Optional[ToolAccessConfigCoordinator] = None,
    vertical_context: Optional[VerticalContext] = None,
    conversation_state: Optional[ConversationStateMachine] = None,
    prompt_builder: Optional[SystemPromptBuilder] = None,
) -> OrchestratorProtocolAdapter:
    """Factory function to create an OrchestratorProtocolAdapter.

    Args:
        orchestrator: Reference to the orchestrator for fallback access
        state_coordinator: State management coordinator
        provider_coordinator: Provider switching coordinator
        tools: Tool registry (optional)
        conversation: Conversation object (optional)
        mode_controller: Optional mode management controller
        unified_tracker: Optional unified task tracker
        tool_selector: Optional tool selection component
        tool_access_config_coordinator: Optional tool access control coordinator
        vertical_context: Optional vertical context
        conversation_state: Optional conversation state machine
        prompt_builder: Optional system prompt builder

    Returns:
        Configured OrchestratorProtocolAdapter instance
    """
    return OrchestratorProtocolAdapter(
        orchestrator=orchestrator,
        state_coordinator=state_coordinator,
        provider_coordinator=provider_coordinator,
        tools=tools,
        conversation=conversation,
        mode_controller=mode_controller,
        unified_tracker=unified_tracker,
        tool_selector=tool_selector,
        tool_access_config_coordinator=tool_access_config_coordinator,
        vertical_context=vertical_context,
        conversation_state=conversation_state,
        prompt_builder=prompt_builder,
    )


__all__ = [
    "OrchestratorProtocolAdapter",
    "create_orchestrator_protocol_adapter",
]
