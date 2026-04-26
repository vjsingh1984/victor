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

"""Service-owned protocol adapter for orchestration runtime state.

This module provides adapter classes that adapt AgentOrchestrator to various
protocol interfaces, enabling the Dependency Inversion Principle (DIP) across
the orchestration layer.

Design Pattern: Adapter Pattern
- OrchestratorProtocolAdapter: Adapts AgentOrchestrator to protocol interfaces
- Enables coordinators to depend on protocols rather than concrete classes
- Facilitates unit testing with lightweight protocol mocks

The legacy `victor.agent.coordinators.protocol_adapters` module now re-exports
this implementation for compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.message_history import MessageHistory
    from victor.agent.session_state_manager import SessionStateManager
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.services.tool_planning_runtime import ToolPlanner
    from victor.agent.tool_selection import ToolSelector
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider, CompletionResponse, Message

logger = logging.getLogger(__name__)


class OrchestratorProtocolAdapter:
    """Adapts AgentOrchestrator to protocol interfaces.

    The orchestrator implements multiple protocols through this adapter,
    allowing coordinators to depend on protocols rather than the
    concrete orchestrator class.

    This adapter implements:
    - ExecutionProvider: For executing model turns
    - ToolExecutor: For tool execution
    - MessageStore: For message storage
    - StateManager: For state management

    The adapter uses the existing orchestrator methods and properties,
    providing a protocol-based facade to the orchestrator's capabilities.
    """

    def __init__(self, orchestrator: "AgentOrchestrator") -> None:
        """Initialize the adapter with an orchestrator.

        Args:
            orchestrator: The AgentOrchestrator to adapt
        """
        self._orchestrator = orchestrator

    # =====================================================================
    # ExecutionProvider Implementation
    # =====================================================================

    async def execute_turn(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a single model turn.

        Delegates to the orchestrator's provider.chat() method.

        Args:
            messages: Conversation history
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse from the model
        """
        orch = self._orchestrator
        return await orch.provider.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )

    # =====================================================================
    # ToolExecutor Implementation
    # =====================================================================

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single tool with arguments.

        Delegates to the orchestrator's tool_executor.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Dict with tool execution result
        """
        orch = self._orchestrator
        context = {
            "code_manager": orch.code_manager,
            "provider": orch.provider,
            "model": orch.model,
            "tool_registry": orch.tools,
            "workflow_registry": orch.workflow_registry,
            "settings": orch.settings,
        }

        exec_result = await orch.tool_executor.execute(
            tool_name=tool_name,
            arguments=arguments,
            context=context,
        )

        return {
            "success": exec_result.success,
            "result": exec_result.result if exec_result.success else None,
            "error": exec_result.error if not exec_result.success else None,
        }

    # [LEGACY WRAPPER] Bridges to AgentOrchestrator._handle_tool_calls
    async def execute_tool_calls(
        self,
        tool_calls: List[Any],
    ) -> List[Dict[str, Any]]:
        """[LEGACY] Execute multiple tool calls via orchestrator bridge.

        Prefer IToolCoordinator for new implementations.
        """
        orch = self._orchestrator
        return await orch._handle_tool_calls(tool_calls)

    # =====================================================================
    # MessageStore Implementation
    # =====================================================================

    @property
    def messages(self) -> List[Message]:
        """Get current message history."""
        return self._orchestrator.messages

    def add_message(self, role: str, content: str) -> None:
        """Add message to history.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
        """
        self._orchestrator.add_message(role, content)

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from history.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        messages = self._orchestrator.messages
        if limit is not None:
            return messages[-limit:]
        return messages

    def clear_messages(self) -> None:
        """Clear message history."""
        self._orchestrator.messages.clear()

    @property
    def conversation(self) -> "MessageHistory":
        """Get conversation manager."""
        return self._orchestrator.conversation

    # =====================================================================
    # StateManager Implementation
    # =====================================================================

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value or default
        """
        return getattr(self._orchestrator, key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set state value.

        Args:
            key: State key
            value: State value
        """
        setattr(self._orchestrator, key, value)

    def delete_state(self, key: str) -> bool:
        """Delete state value.

        Args:
            key: State key

        Returns:
            True if key was found and deleted, False otherwise
        """
        if hasattr(self._orchestrator, key):
            delattr(self._orchestrator, key)
            return True
        return False

    def get_all_state(self) -> Dict[str, Any]:
        """Get all state values.

        Returns:
            Dictionary of all state key-value pairs
        """
        # Return commonly accessed state attributes
        return {
            "tool_calls_used": self._orchestrator.tool_calls_used,
            "tool_budget": self._orchestrator.tool_budget,
            "model": self._orchestrator.model,
            "temperature": self._orchestrator.temperature,
            "max_tokens": self._orchestrator.max_tokens,
            "_system_added": (
                self._orchestrator.get_capability_value("system_prompt_added")
                if self._orchestrator.has_capability("system_prompt_added")
                else False
            ),
        }

    def clear_state(self) -> None:
        """Clear all state values.

        Note: This only clears temporary state, not configuration.
        """
        self._orchestrator.tool_calls_used = 0

    # =====================================================================
    # ChatContextProtocol Helpers (from chat_protocols.py)
    # =====================================================================

    @property
    def settings(self) -> "Settings":
        """Get settings."""
        return self._orchestrator.settings

    @property
    def conversation_controller(self) -> "ConversationController":
        """Get conversation controller."""
        return self._orchestrator.conversation_controller

    @property
    def _context_compactor(self) -> "ContextCompactor":
        """Get context compactor."""
        return self._orchestrator._context_compactor

    @property
    def _session_state(self) -> "SessionStateManager":
        """Get session state manager."""
        return self._orchestrator._session_state

    @property
    def _cumulative_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage."""
        return self._orchestrator._cumulative_token_usage

    def _check_context_overflow(self, max_context: int) -> bool:
        """Check if conversation exceeds context window limit."""
        return self._orchestrator._check_context_overflow(max_context)

    def _get_max_context_chars(self) -> int:
        """Get maximum context length in characters."""
        return self._orchestrator._get_max_context_chars()

    # =====================================================================
    # ToolContextProtocol Helpers (from chat_protocols.py)
    # =====================================================================

    @property
    def tool_selector(self) -> "ToolSelector":
        """Get tool selector."""
        return self._orchestrator.tool_selector

    @property
    def tool_adapter(self) -> Any:
        """Get tool adapter."""
        return self._orchestrator.tool_adapter

    @property
    def _tool_planner(self) -> "ToolPlanner":
        """Get tool planner."""
        return self._orchestrator._tool_planner

    @property
    def tool_budget(self) -> int:
        """Get tool budget."""
        return self._orchestrator.tool_budget

    @property
    def tool_calls_used(self) -> int:
        """Get tool calls used."""
        return self._orchestrator.tool_calls_used

    @tool_calls_used.setter
    def tool_calls_used(self, value: int) -> None:
        """Set tool calls used."""
        self._orchestrator.tool_calls_used = value

    @property
    def use_semantic_selection(self) -> bool:
        """Get semantic selection flag."""
        return self._orchestrator.use_semantic_selection

    @property
    def observed_files(self) -> set:
        """Get observed files set."""
        return self._orchestrator.observed_files

    async def _handle_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """[LEGACY] Handle tool calls via orchestrator bridge."""
        return await self._orchestrator._handle_tool_calls(tool_calls)

    def _model_supports_tool_calls(self) -> bool:
        """Check if model supports tool calls."""
        return self._orchestrator._model_supports_tool_calls()

    # =====================================================================
    # ProviderContextProtocol Helpers (from chat_protocols.py)
    # =====================================================================

    @property
    def task_classifier(self) -> Any:
        """Get task classifier."""
        return self._orchestrator.task_classifier

    @property
    def task_analyzer(self) -> "TaskAnalyzer":
        """Get task analyzer."""
        return self._orchestrator.task_analyzer

    @property
    def response_completer(self) -> Any:
        """Get response completer."""
        return self._orchestrator.response_completer

    @property
    def provider(self) -> "BaseProvider":
        """Get LLM provider."""
        return self._orchestrator.provider

    @property
    def model(self) -> str:
        """Get current model."""
        return self._orchestrator.model

    @property
    def temperature(self) -> float:
        """Get temperature."""
        return self._orchestrator.temperature

    @property
    def max_tokens(self) -> int:
        """Get max tokens."""
        return self._orchestrator.max_tokens

    @property
    def thinking(self) -> Any:
        """Get thinking configuration."""
        return self._orchestrator.thinking

    @property
    def task_classifier(self) -> Any:
        """Get task classifier."""
        return self._orchestrator.task_classifier

    @property
    def task_analyzer(self) -> "TaskAnalyzer":
        """Get task analyzer."""
        return self._orchestrator.task_analyzer

    @property
    def response_completer(self) -> Any:
        """Get response completer."""
        return self._orchestrator.response_completer

    @property
    def _provider_service(self) -> Any:
        """Get canonical provider service for chat compatibility helpers."""
        return self._orchestrator._provider_service

    @property
    def _cancel_event(self) -> Optional["asyncio.Event"]:
        """Get cancel event."""
        return self._orchestrator._cancel_event

    @property
    def _is_streaming(self) -> bool:
        """Get streaming flag."""
        return self._orchestrator._is_streaming

    def _check_cancellation(self) -> bool:
        """Check if cancellation requested."""
        return self._orchestrator._check_cancellation()


__all__ = [
    "OrchestratorProtocolAdapter",
]
