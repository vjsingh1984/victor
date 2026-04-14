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

"""Factory for creating coordinators with protocol-based dependencies.

This module provides a factory for creating coordinators with protocol-based
dependency injection, reducing direct coupling to AgentOrchestrator.

Design Patterns:
- Factory Pattern: Centralized object creation
- Dependency Injection: Inject protocol-based dependencies
- Builder Pattern: Construct complex objects step-by-step

Example:
    >>> from victor.agent.coordinators.coordinator_factory import CoordinatorFactory
    >>> from victor.core import get_container
    >>>
    >>> factory = CoordinatorFactory(get_container())
    >>> tool_coordinator = factory.create_tool_coordinator()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.agent.coordinators.protocol_dependencies import OrchestratorProtocolAdapter
from victor.agent.orchestrator_protocols import (
    IAgentOrchestrator,
    IToolExecutor,
    IMessageStore,
    IStateManager,
)

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer

logger = logging.getLogger(__name__)

__all__ = [
    "CoordinatorFactory",
]


class CoordinatorFactory:
    """
    Factory for creating coordinators with protocol-based dependencies.

    This factory resolves coordinator dependencies from the DI container
    and provides protocol-based adapters for orchestrator functionality.

    Supported Coordinators:
        - ToolCoordinator: Tool execution and management
        - SystemPromptCoordinator: System prompt building
        - MetricsCoordinator: Metrics collection
        - SafetyCoordinator: Safety checks
        - ConversationCoordinator: Conversation management

    Example:
        >>> factory = CoordinatorFactory(container)
        >>>
        >>> # Create tool coordinator with protocol dependencies
        >>> tool_coordinator = factory.create_tool_coordinator()
        >>>
        >>> # Create system prompt coordinator
        >>> prompt_coordinator = factory.create_system_prompt_coordinator()
    """

    def __init__(self, container: "ServiceContainer"):
        """
        Initialize factory.

        Args:
            container: DI container instance
        """
        self._container = container
        self._orchestrator_adapter: Optional[OrchestratorProtocolAdapter] = None

    def create_tool_coordinator(self) -> Any:
        """
        Create tool coordinator with protocol dependencies.

        Returns:
            ToolCoordinator instance with injected dependencies

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            from victor.agent.coordinators.tool_coordinator import ToolCoordinator

            # Resolve dependencies from container or create adapter
            tool_executor = self._resolve_tool_executor()
            message_store = self._resolve_message_store()

            # Create coordinator with protocol-based dependencies
            coordinator = ToolCoordinator(
                tool_executor=tool_executor,
                message_store=message_store,
            )

            logger.debug("CoordinatorFactory: Created ToolCoordinator")
            return coordinator

        except Exception as e:
            logger.error(f"CoordinatorFactory: Failed to create ToolCoordinator: {e}")
            raise RuntimeError(f"Failed to create ToolCoordinator: {e}") from e

    def create_system_prompt_coordinator(self) -> Any:
        """
        Create system prompt coordinator with protocol dependencies.

        Returns:
            SystemPromptCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            from victor.agent.coordinators.system_prompt_coordinator import SystemPromptCoordinator

            # SystemPromptCoordinator has minimal dependencies
            coordinator = SystemPromptCoordinator()

            logger.debug("CoordinatorFactory: Created SystemPromptCoordinator")
            return coordinator

        except Exception as e:
            logger.error(f"CoordinatorFactory: Failed to create SystemPromptCoordinator: {e}")
            raise RuntimeError(f"Failed to create SystemPromptCoordinator: {e}") from e

    def create_metrics_coordinator(self) -> Any:
        """
        Create metrics coordinator with protocol dependencies.

        Returns:
            MetricsCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator

            # Resolve orchestrator for metrics access
            orchestrator = self._resolve_orchestrator()

            # Create coordinator
            coordinator = MetricsCoordinator(orchestrator=orchestrator)

            logger.debug("CoordinatorFactory: Created MetricsCoordinator")
            return coordinator

        except Exception as e:
            logger.error(f"CoordinatorFactory: Failed to create MetricsCoordinator: {e}")
            raise RuntimeError(f"Failed to create MetricsCoordinator: {e}") from e

    def create_safety_coordinator(self) -> Any:
        """
        Create safety coordinator with protocol dependencies.

        Returns:
            SafetyCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            from victor.agent.coordinators.safety_coordinator import SafetyCoordinator

            # Resolve state manager for safety rules
            state_manager = self._resolve_state_manager()

            # Create coordinator
            coordinator = SafetyCoordinator(state_manager=state_manager)

            logger.debug("CoordinatorFactory: Created SafetyCoordinator")
            return coordinator

        except Exception as e:
            logger.error(f"CoordinatorFactory: Failed to create SafetyCoordinator: {e}")
            raise RuntimeError(f"Failed to create SafetyCoordinator: {e}") from e

    def create_conversation_coordinator(self) -> Any:
        """
        Create conversation coordinator with protocol dependencies.

        Returns:
            ConversationCoordinator instance

        Raises:
            RuntimeError: If coordinator creation fails
        """
        try:
            from victor.agent.coordinators.conversation_coordinator import ConversationCoordinator

            # Resolve message store
            message_store = self._resolve_message_store()
            state_manager = self._resolve_state_manager()

            # Create coordinator
            coordinator = ConversationCoordinator(
                message_store=message_store,
                state_manager=state_manager,
            )

            logger.debug("CoordinatorFactory: Created ConversationCoordinator")
            return coordinator

        except Exception as e:
            logger.error(f"CoordinatorFactory: Failed to create ConversationCoordinator: {e}")
            raise RuntimeError(f"Failed to create ConversationCoordinator: {e}") from e

    def _resolve_orchestrator(self) -> IAgentOrchestrator:
        """Resolve orchestrator from container or create adapter."""
        # Try container first
        orchestrator = self._container.get_optional(IAgentOrchestrator)
        if orchestrator is not None:
            return orchestrator

        # Create adapter from orchestrator in container
        if self._orchestrator_adapter is None:
            # Get orchestrator from container (by class lookup or type)
            from victor.agent.orchestrator import AgentOrchestrator

            orchestrator_impl = self._container.get_optional(AgentOrchestrator)
            if orchestrator_impl is not None:
                self._orchestrator_adapter = OrchestratorProtocolAdapter(orchestrator_impl)
            else:
                logger.warning("CoordinatorFactory: No orchestrator found in container")
                raise RuntimeError("Orchestrator not found in container")

        return self._orchestrator_adapter

    def _resolve_tool_executor(self) -> IToolExecutor:
        """Resolve tool executor from container or create adapter."""
        # Try container first
        executor = self._container.get_optional(IToolExecutor)
        if executor is not None:
            return executor

        # Create adapter from orchestrator
        orchestrator = self._resolve_orchestrator()
        if self._orchestrator_adapter is None:
            self._orchestrator_adapter = OrchestratorProtocolAdapter(orchestrator)

        return self._orchestrator_adapter

    def _resolve_message_store(self) -> IMessageStore:
        """Resolve message store from container or create adapter."""
        # Try container first
        store = self._container.get_optional(IMessageStore)
        if store is not None:
            return store

        # Create adapter from orchestrator
        orchestrator = self._resolve_orchestrator()
        if self._orchestrator_adapter is None:
            self._orchestrator_adapter = OrchestratorProtocolAdapter(orchestrator)

        return self._orchestrator_adapter

    def _resolve_state_manager(self) -> IStateManager:
        """Resolve state manager from container or create adapter."""
        # Try container first
        manager = self._container.get_optional(IStateManager)
        if manager is not None:
            return manager

        # Create adapter from orchestrator
        orchestrator = self._resolve_orchestrator()
        if self._orchestrator_adapter is None:
            self._orchestrator_adapter = OrchestratorProtocolAdapter(orchestrator)

        return self._orchestrator_adapter
