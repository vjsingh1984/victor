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

"""Configuration and workflow optimization builder.

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder
from victor.agent.configuration_manager import create_configuration_manager
from victor.agent.memory_manager import create_memory_manager, create_session_recovery_manager
from victor.agent.streaming import ProgressMetrics
from victor.agent.coordinators.conversation_coordinator import ConversationCoordinator
from victor.agent.coordinators.search_coordinator import SearchCoordinator

import logging

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory

logger = logging.getLogger(__name__)


class ConfigWorkflowBuilder(FactoryAwareBuilder):
    """Build configuration, memory wrappers, and workflow optimizations."""

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(self, orchestrator: "AgentOrchestrator", **_kwargs: Any) -> Dict[str, Any]:
        """Build configuration and workflow optimization components."""
        factory = self._ensure_factory()
        components: Dict[str, Any] = {}

        # ConfigurationManager: Centralized configuration management
        # Handles tiered tool config and provides read-only config access
        orchestrator._configuration_manager = create_configuration_manager()
        components["configuration_manager"] = orchestrator._configuration_manager

        # MemoryManager: Unified memory operations interface
        # Wraps ConversationStore for cleaner session management
        orchestrator._memory_manager_wrapper = create_memory_manager(
            conversation_store=orchestrator.memory_manager,
            session_id=orchestrator._memory_session_id,
            message_history=orchestrator.conversation,
        )
        components["memory_manager_wrapper"] = orchestrator._memory_manager_wrapper

        # Initialize ConversationCoordinator for message management
        orchestrator._conversation_coordinator = ConversationCoordinator(
            conversation=orchestrator.conversation,
            lifecycle_manager=orchestrator._lifecycle_manager,
            memory_manager_wrapper=orchestrator._memory_manager_wrapper,
            usage_logger=(
                orchestrator.usage_logger if hasattr(orchestrator, "usage_logger") else None
            ),
        )
        components["conversation_coordinator"] = orchestrator._conversation_coordinator

        # Initialize SearchCoordinator for search query routing
        orchestrator._search_coordinator = SearchCoordinator(
            search_router=orchestrator.search_router
        )
        components["search_coordinator"] = orchestrator._search_coordinator

        # Initialize TeamCoordinator for team specification and suggestions
        # Note: Defer initialization until mode_workflow_team_coordinator is available
        orchestrator._team_coordinator = None
        components["team_coordinator"] = orchestrator._team_coordinator

        # Initialize workflow optimization components (via factory)
        orchestrator._workflow_optimization = factory.create_workflow_optimization_components(
            timeout_seconds=getattr(self.settings, "execution_timeout", None)
        )
        components["workflow_optimization"] = orchestrator._workflow_optimization

        # SessionRecoveryManager: Handles session recovery operations
        orchestrator._session_recovery_manager = create_session_recovery_manager(
            memory_manager=orchestrator._memory_manager_wrapper,
            lifecycle_manager=(
                orchestrator._lifecycle_manager
                if hasattr(orchestrator, "_lifecycle_manager")
                else None
            ),
        )
        components["session_recovery_manager"] = orchestrator._session_recovery_manager

        # Initialize ProgressMetrics for tracking exploration progress
        orchestrator._progress_metrics = ProgressMetrics()
        components["progress_metrics"] = orchestrator._progress_metrics

        # =========================================================================
        # Phase 5 Coordinators (Orchestrator Integration)
        # =========================================================================

        # Initialize ToolRetryCoordinator for tool execution with retry logic
        from victor.agent.coordinators.tool_retry_coordinator import ToolRetryCoordinator

        tool_retry_coordinator = factory.container.get_optional(ToolRetryCoordinator)
        if tool_retry_coordinator:
            # Wire up task completion detector if available
            if hasattr(orchestrator, "_task_completion_detector"):
                tool_retry_coordinator._task_completion_detector = (
                    orchestrator._task_completion_detector
                )
            components["tool_retry_coordinator"] = tool_retry_coordinator
        else:
            logger.debug("ToolRetryCoordinator not available in container")
            components["tool_retry_coordinator"] = None

        # Initialize MemoryCoordinator for memory management operations
        from victor.agent.coordinators.memory_coordinator import MemoryCoordinator

        memory_coordinator = factory.container.get_optional(MemoryCoordinator)
        if memory_coordinator:
            # Wire up memory manager and session ID
            memory_coordinator._memory_manager = orchestrator._memory_manager_wrapper
            memory_coordinator._session_id = orchestrator._memory_session_id
            # Wire up conversation store for fallback
            if hasattr(orchestrator, "conversation"):
                memory_coordinator._conversation_store = orchestrator.conversation
            components["memory_coordinator"] = memory_coordinator
        else:
            logger.debug("MemoryCoordinator not available in container")
            components["memory_coordinator"] = None

        # Initialize ToolCapabilityCoordinator for provider/model capability checks
        from victor.agent.coordinators.tool_capability_coordinator import ToolCapabilityCoordinator

        tool_capability_coordinator = factory.container.get_optional(ToolCapabilityCoordinator)
        if tool_capability_coordinator:
            # Wire up console for user-facing messages
            tool_capability_coordinator._console = orchestrator.console
            components["tool_capability_coordinator"] = tool_capability_coordinator
        else:
            logger.debug("ToolCapabilityCoordinator not available in container")
            components["tool_capability_coordinator"] = None

        # =========================================================================
        # End Phase 5 Coordinators
        # =========================================================================

        # Initialize token budget based on provider/model context window
        try:
            from victor.config.config_loaders import get_provider_limits

            provider_name = orchestrator.provider_name or ""
            model_name = orchestrator.model or ""

            logger.debug(
                f"Fetching provider limits for provider={provider_name}, model={model_name}"
            )

            provider_limits = get_provider_limits(
                provider=provider_name,
                model=model_name,
            )

            logger.debug(
                f"Got provider_limits: session_idle_timeout={provider_limits.session_idle_timeout}"
            )

            orchestrator._progress_metrics.initialize_token_budget(provider_limits.context_window)
            orchestrator._session_idle_timeout = provider_limits.session_idle_timeout
            logger.info(
                f"Initialized token budget with context_window={provider_limits.context_window:,} "
                f"(soft_limit={int(provider_limits.context_window * 0.3):,}, "
                f"hard_limit={int(provider_limits.context_window * 0.7):,}), "
                f"session_idle_timeout={orchestrator._session_idle_timeout}s"
            )
        except Exception as e:
            orchestrator._session_idle_timeout = getattr(self.settings, "session_idle_timeout", 180)
            logger.warning(f"Unable to initialize token budget, using default timeout: {e}")

        # Update the streaming handler with provider-specific session idle timeout
        if hasattr(orchestrator._streaming_handler, "session_idle_timeout"):
            old_timeout = orchestrator._streaming_handler.session_idle_timeout
            orchestrator._streaming_handler.session_idle_timeout = (
                orchestrator._session_idle_timeout
            )
            logger.info(
                f"Updated streaming handler session_idle_timeout: {old_timeout}s -> {orchestrator._session_idle_timeout}s"
            )
        else:
            logger.warning(
                "Streaming handler does not have session_idle_timeout attribute "
                f"(handler: {type(orchestrator._streaming_handler).__name__})"
            )

        self._register_components(components)
        return components
