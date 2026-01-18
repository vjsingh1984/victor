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

"""Finalization builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder
from victor.agent.vertical_context import create_vertical_context
from victor.agent.vertical_integration_adapter import VerticalIntegrationAdapter

import logging

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory

logger = logging.getLogger(__name__)


class FinalizationBuilder(FactoryAwareBuilder):
    """Finalize orchestration wiring, vertical integration, and lifecycle hooks."""

    def __init__(self, settings, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(self, orchestrator: "AgentOrchestrator", **_kwargs) -> Dict[str, Any]:
        """Finalize component wiring and lifecycle configuration."""
        factory = self._ensure_factory()
        components: Dict[str, Any] = {}

        # Wire component dependencies (via factory)
        factory.wire_component_dependencies(
            recovery_handler=orchestrator._recovery_handler,
            context_compactor=orchestrator._context_compactor,
            observability=orchestrator._observability,
            conversation_state=orchestrator.conversation_state,
        )

        # Initialize VerticalContext for unified vertical state management
        orchestrator._vertical_context = create_vertical_context()
        components["vertical_context"] = orchestrator._vertical_context

        # Initialize VerticalIntegrationAdapter for single-source vertical methods
        orchestrator._vertical_integration_adapter = VerticalIntegrationAdapter(orchestrator)
        components["vertical_integration_adapter"] = orchestrator._vertical_integration_adapter

        # Initialize ModeWorkflowTeamCoordinator for intelligent team/workflow suggestions
        orchestrator._mode_workflow_team_coordinator = None
        components["mode_workflow_team_coordinator"] = orchestrator._mode_workflow_team_coordinator

        # Initialize capability registry for explicit capability discovery
        orchestrator.__init_capability_registry__()

        # Wire up LifecycleManager with dependencies for shutdown
        orchestrator._lifecycle_manager.set_provider(orchestrator.provider)
        orchestrator._lifecycle_manager.set_code_manager(
            orchestrator.code_manager if hasattr(orchestrator, "code_manager") else None
        )
        # semantic_selector is no longer used - replaced by unified tool selector strategy factory
        orchestrator._lifecycle_manager.set_usage_logger(
            orchestrator.usage_logger if hasattr(orchestrator, "usage_logger") else None
        )
        # Note: background_tasks is a set, convert to list for lifecycle manager
        orchestrator._lifecycle_manager.set_background_tasks(list(orchestrator._background_tasks))
        # Set callbacks for orchestrator-specific shutdown logic
        orchestrator._lifecycle_manager.set_flush_analytics_callback(orchestrator.flush_analytics)
        orchestrator._lifecycle_manager.set_stop_health_monitoring_callback(
            orchestrator.stop_health_monitoring
        )

        logger.info(
            "Orchestrator initialized with decomposed components: "
            "ConversationController, ToolPipeline, StreamingController, StreamingChatHandler, "
            "TaskAnalyzer, ContextCompactor, UsageAnalytics, ToolSequenceTracker, "
            "ToolOutputFormatter, RecoveryCoordinator, ChunkGenerator, ToolPlanner, TaskCoordinator, "
            "ObservabilityIntegration, WorkflowOptimization, VerticalContext, ModeWorkflowTeamCoordinator, "
            "CapabilityRegistry"
        )

        self._register_components(components)
        return components
