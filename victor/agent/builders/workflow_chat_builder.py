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

"""Workflow chat builder for orchestrator initialization.

Part of Phase 1: Domain-Agnostic Workflow Chat
Creates WorkflowOrchestrator and WorkflowChatCoordinator for workflow-based chat execution.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

import logging

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory

logger = logging.getLogger(__name__)


class WorkflowChatBuilder:
    """Build workflow chat components for domain-agnostic chat execution.

    This builder creates the WorkflowOrchestrator and related components
    when the feature flag is enabled. The workflow chat system provides
    a domain-agnostic way to execute chat workflows through the framework
    workflow engine.

    Feature Flag:
        VICTOR_USE_WORKFLOW_CHAT=true enables workflow-based chat execution
    """

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        self.settings = settings
        self._factory = factory

    def build(self, orchestrator: "AgentOrchestrator", **_kwargs: Any) -> Dict[str, Any]:
        """Build workflow chat components and attach them to orchestrator.

        This method conditionally creates workflow chat components based on
        the VICTOR_USE_WORKFLOW_CHAT feature flag. When enabled, it creates:
        - WorkflowOrchestrator: Domain-agnostic workflow execution
        - WorkflowChatCoordinator: Workflow-based chat coordination

        Args:
            orchestrator: The AgentOrchestrator instance
            **_kwargs: Additional keyword arguments (ignored)

        Returns:
            Dictionary of built components
        """
        components: Dict[str, Any] = {}

        # Check feature flag
        use_workflow_chat = getattr(self.settings, "use_workflow_chat", False)

        if not use_workflow_chat:
            logger.debug("Workflow chat disabled via VICTOR_USE_WORKFLOW_CHAT=false")
            return components

        try:
            # Import workflow chat components
            from victor.framework.workflow_orchestrator import WorkflowOrchestrator
            from victor.framework.coordinators.workflow_chat_coordinator import (
                WorkflowChatCoordinator,
                ChatExecutionConfig,
            )
            from victor.framework.coordinators.graph_coordinator import (
                GraphExecutionCoordinator,
            )

            # Create GraphExecutionCoordinator if not already present
            if not hasattr(orchestrator, "_graph_coordinator"):
                orchestrator._graph_coordinator = GraphExecutionCoordinator(
                    runner_registry=None,  # Can be configured later
                )
            components["graph_coordinator"] = orchestrator._graph_coordinator

            # Create WorkflowChatCoordinator
            chat_config = ChatExecutionConfig(
                max_iterations=getattr(self.settings, "workflow_max_iterations", 50),
                enable_streaming=True,
                enable_checkpoints=getattr(self.settings, "workflow_checkpoints_enabled", True),
                timeout_seconds=getattr(self.settings, "workflow_timeout_seconds", 300),
            )

            orchestrator._workflow_chat_coordinator = WorkflowChatCoordinator(
                workflow_registry=orchestrator.workflow_registry,
                graph_coordinator=orchestrator._graph_coordinator,
                config=chat_config,
            )
            components["workflow_chat_coordinator"] = orchestrator._workflow_chat_coordinator

            # Create WorkflowOrchestrator
            orchestrator._workflow_orchestrator = WorkflowOrchestrator(
                workflow_coordinator=orchestrator._workflow_coordinator,
                graph_coordinator=orchestrator._graph_coordinator,
            )
            components["workflow_orchestrator"] = orchestrator._workflow_orchestrator

            logger.info(
                f"Workflow chat enabled: WorkflowOrchestrator created with "
                f"{len(orchestrator._workflow_orchestrator.get_available_workflows())} workflows"
            )

        except ImportError as e:
            logger.warning(f"Failed to import workflow chat components: {e}")
        except Exception as e:
            logger.error(f"Failed to create workflow chat components: {e}", exc_info=True)

        return components


__all__ = ["WorkflowChatBuilder"]
