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

"""Recovery and observability builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder
from victor.agent.coordinators.checkpoint_coordinator import CheckpointCoordinator
from victor.agent.coordinators.validation_coordinator import ValidationCoordinator
from victor.agent.coordinators.chat_coordinator import ChatCoordinator

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory


class RecoveryObservabilityBuilder(FactoryAwareBuilder):
    """Build recovery and observability components."""

    def __init__(self, settings, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(self, orchestrator: "AgentOrchestrator", **_kwargs) -> Dict[str, Any]:
        """Build recovery and observability components and attach them to orchestrator."""
        factory = self._ensure_factory()
        components: Dict[str, Any] = {}

        # Initialize RecoveryHandler for handling model failures and stuck states (via factory)
        orchestrator._recovery_handler = factory.create_recovery_handler()
        components["recovery_handler"] = orchestrator._recovery_handler

        # Create recovery integration submodule for clean delegation (via factory)
        orchestrator._recovery_integration = factory.create_recovery_integration(
            orchestrator._recovery_handler
        )
        components["recovery_integration"] = orchestrator._recovery_integration

        # Initialize RecoveryCoordinator for centralized recovery logic (via factory, DI)
        try:
            orchestrator._recovery_coordinator = factory.create_recovery_coordinator()
            components["recovery_coordinator"] = orchestrator._recovery_coordinator
        except Exception:  # pragma: no cover
            # RecoveryCoordinator may not be available in test environments
            orchestrator._recovery_coordinator = None
            components["recovery_coordinator"] = None

        # Initialize ChunkGenerator for centralized chunk generation (via factory, DI)
        try:
            orchestrator._chunk_generator = factory.create_chunk_generator()
            components["chunk_generator"] = orchestrator._chunk_generator
        except Exception:  # pragma: no cover
            # ChunkGenerator may not be available in test environments
            orchestrator._chunk_generator = None
            components["chunk_generator"] = None

        # Initialize ToolPlanner for centralized tool planning (via factory, DI)
        try:
            orchestrator._tool_planner = factory.create_tool_planner()
            components["tool_planner"] = orchestrator._tool_planner
        except Exception:  # pragma: no cover
            # ToolPlanner may not be available in test environments
            orchestrator._tool_planner = None
            components["tool_planner"] = None

        # Initialize TaskCoordinator for centralized task coordination (via factory, DI)
        try:
            orchestrator._task_coordinator = factory.create_task_coordinator()
            components["task_coordinator"] = orchestrator._task_coordinator
        except Exception:  # pragma: no cover
            # TaskCoordinator may not be available in test environments
            orchestrator._task_coordinator = None
            components["task_coordinator"] = None

        # Initialize ObservabilityIntegration for unified event bus (via factory)
        orchestrator._observability = factory.create_observability()
        components["observability"] = orchestrator._observability

        # Initialize ConversationCheckpointManager for time-travel debugging (via factory)
        orchestrator._checkpoint_manager = factory.create_checkpoint_manager()
        components["checkpoint_manager"] = orchestrator._checkpoint_manager

        # Initialize CheckpointCoordinator for checkpoint operations (SOLID refactoring)
        orchestrator._checkpoint_coordinator = CheckpointCoordinator(
            checkpoint_manager=orchestrator._checkpoint_manager,
            session_id=None,  # Will be set during session initialization
            get_state_fn=orchestrator._get_checkpoint_state,
            apply_state_fn=orchestrator._apply_checkpoint_state,
        )
        components["checkpoint_coordinator"] = orchestrator._checkpoint_coordinator

        # Initialize ChatCoordinator for chat and streaming operations (SOLID refactoring)
        orchestrator._chat_coordinator = ChatCoordinator(orchestrator=orchestrator)
        components["chat_coordinator"] = orchestrator._chat_coordinator

        # Initialize ValidationCoordinator for validation logic (SOLID refactoring)
        orchestrator._validation_coordinator = ValidationCoordinator(
            intelligent_integration=None,  # Will be set via lazy property
            context_manager=orchestrator._context_manager,
            response_coordinator=orchestrator._response_coordinator,
            cancel_event=getattr(orchestrator, "_cancel_event", None),
            metrics_coordinator=getattr(orchestrator, "_metrics_coordinator", None),
        )
        components["validation_coordinator"] = orchestrator._validation_coordinator

        self._register_components(components)
        return components
