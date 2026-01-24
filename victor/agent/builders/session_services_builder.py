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

"""Session and core services builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder
from victor.agent.session_state_manager import create_session_state_manager
from victor.agent.task_completion import TaskCompletionDetector

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory

logger = logging.getLogger(__name__)


class SessionServicesBuilder(FactoryAwareBuilder):
    """Build session state, budgets, and core service wiring."""

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(self, orchestrator: "AgentOrchestrator", **_kwargs) -> Dict[str, Any]:
        """Build session state and core services and attach them to orchestrator."""
        factory = self._ensure_factory()
        components: Dict[str, Any] = {}

        # Initialize tool call budget using adapter recommendations (via factory)
        orchestrator.tool_budget = factory.initialize_tool_budget(
            orchestrator._tool_calling_caps_internal
        )
        components["tool_budget"] = orchestrator.tool_budget

        # Consolidated execution state tracking (TD-002)
        orchestrator._session_state = create_session_state_manager(
            tool_budget=orchestrator.tool_budget
        )
        components["session_state"] = orchestrator._session_state

        # Gap implementations via factory
        orchestrator.task_classifier = factory.create_complexity_classifier()
        orchestrator.intent_detector = factory.create_action_authorizer()
        orchestrator.search_router = factory.create_search_router()
        components["task_classifier"] = orchestrator.task_classifier
        components["intent_detector"] = orchestrator.intent_detector
        components["search_router"] = orchestrator.search_router

        # Presentation adapter for icon/emoji rendering (via factory, DI)
        orchestrator._presentation = factory.create_presentation_adapter()
        components["presentation_adapter"] = orchestrator._presentation

        # Task completion detection (signal-based)
        orchestrator._task_completion_detector = TaskCompletionDetector()
        components["task_completion_detector"] = orchestrator._task_completion_detector
        logger.info("TaskCompletionDetector initialized (signal-based completion)")

        # Context reminder manager (via factory, DI)
        orchestrator.reminder_manager = factory.create_reminder_manager(
            provider=orchestrator.provider_name,
            task_complexity="medium",
            tool_budget=orchestrator.tool_budget,
        )
        components["reminder_manager"] = orchestrator.reminder_manager

        self._register_components(components)
        return components
