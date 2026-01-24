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

"""Context and intelligence builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder
from victor.agent.context_manager import create_context_manager
from victor.agent.coordinators.evaluation_coordinator import EvaluationCoordinator
from victor.agent.task_analyzer import get_task_analyzer

import logging

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory

logger = logging.getLogger(__name__)


class ContextIntelligenceBuilder(FactoryAwareBuilder):
    """Build context management and intelligence components."""

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(self, orchestrator: "AgentOrchestrator", **_kwargs: Any) -> Dict[str, Any]:
        """Build context and intelligence components and attach them to orchestrator."""
        factory = self._ensure_factory()
        components: Dict[str, Any] = {}

        # TaskAnalyzer: Unified task analysis facade
        orchestrator._task_analyzer = get_task_analyzer()
        components["task_analyzer"] = orchestrator._task_analyzer

        # RLCoordinator: Framework-level RL with unified SQLite storage (via factory)
        orchestrator._rl_coordinator = factory.create_rl_coordinator()
        components["rl_coordinator"] = orchestrator._rl_coordinator

        # Get context pruning learner from RL coordinator (if available)
        pruning_learner = None
        if orchestrator._rl_coordinator is not None:
            try:
                pruning_learner = orchestrator._rl_coordinator.get_learner("context_pruning")
            except KeyError:
                logger.debug("context_pruning learner not registered in RL coordinator")
            except AttributeError:
                logger.debug("RL coordinator interface mismatch (missing get_learner)")

        # ContextCompactor: Proactive context management and tool result truncation (via factory)
        orchestrator._context_compactor = factory.create_context_compactor(
            conversation_controller=orchestrator._conversation_controller,
            pruning_learner=pruning_learner,
        )
        components["context_compactor"] = orchestrator._context_compactor

        # ContextManager: Centralized context window management (TD-002 refactoring)
        # Consolidates _get_model_context_window, _get_max_context_chars,
        # _check_context_overflow, and _handle_compaction
        orchestrator._context_manager = create_context_manager(
            provider_name=orchestrator.provider_name,
            model=orchestrator.model,
            conversation_controller=orchestrator._conversation_controller,
            context_compactor=orchestrator._context_compactor,
            debug_logger=orchestrator.debug_logger,
            settings=orchestrator.settings,
        )
        components["context_manager"] = orchestrator._context_manager

        # Initialize UsageAnalytics singleton for data-driven optimization (via factory)
        orchestrator._usage_analytics = factory.create_usage_analytics()
        components["usage_analytics"] = orchestrator._usage_analytics

        # Initialize ToolSequenceTracker for intelligent next-tool suggestions (via factory)
        orchestrator._sequence_tracker = factory.create_sequence_tracker()
        components["sequence_tracker"] = orchestrator._sequence_tracker

        # Initialize EvaluationCoordinator for evaluation and analytics (SOLID refactoring)
        # Extracts RL feedback, usage analytics, and intelligent outcome recording
        orchestrator._evaluation_coordinator = EvaluationCoordinator(
            usage_analytics=orchestrator._usage_analytics,
            sequence_tracker=orchestrator._sequence_tracker,
            get_rl_coordinator_fn=lambda: getattr(orchestrator, "_rl_coordinator", None),
            get_vertical_context_fn=lambda: getattr(orchestrator, "_vertical_context", None),
            get_stream_context_fn=lambda: getattr(orchestrator, "_current_stream_context", None),
            get_provider_fn=lambda: getattr(orchestrator, "provider", None),
            get_model_fn=lambda: getattr(orchestrator, "model", ""),
            get_tool_calls_used_fn=lambda: getattr(orchestrator, "tool_calls_used", 0),
            get_intelligent_integration_fn=lambda: getattr(
                orchestrator, "intelligent_integration", None
            ),
        )
        components["evaluation_coordinator"] = orchestrator._evaluation_coordinator

        # Initialize ToolOutputFormatter for LLM-context-aware output formatting (via factory)
        orchestrator._tool_output_formatter = factory.create_tool_output_formatter(
            orchestrator._context_compactor
        )
        components["tool_output_formatter"] = orchestrator._tool_output_formatter

        self._register_components(components)
        return components
