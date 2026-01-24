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

"""Conversation pipeline builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory
    from victor.providers.base import BaseProvider


class ConversationPipelineBuilder(FactoryAwareBuilder):
    """Build conversation controller, lifecycle manager, and streaming pipeline."""

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(
        self,
        orchestrator: "AgentOrchestrator",
        provider: "BaseProvider",
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build conversation pipeline components and attach them to orchestrator."""
        factory = self._ensure_factory(provider=provider, model=model)
        components: Dict[str, Any] = {}

        # ConversationController: Manages message history and conversation state (via factory)
        orchestrator._conversation_controller = factory.create_conversation_controller(
            provider=provider,
            model=model,
            conversation=orchestrator.conversation,
            conversation_state=orchestrator.conversation_state,
            memory_manager=orchestrator.memory_manager,
            memory_session_id=orchestrator._memory_session_id,
            system_prompt=orchestrator._system_prompt,
        )
        components["conversation_controller"] = orchestrator._conversation_controller

        # LifecycleManager: Coordinate session lifecycle and resource cleanup (via factory)
        orchestrator._lifecycle_manager = factory.create_lifecycle_manager(
            conversation_controller=orchestrator._conversation_controller,
            metrics_collector=(
                orchestrator._metrics_collector
                if hasattr(orchestrator, "_metrics_collector")
                else None
            ),
            context_compactor=(
                orchestrator._context_compactor
                if hasattr(orchestrator, "_context_compactor")
                else None
            ),
            sequence_tracker=(
                orchestrator._sequence_tracker
                if hasattr(orchestrator, "_sequence_tracker")
                else None
            ),
            usage_analytics=(
                orchestrator._usage_analytics if hasattr(orchestrator, "_usage_analytics") else None
            ),
            reminder_manager=(
                orchestrator._reminder_manager
                if hasattr(orchestrator, "_reminder_manager")
                else None
            ),
        )
        components["lifecycle_manager"] = orchestrator._lifecycle_manager

        # Tool deduplication tracker for preventing redundant calls (via factory)
        orchestrator._deduplication_tracker = factory.create_tool_deduplication_tracker()
        components["deduplication_tracker"] = orchestrator._deduplication_tracker

        # ToolPipeline: Coordinates tool execution flow (via factory)
        orchestrator._tool_pipeline = factory.create_tool_pipeline(
            tools=orchestrator.tools,
            tool_executor=orchestrator.tool_executor,
            tool_budget=orchestrator.tool_budget,
            tool_cache=orchestrator.tool_cache,
            argument_normalizer=orchestrator.argument_normalizer,
            on_tool_start=orchestrator._on_tool_start_callback,
            on_tool_complete=orchestrator._on_tool_complete_callback,
            deduplication_tracker=orchestrator._deduplication_tracker,
            middleware_chain=orchestrator._middleware_chain,
        )
        components["tool_pipeline"] = orchestrator._tool_pipeline

        # Wire pending semantic cache to tool pipeline (deferred from embedding store init)
        if (
            hasattr(orchestrator, "_pending_semantic_cache")
            and orchestrator._pending_semantic_cache is not None
        ):
            orchestrator._tool_pipeline.set_semantic_cache(orchestrator._pending_semantic_cache)
            logger.info("[AgentOrchestrator] Semantic tool result cache enabled")
            orchestrator._pending_semantic_cache = None  # Clear reference

        # StreamingController: Manages streaming sessions and metrics (via factory)
        orchestrator._streaming_controller = factory.create_streaming_controller(
            streaming_metrics_collector=orchestrator.streaming_metrics_collector,
            on_session_complete=orchestrator._on_streaming_session_complete,
        )
        components["streaming_controller"] = orchestrator._streaming_controller

        # StreamingCoordinator: Coordinates streaming response processing (via factory)
        orchestrator._streaming_coordinator = factory.create_streaming_coordinator(
            streaming_controller=orchestrator._streaming_controller,
        )
        components["streaming_coordinator"] = orchestrator._streaming_coordinator

        # StreamingChatHandler: Testable extraction of streaming loop logic (via factory)
        orchestrator._streaming_handler = factory.create_streaming_chat_handler(
            message_adder=orchestrator
        )
        components["streaming_handler"] = orchestrator._streaming_handler

        # IterationCoordinator: Loop control for streaming chat (using handler)
        orchestrator._iteration_coordinator = None
        components["iteration_coordinator"] = orchestrator._iteration_coordinator

        self._register_components(components)
        return components
