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

"""Workflow and memory builder for orchestrator initialization.

Part of HIGH-005: Initialization Complexity reduction.
"""

from typing import Any, Optional, TYPE_CHECKING

from victor.agent.builders.base import FactoryAwareBuilder
from victor.agent.coordinators.workflow_coordinator import WorkflowCoordinator

import logging

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory

logger = logging.getLogger(__name__)


class WorkflowMemoryBuilder(FactoryAwareBuilder):
    """Build workflow registry, conversation memory, and state components."""

    def __init__(self, settings: Any, factory: Optional["OrchestratorFactory"] = None):
        """Initialize the builder.

        Args:
            settings: Application settings
            factory: Optional OrchestratorFactory instance
        """
        super().__init__(settings, factory)

    def build(self, orchestrator: "AgentOrchestrator", **_kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Build workflow and memory components and attach them to orchestrator."""
        factory = self._ensure_factory()
        components: dict[str, Any] = {}

        # Result cache for pure/idempotent tools (via factory)
        orchestrator.tool_cache = factory.create_tool_cache()
        components["tool_cache"] = orchestrator.tool_cache

        # Minimal dependency graph (used for planning search→read→analyze) (via factory, DI)
        # Tool dependencies are registered via ToolRegistrar after it's created
        orchestrator.tool_graph = factory.create_tool_dependency_graph()
        components["tool_graph"] = orchestrator.tool_graph

        # Code execution manager for Docker-based code execution (via factory, DI with fallback)
        orchestrator.code_manager = factory.create_code_execution_manager()
        components["code_manager"] = orchestrator.code_manager

        # Workflow registry (via factory, DI with fallback)
        orchestrator.workflow_registry = factory.create_workflow_registry()
        components["workflow_registry"] = orchestrator.workflow_registry

        # Initialize WorkflowCoordinator for workflow operations (SOLID refactoring)
        orchestrator._workflow_coordinator = WorkflowCoordinator(
            workflow_registry=orchestrator.workflow_registry,
        )
        orchestrator._register_default_workflows()

        # Conversation history (via factory) - MessageHistory for better encapsulation
        orchestrator.conversation = factory.create_message_history(orchestrator._system_prompt)
        components["conversation"] = orchestrator.conversation

        # Persistent conversation memory with SQLite backing (via factory)
        # Provides session recovery, token-aware pruning, and multi-turn context retention
        memory_result = factory.create_memory_components(
            orchestrator.provider_name,
            orchestrator._tool_calling_caps_internal.native_tool_calls,
        )
        orchestrator.memory_manager, orchestrator._memory_session_id = memory_result  # type: ignore[assignment]
        components["memory_manager"] = orchestrator.memory_manager
        components["memory_session_id"] = orchestrator._memory_session_id

        # Initialize LanceDB embedding store for efficient semantic retrieval if memory enabled
        if orchestrator.memory_manager and getattr(
            self.settings, "conversation_embeddings_enabled", True
        ):
            try:
                orchestrator._init_conversation_embedding_store()
            except ImportError as embed_err:
                logger.debug(f"ConversationEmbeddingStore dependencies not available: {embed_err}")
            except (OSError, IOError) as embed_err:
                logger.warning(
                    f"Failed to initialize ConversationEmbeddingStore (I/O error): {embed_err}"
                )
            except (ValueError, TypeError) as embed_err:
                logger.warning(
                    f"Failed to initialize ConversationEmbeddingStore (config error): {embed_err}"
                )

        # Conversation state machine for intelligent stage detection (via factory, DI with fallback)
        orchestrator.conversation_state = factory.create_conversation_state_machine()
        components["conversation_state"] = orchestrator.conversation_state

        # StateCoordinator for unified state management (via factory)
        # Consolidates access to SessionStateManager, ConversationStateMachine, and checkpoint state
        # Provides state change notifications (Observer pattern) and state history tracking
        orchestrator._state_coordinator = factory.create_state_coordinator(
            session_state_manager=orchestrator._session_state if orchestrator._session_state else None,  # type: ignore[arg-type]
            conversation_state_machine=orchestrator.conversation_state,
        )
        components["state_coordinator"] = orchestrator._state_coordinator

        # Intent classifier for semantic continuation/completion detection (via factory)
        orchestrator.intent_classifier = factory.create_intent_classifier()
        components["intent_classifier"] = orchestrator.intent_classifier

        self._register_components(components)
        return components
