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

"""Orchestration domain facade for orchestrator decomposition.

Groups coordinators, protocol adapters, streaming handlers, intelligent
pipeline integration, and subagent orchestration components behind a
single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OrchestrationFacade:
    """Groups top-level orchestration coordinators and integration components.

    Satisfies ``OrchestrationFacadeProtocol`` structurally.  The orchestrator
    creates this facade after all orchestration-domain components are
    initialized, passing references to the already-created instances.

    Components managed:
        - interaction_runtime: Interaction runtime boundary components
        - chat_coordinator: Chat coordinator for chat operations
        - tool_coordinator: Tool coordinator for tool operations
        - session_coordinator: Session coordinator for session operations
        - turn_executor: Execution coordinator for agentic loop
        - sync_chat_coordinator: Sync chat coordinator (non-streaming)
        - streaming_chat_coordinator: Streaming chat coordinator
        - unified_chat_coordinator: Unified chat coordinator facade
        - protocol_adapter: Protocol adapter for coordinator communication
        - streaming_handler: StreamingChatHandler for streaming loop logic
        - streaming_controller: StreamingController for session/metrics
        - streaming_coordinator: StreamingCoordinator for response processing
        - iteration_coordinator: IterationCoordinator for loop control
        - task_analyzer: Unified task analysis facade
        - presentation: Presentation adapter for icon/emoji rendering
        - vertical_integration_adapter: VerticalIntegrationAdapter
        - vertical_context: VerticalContext for unified vertical state
        - observability: ObservabilityIntegration for unified event bus
        - execution_tracer: Execution tracing (if available)
        - tool_call_tracer: Tool call tracing (if available)
        - intelligent_integration: Intelligent pipeline integration
        - subagent_orchestrator: Sub-agent orchestration
    """

    def __init__(
        self,
        *,
        interaction_runtime: Optional[Any] = None,
        chat_coordinator: Optional[Any] = None,
        tool_coordinator: Optional[Any] = None,
        session_coordinator: Optional[Any] = None,
        turn_executor: Optional[Any] = None,
        sync_chat_coordinator: Optional[Any] = None,
        streaming_chat_coordinator: Optional[Any] = None,
        unified_chat_coordinator: Optional[Any] = None,
        protocol_adapter: Optional[Any] = None,
        streaming_handler: Optional[Any] = None,
        streaming_controller: Optional[Any] = None,
        streaming_coordinator: Optional[Any] = None,
        iteration_coordinator: Optional[Any] = None,
        task_analyzer: Optional[Any] = None,
        presentation: Optional[Any] = None,
        vertical_integration_adapter: Optional[Any] = None,
        vertical_context: Optional[Any] = None,
        observability: Optional[Any] = None,
        execution_tracer: Optional[Any] = None,
        tool_call_tracer: Optional[Any] = None,
        intelligent_integration: Optional[Any] = None,
        subagent_orchestrator: Optional[Any] = None,
    ) -> None:
        self._interaction_runtime = interaction_runtime
        self._chat_coordinator = chat_coordinator
        self._tool_coordinator = tool_coordinator
        self._session_coordinator = session_coordinator
        self._turn_executor = turn_executor
        self._sync_chat_coordinator = sync_chat_coordinator
        self._streaming_chat_coordinator = streaming_chat_coordinator
        self._unified_chat_coordinator = unified_chat_coordinator
        self._protocol_adapter = protocol_adapter
        self._streaming_handler = streaming_handler
        self._streaming_controller = streaming_controller
        self._streaming_coordinator = streaming_coordinator
        self._iteration_coordinator = iteration_coordinator
        self._task_analyzer = task_analyzer
        self._presentation = presentation
        self._vertical_integration_adapter = vertical_integration_adapter
        self._vertical_context = vertical_context
        self._observability = observability
        self._execution_tracer = execution_tracer
        self._tool_call_tracer = tool_call_tracer
        self._intelligent_integration = intelligent_integration
        self._subagent_orchestrator = subagent_orchestrator

        logger.debug(
            "OrchestrationFacade initialized (protocol_adapter=%s, observability=%s, "
            "subagent=%s)",
            protocol_adapter is not None,
            observability is not None,
            subagent_orchestrator is not None,
        )

    # ------------------------------------------------------------------
    # Properties (satisfy OrchestrationFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def interaction_runtime(self) -> Optional[Any]:
        """Interaction runtime boundary components."""
        return self._interaction_runtime

    @property
    def chat_coordinator(self) -> Optional[Any]:
        """Chat coordinator for chat operations."""
        return self._chat_coordinator

    @property
    def tool_coordinator(self) -> Optional[Any]:
        """Tool coordinator for tool operations."""
        return self._tool_coordinator

    @property
    def session_coordinator(self) -> Optional[Any]:
        """Session coordinator for session operations."""
        return self._session_coordinator

    @property
    def turn_executor(self) -> Optional[Any]:
        """Execution coordinator for agentic loop."""
        return self._turn_executor

    @turn_executor.setter
    def turn_executor(self, value: Any) -> None:
        """Update the execution coordinator."""
        self._turn_executor = value

    @property
    def sync_chat_coordinator(self) -> Optional[Any]:
        """Sync chat coordinator (non-streaming)."""
        return self._sync_chat_coordinator

    @sync_chat_coordinator.setter
    def sync_chat_coordinator(self, value: Any) -> None:
        """Update the sync chat coordinator."""
        self._sync_chat_coordinator = value

    @property
    def streaming_chat_coordinator(self) -> Optional[Any]:
        """Streaming chat coordinator."""
        return self._streaming_chat_coordinator

    @streaming_chat_coordinator.setter
    def streaming_chat_coordinator(self, value: Any) -> None:
        """Update the streaming chat coordinator."""
        self._streaming_chat_coordinator = value

    @property
    def unified_chat_coordinator(self) -> Optional[Any]:
        """Unified chat coordinator facade."""
        return self._unified_chat_coordinator

    @unified_chat_coordinator.setter
    def unified_chat_coordinator(self, value: Any) -> None:
        """Update the unified chat coordinator."""
        self._unified_chat_coordinator = value

    @property
    def protocol_adapter(self) -> Optional[Any]:
        """Protocol adapter for coordinator communication."""
        return self._protocol_adapter

    @protocol_adapter.setter
    def protocol_adapter(self, value: Any) -> None:
        """Update the protocol adapter."""
        self._protocol_adapter = value

    @property
    def streaming_handler(self) -> Optional[Any]:
        """StreamingChatHandler for streaming loop logic."""
        return self._streaming_handler

    @property
    def streaming_controller(self) -> Optional[Any]:
        """StreamingController for session/metrics management."""
        return self._streaming_controller

    @property
    def streaming_coordinator(self) -> Optional[Any]:
        """StreamingCoordinator for response processing."""
        return self._streaming_coordinator

    @property
    def iteration_coordinator(self) -> Optional[Any]:
        """IterationCoordinator for loop control."""
        return self._iteration_coordinator

    @iteration_coordinator.setter
    def iteration_coordinator(self, value: Any) -> None:
        """Update the iteration coordinator."""
        self._iteration_coordinator = value

    @property
    def task_analyzer(self) -> Optional[Any]:
        """Unified task analysis facade."""
        return self._task_analyzer

    @property
    def presentation(self) -> Optional[Any]:
        """Presentation adapter for icon/emoji rendering."""
        return self._presentation

    @property
    def vertical_integration_adapter(self) -> Optional[Any]:
        """VerticalIntegrationAdapter for single-source vertical methods."""
        return self._vertical_integration_adapter

    @property
    def vertical_context(self) -> Optional[Any]:
        """VerticalContext for unified vertical state management."""
        return self._vertical_context

    @property
    def observability(self) -> Optional[Any]:
        """ObservabilityIntegration for unified event bus."""
        return self._observability

    @observability.setter
    def observability(self, value: Any) -> None:
        """Update the observability integration."""
        self._observability = value

    @property
    def execution_tracer(self) -> Optional[Any]:
        """Execution tracing (if available)."""
        return self._execution_tracer

    @property
    def tool_call_tracer(self) -> Optional[Any]:
        """Tool call tracing (if available)."""
        return self._tool_call_tracer

    @property
    def intelligent_integration(self) -> Optional[Any]:
        """Intelligent pipeline integration."""
        return self._intelligent_integration

    @intelligent_integration.setter
    def intelligent_integration(self, value: Any) -> None:
        """Update the intelligent integration."""
        self._intelligent_integration = value

    @property
    def subagent_orchestrator(self) -> Optional[Any]:
        """Sub-agent orchestration."""
        return self._subagent_orchestrator

    @subagent_orchestrator.setter
    def subagent_orchestrator(self, value: Any) -> None:
        """Update the subagent orchestrator."""
        self._subagent_orchestrator = value
