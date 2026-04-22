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
import warnings
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class OrchestrationFacade:
    """Groups top-level orchestration coordinators and integration components.

    Satisfies ``OrchestrationFacadeProtocol`` structurally.  The orchestrator
    creates this facade after all orchestration-domain components are
    initialized, passing references to the already-created instances.

    Components managed:
        - interaction_runtime: Interaction runtime boundary components
        - chat_service: Canonical chat service
        - tool_service: Canonical tool service
        - session_service: Canonical session service
        - context_service: Canonical context service
        - provider_service: Canonical provider service
        - recovery_service: Canonical recovery service
        - chat_coordinator: Deprecated chat coordinator compatibility shim
        - tool_coordinator: Deprecated tool coordinator compatibility shim
        - session_coordinator: Deprecated session coordinator compatibility shim
        - turn_executor: Execution coordinator for agentic loop
        - sync_chat_coordinator: Deprecated sync chat coordinator compatibility shim
        - streaming_chat_coordinator: Deprecated streaming chat coordinator compatibility shim
        - unified_chat_coordinator: Deprecated unified chat coordinator compatibility shim
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
        chat_service: Optional[Any] = None,
        tool_service: Optional[Any] = None,
        session_service: Optional[Any] = None,
        context_service: Optional[Any] = None,
        provider_service: Optional[Any] = None,
        recovery_service: Optional[Any] = None,
        chat_coordinator: Optional[Any] = None,
        deprecated_chat_coordinator: Optional[Any] = None,
        tool_coordinator: Optional[Any] = None,
        deprecated_tool_coordinator: Optional[Any] = None,
        get_tool_coordinator: Optional[Callable[[], Optional[Any]]] = None,
        session_coordinator: Optional[Any] = None,
        deprecated_session_coordinator: Optional[Any] = None,
        turn_executor: Optional[Any] = None,
        sync_chat_coordinator: Optional[Any] = None,
        deprecated_sync_chat_coordinator: Optional[Any] = None,
        streaming_chat_coordinator: Optional[Any] = None,
        deprecated_streaming_chat_coordinator: Optional[Any] = None,
        unified_chat_coordinator: Optional[Any] = None,
        deprecated_unified_chat_coordinator: Optional[Any] = None,
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
        if chat_coordinator is not None and deprecated_chat_coordinator is not None:
            raise TypeError(
                "Use only one of chat_coordinator or deprecated_chat_coordinator."
            )
        if tool_coordinator is not None and deprecated_tool_coordinator is not None:
            raise TypeError(
                "Use only one of tool_coordinator or deprecated_tool_coordinator."
            )
        if session_coordinator is not None and deprecated_session_coordinator is not None:
            raise TypeError(
                "Use only one of session_coordinator or deprecated_session_coordinator."
            )
        if sync_chat_coordinator is not None and deprecated_sync_chat_coordinator is not None:
            raise TypeError(
                "Use only one of sync_chat_coordinator or deprecated_sync_chat_coordinator."
            )
        if (
            streaming_chat_coordinator is not None
            and deprecated_streaming_chat_coordinator is not None
        ):
            raise TypeError(
                "Use only one of streaming_chat_coordinator or "
                "deprecated_streaming_chat_coordinator."
            )
        if unified_chat_coordinator is not None and deprecated_unified_chat_coordinator is not None:
            raise TypeError(
                "Use only one of unified_chat_coordinator or deprecated_unified_chat_coordinator."
            )

        resolved_deprecated_chat_coordinator = deprecated_chat_coordinator
        if chat_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade(chat_coordinator=...) is deprecated. "
                "Use deprecated_chat_coordinator=... or chat_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_deprecated_chat_coordinator = chat_coordinator

        resolved_deprecated_tool_coordinator = deprecated_tool_coordinator
        if tool_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade(tool_coordinator=...) is deprecated. "
                "Use deprecated_tool_coordinator=... or tool_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_deprecated_tool_coordinator = tool_coordinator

        resolved_deprecated_session_coordinator = deprecated_session_coordinator
        if session_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade(session_coordinator=...) is deprecated. "
                "Use deprecated_session_coordinator=... or session_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_deprecated_session_coordinator = session_coordinator

        resolved_deprecated_sync_chat_coordinator = deprecated_sync_chat_coordinator
        if sync_chat_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade(sync_chat_coordinator=...) is deprecated. "
                "Use deprecated_sync_chat_coordinator=... or chat_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_deprecated_sync_chat_coordinator = sync_chat_coordinator

        resolved_deprecated_streaming_chat_coordinator = (
            deprecated_streaming_chat_coordinator
        )
        if streaming_chat_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade(streaming_chat_coordinator=...) is deprecated. "
                "Use deprecated_streaming_chat_coordinator=... or chat_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_deprecated_streaming_chat_coordinator = streaming_chat_coordinator

        resolved_deprecated_unified_chat_coordinator = deprecated_unified_chat_coordinator
        if unified_chat_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade(unified_chat_coordinator=...) is deprecated. "
                "Use deprecated_unified_chat_coordinator=... or chat_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_deprecated_unified_chat_coordinator = unified_chat_coordinator

        self._interaction_runtime = interaction_runtime
        self._chat_service = chat_service
        self._tool_service = tool_service
        self._session_service = session_service
        self._context_service = context_service
        self._provider_service = provider_service
        self._recovery_service = recovery_service
        self._deprecated_chat_coordinator = resolved_deprecated_chat_coordinator
        self._deprecated_tool_coordinator = resolved_deprecated_tool_coordinator
        self._get_tool_coordinator = get_tool_coordinator
        self._deprecated_session_coordinator = resolved_deprecated_session_coordinator
        self._turn_executor = turn_executor
        self._deprecated_sync_chat_coordinator = resolved_deprecated_sync_chat_coordinator
        self._deprecated_streaming_chat_coordinator = (
            resolved_deprecated_streaming_chat_coordinator
        )
        self._deprecated_unified_chat_coordinator = resolved_deprecated_unified_chat_coordinator
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
    def chat_service(self) -> Optional[Any]:
        """Canonical chat service."""
        return self._chat_service

    @property
    def tool_service(self) -> Optional[Any]:
        """Canonical tool service."""
        return self._tool_service

    @property
    def session_service(self) -> Optional[Any]:
        """Canonical session service."""
        return self._session_service

    @property
    def context_service(self) -> Optional[Any]:
        """Canonical context service."""
        return self._context_service

    @property
    def provider_service(self) -> Optional[Any]:
        """Canonical provider service."""
        return self._provider_service

    @property
    def recovery_service(self) -> Optional[Any]:
        """Canonical recovery service."""
        return self._recovery_service

    @property
    def chat_coordinator(self) -> Optional[Any]:
        """Deprecated chat coordinator compatibility shim."""
        if self._deprecated_chat_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade.chat_coordinator is deprecated. "
                "Use OrchestrationFacade.chat_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._deprecated_chat_coordinator

    @property
    def tool_coordinator(self) -> Optional[Any]:
        """Deprecated tool coordinator compatibility shim."""
        if self._deprecated_tool_coordinator is None and self._get_tool_coordinator is not None:
            self._deprecated_tool_coordinator = self._get_tool_coordinator()
        if self._deprecated_tool_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade.tool_coordinator is deprecated. "
                "Use OrchestrationFacade.tool_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._deprecated_tool_coordinator

    @property
    def session_coordinator(self) -> Optional[Any]:
        """Deprecated session coordinator compatibility shim."""
        if self._deprecated_session_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade.session_coordinator is deprecated. "
                "Use OrchestrationFacade.session_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._deprecated_session_coordinator

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
        """Deprecated sync chat coordinator compatibility shim."""
        if self._deprecated_sync_chat_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade.sync_chat_coordinator is deprecated. "
                "Use OrchestrationFacade.chat_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._deprecated_sync_chat_coordinator

    @sync_chat_coordinator.setter
    def sync_chat_coordinator(self, value: Any) -> None:
        """Update the sync chat coordinator."""
        self._deprecated_sync_chat_coordinator = value

    @property
    def streaming_chat_coordinator(self) -> Optional[Any]:
        """Deprecated streaming chat coordinator compatibility shim."""
        if self._deprecated_streaming_chat_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade.streaming_chat_coordinator is deprecated. "
                "Use OrchestrationFacade.chat_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._deprecated_streaming_chat_coordinator

    @streaming_chat_coordinator.setter
    def streaming_chat_coordinator(self, value: Any) -> None:
        """Update the streaming chat coordinator."""
        self._deprecated_streaming_chat_coordinator = value

    @property
    def unified_chat_coordinator(self) -> Optional[Any]:
        """Deprecated unified chat coordinator compatibility shim."""
        if self._deprecated_unified_chat_coordinator is not None:
            warnings.warn(
                "OrchestrationFacade.unified_chat_coordinator is deprecated. "
                "Use OrchestrationFacade.chat_service instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._deprecated_unified_chat_coordinator

    @unified_chat_coordinator.setter
    def unified_chat_coordinator(self, value: Any) -> None:
        """Update the unified chat coordinator."""
        self._deprecated_unified_chat_coordinator = value

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
