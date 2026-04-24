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

"""Service-owned compatibility shim for the deprecated ChatCoordinator.

`ChatCoordinator` no longer owns the live chat runtime. The canonical path is
`victor.agent.services.chat_service.ChatService`, with streaming owned by the
service runtime layer.

This implementation lives under `victor.agent.services` so coordinator modules
can shrink to explicit deprecated re-export shims while external callers
migrate away from coordinator/facade-first access patterns.
"""

import logging
import warnings
from typing import Any, AsyncIterator, Callable, Optional, TYPE_CHECKING

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.providers.base import CompletionResponse, StreamChunk

if TYPE_CHECKING:
    # Type-only imports
    from victor.agent.streaming.context import StreamingChatContext
    from victor.agent.services.protocols.chat_runtime import ChatOrchestratorProtocol
    from victor.agent.streaming.intent_classification import IntentClassificationHandler
    from victor.agent.streaming.continuation import ContinuationHandler
    from victor.agent.streaming.tool_execution import ToolExecutionHandler
    from victor.agent.services.planning_runtime import PlanningCoordinator
    from victor.agent.streaming.pipeline import StreamingChatPipeline
    from victor.agent.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class ChatCoordinator(ChatStreamHelperMixin):
    """[DEPRECATED - Use ChatService] Adapter for backward compatibility.

    **WARNING**: This class is deprecated and will be removed in a future release.
    All new code should use ``victor.agent.services.ChatService`` instead.

    Migration guide:
    - Old: ``ChatCoordinator(orchestrator)``
    - New: ``ChatService(config, provider_service, tool_service, ...)``

    This coordinator now acts as an adapter that delegates to ChatService
    internally while maintaining the old interface for backward compatibility.

    The coordinator depends on ``ChatOrchestratorProtocol`` (hosted in
    ``victor.agent.services.protocols.chat_runtime``) rather than the concrete
    ``AgentOrchestrator``.
    This enables unit testing with lightweight mocks.

    Args:
        orchestrator: Any object satisfying ChatOrchestratorProtocol
    """

    def __init__(
        self,
        orchestrator: "ChatOrchestratorProtocol",
        token_tracker: Optional["TokenTracker"] = None,
    ) -> None:
        """Initialize the ChatCoordinator with deprecation warning.

        Args:
            orchestrator: Object satisfying ChatOrchestratorProtocol
            token_tracker: Optional centralized token tracker. When provided,
                streaming token usage is accumulated through the tracker
                instead of direct dict mutation on the orchestrator.
        """
        warnings.warn(
            "ChatCoordinator is deprecated. Use ChatService from "
            "victor.agent.services.chat_service instead. "
            "This adapter will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

        self._orchestrator = orchestrator
        self._token_tracker = token_tracker
        self._chat_service: Optional[Any] = None  # Lazy-loaded ChatService
        self._chat_service_getter: Optional[Callable[[], Optional[Any]]] = None
        self._turn_executor: Optional[Any] = None

        # Lazy-initialized handlers (deprecated, kept for compatibility)
        self._intent_classification_handler: Optional["IntentClassificationHandler"] = None
        self._continuation_handler: Optional["ContinuationHandler"] = None
        self._tool_execution_handler: Optional["ToolExecutionHandler"] = None
        self._planning_coordinator: Optional["PlanningCoordinator"] = None
        self._streaming_pipeline: Optional["StreamingChatPipeline"] = None

    def set_streaming_pipeline(self, pipeline: "StreamingChatPipeline") -> None:
        """[DEPRECATED] Previously injected a pre-built streaming pipeline.

        The shim no longer owns pipeline construction — ServiceStreamingRuntime
        holds the live path.  This method is now a no-op kept only for callers
        that have not yet migrated away.
        """
        warnings.warn(
            "set_streaming_pipeline() is deprecated and has no effect. "
            "Pipeline construction is owned by ServiceStreamingRuntime.",
            DeprecationWarning,
            stacklevel=2,
        )

    def bind_chat_service(self, chat_service: Any) -> None:
        """Bind the canonical ChatService for backward-compatible delegation."""
        self._chat_service = chat_service
        self._chat_service_getter = None

    def bind_chat_service_getter(self, getter: Callable[[], Optional[Any]]) -> None:
        """Bind a lazy ChatService getter for deprecated shim compatibility."""
        self._chat_service_getter = getter
        self._chat_service = None

    def _resolve_chat_service(self) -> Optional[Any]:
        """Resolve the currently bound canonical ChatService if available."""
        if self._chat_service is not None:
            return self._chat_service
        if self._chat_service_getter is not None:
            try:
                return self._chat_service_getter()
            except Exception:
                return None
        return None

    def _get_orchestrator_runtime_helper(self, name: str) -> Any:
        """Resolve a real orchestrator runtime helper without MagicMock false-positives."""
        orch = self._orchestrator

        instance_dict = getattr(orch, "__dict__", None)
        if instance_dict:
            helper = instance_dict.get(name)
            if callable(helper):
                return helper

        type_helper = getattr(type(orch), name, None)
        if callable(type_helper):

            async def _bound_helper(*args: Any, **kwargs: Any) -> Any:
                return await type_helper(orch, *args, **kwargs)

            return _bound_helper

        return None

    def _get_orchestrator_runtime_property(self, name: str) -> Any:
        """Resolve a real orchestrator property without MagicMock false-positives."""
        orch = self._orchestrator

        instance_dict = getattr(orch, "__dict__", None)
        if instance_dict and name in instance_dict:
            return instance_dict.get(name)

        type_attr = getattr(type(orch), name, None)
        if isinstance(type_attr, property):
            return getattr(orch, name)

        return None

    # =====================================================================
    # Public API
    # =====================================================================

    @property
    def turn_executor(self) -> Any:
        """[DEPRECATED] TurnExecutor owned by ChatService, not the coordinator shim.

        Access via ``chat_service.turn_executor`` instead.
        """
        warnings.warn(
            "ChatCoordinator.turn_executor is deprecated. "
            "Access turn_executor via ChatService directly.",
            DeprecationWarning,
            stacklevel=2,
        )

        chat_service = self._resolve_chat_service()
        if chat_service is not None and hasattr(chat_service, "turn_executor"):
            return chat_service.turn_executor

        orchestrator_executor = self._get_orchestrator_runtime_property("turn_executor")
        if orchestrator_executor is not None:
            return orchestrator_executor

        if self._turn_executor is None:
            warnings.warn(
                "ChatCoordinator.turn_executor is materializing a legacy local "
                "TurnExecutor because no ChatService or orchestrator runtime "
                "executor is bound. This fallback is deprecated compatibility "
                "behavior.",
                DeprecationWarning,
                stacklevel=2,
            )
            from victor.agent.services.turn_execution_runtime import (
                TurnExecutor,
            )

            # Create protocol adapter for orchestrator
            from victor.agent.services.orchestrator_protocol_adapter import (
                OrchestratorProtocolAdapter,
            )

            adapter = OrchestratorProtocolAdapter(self._orchestrator)

            # Initialize execution coordinator with protocol-based dependencies
            self._turn_executor = TurnExecutor(
                chat_context=adapter,
                tool_context=adapter,
                provider_context=adapter,
                execution_provider=adapter,
                token_tracker=self._token_tracker,
            )
        return self._turn_executor

    async def chat(
        self,
        user_message: str,
        use_planning: Optional[bool] = False,
    ) -> CompletionResponse:
        """Send a chat message and get response with full agentic loop.

        This method implements a proper agentic loop that:
        1. Optionally uses structured planning for complex tasks
        2. Delegates to execution coordinator for agentic loop
        3. Ensures non-empty response on tool failures

        Args:
            user_message: User's message
            use_planning: Whether to use structured planning for complex tasks.
                None = auto-detect based on task complexity.
                True = use planning if task qualifies.
                False = skip planning entirely.

        Returns:
            CompletionResponse from the model with complete response
        """
        warnings.warn(
            "ChatCoordinator.chat() is deprecated compatibility surface. "
            "Use ChatService.chat() or AgentOrchestrator.chat() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        chat_service = self._resolve_chat_service()
        if chat_service is not None:
            return await chat_service.chat(
                user_message,
                use_planning=use_planning,
            )

        warnings.warn(
            "ChatCoordinator.chat() called without a bound ChatService. "
            "Bind via bind_chat_service() or migrate to ChatService directly.",
            DeprecationWarning,
            stacklevel=2,
        )

        # If planning is explicitly disabled, skip planning check
        if use_planning is False:
            return await self.turn_executor.execute_agentic_loop(user_message)

        # Check if we should use planning for this task (explicit or auto-detected)
        if (use_planning is True or use_planning is None) and self._should_use_planning(
            user_message
        ):
            return await self._chat_with_planning(user_message)

        # Default: delegate to execution coordinator for agentic loop
        return await self.turn_executor.execute_agentic_loop(user_message)

    def _should_use_planning(self, user_message: str) -> bool:
        """Determine if planning should be used for this task.

        Checks for:
        1. Planning coordinator is available
        2. Task complexity threshold
        3. Multi-step indicators

        Args:
            user_message: User's message

        Returns:
            True if planning should be used
        """
        # Planning coordinator must be initialized
        if self._planning_coordinator is None:
            return False

        # Check orchestrator settings
        orch = self._orchestrator
        planning_enabled = getattr(orch.settings, "enable_planning", False)
        if not planning_enabled:
            return False

        # Simple heuristic: multi-step keywords
        # Includes analysis/document-oriented terms alongside code-oriented ones
        multi_step_indicators = [
            "analyze",
            "architecture",
            "design",
            "evaluate",
            "compare",
            "roadmap",
            "implementation",
            "refactor",
            "migration",
            "step",
            "phase",
            "stage",
            "deliverable",
            # Document analysis / review tasks
            "review",
            "criteria",
            "milestone",
            "assessment",
            "audit",
            "comprehensive",
            "document",
            "provide",
        ]
        message_lower = user_message.lower()
        keyword_count = sum(1 for kw in multi_step_indicators if kw in message_lower)

        # Use planning if 2+ keywords or task is complex
        if keyword_count >= 2:
            return True

        # Check task complexity
        from victor.framework.task import TaskComplexity as FrameworkTaskComplexity

        task_classification = orch.task_classifier.classify(user_message)
        return task_classification.complexity in (
            FrameworkTaskComplexity.MEDIUM,
            FrameworkTaskComplexity.COMPLEX,
            FrameworkTaskComplexity.ANALYSIS,
        )

    async def _chat_with_planning(self, user_message: str) -> CompletionResponse:
        """Chat using structured planning for complex tasks.

        Args:
            user_message: User's message

        Returns:
            CompletionResponse from planning-based execution
        """
        orch = self._orchestrator
        runtime_helper = self._get_orchestrator_runtime_helper("_run_planning_chat_runtime")
        if callable(runtime_helper):
            return await runtime_helper(user_message)

        from victor.agent.services.planning_runtime import PlanningCoordinator

        if self._planning_coordinator is None:
            # Lazy initialization
            self._planning_coordinator = PlanningCoordinator(orch)

        # Get task analysis for planning
        task_analysis = orch.task_analyzer.analyze(user_message)

        # Use planning coordinator
        response = await self._planning_coordinator.chat_with_planning(
            user_message,
            task_analysis=task_analysis,
        )

        # Add messages to conversation history
        if not orch._system_added:
            orch.conversation.ensure_system_prompt()
            orch._system_added = True

        orch.add_message("user", user_message)
        if response.content:
            orch.add_message("assistant", response.content)

        return response

    async def stream_chat(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream a chat response (public entrypoint).

        Delegates to the canonical StreamingChatPipeline that coordinates the
        streaming lifecycle (context prep, provider streaming, tool execution,
        continuation handling).

        Args:
            user_message: User's input message
            **kwargs: Additional options, including internal parameters:
                - _preserve_iteration: If True, preserve iteration from failed attempt
                - _current_iteration: Current iteration count to preserve
                - _fallback_iteration: Iteration number from previous attempt

        Returns:
            AsyncIterator yielding StreamChunk objects with incremental response
        """
        warnings.warn(
            "ChatCoordinator.stream_chat() is deprecated compatibility surface. "
            "Use ChatService.stream_chat() or AgentOrchestrator.stream_chat() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        runtime_helper = self._get_orchestrator_runtime_helper("_stream_chat_runtime")
        if callable(runtime_helper):
            async for chunk in runtime_helper(user_message, **kwargs):
                yield chunk
            return

        # Fallback path removed — the shim no longer owns pipeline construction.
        # _stream_chat_runtime is always registered in production via ServiceStreamingRuntime.
        warnings.warn(
            "ChatCoordinator.stream_chat() called without a wired _stream_chat_runtime. "
            "This shim no longer owns streaming execution. "
            "Ensure ServiceStreamingRuntime is registered on the orchestrator.",
            DeprecationWarning,
            stacklevel=2,
        )

    # =====================================================================
    # Planning Integration
    # =====================================================================

    async def chat_with_planning(
        self,
        user_message: str,
        use_planning: Optional[bool] = None,
    ) -> CompletionResponse:
        """Chat with automatic planning for complex multi-step tasks.

        Convenience method that delegates to chat() with planning support.

        Args:
            user_message: User's message
            use_planning: Force planning on/off. None = auto-detect

        Returns:
            CompletionResponse from the model
        """
        warnings.warn(
            "ChatCoordinator.chat_with_planning() is deprecated compatibility surface. "
            "Use ChatService.chat_with_planning() or AgentOrchestrator.chat() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.chat(user_message, use_planning=use_planning)

    # =====================================================================
    # Message Persistence (extracted from AgentOrchestrator.add_message)
    # =====================================================================

    @staticmethod
    def persist_message(
        role: str,
        content: str,
        memory_manager: Any,
        memory_session_id: Optional[str],
        usage_logger: Any,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[list] = None,
    ) -> None:
        """Deprecated compatibility delegate to ``ChatService.persist_message``."""
        warnings.warn(
            "ChatCoordinator.persist_message() is deprecated compatibility surface. "
            "Use ChatService.persist_message() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.agent.services.chat_service import ChatService

        ChatService.persist_message(
            role=role,
            content=content,
            memory_manager=memory_manager,
            memory_session_id=memory_session_id,
            usage_logger=usage_logger,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )
