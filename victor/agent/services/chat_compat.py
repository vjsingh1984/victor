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

from victor.agent.services.chat_compat_telemetry import (
    record_deprecated_chat_shim_access,
)
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

    def _get_orchestrator_callable(self, name: str) -> Any:
        """Resolve a real orchestrator callable without MagicMock false-positives."""
        orch = self._orchestrator

        instance_dict = getattr(orch, "__dict__", None)
        if instance_dict:
            helper = instance_dict.get(name)
            if callable(helper):
                return helper

        type_helper = getattr(type(orch), name, None)
        if callable(type_helper):

            def _bound_helper(*args: Any, **kwargs: Any) -> Any:
                return type_helper(orch, *args, **kwargs)

            return _bound_helper

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
            record_deprecated_chat_shim_access(
                "chat_coordinator", "turn_executor", "chat_service"
            )
            return chat_service.turn_executor

        record_deprecated_chat_shim_access("chat_coordinator", "turn_executor", "missing_runtime")
        raise RuntimeError(
            "ChatCoordinator turn_executor requires a bound ChatService runtime. "
            "Bind ChatService and access ChatService.turn_executor directly."
        )

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
            record_deprecated_chat_shim_access("chat_coordinator", "chat", "chat_service")
            return await chat_service.chat(
                user_message,
                use_planning=use_planning,
            )

        runtime_chat = self._get_orchestrator_runtime_helper("chat")
        if callable(runtime_chat):
            record_deprecated_chat_shim_access(
                "chat_coordinator", "chat", "orchestrator_public"
            )
            return await runtime_chat(user_message, use_planning=use_planning)

        record_deprecated_chat_shim_access("chat_coordinator", "chat", "missing_runtime")
        raise RuntimeError(
            "ChatCoordinator has no bound ChatService or orchestrator chat runtime. "
            "Bind ChatService via bind_chat_service()."
        )

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
        chat_service = self._resolve_chat_service()
        if chat_service is not None:
            record_deprecated_chat_shim_access(
                "chat_coordinator", "chat_with_planning", "chat_service"
            )
            return await chat_service.chat(user_message, use_planning=True)

        runtime_helper = self._get_orchestrator_runtime_helper("chat")
        if callable(runtime_helper):
            record_deprecated_chat_shim_access(
                "chat_coordinator", "chat_with_planning", "orchestrator_public"
            )
            return await runtime_helper(user_message, use_planning=True)

        record_deprecated_chat_shim_access(
            "chat_coordinator", "chat_with_planning", "missing_runtime"
        )
        raise RuntimeError(
            "ChatCoordinator planning requires a bound ChatService or orchestrator chat runtime. "
            "Bind ChatService before using deprecated compatibility shims."
        )

    async def stream_chat(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Stream a chat response (public entrypoint).

        Compatibility resolution order:
        1. Bound ``ChatService.stream_chat()``
        2. Orchestrator ``stream_chat()``

        The shim does not own streaming execution anymore; it only forwards to
        the canonical service/runtime path.

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

        chat_service = self._resolve_chat_service()
        if chat_service is not None and hasattr(chat_service, "stream_chat"):
            record_deprecated_chat_shim_access(
                "chat_coordinator", "stream_chat", "chat_service"
            )
            async for chunk in chat_service.stream_chat(user_message, **kwargs):
                yield chunk
            return

        runtime_helper = self._get_orchestrator_runtime_helper("stream_chat")
        if callable(runtime_helper):
            record_deprecated_chat_shim_access(
                "chat_coordinator", "stream_chat", "orchestrator_public"
            )
            async for chunk in runtime_helper(user_message, **kwargs):
                yield chunk
            return

        record_deprecated_chat_shim_access("chat_coordinator", "stream_chat", "missing_runtime")
        raise RuntimeError(
            "ChatCoordinator has no bound ChatService or streaming runtime. "
            "Bind ChatService or use AgentOrchestrator.stream_chat()."
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
        metadata: Optional[dict] = None,
    ) -> None:
        """Deprecated compatibility delegate to ``ChatService.persist_message``."""
        warnings.warn(
            "ChatCoordinator.persist_message() is deprecated compatibility surface. "
            "Use ChatService.persist_message() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        record_deprecated_chat_shim_access("chat_coordinator", "persist_message", "chat_service")
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
            metadata=metadata,
        )
