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

"""Service-owned compatibility shim for deprecated sync chat coordination.

Non-streaming chat is now owned by `ChatService` and its bound runtime
components. `SyncChatCoordinator` remains only for backwards-compatible
callers, but its implementation now lives under `victor.agent.services`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, TYPE_CHECKING

from victor.agent.services.chat_compat_telemetry import (
    record_deprecated_chat_shim_access,
)
from victor.framework.task import TaskComplexity
from victor.providers.base import CompletionResponse

if TYPE_CHECKING:
    from victor.agent.services.protocols.chat_runtime import (
        ChatContextProtocol,
        ToolContextProtocol,
        ProviderContextProtocol,
        SyncChatRuntimeProtocol,
    )
    from victor.agent.services.turn_execution_runtime import TurnExecutor
    from victor.agent.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


class SyncChatCoordinator:
    """Deprecated adapter for non-streaming chat compatibility.

    Args:
        chat_context: Protocol providing conversation/message access
        tool_context: Protocol providing tool selection/execution
        provider_context: Protocol providing LLM provider access
        turn_executor: Deprecated legacy compatibility dependency. Unused by the
            canonical service-bound shim path.
    """

    def __init__(
        self,
        chat_context: "ChatContextProtocol",
        tool_context: "ToolContextProtocol",
        provider_context: "ProviderContextProtocol",
        turn_executor: Optional["TurnExecutor"] = None,
        orchestrator: Optional["SyncChatRuntimeProtocol"] = None,
        query_classifier: Optional["QueryClassifier"] = None,
        chat_service: Optional[Any] = None,
    ) -> None:
        """Initialize the SyncChatCoordinator.

        Args:
            chat_context: Chat context protocol implementation
            tool_context: Tool context protocol implementation
            provider_context: Provider context protocol implementation
            turn_executor: Deprecated compatibility dependency retained for
                constructor stability. No longer required by the canonical shim path.
            orchestrator: Optional sync chat runtime surface for fallback compatibility
            query_classifier: Optional query classifier for auto-planning detection
        """
        self._chat_context = chat_context
        self._tool_context = tool_context
        self._provider_context = provider_context
        self._turn_executor = turn_executor
        self._orchestrator = orchestrator
        self._query_classifier = query_classifier
        self._chat_service = chat_service

        if self._chat_service is None:
            warnings.warn(
                "SyncChatCoordinator without a bound ChatService is deprecated "
                "compatibility construction. Prefer ChatService instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    def bind_chat_service(self, chat_service: Any) -> None:
        """Bind the canonical ChatService for backward-compatible delegation."""
        self._chat_service = chat_service

    # =====================================================================
    # Public API
    # =====================================================================

    async def chat(
        self,
        user_message: str,
        use_planning: Optional[bool] = False,
    ) -> CompletionResponse:
        """Execute chat without streaming.

        Direct path to model response, optimized for sync use cases.

        This method provides the same functionality as ChatCoordinator.chat()
        but is optimized for non-streaming execution by avoiding the overhead
        of event collection and aggregation.

        Args:
            user_message: User's message
            use_planning: Whether to use structured planning for complex tasks.
                None = auto-detect via QueryClassifier.
                True = always use planning (if should_use_planning passes).
                False = never use planning.

        Returns:
            CompletionResponse with complete response
        """
        warnings.warn(
            "SyncChatCoordinator.chat() is deprecated compatibility surface. "
            "Use ChatService.chat() or AgentOrchestrator.chat() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if self._chat_service is not None:
            record_deprecated_chat_shim_access(
                "sync_chat_coordinator", "chat", "chat_service"
            )
            return await self._chat_service.chat(
                user_message,
                use_planning=use_planning,
            )

        orchestrator = self._orchestrator
        if orchestrator is not None and hasattr(orchestrator, "chat"):
            record_deprecated_chat_shim_access(
                "sync_chat_coordinator", "chat", "orchestrator_public"
            )
            response = await orchestrator.chat(user_message, use_planning=use_planning)
            return self._attach_skill_metadata(
                response,
                (
                    orchestrator.get_last_skill_match_info()
                    if hasattr(orchestrator, "get_last_skill_match_info")
                    else None
                ),
            )

        record_deprecated_chat_shim_access(
            "sync_chat_coordinator", "chat", "missing_runtime"
        )
        raise RuntimeError(
            "SyncChatCoordinator has no bound ChatService or orchestrator chat runtime. "
            "Bind ChatService before using deprecated compatibility shims."
        )

    # =====================================================================
    # Private Methods
    # =====================================================================

    @staticmethod
    def _attach_skill_metadata(response: Any, skill_info: Any) -> Any:
        """Attach skill match metadata to response if available."""
        if skill_info and response is not None:
            if not hasattr(response, "metadata") or response.metadata is None:
                try:
                    response.metadata = {}
                except (AttributeError, TypeError):
                    return response
            if isinstance(response.metadata, dict):
                response.metadata.update(skill_info)
        return response

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
        # Check orchestrator settings
        planning_enabled = getattr(self._chat_context.settings, "enable_planning", False)
        if not planning_enabled:
            return False

        # Simple heuristic: multi-step keywords
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
        ]
        message_lower = user_message.lower()
        keyword_count = sum(1 for kw in multi_step_indicators if kw in message_lower)

        # Use planning if 2+ keywords
        if keyword_count >= 2:
            return True

        # Check task complexity
        task_classification = self._provider_context.task_classifier.classify(user_message)
        return task_classification.complexity in (
            TaskComplexity.MEDIUM,
            TaskComplexity.COMPLEX,
            TaskComplexity.ANALYSIS,
        )

    async def _chat_with_planning(self, user_message: str) -> CompletionResponse:
        """Chat using structured planning for complex tasks.

        Args:
            user_message: User's message

        Returns:
            CompletionResponse from planning-based execution
        """
        if self._chat_service is not None:
            record_deprecated_chat_shim_access(
                "sync_chat_coordinator", "chat_with_planning", "chat_service"
            )
            return await self._chat_service.chat(user_message, use_planning=True)

        if self._orchestrator is None:
            record_deprecated_chat_shim_access(
                "sync_chat_coordinator", "chat_with_planning", "missing_runtime"
            )
            raise RuntimeError(
                "SyncChatCoordinator planning requires a bound ChatService or orchestrator. "
                "Bind ChatService before using deprecated compatibility shims."
            )

        record_deprecated_chat_shim_access(
            "sync_chat_coordinator", "chat_with_planning", "orchestrator_public"
        )
        response = await self._orchestrator.chat(
            user_message,
            use_planning=True,
        )
        return self._attach_skill_metadata(
            response,
            (
                self._orchestrator.get_last_skill_match_info()
                if hasattr(self._orchestrator, "get_last_skill_match_info")
                else None
            ),
        )


__all__ = [
    "SyncChatCoordinator",
]
