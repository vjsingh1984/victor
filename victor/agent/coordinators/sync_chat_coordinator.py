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

"""Sync chat coordinator for non-streaming execution.

This module contains the SyncChatCoordinator class that provides optimized
non-streaming chat execution.

The SyncChatCoordinator handles:
- Direct model calls without streaming overhead
- Efficient tool call batching
- Single response aggregation
- Lower memory usage (no event accumulation)

Architecture:
------------
The SyncChatCoordinator depends on protocol-based abstractions:
- ChatContextProtocol: For message/conversation access
- ToolContextProtocol: For tool execution
- ProviderContextProtocol: For LLM calls

Phase 2: Split Sync/Streaming Paths
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from victor.framework.task import TaskComplexity
from victor.providers.base import CompletionResponse

if TYPE_CHECKING:
    from victor.agent.coordinators.chat_protocols import (
        ChatContextProtocol,
        ToolContextProtocol,
        ProviderContextProtocol,
    )
    from victor.agent.coordinators.turn_executor import TurnExecutor
    from victor.agent.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


class SyncChatCoordinator:
    """Coordinator for non-streaming chat execution.

    This coordinator provides an optimized path for synchronous chat execution,
    eliminating the overhead of streaming event collection.

    Key optimizations:
    - Direct model calls without streaming overhead
    - Efficient tool call batching
    - Single response aggregation
    - Lower memory usage (no event accumulation)

    Args:
        chat_context: Protocol providing conversation/message access
        tool_context: Protocol providing tool selection/execution
        provider_context: Protocol providing LLM provider access
        turn_executor: Coordinator for agentic loop execution
    """

    def __init__(
        self,
        chat_context: "ChatContextProtocol",
        tool_context: "ToolContextProtocol",
        provider_context: "ProviderContextProtocol",
        turn_executor: "TurnExecutor",
        orchestrator: Any = None,
        query_classifier: Optional["QueryClassifier"] = None,
    ) -> None:
        """Initialize the SyncChatCoordinator.

        Args:
            chat_context: Chat context protocol implementation
            tool_context: Tool context protocol implementation
            provider_context: Provider context protocol implementation
            turn_executor: Execution coordinator for agentic loop
            orchestrator: Optional orchestrator (required for planning path)
            query_classifier: Optional query classifier for auto-planning detection
        """
        self._chat_context = chat_context
        self._tool_context = tool_context
        self._provider_context = provider_context
        self._turn_executor = turn_executor
        self._orchestrator = orchestrator
        self._query_classifier = query_classifier

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
        # Skill auto-selection (shared logic lives on orchestrator)
        if self._orchestrator and hasattr(self._orchestrator, "apply_skill_for_turn"):
            self._orchestrator.apply_skill_for_turn(user_message)

        # Reset manual skill flag after use
        if getattr(self._orchestrator, "manual_skill_active", False):
            self._orchestrator.manual_skill_active = False

        # Auto-detect planning via QueryClassifier when use_planning is None
        if use_planning is None:
            if self._query_classifier:
                classification = self._query_classifier.classify(user_message)
                # Update system prompt with task-aware guidance
                if self._orchestrator and hasattr(
                    self._orchestrator, "update_system_prompt_for_query"
                ):
                    self._orchestrator.update_system_prompt_for_query(classification)
                if classification.should_plan:
                    response = await self._chat_with_planning(user_message)
                    return self._attach_skill_metadata(
                        response, self._orchestrator.get_last_skill_match_info()
                    )
            else:
                # Fallback to keyword heuristic
                if self._should_use_planning(user_message):
                    response = await self._chat_with_planning(user_message)
                    return self._attach_skill_metadata(
                        response, self._orchestrator.get_last_skill_match_info()
                    )
        elif use_planning and self._should_use_planning(user_message):
            response = await self._chat_with_planning(user_message)
            return self._attach_skill_metadata(
                response, self._orchestrator.get_last_skill_match_info()
            )

        # Use execution coordinator for agentic loop
        response = await self._turn_executor.execute_agentic_loop(user_message)
        return self._attach_skill_metadata(response, self._orchestrator.get_last_skill_match_info())

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
        # PlanningCoordinator currently requires orchestrator.
        # Fall back to normal execution if orchestrator is unavailable.
        if self._orchestrator is None:
            logger.warning(
                "Planning requested but orchestrator not provided to SyncChatCoordinator; "
                "falling back to standard execution."
            )
            return await self._turn_executor.execute_agentic_loop(user_message)

        # Import here to avoid circular dependency
        from victor.agent.coordinators.planning_coordinator import PlanningCoordinator

        planning_coordinator = PlanningCoordinator(self._orchestrator)

        # Get task analysis for planning
        task_analysis = self._provider_context.task_analyzer.analyze(user_message)

        # Use planning coordinator
        response = await planning_coordinator.chat_with_planning(
            user_message,
            task_analysis=task_analysis,
        )

        # Add messages to conversation history
        if not self._chat_context._system_added:
            self._chat_context.conversation.ensure_system_prompt()
            self._chat_context._system_added = True

        self._chat_context.add_message("user", user_message)
        if response.content:
            self._chat_context.add_message("assistant", response.content)

        return response


__all__ = [
    "SyncChatCoordinator",
]
