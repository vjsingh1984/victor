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

"""Chat coordinator for managing chat and streaming chat operations.

This module contains the ChatCoordinator class that has been extracted from
the AgentOrchestrator to improve separation of concerns and testability.

The ChatCoordinator handles:
- Non-streaming chat with agentic loop
- Streaming chat with full iteration management
- Tool calling and execution coordination
- Response validation and recovery

Architecture:
------------
The ChatCoordinator acts as a facade for chat operations, delegating to
specialized handlers for specific concerns:
- IntentClassificationHandler: Determines continuation actions
- ContinuationHandler: Executes continuation decisions
- ToolExecutionHandler: Manages tool execution
- RecoveryCoordinator: Handles error recovery

This design enables:
- Independent testing of chat logic
- Clear separation of concerns
- Easier refactoring and maintenance
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from victor.framework.task import TaskComplexity
from victor.agent.unified_task_tracker import TrackerTaskType
from victor.agent.response_completer import ToolFailureContext
from victor.agent.prompt_requirement_extractor import extract_prompt_requirements
from victor.agent.task_analyzer import TaskAnalysis
from victor.providers.base import CompletionResponse, Message, StreamChunk
from victor.core.errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

if TYPE_CHECKING:
    # Type-only imports
    from victor.agent.streaming.context import StreamingChatContext
    from victor.agent.coordinators.chat_protocols import ChatOrchestratorProtocol
    from victor.agent.streaming.intent_classification import IntentClassificationHandler
    from victor.agent.streaming.continuation import ContinuationHandler
    from victor.agent.streaming.tool_execution import ToolExecutionHandler
    from victor.agent.coordinators.planning_coordinator import PlanningCoordinator
    from victor.agent.streaming.pipeline import StreamingChatPipeline

logger = logging.getLogger(__name__)


class ChatCoordinator:
    """Coordinator for chat and streaming chat operations.

    This class extracts chat-related logic from the orchestrator,
    providing a clean interface for managing conversations.

    The coordinator depends on ``ChatOrchestratorProtocol`` (defined in
    ``chat_protocols.py``) rather than the concrete ``AgentOrchestrator``.
    This enables unit testing with lightweight mocks.

    Args:
        orchestrator: Any object satisfying ChatOrchestratorProtocol
    """

    def __init__(self, orchestrator: "ChatOrchestratorProtocol") -> None:
        """Initialize the ChatCoordinator.

        Args:
            orchestrator: Object satisfying ChatOrchestratorProtocol
        """
        self._orchestrator = orchestrator

        # Lazy-initialized handlers
        self._intent_classification_handler: Optional["IntentClassificationHandler"] = None
        self._continuation_handler: Optional["ContinuationHandler"] = None
        self._tool_execution_handler: Optional["ToolExecutionHandler"] = None
        self._planning_coordinator: Optional["PlanningCoordinator"] = None
        self._streaming_pipeline: Optional["StreamingChatPipeline"] = None

    def set_streaming_pipeline(self, pipeline: "StreamingChatPipeline") -> None:
        """Inject a pre-built streaming pipeline (used by orchestrator factory)."""
        self._streaming_pipeline = pipeline

        # NEW: Execution coordinator for agentic loop (Phase 1)
        self._execution_coordinator: Optional[Any] = None

    # =====================================================================
    # Public API
    # =====================================================================

    @property
    def execution_coordinator(self) -> Any:
        """Get the execution coordinator for agentic loop.

        Returns:
            ExecutionCoordinator instance (lazy initialized)
        """
        if self._execution_coordinator is None:
            from victor.agent.coordinators.execution_coordinator import ExecutionCoordinator

            # Create protocol adapter for orchestrator
            from victor.agent.coordinators.protocol_adapters import OrchestratorProtocolAdapter

            adapter = OrchestratorProtocolAdapter(self._orchestrator)

            # Initialize execution coordinator with protocol-based dependencies
            self._execution_coordinator = ExecutionCoordinator(
                chat_context=adapter,
                tool_context=adapter,
                provider_context=adapter,
                execution_provider=adapter,
            )
        return self._execution_coordinator

    async def chat(
        self,
        user_message: str,
        use_planning: bool = False,
    ) -> CompletionResponse:
        """Send a chat message and get response with full agentic loop.

        This method implements a proper agentic loop that:
        1. Optionally uses structured planning for complex tasks
        2. Delegates to execution coordinator for agentic loop
        3. Ensures non-empty response on tool failures

        Args:
            user_message: User's message
            use_planning: Whether to use structured planning for complex tasks

        Returns:
            CompletionResponse from the model with complete response
        """
        # Check if we should use planning for this task
        if use_planning and self._should_use_planning(user_message):
            return await self._chat_with_planning(user_message)

        # NEW: Delegate to execution coordinator for agentic loop
        return await self.execution_coordinator.execute_agentic_loop(user_message)

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
        from victor.agent.coordinators.planning_coordinator import PlanningCoordinator

        if self._planning_coordinator is None:
            # Lazy initialization
            self._planning_coordinator = PlanningCoordinator(self._orchestrator)

        # Get task analysis for planning
        orch = self._orchestrator
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

    async def stream_chat(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Stream a chat response (public entrypoint).

        Delegates to the canonical StreamingChatPipeline that coordinates the
        streaming lifecycle (context prep, provider streaming, tool execution,
        continuation handling).

        Args:
            user_message: User's input message

        Returns:
            AsyncIterator yielding StreamChunk objects with incremental response
        """
        orch = self._orchestrator
        if self._streaming_pipeline is None:
            from victor.agent.streaming import create_streaming_chat_pipeline

            self._streaming_pipeline = create_streaming_chat_pipeline(self)

        pipeline = self._streaming_pipeline
        try:
            async for chunk in pipeline.run(user_message):
                yield chunk
        finally:
            # Update cumulative token usage after stream completes
            # This enables accurate token tracking for evaluations/benchmarks
            if hasattr(orch, "_current_stream_context") and orch._current_stream_context:
                ctx = orch._current_stream_context
                if hasattr(ctx, "cumulative_usage"):
                    for key in orch._cumulative_token_usage:
                        if key in ctx.cumulative_usage:
                            orch._cumulative_token_usage[key] += ctx.cumulative_usage[key]
                    # Calculate total if not tracked by provider
                    if orch._cumulative_token_usage["total_tokens"] == 0:
                        orch._cumulative_token_usage["total_tokens"] = (
                            orch._cumulative_token_usage["prompt_tokens"]
                            + orch._cumulative_token_usage["completion_tokens"]
                        )

    # =====================================================================
    # Stream Preparation and Context
    # =====================================================================

    async def _prepare_stream(self, user_message: str) -> tuple[
        Any,
        float,
        float,
        Dict[str, int],
        int,
        int,
        int,
        bool,
        TrackerTaskType,
        Any,
        int,
    ]:
        """Prepare streaming state and return commonly used values."""
        orch = self._orchestrator

        # Initialize cancellation support
        orch._cancel_event = asyncio.Event()
        orch._is_streaming = True

        # Track performance metrics using StreamMetrics
        stream_metrics = orch._metrics_collector.init_stream_metrics()
        start_time = stream_metrics.start_time
        total_tokens: float = 0

        # Cumulative token usage from provider
        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        # Ensure system prompt is included once at start of conversation
        orch.conversation.ensure_system_prompt()
        orch._system_added = True
        # Reset session state for new stream via SessionStateManager
        orch._session_state.reset_for_new_turn()

        # Reset unified tracker for new conversation
        orch.unified_tracker.reset()

        # Reset context reminder manager for new conversation turn
        orch.reminder_manager.reset()

        # Start UsageAnalytics session for this conversation
        if hasattr(orch, "_usage_analytics") and orch._usage_analytics:
            orch._usage_analytics.start_session()

        # Clear ToolSequenceTracker history for new conversation
        if hasattr(orch, "_sequence_tracker") and orch._sequence_tracker:
            orch._sequence_tracker.clear_history()

        # PERF: Start background compaction for async context management
        if orch._context_manager and hasattr(orch._context_manager, "start_background_compaction"):
            await orch._context_manager.start_background_compaction(interval_seconds=15.0)

        # Local aliases for frequently-used values
        max_total_iterations = orch.unified_tracker.config.get("max_total_iterations", 50)
        total_iterations = 0
        force_completion = False

        # Add user message to history
        orch.add_message("user", user_message)

        # Record this turn in UsageAnalytics
        if hasattr(orch, "_usage_analytics") and orch._usage_analytics:
            orch._usage_analytics.record_turn()

        # Detect task type using unified tracker
        unified_task_type = orch.unified_tracker.detect_task_type(user_message)
        logger.info(f"Task type detected: {unified_task_type.value}")

        # Extract prompt requirements for dynamic budgets
        prompt_requirements = extract_prompt_requirements(user_message)
        if prompt_requirements.has_explicit_requirements():
            orch.unified_tracker._progress.has_prompt_requirements = True

            if (
                prompt_requirements.tool_budget
                and prompt_requirements.tool_budget > orch.unified_tracker._progress.tool_budget
            ):
                orch.unified_tracker.set_tool_budget(prompt_requirements.tool_budget)
                logger.info(
                    f"Dynamic budget from prompt: {prompt_requirements.tool_budget} "
                    f"(files={prompt_requirements.file_count}, fixes={prompt_requirements.fix_count})"
                )

            if (
                prompt_requirements.iteration_budget
                and prompt_requirements.iteration_budget
                > orch.unified_tracker._task_config.max_exploration_iterations
            ):
                orch.unified_tracker.set_max_iterations(prompt_requirements.iteration_budget)
                logger.info(
                    f"Dynamic iterations from prompt: {prompt_requirements.iteration_budget}"
                )

        # Intelligent pipeline pre-request hook
        intelligent_task = asyncio.create_task(
            orch._prepare_intelligent_request(
                task=user_message,
                task_type=unified_task_type.value,
            )
        )

        # Get exploration iterations from unified tracker
        max_exploration_iterations = orch.unified_tracker.max_exploration_iterations

        # Task prep: hints, complexity, reminders
        task_classification, complexity_tool_budget = self._prepare_task(
            user_message, unified_task_type
        )

        # Await intelligent request after sync work completes
        intelligent_context = await intelligent_task
        if intelligent_context:
            if intelligent_context.get("system_prompt_addition"):
                orch.add_message("system", intelligent_context["system_prompt_addition"])
                logger.debug("Injected intelligent pipeline optimized prompt")

        return (
            stream_metrics,
            start_time,
            total_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        )

    async def _create_stream_context(self, user_message: str) -> "StreamingChatContext":
        """Create a StreamingChatContext with all prepared data.

        Args:
            user_message: The user's message

        Returns:
            Populated StreamingChatContext ready for the streaming loop
        """
        from victor.agent.streaming import create_stream_context

        orch = self._orchestrator

        # Get all prepared data from _prepare_stream
        (
            stream_metrics,
            start_time,
            total_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        ) = await self._prepare_stream(user_message)

        # Classify task type based on keywords
        task_keywords = self._classify_task_keywords(user_message)

        # Create and populate context
        ctx = create_stream_context(
            user_message=user_message,
            max_iterations=max_total_iterations,
            max_exploration=max_exploration_iterations,
            tool_budget=complexity_tool_budget,
        )

        # Populate context with prepared data
        ctx.stream_metrics = stream_metrics
        ctx.start_time = start_time
        ctx.total_tokens = total_tokens
        ctx.cumulative_usage = cumulative_usage
        ctx.total_iterations = total_iterations
        ctx.force_completion = force_completion
        ctx.unified_task_type = unified_task_type
        ctx.task_classification = task_classification
        ctx.complexity_tool_budget = complexity_tool_budget

        # Add task keyword results
        ctx.is_analysis_task = task_keywords["is_analysis_task"] or unified_task_type.value in (
            "analyze",
            "analysis",
        )
        ctx.is_action_task = task_keywords["is_action_task"]
        ctx.needs_execution = task_keywords["needs_execution"]
        ctx.coarse_task_type = task_keywords["coarse_task_type"]

        # Set is_complex_task from ComplexityClassifier for lenient progress checking
        if task_classification and hasattr(task_classification, "complexity"):
            ctx.is_complex_task = task_classification.complexity in (
                TaskComplexity.COMPLEX,
                TaskComplexity.ANALYSIS,
            )

        # Set goals for tool selection
        ctx.goals = orch._tool_planner.infer_goals_from_message(user_message)

        # Sync tool tracking from orchestrator to context
        ctx.tool_budget = orch.tool_budget
        ctx.tool_calls_used = orch.tool_calls_used

        # Task Completion Detection Enhancement
        ctx.task_completion_detector = orch._task_completion_detector

        return ctx

    # =====================================================================
    # Helper Methods
    # =====================================================================

    def _prepare_task(
        self, user_message: str, unified_task_type: TrackerTaskType
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments.

        Args:
            user_message: The user's message
            unified_task_type: The detected task type

        Returns:
            Tuple of (task_classification, complexity_tool_budget)
        """
        orch = self._orchestrator

        # Wire reminder_manager dependency if not already set
        if orch.task_coordinator._reminder_manager is None:
            orch.task_coordinator.set_reminder_manager(orch.reminder_manager)

        # Delegate to TaskCoordinator
        return orch.task_coordinator.prepare_task(
            user_message, unified_task_type, orch.conversation_controller
        )

    def _classify_task_keywords(self, user_message: str) -> Dict[str, Any]:
        """Classify task based on keywords in the user message.

        Delegates to TaskAnalyzer.classify_task_keywords() for consistency
        with orchestrator behavior.

        Args:
            user_message: The user's message

        Returns:
            Dictionary with classification results including is_analysis_task,
            is_action_task, needs_execution, and coarse_task_type
        """
        orch = self._orchestrator
        return orch._task_analyzer.classify_task_keywords(user_message)

    def _apply_intent_guard(self, user_message: str) -> None:
        """Detect intent and inject prompt guards for read-only tasks.

        Args:
            user_message: The user's message
        """
        orch = self._orchestrator

        # Delegate to TaskCoordinator
        orch.task_coordinator.apply_intent_guard(user_message, orch.conversation_controller)

        # Sync current_intent back to orchestrator
        orch._current_intent = orch.task_coordinator.current_intent

    def _apply_task_guidance(
        self,
        user_message: str,
        unified_task_type: TrackerTaskType,
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
    ) -> None:
        """Apply guidance and budget tweaks for analysis/action tasks.

        Args:
            user_message: The user's message
            unified_task_type: The detected task type
            is_analysis_task: Whether this is an analysis task
            is_action_task: Whether this is an action task
            needs_execution: Whether execution is needed
            max_exploration_iterations: Maximum exploration iterations
        """
        orch = self._orchestrator

        # Set initial temperature and tool_budget in TaskCoordinator
        orch.task_coordinator.temperature = orch.temperature
        orch.task_coordinator.tool_budget = orch.tool_budget

        # Delegate to TaskCoordinator
        orch.task_coordinator.apply_task_guidance(
            user_message=user_message,
            unified_task_type=unified_task_type,
            is_analysis_task=is_analysis_task,
            is_action_task=is_action_task,
            needs_execution=needs_execution,
            max_exploration_iterations=max_exploration_iterations,
            conversation_controller=orch.conversation_controller,
        )

        # Sync temperature and tool_budget back to orchestrator
        orch.temperature = orch.task_coordinator.temperature
        orch.tool_budget = orch.task_coordinator.tool_budget

    async def _select_tools_for_turn(self, context_msg: str, goals: Any) -> Any:
        """Select and prioritize tools for the current turn.

        Args:
            context_msg: The context message for tool selection
            goals: Inferred goals from the user message

        Returns:
            Selected and prioritized tools, or None if tooling not allowed
        """
        orch = self._orchestrator

        provider_supports_tools = orch.provider.supports_tools()
        tooling_allowed = provider_supports_tools and orch._model_supports_tool_calls()

        if not tooling_allowed:
            return None

        planned_tools = None
        if goals:
            available_inputs = ["query"]
            if orch.observed_files:
                available_inputs.append("file_contents")
            planned_tools = orch._tool_planner.plan_tools(goals, available_inputs)
            logger.info(f"available_inputs={available_inputs}")

        conversation_depth = orch.conversation.message_count()
        conversation_history = (
            [msg.model_dump() for msg in orch.messages] if orch.messages else None
        )
        tools = await orch.tool_selector.select_tools(
            context_msg,
            use_semantic=orch.use_semantic_selection,
            conversation_history=conversation_history,
            conversation_depth=conversation_depth,
            planned_tools=planned_tools,
        )
        logger.info(
            f"context_msg={context_msg}\nuse_semantic={orch.use_semantic_selection}\nconversation_depth={conversation_depth}"
        )
        tools = orch.tool_selector.prioritize_by_stage(context_msg, tools)
        current_intent = getattr(orch, "_current_intent", None)
        tools = orch._tool_planner.filter_tools_by_intent(tools, current_intent)
        return tools

    def _get_decision_service(self) -> Optional[Any]:
        """Get the LLM decision service from the container if available.

        Returns:
            LLMDecisionService instance or None if not configured.
        """
        try:
            from victor.core.feature_flags import FeatureFlag, is_feature_enabled

            if not is_feature_enabled(FeatureFlag.USE_LLM_DECISION_SERVICE):
                return None
            from victor.agent.services.protocols.decision_service import (
                LLMDecisionServiceProtocol,
            )

            container = getattr(self._orchestrator, "_container", None)
            if container is not None:
                return container.get(LLMDecisionServiceProtocol)
        except Exception:
            pass
        return None

    def _extract_required_files_from_prompt(self, user_message: str) -> List[str]:
        """Extract required file paths from the user message.

        Args:
            user_message: The user's message

        Returns:
            List of required file paths mentioned in the message
        """
        import re

        # Extract file paths from the message (e.g., /path/to/file.py, ./relative/path)
        pattern = r'(?:^|[\s"\'`])(/[\w./\-]+\.\w+|\.{1,2}/[\w./\-]+\.\w+)'
        matches = re.findall(pattern, user_message)
        return list(set(matches))

    def _extract_required_outputs_from_prompt(self, user_message: str) -> List[str]:
        """Extract required outputs from the user message.

        Args:
            user_message: The user's message

        Returns:
            List of required outputs mentioned in the message
        """
        # Required outputs are not easily extractable from raw text;
        # return empty list as these are advisory hints for the streaming loop
        return []

    def _get_max_context_chars(self) -> int:
        """Get the maximum context length in characters.

        Delegates to orchestrator's _get_max_context_chars which uses ContextManager.

        Returns:
            Maximum context length for the current provider/model
        """
        return self._orchestrator._get_max_context_chars()

    # =====================================================================
    # Iteration Pre-Checks
    # =====================================================================

    async def _run_iteration_pre_checks(
        self,
        stream_ctx: "StreamingChatContext",
        user_message: str,
    ) -> AsyncIterator[StreamChunk]:
        """Run pre-iteration checks: cancellation, compaction, time limit.

        Args:
            stream_ctx: The streaming context
            user_message: The user's message

        Yields:
            StreamChunk objects for notifications or final chunks
        """
        orch = self._orchestrator

        # 1. Check for cancellation
        if orch._check_cancellation():
            logger.info("Stream cancelled by user request")
            orch._is_streaming = False
            orch._record_intelligent_outcome(
                success=False,
                quality_score=stream_ctx.last_quality_score,
                user_satisfied=False,
                completed=False,
            )
            yield StreamChunk(
                content="\n\n[Cancelled by user]\n",
                is_final=True,
            )
            return

        # 2. Check context compaction
        if orch._context_compactor:
            compaction_action = orch._context_compactor.check_and_compact(
                current_query=user_message,
                force=False,
                tool_call_count=orch.tool_calls_used,
                task_complexity=TaskComplexity.COMPLEX.value,
            )
            if compaction_action.action_taken:
                logger.info(
                    f"Compacted context: {compaction_action.messages_removed} messages removed, "
                    f"{compaction_action.tokens_freed} tokens freed"
                )

        # 3. Check time limit
        time_limit = getattr(orch.settings, "stream_idle_timeout_seconds", 300)
        if stream_ctx.is_over_time_limit(time_limit):
            logger.warning(f"Stream time limit exceeded: {stream_ctx.elapsed_time():.1f}s")
            yield StreamChunk(
                content=f"\n\n[Session exceeded {time_limit}s idle timeout - providing summary]\n",
                is_final=False,
            )
            stream_ctx.force_completion = True

        # 4. Increment iteration
        stream_ctx.increment_iteration()

        # 5. Inject grounding feedback if pending
        if stream_ctx.pending_grounding_feedback:
            logger.info("Injecting pending grounding feedback as system message")
            orch.add_message("system", stream_ctx.pending_grounding_feedback)
            stream_ctx.pending_grounding_feedback = ""

    def _log_iteration_debug(
        self,
        stream_ctx: "StreamingChatContext",
        max_total_iterations: int,
    ) -> None:
        """Log iteration debug information.

        Args:
            stream_ctx: The streaming context
            max_total_iterations: Maximum iterations allowed
        """
        orch = self._orchestrator
        unique_resources = orch.unified_tracker.unique_resources
        logger.debug(
            f"Iteration {stream_ctx.total_iterations}/{max_total_iterations}: "
            f"tool_calls_used={orch.tool_calls_used}/{orch.tool_budget}, "
            f"unique_resources={len(unique_resources)}, "
            f"force_completion={stream_ctx.force_completion}"
        )

        orch.debug_logger.log_iteration_start(
            stream_ctx.total_iterations,
            tool_calls=orch.tool_calls_used,
            files_read=len(unique_resources),
        )
        orch.debug_logger.log_limits(
            tool_budget=orch.tool_budget,
            tool_calls_used=orch.tool_calls_used,
            max_iterations=max_total_iterations,
            current_iteration=stream_ctx.total_iterations,
            is_analysis_task=stream_ctx.is_analysis_task,
        )

    # =====================================================================
    # Context and Iteration Limits
    # =====================================================================

    async def _handle_context_and_iteration_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional[StreamChunk]]:
        """Handle context overflow and hard iteration limits.

        Returns:
            handled (bool): True if the caller should stop processing
            chunk (Optional[StreamChunk]): Chunk to yield if produced
        """
        orch = self._orchestrator

        # Context overflow handling
        if orch._check_context_overflow(max_context):
            logger.warning("Context overflow detected. Attempting smart compaction...")
            removed = orch._conversation_controller.smart_compact_history(
                current_query=user_message
            )
            if removed > 0:
                logger.info(f"Smart compaction removed {removed} messages")
                chunk = StreamChunk(
                    content=f"\n[context] Compacted history ({removed} messages) to continue.\n"
                )
                orch._conversation_controller.inject_compaction_context()
                return False, chunk

            # If still overflowing, force completion
            if orch._check_context_overflow(max_context):
                logger.warning("Still overflowing after compaction. Forcing completion.")
                chunk = StreamChunk(
                    content=f"\n[tool] {orch._presentation.icon('warning', with_color=False)} Context size limit reached. Providing summary.\n"
                )
                completion_prompt = orch._get_thinking_disabled_prompt(
                    "Context limit reached. Summarize in 2-3 sentences."
                )
                recent_messages = orch.messages[-8:] if len(orch.messages) > 8 else orch.messages[:]
                completion_messages = recent_messages + [
                    Message(role="user", content=completion_prompt)
                ]

                try:
                    response = await orch.provider.chat(
                        messages=completion_messages,
                        model=orch.model,
                        temperature=orch.temperature,
                        max_tokens=min(orch.max_tokens, 1024),
                        tools=None,
                    )
                    if response and response.content:
                        sanitized = orch.sanitizer.sanitize(response.content)
                        if sanitized:
                            orch.add_message("assistant", sanitized)
                            chunk = StreamChunk(content=sanitized, is_final=True)
                            orch._record_intelligent_outcome(
                                success=True,
                                quality_score=last_quality_score,
                                user_satisfied=True,
                                completed=True,
                            )
                            return True, chunk
                except Exception as e:
                    logger.warning(f"Final response after context overflow failed: {e}")
                orch._record_intelligent_outcome(
                    success=True,
                    quality_score=last_quality_score,
                    user_satisfied=True,
                    completed=True,
                )
                return True, StreamChunk(content="", is_final=True)

        # Iteration limit handling
        if total_iterations > max_total_iterations:
            logger.warning(
                f"Hard iteration limit reached ({max_total_iterations}). Forcing completion."
            )
            iteration_prompt = orch._get_thinking_disabled_prompt(
                "Max iterations reached. Summarize key findings in 3-4 sentences. "
                "Do NOT attempt any more tool calls."
            )
            recent_messages = orch.messages[-10:] if len(orch.messages) > 10 else orch.messages[:]
            completion_messages = recent_messages + [Message(role="user", content=iteration_prompt)]

            chunk = StreamChunk(
                content=f"\n[tool] {orch._presentation.icon('warning', with_color=False)} Maximum iterations ({max_total_iterations}) reached. Providing summary.\n"
            )

            try:
                response = await orch.provider.chat(
                    messages=completion_messages,
                    model=orch.model,
                    temperature=orch.temperature,
                    max_tokens=min(orch.max_tokens, 1024),
                    tools=None,
                )
                if response and response.content:
                    sanitized = orch.sanitizer.sanitize(response.content)
                    if sanitized:
                        orch.add_message("assistant", sanitized)
                        chunk = StreamChunk(content=sanitized)
            except (ProviderRateLimitError, ProviderTimeoutError) as e:
                logger.error(f"Rate limit/timeout during final response: {e}")
                chunk = StreamChunk(content="Rate limited or timeout. Please retry in a moment.\n")
            except ProviderAuthError as e:
                logger.error(f"Auth error during final response: {e}")
                chunk = StreamChunk(content="Authentication error. Check API credentials.\n")
            except (ConnectionError, TimeoutError) as e:
                logger.error(f"Network error during final response: {e}")
                chunk = StreamChunk(content="Network error. Check connection.\n")
            except Exception:
                logger.exception("Unexpected error during final response generation")
                chunk = StreamChunk(
                    content="Unable to generate final summary due to iteration limit.\n"
                )

            orch._record_intelligent_outcome(
                success=True,
                quality_score=last_quality_score,
                user_satisfied=True,
                completed=True,
            )
            return True, StreamChunk(content="", is_final=True)

        return False, None

    # =====================================================================
    # Provider Response Streaming
    # =====================================================================

    async def _stream_provider_response(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """Stream response from provider with rate limit retry.

        Args:
            tools: Available tools for the provider
            provider_kwargs: Additional kwargs for the provider
            stream_ctx: The streaming context

        Returns:
            Tuple of (full_content, tool_calls, total_tokens, garbage_detected)
        """
        return await self._stream_with_rate_limit_retry(tools, provider_kwargs, stream_ctx)

    def _get_rate_limit_wait_time(self, exc: Exception, attempt: int) -> float:
        """Get wait time for rate limit retry.

        Args:
            exc: The rate limit exception
            attempt: Current retry attempt number

        Returns:
            Number of seconds to wait before retrying
        """
        orch = self._orchestrator
        base_wait = orch._provider_coordinator.get_rate_limit_wait_time(exc)
        backoff_multiplier = 2**attempt
        wait_time = base_wait * backoff_multiplier
        return min(wait_time, 300.0)

    async def _stream_with_rate_limit_retry(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
        max_retries: int = 3,
    ) -> tuple[str, Any, float, bool]:
        """Stream provider response with automatic rate limit retry.

        Args:
            tools: Available tools for the provider
            provider_kwargs: Additional kwargs for the provider
            stream_ctx: The streaming context
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (full_content, tool_calls, total_tokens, garbage_detected)
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await self._stream_provider_response_inner(
                    tools, provider_kwargs, stream_ctx
                )
            except ProviderRateLimitError as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = self._get_rate_limit_wait_time(e, attempt)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {wait_time:.1f}s before retry..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
            except Exception as e:
                exc_str = str(e).lower()
                if "rate_limit" in exc_str or "429" in exc_str or "rate limit" in exc_str:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = self._get_rate_limit_wait_time(e, attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Waiting {wait_time:.1f}s before retry..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
                else:
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Rate limit retry exhausted without exception")

    async def _stream_provider_response_inner(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """Inner implementation of stream_provider_response without retry logic.

        Args:
            tools: Available tools for the provider
            provider_kwargs: Additional kwargs for the provider
            stream_ctx: The streaming context

        Returns:
            Tuple of (full_content, tool_calls, total_tokens, garbage_detected)
        """
        orch = self._orchestrator

        full_content = ""
        tool_calls = None
        garbage_detected = False
        consecutive_garbage_chunks = 0
        max_garbage_chunks = 3
        total_tokens: float = 0

        async for chunk in orch.provider.stream(
            messages=orch.messages,
            model=orch.model,
            temperature=orch.temperature,
            max_tokens=orch.max_tokens,
            tools=tools,
            **provider_kwargs,
        ):
            chunk, consecutive_garbage_chunks, garbage_detected = self._handle_stream_chunk(
                chunk, consecutive_garbage_chunks, max_garbage_chunks, garbage_detected
            )
            if chunk is None:
                continue

            full_content += chunk.content
            stream_ctx.stream_metrics.total_chunks += 1
            if chunk.content:
                orch._metrics_collector.record_first_token()
                total_tokens += len(chunk.content) / 4
                stream_ctx.stream_metrics.total_content_length += len(chunk.content)

            if chunk.tool_calls:
                logger.debug(f"Received tool_calls in chunk: {chunk.tool_calls}")
                tool_calls = chunk.tool_calls
                stream_ctx.stream_metrics.tool_calls_count += len(chunk.tool_calls)

            if chunk.usage:
                for key in stream_ctx.cumulative_usage:
                    stream_ctx.cumulative_usage[key] += chunk.usage.get(key, 0)
                logger.debug(
                    f"Chunk usage: in={chunk.usage.get('prompt_tokens', 0)} "
                    f"out={chunk.usage.get('completion_tokens', 0)} "
                    f"cache_read={chunk.usage.get('cache_read_input_tokens', 0)}"
                )

            if tool_calls:
                break

        if garbage_detected and not tool_calls:
            logger.info("Setting force_completion due to garbage detection")

        stream_ctx.total_tokens = total_tokens
        return full_content, tool_calls, total_tokens, garbage_detected

    def _handle_stream_chunk(
        self,
        chunk: Any,
        consecutive_garbage_chunks: int,
        max_garbage_chunks: int,
        garbage_detected: bool,
    ) -> tuple[Any, int, bool]:
        """Handle garbage detection for a streaming chunk.

        Args:
            chunk: The stream chunk to check
            consecutive_garbage_chunks: Current count of consecutive garbage chunks
            max_garbage_chunks: Maximum consecutive garbage chunks allowed
            garbage_detected: Whether garbage has been detected

        Returns:
            Tuple of (chunk, consecutive_garbage_chunks, garbage_detected)
        """
        orch = self._orchestrator
        if chunk.content and orch.sanitizer.is_garbage_content(chunk.content):
            consecutive_garbage_chunks += 1
            if consecutive_garbage_chunks >= max_garbage_chunks:
                if not garbage_detected:
                    garbage_detected = True
                    logger.warning(
                        f"Garbage content detected after {len(chunk.content)} chars - stopping stream early"
                    )
                return None, consecutive_garbage_chunks, garbage_detected
        else:
            consecutive_garbage_chunks = 0
        return chunk, consecutive_garbage_chunks, garbage_detected

    # =====================================================================
    # Tool Call Processing
    # =====================================================================

    def _parse_and_validate_tool_calls(self, tool_calls: Any, full_content: str) -> tuple[Any, str]:
        """Parse, validate, and normalize tool calls.

        Delegates to ToolCoordinator which consolidates all tool-call processing.

        Args:
            tool_calls: Raw tool calls from the provider
            full_content: The full content from the response

        Returns:
            Tuple of (validated_tool_calls, full_content)
        """
        orch = self._orchestrator
        return orch._tool_coordinator.parse_and_validate_tool_calls(
            tool_calls, full_content, orch.tool_adapter
        )

    # =====================================================================
    # Recovery Integration
    # =====================================================================

    def _create_recovery_context(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> Any:
        """Create RecoveryContext from current orchestrator state.

        Delegates to orchestrator for consistency (includes model, temperature,
        unified_task_type, is_analysis_task, is_action_task fields).
        """
        return self._orchestrator._create_recovery_context(stream_ctx)

    async def _handle_recovery_with_integration(
        self,
        stream_ctx: "StreamingChatContext",
        full_content: str,
        tool_calls: Any,
        mentioned_tools: Optional[List[str]] = None,
    ) -> Any:
        """Handle recovery using the recovery integration system.

        Delegates to orchestrator which uses _recovery_coordinator for
        consistent recovery handling.
        """
        return await self._orchestrator._handle_recovery_with_integration(
            stream_ctx=stream_ctx,
            full_content=full_content,
            tool_calls=tool_calls,
            mentioned_tools=mentioned_tools,
        )

    def _apply_recovery_action(
        self, recovery_action: Any, stream_ctx: "StreamingChatContext"
    ) -> Optional[StreamChunk]:
        """Apply a recovery action and return any chunk to yield.

        Delegates to orchestrator which uses _recovery_coordinator for
        consistent recovery action application.
        """
        return self._orchestrator._apply_recovery_action(recovery_action, stream_ctx)

    async def _handle_empty_response_recovery(
        self,
        stream_ctx: "StreamingChatContext",
        tools: Any,
    ) -> tuple[bool, Any, Optional[StreamChunk]]:
        """Handle empty response recovery with multi-strategy retry.

        Attempts to recover from an empty provider response by retrying
        with increasing temperature and a nudge prompt. If recovery
        produces content or tool calls, returns them for the main loop
        to consume.

        Args:
            stream_ctx: The streaming context
            tools: Available tools

        Returns:
            Tuple of (success, tool_calls, final_chunk)
        """
        orch = self._orchestrator

        # Try recovery with increasing temperature
        recovery_temps = [0.7, 0.9]
        for temp in recovery_temps:
            try:
                # Add a nudge to prompt the model for a response
                orch.add_message(
                    "system",
                    "Please provide a response to the user's question. "
                    "If you need to use tools, go ahead. Otherwise, provide a text answer.",
                )

                provider_kwargs: Dict[str, Any] = {}
                if orch.thinking:
                    provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

                full_content = ""
                recovered_tool_calls = None

                async for chunk in orch.provider.stream(
                    messages=orch.messages,
                    model=orch.model,
                    temperature=temp,
                    max_tokens=orch.max_tokens,
                    tools=tools,
                    **provider_kwargs,
                ):
                    if chunk.content:
                        full_content += chunk.content
                    if chunk.tool_calls:
                        recovered_tool_calls = chunk.tool_calls
                        break

                if recovered_tool_calls:
                    logger.info(f"Recovery at temperature {temp} produced tool calls")
                    return True, recovered_tool_calls, None

                if full_content.strip():
                    logger.info(
                        f"Recovery at temperature {temp} produced content "
                        f"({len(full_content)} chars)"
                    )
                    sanitized = orch.sanitizer.sanitize(full_content)
                    if sanitized:
                        orch.add_message("assistant", sanitized)
                    final_chunk = orch._chunk_generator.generate_content_chunk(
                        sanitized or full_content, is_final=True
                    )
                    return True, None, final_chunk

            except Exception as e:
                logger.warning(f"Recovery attempt at temperature {temp} failed: {e}")
                continue

        # All recovery attempts failed
        return False, None, None

    # =====================================================================
    # Intelligent Response Validation
    # =====================================================================

    async def _validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Validate response quality using the intelligent pipeline.

        Delegates to orchestrator's implementation which uses the correct
        intelligent_integration property.
        """
        return await self._orchestrator._validate_intelligent_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            task_type=task_type,
        )

    # =====================================================================
    # Planning Integration
    # =====================================================================

    def _get_planning_coordinator(self) -> "PlanningCoordinator":
        """Get or create the planning coordinator.

        Returns:
            PlanningCoordinator instance
        """
        if self._planning_coordinator is None:
            from victor.agent.coordinators.planning_coordinator import PlanningCoordinator

            self._planning_coordinator = PlanningCoordinator(self._orchestrator)

        return self._planning_coordinator

    async def chat_with_planning(
        self,
        user_message: str,
        use_planning: Optional[bool] = None,
    ) -> CompletionResponse:
        """Chat with automatic planning for complex multi-step tasks.

        Delegates to _chat_with_planning when planning is enabled or auto-detected.
        For simple tasks or when planning is disabled, uses regular chat.

        Args:
            user_message: User's message
            use_planning: Force planning on/off. None = auto-detect

        Returns:
            CompletionResponse from the model

        Example:
            # Auto-detect if planning is needed
            response = await coordinator.chat_with_planning(
                "Analyze the codebase architecture and provide SOLID evaluation"
            )

            # Force planning mode
            response = await coordinator.chat_with_planning(
                "Implement user auth",
                use_planning=True
            )
        """
        # If planning is explicitly disabled, use regular chat
        if use_planning is False:
            return await self.chat(user_message)

        # If planning is explicitly enabled or auto-detected, delegate
        if use_planning is True or self._should_use_planning(user_message):
            return await self._chat_with_planning(user_message)

        # Default to regular chat for simple tasks
        return await self.chat(user_message)
