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
from typing import Any, AsyncIterator, Dict, Optional, TYPE_CHECKING

from victor.framework.task import TaskComplexity
from victor.agent.unified_task_tracker import TrackerTaskType
from victor.agent.prompt_requirement_extractor import extract_prompt_requirements
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
    from victor.agent.token_tracker import TokenTracker

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

    def __init__(
        self,
        orchestrator: "ChatOrchestratorProtocol",
        token_tracker: Optional["TokenTracker"] = None,
    ) -> None:
        """Initialize the ChatCoordinator.

        Args:
            orchestrator: Object satisfying ChatOrchestratorProtocol
            token_tracker: Optional centralized token tracker. When provided,
                streaming token usage is accumulated through the tracker
                instead of direct dict mutation on the orchestrator.
        """
        self._orchestrator = orchestrator
        self._token_tracker = token_tracker

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
            from victor.agent.coordinators.execution_coordinator import (
                ExecutionCoordinator,
            )

            # Create protocol adapter for orchestrator
            from victor.agent.coordinators.protocol_adapters import (
                OrchestratorProtocolAdapter,
            )

            adapter = OrchestratorProtocolAdapter(self._orchestrator)

            # Initialize execution coordinator with protocol-based dependencies
            self._execution_coordinator = ExecutionCoordinator(
                chat_context=adapter,
                tool_context=adapter,
                provider_context=adapter,
                execution_provider=adapter,
                token_tracker=self._token_tracker,
            )
        return self._execution_coordinator

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
        # If planning is explicitly disabled, skip planning check
        if use_planning is False:
            return await self.execution_coordinator.execute_agentic_loop(user_message)

        # Check if we should use planning for this task (explicit or auto-detected)
        if (use_planning is True or use_planning is None) and self._should_use_planning(
            user_message
        ):
            return await self._chat_with_planning(user_message)

        # Default: delegate to execution coordinator for agentic loop
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
            if orch.has_capability("current_stream_context") and orch.get_capability_value(
                "current_stream_context"
            ):
                ctx = orch.get_capability_value("current_stream_context")
                if hasattr(ctx, "cumulative_usage"):
                    if self._token_tracker is not None:
                        self._token_tracker.accumulate(ctx.cumulative_usage)
                    else:
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
        if orch.has_capability("usage_analytics") and orch.get_capability_value("usage_analytics"):
            orch.get_capability_value("usage_analytics").start_session()

        # Clear ToolSequenceTracker history for new conversation
        if orch.has_capability("tool_sequence_tracker") and orch.get_capability_value(
            "tool_sequence_tracker"
        ):
            orch.get_capability_value("tool_sequence_tracker").clear_history()

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
        if orch.has_capability("usage_analytics") and orch.get_capability_value("usage_analytics"):
            orch.get_capability_value("usage_analytics").record_turn()

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
        task_keywords = orch._classify_task_keywords(user_message)

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

        # Q&A detection: skip tools for pure question/display-only tasks
        from victor.agent.coordinators.execution_coordinator import ExecutionCoordinator

        ctx.is_qa_task = ExecutionCoordinator._is_question_only(user_message)

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

        assembled = orch.get_assembled_messages(
            current_query=stream_ctx.user_message if stream_ctx else None
        )
        async for chunk in orch.provider.stream(
            messages=assembled,
            model=orch.model,
            temperature=orch.temperature,
            max_tokens=orch.max_tokens,
            tools=tools,
            **provider_kwargs,
        ):
            chunk, consecutive_garbage_chunks, garbage_detected = self._handle_stream_chunk(
                chunk,
                consecutive_garbage_chunks,
                max_garbage_chunks,
                garbage_detected,
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
    # Recovery Integration
    # =====================================================================

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
                    provider_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": 10000,
                    }

                full_content = ""
                recovered_tool_calls = None

                retry_assembled = orch.get_assembled_messages(
                    current_query=stream_ctx.user_message if stream_ctx else None
                )
                async for chunk in orch.provider.stream(
                    messages=retry_assembled,
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
    ) -> None:
        """Persist a message to memory and log the event.

        Offloads blocking SQLite I/O to the thread pool when an event
        loop is running, preventing async caller blocking.

        Args:
            role: Message role (user, assistant, system).
            content: Message content text.
            memory_manager: MemoryManager instance (or None).
            memory_session_id: Active memory session ID (or None).
            usage_logger: UsageLogger for event logging.
        """
        # Persist to memory manager if available
        if memory_manager and memory_session_id:
            try:
                from victor.agent.conversation_memory import MessageRole

                role_map = {
                    "user": MessageRole.USER,
                    "assistant": MessageRole.ASSISTANT,
                    "system": MessageRole.SYSTEM,
                }
                msg_role = role_map.get(role, MessageRole.USER)

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is not None and loop.is_running():
                    loop.run_in_executor(
                        None,
                        memory_manager.add_message,
                        memory_session_id,
                        msg_role,
                        content,
                    )
                else:
                    memory_manager.add_message(
                        session_id=memory_session_id,
                        role=msg_role,
                        content=content,
                    )
            except Exception as e:
                logging.getLogger(__name__).debug("Failed to persist message: %s", e)

        # Log usage event
        if role == "user":
            usage_logger.log_event("user_prompt", {"content": content})
        elif role == "assistant":
            usage_logger.log_event("assistant_response", {"content": content})
            if hasattr(usage_logger, "set_reasoning_context") and content:
                usage_logger.set_reasoning_context(content)
