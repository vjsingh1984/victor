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
from victor.providers.base import CompletionResponse, StreamChunk
from victor.core.errors import ProviderRateLimitError

if TYPE_CHECKING:
    # Type-only imports
    from victor.agent.streaming.context import StreamingChatContext
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.streaming.intent_classification import IntentClassificationHandler
    from victor.agent.streaming.continuation import ContinuationHandler
    from victor.agent.streaming.tool_execution import ToolExecutionHandler
    from victor.agent.streaming import apply_tracking_state_updates, create_tracking_state

logger = logging.getLogger(__name__)


class ChatCoordinator:
    """Coordinator for chat and streaming chat operations.

    This class extracts chat-related logic from the orchestrator,
    providing a clean interface for managing conversations.

    The coordinator maintains a reference to the orchestrator for
    accessing shared state and delegated operations.

    Args:
        orchestrator: The AgentOrchestrator instance for accessing shared state
    """

    def __init__(self, orchestrator: "AgentOrchestrator") -> None:
        """Initialize the ChatCoordinator.

        Args:
            orchestrator: The parent orchestrator instance
        """
        self._orchestrator = orchestrator

        # Lazy-initialized handlers
        self._intent_classification_handler: Optional["IntentClassificationHandler"] = None
        self._continuation_handler: Optional["ContinuationHandler"] = None
        self._tool_execution_handler: Optional["ToolExecutionHandler"] = None

    # =====================================================================
    # Public API
    # =====================================================================

    async def chat(self, user_message: str) -> CompletionResponse:
        """Send a chat message and get response with full agentic loop.

        This method implements a proper agentic loop that:
        1. Gets model response
        2. Executes any tool calls
        3. Continues until model provides a final response (no tool calls)
        4. Ensures non-empty response on tool failures

        Args:
            user_message: User's message

        Returns:
            CompletionResponse from the model with complete response
        """
        orch = self._orchestrator

        # Ensure system prompt is included once at start of conversation
        orch.conversation.ensure_system_prompt()
        orch._system_added = True
        # Add user message to history
        orch.add_message("user", user_message)

        # Initialize tracking for this conversation turn
        orch.tool_calls_used = 0
        failure_context = ToolFailureContext()
        max_iterations = getattr(orch.settings, "chat_max_iterations", 10)
        iteration = 0

        # Classify task complexity for appropriate budgeting
        task_classification = orch.task_classifier.classify(user_message)
        iteration_budget = min(
            task_classification.tool_budget * 2, max_iterations  # Allow 2x budget for iterations
        )

        # Agentic loop: continue until no tool calls or budget exhausted
        final_response: Optional[CompletionResponse] = None

        while iteration < iteration_budget:
            iteration += 1

            # Get tool definitions if provider supports them
            tools = None
            if orch.provider.supports_tools() and orch.tool_calls_used < orch.tool_budget:
                conversation_depth = orch.conversation.message_count()
                # Use new IToolSelector API with ToolSelectionContext
                from victor.protocols import ToolSelectionContext

                context = ToolSelectionContext(
                    task_description=user_message,
                    conversation_stage=orch.conversation_state.state.stage.value if orch.conversation_state.state.stage else None,
                    previous_tools=[],
                )
                tools = await orch.tool_selector.select_tools(
                    user_message,
                    context,
                )
                tools = orch.tool_selector.prioritize_by_stage(user_message, tools)

            # Prepare optional thinking parameter
            provider_kwargs = {}
            if orch.thinking:
                provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            # Check context and compact before API call to prevent overflow
            if orch._context_compactor:
                compaction_action = orch._context_compactor.check_and_compact(
                    current_query=user_message,
                    force=False,
                    tool_call_count=orch.tool_calls_used,
                    task_complexity=task_classification.complexity.value,
                )
                if compaction_action.action_taken:
                    logger.info(
                        f"Compacted context before API call: {compaction_action.messages_removed} messages removed, "
                        f"{compaction_action.tokens_freed} tokens freed"
                    )

            # Get response from provider
            response = await orch.provider.chat(
                messages=orch.messages,
                model=orch.model,
                temperature=orch.temperature,
                max_tokens=orch.max_tokens,
                tools=tools,
                **provider_kwargs,
            )

            # Accumulate token usage for evaluation tracking (P1: Token Tracking Fix)
            if response.usage:
                orch._cumulative_token_usage["prompt_tokens"] += response.usage.get(
                    "prompt_tokens", 0
                )
                orch._cumulative_token_usage["completion_tokens"] += response.usage.get(
                    "completion_tokens", 0
                )
                orch._cumulative_token_usage["total_tokens"] += response.usage.get(
                    "total_tokens", 0
                )

            # Add assistant response to history if has content
            if response.content:
                orch.add_message("assistant", response.content)

                # Check compaction after adding assistant response
                if orch._context_compactor:
                    compaction_action = orch._context_compactor.check_and_compact(
                        current_query=user_message,
                        force=False,
                        tool_call_count=orch.tool_calls_used,
                        task_complexity=task_classification.complexity.value,
                    )
                    if compaction_action.action_taken:
                        logger.info(
                            f"Compacted context after response: {compaction_action.messages_removed} messages removed, "
                            f"{compaction_action.tokens_freed} tokens freed"
                        )

            # Check if model wants to use tools
            if response.tool_calls:
                # Handle tool calls and track results
                tool_results = await orch._handle_tool_calls(response.tool_calls)

                # Update failure context
                for result in tool_results:
                    if result.get("success"):
                        failure_context.successful_tools.append(result)
                    else:
                        failure_context.failed_tools.append(result)
                        failure_context.last_error = result.get("error")

                # Continue loop to get follow-up response
                continue

            # No tool calls - this is the final response
            final_response = response
            break

        # Ensure we have a complete response
        if final_response is None or not final_response.content:
            # Use response completer to generate a response
            completion_result = await orch.response_completer.ensure_response(
                messages=orch.messages,
                model=orch.model,
                temperature=orch.temperature,
                max_tokens=orch.max_tokens,
                failure_context=failure_context if failure_context.failed_tools else None,
            )

            if completion_result.content:
                orch.add_message("assistant", completion_result.content)
                # Create a synthetic response
                final_response = CompletionResponse(
                    content=completion_result.content,
                    role="assistant",
                    tool_calls=None,
                )
            else:
                # Last resort fallback
                fallback_content = (
                    "I was unable to generate a complete response. "
                    "Please try rephrasing your request."
                )
                if failure_context.failed_tools:
                    fallback_content = orch.response_completer.format_tool_failure_message(
                        failure_context
                    )
                # Add fallback to history and return synthetic response
                orch.add_message("assistant", fallback_content)
                final_response = CompletionResponse(
                    content=fallback_content,
                    role="assistant",
                    tool_calls=None,
                )

        return final_response

    async def stream_chat(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Stream a chat response (public entrypoint).

        This method wraps the implementation to make phased refactors safer.

        Args:
            user_message: User's input message

        Returns:
            AsyncIterator yielding StreamChunk objects with incremental response
        """
        orch = self._orchestrator
        try:
            async for chunk in self._stream_chat_impl(user_message):
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
    # Streaming Implementation
    # =====================================================================

    async def _stream_chat_impl(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Implementation for streaming chat.

        Args:
            user_message: User's message

        Yields:
            StreamChunk objects with incremental response

        Note:
            Stream metrics (TTFT, throughput) are available via get_last_stream_metrics()
            after the stream completes.

            The stream can be cancelled by calling request_cancellation(). When cancelled,
            the stream will yield a final chunk indicating cancellation and stop.

            This method uses StreamingChatContext to centralize state management.
        """
        orch = self._orchestrator

        # Initialize and prepare using StreamingChatContext
        stream_ctx = await self._create_stream_context(user_message)

        # Store context reference for handler delegation methods
        orch._current_stream_context = stream_ctx

        # Extract required files and outputs from user prompt for task completion tracking
        orch._required_files = self._extract_required_files_from_prompt(user_message)
        orch._required_outputs = self._extract_required_outputs_from_prompt(user_message)
        orch._read_files_session.clear()
        orch._all_files_read_nudge_sent = False
        logger.debug(
            f"Task requirements extracted - files: {orch._required_files}, "
            f"outputs: {orch._required_outputs}"
        )

        # Emit task requirements extracted event
        if orch._required_files or orch._required_outputs:
            from victor.core.events import get_observability_bus

            event_bus = get_observability_bus()
            event_bus.emit(
                topic="state.task.requirements_extracted",
                data={
                    "required_files": orch._required_files,
                    "required_outputs": orch._required_outputs,
                    "file_count": len(orch._required_files),
                    "output_count": len(orch._required_outputs),
                    "category": "state",
                },
            )

        # Iteration limits - kept as read-only local references for readability
        max_total_iterations = stream_ctx.max_total_iterations
        max_exploration_iterations = stream_ctx.max_exploration_iterations

        # Detect intent and inject prompt guard for non-write tasks
        self._apply_intent_guard(user_message)

        # For compound analysis+edit tasks, unified_tracker handles exploration limits
        if stream_ctx.is_analysis_task and stream_ctx.unified_task_type.value in ("edit", "create"):
            logger.info(
                f"Compound task detected (analysis+{stream_ctx.unified_task_type.value}): "
                f"unified_tracker will use appropriate exploration limits"
            )

        logger.info(
            f"Task type classification: coarse={stream_ctx.coarse_task_type}, "
            f"unified={stream_ctx.unified_task_type.value}, is_analysis={stream_ctx.is_analysis_task}, "
            f"is_action={stream_ctx.is_action_task}"
        )

        # Apply guidance for analysis/action tasks
        self._apply_task_guidance(
            user_message,
            stream_ctx.unified_task_type,
            stream_ctx.is_analysis_task,
            stream_ctx.is_action_task,
            stream_ctx.needs_execution,
            max_exploration_iterations,
        )

        # Add guidance for action-oriented tasks
        if stream_ctx.is_action_task:
            logger.info(
                f"Detected action-oriented task - allowing up to {max_exploration_iterations} exploration iterations"
            )

            if stream_ctx.needs_execution:
                orch.add_message(
                    "system",
                    "This is an action-oriented task requiring execution. "
                    "Follow this workflow: "
                    "1. CREATE the file/script with write_file or edit_files "
                    "2. EXECUTE it immediately with execute_bash (don't skip this step!) "
                    "3. SHOW the output to the user. "
                    "Minimize exploration and proceed directly to create->execute->show results.",
                )
            else:
                orch.add_message(
                    "system",
                    "This is an action-oriented task (create/write/build). "
                    "Minimize exploration and proceed directly to creating what was requested. "
                    "Only explore if absolutely necessary to complete the task.",
                )

        goals = orch._tool_planner.infer_goals_from_message(user_message)

        # Log all limits for debugging
        logger.info(
            f"Stream chat limits: "
            f"tool_budget={orch.tool_budget}, "
            f"max_total_iterations={max_total_iterations}, "
            f"max_exploration_iterations={max_exploration_iterations}, "
            f"is_analysis_task={stream_ctx.is_analysis_task}, "
            f"is_action_task={stream_ctx.is_action_task}"
        )

        # Reset debug logger for new conversation turn
        orch.debug_logger.reset()

        while True:
            # === PRE-ITERATION CHECKS (via coordinator helper) ===
            cancelled = False
            async for pre_chunk in self._run_iteration_pre_checks(stream_ctx, user_message):
                yield pre_chunk
                if pre_chunk.content == "" and getattr(pre_chunk, "is_final", False):
                    cancelled = True
            if cancelled:
                return

            # Log iteration debug info
            self._log_iteration_debug(stream_ctx, max_total_iterations)

            # === CONTEXT AND ITERATION LIMIT CHECKS ===
            max_context = self._get_max_context_chars()
            handled, iter_chunk = await self._handle_context_and_iteration_limits(
                user_message,
                max_total_iterations,
                max_context,
                stream_ctx.total_iterations,
                stream_ctx.last_quality_score,
            )
            if iter_chunk:
                yield iter_chunk
            if handled:
                break

            tools = await self._select_tools_for_turn(stream_ctx.context_msg, goals)

            # Prepare optional thinking parameter for providers that support it
            provider_kwargs = {}
            if orch.thinking:
                provider_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            full_content, tool_calls, _, garbage_detected = await self._stream_provider_response(
                tools=tools,
                provider_kwargs=provider_kwargs,
                stream_ctx=stream_ctx,
            )

            # Debug: Log response details
            content_preview = full_content[:200] if full_content else "(empty)"
            logger.debug(
                f"_stream_provider_response returned: content_len={len(full_content) if full_content else 0}, "
                f"native_tool_calls={len(tool_calls) if tool_calls else 0}, tokens={stream_ctx.total_tokens}, "
                f"garbage={garbage_detected}, content_preview={content_preview!r}"
            )

            # If garbage was detected, force completion on next iteration
            if garbage_detected and not tool_calls:
                stream_ctx.force_completion = True
                logger.info("Setting force_completion due to garbage detection")

            # Parse, validate, and normalize tool calls
            tool_calls, full_content = self._parse_and_validate_tool_calls(tool_calls, full_content)

            # Task Completion Detection Enhancement
            if orch._task_completion_detector and full_content:
                from victor.agent.task_completion import CompletionConfidence

                orch._task_completion_detector.analyze_response(full_content)
                confidence = orch._task_completion_detector.get_completion_confidence()

                if confidence == CompletionConfidence.HIGH:
                    logger.info(
                        "Task completion: HIGH confidence detected (active signal), "
                        "forcing completion after this response"
                    )
                    stream_ctx.force_completion = True
                elif confidence == CompletionConfidence.MEDIUM:
                    logger.info(
                        "Task completion: MEDIUM confidence detected (file mods + passive signal)"
                    )

            # Initialize mentioned_tools_detected for later use in continuation action
            mentioned_tools_detected: List[str] = []

            # Check for mentioned tools early for recovery integration
            from victor.agent.continuation_strategy import ContinuationStrategy
            from victor.tools.tool_names import get_all_canonical_names, TOOL_ALIASES

            if full_content and not tool_calls:
                mentioned_tools_detected = ContinuationStrategy.detect_mentioned_tools(
                    full_content, list(get_all_canonical_names()), TOOL_ALIASES
                )

            # Use recovery integration to detect and handle failures
            recovery_action = await self._handle_recovery_with_integration(
                stream_ctx=stream_ctx,
                full_content=full_content,
                tool_calls=tool_calls,
                mentioned_tools=mentioned_tools_detected or None,
            )

            # Apply recovery action if not just "continue"
            if recovery_action.action != "continue":
                recovery_chunk = self._apply_recovery_action(recovery_action, stream_ctx)
                if recovery_chunk:
                    yield recovery_chunk
                    if recovery_chunk.is_final:
                        orch._recovery_integration.record_outcome(success=False)
                        return
                if recovery_action.action in ("retry", "force_summary"):
                    continue

            if full_content:
                # Sanitize response to remove malformed patterns from local models
                sanitized = orch.sanitizer.sanitize(full_content)
                if sanitized:
                    orch.add_message("assistant", sanitized)
                else:
                    plain_text = orch.sanitizer.strip_markup(full_content)
                    if plain_text:
                        orch.add_message("assistant", plain_text)

                # Log if model mentioned tools but didn't execute them
                if mentioned_tools_detected:
                    tools_str = ", ".join(mentioned_tools_detected)
                    logger.info(
                        f"Model mentioned tool(s) [{tools_str}] in text without executing. "
                        "Common with local models - tool syntax detected in response content."
                    )
            elif not tool_calls:
                # No content and no tool calls - check for natural completion
                recovery_ctx = self._create_recovery_context(stream_ctx)
                final_chunk = orch._recovery_coordinator.check_natural_completion(
                    recovery_ctx, has_tool_calls=False, content_length=0
                )
                if final_chunk:
                    yield final_chunk
                    return

                # No substantial content yet - attempt aggressive recovery
                logger.warning("Model returned empty response - attempting aggressive recovery")

                recovery_ctx = self._create_recovery_context(stream_ctx)
                recovery_chunk, should_force = orch._recovery_coordinator.handle_empty_response(
                    recovery_ctx
                )
                if recovery_chunk:
                    yield recovery_chunk
                    continue

                # Delegate empty response recovery to helper method
                recovery_success, recovered_tool_calls, final_chunk = (
                    await self._handle_empty_response_recovery(stream_ctx, tools)
                )

                if recovery_success:
                    if final_chunk:
                        yield final_chunk
                        return
                    elif recovered_tool_calls:
                        tool_calls = recovered_tool_calls
                        logger.info(
                            f"Recovery produced {len(tool_calls)} tool call(s) - continuing main loop"
                        )
                else:
                    recovery_ctx = self._create_recovery_context(stream_ctx)
                    fallback_msg = orch._recovery_coordinator.get_recovery_fallback_message(
                        recovery_ctx
                    )
                    orch._record_intelligent_outcome(
                        success=False,
                        quality_score=0.3,
                        user_satisfied=False,
                        completed=False,
                    )
                    yield orch._chunk_generator.generate_content_chunk(fallback_msg, is_final=True)
                    return

            # Record tool calls in progress tracker for loop detection
            for tc in tool_calls or []:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                orch.unified_tracker.record_tool_call(tool_name, tool_args)

            content_length = len(full_content.strip())

            # Record iteration in unified tracker
            orch.unified_tracker.record_iteration(content_length)

            # Increment iteration counter AFTER completing iteration work
            stream_ctx.increment_iteration()

            # Intelligent pipeline post-iteration hook: validate response quality
            if full_content and len(full_content.strip()) > 50:
                quality_result = await self._validate_intelligent_response(
                    response=full_content,
                    query=user_message,
                    tool_calls=orch.tool_calls_used,
                    task_type=stream_ctx.unified_task_type.value,
                )
                if quality_result and not quality_result.is_grounded:
                    issues = quality_result.grounding_issues
                    if issues:
                        logger.warning(
                            f"IntelligentPipeline detected grounding issues: {issues[:3]}"
                        )
                    if quality_result.should_retry:
                        grounding_feedback = quality_result.grounding_feedback
                        if grounding_feedback:
                            logger.info(
                                f"Injecting grounding feedback for retry: {len(grounding_feedback)} chars"
                            )
                            stream_ctx.pending_grounding_feedback = grounding_feedback

                if quality_result:
                    new_score = quality_result.quality_score
                    stream_ctx.update_quality_score(new_score)

                if quality_result and quality_result.should_finalize:
                    finalize_reason = quality_result.finalize_reason or "grounding limit exceeded"
                    logger.warning(
                        f"Force finalize triggered: {finalize_reason}. "
                        "Stopping continuation to prevent infinite loop."
                    )
                    orch._force_finalize = True

            # Check for loop warning via streaming handler
            unified_loop_warning = orch.unified_tracker.check_loop_warning()
            loop_warning_chunk = orch._streaming_handler.handle_loop_warning(
                stream_ctx, unified_loop_warning
            )
            if loop_warning_chunk:
                logger.warning(f"UnifiedTaskTracker loop warning: {unified_loop_warning}")
                yield loop_warning_chunk
            else:
                # Check UnifiedTaskTracker for stop decision via recovery coordinator
                recovery_ctx = self._create_recovery_context(stream_ctx)
                was_triggered, hint = orch._recovery_coordinator.check_force_action(recovery_ctx)
                if was_triggered:
                    logger.info(
                        f"UnifiedTaskTracker forcing action: {hint}, "
                        f"metrics={orch.unified_tracker.get_metrics()}"
                    )

                logger.debug(f"After streaming pass, tool_calls = {tool_calls}")

                if not tool_calls:
                    # === INTENT CLASSIFICATION (P0 SRP refactor) ===
                    if not self._intent_classification_handler:
                        from victor.agent.streaming import create_intent_classification_handler

                        self._intent_classification_handler = create_intent_classification_handler(
                            orch
                        )

                    # Import create_tracking_state for intent classification
                    from victor.agent.streaming.intent_classification import create_tracking_state, apply_tracking_state_updates

                    # Ensure tracking variables are initialized
                    if not hasattr(orch, "_continuation_prompts"):
                        orch._continuation_prompts = 0
                    if not hasattr(orch, "_asking_input_prompts"):
                        orch._asking_input_prompts = 0
                    if not hasattr(orch, "_consecutive_blocked_attempts"):
                        orch._consecutive_blocked_attempts = 0
                    if not hasattr(orch, "_cumulative_prompt_interventions"):
                        orch._cumulative_prompt_interventions = 0

                    tracking_state = create_tracking_state(orch)

                    intent_result = (
                        self._intent_classification_handler.classify_and_determine_action(
                            stream_ctx=stream_ctx,
                            full_content=full_content,
                            content_length=content_length,
                            mentioned_tools=mentioned_tools_detected,
                            tracking_state=tracking_state,
                        )
                    )

                    for chunk in intent_result.chunks:
                        yield chunk

                    if intent_result.content_cleared:
                        full_content = ""

                    force_finalize_used = (
                        tracking_state.force_finalize and intent_result.action == "finish"
                    )
                    apply_tracking_state_updates(
                        orch, intent_result.state_updates, force_finalize_used
                    )

                    action_result = intent_result.action_result
                    action = intent_result.action

                    logger.info(
                        f"Continuation action: {action} - {action_result.get('reason', 'unknown')}"
                    )

                    # === CONTINUATION ACTION HANDLING (P0 SRP refactor) ===
                    if not self._continuation_handler:
                        from victor.agent.streaming import create_continuation_handler

                        self._continuation_handler = create_continuation_handler(orch)

                    action_result["action"] = action

                    continuation_result = await self._continuation_handler.handle_action(
                        action_result=action_result,
                        stream_ctx=stream_ctx,
                        full_content=full_content,
                    )

                    for chunk in continuation_result.chunks:
                        yield chunk

                    if "cumulative_prompt_interventions" in continuation_result.state_updates:
                        orch._cumulative_prompt_interventions = continuation_result.state_updates[
                            "cumulative_prompt_interventions"
                        ]

                    if continuation_result.should_return:
                        return

                # === TOOL EXECUTION PHASE (P0 SRP refactor) ===
                if not self._tool_execution_handler:
                    from victor.agent.streaming import create_tool_execution_handler

                    self._tool_execution_handler = create_tool_execution_handler(orch)

                self._tool_execution_handler.update_observed_files(
                    set(orch.observed_files) if orch.observed_files else set()
                )

                tool_exec_result = await self._tool_execution_handler.execute_tools(
                    stream_ctx=stream_ctx,
                    tool_calls=tool_calls,
                    user_message=user_message,
                    full_content=full_content,
                    tool_calls_used=orch.tool_calls_used,
                    tool_budget=orch.tool_budget,
                )

                for chunk in tool_exec_result.chunks:
                    yield chunk

                orch.tool_calls_used += tool_exec_result.tool_calls_executed

                if tool_exec_result.should_return:
                    return

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

        Args:
            user_message: The user's message

        Returns:
            Dictionary with classification results including is_analysis_task,
            is_action_task, needs_execution, and coarse_task_type
        """
        # Check for analysis keywords
        analysis_keywords = [
            "analyze",
            "analysis",
            "investigate",
            "explore",
            "examine",
            "review",
            "audit",
            "inspect",
            "study",
            "understand",
            "explain",
            "find",
            "search",
            "look for",
            "identify",
            "locate",
            "trace",
            "debug",
            "troubleshoot",
            "diagnose",
        ]
        is_analysis_task = any(keyword in user_message.lower() for keyword in analysis_keywords)

        # Check for action keywords
        action_keywords = [
            "create",
            "write",
            "build",
            "implement",
            "add",
            "generate",
            "make",
            "construct",
            "develop",
            "produce",
            "form",
            "compose",
        ]
        is_action_task = any(keyword in user_message.lower() for keyword in action_keywords)

        # Check for execution keywords
        execution_keywords = ["run", "execute", "test", "deploy", "launch"]
        needs_execution = any(keyword in user_message.lower() for keyword in execution_keywords)

        # Determine coarse task type
        if is_analysis_task:
            coarse_task_type = "analysis"
        elif is_action_task:
            coarse_task_type = "action"
        elif needs_execution:
            coarse_task_type = "execution"
        else:
            coarse_task_type = "default"

        return {
            "is_analysis_task": is_analysis_task,
            "is_action_task": is_action_task,
            "needs_execution": needs_execution,
            "coarse_task_type": coarse_task_type,
        }

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
        # Use new IToolSelector API with ToolSelectionContext
        from victor.protocols import ToolSelectionContext

        context = ToolSelectionContext(
            task_description=context_msg,
            conversation_stage=orch.conversation_state.state.stage.value if orch.conversation_state.state.stage else None,
            previous_tools=[],
            planned_tools=planned_tools if planned_tools else [],
        )
        tools = await orch.tool_selector.select_tools(
            context_msg,
            context,
        )
        logger.info(
            f"context_msg={context_msg}\nconversation_stage={orch.conversation_state.state.stage.value if orch.conversation_state.state.stage else None}"
        )
        tools = orch.tool_selector.prioritize_by_stage(context_msg, tools)
        current_intent = getattr(orch, "_current_intent", None)
        tools = orch._tool_planner.filter_tools_by_intent(tools, current_intent)
        return tools

    def _extract_required_files_from_prompt(self, user_message: str) -> List[str]:
        """Extract required file paths from the user message.

        Args:
            user_message: The user's message

        Returns:
            List of required file paths mentioned in the message

        Note:
            PromptRequirements extracts counts, not actual file paths.
            This method returns empty list since file path extraction
            is not implemented. Use requirements.file_count for budgeting.
        """
        from victor.agent.prompt_requirement_extractor import extract_prompt_requirements

        requirements = extract_prompt_requirements(user_message)
        # PromptRequirements only has counts, not file paths
        # Return empty list - file paths extracted elsewhere via patterns
        return []

    def _extract_required_outputs_from_prompt(self, user_message: str) -> List[str]:
        """Extract required outputs from the user message.

        Args:
            user_message: The user's message

        Returns:
            List of required outputs mentioned in the message

        Note:
            PromptRequirements extracts counts, not actual output paths.
            This method returns empty list since output path extraction
            is not implemented. Use requirements for budgeting only.
        """
        from victor.agent.prompt_requirement_extractor import extract_prompt_requirements

        requirements = extract_prompt_requirements(user_message)
        # PromptRequirements only has counts, not output paths
        return []

    def _get_max_context_chars(self) -> int:
        """Get the maximum context length in characters.

        Returns:
            Maximum context length for the current provider/model
        """
        orch = self._orchestrator
        # Use context_manager which has provider-aware context limits
        return orch._context_manager.get_max_context_chars()

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

        # 4. Inject grounding feedback if pending
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
        """Handle context length and iteration limit checks.

        Args:
            user_message: The user's message
            max_total_iterations: Maximum total iterations
            max_context: Maximum context length
            total_iterations: Current iteration count
            last_quality_score: Last quality score

        Returns:
            Tuple of (handled, chunk) where handled is True if limits were hit
        """
        orch = self._orchestrator

        # Check context length
        context_length = sum(len(msg.content) for msg in orch.messages)
        if context_length > max_context * 0.9:  # 90% threshold
            logger.warning(f"Context length approaching limit: {context_length}/{max_context}")
            # Trigger compaction
            if orch._context_compactor:
                compaction_action = orch._context_compactor.check_and_compact(
                    current_query=user_message,
                    force=True,
                    tool_call_count=orch.tool_calls_used,
                    task_complexity=TaskComplexity.COMPLEX.value,
                )
                if compaction_action.action_taken:
                    logger.info(
                        f"Emergency compaction: {compaction_action.messages_removed} messages removed"
                    )

        # Check iteration limit
        if total_iterations >= max_total_iterations:
            logger.warning(f"Iteration limit reached: {total_iterations}/{max_total_iterations}")
            chunk = StreamChunk(
                content=f"\n\n[Reached maximum iterations ({max_total_iterations}) - providing final response]\n",
                is_final=False,
            )
            return True, chunk

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

        Args:
            tool_calls: Raw tool calls from the provider
            full_content: The full content from the response

        Returns:
            Tuple of (validated_tool_calls, full_content)
        """
        if tool_calls:
            for tc in tool_calls:
                if tc.get("name"):
                    logger.debug(f"Tool call: {tc.get('name')}")
                # Normalize arguments if needed
                if tc.get("arguments") is None:
                    tc["arguments"] = {}

        return tool_calls, full_content

    # =====================================================================
    # Recovery Integration
    # =====================================================================

    def _create_recovery_context(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> Any:
        """Create RecoveryContext from current orchestrator state.

        Args:
            stream_ctx: The streaming context

        Returns:
            StreamingRecoveryContext with all necessary state
        """
        import time
        from victor.agent.recovery_coordinator import StreamingRecoveryContext

        orch = self._orchestrator

        elapsed_time = 0.0
        if orch._streaming_controller.current_session:
            elapsed_time = time.time() - orch._streaming_controller.current_session.start_time

        return StreamingRecoveryContext(
            iteration=stream_ctx.total_iterations,
            elapsed_time=elapsed_time,
            tool_calls_used=orch.tool_calls_used,
            tool_budget=orch.tool_budget,
            max_iterations=stream_ctx.max_total_iterations,
            session_start_time=(
                orch._streaming_controller.current_session.start_time
                if orch._streaming_controller.current_session
                else time.time()
            ),
            last_quality_score=stream_ctx.last_quality_score,
            streaming_context=stream_ctx,
            provider_name=orch.provider_name,
            model=orch.model,
            temperature=getattr(orch, "temperature", 0.7),
            unified_task_type=getattr(orch._task_tracker, "current_task_type", None) if hasattr(orch, "_task_tracker") else None,
            is_analysis_task=getattr(orch._task_tracker, "is_analysis_task", False) if hasattr(orch, "_task_tracker") else False,
            is_action_task=getattr(orch._task_tracker, "is_action_task", False) if hasattr(orch, "_task_tracker") else False,
        )

    async def _handle_recovery_with_integration(
        self,
        stream_ctx: "StreamingChatContext",
        full_content: str,
        tool_calls: Any,
        mentioned_tools: Optional[List[str]] = None,
    ) -> Any:
        """Handle recovery using the recovery integration system.

        Args:
            stream_ctx: The streaming context
            full_content: The full content from the response
            tool_calls: Tool calls from the response
            mentioned_tools: Tools mentioned in the content

        Returns:
            RecoveryAction with the action to take
        """
        orch = self._orchestrator
        # Call handle_response with individual parameters instead of detect_and_handle
        return await orch._recovery_integration.handle_response(
            content=full_content,
            tool_calls=tool_calls,
            mentioned_tools=mentioned_tools,
            provider_name=orch.provider_name,
            model_name=orch.model,
            tool_calls_made=orch.tool_calls_used,
            tool_budget=orch.tool_budget,
            iteration_count=stream_ctx.total_iterations,
            max_iterations=stream_ctx.max_total_iterations,
            current_temperature=getattr(orch, "temperature", 0.7),
            quality_score=stream_ctx.last_quality_score,
            task_type=getattr(orch._task_tracker, "current_task_type", "general") if hasattr(orch, "_task_tracker") else "general",
            is_analysis_task=getattr(orch._task_tracker, "is_analysis_task", False) if hasattr(orch, "_task_tracker") else False,
            is_action_task=getattr(orch._task_tracker, "is_action_task", False) if hasattr(orch, "_task_tracker") else False,
            context_utilization=None,
        )

    def _apply_recovery_action(
        self, recovery_action: Any, stream_ctx: "StreamingChatContext"
    ) -> Optional[StreamChunk]:
        """Apply a recovery action and return any chunk to yield.

        Args:
            recovery_action: The recovery action to apply
            stream_ctx: The streaming context

        Returns:
            StreamChunk to yield, or None
        """
        orch = self._orchestrator

        if recovery_action.action == "force_summary":
            stream_ctx.force_completion = True
            return orch._chunk_generator.generate_content_chunk(
                "Providing summary based on information gathered so far.",
                is_final=True
            )
        elif recovery_action.action == "retry":
            orch.add_message("system", recovery_action.message or "Please try again.")
            return None
        elif recovery_action.action == "finalize":
            return orch._chunk_generator.generate_content_chunk(
                recovery_action.message or "", is_final=True
            )
        return None

    async def _handle_empty_response_recovery(
        self,
        stream_ctx: "StreamingChatContext",
        tools: Any,
    ) -> tuple[bool, Any, Optional[StreamChunk]]:
        """Handle empty response recovery.

        Args:
            stream_ctx: The streaming context
            tools: Available tools

        Returns:
            Tuple of (success, tool_calls, final_chunk)
        """
        orch = self._orchestrator

        # Try to recover with a simple prompt
        try:
            orch.add_message("system", "Please provide a response.")
            response = await orch.provider.chat(
                messages=orch.messages,
                model=orch.model,
                temperature=orch.temperature,
                max_tokens=orch.max_tokens,
                tools=None,
            )
            if response and response.content:
                return True, None, StreamChunk(content=response.content, is_final=True)
        except Exception as e:
            logger.warning(f"Empty response recovery failed: {e}")

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

        Args:
            response: The response to validate
            query: The original query
            tool_calls: Number of tool calls used
            task_type: The task type

        Returns:
            Validation result dict or None
        """
        orch = self._orchestrator

        if not orch._intelligent_integration:
            return None

        try:
            return await orch._intelligent_integration.validate_response(
                response=response,
                query=query,
                tool_calls=tool_calls,
                task_type=task_type,
            )
        except Exception as e:
            logger.warning(f"Intelligent validation failed: {e}")
            return None
