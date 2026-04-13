"""Streaming chat pipeline implementation."""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator, List, Optional, TYPE_CHECKING

from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.coordinators.chat_coordinator import ChatCoordinator

logger = logging.getLogger(__name__)


def _extract_required_files_from_prompt(user_message: str) -> List[str]:
    """Extract required file paths from the user message."""
    pattern = r'(?:^|[\s"\'`])(/[\w./\-]+\.\w+|\.{1,2}/[\w./\-]+\.\w+)'
    matches = re.findall(pattern, user_message)
    return list(set(matches))


def _extract_required_outputs_from_prompt(user_message: str) -> List[str]:
    """Extract required outputs from the user message."""
    return []


def _get_decision_service(orchestrator: object) -> object | None:
    """Get the LLM decision service from the container if available."""
    try:
        from victor.core.feature_flags import FeatureFlag, is_feature_enabled

        if not is_feature_enabled(FeatureFlag.USE_LLM_DECISION_SERVICE):
            return None
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )

        container = getattr(orchestrator, "_container", None)
        if container is not None:
            return container.get(LLMDecisionServiceProtocol)
    except Exception as e:
        logger.debug("Failed to resolve LLM decision service: %s", e)
    return None


class StreamingChatPipeline:
    """Canonical streaming pipeline wired to ChatCoordinator helpers."""

    def __init__(self, coordinator: "ChatCoordinator") -> None:
        self._coordinator = coordinator
        # Tool selection cache — avoids redundant selection within same turn
        self._last_tool_context: Optional[str] = None
        self._last_tools: Optional[Any] = None

    async def _get_tools_cached(self, orch: Any, context_msg: str, goals: Any) -> Any:
        """Select tools with per-turn caching.

        If context_msg hasn't changed since the last call, returns the
        previously selected tools. This avoids redundant semantic search
        and tool schema serialization within the same streaming turn.
        """
        if self._last_tool_context == context_msg and self._last_tools is not None:
            return self._last_tools

        tools = await orch._select_tools_for_turn(context_msg, goals)
        if tools is not None:
            self._last_tool_context = context_msg
            self._last_tools = tools
        return tools

    async def run(self, user_message: str) -> AsyncIterator[StreamChunk]:
        """Run the streaming pipeline for the provided message."""
        coord = self._coordinator
        orch = coord._orchestrator

        # Initialize and prepare using StreamingChatContext
        stream_ctx = await coord._create_stream_context(user_message)

        # Store context reference for handler delegation methods
        orch._current_stream_context = stream_ctx

        # Extract required files and outputs from user prompt for task completion tracking
        orch._required_files = _extract_required_files_from_prompt(user_message)
        orch._required_outputs = _extract_required_outputs_from_prompt(user_message)
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
            await event_bus.emit(
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
        orch._apply_intent_guard(user_message)

        # For compound analysis+edit tasks, unified_tracker handles exploration limits
        if stream_ctx.is_analysis_task and stream_ctx.unified_task_type.value in (
            "edit",
            "create",
        ):
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
        orch._apply_task_guidance(
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

        # Reset LLM decision service budget for this turn (if available)
        decision_service = _get_decision_service(orch)
        if decision_service is not None:
            decision_service.reset_budget()

        # Spin detection: consecutive non-tool short responses indicate the model
        # is stuck describing actions instead of executing them (common with local models)
        _consecutive_empty_tool_responses = 0
        _MAX_CONSECUTIVE_EMPTY = 3  # Break after 3 consecutive non-tool short responses

        while True:
            # === PRE-ITERATION CHECKS (via coordinator helper) ===
            cancelled = False
            async for pre_chunk in coord._run_iteration_pre_checks(stream_ctx, user_message):
                yield pre_chunk
                if pre_chunk.content == "" and getattr(pre_chunk, "is_final", False):
                    cancelled = True
            if cancelled:
                return

            # Log iteration debug info
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

            # === CONTEXT AND ITERATION LIMIT CHECKS ===
            max_context = orch._get_max_context_chars()
            handled, iter_chunk = await orch._handle_context_and_iteration_limits(
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

            # Q&A bypass: skip tools entirely for pure Q&A tasks.
            # This avoids sending 48 tool schemas to local models (15K tokens overhead).
            if getattr(stream_ctx, "is_qa_task", False):
                tools = None
            # Caching providers: use session-locked full tool set (90% discount)
            elif orch.get_session_tools() is not None:
                tools = orch.get_session_tools()
            # Non-caching providers: per-turn semantic selection
            else:
                tools = await self._get_tools_cached(orch, stream_ctx.context_msg, goals)

            # Prepare optional thinking parameter for providers that support it
            provider_kwargs = {}
            if orch.thinking:
                provider_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000,
                }

            full_content, tool_calls, _, garbage_detected = await coord._stream_provider_response(
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
            tool_calls, full_content = orch._parse_and_validate_tool_calls(tool_calls, full_content)

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
                all_tool_names = get_all_canonical_names() | set(TOOL_ALIASES.keys())
                mentioned_tools_detected = ContinuationStrategy.detect_mentioned_tools(
                    full_content, list(all_tool_names), TOOL_ALIASES
                )

                # Tool format assist: if model describes a tool but didn't call it,
                # inject a format hint as a system message so it can self-correct.
                # This is more effective than generic "use tools" encouragement.
                if mentioned_tools_detected and _consecutive_empty_tool_responses >= 1:
                    tool_hint = mentioned_tools_detected[0]
                    logger.info(
                        f"[tool-format-assist] Model mentioned '{tool_hint}' without calling it "
                        f"(spin #{_consecutive_empty_tool_responses}). Injecting format hint."
                    )
                    orch.add_message(
                        "system",
                        f"You described wanting to use '{tool_hint}' but didn't call it. "
                        f"Call the tool directly — don't describe what you want to do, execute it. "
                        f"If you've already modified the file successfully, say _DONE_.",
                    )

            # Use recovery integration to detect and handle failures
            recovery_action = await orch._handle_recovery_with_integration(
                stream_ctx=stream_ctx,
                full_content=full_content,
                tool_calls=tool_calls,
                mentioned_tools=mentioned_tools_detected or None,
            )

            # Apply recovery action if not just "continue"
            if recovery_action.action != "continue":
                recovery_chunk = orch._apply_recovery_action(recovery_action, stream_ctx)
                if recovery_chunk:
                    yield recovery_chunk
                    if recovery_chunk.is_final:
                        orch._recovery_integration.record_outcome(success=False)
                        return
                if recovery_action.action in ("retry", "force_summary"):
                    continue

            # Spin detection for ALL paths (including recovery "continue"):
            # Check if the tool pipeline blocked all calls in the last batch.
            # This catches loops where the model repeatedly calls the same tool
            # and dedup blocks every attempt — regardless of recovery action.
            _pipeline_obj = getattr(orch, "_tool_pipeline", None)
            if _pipeline_obj and getattr(_pipeline_obj, "last_batch_all_skipped", False):
                _consecutive_empty_tool_responses += 1
                logger.info(
                    f"[spin-check] all_blocked=True spin={_consecutive_empty_tool_responses}/{_MAX_CONSECUTIVE_EMPTY}"
                )
                if _consecutive_empty_tool_responses == 2:
                    logger.info("[approach-pivot] All tools blocked. Injecting approach change.")
                    orch.add_message(
                        "system",
                        "Your last tool calls were blocked because you already called them. "
                        "Try a DIFFERENT tool or different arguments. "
                        "If you've found what you need, proceed to edit the code. "
                        "If you've already made the fix, say _DONE_.",
                    )
                if _consecutive_empty_tool_responses >= _MAX_CONSECUTIVE_EMPTY:
                    logger.warning(
                        f"[spin-detect] {_consecutive_empty_tool_responses} consecutive "
                        f"blocked tool calls. Breaking loop."
                    )
                    yield orch._chunk_generator.generate_content_chunk(
                        "\n\n[Agent detected a tool-call loop — all attempts were "
                        "blocked by deduplication. Try a different approach.]",
                        is_final=True,
                    )
                    return

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
                recovery_ctx = orch._create_recovery_context(stream_ctx)
                final_chunk = orch._recovery_coordinator.check_natural_completion(
                    recovery_ctx, has_tool_calls=False, content_length=0
                )
                if final_chunk:
                    yield final_chunk
                    return

                # No substantial content yet - attempt aggressive recovery
                logger.warning("Model returned empty response - attempting aggressive recovery")

                recovery_ctx = orch._create_recovery_context(stream_ctx)
                recovery_chunk, should_force = orch._recovery_coordinator.handle_empty_response(
                    recovery_ctx
                )
                if recovery_chunk:
                    yield recovery_chunk
                    continue

                # Delegate empty response recovery to helper method
                recovery_success, recovered_tool_calls, final_chunk = (
                    await coord._handle_empty_response_recovery(stream_ctx, tools)
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
                    recovery_ctx = orch._create_recovery_context(stream_ctx)
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

            # Spin detection: count iterations with no meaningful progress.
            # Check AFTER tool execution (last_batch_all_skipped is set by pipeline)
            _all_tools_blocked = False
            _pipeline_obj = getattr(orch, "_tool_pipeline", None)
            if _pipeline_obj and getattr(_pipeline_obj, "last_batch_all_skipped", False):
                _all_tools_blocked = True

            _no_progress = (not tool_calls and content_length < 120) or _all_tools_blocked
            logger.info(
                f"[spin-check] tool_calls={len(tool_calls) if tool_calls else 0} "
                f"content_len={content_length} all_blocked={_all_tools_blocked} "
                f"no_progress={_no_progress} spin={_consecutive_empty_tool_responses}"
            )
            if _no_progress:
                _consecutive_empty_tool_responses += 1

                # Approach pivot: at spin 2, suggest the model try a different approach
                if _consecutive_empty_tool_responses == 2:
                    logger.info(
                        "[approach-pivot] Model stuck for 2 iterations. "
                        "Injecting approach change suggestion."
                    )
                    orch.add_message(
                        "system",
                        "You seem stuck. Try a DIFFERENT approach: "
                        "if reading failed, try code_search instead. "
                        "If editing failed, re-read the file first to get exact content. "
                        "If you've already made the fix, say _DONE_.",
                    )

                if _consecutive_empty_tool_responses >= _MAX_CONSECUTIVE_EMPTY:
                    logger.warning(
                        f"[spin-detect] {_consecutive_empty_tool_responses} consecutive non-tool "
                        f"short responses — model is stuck describing actions instead of "
                        f"executing them. Breaking loop. Last response: {full_content[:100]!r}"
                    )
                    yield orch._chunk_generator.generate_content_chunk(
                        "\n\n[Agent detected a response loop — breaking to prevent wasted time. "
                        "The model described tool calls without executing them.]",
                        is_final=True,
                    )
                    return
                logger.info(
                    f"[spin-detect] Non-tool short response #{_consecutive_empty_tool_responses}"
                    f"/{_MAX_CONSECUTIVE_EMPTY}: {content_length} chars, "
                    f"content: {full_content[:80]!r}"
                )
            else:
                _consecutive_empty_tool_responses = 0  # Reset on successful tool call

            # Record iteration in unified tracker
            orch.unified_tracker.record_iteration(content_length)

            # Intelligent pipeline post-iteration hook: validate response quality
            if full_content and len(full_content.strip()) > 50:
                quality_result = await orch._validate_intelligent_response(
                    response=full_content,
                    query=user_message,
                    tool_calls=orch.tool_calls_used,
                    task_type=stream_ctx.unified_task_type.value,
                )
                if quality_result and not quality_result.get("is_grounded", True):
                    issues = quality_result.get("grounding_issues", [])
                    if issues:
                        logger.debug(
                            "IntelligentPipeline detected grounding issues: %s",
                            issues[:3],
                        )
                    if quality_result.get("should_retry"):
                        grounding_feedback = quality_result.get("grounding_feedback", "")
                        if grounding_feedback:
                            logger.info(
                                f"Injecting grounding feedback for retry: {len(grounding_feedback)} chars"
                            )
                            stream_ctx.pending_grounding_feedback = grounding_feedback

                if quality_result:
                    new_score = quality_result.get("quality_score", stream_ctx.last_quality_score)
                    stream_ctx.update_quality_score(new_score)

                if quality_result and quality_result.get("should_finalize"):
                    finalize_reason = quality_result.get(
                        "finalize_reason", "grounding limit exceeded"
                    )
                    logger.debug(
                        "Force finalize triggered: %s. "
                        "Stopping continuation to prevent infinite loop.",
                        finalize_reason,
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
                recovery_ctx = orch._create_recovery_context(stream_ctx)
                was_triggered, hint = orch._recovery_coordinator.check_force_action(recovery_ctx)
                if was_triggered:
                    logger.info(
                        f"UnifiedTaskTracker forcing action: {hint}, "
                        f"metrics={orch.unified_tracker.get_metrics()}"
                    )

                logger.debug(f"After streaming pass, tool_calls = {tool_calls}")

                if not tool_calls:
                    # === INTENT CLASSIFICATION (P0 SRP refactor) ===
                    if not coord._intent_classification_handler:
                        from victor.agent.streaming import (
                            create_intent_classification_handler,
                        )

                        coord._intent_classification_handler = create_intent_classification_handler(
                            orch
                        )

                    # Tracking variables now initialized in AgentOrchestrator.__init__
                    from victor.agent.streaming import create_tracking_state

                    tracking_state = create_tracking_state(orch)

                    intent_result = (
                        coord._intent_classification_handler.classify_and_determine_action(
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
                    from victor.agent.streaming import apply_tracking_state_updates

                    apply_tracking_state_updates(
                        orch, intent_result.state_updates, force_finalize_used
                    )

                    action_result = intent_result.action_result
                    action = intent_result.action

                    logger.info(
                        f"[continuation] action={action} reason={action_result.get('reason', 'unknown')} "
                        f"content_len={content_length} tool_calls={len(tool_calls) if tool_calls else 0} "
                        f"iteration={stream_ctx.total_iterations} spin={_consecutive_empty_tool_responses}"
                    )

                    # === CONTINUATION ACTION HANDLING (P0 SRP refactor) ===
                    if not coord._continuation_handler:
                        from victor.agent.streaming import create_continuation_handler

                        coord._continuation_handler = create_continuation_handler(orch)

                    action_result["action"] = action

                    continuation_result = await coord._continuation_handler.handle_action(
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
                if not coord._tool_execution_handler:
                    from victor.agent.streaming import create_tool_execution_handler

                    coord._tool_execution_handler = create_tool_execution_handler(orch)

                coord._tool_execution_handler.update_observed_files(
                    set(orch.observed_files) if orch.observed_files else set()
                )

                tool_exec_result = await coord._tool_execution_handler.execute_tools(
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


def create_streaming_chat_pipeline(
    coordinator: "ChatCoordinator",
) -> StreamingChatPipeline:
    """Factory helper for creating a streaming pipeline bound to a coordinator."""
    return StreamingChatPipeline(coordinator)
