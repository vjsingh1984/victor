"""Streaming chat pipeline implementation."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, AsyncIterator, List, Optional

from victor.agent.services.protocols.streaming_runtime import StreamingPipelineRuntimeProtocol
from victor.core.completion_markers import FILE_DONE_MARKER
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy
from victor.providers.base import StreamChunk

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
    """Canonical streaming pipeline wired to a runtime owner.

    The canonical owner is now the service/runtime path. Deprecated chat
    coordinators may still satisfy this structural contract for backward
    compatibility, but they are no longer the primary execution owner.
    """

    def __init__(
        self,
        runtime_owner: StreamingPipelineRuntimeProtocol,
        runtime_intelligence: Optional[Any] = None,
        perception: Optional[Any] = None,
        fulfillment: Optional[Any] = None,
        confidence_monitor: Optional[Any] = None,
    ) -> None:
        self._runtime_owner = runtime_owner
        self._runtime_intelligence = runtime_intelligence
        resolved_policy = getattr(runtime_intelligence, "evaluation_policy", None)
        if not isinstance(resolved_policy, RuntimeEvaluationPolicy):
            resolved_policy = RuntimeEvaluationPolicy()
        self._evaluation_policy = resolved_policy
        # AgenticLoop component integration (streaming parity)
        self._perception = perception  # PerceptionIntegration instance
        self._fulfillment = fulfillment  # FulfillmentDetector instance
        self._confidence_monitor = confidence_monitor  # StreamingConfidenceMonitor instance
        self._progress_scores: List[float] = []  # For adaptive iteration
        # Tool selection cache — avoids redundant selection within same turn
        self._last_tool_context: Optional[str] = None
        self._last_tools: Optional[Any] = None

        # Content repetition tracking (Fix 1 & 4: prevents infinite content loops)
        self._content_hashes: List[str] = []  # Rolling window of last N content hashes
        self._prev_full_content: str = ""  # Previous iteration's full_content for overlap check
        self._repetition_count: int = 0  # Consecutive iterations with highly similar content

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

    async def run(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Run the streaming pipeline for the provided message."""
        runtime_owner = self._runtime_owner
        orch = runtime_owner._orchestrator
        recovery = getattr(orch, "_recovery_service", None) or orch._recovery_coordinator

        # Initialize and prepare using StreamingChatContext
        stream_ctx = await runtime_owner._create_stream_context(user_message, **kwargs)

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
                    "user",
                    "[ACTION-GUIDANCE: This is an action-oriented task requiring execution. "
                    "Follow this workflow: "
                    "1. CREATE the file/script with write or edit "
                    "2. EXECUTE it immediately with shell (don't skip this step!) "
                    "3. SHOW the output to the user. "
                    "Minimize exploration and proceed directly to create→execute→show results.]",
                )
            else:
                orch.add_message(
                    "user",
                    "[ACTION-GUIDANCE: This is an action-oriented task (create/write/build). "
                    "Minimize exploration and proceed directly to creating what was requested. "
                    "Only explore if absolutely necessary to complete the task.]",
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
        if self._runtime_intelligence is not None:
            self._runtime_intelligence.reset_decision_budget()
        else:
            decision_service = _get_decision_service(orch)
            if decision_service is not None and hasattr(decision_service, "reset_budget"):
                decision_service.reset_budget()

        # Spin detection via shared turn_policy (consistent with batch/AgenticLoop path)
        from victor.agent.turn_policy import SpinDetector, NudgePolicy, SpinState

        _spin = SpinDetector()
        _nudge_policy = NudgePolicy()

        # === PRE-LOOP PERCEPTION (AgenticLoop parity) ===
        # Structured task understanding before iteration begins. Reuses
        # PerceptionIntegration from the AgenticLoop's PERCEIVE phase.
        _perception = None
        if self._runtime_intelligence is not None:
            try:
                snapshot = await self._runtime_intelligence.analyze_turn(
                    user_message,
                    context={
                        "conversation_stage": getattr(stream_ctx, "conversation_stage", "initial"),
                        "is_analysis_task": stream_ctx.is_analysis_task,
                        "is_action_task": stream_ctx.is_action_task,
                    },
                )
                _perception = snapshot.perception
                if _perception:
                    stream_ctx.perception = _perception
                    logger.info(
                        "Streaming perception: intent=%s, complexity=%s, confidence=%.2f",
                        getattr(_perception, "intent", "unknown"),
                        getattr(_perception, "complexity", "unknown"),
                        getattr(_perception, "confidence", 0.0),
                    )
                    clarification = self._evaluation_policy.get_clarification_decision(_perception)
                    if clarification.requires_clarification:
                        yield orch._chunk_generator.generate_content_chunk(
                            clarification.prompt
                            or self._evaluation_policy.default_clarification_prompt,
                            is_final=True,
                        )
                        return
            except Exception as e:
                logger.debug("Runtime intelligence perception skipped: %s", e)
        elif self._perception is not None:
            try:
                _perception = await self._perception.perceive(
                    user_message,
                    context={
                        "conversation_stage": getattr(stream_ctx, "conversation_stage", "initial"),
                        "is_analysis_task": stream_ctx.is_analysis_task,
                        "is_action_task": stream_ctx.is_action_task,
                    },
                )
                if _perception:
                    stream_ctx.perception = _perception
                    logger.info(
                        "Streaming perception: intent=%s, complexity=%s, confidence=%.2f",
                        getattr(_perception, "intent", "unknown"),
                        getattr(_perception, "complexity", "unknown"),
                        getattr(_perception, "confidence", 0.0),
                    )
                    clarification = self._evaluation_policy.get_clarification_decision(_perception)
                    if clarification.requires_clarification:
                        yield orch._chunk_generator.generate_content_chunk(
                            clarification.prompt
                            or self._evaluation_policy.default_clarification_prompt,
                            is_final=True,
                        )
                        return
            except Exception as e:
                logger.debug("Perception phase skipped: %s", e)

        # Reset progress tracking for adaptive iteration
        self._progress_scores.clear()
        _prev_iteration_had_content = False

        # Reset content repetition tracking for this turn
        self._content_hashes.clear()
        self._prev_full_content = ""
        self._repetition_count = 0

        while True:
            # Yield separator between iterations when content was emitted
            if _prev_iteration_had_content and stream_ctx.total_iterations > 1:
                yield StreamChunk(content="\n\n")
            _prev_iteration_had_content = False

            # === PRE-ITERATION CHECKS (via runtime helper) ===
            cancelled = False
            async for pre_chunk in runtime_owner._run_iteration_pre_checks(
                stream_ctx,
                user_message,
            ):
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

            (
                full_content,
                tool_calls,
                _,
                garbage_detected,
            ) = await runtime_owner._stream_provider_response(
                tools=tools,
                provider_kwargs=provider_kwargs,
                stream_ctx=stream_ctx,
            )

            # Confidence monitor: check if generation should stop early (ATCC, arXiv 2603.13906)
            if self._confidence_monitor is not None and not tool_calls:
                try:
                    from victor.core.feature_flags import FeatureFlag, is_feature_enabled

                    if is_feature_enabled(FeatureFlag.USE_CONFIDENCE_MONITOR):
                        self._confidence_monitor.record(full_content or "", stream_ctx.total_tokens)
                        if self._confidence_monitor.should_stop():
                            logger.info(
                                "[ConfidenceMonitor] Stopping iteration early — confidence threshold met"
                            )
                            break
                except Exception:
                    pass

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

            # === CONTENT REPETITION DETECTION (Fix 1 & 4) ===
            # Detect when the LLM regenerates the same or highly similar content
            # across consecutive iterations — the root cause of infinite buffering.
            if full_content and len(full_content.strip()) > 20:
                # Compute a hash of the normalized content for exact-match detection
                normalized = re.sub(r"\s+", " ", full_content.strip().lower())
                content_hash = hashlib.md5(normalized.encode()).hexdigest()
                self._content_hashes.append(content_hash)

                # Keep only the last 5 hashes for sliding-window comparison
                if len(self._content_hashes) > 5:
                    self._content_hashes.pop(0)

                # Check 1: Exact hash match — same content repeated
                if len(self._content_hashes) >= 3:
                    last_3 = self._content_hashes[-3:]
                    if len(set(last_3)) == 1:
                        self._repetition_count += 1
                        logger.warning(
                            f"[content-repetition] Exact content match detected "
                            f"(consecutive={self._repetition_count}, "
                            f"content_len={len(full_content)}). "
                            f"Forcing completion to break feedback loop."
                        )
                        stream_ctx.force_completion = True
                        stream_ctx.skip_continuation = True
                        yield orch._chunk_generator.generate_content_chunk(
                            "\n\n[Content repetition detected — stopping to prevent "
                            "infinite output loop.]",
                            is_final=True,
                        )
                        return

                # Check 2: Text overlap — measure how much of previous content
                # appears in current content (catches partial regeneration)
                if self._prev_full_content and len(self._prev_full_content) > 50:
                    prev_norm = re.sub(r"\s+", " ", self._prev_full_content.strip().lower())
                    curr_norm = normalized
                    # Use word-level Jaccard similarity for overlap detection
                    prev_words = set(prev_norm.split())
                    curr_words = set(curr_norm.split())
                    if prev_words and curr_words:
                        overlap = len(prev_words & curr_words) / len(prev_words | curr_words)
                        if overlap > 0.6:  # >60% word overlap
                            self._repetition_count += 1
                            logger.warning(
                                f"[content-repetition] High content overlap detected "
                                f"(overlap={overlap:.1%}, consecutive={self._repetition_count}, "
                                f"content_len={len(full_content)})"
                            )
                            if self._repetition_count >= 3:
                                logger.warning(
                                    "[content-repetition] 3+ consecutive high-overlap iterations — "
                                    "forcing completion."
                                )
                                stream_ctx.force_completion = True
                                stream_ctx.skip_continuation = True
                                yield orch._chunk_generator.generate_content_chunk(
                                    "\n\n[Content repetition detected — stopping to prevent "
                                    "infinite output loop.]",
                                    is_final=True,
                                )
                                return
                        else:
                            # Content has diverged — reset counter
                            self._repetition_count = 0
                else:
                    self._repetition_count = 0

                self._prev_full_content = full_content

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
                        "forcing completion NOW (immediate stop, skip continuation)"
                    )
                    stream_ctx.force_completion = True
                    stream_ctx.skip_continuation = (
                        True  # NEW: Prevent continuation strategy override
                    )
                    # Persist VICTOR_SUMMARY text as a compaction summary so it
                    # survives context compaction and is injected into the next turn.
                    last_summary = getattr(
                        orch._task_completion_detector._state, "last_summary", ""
                    )
                    if last_summary and hasattr(orch, "_conversation_controller"):
                        try:
                            orch._conversation_controller.persist_compaction_summary(
                                last_summary, []
                            )
                            orch._conversation_controller.inject_compaction_context()
                            logger.info("VICTOR_SUMMARY persisted for next-turn context injection")
                        except Exception as _e:
                            logger.debug(f"Failed to persist VICTOR_SUMMARY: {_e}")
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
                if mentioned_tools_detected and _spin.consecutive_no_tool_turns >= 1:
                    tool_hint = mentioned_tools_detected[0]
                    logger.info(
                        f"[tool-format-assist] Model mentioned '{tool_hint}' without calling it "
                        f"(state={_spin.state.value}). Injecting format hint."
                    )
                    orch.add_message(
                        "user",
                        f"[TOOL-FORMAT-HINT: You described wanting to use '{tool_hint}' but didn't call it. "
                        f"Call the tool directly — don't describe what you want to do, execute it. "
                        f"If you've already modified the file successfully, say {FILE_DONE_MARKER}]",
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

            # Spin detection for ALL paths (via shared SpinDetector)
            _pipeline_obj = getattr(orch, "_tool_pipeline", None)
            if _pipeline_obj and getattr(_pipeline_obj, "last_batch_all_skipped", False):
                _spin.record_turn(has_tool_calls=True, all_blocked=True)
                logger.info(f"[spin-check] all_blocked=True state={_spin.state.value}")
                _intent = getattr(orch, "_current_intent", None)
                nudge = _nudge_policy.evaluate(_spin, intent=_intent)
                if nudge.should_inject:
                    orch.add_message(nudge.role, nudge.message)
                if _spin.state == SpinState.TERMINATED:
                    logger.warning(
                        f"[spin-detect] {_spin.consecutive_all_blocked} consecutive "
                        f"blocked tool calls. Breaking loop."
                    )
                    yield orch._chunk_generator.generate_content_chunk(
                        "\n\n[Agent detected a tool-call loop — all attempts were "
                        "blocked by deduplication. Try a different approach.]",
                        is_final=True,
                    )
                    return

            if full_content:
                _prev_iteration_had_content = True
                # Sanitize response to remove malformed patterns from local models
                sanitized = orch.sanitizer.sanitize(full_content)
                if sanitized:
                    orch.add_message("assistant", sanitized, tool_calls=tool_calls)
                    # Only yield here when tool_calls are present — the intent
                    # classification handler (Step 1) is skipped for tool-call
                    # turns and yields nothing, so we must display the content.
                    # For non-tool turns, intent classification yields it at
                    # classify_and_determine_action() to avoid double display.
                    if tool_calls:
                        yield orch._chunk_generator.generate_content_chunk(sanitized)
                else:
                    plain_text = orch.sanitizer.strip_markup(full_content)
                    if plain_text:
                        orch.add_message("assistant", plain_text, tool_calls=tool_calls)
                        if tool_calls:
                            yield orch._chunk_generator.generate_content_chunk(plain_text)
            elif tool_calls:
                # OpenAI spec: assistant message with tool_calls must be in
                # conversation even when content is empty. Without this,
                # tool responses have no matching assistant message and
                # the model can't see its own tool_calls/results.
                orch.add_message("assistant", "", tool_calls=tool_calls)
            else:
                # No content and no tool calls - check for natural completion
                recovery_ctx = orch._create_recovery_context(stream_ctx)
                final_chunk = recovery.check_natural_completion(
                    recovery_ctx, has_tool_calls=False, content_length=0
                )
                if final_chunk:
                    yield final_chunk
                    return

                # No substantial content yet - attempt aggressive recovery
                logger.warning("Model returned empty response - attempting aggressive recovery")

                recovery_ctx = orch._create_recovery_context(stream_ctx)
                recovery_chunk, should_force = recovery.handle_empty_response(recovery_ctx)
                if recovery_chunk:
                    yield recovery_chunk
                    continue

                # Delegate empty response recovery to helper method
                recovery_success, recovered_tool_calls, final_chunk = (
                    await runtime_owner._handle_empty_response_recovery(stream_ctx, tools)
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
                    fallback_msg = recovery.get_recovery_fallback_message(recovery_ctx)
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

            # Spin detection via shared SpinDetector (consistent with batch path)
            _all_tools_blocked = False
            _pipeline_obj = getattr(orch, "_tool_pipeline", None)
            if _pipeline_obj and getattr(_pipeline_obj, "last_batch_all_skipped", False):
                _all_tools_blocked = True

            _has_tools = bool(tool_calls)
            _no_progress = (not _has_tools and content_length < 120) or _all_tools_blocked

            # Record turn in shared detector
            tool_names_set = {tc.get("name", "") for tc in tool_calls} if tool_calls else set()
            _spin.record_turn(
                has_tool_calls=_has_tools,
                all_blocked=_all_tools_blocked,
                tool_names=tool_names_set,
                tool_count=len(tool_calls) if tool_calls else 0,
            )

            logger.info(
                f"[spin-check] tool_calls={len(tool_calls) if tool_calls else 0} "
                f"content_len={content_length} all_blocked={_all_tools_blocked} "
                f"no_progress={_no_progress} state={_spin.state.value}"
            )

            if _no_progress:
                # Inject nudge via shared NudgePolicy
                _intent = getattr(orch, "_current_intent", None)
                nudge = _nudge_policy.evaluate(_spin, intent=_intent)
                if nudge.should_inject:
                    orch.add_message(nudge.role, nudge.message)
                    logger.info(f"[nudge] {nudge.nudge_type.value}")

                if _spin.state == SpinState.TERMINATED:
                    logger.warning(
                        f"[spin-detect] Terminated after "
                        f"no_tool={_spin.consecutive_no_tool_turns} "
                        f"blocked={_spin.consecutive_all_blocked}. "
                        f"Last response: {full_content[:100]!r}"
                    )
                    yield orch._chunk_generator.generate_content_chunk(
                        "\n\n[Agent detected a response loop — breaking to "
                        "prevent wasted time.]",
                        is_final=True,
                    )
                    return

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
                was_triggered, hint = recovery.check_force_action(recovery_ctx)
                if was_triggered:
                    logger.info(
                        f"UnifiedTaskTracker forcing action: {hint}, "
                        f"metrics={orch.unified_tracker.get_metrics()}"
                    )

                logger.debug(f"After streaming pass, tool_calls = {tool_calls}")

                if not tool_calls:
                    # === INTENT CLASSIFICATION (P0 SRP refactor) ===
                    if not runtime_owner._intent_classification_handler:
                        from victor.agent.streaming import (
                            create_intent_classification_handler,
                        )

                        runtime_owner._intent_classification_handler = (
                            create_intent_classification_handler(orch)
                        )

                    # Tracking variables now initialized in AgentOrchestrator.__init__
                    from victor.agent.streaming import create_tracking_state

                    tracking_state = create_tracking_state(orch)

                    intent_result = (
                        runtime_owner._intent_classification_handler.classify_and_determine_action(
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
                        f"iteration={stream_ctx.total_iterations} spin_state={_spin.state.value}"
                    )

                    # === CONTINUATION ACTION HANDLING (P0 SRP refactor) ===
                    if not runtime_owner._continuation_handler:
                        from victor.agent.streaming import create_continuation_handler

                        runtime_owner._continuation_handler = create_continuation_handler(orch)

                    action_result["action"] = action

                    continuation_result = await runtime_owner._continuation_handler.handle_action(
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
                if not runtime_owner._tool_execution_handler:
                    from victor.agent.streaming import create_tool_execution_handler

                    runtime_owner._tool_execution_handler = create_tool_execution_handler(orch)

                runtime_owner._tool_execution_handler.update_observed_files(
                    set(orch.observed_files) if orch.observed_files else set()
                )

                tool_exec_result = await runtime_owner._tool_execution_handler.execute_tools(
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

                # === FULFILLMENT CHECK (AgenticLoop parity) ===
                # After tool execution, check if the task is fulfilled using
                # the same FulfillmentDetector as the non-streaming path.
                if self._fulfillment and _perception:
                    try:
                        from victor.framework.fulfillment import TaskType
                        from victor.agent.turn_policy import FulfillmentCriteriaBuilder

                        task_analysis = getattr(_perception, "task_analysis", None)
                        task_type_str = (
                            getattr(task_analysis, "task_type", "unknown")
                            if task_analysis
                            else "unknown"
                        )
                        try:
                            task_type = TaskType(task_type_str)
                        except (ValueError, KeyError):
                            task_type = TaskType.UNKNOWN

                        criteria = FulfillmentCriteriaBuilder.from_tool_results(
                            tool_exec_result.tool_results
                            if hasattr(tool_exec_result, "tool_results")
                            else []
                        )
                        fulfillment_result = await self._fulfillment.check_fulfillment(
                            task_type=task_type,
                            criteria=criteria,
                            context={"full_content": full_content, "user_message": user_message},
                        )
                        if fulfillment_result.is_fulfilled:
                            logger.info(
                                "Streaming fulfillment: task fulfilled (score=%.2f, reason=%s)",
                                fulfillment_result.score,
                                fulfillment_result.reason,
                            )
                            return  # Exit loop — task complete
                    except Exception as e:
                        logger.debug("Streaming fulfillment check skipped: %s", e)

                # === PROGRESS TRACKING (AgenticLoop parity) ===
                # Track progress for adaptive iteration (plateau detection).
                tool_count = tool_exec_result.tool_calls_executed if tool_exec_result else 0
                content_len = len(full_content) if full_content else 0
                progress = min(1.0, (tool_count * 0.3 + min(content_len / 2000, 0.7)))
                self._progress_scores.append(progress)

                # Plateau detection: 3+ iterations with < 5% progress change
                if len(self._progress_scores) >= 3:
                    recent = self._progress_scores[-3:]
                    if max(recent) - min(recent) < 0.05 and recent[-1] < 0.8:
                        logger.info(
                            "Streaming progress plateau detected (scores=%s), injecting nudge",
                            [f"{s:.2f}" for s in recent],
                        )
                        _plateau_intent = getattr(orch, "_current_intent", None)
                        _is_write = (
                            _plateau_intent is not None
                            and hasattr(_plateau_intent, "value")
                            and _plateau_intent.value == "write_allowed"
                        )
                        if _is_write:
                            _plateau_msg = (
                                "Progress stalled. You have enough context — stop reading "
                                "and apply the change now with edit(ops=[{\"type\": \"replace\", "
                                "\"path\": \"file\", \"old_str\": \"exact text\", "
                                "\"new_str\": \"replacement\"}])."
                            )
                        else:
                            _plateau_msg = (
                                "Progress seems stalled. Try a different approach or "
                                "summarize what you've found so far."
                            )
                        orch.add_message("system", _plateau_msg)


def create_streaming_chat_pipeline(
    runtime_owner: StreamingPipelineRuntimeProtocol,
    runtime_intelligence: Optional[Any] = None,
    perception: Optional[Any] = None,
    fulfillment: Optional[Any] = None,
) -> StreamingChatPipeline:
    """Factory helper for creating a streaming pipeline bound to a runtime owner."""
    return StreamingChatPipeline(
        runtime_owner,
        runtime_intelligence=runtime_intelligence,
        perception=perception,
        fulfillment=fulfillment,
    )
