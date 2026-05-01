# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical service-owned executor for streaming chat turns.

This module owns the live streaming loop used by the canonical chat service
path. Deprecated pipeline surfaces should reuse this executor instead of
owning a parallel implementation.
"""

from __future__ import annotations

import hashlib
import logging
import re
from types import SimpleNamespace
from typing import Any, AsyncIterator, List, Optional

from victor.agent.output_deduplicator import OutputDeduplicator
from victor.agent.services.protocols.streaming_runtime import StreamingExecutionRuntimeProtocol
from victor.core.completion_markers import FILE_DONE_MARKER, strip_active_completion_markers
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
    except Exception as exc:
        logger.debug("Failed to resolve LLM decision service: %s", exc)
    return None


class StreamingChatExecutor:
    """Canonical streaming chat executor bound to a runtime owner."""

    def __init__(
        self,
        runtime_owner: StreamingExecutionRuntimeProtocol,
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
        self._perception = perception
        self._fulfillment = fulfillment
        self._confidence_monitor = confidence_monitor
        self._progress_scores: List[float] = []
        self._last_tool_context: Optional[str] = None
        self._last_tools: Optional[Any] = None
        self._content_hashes: List[str] = []
        self._prev_full_content: str = ""
        self._repetition_count: int = 0
        self._visible_output_deduplicator = OutputDeduplicator(min_block_length=40)
        self._prev_visible_content: str = ""

    @staticmethod
    def _normalize_visible_content_key(content: str) -> str:
        """Return a stable comparison key for visible content suppression."""
        return re.sub(r"\s+", " ", content).strip().lower()

    def _prepare_visible_content(self, content: str, *, user_message: str = "") -> str:
        """Normalize model output for display and conversation history."""
        if not content or not content.strip():
            return ""

        display_content = strip_active_completion_markers(content)
        if not display_content:
            return ""

        if user_message:
            from victor.framework.task.direct_response import normalize_direct_response_output

            display_content = normalize_direct_response_output(
                user_message, display_content
            ).strip()
            if not display_content:
                return ""

        display_content = self._visible_output_deduplicator.process(display_content).strip()
        if not display_content:
            return ""

        normalized_key = self._normalize_visible_content_key(display_content)
        if not normalized_key:
            return ""
        if normalized_key == self._prev_visible_content:
            logger.info("Suppressing repeated visible output block")
            return ""

        self._prev_visible_content = normalized_key
        return display_content

    @staticmethod
    def _append_stream_event(stream_ctx: Any, field_name: str, event: dict[str, Any]) -> None:
        """Append a structured event list onto the mutable stream context."""
        events = getattr(stream_ctx, field_name, None)
        if not isinstance(events, list):
            events = []
            setattr(stream_ctx, field_name, events)
        events.append(event)

    @staticmethod
    def _normalize_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(getattr(value, "value", value)).strip()
        return text or None

    def _record_confidence_early_stop(self, stream_ctx: Any) -> None:
        """Persist an early-stop signal into the shared degradation event path."""
        task_type = self._normalize_optional_text(
            getattr(getattr(stream_ctx, "unified_task_type", None), "value", None)
            or getattr(stream_ctx, "coarse_task_type", None)
        )
        event = {
            "source": "streaming_confidence",
            "kind": "confidence_early_stop",
            "task_type": task_type,
            "iteration": getattr(stream_ctx, "total_iterations", 0),
            "pre_degraded": False,
            "post_degraded": False,
            "recovered": False,
            "adaptation_cost": 0.0,
            "degradation_reasons": ["confidence_threshold_reached"],
            "total_tokens": getattr(stream_ctx, "total_tokens", 0.0),
        }
        self._append_stream_event(stream_ctx, "degradation_events", event)

    def _record_recovery_action(self, stream_ctx: Any, recovery_action: Any) -> None:
        """Persist structured recovery metadata for stream teardown normalization."""
        action = self._normalize_optional_text(getattr(recovery_action, "action", None))
        if action is None or action == "continue":
            return

        task_type = self._normalize_optional_text(
            getattr(getattr(stream_ctx, "unified_task_type", None), "value", None)
            or getattr(stream_ctx, "coarse_task_type", None)
        )
        confidence = getattr(recovery_action, "confidence", None)
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        event = {
            "source": "streaming_recovery",
            "kind": "recovery_action",
            "action": action,
            "failure_type": self._normalize_optional_text(
                getattr(recovery_action, "failure_type", None)
            ),
            "task_type": task_type,
            "iteration": getattr(stream_ctx, "total_iterations", 0),
            "reason": self._normalize_optional_text(getattr(recovery_action, "reason", None)),
            "strategy_name": self._normalize_optional_text(
                getattr(recovery_action, "strategy_name", None)
            ),
            "confidence": confidence_value,
            "fallback_provider": self._normalize_optional_text(
                getattr(recovery_action, "fallback_provider", None)
            ),
            "fallback_model": self._normalize_optional_text(
                getattr(recovery_action, "fallback_model", None)
            ),
        }
        self._append_stream_event(stream_ctx, "recovery_events", event)

    def _reset_streaming_turn_state(self, orch: Any) -> None:
        """Reset per-turn state shared across streaming iterations."""
        self._last_tool_context = None
        self._last_tools = None

        if hasattr(orch, "tool_calls_used"):
            orch.tool_calls_used = 0

        tool_pipeline = getattr(orch, "_tool_pipeline", None)
        reset_pipeline = getattr(tool_pipeline, "reset", None)
        if callable(reset_pipeline):
            reset_pipeline()

        detector = getattr(orch, "_task_completion_detector", None)
        reset_detector = getattr(detector, "reset", None)
        if callable(reset_detector):
            reset_detector()

    @staticmethod
    def _clear_deferred_active_completion_signal(detector: Any) -> None:
        """Clear a completion marker that arrived alongside additional tool calls."""
        clear_active_signal = getattr(detector, "clear_active_signal", None)
        if callable(clear_active_signal):
            clear_active_signal()
            return

        state = getattr(detector, "_state", None)
        if state is None:
            return

        if hasattr(state, "active_signal_detected"):
            state.active_signal_detected = False

        signals = getattr(state, "completion_signals", None)
        if isinstance(signals, set):
            signals_to_keep = {
                signal for signal in signals if not str(signal).startswith("active:")
            }
            state.completion_signals = signals_to_keep

    @staticmethod
    def _serialize_conversation_message(message: Any) -> dict[str, Any] | None:
        """Normalize a conversation message into a mapping for perception services."""
        if isinstance(message, dict):
            payload = dict(message)
        elif hasattr(message, "model_dump"):
            payload = message.model_dump()
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role is None and content is None:
                return None
            payload = {"role": role, "content": content}

        if not isinstance(payload, dict):
            return None
        if payload.get("role") is None and payload.get("content") is None:
            return None
        return payload

    def _get_conversation_history(
        self,
        runtime_owner: Any,
        orch: Any,
        user_message: str,
    ) -> Optional[List[dict[str, Any]]]:
        """Return serialized conversation history excluding the current user turn."""
        messages = None
        assembled_getter = getattr(orch, "get_assembled_messages", None)
        if callable(assembled_getter):
            try:
                messages = assembled_getter(current_query=user_message)
            except Exception as exc:
                logger.debug("Falling back to raw conversation history: %s", exc)

        if messages is None:
            chat_context = getattr(runtime_owner, "_chat_context", None)
            messages = getattr(chat_context, "messages", None)
        if messages is None:
            messages = getattr(orch, "messages", None)
        if not messages:
            return None

        history = [
            payload
            for message in messages
            if (payload := self._serialize_conversation_message(message)) is not None
        ]
        if (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content") == user_message
        ):
            history = history[:-1]
        return history or None

    async def _get_tools_cached(self, orch: Any, context_msg: str, goals: Any) -> Any:
        """Select tools with per-turn caching."""
        if self._last_tool_context == context_msg and self._last_tools is not None:
            return self._last_tools

        tools = await orch._select_tools_for_turn(context_msg, goals)
        if tools is not None:
            self._last_tool_context = context_msg
            self._last_tools = tools
        return tools

    @staticmethod
    def _should_execute_prepared_team(stream_ctx: Any) -> bool:
        """Check whether streaming topology resolved a concrete team execution."""
        runtime_context_overrides = getattr(stream_ctx, "runtime_context_overrides", None)
        return bool(
            isinstance(runtime_context_overrides, dict)
            and runtime_context_overrides.get("execution_mode") == "team_execution"
            and runtime_context_overrides.get("team_name")
        )

    @staticmethod
    def _build_stream_team_context(
        user_message: str,
        stream_ctx: Any,
    ) -> dict[str, Any]:
        """Build shared context for framework team execution in streaming mode."""
        runtime_context_overrides = getattr(stream_ctx, "runtime_context_overrides", {}) or {}
        context = {
            "query": user_message,
            "task_type": getattr(stream_ctx, "coarse_task_type", None),
            "task_complexity": getattr(
                getattr(getattr(stream_ctx, "task_classification", None), "complexity", None),
                "value",
                getattr(getattr(stream_ctx, "task_classification", None), "complexity", None),
            ),
            "topology_plan": getattr(stream_ctx, "topology_plan", None),
            "topology_decision": getattr(stream_ctx, "topology_decision", None),
            "perception": getattr(stream_ctx, "perception", None),
        }
        for key in (
            "team_name",
            "team_display_name",
            "formation_hint",
            "topology_action",
            "topology_kind",
            "topology_metadata",
            "provider_hint",
            "max_workers",
            "worktree_isolation",
            "materialize_worktrees",
            "dry_run_worktrees",
            "cleanup_worktrees",
        ):
            value = runtime_context_overrides.get(key)
            if value is not None:
                context[key] = value
        return context

    async def _execute_prepared_team(
        self,
        orch: Any,
        user_message: str,
        stream_ctx: Any,
    ) -> Optional[StreamChunk]:
        """Execute a prepared framework team and convert it into a final stream chunk."""
        runtime_context_overrides = getattr(stream_ctx, "runtime_context_overrides", None)
        if not isinstance(runtime_context_overrides, dict):
            return None

        from victor.framework.team_runtime import run_configured_team

        complexity_value = getattr(
            getattr(stream_ctx, "task_classification", None), "complexity", None
        )
        if hasattr(complexity_value, "value"):
            complexity_value = complexity_value.value
        if complexity_value is None:
            complexity_value = "medium"

        team_execution = await run_configured_team(
            orch,
            goal=user_message,
            task_type=str(getattr(stream_ctx, "coarse_task_type", None) or "unknown"),
            complexity=str(complexity_value),
            preferred_team=runtime_context_overrides.get("team_name"),
            preferred_formation=runtime_context_overrides.get("formation_hint"),
            max_workers=runtime_context_overrides.get("max_workers"),
            tool_budget=runtime_context_overrides.get("tool_budget"),
            context=self._build_stream_team_context(user_message, stream_ctx),
        )
        if team_execution is None:
            return None

        resolved_team, team_result = team_execution
        final_output = (
            team_result.final_output.strip() or team_result.error or "Team execution completed."
        )
        if final_output:
            orch.add_message("assistant", final_output)
            if hasattr(stream_ctx, "accumulate_content"):
                stream_ctx.accumulate_content(final_output)
            if hasattr(stream_ctx, "update_context_message"):
                stream_ctx.update_context_message(final_output)
        if hasattr(stream_ctx, "reset_activity_timer"):
            stream_ctx.reset_activity_timer()
        if hasattr(stream_ctx, "update_quality_score"):
            stream_ctx.update_quality_score(0.92 if team_result.success else 0.2)

        stream_ctx.full_content = final_output
        stream_ctx.force_completion = True
        stream_ctx.total_iterations = max(1, getattr(stream_ctx, "total_iterations", 0))
        stream_ctx.tool_calls_used = team_result.total_tool_calls
        if hasattr(orch, "tool_calls_used"):
            orch.tool_calls_used = team_result.total_tool_calls

        metadata = getattr(stream_ctx, "topology_plan", None)
        if isinstance(metadata, dict):
            metadata["team_name"] = resolved_team.team_name
            metadata["team_display_name"] = resolved_team.display_name
            metadata["member_count"] = resolved_team.member_count

        return orch._chunk_generator.generate_content_chunk(final_output, is_final=True)

    async def run(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Run the streaming executor for the provided message."""
        runtime_owner = self._runtime_owner
        orch = runtime_owner._orchestrator
        recovery = getattr(orch, "_recovery_service", None) or orch._recovery_coordinator
        create_recovery_context = orch.create_recovery_context

        self._reset_streaming_turn_state(orch)

        stream_ctx = await runtime_owner._create_stream_context(user_message, **kwargs)
        orch._current_stream_context = stream_ctx

        orch._required_files = _extract_required_files_from_prompt(user_message)
        orch._required_outputs = _extract_required_outputs_from_prompt(user_message)
        orch._read_files_session.clear()
        orch._all_files_read_nudge_sent = False
        logger.debug(
            "Task requirements extracted - files: %s, outputs: %s",
            orch._required_files,
            orch._required_outputs,
        )

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

        max_total_iterations = stream_ctx.max_total_iterations
        max_exploration_iterations = stream_ctx.max_exploration_iterations

        orch._apply_intent_guard(user_message)

        if stream_ctx.is_analysis_task and stream_ctx.unified_task_type.value in ("edit", "create"):
            logger.info(
                "Compound task detected (analysis+%s): unified_tracker will use appropriate "
                "exploration limits",
                stream_ctx.unified_task_type.value,
            )

        logger.info(
            "Task type classification: coarse=%s, unified=%s, is_analysis=%s, is_action=%s",
            stream_ctx.coarse_task_type,
            stream_ctx.unified_task_type.value,
            stream_ctx.is_analysis_task,
            stream_ctx.is_action_task,
        )

        orch._apply_task_guidance(
            user_message,
            stream_ctx.unified_task_type,
            stream_ctx.is_analysis_task,
            stream_ctx.is_action_task,
            stream_ctx.needs_execution,
            max_exploration_iterations,
        )

        if stream_ctx.is_action_task:
            logger.info(
                "Detected action-oriented task - allowing up to %s exploration iterations",
                max_exploration_iterations,
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

        logger.info(
            "Stream chat limits: tool_budget=%s, max_total_iterations=%s, "
            "max_exploration_iterations=%s, is_analysis_task=%s, is_action_task=%s",
            orch.tool_budget,
            max_total_iterations,
            max_exploration_iterations,
            stream_ctx.is_analysis_task,
            stream_ctx.is_action_task,
        )

        orch.debug_logger.reset()

        if self._runtime_intelligence is not None:
            self._runtime_intelligence.reset_decision_budget()
        else:
            decision_service = _get_decision_service(orch)
            if decision_service is not None and hasattr(decision_service, "reset_budget"):
                decision_service.reset_budget()

        from victor.agent.turn_policy import NudgePolicy, SpinDetector, SpinState

        _spin = SpinDetector()
        _nudge_policy = NudgePolicy()

        _perception = None
        conversation_history = self._get_conversation_history(runtime_owner, orch, user_message)
        if self._runtime_intelligence is not None:
            try:
                snapshot = await self._runtime_intelligence.analyze_turn(
                    user_message,
                    context={
                        "conversation_stage": getattr(stream_ctx, "conversation_stage", "initial"),
                        "is_analysis_task": stream_ctx.is_analysis_task,
                        "is_action_task": stream_ctx.is_action_task,
                    },
                    conversation_history=conversation_history,
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
            except Exception as exc:
                logger.debug("Runtime intelligence perception skipped: %s", exc)
        elif self._perception is not None:
            try:
                _perception = await self._perception.perceive(
                    user_message,
                    context={
                        "conversation_stage": getattr(stream_ctx, "conversation_stage", "initial"),
                        "is_analysis_task": stream_ctx.is_analysis_task,
                        "is_action_task": stream_ctx.is_action_task,
                    },
                    conversation_history=conversation_history,
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
            except Exception as exc:
                logger.debug("Perception phase skipped: %s", exc)

        self._progress_scores.clear()
        _prev_iteration_had_content = False

        self._content_hashes.clear()
        self._prev_full_content = ""
        self._repetition_count = 0
        self._visible_output_deduplicator.reset()
        self._prev_visible_content = ""

        if self._should_execute_prepared_team(stream_ctx):
            team_chunk = await self._execute_prepared_team(orch, user_message, stream_ctx)
            if team_chunk is not None:
                yield team_chunk
                return

        while True:
            if _prev_iteration_had_content and stream_ctx.total_iterations > 1:
                yield StreamChunk(content="\n\n")
            _prev_iteration_had_content = False

            cancelled = False
            async for pre_chunk in runtime_owner._run_iteration_pre_checks(
                stream_ctx, user_message
            ):
                yield pre_chunk
                if pre_chunk.content == "" and getattr(pre_chunk, "is_final", False):
                    cancelled = True
            if cancelled:
                return

            unique_resources = orch.unified_tracker.unique_resources
            logger.debug(
                "Iteration %s/%s: tool_calls_used=%s/%s, unique_resources=%s, force_completion=%s",
                stream_ctx.total_iterations,
                max_total_iterations,
                orch.tool_calls_used,
                orch.tool_budget,
                len(unique_resources),
                stream_ctx.force_completion,
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

            if getattr(stream_ctx, "is_qa_task", False):
                tools = None
            elif orch.get_session_tools() is not None:
                tools = orch.get_session_tools()
            else:
                tools = await self._get_tools_cached(orch, stream_ctx.context_msg, goals)

            provider_kwargs = dict(getattr(stream_ctx, "provider_kwargs", {}) or {})
            if orch.thinking or provider_kwargs.get("execution_mode") == "escalated_single_agent":
                provider_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000,
                }

            full_content, tool_calls, _, garbage_detected = (
                await runtime_owner._stream_provider_response(
                    tools=tools,
                    provider_kwargs=provider_kwargs,
                    stream_ctx=stream_ctx,
                )
            )

            if self._confidence_monitor is not None and not tool_calls:
                try:
                    from victor.core.feature_flags import FeatureFlag, is_feature_enabled

                    if is_feature_enabled(FeatureFlag.USE_CONFIDENCE_MONITOR):
                        self._confidence_monitor.record(full_content or "", stream_ctx.total_tokens)
                        if self._confidence_monitor.should_stop():
                            self._record_confidence_early_stop(stream_ctx)
                            logger.info(
                                "[ConfidenceMonitor] Stopping iteration early — confidence threshold met"
                            )
                            break
                except Exception:
                    pass

            content_preview = full_content[:200] if full_content else "(empty)"
            logger.debug(
                "_stream_provider_response returned: content_len=%s, native_tool_calls=%s, "
                "tokens=%s, garbage=%s, content_preview=%r",
                len(full_content) if full_content else 0,
                len(tool_calls) if tool_calls else 0,
                stream_ctx.total_tokens,
                garbage_detected,
                content_preview,
            )

            if garbage_detected and not tool_calls:
                stream_ctx.force_completion = True
                logger.info("Setting force_completion due to garbage detection")

            if full_content and len(full_content.strip()) > 20:
                normalized = re.sub(r"\s+", " ", full_content.strip().lower())
                content_hash = hashlib.md5(normalized.encode()).hexdigest()
                self._content_hashes.append(content_hash)

                if len(self._content_hashes) > 5:
                    self._content_hashes.pop(0)

                if len(self._content_hashes) >= 3:
                    last_3 = self._content_hashes[-3:]
                    if len(set(last_3)) == 1:
                        self._repetition_count += 1
                        logger.warning(
                            "[content-repetition] Exact content match detected "
                            "(consecutive=%s, content_len=%s). "
                            "Forcing completion to break feedback loop.",
                            self._repetition_count,
                            len(full_content),
                        )
                        stream_ctx.force_completion = True
                        stream_ctx.skip_continuation = True
                        yield orch._chunk_generator.generate_content_chunk(
                            "\n\n[Content repetition detected — stopping to prevent "
                            "infinite output loop.]",
                            is_final=True,
                        )
                        return

                if self._prev_full_content and len(self._prev_full_content) > 50:
                    prev_norm = re.sub(r"\s+", " ", self._prev_full_content.strip().lower())
                    curr_norm = normalized
                    prev_words = set(prev_norm.split())
                    curr_words = set(curr_norm.split())
                    if prev_words and curr_words:
                        overlap = len(prev_words & curr_words) / len(prev_words | curr_words)
                        if overlap > 0.6:
                            self._repetition_count += 1
                            logger.warning(
                                "[content-repetition] High content overlap detected "
                                "(overlap=%.1f%%, consecutive=%s, content_len=%s)",
                                overlap * 100,
                                self._repetition_count,
                                len(full_content),
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
                        elif overlap > 0.4:
                            self._repetition_count = max(0, self._repetition_count - 1)
                        else:
                            self._repetition_count = 0
                else:
                    self._repetition_count = 0

                self._prev_full_content = full_content

            tool_calls, full_content = orch._parse_and_validate_tool_calls(tool_calls, full_content)

            forced_task_completion = False
            if orch._task_completion_detector and full_content:
                from victor.agent.task_completion import CompletionConfidence

                orch._task_completion_detector.analyze_response(full_content)
                confidence = orch._task_completion_detector.get_completion_confidence()

                if confidence == CompletionConfidence.HIGH:
                    if tool_calls:
                        logger.info(
                            "Task completion: HIGH confidence marker deferred because response "
                            "still contains %s tool call(s)",
                            len(tool_calls),
                        )
                        self._clear_deferred_active_completion_signal(
                            orch._task_completion_detector
                        )
                    else:
                        logger.info(
                            "Task completion: HIGH confidence detected (active signal), "
                            "forcing completion NOW (immediate stop, skip continuation)"
                        )
                        forced_task_completion = True
                        stream_ctx.force_completion = True
                        stream_ctx.skip_continuation = True
                        last_summary = getattr(
                            orch._task_completion_detector._state, "last_summary", ""
                        )
                        sanitized_summary = strip_active_completion_markers(last_summary).strip()
                        if sanitized_summary and hasattr(orch, "_conversation_controller"):
                            try:
                                orch._conversation_controller.persist_compaction_summary(
                                    sanitized_summary, []
                                )
                                orch._conversation_controller.inject_compaction_context()
                                logger.info(
                                    "VICTOR_SUMMARY persisted for next-turn context injection"
                                )
                            except Exception as exc:
                                logger.debug("Failed to persist VICTOR_SUMMARY: %s", exc)
                elif confidence == CompletionConfidence.MEDIUM:
                    logger.info(
                        "Task completion: MEDIUM confidence detected (file mods + passive signal)"
                    )

            mentioned_tools_detected: List[str] = []

            from victor.agent.continuation_strategy import ContinuationStrategy
            from victor.tools.tool_names import TOOL_ALIASES, get_all_canonical_names

            if full_content and not tool_calls and not forced_task_completion:
                all_tool_names = get_all_canonical_names() | set(TOOL_ALIASES.keys())
                mentioned_tools_detected = ContinuationStrategy.detect_mentioned_tools(
                    full_content, list(all_tool_names), TOOL_ALIASES
                )

                if mentioned_tools_detected and _spin.consecutive_no_tool_turns >= 1:
                    tool_hint = mentioned_tools_detected[0]
                    logger.info(
                        "[tool-format-assist] Model mentioned '%s' without calling it "
                        "(state=%s). Injecting format hint.",
                        tool_hint,
                        _spin.state.value,
                    )
                    orch.add_message(
                        "user",
                        f"[TOOL-FORMAT-HINT: You described wanting to use '{tool_hint}' but didn't call it. "
                        f"Call the tool directly — don't describe what you want to do, execute it. "
                        f"If you've already modified the file successfully, say {FILE_DONE_MARKER}]",
                    )

            if forced_task_completion and not tool_calls:
                recovery_action = SimpleNamespace(action="continue")
            else:
                recovery_action = await orch._handle_recovery_with_integration(
                    stream_ctx=stream_ctx,
                    full_content=full_content,
                    tool_calls=tool_calls,
                    mentioned_tools=mentioned_tools_detected or None,
                )

            if recovery_action.action != "continue":
                self._record_recovery_action(stream_ctx, recovery_action)
                recovery_chunk = orch._apply_recovery_action(recovery_action, stream_ctx)
                if recovery_chunk:
                    yield recovery_chunk
                    if recovery_chunk.is_final:
                        orch._recovery_integration.record_outcome(success=False)
                        return
                if recovery_action.action in ("retry", "force_summary"):
                    continue

            if full_content:
                visible_content = self._prepare_visible_content(
                    full_content,
                    user_message=getattr(stream_ctx, "user_message", ""),
                )
                if visible_content:
                    _prev_iteration_had_content = True
                sanitized = orch.sanitizer.sanitize(visible_content)
                if sanitized:
                    orch.add_message("assistant", sanitized, tool_calls=tool_calls)
                    if tool_calls or (forced_task_completion and not tool_calls):
                        yield orch._chunk_generator.generate_content_chunk(
                            sanitized,
                            is_final=forced_task_completion and not tool_calls,
                        )
                        if forced_task_completion and not tool_calls:
                            return
                else:
                    plain_text = orch.sanitizer.strip_markup(visible_content)
                    if plain_text:
                        orch.add_message("assistant", plain_text, tool_calls=tool_calls)
                        if tool_calls or (forced_task_completion and not tool_calls):
                            yield orch._chunk_generator.generate_content_chunk(
                                plain_text,
                                is_final=forced_task_completion and not tool_calls,
                            )
                            if forced_task_completion and not tool_calls:
                                return
            elif tool_calls:
                orch.add_message("assistant", "", tool_calls=tool_calls)
            else:
                recovery_ctx = create_recovery_context(stream_ctx)
                final_chunk = recovery.check_natural_completion(
                    recovery_ctx, has_tool_calls=False, content_length=0
                )
                if final_chunk:
                    yield final_chunk
                    return

                logger.warning("Model returned empty response - attempting aggressive recovery")

                recovery_ctx = create_recovery_context(stream_ctx)
                recovery_chunk, should_force = recovery.handle_empty_response(recovery_ctx)
                _ = should_force
                if recovery_chunk:
                    yield recovery_chunk
                    continue

                recovery_success, recovered_tool_calls, final_chunk = (
                    await runtime_owner._handle_empty_response_recovery(stream_ctx, tools)
                )

                if recovery_success:
                    if final_chunk:
                        yield final_chunk
                        return
                    if recovered_tool_calls:
                        tool_calls = recovered_tool_calls
                        logger.info(
                            "Recovery produced %s tool call(s) - continuing main loop",
                            len(tool_calls),
                        )
                else:
                    recovery_ctx = create_recovery_context(stream_ctx)
                    fallback_msg = recovery.get_recovery_fallback_message(recovery_ctx)
                    orch._record_runtime_intelligence_outcome(
                        success=False,
                        quality_score=0.3,
                        user_satisfied=False,
                        completed=False,
                    )
                    yield orch._chunk_generator.generate_content_chunk(fallback_msg, is_final=True)
                    return

            for tc in tool_calls or []:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                orch.unified_tracker.record_tool_call(tool_name, tool_args)

            content_length = len(full_content.strip())
            _has_tools = bool(tool_calls)
            _no_progress = not _has_tools and content_length < 120

            tool_names_set = {tc.get("name", "") for tc in tool_calls} if tool_calls else set()
            _spin.record_turn(
                has_tool_calls=_has_tools,
                tool_names=tool_names_set,
                tool_count=len(tool_calls) if tool_calls else 0,
            )

            logger.info(
                "[spin-check] tool_calls=%s content_len=%s no_progress=%s state=%s",
                len(tool_calls) if tool_calls else 0,
                content_length,
                _no_progress,
                _spin.state.value,
            )

            if _no_progress:
                _intent = getattr(orch, "_current_intent", None)
                nudge = _nudge_policy.evaluate(_spin, intent=_intent)
                if nudge.should_inject:
                    orch.add_message(nudge.role, nudge.message)
                    logger.info("[nudge] %s", nudge.nudge_type.value)

                if _spin.state == SpinState.TERMINATED:
                    logger.warning(
                        "[spin-detect] Terminated after no_tool=%s blocked=%s. Last response: %r",
                        _spin.consecutive_no_tool_turns,
                        _spin.consecutive_all_blocked,
                        full_content[:100],
                    )
                    yield orch._chunk_generator.generate_content_chunk(
                        "\n\n[Agent detected a response loop — breaking to prevent wasted time.]",
                        is_final=True,
                    )
                    return

            orch.unified_tracker.record_iteration(content_length)

            if (
                full_content
                and len(full_content.strip()) > 50
                and not getattr(stream_ctx, "is_qa_task", False)
            ):
                quality_result = await orch._validate_runtime_intelligence_response(
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
                                "Injecting grounding feedback for retry: %s chars",
                                len(grounding_feedback),
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
                        "Force finalize triggered: %s. Stopping continuation to prevent "
                        "infinite loop.",
                        finalize_reason,
                    )
                    orch._force_finalize = True

            unified_loop_warning = orch.unified_tracker.check_loop_warning()
            loop_warning_chunk = orch._streaming_handler.handle_loop_warning(
                stream_ctx, unified_loop_warning
            )
            if loop_warning_chunk:
                logger.warning("UnifiedTaskTracker loop warning: %s", unified_loop_warning)
                yield loop_warning_chunk
            else:
                recovery_ctx = create_recovery_context(stream_ctx)
                was_triggered, hint = recovery.check_force_action(recovery_ctx)
                if was_triggered:
                    logger.info(
                        "UnifiedTaskTracker forcing action: %s, metrics=%s",
                        hint,
                        orch.unified_tracker.get_metrics(),
                    )

                logger.debug("After streaming pass, tool_calls = %s", tool_calls)

                if not tool_calls:
                    if not runtime_owner._intent_classification_handler:
                        from victor.agent.streaming import create_intent_classification_handler

                        runtime_owner._intent_classification_handler = (
                            create_intent_classification_handler(orch)
                        )

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
                        "[continuation] action=%s reason=%s content_len=%s tool_calls=%s "
                        "iteration=%s spin_state=%s",
                        action,
                        action_result.get("reason", "unknown"),
                        content_length,
                        len(tool_calls) if tool_calls else 0,
                        stream_ctx.total_iterations,
                        _spin.state.value,
                    )

                    if (
                        stream_ctx.compaction_occurred
                        and not stream_ctx.force_completion
                        and not forced_task_completion
                    ):
                        turns_since_compaction = (
                            stream_ctx.total_iterations - stream_ctx.last_compaction_turn
                        )
                        if turns_since_compaction <= 2:
                            is_asking_input = action_result.get("is_asking_input", False)
                            is_completion = action_result.get("is_completion", False)

                            if not is_asking_input and not is_completion:
                                logger.info(
                                    "[post-compaction-continuation] Forcing continuation after "
                                    "compaction (turn %s, %s messages removed)",
                                    stream_ctx.total_iterations,
                                    stream_ctx.compaction_message_removed_count,
                                )
                                summary_hint = ""
                                if stream_ctx.compaction_summary:
                                    summary_hint = (
                                        f" Previous work: {stream_ctx.compaction_summary[:200]}"
                                    )

                                orch.add_message(
                                    "user",
                                    f"[CONTEXT COMPACTED: Context was compacted to continue the session. "
                                    f"You were in the middle of a task.{summary_hint} "
                                    f"Please continue with what you were doing - "
                                    f"use tools to gather more information or complete your analysis.]",
                                )
                                if turns_since_compaction == 2:
                                    stream_ctx.compaction_occurred = False

                    if not runtime_owner._continuation_handler:
                        from victor.agent.streaming import create_continuation_handler

                        runtime_owner._continuation_handler = create_continuation_handler(orch)

                    action_result = action_result.with_action(action)

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

                if self._fulfillment and _perception:
                    try:
                        from victor.agent.turn_policy import FulfillmentCriteriaBuilder
                        from victor.framework.fulfillment import TaskType

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
                            return
                    except Exception as exc:
                        logger.debug("Streaming fulfillment check skipped: %s", exc)

                tool_count = tool_exec_result.tool_calls_executed if tool_exec_result else 0
                content_len = len(full_content) if full_content else 0
                progress = min(1.0, (tool_count * 0.3 + min(content_len / 2000, 0.7)))
                self._progress_scores.append(progress)

                if len(self._progress_scores) >= 3:
                    recent = self._progress_scores[-3:]
                    if max(recent) - min(recent) < 0.05 and recent[-1] < 0.8:
                        logger.info(
                            "Streaming progress plateau detected (scores=%s), injecting nudge",
                            [f"{score:.2f}" for score in recent],
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
                                'and apply the change now with edit(ops=[{"type": "replace", '
                                '"path": "file", "old_str": "exact text", '
                                '"new_str": "replacement"}]).'
                            )
                        else:
                            _plateau_msg = (
                                "Progress seems stalled. Try a different approach or "
                                "summarize what you've found so far."
                            )
                        orch.add_message("system", _plateau_msg)


def create_streaming_chat_executor(
    runtime_owner: StreamingExecutionRuntimeProtocol,
    runtime_intelligence: Optional[Any] = None,
    perception: Optional[Any] = None,
    fulfillment: Optional[Any] = None,
) -> StreamingChatExecutor:
    """Factory helper for creating the canonical service-owned executor."""
    return StreamingChatExecutor(
        runtime_owner,
        runtime_intelligence=runtime_intelligence,
        perception=perception,
        fulfillment=fulfillment,
    )
