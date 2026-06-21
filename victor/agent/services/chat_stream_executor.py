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
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Tuple

from victor.agent.output_deduplicator import OutputDeduplicator
from victor.agent.services.protocols.streaming_runtime import (
    StreamingExecutionRuntimeProtocol,
)
from victor.core.completion_markers import (
    FILE_DONE_MARKER,
    strip_active_completion_markers,
)
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy
from victor.providers.base import StreamChunk

logger = logging.getLogger(__name__)


def _tool_call_signatures(tool_calls: Optional[List[dict]]) -> set:
    """Stable per-turn tool-call signatures for spin detection.

    Mirrors ``TurnToolExecutionResult.tool_signatures`` (name + sorted, stringified args)
    so the streaming path can feed the SpinDetector the same signature scheme the
    single-turn primitive uses. Repeated identical signatures across turns (e.g. the same
    ``code_search(query=...)`` issued every iteration) are how a tool-call loop is detected
    early, instead of waiting for assistant content to become near-identical.
    """
    signatures: set = set()
    for tc in tool_calls or []:
        name = tc.get("name", "unknown")
        args = tc.get("arguments", {})
        if isinstance(args, dict):
            args_str = str([(k, str(v)) for k, v in sorted(args.items())])
        else:
            args_str = str(args)
        signatures.add(f"{name}:{args_str}")
    return signatures


def _count_productive_tools(tool_exec_result: Any) -> int:
    """Count tool calls that succeeded with output for progress/plateau accounting.

    The streaming ``ToolExecutionResult`` (victor/agent/streaming/tool_execution.py) exposes
    ``tool_results`` (a list of dicts) and ``tool_calls_executed`` — it does NOT have the
    ``successful_tool_count`` property of the single-turn ``TurnToolExecutionResult``. Derive
    the count from ``tool_results`` so a turn whose tools all failed/were blocked/returned
    nothing does not count as progress (and still gets nudged). When no per-result detail is
    available, fall back to whether any tool executed at all.
    """
    if not tool_exec_result:
        return 0
    results = getattr(tool_exec_result, "tool_results", None)
    if isinstance(results, list) and results:
        return sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    return int(getattr(tool_exec_result, "tool_calls_executed", 0) or 0)


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


@dataclass
class _StreamTurnGuards:
    """Per-turn guard components built once per streaming run.

    Bundling these makes the per-turn evaluation boundary explicit — the first step of
    FEP-0007 Phase 2 toward a ``stream_turn()`` primitive that both loops can share. The
    streaming loop reads/feeds these each iteration (content-repetition via ``turn_eval``,
    plateau via ``plateau``, search-novelty via ``novelty``, spin via ``spin``).
    """

    spin: Any
    nudge_policy: Any
    turn_eval: Any
    plateau: Any
    novelty: Any
    novelty_enabled: bool


@dataclass
class _ToolTurnOutcome:
    """Single-slot mutable holder for the tool-execution ACT sub-step's result.

    An async generator can't ``return`` a value, so ``_execute_tools_turn`` yields tool
    chunks and writes its ``ToolExecutionResult`` here for ``run()`` to read — both for
    loop control (``should_return``) and downstream evaluation (fulfillment / plateau /
    search-novelty). Part of FEP-0007 Phase 2's move toward a shared ``stream_turn()``.
    """

    result: Any = None


@dataclass
class _ProviderTurnEval:
    """Single-slot mutable holder for the post-provider early-stop evaluation.

    ``_evaluate_provider_turn_stops`` records per-turn spin signals and runs the early-stop
    checks (TERMINATED spin, content-repetition, skip_continuation). Because it's an async
    generator (it yields the final stop chunk), it writes its decision here for ``run()`` to
    read: ``should_return`` drives loop exit, while ``content_length`` / ``has_tools`` are the
    derived turn metrics the rest of the loop body reuses. Part of FEP-0007 Phase 2's move
    toward a shared ``stream_turn()``.
    """

    should_return: bool = False
    content_length: int = 0
    has_tools: bool = False


@dataclass
class _PostToolEval:
    """Single-slot mutable holder for the post-tool-execution evaluation band.

    ``_evaluate_post_tool_turn`` runs fulfillment / plateau / search-novelty after tool
    execution; it's an async generator (the search-novelty safety-net yields a final chunk),
    so it records its loop-exit decision here for ``run()`` to read via ``should_return``.
    Part of FEP-0007 Phase 2's move toward a shared ``stream_turn()``.
    """

    should_return: bool = False


class StreamingChatExecutor:
    """Canonical streaming chat executor bound to a runtime owner."""

    @staticmethod
    def _create_stream_turn_guards(orch: Any) -> "_StreamTurnGuards":
        """Build the per-turn guard components for one streaming run from settings.

        Mirrors the headless loop's controller setup: content-repetition + spin go through the
        shared ``TurnEvaluationController`` (plateau/budget/novelty disabled on it because this
        loop checks plateau/novelty at the post-tool point); plateau and search-novelty run as
        dedicated trackers fed after tool execution. Thresholds + the novelty on/off come from
        ``ExplorationSettings``.
        """
        from victor.agent.turn_policy import (
            NudgePolicy,
            PlateauDetector,
            SpinDetector,
            TurnEvaluationController,
        )
        from victor.framework.search_novelty import NoveltyConfig, SearchNoveltyTracker

        spin = SpinDetector()
        nudge_policy = NudgePolicy()
        turn_eval = TurnEvaluationController(
            spin_detector=spin,
            nudge_policy=nudge_policy,
            enable_plateau_nudge=False,
            enable_budget_warning=False,
            enable_search_novelty=False,
        )
        explore = getattr(getattr(orch, "settings", None), "exploration", None)
        return _StreamTurnGuards(
            spin=spin,
            nudge_policy=nudge_policy,
            turn_eval=turn_eval,
            plateau=PlateauDetector(),
            novelty=SearchNoveltyTracker(NoveltyConfig.from_exploration(explore)),
            novelty_enabled=bool(getattr(explore, "search_novelty_guard_enabled", True)),
        )

    _WRITE_MUTATION_TOOL_NAMES = {
        "apply_patch",
        "edit",
        "multi_edit",
        "patch",
        "replace",
        "str_replace_editor",
        "write",
    }
    _WRITE_EXPLORATION_TOOL_NAMES = {
        "cat",
        "code_search",
        "find",
        "grep",
        "ls",
        "read",
        "rg",
        "search",
        "shell",
    }
    _WRITE_ACTION_GUARD_THRESHOLD = 8
    _WRITE_ACTION_GUARD_ESCALATE_THRESHOLD = 12

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
        self._last_tool_context: Optional[str] = None
        self._last_tools: Optional[Any] = None
        self._visible_output_deduplicator = OutputDeduplicator(min_block_length=40)
        self._prev_visible_content: str = ""

    @staticmethod
    def _normalize_visible_content_key(content: str) -> str:
        """Return a stable comparison key for visible content suppression."""
        return re.sub(r"\s+", " ", content).strip().lower()

    @staticmethod
    def _normalize_visible_candidate(content: str, *, user_message: str = "") -> str:
        """Normalize raw model output into a display candidate without deduping."""
        if not content or not content.strip():
            return ""

        display_content = strip_active_completion_markers(content)
        if not display_content or not display_content.strip():
            return ""

        if user_message:
            from victor.framework.task.direct_response import (
                normalize_direct_response_output,
            )

            display_content = normalize_direct_response_output(
                user_message, display_content
            ).strip()
            if not display_content:
                return ""

        return display_content

    def _prepare_visible_content(self, content: str, *, user_message: str = "") -> str:
        """Normalize model output for display and conversation history."""
        display_content = self._normalize_visible_candidate(content, user_message=user_message)
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
    def _get_task_completion_summary(detector: Any) -> str:
        """Return the detector's last completion summary without active markers."""
        state = getattr(detector, "_state", None)
        summary = getattr(state, "last_summary", "") if state is not None else ""
        return strip_active_completion_markers(summary).strip()

    @staticmethod
    def _build_final_marker_chunk(orch: Any) -> StreamChunk:
        """Return a terminal marker chunk even when no text is available."""
        generator = getattr(orch, "_chunk_generator", None)
        marker_factory = getattr(generator, "generate_final_marker_chunk", None)
        if callable(marker_factory):
            return marker_factory()
        return StreamChunk(content="", is_final=True)

    def _resolve_terminal_visible_output(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        full_content: str,
        user_message: str,
    ) -> tuple[str, Optional[str]]:
        """Pick the best visible terminal response from provider and summary fallbacks."""
        detector = getattr(orch, "_task_completion_detector", None)
        candidates = (
            ("provider_response", full_content),
            ("completion_summary", self._get_task_completion_summary(detector)),
            ("compaction_summary", getattr(stream_ctx, "compaction_summary", "")),
        )

        sanitizer = getattr(orch, "sanitizer", None)
        for source, candidate in candidates:
            normalized = self._normalize_visible_candidate(candidate, user_message=user_message)
            if not normalized:
                continue

            sanitized = sanitizer.sanitize(normalized) if sanitizer else normalized
            if isinstance(sanitized, str) and sanitized.strip():
                return sanitized.strip(), source

            plain_text = sanitizer.strip_markup(normalized) if sanitizer else normalized
            if isinstance(plain_text, str) and plain_text.strip():
                return plain_text.strip(), source

        return "", None

    async def _govern_final_response(self, orch: Any, text: str) -> tuple[str, bool]:
        """RESPONSE-phase gate for the streaming path's final assistant output.

        Returns ``(text_to_use, blocked)``. When no message policy gate is
        configured the text is returned unchanged. Streaming cannot un-send
        tokens that were already emitted incrementally, so this governs the
        persisted history copy and the final delivered chunk (the documented
        limitation of message governance on the streaming path).
        """
        gate = getattr(orch, "_message_policy_gate", None)
        if gate is None or not text:
            return text, False
        result = await gate.gate_response(text)
        if not result.allowed:
            return result.reason or "The response was withheld by policy.", True
        return result.content, False

    async def _build_terminal_delivery_chunk(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        full_content: str,
        user_message: str,
    ) -> StreamChunk:
        """Guarantee a visible terminal assistant chunk or an explicit final marker."""
        terminal_content, content_source = self._resolve_terminal_visible_output(
            orch,
            stream_ctx,
            full_content=full_content,
            user_message=user_message,
        )

        if terminal_content:
            # RESPONSE-phase governance on the terminal (final) assistant output.
            terminal_content, _blocked = await self._govern_final_response(orch, terminal_content)

            from victor.agent.conversation.types import (
                MESSAGE_SOURCE_METADATA_KEY,
                MessageSource,
            )

            orch.add_message(
                "assistant",
                terminal_content,
                persist_synchronously=True,
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
            )
            self._append_stream_event(
                stream_ctx,
                "provider_status_events",
                {
                    "source": "streaming_executor",
                    "kind": "final_chunk_emitted",
                    "response_source": content_source,
                    "iteration": getattr(stream_ctx, "total_iterations", 0),
                    "content_length": len(terminal_content),
                },
            )
            if content_source and content_source != "provider_response":
                logger.warning(
                    "Terminal response content recovered from %s fallback (provider output was not visible)",
                    content_source,
                )
            return orch._chunk_generator.generate_content_chunk(terminal_content, is_final=True)

        self._append_stream_event(
            stream_ctx,
            "provider_status_events",
            {
                "source": "streaming_executor",
                "kind": "final_marker_emitted",
                "iteration": getattr(stream_ctx, "total_iterations", 0),
            },
        )
        logger.warning(
            "No visible terminal assistant content available; emitting final marker only"
        )
        return self._build_final_marker_chunk(orch)

    @staticmethod
    def _resolve_provider_identity(orch: Any) -> tuple[str, str]:
        """Resolve provider and model names for model-aware continuation policy."""
        provider = getattr(orch, "provider", None)
        provider_name = getattr(provider, "name", None) or getattr(orch, "provider_name", "")
        model = getattr(orch, "model", "") or ""
        return str(provider_name or "unknown"), str(model or "unknown")

    def _build_post_compaction_continuation_prompt(
        self,
        orch: Any,
        stream_ctx: Any,
    ) -> str:
        """Build a model-aware post-compaction continuation prompt."""
        from victor.agent.compaction_continuation_bonus import get_compaction_bonus
        from victor.agent.context_reminder_templates import get_post_compaction_reminder

        task_type = (
            getattr(stream_ctx, "coarse_task_type", None)
            or getattr(getattr(stream_ctx, "unified_task_type", None), "value", None)
            or "default"
        )
        provider_name, model_name = self._resolve_provider_identity(orch)
        messages_removed = int(getattr(stream_ctx, "compaction_message_removed_count", 0) or 0)
        reminder = get_post_compaction_reminder(
            task_type=task_type,
            compaction_summary=str(getattr(stream_ctx, "compaction_summary", "") or ""),
            messages_removed=messages_removed,
        )

        try:
            compaction_bonus = get_compaction_bonus().get_bonus(
                provider=provider_name,
                model=model_name,
                compaction_occurred=True,
                messages_removed=messages_removed,
            )
        except Exception:
            compaction_bonus = 1

        ledger_text = ""
        if hasattr(stream_ctx, "build_continuation_ledger"):
            ledger_text = stream_ctx.build_continuation_ledger(
                max_events=max(3, min(6, compaction_bonus + 2)),
                max_plan_steps=max(3, min(5, compaction_bonus + 1)),
                max_chars=900 if compaction_bonus >= 3 else 600,
            )

        parts = ["[CONTEXT COMPACTED]", reminder]
        if ledger_text:
            parts.append("Continuation ledger:\n" + ledger_text)
        if compaction_bonus >= 3:
            parts.append(
                "Resume from the recorded plan before you summarize or ask the user for input."
            )
        else:
            parts.append(
                "Continue from the recorded plan and use tools if more evidence is needed."
            )
        return " ".join(part for part in parts[:2]) + (
            "\n\n" + "\n\n".join(parts[2:]) if len(parts) > 2 else ""
        )

    @staticmethod
    def _append_stream_event(stream_ctx: Any, field_name: str, event: dict[str, Any]) -> None:
        """Append a structured event list onto the mutable stream context."""
        events = getattr(stream_ctx, field_name, None)
        if not isinstance(events, list):
            events = []
            setattr(stream_ctx, field_name, events)
        events.append(event)

    @staticmethod
    def _tool_name_value(tool_name: Any) -> str:
        return str(tool_name or "").split(".")[-1].strip().lower()

    def _is_write_action_turn(self, orch: Any, stream_ctx: Any) -> bool:
        """Return True for turns where continued read-only exploration is risky."""
        if not bool(getattr(stream_ctx, "is_action_task", False)):
            return False

        intent = getattr(orch, "_current_intent", None)
        intent_value = str(getattr(intent, "value", intent) or "").lower()
        if intent_value in {"write_allowed", "edit", "write"}:
            return True

        task_type = str(getattr(getattr(stream_ctx, "unified_task_type", None), "value", "") or "")
        coarse_type = str(getattr(stream_ctx, "coarse_task_type", "") or "")
        task_text = f"{task_type} {coarse_type}".lower()
        return any(token in task_text for token in ("edit", "write", "code", "coding"))

    def _has_mutation_tool_executed(self, stream_ctx: Any) -> bool:
        tool_names = {
            self._tool_name_value(name)
            for name in (getattr(stream_ctx, "executed_tool_names", set()) or set())
            if name
        }
        return bool(tool_names & self._WRITE_MUTATION_TOOL_NAMES)

    def _maybe_inject_write_action_guard(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        user_message: str,
        tool_calls_used: int,
    ) -> bool:
        """Nudge write-intent turns out of repeated read-only exploration."""
        if not self._is_write_action_turn(orch, stream_ctx):
            return False
        if self._has_mutation_tool_executed(stream_ctx):
            return False

        executed_names = {
            self._tool_name_value(name)
            for name in (getattr(stream_ctx, "executed_tool_names", set()) or set())
            if name
        }
        if not executed_names:
            return False
        if not (executed_names & self._WRITE_EXPLORATION_TOOL_NAMES):
            return False

        threshold = self._WRITE_ACTION_GUARD_THRESHOLD
        level = "initial"
        if tool_calls_used >= self._WRITE_ACTION_GUARD_ESCALATE_THRESHOLD:
            threshold = self._WRITE_ACTION_GUARD_ESCALATE_THRESHOLD
            level = "escalated"
        if tool_calls_used < threshold:
            return False

        marker = f"_write_action_guard_{level}_injected"
        if getattr(stream_ctx, marker, False):
            return False
        setattr(stream_ctx, marker, True)

        from victor.agent.conversation.types import (
            MESSAGE_SOURCE_METADATA_KEY,
            MessageSource,
        )

        if level == "escalated":
            message = (
                "[SYSTEM: This is still an edit/action task. You have already used "
                f"{tool_calls_used} tool call(s) and no mutation tool has run. Do not call "
                "read, grep, code_search, or shell again unless it directly executes the edit. "
                "Next response must either call edit/write/apply_patch with a concrete change, "
                "or return a concise blocker explaining the exact file/location that prevents "
                "a safe edit.]"
            )
        else:
            message = (
                "[SYSTEM: This is an edit/action task, and enough repository context has been "
                f"gathered ({tool_calls_used} tool call(s)) without any mutation. Stop broad "
                "read-only exploration. On the next turn, apply the smallest concrete code "
                "change with an edit/write/apply_patch-style tool, or state the precise blocker "
                "that makes a safe edit impossible.]"
            )

        orch.add_message(
            "system",
            message,
            metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.SYSTEM_INJECTED.value},
        )
        self._append_stream_event(
            stream_ctx,
            "provider_status_events",
            {
                "source": "streaming_executor",
                "kind": "write_action_guard_injected",
                "level": level,
                "tool_calls_used": tool_calls_used,
                "executed_tool_names": sorted(executed_names),
                "user_message_preview": user_message[:160],
            },
        )
        logger.info(
            "Injected write-action guard after %s read-only tool calls (level=%s, tools=%s)",
            tool_calls_used,
            level,
            sorted(executed_names),
        )
        return True

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

    async def _get_tools_cached(
        self,
        orch: Any,
        context_msg: str,
        goals: Any,
        planned_tools: Any = None,
    ) -> Any:
        """Select tools with per-turn caching."""
        if self._last_tool_context == context_msg and self._last_tools is not None:
            return self._last_tools

        tools = await orch._select_tools_for_turn(
            context_msg,
            goals,
            planned_tools=planned_tools,
        )
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
            from victor.agent.conversation.types import (
                MESSAGE_SOURCE_METADATA_KEY,
                MessageSource,
            )

            orch.add_message(
                "assistant",
                final_output,
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
            )
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

    @staticmethod
    async def _extract_task_requirements(orch: Any, user_message: str) -> None:
        """Extract required files/outputs from the prompt onto orch + emit a state event.

        A cohesive piece of run()'s preamble (FEP-0007 Phase 2 decomposition). Mutates orch
        run-state; yields nothing.
        """
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

    @staticmethod
    def _apply_run_guidance(
        orch: Any, stream_ctx: Any, user_message: str, max_exploration_iterations: int
    ) -> None:
        """Apply intent guard + task-type guidance + action-task guidance for one run.

        A cohesive piece of run()'s preamble (FEP-0007 Phase 2 decomposition). Mutates orch
        (guards + guidance messages); yields nothing.
        """
        orch._apply_intent_guard(user_message)

        if stream_ctx.is_analysis_task and stream_ctx.unified_task_type.value in (
            "edit",
            "create",
        ):
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

            from victor.agent.conversation.history_metadata import (
                build_internal_history_metadata,
            )
            from victor.agent.conversation.types import MessageSource

            _guidance_meta = build_internal_history_metadata(
                "action_guidance", source=MessageSource.AGENT_GUIDANCE
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
                    metadata=_guidance_meta,
                )
            else:
                orch.add_message(
                    "user",
                    "[ACTION-GUIDANCE: This is an action-oriented task (create/write/build). "
                    "Minimize exploration and proceed directly to creating what was requested. "
                    "Only explore if absolutely necessary to complete the task.]",
                    metadata=_guidance_meta,
                )

    @staticmethod
    def _initialize_task_intent(orch: Any, stream_ctx: Any, user_message: str) -> Any:
        """Seed task intent on the stream context and return the inferred goals.

        Final cohesive piece of run()'s preamble (FEP-0007 Phase 2 decomposition). Mutates
        stream_ctx (task intent, plan steps, task-start event); yields nothing. Returns the
        inferred ``goals`` so the loop can use them for tool planning.
        """
        goals = orch._tool_planner.infer_goals_from_message(user_message)
        if hasattr(stream_ctx, "set_task_intent"):
            stream_ctx.set_task_intent(user_message)
        if hasattr(stream_ctx, "extend_plan_steps"):
            stream_ctx.extend_plan_steps(goals)
        if hasattr(stream_ctx, "record_intent_event"):
            stream_ctx.record_intent_event(
                "task_start",
                f"task start ({stream_ctx.coarse_task_type})",
                task_type=stream_ctx.coarse_task_type,
            )
        return goals

    async def _stream_provider_turn(
        self,
        orch: Any,
        runtime_owner: Any,
        stream_ctx: Any,
        goals: Any,
    ) -> tuple[Any, str, Any, bool]:
        """Resolve this turn's tools and stream the provider response.

        The ACT provider sub-step of run()'s per-turn body (FEP-0007 Phase 2, toward a
        shared stream_turn() boundary). Plans tools, resolves the active tool set, builds
        provider kwargs, and streams the provider response for ONE turn. Sets
        stream_ctx.planned_tools and returns the ``(tools, full_content, tool_calls,
        garbage_detected)`` tuple (tools is needed by run() for empty-response recovery).
        Token streaming happens inside ``_stream_provider_response``; this helper yields
        nothing.
        """
        planned_tools = None
        if goals:
            available_inputs = ["query"]
            if orch.observed_files:
                available_inputs.append("file_contents")
            planned_tools = orch._tool_planner.plan_tools(goals, available_inputs)
        stream_ctx.planned_tools = planned_tools

        if getattr(stream_ctx, "is_qa_task", False):
            tools = None
        elif orch.get_session_tools() is not None:
            tools = orch.get_session_tools()
        else:
            tools = await self._get_tools_cached(
                orch,
                stream_ctx.context_msg,
                goals,
                planned_tools=planned_tools,
            )

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
        return tools, full_content, tool_calls, garbage_detected

    async def _execute_tools_turn(
        self,
        orch: Any,
        runtime_owner: Any,
        stream_ctx: Any,
        *,
        user_message: str,
        tool_calls: Any,
        full_content: str,
        result_holder: _ToolTurnOutcome,
    ) -> AsyncIterator[StreamChunk]:
        """Execute this turn's tool calls, streaming chunks; record the result on the holder.

        The ACT tool-execution sub-step of run()'s per-turn body (FEP-0007 Phase 2, toward a
        shared stream_turn() boundary). Lazily builds the tool-execution handler, streams (or
        buffers) tool-result chunks, then applies post-execution bookkeeping (tool-call
        accounting + write-action guard). Because an async generator can't return a value, the
        produced ``ToolExecutionResult`` is written to ``result_holder.result`` for run() to read
        (``should_return`` loop control + downstream fulfillment/plateau/novelty evaluation).
        """
        if not runtime_owner._tool_execution_handler:
            from victor.agent.streaming import create_tool_execution_handler

            runtime_owner._tool_execution_handler = create_tool_execution_handler(orch)

        runtime_owner._tool_execution_handler.update_observed_files(
            set(orch.observed_files) if orch.observed_files else set()
        )

        if hasattr(runtime_owner._tool_execution_handler, "execute_tools_streaming"):
            from victor.agent.streaming.tool_execution import (
                ToolExecutionResult,
            )

            tool_exec_result = ToolExecutionResult()
            async for chunk in runtime_owner._tool_execution_handler.execute_tools_streaming(
                stream_ctx=stream_ctx,
                tool_calls=tool_calls,
                user_message=user_message,
                full_content=full_content,
                tool_calls_used=orch.tool_calls_used,
                tool_budget=orch.tool_budget,
                result=tool_exec_result,
            ):
                yield chunk
        else:
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
        stream_ctx.tool_calls_used = orch.tool_calls_used
        _record = getattr(stream_ctx, "record_iteration_tool_count", None)
        if callable(_record):
            _record(tool_exec_result.tool_calls_executed)
        self._maybe_inject_write_action_guard(
            orch,
            stream_ctx,
            user_message=user_message,
            tool_calls_used=orch.tool_calls_used,
        )

        result_holder.result = tool_exec_result

    async def _evaluate_provider_turn_stops(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        full_content: str,
        tool_calls: Any,
        spin: Any,
        turn_eval: Any,
        outcome: _ProviderTurnEval,
    ) -> AsyncIterator[StreamChunk]:
        """Record per-turn spin signals and run the early-stop checks for the provider turn.

        The EVALUATE sub-step that runs right after the provider response (FEP-0007 Phase 2,
        toward a shared stream_turn() boundary). Feeds the SpinDetector this turn's tool
        signatures, then applies the three early stops — TERMINATED spin, content-repetition
        (shared TurnEvaluationController), and a pre-set ``skip_continuation`` — each of which
        forces completion and ends the turn. Because this yields the final stop chunk, it can't
        return a value: it writes ``should_return`` plus the derived ``content_length`` /
        ``has_tools`` metrics onto ``outcome`` for run() to read.
        """
        from victor.agent.turn_policy import TurnObservation

        # P1 FIX: Earlier spin state detection
        # Move spin check earlier (before content repetition detection) so that spin state
        # can influence continuation decisions. This helps detect stuck patterns sooner.
        content_length = len(full_content.strip()) if full_content else 0
        _has_tools = bool(tool_calls)
        _no_progress = not _has_tools and content_length < 120
        outcome.content_length = content_length
        outcome.has_tools = _has_tools

        tool_names_set = {tc.get("name", "") for tc in tool_calls} if tool_calls else set()
        # Feed per-turn tool-call signatures so the SpinDetector's repeated-signature
        # path is live in streaming. Without these it only tracked names/counts and
        # never tripped on a model re-issuing the *same* tool call (same args) every
        # turn — the exact shape of the observed code_search loop.
        spin.record_turn(
            has_tool_calls=_has_tools,
            tool_names=tool_names_set,
            tool_count=len(tool_calls) if tool_calls else 0,
            tool_signatures=_tool_call_signatures(tool_calls) if tool_calls else None,
        )

        logger.debug(
            "[spin-check] tool_calls=%s content_len=%s no_progress=%s state=%s",
            len(tool_calls) if tool_calls else 0,
            content_length,
            _no_progress,
            spin.state.value,
        )

        # P1 FIX: Check for terminated spin state early
        # If already in TERMINATED state, we should force completion immediately
        # instead of waiting for later checks
        if spin.state.name == "TERMINATED":
            logger.warning(
                "[spin-detect-early] Spin state TERMINATED detected early - forcing completion"
            )
            stream_ctx.force_completion = True
            stream_ctx.skip_continuation = True
            outcome.should_return = True
            yield orch._chunk_generator.generate_content_chunk(
                "\n\n[Agent detected a response loop pattern — breaking to prevent wasted time.]",
                is_final=True,
            )
            return

        # Content-repetition detection via the shared TurnEvaluationController (same
        # hash/overlap detector as the headless loop). The loop already fed the spin
        # detector above, so don't re-record. On a repetition stop, force completion and
        # emit a final chunk to break the feedback loop.
        if full_content:
            _decision = turn_eval.evaluate(TurnObservation(content=full_content), record_spin=False)
            if _decision.stop:
                logger.warning(
                    "[content-repetition] %s (content_len=%s) — forcing completion.",
                    _decision.stop_reason,
                    len(full_content),
                )
                stream_ctx.force_completion = True
                stream_ctx.skip_continuation = True
                outcome.should_return = True
                yield orch._chunk_generator.generate_content_chunk(
                    _decision.stop_message
                    or "\n\n[Content repetition detected — stopping to prevent "
                    "infinite output loop.]",
                    is_final=True,
                )
                return

        # P0 FIX: Explicit exit check after content repetition detection
        # This ensures that if skip_continuation was set by content repetition or any other
        # reason, we exit the loop immediately instead of continuing to continuation handling
        if getattr(stream_ctx, "skip_continuation", False):
            logger.info(
                "Exiting loop: skip_continuation flag set (content repetition or other forced "
                "completion)"
            )
            outcome.should_return = True
            return

    async def _evaluate_post_tool_turn(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        perception: Any,
        tool_exec_result: Any,
        full_content: str,
        user_message: str,
        plateau: Any,
        novelty: Any,
        novelty_enabled: bool,
        outcome: _PostToolEval,
    ) -> AsyncIterator[StreamChunk]:
        """Run fulfillment + plateau + search-novelty evaluation after tool execution.

        The post-tool EVALUATE band of run()'s per-turn body (FEP-0007 Phase 2, toward a
        shared stream_turn() boundary). Checks task fulfillment (ends the turn when fulfilled),
        runs plateau detection (injects a nudge when unproductive), and the search-novelty
        safety-net (forces synthesis — yielding a final chunk — when successive searches stop
        surfacing new files, else nudges). Because the safety-net yields, the loop-exit decision
        is written to ``outcome.should_return`` for run() to read rather than returned.
        """
        if self._fulfillment and perception:
            try:
                from victor.agent.turn_policy import FulfillmentCriteriaBuilder
                from victor.framework.fulfillment import TaskType

                task_analysis = getattr(perception, "task_analysis", None)
                task_type_str = (
                    getattr(task_analysis, "task_type", "unknown") if task_analysis else "unknown"
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
                    context={
                        "full_content": full_content,
                        "user_message": user_message,
                    },
                )
                if fulfillment_result.is_fulfilled:
                    logger.info(
                        "Streaming fulfillment: task fulfilled (score=%.2f, reason=%s)",
                        fulfillment_result.score,
                        fulfillment_result.reason,
                    )
                    outcome.should_return = True
                    return
            except Exception as exc:
                logger.debug("Streaming fulfillment check skipped: %s", exc)

        # Plateau detection via the shared PlateauDetector (same productivity-weighted
        # formula and "nudge only when unproductive" rule the headless loop uses).
        productive_count = _count_productive_tools(tool_exec_result)
        content_len = len(full_content) if full_content else 0
        _plateau_res = plateau.record(productive_count, content_len)
        if _plateau_res.is_plateau and not _plateau_res.should_nudge:
            logger.info(
                "Skipping plateau nudge: agent made %d productive tool call(s) this iteration",
                productive_count,
            )
        elif _plateau_res.should_nudge:
            logger.info(
                "Streaming progress plateau detected (scores=%s), injecting nudge",
                [f"{score:.2f}" for score in _plateau_res.recent_scores],
            )
            _plateau_intent = getattr(orch, "_current_intent", None)
            _is_write = (
                _plateau_intent is not None
                and getattr(_plateau_intent, "value", None) == "write_allowed"
            )
            from victor.agent.conversation.types import (
                MESSAGE_SOURCE_METADATA_KEY,
                MessageSource,
            )
            from victor.agent.turn_policy import plateau_nudge_message

            orch.add_message(
                "system",
                plateau_nudge_message(_is_write),
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.SYSTEM_INJECTED.value},
            )

        # Search-novelty safety-net: if successive searches stop surfacing new files,
        # synthesize instead of thrashing. Skip when actively editing (real work).
        from victor.tools.tool_names import get_canonical_name

        _nov_results = getattr(tool_exec_result, "tool_results", None) or []
        _nov = novelty.record_turn(_nov_results)
        _editing = any(
            get_canonical_name(r.get("tool_name") or r.get("name") or "")
            in {"edit", "write", "create_file", "replace_in_file"}
            for r in _nov_results
            if isinstance(r, dict)
        )
        _iter_no = getattr(stream_ctx, "total_iterations", 0)
        if novelty_enabled and _nov.should_force_complete and not _editing and _iter_no >= 2:
            logger.info(
                "[search-novelty] %d consecutive low-novelty searches — synthesizing.",
                _nov.consecutive_low_novelty,
            )
            stream_ctx.force_completion = True
            stream_ctx.skip_continuation = True
            outcome.should_return = True
            yield orch._chunk_generator.generate_content_chunk(
                "\n\n[Enough context gathered — synthesizing the answer.]",
                is_final=True,
            )
            return
        if novelty_enabled and _nov.should_nudge:
            from victor.agent.conversation.types import (
                MESSAGE_SOURCE_METADATA_KEY,
                MessageSource,
            )
            from victor.framework.search_novelty import synthesize_nudge_message

            orch.add_message(
                "system",
                synthesize_nudge_message(),
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.SYSTEM_INJECTED.value},
            )

    async def run(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Run the streaming executor for the provided message."""
        runtime_owner = self._runtime_owner
        orch = runtime_owner._orchestrator
        recovery = getattr(orch, "_recovery_service", None) or orch._recovery_coordinator
        create_recovery_context = orch.create_recovery_context

        # Governance REQUEST phase: gate/redact the user message before any state
        # is set up or the LLM is called. A block short-circuits the stream with
        # a single refusal chunk; a redaction substitutes the message text used
        # downstream (and stored in history).
        _gate = getattr(orch, "_message_policy_gate", None)
        if _gate is not None:
            _req = await _gate.gate_request(user_message)
            if not _req.allowed:
                yield orch._chunk_generator.generate_content_chunk(
                    _req.reason or "Your message was blocked by policy.",
                    is_final=True,
                )
                return
            user_message = _req.content

        self._reset_streaming_turn_state(orch)

        stream_ctx = await runtime_owner._create_stream_context(user_message, **kwargs)
        orch._current_stream_context = stream_ctx

        await self._extract_task_requirements(orch, user_message)

        max_total_iterations = stream_ctx.max_total_iterations
        max_exploration_iterations = stream_ctx.max_exploration_iterations

        self._apply_run_guidance(orch, stream_ctx, user_message, max_exploration_iterations)

        goals = self._initialize_task_intent(orch, stream_ctx, user_message)

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

        # Loop-scoped names used in the per-turn body below.
        from victor.agent.turn_policy import SpinState

        # Per-turn guard components (built once per run via the extracted factory — the first
        # step of FEP-0007 Phase 2 toward a shared stream_turn() boundary).
        _guards = self._create_stream_turn_guards(orch)
        _spin = _guards.spin
        _nudge_policy = _guards.nudge_policy
        _turn_eval = _guards.turn_eval
        _plateau = _guards.plateau
        _novelty = _guards.novelty
        _novelty_enabled = _guards.novelty_enabled

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

        _prev_iteration_had_content = False

        _turn_eval.reset()  # resets spin_detector + shared content-repetition detector
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
            chat_service = getattr(orch, "_chat_service", None)
            if chat_service is not None and hasattr(
                chat_service, "handle_context_and_iteration_limits"
            ):
                handled, iter_chunk = await chat_service.handle_context_and_iteration_limits(
                    user_message,
                    max_total_iterations,
                    max_context,
                    stream_ctx.total_iterations,
                    stream_ctx.last_quality_score,
                )
            else:
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

            tools, full_content, tool_calls, garbage_detected = await self._stream_provider_turn(
                orch, runtime_owner, stream_ctx, goals
            )

            if self._confidence_monitor is not None and not tool_calls:
                try:
                    from victor.core.feature_flags import (
                        FeatureFlag,
                        is_feature_enabled,
                    )

                    if is_feature_enabled(FeatureFlag.USE_CONFIDENCE_MONITOR):
                        self._confidence_monitor.record(
                            full_content or "", stream_ctx.estimated_content_tokens
                        )
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
                stream_ctx.estimated_content_tokens,
                garbage_detected,
                content_preview,
            )

            if garbage_detected and not tool_calls:
                stream_ctx.force_completion = True
                logger.info("Setting force_completion due to garbage detection")

            _turn_stop = _ProviderTurnEval()
            async for chunk in self._evaluate_provider_turn_stops(
                orch,
                stream_ctx,
                full_content=full_content,
                tool_calls=tool_calls,
                spin=_spin,
                turn_eval=_turn_eval,
                outcome=_turn_stop,
            ):
                yield chunk
            if _turn_stop.should_return:
                return
            content_length = _turn_stop.content_length
            _has_tools = _turn_stop.has_tools

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
                    from victor.agent.conversation.history_metadata import (
                        build_internal_history_metadata,
                    )
                    from victor.agent.conversation.types import MessageSource

                    orch.add_message(
                        "user",
                        f"[TOOL-FORMAT-HINT: You described wanting to use '{tool_hint}' but didn't call it. "
                        f"Call the tool directly — don't describe what you want to do, execute it. "
                        f"If you've already modified the file successfully, say {FILE_DONE_MARKER}]",
                        metadata=build_internal_history_metadata(
                            "tool_format_hint", source=MessageSource.AGENT_GUIDANCE
                        ),
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

            assistant_content_yielded = False
            if full_content:
                visible_content = self._prepare_visible_content(
                    full_content,
                    user_message=getattr(stream_ctx, "user_message", ""),
                )
                if visible_content:
                    _prev_iteration_had_content = True
                sanitized = orch.sanitizer.sanitize(visible_content)
                if sanitized:
                    from victor.agent.conversation.types import (
                        MESSAGE_SOURCE_METADATA_KEY,
                        MessageSource,
                    )

                    # RESPONSE-phase governance only on the final emit; intermediate
                    # content turns continue the loop and must not be blocked.
                    _is_final_emit = forced_task_completion and not tool_calls
                    if _is_final_emit:
                        sanitized, _ = await self._govern_final_response(orch, sanitized)
                    orch.add_message(
                        "assistant",
                        sanitized,
                        tool_calls=tool_calls,
                        persist_synchronously=_is_final_emit,
                        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                    )
                    assistant_content_yielded = True
                    yield orch._chunk_generator.generate_content_chunk(
                        sanitized,
                        is_final=_is_final_emit,
                    )
                    if _is_final_emit:
                        return
                else:
                    plain_text = orch.sanitizer.strip_markup(
                        visible_content
                        or self._normalize_visible_candidate(
                            full_content,
                            user_message=getattr(stream_ctx, "user_message", ""),
                        )
                    )
                    if plain_text:
                        from victor.agent.conversation.types import (
                            MESSAGE_SOURCE_METADATA_KEY,
                            MessageSource,
                        )

                        _is_final_emit = forced_task_completion and not tool_calls
                        if _is_final_emit:
                            plain_text, _ = await self._govern_final_response(orch, plain_text)
                        orch.add_message(
                            "assistant",
                            plain_text,
                            tool_calls=tool_calls,
                            persist_synchronously=_is_final_emit,
                            metadata={
                                MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value
                            },
                        )
                        assistant_content_yielded = True
                        yield orch._chunk_generator.generate_content_chunk(
                            plain_text,
                            is_final=_is_final_emit,
                        )
                        if _is_final_emit:
                            return
                    elif forced_task_completion and not tool_calls:
                        yield await self._build_terminal_delivery_chunk(
                            orch,
                            stream_ctx,
                            full_content=full_content,
                            user_message=user_message,
                        )
                        return
            elif tool_calls:
                from victor.agent.conversation.types import (
                    MESSAGE_SOURCE_METADATA_KEY,
                    MessageSource,
                )

                orch.add_message(
                    "assistant",
                    "",
                    tool_calls=tool_calls,
                    metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                )
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

            # P1 FIX: Spin state already recorded earlier (before content repetition check)
            # This section now only handles nudge policy and detailed termination messages
            _no_progress = not _has_tools and content_length < 120

            if _no_progress:
                _intent = getattr(orch, "_current_intent", None)
                nudge = _nudge_policy.evaluate(_spin, intent=_intent)
                if nudge.should_inject:
                    from victor.agent.conversation.history_metadata import (
                        build_internal_history_metadata,
                    )
                    from victor.agent.conversation.types import MessageSource

                    orch.add_message(
                        nudge.role,
                        nudge.message,
                        metadata=build_internal_history_metadata(
                            "nudge", source=MessageSource.AGENT_NUDGE
                        ),
                    )
                    logger.info("[nudge] %s", nudge.nudge_type.value)

                if _spin.state == SpinState.TERMINATED:
                    logger.warning(
                        "[spin-detect] Terminated after no_tool=%s blocked=%s. Last response: %r",
                        _spin.consecutive_no_tool_turns,
                        _spin.consecutive_all_blocked,
                        full_content[:100],
                    )
                    if getattr(stream_ctx, "is_action_task", False):
                        loop_break_message = (
                            "\n\n[Agent detected a repeated no-tool response loop while trying to "
                            "resume an action task. The previous action did not recover cleanly. "
                            "Start a follow-up turn or make one concrete tool call to continue.]"
                        )
                    elif getattr(stream_ctx, "is_analysis_task", False):
                        loop_break_message = (
                            "\n\n[Agent detected a repeated no-tool response loop while trying to "
                            "continue analysis. Start a follow-up turn or make one concrete discovery "
                            "tool call to continue.]"
                        )
                    else:
                        loop_break_message = "\n\n[Agent detected a response loop — breaking to prevent wasted time.]"
                    yield orch._chunk_generator.generate_content_chunk(
                        loop_break_message,
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
                        from victor.agent.streaming import (
                            create_intent_classification_handler,
                        )

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
                                post_compaction_prompt = (
                                    self._build_post_compaction_continuation_prompt(
                                        orch,
                                        stream_ctx,
                                    )
                                )
                                orch.add_message(
                                    "user",
                                    post_compaction_prompt,
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
                        continuation_visible = any(
                            bool(getattr(chunk, "content", "").strip())
                            for chunk in continuation_result.chunks
                        )
                        if (
                            not tool_calls
                            and not assistant_content_yielded
                            and not continuation_visible
                        ):
                            yield await self._build_terminal_delivery_chunk(
                                orch,
                                stream_ctx,
                                full_content=full_content,
                                user_message=user_message,
                            )
                        elif not continuation_result.chunks:
                            yield self._build_final_marker_chunk(orch)
                        return

                _tool_outcome = _ToolTurnOutcome()
                async for chunk in self._execute_tools_turn(
                    orch,
                    runtime_owner,
                    stream_ctx,
                    user_message=user_message,
                    tool_calls=tool_calls,
                    full_content=full_content,
                    result_holder=_tool_outcome,
                ):
                    yield chunk
                tool_exec_result = _tool_outcome.result

                if tool_exec_result.should_return:
                    return

                _post_tool = _PostToolEval()
                async for chunk in self._evaluate_post_tool_turn(
                    orch,
                    stream_ctx,
                    perception=_perception,
                    tool_exec_result=tool_exec_result,
                    full_content=full_content,
                    user_message=user_message,
                    plateau=_plateau,
                    novelty=_novelty,
                    novelty_enabled=_novelty_enabled,
                    outcome=_post_tool,
                ):
                    yield chunk
                if _post_tool.should_return:
                    return


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
