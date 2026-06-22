# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned tool selection runtime helper."""

from __future__ import annotations

import logging
from typing import Any

from victor.tools.tool_supply_trace import TOOL_SUPPLY_TOPIC, ToolSupplyTrace

logger = logging.getLogger(__name__)


def _emit_tool_supply_trace(trace: ToolSupplyTrace) -> None:
    """Emit the per-turn tool-supply event. Best-effort; never breaks selection.

    Mirrors the ``tool.intent`` emission pattern: fetch the observability bus and
    publish a structured payload. Any failure is swallowed — telemetry must never
    affect which tools the model receives.
    """
    try:
        from victor.core.events import get_observability_bus

        bus = get_observability_bus()
        if bus is None:
            return
        bus.emit_sync(
            topic=TOOL_SUPPLY_TOPIC,
            data=trace.to_payload(),
            source="ToolSelectionRuntime",
        )
    except Exception:  # telemetry is non-critical
        logger.debug("tool-supply trace emit failed", exc_info=True)


# Read-oriented task types: WRITE_ALLOWED is a *permission*, not an intent to write, so we
# do not force mutation tools (edit/write/shell) onto these pure analysis/search turns.
_READ_ORIENTED_TASK_TYPES = frozenset({"analyze", "search", "research"})

# Minimal read-only core handed to borderline Q&A turns (tool-supply P3) instead of None.
_READ_CORE_TOOL_NAMES = ("read", "code_search", "ls")


# --- E3-TIR experience-replay exploration (opt-in via USE_E3_TIR_EXPLORATION) ---------
# The experience STORE holds user-global RL data, so it is shared across sessions and fed
# by a single idempotent TOOL_EXECUTED hook. The per-session SELECTOR (which owns the
# exploration-phase warm-up schedule) reads that shared store. Keeping the store shared
# but the selector per-session avoids cross-session contamination while still letting each
# session run its own demonstration -> self-play -> exploration ramp.
_e3tir_shared_store: Any = None
_e3tir_hook_subscribed: bool = False


def _warm_start_experience_store(store: Any, limit: int = 2000) -> None:
    """Warm-start the experience projection from durable RL_OUTCOME tool rows (R3).

    The store is an in-memory projection of the durable outcome stream but starts
    empty each process, drifting from the durable truth. Replaying recent durable
    tool outcomes (chronological) makes it a true projection. Best-effort: any
    failure leaves the store empty rather than breaking selection. Reward uses the
    R2 canonical ``reward_from_signals`` so it matches the live feed exactly.
    """
    try:
        import json

        from victor.core.database import get_database
        from victor.framework.rl.reward import reward_from_signals

        rows = get_database().query(
            "SELECT task_type, success, quality_score, metadata FROM rl_outcome "
            "WHERE learner_id = ? ORDER BY created_at DESC LIMIT ?",
            ("tool_selector", limit),
        )
        outcomes = []
        for row in reversed(rows or []):  # oldest -> newest for faithful running stats
            task_type, success, quality_score, metadata = row[0], row[1], row[2], row[3]
            try:
                meta = json.loads(metadata) if metadata else {}
            except (TypeError, ValueError):
                meta = {}
            tool = meta.get("tool_name")
            if not tool:
                continue
            reward = reward_from_signals(success=bool(success), quality_score=quality_score)
            outcomes.append((tool, task_type, bool(success), reward))
        replayed = store.warm_start_from_outcomes(outcomes)
        if replayed:
            logger.info(
                "E3-TIR: warm-started experience projection from %d durable outcomes",
                replayed,
            )
    except Exception:  # warm-start is best-effort; live feed still populates the store
        logger.debug("E3-TIR experience warm-start skipped", exc_info=True)


def _get_e3tir_shared_store() -> Any:
    """Return the process-shared ToolExperienceStore, subscribing the outcome hook once."""
    global _e3tir_shared_store, _e3tir_hook_subscribed
    if _e3tir_shared_store is None:
        from victor.tools.experience_store import ToolExperienceStore

        _e3tir_shared_store = ToolExperienceStore()
        # Make the in-memory store a projection of the durable outcome stream before
        # the live feed begins, so it doesn't start blank (and drift) each process.
        _warm_start_experience_store(_e3tir_shared_store)
    if not _e3tir_hook_subscribed:
        from victor.framework.rl.hooks import RLEventType, get_rl_hooks

        store = _e3tir_shared_store

        def _record(event: Any) -> None:
            try:
                tool = getattr(event, "tool_name", None)
                if not tool:
                    return
                success = event.success if event.success is not None else True
                # Canonical default reward (R2) — was an inline
                # `quality_score or (1.0 if success else 0.3)` here; now the single
                # source in victor.framework.rl.reward so it can't drift.
                from victor.framework.rl.reward import reward_from_signals

                reward = reward_from_signals(
                    success=bool(success), quality_score=event.quality_score
                )
                store.record_outcome(
                    tool_name=tool,
                    task_type=getattr(event, "task_type", None) or "general",
                    success=bool(success),
                    reward=reward,
                )
            except Exception:
                logger.debug("E3-TIR outcome recording failed", exc_info=True)

        get_rl_hooks().add_handler(RLEventType.TOOL_EXECUTED, _record)
        _e3tir_hook_subscribed = True
    return _e3tir_shared_store


def _reset_e3tir_state_for_tests() -> None:
    """Reset the process-shared E3-TIR store/hook flag (test isolation only)."""
    global _e3tir_shared_store, _e3tir_hook_subscribed
    _e3tir_shared_store = None
    _e3tir_hook_subscribed = False


class ToolSelectionRuntime:
    """Bridge orchestrator runtime state to the canonical tool-selection path."""

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    async def select_tools_for_turn(
        self,
        context_msg: str,
        goals: Any,
        planned_tools: Any = None,
    ) -> Any:
        """Select, prioritize, and filter tools for the current turn."""
        runtime = self._runtime
        provider_supports_tools = runtime.provider.supports_tools()
        tooling_allowed = provider_supports_tools and runtime._model_supports_tool_calls()
        # Intent and mutation authorization are turn-scoped user-prompt state.
        # Assistant progress narration may refine semantic context below, but it
        # must not become the anchor for intent filtering or stage prioritization.
        user_message_anchor = getattr(runtime, "_current_user_message", None) or context_msg

        # Per-turn tool-supply telemetry (observe-only; never alters the value
        # flowing through this method). Captures the registered set and every
        # narrowing stage so over-restriction is queryable, not just log-grep-able.
        trace = ToolSupplyTrace.begin(self._registered_tools(runtime))

        if not tooling_allowed:
            _emit_tool_supply_trace(trace.mark_skipped("provider_or_model_no_tools"))
            return None

        # Q&A necessity gate (tool-supply P3). A trivially-safe greeting still hard-skips
        # (no tools); a borderline Q&A turn gets a minimal read-only core instead of None,
        # so "how does X work?" can still read X rather than looping tool-less.
        skip_mode_fn = getattr(runtime, "_tool_skip_mode", None)
        if skip_mode_fn is not None:
            skip_mode = skip_mode_fn(user_message_anchor)
        else:  # back-compat for hosts without the 3-valued gate
            skip_mode = (
                "skip" if runtime._should_skip_tools_for_turn(user_message_anchor) else "tools"
            )
        if skip_mode == "skip":
            _emit_tool_supply_trace(trace.mark_skipped("qa_greeting"))
            return None
        if skip_mode == "read_core":
            core = self._read_core_tools(runtime)
            trace.set_candidates(core)
            _emit_tool_supply_trace(trace.finalize(core))
            return core or None

        if planned_tools is None and goals:
            available_inputs = ["query"]
            if runtime.observed_files:
                available_inputs.append("file_contents")
            planned_tools = runtime._tool_planner.plan_tools(goals, available_inputs)
            logger.info("available_inputs=%s", available_inputs)

        conversation_depth = runtime.conversation.message_count()
        conversation_history = (
            [msg.model_dump() for msg in runtime.messages] if runtime.messages else None
        )
        tools = await runtime.tool_selector.select_tools(
            user_message_anchor,
            use_semantic=runtime.use_semantic_selection,
            conversation_history=conversation_history,
            conversation_depth=conversation_depth,
            planned_tools=planned_tools,
        )
        trace.set_candidates(tools)
        logger.info(
            "context_msg=%s\nuse_semantic=%s\nconversation_depth=%s",
            context_msg,
            runtime.use_semantic_selection,
            conversation_depth,
        )
        # Stage prioritization should stay anchored to the user's request.
        # Feeding assistant progress narration here can wrongly push the
        # conversation state machine back toward analysis/search-heavy tools.
        _prev = tools
        tools = runtime.tool_selector.prioritize_by_stage(user_message_anchor, tools)
        trace.record("stage_priority", _prev, tools)
        current_intent = getattr(runtime, "_current_intent", None)
        _prev = tools
        tools = runtime._tool_planner.filter_tools_by_intent(
            tools,
            current_intent,
            user_message=user_message_anchor,
        )
        trace.record("intent_filter", _prev, tools, reason=str(current_intent or ""))
        _prev = tools
        tools = self._ensure_write_tools_for_write_intent(tools, current_intent)
        trace.record("ensure_write", _prev, tools)
        _prev = tools
        tools = self._prioritize_explicit_database_tools(tools, user_message_anchor)
        trace.record("explicit_db", _prev, tools)
        _prev = tools
        tools = self._apply_e3tir_exploration(tools, user_message_anchor)
        trace.record("e3tir_rerank", _prev, tools)
        _prev = tools
        tools = runtime._apply_kv_tool_strategy(tools)
        trace.record("kv_strategy", _prev, tools)
        _prev = tools
        final_tools = runtime._sort_tools_for_kv_stability(tools)
        trace.record("kv_sort", _prev, final_tools)
        # Log the tools actually dispatched to the provider. The selector's earlier
        # "Selected N tools" line is emitted before stage pruning / intent filtering and
        # can overstate the callable set (a tool can be selected then pruned away). This
        # post-transform line is the source of truth for what the model receives.
        try:
            logger.info(
                "Tools dispatched to provider (%d): %s",
                len(final_tools),
                ", ".join(getattr(t, "name", "?") for t in final_tools),
            )
        except Exception:  # logging must never break tool selection
            logger.debug("dispatched-tools log failed", exc_info=True)
        _emit_tool_supply_trace(trace.finalize(final_tools))
        return final_tools

    @staticmethod
    def _registered_tools(runtime: Any) -> Any:
        """Best-effort snapshot of the enabled registered tool set (for telemetry).

        Returns ``None`` on any failure — the trace simply records a zero registered
        count rather than letting telemetry interfere with selection.
        """
        try:
            selector = getattr(runtime, "tool_selector", None)
            registry = getattr(selector, "tools", None)
            if registry is not None:
                return registry.list_tools(only_enabled=True)
        except Exception:
            logger.debug("registered-tools snapshot failed", exc_info=True)
        return None

    @staticmethod
    def _read_core_tools(runtime: Any) -> Any:
        """Minimal read-only core for borderline Q&A turns (tool-supply P3).

        A few STUB-schema read tools (read/code_search/ls — tens of tokens each) so the
        model can look something up if the "question" actually needs a file or search,
        rather than being handed no tools at all. Best-effort: returns ``[]`` if the
        registry is unavailable.
        """
        try:
            from victor.agent.tool_selection import tool_to_definition
            from victor.tools.enums import SchemaLevel

            selector = getattr(runtime, "tool_selector", None)
            registry = getattr(selector, "tools", None)
            if registry is None or not hasattr(registry, "get"):
                return []
            out = []
            for name in _READ_CORE_TOOL_NAMES:
                tool = registry.get(name)
                if tool is not None:
                    out.append(tool_to_definition(tool, SchemaLevel.STUB))
            return out
        except Exception:
            logger.debug("read-core build failed", exc_info=True)
            return []

    def _get_e3tir_reranker(self) -> Any:
        """Lazily build the per-session E3-TIR reranker when enabled, else None.

        Cached on the runtime (sentinel-distinguished from "disabled") so the flag is
        checked once and the exploration-phase schedule persists for the session. Reads
        the process-shared experience store fed by the TOOL_EXECUTED hook.
        """
        runtime = self._runtime
        cached = getattr(runtime, "_e3tir_reranker", "unset")
        if cached != "unset":
            return cached
        reranker = None
        try:
            from victor.core.feature_flags import FeatureFlag, is_feature_enabled

            if is_feature_enabled(FeatureFlag.USE_E3_TIR_EXPLORATION) and is_feature_enabled(
                FeatureFlag.USE_LEARNING_FROM_EXECUTION
            ):
                from victor.tools.e3_tir_selector import E3TIRToolSelector

                reranker = E3TIRToolSelector(store=_get_e3tir_shared_store())
        except Exception:
            logger.debug("E3-TIR reranker unavailable", exc_info=True)
            reranker = None
        runtime._e3tir_reranker = reranker
        return reranker

    def _apply_e3tir_exploration(self, tools: Any, user_message: str) -> Any:
        """Rerank the selected tools via E3-TIR experience-based exploration.

        No-op when the reranker is disabled, or when KV tool-stability applies — E3-TIR's
        per-turn reordering would break the byte-stable tool prefix that KV-prefix-caching
        providers depend on (``_sort_tools_for_kv_stability`` only sorts in that case).
        """
        if not tools:
            return tools
        runtime = self._runtime
        if getattr(runtime, "_kv_optimization_enabled", False):
            return tools
        reranker = self._get_e3tir_reranker()
        if reranker is None:
            return tools
        try:
            by_name = {self._tool_name(t): t for t in tools}
            base_ranking = [name for name in by_name if name]
            if not base_ranking:
                return tools
            reranked = reranker.select(
                available_tools=base_ranking,
                task_type=str(getattr(runtime, "_current_task_type", "") or "general"),
                user_message=user_message or "",
                base_ranking=base_ranking,
                max_tools=len(base_ranking),
            )
            seen = set()
            reordered = []
            for name in reranked:
                if name in by_name and name not in seen:
                    reordered.append(by_name[name])
                    seen.add(name)
            # Preserve any tools E3-TIR dropped/omitted at the tail (never lose tools).
            reordered.extend(tool for name, tool in by_name.items() if name not in seen)
            return reordered
        except Exception:
            logger.debug("E3-TIR rerank failed; using base order", exc_info=True)
            return tools

    @staticmethod
    def _tool_name(tool: Any) -> str:
        """Extract a tool name from dict- or object-style definitions."""
        if hasattr(tool, "name"):
            return str(tool.name or "")
        if isinstance(tool, dict):
            return str(tool.get("name", "") or "")
        return ""

    def _prioritize_explicit_database_tools(
        self,
        tools: Any,
        user_message: str | None,
    ) -> Any:
        """Move db/database and shell to the front for explicit DB inspection requests."""
        if not tools:
            return tools

        from victor.agent.action_authorizer import has_explicit_readonly_shell_request
        from victor.tools.core_tool_aliases import canonicalize_core_tool_name
        from victor.tools.decorators import resolve_tool_name

        if not has_explicit_readonly_shell_request(user_message):
            return tools

        preferred_groups = ({"db", "database"}, {"shell"})
        grouped: list[list[Any]] = [[] for _ in preferred_groups]
        remainder: list[Any] = []

        for tool in tools:
            name = self._tool_name(tool)
            canonical_name = canonicalize_core_tool_name(resolve_tool_name(name))
            for idx, preferred_names in enumerate(preferred_groups):
                if canonical_name in preferred_names:
                    grouped[idx].append(tool)
                    break
            else:
                remainder.append(tool)

        reordered = [tool for group in grouped for tool in group] + remainder
        if reordered != list(tools):
            logger.info(
                "Explicit database request detected: prioritizing database inspection tools "
                "at the front of the candidate list"
            )
        return reordered

    def _current_task_type(self) -> str:
        """Resolve the current turn's task type from a *populated* source.

        The prior guard read ``self._runtime._current_task_type``, an attribute that is
        never assigned anywhere — so it was always empty and the read-oriented guard below
        never fired, forcing edit/write/shell onto pure analysis turns every iteration. The
        UnifiedTaskTracker's ``task_type`` is the canonical per-turn classification
        (e.g. "analyze"/"search"/"research"); fall back to a runtime-level attribute if a
        future path sets one.
        """
        runtime = self._runtime
        tracker = getattr(runtime, "unified_tracker", None)
        for candidate in (
            getattr(tracker, "task_type", None),
            getattr(runtime, "_current_task_type", None),
        ):
            if candidate is None:
                continue
            value = str(getattr(candidate, "value", candidate) or "").lower()
            if value:
                return value
        return ""

    def _ensure_write_tools_for_write_intent(self, tools: Any, current_intent: Any) -> Any:
        """Ensure semantic selection does not omit edit/write on write-authorized turns."""
        if not tools:
            return tools

        from victor.agent.action_authorizer import ActionIntent
        from victor.tools.core_tool_aliases import canonicalize_core_tool_name
        from victor.tools.decorators import resolve_tool_name

        intent_value = getattr(current_intent, "value", current_intent)
        if intent_value != ActionIntent.WRITE_ALLOWED.value:
            return tools

        # WRITE_ALLOWED only means writes aren't blocked — not that the user wants to write.
        # Don't force edit/write/shell onto read-oriented analysis/search/research turns every
        # iteration; let semantic selection decide (the agent can still surface them if needed).
        if self._current_task_type() in _READ_ORIENTED_TASK_TYPES:
            return tools

        selected = list(tools)
        selected_names = {
            canonicalize_core_tool_name(resolve_tool_name(self._tool_name(tool)))
            for tool in selected
            if self._tool_name(tool)
        }
        needed = [name for name in ("edit", "write", "shell") if name not in selected_names]
        if not needed:
            return tools

        available_by_name = self._available_tool_defs_by_name()
        additions = [available_by_name[name] for name in needed if name in available_by_name]
        if not additions:
            return tools

        logger.info(
            "Write intent detected: adding missing mutation tool(s) to selected set: %s",
            [self._tool_name(tool) for tool in additions],
        )
        return selected + additions

    def _available_tool_defs_by_name(self) -> dict[str, Any]:
        """Return available tool definitions keyed by canonical core tool name."""
        from victor.tools.core_tool_aliases import canonicalize_core_tool_name
        from victor.tools.decorators import resolve_tool_name

        registry = getattr(self._runtime, "tools", None) or getattr(
            self._runtime, "tool_registry", None
        )
        if registry is None:
            return {}

        list_tools = getattr(registry, "list_tools", None)
        if not callable(list_tools):
            return {}

        try:
            available = list_tools(only_enabled=True)
        except TypeError:
            available = list_tools()
        except Exception:
            return {}

        by_name: dict[str, Any] = {}
        for tool in available or []:
            name = self._tool_name(tool)
            if not name:
                continue
            canonical = canonicalize_core_tool_name(resolve_tool_name(name))
            by_name.setdefault(canonical, tool)
        return by_name
