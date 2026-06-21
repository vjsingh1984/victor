# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned tool selection runtime helper."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Read-oriented task types: WRITE_ALLOWED is a *permission*, not an intent to write, so we
# do not force mutation tools (edit/write/shell) onto these pure analysis/search turns.
_READ_ORIENTED_TASK_TYPES = frozenset({"analyze", "search", "research"})


# --- E3-TIR experience-replay exploration (opt-in via USE_E3_TIR_EXPLORATION) ---------
# The experience STORE holds user-global RL data, so it is shared across sessions and fed
# by a single idempotent TOOL_EXECUTED hook. The per-session SELECTOR (which owns the
# exploration-phase warm-up schedule) reads that shared store. Keeping the store shared
# but the selector per-session avoids cross-session contamination while still letting each
# session run its own demonstration -> self-play -> exploration ramp.
_e3tir_shared_store: Any = None
_e3tir_hook_subscribed: bool = False


def _get_e3tir_shared_store() -> Any:
    """Return the process-shared ToolExperienceStore, subscribing the outcome hook once."""
    global _e3tir_shared_store, _e3tir_hook_subscribed
    if _e3tir_shared_store is None:
        from victor.tools.experience_store import ToolExperienceStore

        _e3tir_shared_store = ToolExperienceStore()
    if not _e3tir_hook_subscribed:
        from victor.framework.rl.hooks import RLEventType, get_rl_hooks

        store = _e3tir_shared_store

        def _record(event: Any) -> None:
            try:
                tool = getattr(event, "tool_name", None)
                if not tool:
                    return
                success = event.success if event.success is not None else True
                reward = (
                    float(event.quality_score)
                    if event.quality_score is not None
                    else (1.0 if success else 0.3)
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

        if not tooling_allowed:
            return None

        if runtime._should_skip_tools_for_turn(user_message_anchor):
            return None

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
        logger.info(
            "context_msg=%s\nuse_semantic=%s\nconversation_depth=%s",
            context_msg,
            runtime.use_semantic_selection,
            conversation_depth,
        )
        # Stage prioritization should stay anchored to the user's request.
        # Feeding assistant progress narration here can wrongly push the
        # conversation state machine back toward analysis/search-heavy tools.
        tools = runtime.tool_selector.prioritize_by_stage(user_message_anchor, tools)
        current_intent = getattr(runtime, "_current_intent", None)
        tools = runtime._tool_planner.filter_tools_by_intent(
            tools,
            current_intent,
            user_message=user_message_anchor,
        )
        tools = self._ensure_write_tools_for_write_intent(tools, current_intent)
        tools = self._prioritize_explicit_database_tools(tools, user_message_anchor)
        tools = self._apply_e3tir_exploration(tools, user_message_anchor)
        tools = runtime._apply_kv_tool_strategy(tools)
        return runtime._sort_tools_for_kv_stability(tools)

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
        task_type_raw = getattr(self._runtime, "_current_task_type", None)
        task_type = str(getattr(task_type_raw, "value", task_type_raw) or "").lower()
        if task_type in _READ_ORIENTED_TASK_TYPES:
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
