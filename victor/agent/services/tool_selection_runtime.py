# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned tool selection runtime helper."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
        user_message_anchor = getattr(runtime, "_current_user_message", None) or context_msg

        if not tooling_allowed:
            return None

        if runtime._should_skip_tools_for_turn(context_msg):
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
        selection_context = context_msg
        if user_message_anchor and user_message_anchor != context_msg:
            selection_context = f"{user_message_anchor}\n\n" f"Current working step: {context_msg}"
        tools = await runtime.tool_selector.select_tools(
            selection_context,
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
        tools = runtime._apply_kv_tool_strategy(tools)
        return runtime._sort_tools_for_kv_stability(tools)

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
