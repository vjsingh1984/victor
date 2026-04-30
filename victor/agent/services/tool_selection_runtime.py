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

    async def select_tools_for_turn(self, context_msg: str, goals: Any) -> Any:
        """Select, prioritize, and filter tools for the current turn."""
        runtime = self._runtime
        provider_supports_tools = runtime.provider.supports_tools()
        tooling_allowed = provider_supports_tools and runtime._model_supports_tool_calls()

        if not tooling_allowed:
            return None

        if runtime._should_skip_tools_for_turn(context_msg):
            return None

        planned_tools = None
        if goals:
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
            context_msg,
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
        tools = runtime.tool_selector.prioritize_by_stage(context_msg, tools)
        current_intent = getattr(runtime, "_current_intent", None)
        tools = runtime._tool_planner.filter_tools_by_intent(
            tools,
            current_intent,
            user_message=context_msg,
        )
        tools = runtime._apply_kv_tool_strategy(tools)
        return runtime._sort_tools_for_kv_stability(tools)
