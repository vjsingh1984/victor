# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned tool execution runtime helper."""

from __future__ import annotations

from typing import Any, Dict, List


class ToolExecutionRuntime:
    """Bridge orchestrator tool pipeline state into the canonical tool service."""

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and post-process results through the canonical services."""
        runtime = self._runtime
        if not tool_calls:
            return []

        tool_calls = [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]
        if not tool_calls:
            return []

        pipeline_result = await runtime._tool_pipeline.execute_tool_calls(
            tool_calls=tool_calls,
            context=runtime._get_tool_context(),
        )
        runtime.tool_calls_used = runtime._tool_pipeline.calls_used

        from victor.agent.services.tool_service import ToolResultContext

        ctx = ToolResultContext(
            executed_tools=runtime.executed_tools,
            observed_files=runtime.observed_files,
            failed_tool_signatures=runtime.failed_tool_signatures,
            shown_tool_errors=runtime._shown_tool_errors,
            continuation_prompts=runtime._continuation_prompts,
            asking_input_prompts=runtime._asking_input_prompts,
            tool_calls_used=runtime.tool_calls_used,
            record_tool_execution=runtime._record_tool_execution,
            conversation_state=runtime.conversation_state,
            unified_tracker=runtime.unified_tracker,
            usage_logger=runtime.usage_logger,
            add_message=runtime.add_message,
            format_tool_output=runtime._format_tool_output,
            console=runtime.console,
            presentation=runtime._presentation,
            stream_context=(
                runtime._current_stream_context
                if hasattr(runtime, "_current_stream_context")
                else None
            ),
            task_type=getattr(
                runtime, "_current_task_type", getattr(runtime, "_task_type", "unknown")
            ),
        )

        results = runtime._tool_service.process_tool_results(pipeline_result, ctx)
        runtime._continuation_prompts = ctx.continuation_prompts
        runtime._asking_input_prompts = ctx.asking_input_prompts
        return results
