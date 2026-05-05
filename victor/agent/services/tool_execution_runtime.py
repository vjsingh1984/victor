# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned tool execution runtime helper."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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
        results = self._ensure_tool_response_coverage(runtime, tool_calls, results)
        runtime._continuation_prompts = ctx.continuation_prompts
        runtime._asking_input_prompts = ctx.asking_input_prompts
        return results

    @staticmethod
    def _ensure_tool_response_coverage(
        runtime: Any,
        tool_calls: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Backfill missing provider-visible tool responses for dropped call IDs."""
        expected_by_id = {
            tool_call.get("id"): tool_call
            for tool_call in tool_calls
            if isinstance(tool_call, dict) and tool_call.get("id")
        }
        if not expected_by_id:
            return results

        observed_result_ids = {
            result.get("tool_call_id")
            for result in results
            if isinstance(result, dict) and result.get("tool_call_id")
        }

        conversation = getattr(runtime, "conversation", None)
        conversation_messages = getattr(conversation, "_messages", None)
        message_by_id: Dict[str, Any] = {}
        if conversation_messages:
            for message in conversation_messages:
                if getattr(message, "role", None) != "tool":
                    continue
                tool_call_id = getattr(message, "tool_call_id", None)
                if tool_call_id:
                    message_by_id[tool_call_id] = message

        missing_ids = [
            tool_call_id
            for tool_call_id in expected_by_id
            if tool_call_id not in observed_result_ids and tool_call_id not in message_by_id
        ]
        if missing_ids:
            logger.error(
                "Tool response coverage gap detected for %d call(s): %s",
                len(missing_ids),
                missing_ids,
            )

        for tool_call_id in missing_ids:
            tool_call = expected_by_id[tool_call_id]
            tool_name = tool_call.get("name", "tool")
            tool_args = tool_call.get("arguments", {}) or {}
            fallback_content = (
                f"Tool result unavailable for '{tool_name}'. Victor did not complete "
                "post-processing for this tool call, so treat it as failed and continue "
                "with the available context."
            )
            try:
                runtime.add_message(
                    "tool",
                    fallback_content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    persist_synchronously=True,
                )
            except TypeError:
                try:
                    runtime.add_message(
                        "tool",
                        fallback_content,
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist fallback tool response for %s (tool_call_id=%s)",
                        tool_name,
                        tool_call_id,
                    )
            except Exception:
                logger.exception(
                    "Failed to persist fallback tool response for %s (tool_call_id=%s)",
                    tool_name,
                    tool_call_id,
                )
            results.append(
                {
                    "name": tool_name,
                    "success": False,
                    "elapsed": 0.0,
                    "args": tool_args,
                    "error": "Tool response coverage gap",
                    "result": fallback_content,
                    "full_result": fallback_content,
                    "follow_up_suggestions": None,
                    "was_pruned": False,
                    "tool_call_id": tool_call_id,
                    "content": fallback_content,
                    "skipped": True,
                    "outcome_kind": "tool_response_missing",
                    "block_source": "tool_result_postprocessing",
                    "retryable": False,
                    "user_message": fallback_content,
                }
            )

        for tool_call_id, message in message_by_id.items():
            if tool_call_id in observed_result_ids or tool_call_id not in expected_by_id:
                continue
            message_content = getattr(message, "content", "")
            results.append(
                {
                    "name": getattr(message, "tool_name", None)
                    or getattr(message, "name", None)
                    or expected_by_id[tool_call_id].get("name", "tool"),
                    "success": False,
                    "elapsed": 0.0,
                    "args": expected_by_id[tool_call_id].get("arguments", {}) or {},
                    "error": "Tool response recovered from conversation history",
                    "result": message_content,
                    "full_result": message_content,
                    "follow_up_suggestions": None,
                    "was_pruned": False,
                    "tool_call_id": tool_call_id,
                    "content": message_content,
                    "skipped": True,
                    "outcome_kind": "tool_response_recovered",
                    "block_source": "conversation_history",
                    "retryable": False,
                    "user_message": message_content,
                }
            )

        return results
