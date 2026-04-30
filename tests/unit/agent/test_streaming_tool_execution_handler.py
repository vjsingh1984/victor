from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.tool_execution import (
    ToolExecutionHandler,
    create_tool_execution_handler,
)
from victor.providers.base import StreamChunk


def test_factory_prefers_canonical_recovery_context_builder():
    def canonical(stream_ctx):
        return {"stream_ctx": stream_ctx}

    orchestrator = SimpleNamespace(
        create_recovery_context=canonical,
        _recovery_service=None,
        _recovery_coordinator=MagicMock(),
        _chunk_generator=MagicMock(),
        reminder_manager=MagicMock(),
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=MagicMock(),
        execute_tool_calls=MagicMock(),
        observed_files=set(),
    )

    handler = create_tool_execution_handler(orchestrator)

    assert handler._recovery_context_factory is canonical
    assert handler._recovery_context_factory("ctx") == {"stream_ctx": "ctx"}


@pytest.mark.asyncio
async def test_execute_tools_invokes_callback_without_signature_collision():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="done")]),
    )
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value=None),
    )
    execute_tool_calls = AsyncMock(
        return_value=[{"name": "read", "success": True, "args": {}, "elapsed": 1.0}]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=SimpleNamespace(add_message=MagicMock()),
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Run the read tool")
    tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/orchestrator.py"}}]

    result = await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=tool_calls,
        user_message="Run the read tool",
        full_content="read the file",
        tool_calls_used=0,
        tool_budget=5,
    )

    execute_tool_calls.assert_awaited_once_with(tool_calls)
    assert result.tool_calls_executed == 1
    assert [chunk.content for chunk in result.chunks] == ["start", "done"]
    assert stream_ctx.executed_tool_names == {"read"}


@pytest.mark.asyncio
async def test_execute_tools_marks_system_reminders_as_noninteractive_history():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="done")]),
    )
    message_adder = SimpleNamespace(add_message=MagicMock())
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value="use more tools"),
    )
    execute_tool_calls = AsyncMock(
        return_value=[{"name": "read", "success": True, "args": {}, "elapsed": 1.0}]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=message_adder,
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Run the read tool")
    tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/orchestrator.py"}}]

    await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=tool_calls,
        user_message="Run the read tool",
        full_content="read the file",
        tool_calls_used=0,
        tool_budget=5,
    )

    message_adder.add_message.assert_any_call(
        "user",
        "[SYSTEM-REMINDER: use more tools]",
        metadata=build_internal_history_metadata("system_reminder"),
    )


@pytest.mark.asyncio
async def test_execute_tools_forces_completion_after_terminal_skips():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="blocked")]),
    )
    message_adder = SimpleNamespace(add_message=MagicMock())
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value=None),
    )
    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "name": "read",
                "success": False,
                "skipped": True,
                "outcome_kind": "budget_exhausted",
            }
        ]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=message_adder,
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Run the read tool")

    await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=[{"name": "read", "arguments": {"path": "victor/agent/orchestrator.py"}}],
        user_message="Run the read tool",
        full_content="read the file",
        tool_calls_used=0,
        tool_budget=5,
    )

    assert stream_ctx.force_completion is True
    message_adder.add_message.assert_any_call(
        "user",
        (
            "[SYSTEM: Tool execution is no longer making progress. "
            "Do not request more blocked tools; summarize the current findings or explain "
            "the blocker directly.]"
        ),
        metadata=build_internal_history_metadata("force_completion"),
    )


@pytest.mark.asyncio
async def test_execute_tools_records_shell_alias_canonically():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="done")]),
    )
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value=None),
    )
    execute_tool_calls = AsyncMock(
        return_value=[{"name": "bash", "success": True, "args": {}, "elapsed": 1.0}]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=SimpleNamespace(add_message=MagicMock()),
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Run bash")

    await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=[{"name": "bash", "arguments": {"cmd": "sqlite3 data.db \".tables\""}}],
        user_message="Run bash",
        full_content="inspect the db",
        tool_calls_used=0,
        tool_budget=5,
    )

    assert stream_ctx.executed_tool_names == {"shell"}
