from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.tool_execution_runtime import ToolExecutionRuntime
from victor.agent.streaming.context import StreamingChatContext


def _make_runtime_host(**overrides):
    values = {
        "_tool_pipeline": SimpleNamespace(
            execute_tool_calls=AsyncMock(return_value=[{"name": "read", "success": True}]),
            calls_used=4,
        ),
        "_tool_service": MagicMock(),
        "_get_tool_context": MagicMock(return_value={"provider": "mock"}),
        "executed_tools": ["read"],
        "observed_files": {"app.py"},
        "failed_tool_signatures": set(),
        "_shown_tool_errors": set(),
        "_continuation_prompts": 1,
        "_asking_input_prompts": 2,
        "tool_calls_used": 0,
        "_record_tool_execution": MagicMock(),
        "conversation_state": MagicMock(),
        "unified_tracker": MagicMock(),
        "usage_logger": MagicMock(),
        "add_message": MagicMock(),
        "_format_tool_output": MagicMock(),
        "console": MagicMock(),
        "_presentation": MagicMock(),
        "_current_stream_context": MagicMock(),
        "_current_task_type": "analysis",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_tool_execution_runtime_returns_empty_for_empty_calls():
    host = _make_runtime_host()
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    assert await runtime.execute_tool_calls([]) == []


@pytest.mark.asyncio
async def test_tool_execution_runtime_filters_nondict_calls():
    host = _make_runtime_host()
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    assert await runtime.execute_tool_calls(["bad-payload"]) == []


@pytest.mark.asyncio
async def test_tool_execution_runtime_executes_pipeline_and_syncs_mutable_state():
    host = _make_runtime_host()

    def _process_results(pipeline_result, ctx):
        assert pipeline_result == [{"name": "read", "success": True}]
        ctx.continuation_prompts = 5
        ctx.asking_input_prompts = 6
        return [{"name": "read", "success": True, "elapsed": 0.1}]

    host._tool_service.process_tool_results.side_effect = _process_results
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.execute_tool_calls([{"name": "read", "arguments": {}}, "bad-payload"])

    assert result == [{"name": "read", "success": True, "elapsed": 0.1}]
    host._tool_pipeline.execute_tool_calls.assert_awaited_once_with(
        tool_calls=[{"name": "read", "arguments": {}}],
        context={"provider": "mock"},
    )
    assert host.tool_calls_used == 4
    assert host._continuation_prompts == 5
    assert host._asking_input_prompts == 6


@pytest.mark.asyncio
async def test_tool_execution_runtime_backfills_missing_tool_response_ids():
    host = _make_runtime_host(
        conversation=SimpleNamespace(_messages=[]),
    )
    host._tool_service.process_tool_results.return_value = [
        {"name": "read", "success": True, "elapsed": 0.1, "tool_call_id": "call_1"}
    ]
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.execute_tool_calls(
        [
            {"id": "call_1", "name": "read", "arguments": {"path": "victor/framework/graph.py"}},
            {"id": "call_2", "name": "metrics", "arguments": {"path": "victor/framework/graph.py"}},
        ]
    )

    assert len(result) == 2
    assert any(entry.get("tool_call_id") == "call_2" for entry in result)
    host.add_message.assert_called_once_with(
        "tool",
        (
            "Tool result unavailable for 'metrics'. Victor did not complete "
            "post-processing for this tool call, so treat it as failed and continue "
            "with the available context."
        ),
        name="metrics",
        tool_call_id="call_2",
        persist_synchronously=True,
    )


@pytest.mark.asyncio
async def test_tool_execution_runtime_backfill_survives_add_message_failure():
    host = _make_runtime_host(
        conversation=SimpleNamespace(_messages=[]),
    )
    host.add_message.side_effect = RuntimeError("conversation write failed")
    host._tool_service.process_tool_results.return_value = []
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.execute_tool_calls(
        [
            {
                "id": "call_2",
                "name": "metrics",
                "arguments": {"path": "victor/framework/graph.py"},
            },
        ]
    )

    assert len(result) == 1
    assert result[0]["tool_call_id"] == "call_2"
    assert result[0]["outcome_kind"] == "tool_response_missing"


@pytest.mark.asyncio
async def test_tool_execution_runtime_compacts_before_large_tool_output_and_records_ledger():
    stream_ctx = StreamingChatContext(user_message="investigate runtime", total_iterations=2)
    pipeline_result = SimpleNamespace(
        results=[
            SimpleNamespace(
                result={"content": "x" * 9000},
                error=None,
            )
        ]
    )
    context_service = SimpleNamespace(
        prepare_for_tool_output_injection=AsyncMock(
            return_value={
                "should_compact": True,
                "compacted": True,
                "messages_removed": 3,
                "saved_tokens": 120,
                "strategy": "hybrid",
                "reason": "pre_tool_output",
                "policy_reason": "high_utilization_large_tool_output",
            }
        )
    )
    host = _make_runtime_host(
        _tool_pipeline=SimpleNamespace(
            execute_tool_calls=AsyncMock(return_value=pipeline_result),
            calls_used=1,
        ),
        _tool_service=MagicMock(),
        _context_service=context_service,
        settings=SimpleNamespace(context_compaction_strategy="tiered"),
        provider=SimpleNamespace(name="deepseek"),
        model="deepseek-chat",
        _conversation_controller=SimpleNamespace(
            get_compaction_summaries=MagicMock(return_value=["trimmed stale tool chatter"])
        ),
        _current_stream_context=stream_ctx,
    )
    host._tool_service.process_tool_results.return_value = [
        {"name": "read", "success": True, "elapsed": 0.1, "args": {"path": "app.py"}}
    ]

    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    await runtime.execute_tool_calls([{"name": "read", "arguments": {"path": "app.py"}}])

    context_service.prepare_for_tool_output_injection.assert_awaited_once()
    args, kwargs = context_service.prepare_for_tool_output_injection.await_args
    assert args and args[0] >= 2200
    assert kwargs == {
        "provider_name": "deepseek",
        "model_name": "deepseek-chat",
        "task_type": "analysis",
        "min_messages": 6,
        "default_strategy": "tiered",
    }
    assert stream_ctx.compaction_occurred is True
    assert stream_ctx.last_compaction_strategy == "hybrid"
    assert stream_ctx.last_compaction_reason == "pre_tool_output"
    assert stream_ctx.last_compaction_policy_reason == "high_utilization_large_tool_output"
    assert any(event["kind"] == "tool_intent" for event in stream_ctx.intent_log)
    assert any(event["kind"] == "tool_result" for event in stream_ctx.intent_log)
