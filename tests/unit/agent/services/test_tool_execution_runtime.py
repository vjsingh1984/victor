from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.tool_execution_runtime import ToolExecutionRuntime


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
