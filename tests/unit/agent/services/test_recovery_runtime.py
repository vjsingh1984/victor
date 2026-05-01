from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.recovery_runtime import RecoveryRuntime
from victor.providers.base import StreamChunk


def _make_stream_ctx(**overrides):
    values = {
        "total_iterations": 4,
        "max_total_iterations": 12,
        "last_quality_score": 0.85,
        "unified_task_type": "analysis",
        "is_analysis_task": True,
        "is_action_task": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_runtime_host(**overrides):
    values = {
        "_streaming_controller": SimpleNamespace(current_session=SimpleNamespace(start_time=100.0)),
        "_recovery_service": MagicMock(),
        "tool_calls_used": 3,
        "tool_budget": 10,
        "provider_name": "openai",
        "model": "gpt-4.1",
        "temperature": 0.2,
        "add_message": MagicMock(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_recovery_runtime_create_recovery_context_uses_runtime_state():
    host = _make_runtime_host()
    runtime = RecoveryRuntime(OrchestratorProtocolAdapter(host))
    stream_ctx = _make_stream_ctx()

    with patch("victor.agent.services.recovery_runtime.time.time", return_value=130.0):
        recovery_ctx = runtime.create_recovery_context(stream_ctx)

    assert recovery_ctx.iteration == 4
    assert recovery_ctx.elapsed_time == 30.0
    assert recovery_ctx.tool_calls_used == 3
    assert recovery_ctx.tool_budget == 10
    assert recovery_ctx.session_start_time == 100.0
    assert recovery_ctx.provider_name == "openai"
    assert recovery_ctx.model == "gpt-4.1"
    assert recovery_ctx.temperature == 0.2


def test_recovery_runtime_create_recovery_context_without_session_uses_now():
    host = _make_runtime_host(
        _streaming_controller=SimpleNamespace(current_session=None),
    )
    runtime = RecoveryRuntime(OrchestratorProtocolAdapter(host))
    stream_ctx = _make_stream_ctx()

    with patch("victor.agent.services.recovery_runtime.time.time", return_value=250.0):
        recovery_ctx = runtime.create_recovery_context(stream_ctx)

    assert recovery_ctx.elapsed_time == 0.0
    assert recovery_ctx.session_start_time == 250.0


async def test_recovery_runtime_handle_recovery_with_integration_delegates_to_service():
    host = _make_runtime_host()
    host._recovery_service.handle_recovery_with_integration = AsyncMock(
        return_value=SimpleNamespace(action="continue")
    )
    runtime = RecoveryRuntime(OrchestratorProtocolAdapter(host))
    stream_ctx = _make_stream_ctx()

    result = await runtime.handle_recovery_with_integration(
        stream_ctx,
        "response",
        [{"name": "read_file"}],
        ["read_file"],
    )

    assert result.action == "continue"
    host._recovery_service.handle_recovery_with_integration.assert_awaited_once()
    args = host._recovery_service.handle_recovery_with_integration.await_args.args
    assert args[0].streaming_context is stream_ctx
    assert args[1] == "response"
    assert args[2] == [{"name": "read_file"}]
    assert args[3] == ["read_file"]
    kwargs = host._recovery_service.handle_recovery_with_integration.await_args.kwargs
    assert callable(kwargs["message_adder"])


def test_recovery_runtime_apply_recovery_action_delegates_to_service():
    host = _make_runtime_host()
    host._recovery_service.apply_recovery_action = MagicMock(
        return_value=StreamChunk(content="Recovered", is_final=True)
    )
    runtime = RecoveryRuntime(OrchestratorProtocolAdapter(host))
    stream_ctx = _make_stream_ctx()
    recovery_action = SimpleNamespace(action="abort")

    result = runtime.apply_recovery_action(recovery_action, stream_ctx)

    assert result.content == "Recovered"
    host._recovery_service.apply_recovery_action.assert_called_once()
    args = host._recovery_service.apply_recovery_action.call_args.args
    assert args[0] is recovery_action
    assert args[1].streaming_context is stream_ctx
    kwargs = host._recovery_service.apply_recovery_action.call_args.kwargs
    assert callable(kwargs["message_adder"])
