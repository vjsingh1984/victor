"""Focused tests for service runtime parity helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.session_service import SessionService
from victor.agent.services.provider_service import ProviderService
from victor.agent.services.recovery_service import RecoveryService
from victor.agent.services.tool_service import ToolService, ToolServiceConfig


def _make_tool_service() -> ToolService:
    return ToolService(
        config=ToolServiceConfig(),
        tool_selector=MagicMock(),
        tool_executor=MagicMock(),
        tool_registrar=MagicMock(),
    )


@pytest.mark.asyncio
async def test_tool_service_execute_tool_with_retry_uses_bound_retry_executor():
    service = _make_tool_service()
    retry_executor = MagicMock()
    retry_executor.execute_tool_with_retry = AsyncMock(return_value=("result", True, None))
    service.bind_runtime_components(retry_executor=retry_executor)

    result = await service.execute_tool_with_retry("read", {"path": "a.py"}, {"task_type": "read"})

    retry_executor.execute_tool_with_retry.assert_awaited_once()
    assert result == ("result", True, None)


def test_tool_service_parse_and_validate_tool_calls_matches_runtime_contract():
    service = _make_tool_service()
    parser = MagicMock()
    parser.normalize_args.side_effect = lambda _name, args: args
    service.bind_runtime_components(tool_call_parser=parser)
    service.set_enabled_tools({"read"})

    tool_call = SimpleNamespace(to_dict=lambda: {"name": "read", "arguments": "{\"path\": \"a.py\"}"})
    parse_result = SimpleNamespace(
        tool_calls=[tool_call],
        warnings=[],
        remaining_content="trimmed content",
    )
    tool_adapter = MagicMock()
    tool_adapter.parse_tool_calls.return_value = parse_result

    tool_calls, remaining = service.parse_and_validate_tool_calls(None, "content", tool_adapter)

    assert tool_calls == [{"name": "read", "arguments": {"path": "a.py"}}]
    assert remaining == "trimmed content"


def test_tool_service_build_tool_access_context_uses_bound_mode_controller():
    service = _make_tool_service()
    mode_controller = SimpleNamespace(config=SimpleNamespace(name="review"))
    service.bind_runtime_components(mode_controller=mode_controller)
    service.set_enabled_tools({"read", "grep"})

    context = service.build_tool_access_context()

    assert context.session_enabled_tools == {"read", "grep"}
    assert context.current_mode == "review"


@pytest.mark.asyncio
async def test_session_service_save_and_restore_checkpoint_round_trip():
    session_state = SimpleNamespace(
        tool_calls_used=7,
        observed_files={"a.py"},
        get_token_usage=lambda: {"input": 12, "output": 8},
    )
    checkpoint_manager = MagicMock()
    checkpoint_manager.save_checkpoint = AsyncMock(return_value="ckpt-123")
    checkpoint_manager.restore_checkpoint = AsyncMock(
        return_value={
            "session_id": "mem-1",
            "tool_calls_used": 3,
            "token_usage": {"input": 20, "output": 10},
            "observed_files": ["b.py"],
        }
    )

    service = SessionService(
        session_state_manager=session_state,
        memory_manager=None,
        checkpoint_manager=checkpoint_manager,
    )
    service._memory_session_id = "mem-1"

    checkpoint_id = await service.save_checkpoint("before fix", ["manual"])
    restored = await service.restore_checkpoint("ckpt-123")

    checkpoint_manager.save_checkpoint.assert_awaited_once()
    checkpoint_manager.restore_checkpoint.assert_awaited_once_with("ckpt-123")
    assert checkpoint_id == "ckpt-123"
    assert restored is True
    assert service._session_state.tool_calls_used == 3
    assert service._session_state.observed_files == {"b.py"}


@pytest.mark.asyncio
async def test_provider_service_switch_provider_uses_bound_provider_manager():
    manager = MagicMock()
    manager.switch_provider = AsyncMock(return_value=True)
    manager.provider = SimpleNamespace(name="openai", model="gpt-4.1", max_tokens=200000)
    manager.provider_name = "openai"
    manager.model = "gpt-4.1"
    manager.switch_count = 2

    service = ProviderService(registry=MagicMock())
    service.bind_runtime_components(provider_manager=manager)

    await service.switch_provider("openai", "gpt-4.1")

    manager.switch_provider.assert_awaited_once_with("openai", "gpt-4.1")
    assert service.provider_name == "openai"
    assert service.model == "gpt-4.1"
    assert service.get_current_provider() is manager.provider


@pytest.mark.asyncio
async def test_recovery_service_streaming_methods_use_bound_recovery_coordinator():
    service = RecoveryService()
    coordinator = MagicMock()
    coordinator.handle_recovery_with_integration = AsyncMock(
        return_value=SimpleNamespace(action="continue")
    )
    coordinator.apply_recovery_action.return_value = "applied"
    coordinator.check_natural_completion.return_value = "done"
    coordinator.handle_empty_response.return_value = ("chunk", True)
    coordinator.get_recovery_fallback_message.return_value = "fallback"
    coordinator.check_tool_budget.return_value = "warn"
    coordinator.truncate_tool_calls.return_value = ([{"name": "read"}], True)
    coordinator.filter_blocked_tool_calls.return_value = ([{"name": "read"}], [], 0)
    coordinator.check_blocked_threshold.return_value = None
    coordinator.check_force_action.return_value = (False, None)

    service.bind_runtime_components(recovery_coordinator=coordinator)

    ctx = SimpleNamespace()

    recovery_action = await service.handle_recovery_with_integration(
        ctx,
        "content",
        [{"name": "read"}],
    )

    assert recovery_action.action == "continue"
    coordinator.handle_recovery_with_integration.assert_awaited_once()
    assert service.apply_recovery_action("action", ctx) == "applied"
    assert service.check_natural_completion(ctx, False, 0) == "done"
    assert service.handle_empty_response(ctx) == ("chunk", True)
    assert service.get_recovery_fallback_message(ctx) == "fallback"
    assert service.check_tool_budget(ctx, 10) == "warn"
    assert service.truncate_tool_calls(ctx, [{"name": "read"}], 1) == (
        [{"name": "read"}],
        True,
    )
    assert service.filter_blocked_tool_calls(ctx, [{"name": "read"}]) == (
        [{"name": "read"}],
        [],
        0,
    )
    assert service.check_blocked_threshold(ctx, False) is None
    assert service.check_force_action(ctx) == (False, None)
