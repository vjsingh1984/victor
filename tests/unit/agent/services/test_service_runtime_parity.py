"""Focused tests for service runtime parity helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.session_service import SessionService
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
