"""Tests for ToolServiceAdapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.adapters.tool_adapter import ToolServiceAdapter


@pytest.fixture
def mock_tool_service():
    service = SimpleNamespace(
        get_available_tools=MagicMock(return_value={"read", "write", "grep"}),
        get_enabled_tools=MagicMock(return_value={"read", "grep"}),
        set_enabled_tools=MagicMock(),
        is_tool_enabled=MagicMock(return_value=True),
        resolve_tool_alias=MagicMock(return_value="read"),
        execute_tool_with_retry=AsyncMock(return_value=("result", True, None)),
        parse_and_validate_tool_calls=MagicMock(return_value=([{"name": "read"}], "")),
        normalize_tool_arguments=MagicMock(return_value=({"path": "a.py"}, "direct")),
        process_tool_results=MagicMock(return_value=[{"name": "read", "success": True}]),
        on_tool_complete=MagicMock(),
        build_tool_access_context=MagicMock(return_value=MagicMock(name="tool_access_context")),
    )
    return service


@pytest.fixture
def mock_tool_coordinator():
    coordinator = SimpleNamespace(
        get_available_tools=MagicMock(return_value={"fallback"}),
        get_enabled_tools=MagicMock(return_value={"fallback"}),
        set_enabled_tools=MagicMock(),
        is_tool_enabled=MagicMock(return_value=False),
        resolve_tool_alias=MagicMock(return_value="fallback"),
        execute_tool_with_retry=AsyncMock(return_value=("fallback", False, "err")),
        parse_and_validate_tool_calls=MagicMock(return_value=(None, "fallback")),
        validate_tool_call=MagicMock(return_value=SimpleNamespace(valid=True)),
        normalize_arguments_full=MagicMock(return_value=SimpleNamespace(args={"path": "a.py"})),
        normalize_tool_arguments=MagicMock(return_value=({"path": "fallback.py"}, "direct")),
        process_tool_results=MagicMock(return_value=[{"name": "fallback", "success": False}]),
        on_tool_complete=MagicMock(),
        build_tool_access_context=MagicMock(return_value=MagicMock(name="legacy_context")),
    )
    return coordinator


@pytest.fixture
def tool_adapter(mock_tool_service, mock_tool_coordinator):
    return ToolServiceAdapter(
        tool_service=mock_tool_service,
        deprecated_tool_coordinator=mock_tool_coordinator,
    )


def test_get_available_tools_prefers_service(tool_adapter, mock_tool_service, mock_tool_coordinator):
    result = tool_adapter.get_available_tools()

    mock_tool_service.get_available_tools.assert_called_once()
    mock_tool_coordinator.get_available_tools.assert_not_called()
    assert result == {"read", "write", "grep"}


def test_get_enabled_tools_prefers_service(tool_adapter, mock_tool_service, mock_tool_coordinator):
    result = tool_adapter.get_enabled_tools()

    mock_tool_service.get_enabled_tools.assert_called_once()
    mock_tool_coordinator.get_enabled_tools.assert_not_called()
    assert result == {"read", "grep"}


def test_set_enabled_tools_prefers_service(tool_adapter, mock_tool_service, mock_tool_coordinator):
    tool_adapter.set_enabled_tools({"read"})

    mock_tool_service.set_enabled_tools.assert_called_once_with({"read"})
    mock_tool_coordinator.set_enabled_tools.assert_not_called()


def test_is_tool_enabled_prefers_service(tool_adapter, mock_tool_service, mock_tool_coordinator):
    assert tool_adapter.is_tool_enabled("read") is True

    mock_tool_service.is_tool_enabled.assert_called_once_with("read")
    mock_tool_coordinator.is_tool_enabled.assert_not_called()


def test_resolve_tool_alias_prefers_service(tool_adapter, mock_tool_service, mock_tool_coordinator):
    assert tool_adapter.resolve_tool_alias("cat") == "read"

    mock_tool_service.resolve_tool_alias.assert_called_once_with("cat")
    mock_tool_coordinator.resolve_tool_alias.assert_not_called()


@pytest.mark.asyncio
async def test_execute_tool_with_retry_prefers_service(
    tool_adapter,
    mock_tool_service,
    mock_tool_coordinator,
):
    result, success, err = await tool_adapter.execute_tool_with_retry(
        "read",
        {"path": "/tmp/file"},
        {},
    )

    mock_tool_service.execute_tool_with_retry.assert_awaited_once()
    mock_tool_coordinator.execute_tool_with_retry.assert_not_awaited()
    assert (result, success, err) == ("result", True, None)


def test_parse_and_validate_tool_calls_prefers_service(
    tool_adapter,
    mock_tool_service,
    mock_tool_coordinator,
):
    result = tool_adapter.parse_and_validate_tool_calls([{"name": "read"}], "content", MagicMock())

    mock_tool_service.parse_and_validate_tool_calls.assert_called_once()
    mock_tool_coordinator.parse_and_validate_tool_calls.assert_not_called()
    assert result == ([{"name": "read"}], "")


def test_validate_tool_call_falls_back_to_coordinator_when_service_lacks_method(
    tool_adapter,
    mock_tool_coordinator,
):
    validation = tool_adapter.validate_tool_call(MagicMock(), MagicMock())

    mock_tool_coordinator.validate_tool_call.assert_called_once()
    assert validation.valid is True


def test_normalize_arguments_full_falls_back_to_coordinator_when_service_lacks_method(
    tool_adapter,
    mock_tool_coordinator,
):
    normalized = tool_adapter.normalize_arguments_full(
        "read",
        "read",
        {"path": "a.py"},
        MagicMock(),
        MagicMock(),
    )

    mock_tool_coordinator.normalize_arguments_full.assert_called_once()
    assert normalized.args == {"path": "a.py"}


def test_normalize_tool_arguments_prefers_service(
    tool_adapter,
    mock_tool_service,
    mock_tool_coordinator,
):
    result = tool_adapter.normalize_tool_arguments({"path": "a.py"}, "read")

    mock_tool_service.normalize_tool_arguments.assert_called_once_with({"path": "a.py"}, "read")
    mock_tool_coordinator.normalize_tool_arguments.assert_not_called()
    assert result == ({"path": "a.py"}, "direct")


def test_process_tool_results_prefers_service(tool_adapter, mock_tool_service, mock_tool_coordinator):
    result = tool_adapter.process_tool_results(MagicMock(), MagicMock())

    mock_tool_service.process_tool_results.assert_called_once()
    mock_tool_coordinator.process_tool_results.assert_not_called()
    assert result == [{"name": "read", "success": True}]


def test_on_tool_complete_prefers_service(tool_adapter, mock_tool_service, mock_tool_coordinator):
    result = MagicMock()

    tool_adapter.on_tool_complete(
        result=result,
        metrics_collector=MagicMock(),
        read_files_session=set(),
        required_files=["a.py"],
        required_outputs=["summary"],
        nudge_sent_flag=[False],
        add_message=MagicMock(),
        observability=MagicMock(),
        pipeline_calls_used=2,
    )

    mock_tool_service.on_tool_complete.assert_called_once()
    mock_tool_coordinator.on_tool_complete.assert_not_called()


def test_build_tool_access_context_prefers_service(
    tool_adapter,
    mock_tool_service,
    mock_tool_coordinator,
):
    result = tool_adapter.build_tool_access_context()

    mock_tool_service.build_tool_access_context.assert_called_once()
    mock_tool_coordinator.build_tool_access_context.assert_not_called()
    assert result is mock_tool_service.build_tool_access_context.return_value


def test_build_tool_access_context_falls_back_to_public_coordinator_method(mock_tool_coordinator):
    with pytest.warns(
        DeprecationWarning,
        match="coordinator fallback only",
    ):
        adapter = ToolServiceAdapter(
            tool_service=None,
            deprecated_tool_coordinator=mock_tool_coordinator,
        )

    result = adapter.build_tool_access_context()

    mock_tool_coordinator.build_tool_access_context.assert_called_once()
    assert result is mock_tool_coordinator.build_tool_access_context.return_value


def test_old_tool_coordinator_kwarg_warns(mock_tool_service, mock_tool_coordinator):
    with pytest.warns(
        DeprecationWarning,
        match="ToolServiceAdapter\\(tool_coordinator=...\\) is deprecated",
    ):
        adapter = ToolServiceAdapter(
            tool_service=mock_tool_service,
            tool_coordinator=mock_tool_coordinator,
        )

    assert adapter.get_available_tools() == {"read", "write", "grep"}


def test_old_and_new_tool_coordinator_kwargs_conflict(mock_tool_service, mock_tool_coordinator):
    with pytest.raises(
        TypeError,
        match="Use only one of tool_coordinator or deprecated_tool_coordinator",
    ):
        ToolServiceAdapter(
            tool_service=mock_tool_service,
            tool_coordinator=mock_tool_coordinator,
            deprecated_tool_coordinator=mock_tool_coordinator,
        )


def test_is_healthy(tool_adapter):
    assert tool_adapter.is_healthy() is True


def test_is_healthy_with_none():
    adapter = ToolServiceAdapter(tool_service=None)
    assert adapter.is_healthy() is False


def test_legacy_positional_coordinator_is_still_supported(mock_tool_coordinator):
    with pytest.warns(
        DeprecationWarning,
        match="Positional ToolServiceAdapter\\(\\.\\.\\.\\) construction is deprecated",
    ):
        adapter = ToolServiceAdapter(mock_tool_coordinator)

    assert adapter.get_available_tools() == {"fallback"}
    mock_tool_coordinator.get_available_tools.assert_called_once()
