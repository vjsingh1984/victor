"""Tests for ToolServiceAdapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.adapters.tool_adapter import ToolServiceAdapter


@pytest.fixture
def mock_tool_coordinator():
    coordinator = MagicMock()
    coordinator.get_available_tools.return_value = {"read", "write", "grep"}
    coordinator.get_enabled_tools.return_value = {"read", "grep"}
    coordinator.is_tool_enabled.return_value = True
    coordinator.execute_tool_with_retry = AsyncMock(return_value=("result", True, None))
    coordinator.parse_and_validate_tool_calls.return_value = ([{"name": "read"}], "")
    coordinator.validate_tool_call.return_value = MagicMock(valid=True)
    coordinator.normalize_arguments_full.return_value = MagicMock()
    coordinator._build_tool_access_context.return_value = MagicMock()
    return coordinator


@pytest.fixture
def tool_adapter(mock_tool_coordinator):
    return ToolServiceAdapter(mock_tool_coordinator)


def test_get_available_tools(tool_adapter, mock_tool_coordinator):
    result = tool_adapter.get_available_tools()
    mock_tool_coordinator.get_available_tools.assert_called_once()
    assert result == {"read", "write", "grep"}


def test_get_enabled_tools(tool_adapter, mock_tool_coordinator):
    result = tool_adapter.get_enabled_tools()
    mock_tool_coordinator.get_enabled_tools.assert_called_once()
    assert result == {"read", "grep"}


def test_set_enabled_tools(tool_adapter, mock_tool_coordinator):
    tool_adapter.set_enabled_tools({"read"})
    mock_tool_coordinator.set_enabled_tools.assert_called_once_with({"read"})


def test_is_tool_enabled(tool_adapter, mock_tool_coordinator):
    assert tool_adapter.is_tool_enabled("read") is True
    mock_tool_coordinator.is_tool_enabled.assert_called_once_with("read")


async def test_execute_tool_with_retry(tool_adapter, mock_tool_coordinator):
    result, success, err = await tool_adapter.execute_tool_with_retry(
        "read", {"path": "/tmp/file"}, {}
    )
    mock_tool_coordinator.execute_tool_with_retry.assert_awaited_once()
    assert success is True
    assert result == "result"


def test_parse_and_validate_tool_calls(tool_adapter, mock_tool_coordinator):
    result = tool_adapter.parse_and_validate_tool_calls([{"name": "read"}], "content", MagicMock())
    mock_tool_coordinator.parse_and_validate_tool_calls.assert_called_once()
    assert result == ([{"name": "read"}], "")


def test_validate_tool_call(tool_adapter, mock_tool_coordinator):
    validation = tool_adapter.validate_tool_call(MagicMock(), MagicMock())
    assert validation.valid is True


def test_is_healthy(tool_adapter):
    assert tool_adapter.is_healthy() is True


def test_is_healthy_with_none():
    adapter = ToolServiceAdapter(None)
    assert adapter.is_healthy() is False


def test_process_tool_results_delegates(tool_adapter, mock_tool_coordinator):
    mock_tool_coordinator.process_tool_results.return_value = [{"name": "read", "success": True}]
    mock_pipeline_result = MagicMock()
    mock_ctx = MagicMock()

    result = tool_adapter.process_tool_results(mock_pipeline_result, mock_ctx)

    mock_tool_coordinator.process_tool_results.assert_called_once_with(
        mock_pipeline_result, mock_ctx
    )
    assert result == [{"name": "read", "success": True}]
