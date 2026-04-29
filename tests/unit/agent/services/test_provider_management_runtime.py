from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.provider_management_runtime import ProviderManagementRuntime


def _make_provider_info(**overrides):
    values = {
        "provider_name": "openai",
        "model_name": "gpt-4.1",
        "api_key_configured": True,
        "supports_streaming": True,
        "supports_tool_calling": True,
        "max_tokens": 128000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_runtime_host(**overrides):
    values = {
        "_provider_service": MagicMock(),
        "provider": None,
        "provider_name": "anthropic",
        "model": "claude-3-7-sonnet",
        "tool_budget": 8,
        "tool_calls_used": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_provider_management_runtime_switch_provider_syncs_host_state_and_callback():
    provider_service = MagicMock()
    provider_service.switch_provider = AsyncMock()
    provider_service.get_current_provider = MagicMock(return_value=MagicMock(name="provider"))
    provider_service.get_current_provider_info.return_value = _make_provider_info()
    host = _make_runtime_host(_provider_service=provider_service)
    runtime = ProviderManagementRuntime(OrchestratorProtocolAdapter(host))
    callback = MagicMock()

    result = await runtime.switch_provider("openai", "gpt-4.1", callback)

    assert result is True
    provider_service.switch_provider.assert_awaited_once_with("openai", "gpt-4.1")
    assert host.provider is provider_service.get_current_provider.return_value
    assert host.provider_name == "openai"
    assert host.model == "gpt-4.1"
    callback.assert_called_once_with("openai", "gpt-4.1")


@pytest.mark.asyncio
async def test_provider_management_runtime_switch_model_syncs_host_state():
    provider_service = MagicMock()
    provider_service.switch_model = AsyncMock()
    provider_service.get_current_provider = MagicMock(return_value=MagicMock(name="provider"))
    provider_service.get_current_provider_info.return_value = _make_provider_info(
        provider_name="anthropic",
        model_name="claude-3-7-sonnet",
    )
    host = _make_runtime_host(_provider_service=provider_service)
    runtime = ProviderManagementRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.switch_model("claude-3-7-sonnet")

    assert result is True
    provider_service.switch_model.assert_awaited_once_with("claude-3-7-sonnet")
    assert host.provider is provider_service.get_current_provider.return_value
    assert host.provider_name == "anthropic"
    assert host.model == "claude-3-7-sonnet"


@pytest.mark.asyncio
async def test_provider_management_runtime_health_monitoring_handles_missing_service():
    host = _make_runtime_host(_provider_service=None)
    runtime = ProviderManagementRuntime(OrchestratorProtocolAdapter(host))

    assert await runtime.start_health_monitoring() is False
    assert await runtime.stop_health_monitoring() is False


@pytest.mark.asyncio
async def test_provider_management_runtime_get_provider_health_returns_expected_dict():
    provider_service = MagicMock()
    provider_service.check_provider_health = AsyncMock(return_value=True)
    provider_service.get_current_provider_info.return_value = _make_provider_info(
        provider_name="openai",
        model_name="gpt-4.1",
        api_key_configured=False,
    )
    host = _make_runtime_host(_provider_service=provider_service)
    runtime = ProviderManagementRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.get_provider_health()

    assert result == {
        "healthy": True,
        "provider": "openai",
        "model": "gpt-4.1",
        "api_key_configured": False,
    }


def test_provider_management_runtime_get_current_provider_info_merges_rate_limit_stats():
    provider_service = MagicMock()
    provider_service.get_current_provider_info.return_value = _make_provider_info(
        provider_name="openai",
        model_name="gpt-4.1",
    )
    provider_service.get_rate_limit_stats.return_value = {"rate_limits_hit": 7}
    host = _make_runtime_host(_provider_service=provider_service, tool_budget=21, tool_calls_used=4)
    runtime = ProviderManagementRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.get_current_provider_info()

    assert result["provider_name"] == "openai"
    assert result["model_name"] == "gpt-4.1"
    assert result["tool_budget"] == 21
    assert result["tool_calls_used"] == 4
    assert result["rate_limits_hit"] == 7
