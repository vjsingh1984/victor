from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
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


def test_provider_management_runtime_reports_tool_support_from_static_capabilities():
    tool_capabilities = MagicMock()
    tool_capabilities.is_tool_call_supported.return_value = True
    host = _make_runtime_host(
        provider_name="openai",
        model="gpt-4.1",
        tool_capabilities=tool_capabilities,
        _tool_capability_warned=False,
    )
    runtime = ProviderManagementRuntime(OrchestratorProtocolAdapter(host))

    assert runtime.model_supports_tool_calls() is True
    tool_capabilities.is_tool_call_supported.assert_called_once_with("openai", "gpt-4.1")
    assert host._tool_capability_warned is False


def test_provider_management_runtime_warns_once_for_unsupported_tool_model():
    tool_capabilities = MagicMock()
    tool_capabilities.is_tool_call_supported.return_value = False
    tool_capabilities.get_supported_models.return_value = ["model-a", "model-b"]
    console = MagicMock()
    presentation = MagicMock()
    presentation.icon.return_value = "!"
    host = _make_runtime_host(
        provider_name="test-provider",
        model="no-tools",
        provider=SimpleNamespace(name="test-provider"),
        tool_capabilities=tool_capabilities,
        _tool_capability_warned=False,
        console=console,
        _presentation=presentation,
    )
    runtime = ProviderManagementRuntime(OrchestratorProtocolAdapter(host))

    assert runtime.model_supports_tool_calls() is False
    assert host._tool_capability_warned is True
    console.print.assert_called_once()

    console.print.reset_mock()
    assert runtime.model_supports_tool_calls() is False
    console.print.assert_not_called()


# --- F-016m: model-derived state is re-synced on a mid-session switch ----------


class _RecordingTracker:
    def __init__(self):
        self.exploration = None
        self.budget = None

    def set_model_exploration_settings(self, exploration_multiplier=1.0, continuation_patience=10):
        self.exploration = (exploration_multiplier, continuation_patience)

    def set_tool_budget(self, budget, user_override=False):
        self.budget = budget


class _RecordingPipeline:
    def __init__(self):
        self.budget = None

    def set_tool_budget(self, budget):
        self.budget = budget


def _fresh_caps(**overrides):
    values = {
        "exploration_multiplier": 2.0,
        "continuation_patience": 7,
        "recommended_tool_budget": 30,
        "thinking_disable_prefix": "/no_think",
        "native_tool_calls": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _host_with_derived_state(**overrides):
    values = {
        "tool_calling_caps": SimpleNamespace(
            exploration_multiplier=1.0,
            continuation_patience=10,
            recommended_tool_budget=12,
        ),
        "tool_budget": 12,
        "_factory": SimpleNamespace(
            initialize_tool_budget=lambda caps: caps.recommended_tool_budget
        ),
        "unified_tracker": _RecordingTracker(),
        "_tool_pipeline": _RecordingPipeline(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resync_refreshes_caps_budget_and_exploration():
    fresh = _fresh_caps()
    provider_service = SimpleNamespace(_provider_manager=SimpleNamespace(capabilities=fresh))
    host = _host_with_derived_state()

    ProviderManagementRuntime._resync_model_derived_state(host, provider_service)

    assert host.tool_calling_caps is fresh
    assert host.tool_budget == 30
    assert host.unified_tracker.exploration == (2.0, 7)
    assert host.unified_tracker.budget == 30
    assert host._tool_pipeline.budget == 30


def test_resync_noops_without_live_caps():
    # A provider_service without a manager -> no fresh caps -> host untouched.
    provider_service = SimpleNamespace(_provider_manager=None)
    old = SimpleNamespace(exploration_multiplier=1.0, continuation_patience=10)
    host = SimpleNamespace(tool_calling_caps=old, tool_budget=12)

    ProviderManagementRuntime._resync_model_derived_state(host, provider_service)

    assert host.tool_calling_caps is old
    assert host.tool_budget == 12


def test_resync_tolerates_bare_host():
    # A host missing factory/tracker/pipeline must not raise from the switch path.
    fresh = _fresh_caps()
    provider_service = SimpleNamespace(_provider_manager=SimpleNamespace(capabilities=fresh))
    host = SimpleNamespace(tool_calling_caps=None)

    ProviderManagementRuntime._resync_model_derived_state(host, provider_service)

    assert host.tool_calling_caps is fresh


@pytest.mark.asyncio
async def test_switch_provider_resyncs_model_derived_state_end_to_end():
    fresh = _fresh_caps()
    provider_service = MagicMock()
    provider_service._provider_manager = SimpleNamespace(capabilities=fresh)
    provider_service.switch_provider = AsyncMock()
    provider_service.get_current_provider = MagicMock(return_value=MagicMock())
    provider_service.get_current_provider_info.return_value = _make_provider_info(
        provider_name="openai", model_name="gpt-4.1"
    )
    host = _host_with_derived_state(_provider_service=provider_service)
    runtime = ProviderManagementRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.switch_provider("openai", "gpt-4.1")

    assert result is True
    # Provider/model synced AND the model-derived state refreshed for the new model.
    assert host.provider_name == "openai"
    assert host.tool_calling_caps is fresh
    assert host.tool_budget == 30
    assert host.unified_tracker.exploration == (2.0, 7)
    assert host._tool_pipeline.budget == 30
