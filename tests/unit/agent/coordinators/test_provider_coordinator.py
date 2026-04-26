from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.coordinators.provider_coordinator import (
    ProviderCoordinator,
    ProviderCoordinatorConfig,
    RateLimitInfo,
)
from victor.agent.provider.coordinator import (
    ProviderCoordinator as CanonicalProviderCoordinator,
)
from victor.agent.provider.coordinator import (
    ProviderCoordinatorConfig as CanonicalProviderCoordinatorConfig,
)
from victor.agent.provider.coordinator import RateLimitInfo as CanonicalRateLimitInfo


@pytest.fixture
def mock_manager():
    manager = MagicMock()
    manager.provider_name = "openai"
    manager.model = "gpt-4.1"
    manager.provider = None
    manager.tool_adapter = None
    manager.capabilities = None
    manager.switch_count = 0
    manager.get_current_state.return_value = SimpleNamespace(
        is_healthy=True,
        provider_name="openai",
        model="gpt-4.1",
        last_error=None,
        switch_count=1,
    )
    manager.get_healthy_providers = AsyncMock(return_value=["openai"])
    manager.add_switch_callback = MagicMock()
    return manager


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.get_provider_settings.return_value = {"api_key": "test-key"}
    return settings


def test_support_types_are_reexported_from_canonical_module():
    assert ProviderCoordinatorConfig is CanonicalProviderCoordinatorConfig
    assert RateLimitInfo is CanonicalRateLimitInfo


@pytest.mark.asyncio
async def test_switch_provider_uses_settings_and_canonical_async_delegate(
    mock_manager,
    mock_settings,
):
    coordinator = ProviderCoordinator(mock_manager, mock_settings)

    with patch.object(
        CanonicalProviderCoordinator,
        "switch_provider_async",
        new=AsyncMock(return_value=True),
    ) as switch_provider_async:
        result = await coordinator.switch_provider("anthropic", "claude-sonnet")

    assert result is True
    mock_settings.get_provider_settings.assert_called_once_with("anthropic")
    switch_provider_async.assert_awaited_once_with(
        provider_name="anthropic",
        model="claude-sonnet",
        api_key="test-key",
    )


@pytest.mark.asyncio
async def test_switch_model_delegates_to_canonical_async_delegate(
    mock_manager,
    mock_settings,
):
    coordinator = ProviderCoordinator(mock_manager, mock_settings)

    with patch.object(
        CanonicalProviderCoordinator,
        "switch_model_async",
        new=AsyncMock(return_value=True),
    ) as switch_model_async:
        result = await coordinator.switch_model("gpt-5")

    assert result is True
    switch_model_async.assert_awaited_once_with("gpt-5")


@pytest.mark.asyncio
async def test_legacy_helpers_delegate_to_canonical_methods(mock_manager, mock_settings):
    coordinator = ProviderCoordinator(mock_manager, mock_settings)

    with patch.object(
        CanonicalProviderCoordinator,
        "get_current_info",
        return_value={"provider": "openai", "model": "gpt-4.1"},
    ) as get_current_info, patch.object(
        CanonicalProviderCoordinator,
        "get_health",
        new=AsyncMock(return_value={"healthy": True, "provider": "openai"}),
    ) as get_health:
        info = coordinator.get_current_provider_info()
        health = await coordinator.get_provider_health()

    assert info == {"provider": "openai", "model": "gpt-4.1"}
    assert health == {"healthy": True, "provider": "openai"}
    get_current_info.assert_called_once_with()
    get_health.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_get_healthy_providers_delegates_to_manager(mock_manager, mock_settings):
    coordinator = ProviderCoordinator(mock_manager, mock_settings)

    providers = await coordinator.get_healthy_providers()

    assert providers == ["openai"]
    mock_manager.get_healthy_providers.assert_awaited_once_with()


def test_internal_module_is_a_canonical_adapter():
    source = Path("victor/agent/coordinators/provider_coordinator.py").read_text()

    assert "from victor.agent.provider.coordinator import (" in source
    assert "class ProviderCoordinator(CanonicalProviderCoordinator):" in source
    assert "victor.agent.provider_coordinator" not in source
