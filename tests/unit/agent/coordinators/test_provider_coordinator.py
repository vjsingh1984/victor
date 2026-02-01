# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ProviderCoordinator.

This test file verifies the ProviderCoordinator functionality, which manages
LLM provider operations and switching. Tests cover provider management,
health monitoring, rate limiting, and capability discovery.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from victor.agent.coordinators.provider_coordinator import (
    ProviderCoordinator,
    ProviderCoordinatorConfig,
    RateLimitInfo,
    create_provider_coordinator,
)
from victor.agent.model_switcher import SwitchReason


class TestProviderCoordinatorInit:
    """Test ProviderCoordinator initialization and configuration."""

    @pytest.fixture
    def mock_provider_manager(self) -> Mock:
        """Create mock provider manager."""
        manager = Mock()
        manager.provider = Mock()
        manager.model = "claude-sonnet-4-20250514"
        manager.provider_name = "anthropic"
        manager.tool_adapter = Mock()
        manager.capabilities = Mock()
        manager.switch_count = 0
        manager.get_info = Mock(
            return_value={"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        )
        manager.get_current_state = Mock(return_value=None)
        manager.get_healthy_providers = AsyncMock(return_value=["anthropic", "openai"])
        manager.start_health_monitoring = AsyncMock()
        manager.stop_health_monitoring = AsyncMock()
        manager.close = AsyncMock()
        manager.switch_provider = AsyncMock(return_value=True)
        manager.switch_model = AsyncMock(return_value=True)
        manager.add_switch_callback = Mock()
        return manager

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        settings = Mock()
        settings.get_provider_settings = Mock(return_value={"api_key": "test-key"})
        return settings

    def test_init_with_default_config(self, mock_provider_manager: Mock, mock_settings: Mock):
        """Test initialization with default configuration."""
        # Execute
        coordinator = ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

        # Assert
        assert coordinator._manager == mock_provider_manager
        assert coordinator.settings == mock_settings
        assert isinstance(coordinator.config, ProviderCoordinatorConfig)
        assert coordinator.config.max_rate_limit_retries == 3
        assert coordinator.config.default_rate_limit_wait == 60.0
        assert coordinator.config.enable_health_monitoring is True
        assert coordinator._rate_limit_count == 0
        assert coordinator._last_rate_limit_time is None
        assert coordinator._capability_cache == {}

    def test_init_with_custom_config(self, mock_provider_manager: Mock, mock_settings: Mock):
        """Test initialization with custom configuration."""
        # Setup
        config = ProviderCoordinatorConfig(
            max_rate_limit_retries=5,
            default_rate_limit_wait=120.0,
            enable_health_monitoring=False,
        )

        # Execute
        coordinator = ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
            config=config,
        )

        # Assert
        assert coordinator.config.max_rate_limit_retries == 5
        assert coordinator.config.default_rate_limit_wait == 120.0
        assert coordinator.config.enable_health_monitoring is False

    def test_init_post_switch_hooks_initialized_empty(
        self, mock_provider_manager: Mock, mock_settings: Mock
    ):
        """Test that post-switch hooks are initialized as empty list."""
        # Execute
        coordinator = ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

        # Assert
        assert coordinator._post_switch_hooks == []


class TestProviderCoordinatorProperties:
    """Test ProviderCoordinator property accessors."""

    @pytest.fixture
    def mock_provider_manager(self) -> Mock:
        """Create mock provider manager with properties."""
        manager = Mock()
        manager.provider = Mock(name="test_provider")
        manager.model = "test-model"
        manager.provider_name = "test_provider"
        manager.tool_adapter = Mock(name="test_adapter")
        manager.capabilities = Mock(name="test_capabilities")
        manager.switch_count = 5
        return manager

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        return Mock()

    @pytest.fixture
    def coordinator(self, mock_provider_manager: Mock, mock_settings: Mock) -> ProviderCoordinator:
        """Create coordinator instance."""
        return ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

    def test_provider_property(self, coordinator: ProviderCoordinator, mock_provider_manager: Mock):
        """Test provider property returns manager's provider."""
        # Execute
        provider = coordinator.provider

        # Assert
        assert provider == mock_provider_manager.provider

    def test_model_property(self, coordinator: ProviderCoordinator, mock_provider_manager: Mock):
        """Test model property returns manager's model."""
        # Execute
        model = coordinator.model

        # Assert
        assert model == mock_provider_manager.model

    def test_provider_name_property(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test provider_name property returns manager's provider name."""
        # Execute
        provider_name = coordinator.provider_name

        # Assert
        assert provider_name == mock_provider_manager.provider_name

    def test_tool_adapter_property(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test tool_adapter property returns manager's tool adapter."""
        # Execute
        adapter = coordinator.tool_adapter

        # Assert
        assert adapter == mock_provider_manager.tool_adapter

    def test_capabilities_property(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test capabilities property returns manager's capabilities."""
        # Execute
        capabilities = coordinator.capabilities

        # Assert
        assert capabilities == mock_provider_manager.capabilities

    def test_switch_count_property(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test switch_count property returns manager's switch count."""
        # Execute
        switch_count = coordinator.switch_count

        # Assert
        assert switch_count == mock_provider_manager.switch_count


class TestProviderSwitching:
    """Test provider and model switching functionality."""

    @pytest.fixture
    def mock_provider_manager(self) -> Mock:
        """Create mock provider manager."""
        manager = Mock()
        manager.provider = Mock()
        manager.model = "claude-sonnet-4-20250514"
        manager.provider_name = "anthropic"
        manager.switch_count = 0
        manager.get_current_state = Mock(return_value=None)
        manager.switch_provider = AsyncMock(return_value=True)
        manager.switch_model = AsyncMock(return_value=True)
        manager.add_switch_callback = Mock()
        return manager

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        settings = Mock()
        settings.get_provider_settings = Mock(return_value={})
        return settings

    @pytest.fixture
    def coordinator(self, mock_provider_manager: Mock, mock_settings: Mock) -> ProviderCoordinator:
        """Create coordinator instance."""
        return ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

    @pytest.mark.asyncio
    async def test_switch_provider_success(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test successful provider switch."""
        # Execute
        result = await coordinator.switch_provider("openai", "gpt-4")

        # Assert
        assert result is True
        mock_provider_manager.switch_provider.assert_called_once()
        call_kwargs = mock_provider_manager.switch_provider.call_args[1]
        assert call_kwargs["provider_name"] == "openai"
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["reason"] == SwitchReason.USER_REQUEST

    @pytest.mark.asyncio
    async def test_switch_provider_with_custom_kwargs(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test provider switch with custom provider kwargs."""
        # Execute
        result = await coordinator.switch_provider(
            "openai", "gpt-4", api_key="custom-key", base_url="https://custom.api"
        )

        # Assert
        assert result is True
        mock_provider_manager.switch_provider.assert_called_once()
        call_kwargs = mock_provider_manager.switch_provider.call_args[1]
        assert call_kwargs["provider_name"] == "openai"
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["api_key"] == "custom-key"
        assert call_kwargs["base_url"] == "https://custom.api"

    @pytest.mark.asyncio
    async def test_switch_provider_uses_current_model_when_not_specified(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that switch_provider uses current model when model is None."""
        # Setup
        mock_provider_manager.model = "current-model"

        # Execute
        await coordinator.switch_provider("openai")

        # Assert
        mock_provider_manager.switch_provider.assert_called_once()
        call_kwargs = mock_provider_manager.switch_provider.call_args[1]
        assert call_kwargs["model"] == "current-model"

    @pytest.mark.asyncio
    async def test_switch_provider_failure_returns_false(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that switch_provider returns False on failure."""
        # Setup
        mock_provider_manager.switch_provider = AsyncMock(return_value=False)

        # Execute
        result = await coordinator.switch_provider("openai", "gpt-4")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_provider_exception_handling(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that switch_provider handles exceptions gracefully."""
        # Setup
        mock_provider_manager.switch_provider = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        # Execute
        result = await coordinator.switch_provider("openai", "gpt-4")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_model_success(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test successful model switch."""
        # Execute
        result = await coordinator.switch_model("gpt-4")

        # Assert
        assert result is True
        mock_provider_manager.switch_model.assert_called_once()
        call_kwargs = mock_provider_manager.switch_model.call_args[1]
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["reason"] == SwitchReason.USER_REQUEST

    @pytest.mark.asyncio
    async def test_switch_model_failure_returns_false(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that switch_model returns False on failure."""
        # Setup
        mock_provider_manager.switch_model = AsyncMock(return_value=False)

        # Execute
        result = await coordinator.switch_model("gpt-4")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_model_exception_handling(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that switch_model handles exceptions gracefully."""
        # Setup
        mock_provider_manager.switch_model = AsyncMock(side_effect=Exception("Model not found"))

        # Execute
        result = await coordinator.switch_model("invalid-model")

        # Assert
        assert result is False

    def test_get_current_provider_info(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test getting current provider information."""
        # Setup
        mock_provider_manager.get_info = Mock(
            return_value={"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        )
        coordinator._rate_limit_count = 3
        mock_provider_manager.switch_count = 5

        # Execute
        info = coordinator.get_current_provider_info()

        # Assert
        assert info["provider"] == "anthropic"
        assert info["model"] == "claude-sonnet-4-20250514"
        assert info["rate_limit_count"] == 3
        assert info["switch_count"] == 5


class TestHealthMonitoring:
    """Test health monitoring functionality."""

    @pytest.fixture
    def mock_provider_manager(self) -> Mock:
        """Create mock provider manager."""
        manager = Mock()
        manager.get_current_state = Mock(return_value=None)
        manager.get_healthy_providers = AsyncMock(return_value=["anthropic", "openai"])
        manager.start_health_monitoring = AsyncMock()
        manager.stop_health_monitoring = AsyncMock()
        return manager

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        return Mock()

    @pytest.fixture
    def coordinator(self, mock_provider_manager: Mock, mock_settings: Mock) -> ProviderCoordinator:
        """Create coordinator instance."""
        return ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

    @pytest.mark.asyncio
    async def test_start_health_monitoring(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test starting health monitoring."""
        # Execute
        await coordinator.start_health_monitoring()

        # Assert
        mock_provider_manager.start_health_monitoring.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_health_monitoring(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test stopping health monitoring."""
        # Execute
        await coordinator.stop_health_monitoring()

        # Assert
        mock_provider_manager.stop_health_monitoring.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_provider_health_with_no_provider(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test getting provider health when no provider is configured."""
        # Setup
        mock_provider_manager.get_current_state = Mock(return_value=None)

        # Execute
        health = await coordinator.get_provider_health()

        # Assert
        assert health["healthy"] is False
        assert health["provider"] is None
        assert health["model"] is None
        assert "error" in health

    @pytest.mark.asyncio
    async def test_get_provider_health_with_healthy_provider(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test getting provider health when provider is healthy."""
        # Setup
        mock_state = Mock()
        mock_state.is_healthy = True
        mock_state.provider_name = "anthropic"
        mock_state.model = "claude-sonnet-4-20250514"
        mock_state.last_error = None
        mock_state.switch_count = 2
        mock_provider_manager.get_current_state = Mock(return_value=mock_state)

        # Execute
        health = await coordinator.get_provider_health()

        # Assert
        assert health["healthy"] is True
        assert health["provider"] == "anthropic"
        assert health["model"] == "claude-sonnet-4-20250514"
        assert health["last_error"] is None
        assert health["switch_count"] == 2

    @pytest.mark.asyncio
    async def test_get_provider_health_with_unhealthy_provider(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test getting provider health when provider is unhealthy."""
        # Setup
        mock_state = Mock()
        mock_state.is_healthy = False
        mock_state.provider_name = "openai"
        mock_state.model = "gpt-4"
        mock_state.last_error = "Rate limit exceeded"
        mock_state.switch_count = 1
        mock_provider_manager.get_current_state = Mock(return_value=mock_state)

        # Execute
        health = await coordinator.get_provider_health()

        # Assert
        assert health["healthy"] is False
        assert health["provider"] == "openai"
        assert health["model"] == "gpt-4"
        assert health["last_error"] == "Rate limit exceeded"
        assert health["switch_count"] == 1

    @pytest.mark.asyncio
    async def test_get_healthy_providers(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test getting list of healthy providers."""
        # Setup
        mock_provider_manager.get_healthy_providers = AsyncMock(
            return_value=["anthropic", "openai", "google"]
        )

        # Execute
        providers = await coordinator.get_healthy_providers()

        # Assert
        assert providers == ["anthropic", "openai", "google"]
        mock_provider_manager.get_healthy_providers.assert_awaited_once()


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.fixture
    def mock_provider_manager(self) -> Mock:
        """Create mock provider manager."""
        manager = Mock()
        manager.provider = Mock()
        manager.model = "test-model"
        manager.provider_name = "test"
        return manager

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        return Mock()

    @pytest.fixture
    def coordinator(self, mock_provider_manager: Mock, mock_settings: Mock) -> ProviderCoordinator:
        """Create coordinator instance."""
        return ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

    def test_get_rate_limit_wait_time_with_retry_after_attribute(
        self, coordinator: ProviderCoordinator
    ):
        """Test extracting wait time from error with retry_after attribute."""
        # Setup
        error = Exception("Rate limited")
        error.retry_after = 120.0

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert
        assert wait_time == 120.0

    def test_get_rate_limit_wait_time_with_retry_after_exceeding_max(
        self, coordinator: ProviderCoordinator
    ):
        """Test that wait time is capped at max_rate_limit_wait."""
        # Setup
        coordinator.config.max_rate_limit_wait = 300.0
        error = Exception("Rate limited")
        error.retry_after = 500.0

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert
        assert wait_time == 300.0

    def test_get_rate_limit_wait_time_with_invalid_retry_after(
        self, coordinator: ProviderCoordinator
    ):
        """Test handling of invalid retry_after attribute."""
        # Setup
        error = Exception("Rate limited")
        error.retry_after = "invalid"

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert - should fall back to default
        assert wait_time == coordinator.config.default_rate_limit_wait

    def test_get_rate_limit_wait_time_from_error_message_try_again(
        self, coordinator: ProviderCoordinator
    ):
        """Test parsing 'try again in Xs' pattern from error message."""
        # Setup
        error = Exception("Please try again in 45s")

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert
        assert wait_time == 45.0

    def test_get_rate_limit_wait_time_from_error_message_retry_after(
        self, coordinator: ProviderCoordinator
    ):
        """Test parsing 'retry after X seconds' pattern from error message."""
        # Setup
        error = Exception("Retry after 30 seconds")

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert
        assert wait_time == 30.0

    def test_get_rate_limit_wait_time_from_error_message_wait(
        self, coordinator: ProviderCoordinator
    ):
        """Test parsing 'wait X seconds' pattern from error message."""
        # Setup
        error = Exception("Please wait 15 seconds before trying again")

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert
        assert wait_time == 15.0

    def test_get_rate_limit_wait_time_from_error_message_seconds_at_end(
        self, coordinator: ProviderCoordinator
    ):
        """Test parsing 'X seconds' at end of error message."""
        # Setup
        error = Exception("Rate limited. Try again in 60 seconds")

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert
        assert wait_time == 60.0

    def test_get_rate_limit_wait_time_from_error_message_just_number(
        self, coordinator: ProviderCoordinator
    ):
        """Test parsing just a number from error message."""
        # Setup
        error = Exception("20")

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert
        assert wait_time == 20.0

    def test_get_rate_limit_wait_time_from_response_headers(self, coordinator: ProviderCoordinator):
        """Test extracting wait time from Retry-After response header."""
        # Setup
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "90"}
        error = Exception("Rate limited")
        error.response = mock_response

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert
        assert wait_time == 90.0

    def test_get_rate_limit_wait_time_fallback_to_default(self, coordinator: ProviderCoordinator):
        """Test falling back to default wait time when parsing fails."""
        # Setup
        error = Exception("Unknown error")

        # Execute
        wait_time = coordinator.get_rate_limit_wait_time(error)

        # Assert
        assert wait_time == coordinator.config.default_rate_limit_wait

    def test_track_rate_limit_increments_count(self, coordinator: ProviderCoordinator):
        """Test that tracking rate limit increments the count."""
        # Setup
        error = Exception("Rate limited")

        # Execute
        coordinator.track_rate_limit(error)

        # Assert
        assert coordinator._rate_limit_count == 1
        assert coordinator._last_rate_limit_time is not None

    def test_track_rate_limit_multiple_times(self, coordinator: ProviderCoordinator):
        """Test tracking multiple rate limit errors."""
        # Setup
        error = Exception("Rate limited")

        # Execute
        coordinator.track_rate_limit(error)
        coordinator.track_rate_limit(error)
        coordinator.track_rate_limit(error)

        # Assert
        assert coordinator._rate_limit_count == 3

    def test_get_rate_limit_stats(self, coordinator: ProviderCoordinator):
        """Test getting rate limit statistics."""
        # Setup
        coordinator._rate_limit_count = 5
        coordinator._last_rate_limit_time = 1234567890.0

        # Execute
        stats = coordinator.get_rate_limit_stats()

        # Assert
        assert stats["rate_limit_count"] == 5
        assert stats["last_rate_limit_time"] == 1234567890.0


class TestCapabilityDiscovery:
    """Test capability discovery functionality."""

    @pytest.fixture
    def mock_provider_manager(self) -> Mock:
        """Create mock provider manager."""
        manager = Mock()
        manager.provider = Mock()
        manager.provider.name = "anthropic"
        manager.model = "claude-sonnet-4-20250514"
        manager.provider_name = "anthropic"
        manager.provider.supports_tools = Mock(return_value=True)
        manager.provider.supports_streaming = Mock(return_value=True)
        return manager

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        return Mock()

    @pytest.fixture
    def coordinator(self, mock_provider_manager: Mock, mock_settings: Mock) -> ProviderCoordinator:
        """Create coordinator instance."""
        return ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

    @pytest.mark.asyncio
    async def test_discover_capabilities_success(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test successful capability discovery."""
        # Setup
        mock_capabilities = Mock()
        mock_provider_manager.provider.discover_capabilities = AsyncMock(
            return_value=mock_capabilities
        )

        # Execute
        result = await coordinator.discover_capabilities()

        # Assert
        assert result == mock_capabilities
        mock_provider_manager.provider.discover_capabilities.assert_awaited_once_with(
            "claude-sonnet-4-20250514"
        )

    @pytest.mark.asyncio
    async def test_discover_capabilities_caches_result(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that capability discovery results are cached."""
        # Setup
        mock_capabilities = Mock()
        mock_provider_manager.provider.discover_capabilities = AsyncMock(
            return_value=mock_capabilities
        )

        # Execute - call twice
        result1 = await coordinator.discover_capabilities()
        result2 = await coordinator.discover_capabilities()

        # Assert - should only call discover_capabilities once
        assert result1 == mock_capabilities
        assert result2 == mock_capabilities
        mock_provider_manager.provider.discover_capabilities.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_discover_capabilities_with_custom_provider_and_model(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test capability discovery with custom provider and model."""
        # Setup
        mock_provider = Mock()
        mock_provider.name = "openai"
        mock_provider.discover_capabilities = AsyncMock(return_value=Mock())

        # Execute
        await coordinator.discover_capabilities(provider=mock_provider, model="gpt-4")

        # Assert
        mock_provider.discover_capabilities.assert_awaited_once_with("gpt-4")

    @pytest.mark.asyncio
    async def test_discover_capabilities_falls_back_to_config_on_error(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that capability discovery falls back to config on error."""
        # Setup
        mock_provider_manager.provider.discover_capabilities = AsyncMock(
            side_effect=Exception("Discovery failed")
        )

        with patch("victor.config.config_loaders.get_provider_limits") as mock_get_limits:
            mock_limits = Mock()
            mock_limits.context_window = 200000
            mock_get_limits.return_value = mock_limits

            # Execute
            result = await coordinator.discover_capabilities()

            # Assert
            assert result is not None
            assert result.provider == "anthropic"
            assert result.model == "claude-sonnet-4-20250514"
            assert result.context_window == 200000
            assert result.supports_tools is True
            assert result.supports_streaming is True
            assert result.source == "config"

    @pytest.mark.asyncio
    async def test_discover_capabilities_returns_none_when_no_provider(
        self, mock_provider_manager: Mock, mock_settings: Mock
    ):
        """Test that capability discovery returns None when no provider."""
        # Setup
        mock_provider_manager.provider = None
        coordinator = ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

        # Execute
        result = await coordinator.discover_capabilities()

        # Assert
        assert result is None


class TestPostSwitchHooks:
    """Test post-switch hook functionality."""

    @pytest.fixture
    def mock_provider_manager(self) -> Mock:
        """Create mock provider manager."""
        manager = Mock()
        manager.provider = Mock()
        manager.model = "test-model"
        manager.provider_name = "test"
        manager.switch_count = 0

        # Simulate state changes after switch
        async def switch_provider_side_effect(**kwargs):
            manager.provider_name = kwargs.get("provider_name", "openai")
            manager.model = kwargs.get("model", "gpt-4")
            return True

        async def switch_model_side_effect(**kwargs):
            manager.model = kwargs.get("model", "gpt-4")
            return True

        def get_current_state_side_effect():
            if manager.provider_name and manager.model:
                mock_state = Mock()
                mock_state.provider_name = manager.provider_name
                mock_state.model = manager.model
                return mock_state
            return None

        manager.switch_provider = AsyncMock(side_effect=switch_provider_side_effect)
        manager.switch_model = AsyncMock(side_effect=switch_model_side_effect)
        manager.get_current_state = Mock(side_effect=get_current_state_side_effect)
        manager.add_switch_callback = Mock()
        return manager

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        settings = Mock()
        settings.get_provider_settings = Mock(return_value={})
        return settings

    @pytest.fixture
    def coordinator(self, mock_provider_manager: Mock, mock_settings: Mock) -> ProviderCoordinator:
        """Create coordinator instance."""
        return ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

    @pytest.mark.asyncio
    async def test_register_post_switch_hook(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test registering a post-switch hook."""
        # Setup
        hook = Mock()

        # Execute
        coordinator.register_post_switch_hook(hook)

        # Assert
        assert hook in coordinator._post_switch_hooks
        mock_provider_manager.add_switch_callback.assert_called_once_with(hook)

    @pytest.mark.asyncio
    async def test_notify_post_switch_hooks_calls_all_hooks(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that all post-switch hooks are called after switch."""
        # Setup
        hook1 = Mock()
        hook2 = Mock()
        hook3 = Mock()
        coordinator.register_post_switch_hook(hook1)
        coordinator.register_post_switch_hook(hook2)
        coordinator.register_post_switch_hook(hook3)

        # Execute
        await coordinator.switch_provider("openai", "gpt-4")

        # Assert - all hooks should be called once with the new state
        assert hook1.call_count == 1
        assert hook2.call_count == 1
        assert hook3.call_count == 1

        # Verify they were called with the correct state
        hook1.assert_called_once()
        hook2.assert_called_once()
        hook3.assert_called_once()

        # Verify the state has the correct provider and model
        call_args = hook1.call_args[0][0]
        assert call_args.provider_name == "openai"
        assert call_args.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_notify_post_switch_hooks_handles_exceptions(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that hook exceptions don't prevent other hooks from running."""
        # Setup
        hook1 = Mock()
        hook2 = Mock(side_effect=Exception("Hook failed"))
        hook3 = Mock()
        coordinator.register_post_switch_hook(hook1)
        coordinator.register_post_switch_hook(hook2)
        coordinator.register_post_switch_hook(hook3)

        # Execute - should not raise
        await coordinator.switch_provider("openai", "gpt-4")

        # Assert - all hooks should be called despite exception
        assert hook1.call_count == 1
        assert hook2.call_count == 1
        assert hook3.call_count == 1

    @pytest.mark.asyncio
    async def test_notify_post_switch_hooks_with_no_state(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that hooks are not called when there's no state."""
        # Setup
        hook = Mock()
        coordinator.register_post_switch_hook(hook)
        mock_provider_manager.get_current_state = Mock(return_value=None)

        # Execute
        await coordinator.switch_provider("openai", "gpt-4")

        # Assert - hook should not be called
        hook.assert_not_called()


class TestLifecycle:
    """Test lifecycle management."""

    @pytest.fixture
    def mock_provider_manager(self) -> Mock:
        """Create mock provider manager."""
        manager = Mock()
        manager.stop_health_monitoring = AsyncMock()
        manager.close = AsyncMock()
        return manager

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        return Mock()

    @pytest.fixture
    def coordinator(self, mock_provider_manager: Mock, mock_settings: Mock) -> ProviderCoordinator:
        """Create coordinator instance."""
        return ProviderCoordinator(
            provider_manager=mock_provider_manager,
            settings=mock_settings,
        )

    @pytest.mark.asyncio
    async def test_close_stops_health_monitoring(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that close stops health monitoring."""
        # Execute
        await coordinator.close()

        # Assert
        mock_provider_manager.stop_health_monitoring.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_closes_provider_manager(
        self, coordinator: ProviderCoordinator, mock_provider_manager: Mock
    ):
        """Test that close closes the provider manager."""
        # Execute
        await coordinator.close()

        # Assert
        mock_provider_manager.close.assert_awaited_once()


class TestFactoryFunction:
    """Test the factory function."""

    def test_create_provider_coordinator(self):
        """Test that factory function creates coordinator."""
        # Setup
        mock_manager = Mock()
        mock_settings = Mock()

        # Execute
        coordinator = create_provider_coordinator(
            provider_manager=mock_manager,
            settings=mock_settings,
        )

        # Assert
        assert isinstance(coordinator, ProviderCoordinator)
        assert coordinator._manager == mock_manager
        assert coordinator.settings == mock_settings

    def test_create_provider_coordinator_with_custom_config(self):
        """Test that factory function accepts custom config."""
        # Setup
        mock_manager = Mock()
        mock_settings = Mock()
        config = ProviderCoordinatorConfig(max_rate_limit_retries=10)

        # Execute
        coordinator = create_provider_coordinator(
            provider_manager=mock_manager,
            settings=mock_settings,
            config=config,
        )

        # Assert
        assert coordinator.config.max_rate_limit_retries == 10


class TestRateLimitInfo:
    """Test RateLimitInfo dataclass."""

    def test_rate_limit_info_creation(self):
        """Test creating RateLimitInfo instance."""
        # Execute
        info = RateLimitInfo(
            wait_seconds=60.0,
            retry_after=120.0,
            message="Rate limited",
            error_type="rate_limit",
        )

        # Assert
        assert info.wait_seconds == 60.0
        assert info.retry_after == 120.0
        assert info.message == "Rate limited"
        assert info.error_type == "rate_limit"

    def test_rate_limit_info_defaults(self):
        """Test RateLimitInfo with default values."""
        # Execute
        info = RateLimitInfo(wait_seconds=30.0)

        # Assert
        assert info.wait_seconds == 30.0
        assert info.retry_after is None
        assert info.message == ""
        assert info.error_type == "rate_limit"


class TestProviderCoordinatorConfig:
    """Test ProviderCoordinatorConfig dataclass."""

    def test_config_with_defaults(self):
        """Test creating config with default values."""
        # Execute
        config = ProviderCoordinatorConfig()

        # Assert
        assert config.max_rate_limit_retries == 3
        assert config.default_rate_limit_wait == 60.0
        assert config.max_rate_limit_wait == 300.0
        assert config.enable_health_monitoring is True
        assert config.health_check_interval == 60.0

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        # Execute
        config = ProviderCoordinatorConfig(
            max_rate_limit_retries=5,
            default_rate_limit_wait=120.0,
            max_rate_limit_wait=600.0,
            enable_health_monitoring=False,
            health_check_interval=30.0,
        )

        # Assert
        assert config.max_rate_limit_retries == 5
        assert config.default_rate_limit_wait == 120.0
        assert config.max_rate_limit_wait == 600.0
        assert config.enable_health_monitoring is False
        assert config.health_check_interval == 30.0
