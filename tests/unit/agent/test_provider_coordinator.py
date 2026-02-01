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

Tests the provider coordination functionality including:
- Rate limit handling and retry logic
- Health monitoring coordination
- Provider/model switching
- Post-switch hooks
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.provider_coordinator import (
    ProviderCoordinator,
    ProviderCoordinatorConfig,
    RateLimitInfo,
    create_provider_coordinator,
)
from victor.agent.provider_manager import ProviderManager, ProviderState
from victor.agent.tool_calling import ToolCallingCapabilities, ToolCallFormat


class TestProviderCoordinatorConfig:
    """Tests for ProviderCoordinatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProviderCoordinatorConfig()

        assert config.max_rate_limit_retries == 3
        assert config.default_rate_limit_wait == 60.0
        assert config.max_rate_limit_wait == 300.0
        assert config.enable_health_monitoring is True
        assert config.health_check_interval == 60.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProviderCoordinatorConfig(
            max_rate_limit_retries=5,
            default_rate_limit_wait=30.0,
            max_rate_limit_wait=600.0,
            enable_health_monitoring=False,
            health_check_interval=120.0,
        )

        assert config.max_rate_limit_retries == 5
        assert config.default_rate_limit_wait == 30.0
        assert config.max_rate_limit_wait == 600.0
        assert config.enable_health_monitoring is False
        assert config.health_check_interval == 120.0


class TestRateLimitInfo:
    """Tests for RateLimitInfo dataclass."""

    def test_default_values(self):
        """Test default values for RateLimitInfo."""
        info = RateLimitInfo(wait_seconds=60.0)

        assert info.wait_seconds == 60.0
        assert info.retry_after is None
        assert info.message == ""
        assert info.error_type == "rate_limit"

    def test_custom_values(self):
        """Test custom values for RateLimitInfo."""
        info = RateLimitInfo(
            wait_seconds=120.0,
            retry_after=100.0,
            message="Too many requests",
            error_type="quota_exceeded",
        )

        assert info.wait_seconds == 120.0
        assert info.retry_after == 100.0
        assert info.message == "Too many requests"
        assert info.error_type == "quota_exceeded"


class TestProviderCoordinator:
    """Tests for ProviderCoordinator."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider instance."""
        provider = MagicMock()
        provider.name = "anthropic"
        provider.supports_tools.return_value = True
        provider.supports_streaming.return_value = True
        provider.close = AsyncMock()
        return provider

    @pytest.fixture
    def mock_state(self, mock_provider):
        """Create mock provider state."""
        return ProviderState(
            provider=mock_provider,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            is_healthy=True,
            switch_count=0,
        )

    @pytest.fixture
    def mock_manager(self, mock_provider, mock_state):
        """Create mock ProviderManager."""
        manager = MagicMock(spec=ProviderManager)
        manager.provider = mock_provider
        manager.model = "claude-sonnet-4-20250514"
        manager.provider_name = "anthropic"
        manager.tool_adapter = MagicMock()
        manager.capabilities = ToolCallingCapabilities(
            native_tool_calls=True,
            streaming_tool_calls=True,
            parallel_tool_calls=True,
            tool_call_format=ToolCallFormat.ANTHROPIC,
        )
        manager.switch_count = 0
        manager.get_current_state.return_value = mock_state
        manager.get_info.return_value = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "is_healthy": True,
        }
        manager.switch_provider = AsyncMock(return_value=True)
        manager.switch_model = AsyncMock(return_value=True)
        manager.start_health_monitoring = AsyncMock()
        manager.stop_health_monitoring = AsyncMock()
        manager.close = AsyncMock()
        manager.add_switch_callback = MagicMock()
        return manager

    @pytest.fixture
    def coordinator(self, mock_manager):
        """Create coordinator with mock manager."""
        return ProviderCoordinator(provider_manager=mock_manager)

    def test_init_default_config(self, mock_manager):
        """Test initialization with default config."""
        coordinator = ProviderCoordinator(provider_manager=mock_manager)

        assert coordinator.config.max_rate_limit_retries == 3
        assert coordinator.config.default_rate_limit_wait == 60.0
        assert coordinator._manager is mock_manager

    def test_init_custom_config(self, mock_manager):
        """Test initialization with custom config."""
        config = ProviderCoordinatorConfig(
            max_rate_limit_retries=5,
            default_rate_limit_wait=30.0,
        )
        coordinator = ProviderCoordinator(
            provider_manager=mock_manager,
            config=config,
        )

        assert coordinator.config.max_rate_limit_retries == 5
        assert coordinator.config.default_rate_limit_wait == 30.0

    def test_provider_property(self, coordinator, mock_manager):
        """Test provider property delegation."""
        assert coordinator.provider is mock_manager.provider

    def test_model_property(self, coordinator, mock_manager):
        """Test model property delegation."""
        assert coordinator.model == mock_manager.model

    def test_provider_name_property(self, coordinator, mock_manager):
        """Test provider_name property delegation."""
        assert coordinator.provider_name == mock_manager.provider_name

    def test_tool_adapter_property_from_manager(self, coordinator, mock_manager):
        """Test tool_adapter property from manager."""
        assert coordinator.tool_adapter is mock_manager.tool_adapter

    def test_tool_adapter_property_override(self, mock_manager):
        """Test tool_adapter property with override."""
        custom_adapter = MagicMock()
        coordinator = ProviderCoordinator(
            provider_manager=mock_manager,
            tool_adapter=custom_adapter,
        )

        assert coordinator.tool_adapter is custom_adapter

    def test_capabilities_property_from_manager(self, coordinator, mock_manager):
        """Test capabilities property from manager."""
        assert coordinator.capabilities is mock_manager.capabilities

    def test_capabilities_property_override(self, mock_manager):
        """Test capabilities property with override."""
        custom_caps = ToolCallingCapabilities(
            native_tool_calls=False,
            tool_call_format=ToolCallFormat.OPENAI,
        )
        coordinator = ProviderCoordinator(
            provider_manager=mock_manager,
            capabilities=custom_caps,
        )

        assert coordinator.capabilities is custom_caps

    def test_switch_count_property(self, coordinator, mock_manager):
        """Test switch_count property delegation."""
        assert coordinator.switch_count == mock_manager.switch_count

    def test_get_current_info(self, coordinator, mock_manager):
        """Test get_current_info includes coordinator info."""
        info = coordinator.get_current_info()

        assert "provider" in info
        assert "model" in info
        assert "rate_limit_count" in info
        assert "is_health_monitoring" in info
        assert "switch_count" in info
        assert info["rate_limit_count"] == 0
        assert info["is_health_monitoring"] is False

    @pytest.mark.asyncio
    async def test_get_health(self, coordinator, mock_manager, mock_state):
        """Test get_health returns health status."""
        health = await coordinator.get_health()

        assert health["healthy"] is True
        assert health["provider"] == "anthropic"
        assert health["model"] == "claude-sonnet-4-20250514"
        assert health["switch_count"] == 0

    @pytest.mark.asyncio
    async def test_get_health_no_provider(self, mock_manager):
        """Test get_health when no provider configured."""
        mock_manager.get_current_state.return_value = None
        coordinator = ProviderCoordinator(provider_manager=mock_manager)

        health = await coordinator.get_health()

        assert health["healthy"] is False
        assert health["provider"] is None
        assert "error" in health

    @pytest.mark.asyncio
    async def test_start_health_monitoring(self, coordinator, mock_manager):
        """Test start_health_monitoring."""
        await coordinator.start_health_monitoring()

        mock_manager.start_health_monitoring.assert_called_once()
        assert coordinator._is_monitoring is True

    @pytest.mark.asyncio
    async def test_start_health_monitoring_already_running(self, coordinator, mock_manager):
        """Test start_health_monitoring when already running."""
        coordinator._is_monitoring = True

        await coordinator.start_health_monitoring()

        mock_manager.start_health_monitoring.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_health_monitoring(self, coordinator, mock_manager):
        """Test stop_health_monitoring."""
        coordinator._is_monitoring = True

        await coordinator.stop_health_monitoring()

        mock_manager.stop_health_monitoring.assert_called_once()
        assert coordinator._is_monitoring is False

    @pytest.mark.asyncio
    async def test_stop_health_monitoring_not_running(self, coordinator, mock_manager):
        """Test stop_health_monitoring when not running."""
        await coordinator.stop_health_monitoring()

        mock_manager.stop_health_monitoring.assert_not_called()

    def test_get_rate_limit_wait_time_default(self, coordinator):
        """Test get_rate_limit_wait_time with no parseable time."""
        error = Exception("Some error occurred")

        wait_time = coordinator.get_rate_limit_wait_time(error)

        assert wait_time == 60.0  # default

    def test_get_rate_limit_wait_time_retry_after(self, coordinator):
        """Test get_rate_limit_wait_time with 'retry after X seconds'."""
        error = Exception("Rate limited. Please retry after 45 seconds")

        wait_time = coordinator.get_rate_limit_wait_time(error)

        assert wait_time == 45.0

    def test_get_rate_limit_wait_time_wait_pattern(self, coordinator):
        """Test get_rate_limit_wait_time with 'wait X seconds'."""
        error = Exception("Too many requests, wait 30 seconds")

        wait_time = coordinator.get_rate_limit_wait_time(error)

        assert wait_time == 30.0

    def test_get_rate_limit_wait_time_clamped(self, coordinator):
        """Test get_rate_limit_wait_time clamps to max."""
        error = Exception("Please retry after 600 seconds")

        wait_time = coordinator.get_rate_limit_wait_time(error)

        assert wait_time == 300.0  # max_rate_limit_wait

    @pytest.mark.asyncio
    async def test_stream_with_rate_limit_retry_success(self, coordinator):
        """Test stream_with_rate_limit_retry on success."""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        chunks = []
        async for chunk in coordinator.stream_with_rate_limit_retry(mock_stream):
            chunks.append(chunk)

        assert chunks == ["chunk1", "chunk2", "chunk3"]

    @pytest.mark.asyncio
    async def test_stream_with_rate_limit_retry_retries_on_rate_limit(self, coordinator):
        """Test stream_with_rate_limit_retry retries on rate limit."""
        call_count = 0

        async def mock_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Use if False: yield pattern to make this a proper async generator
                if False:
                    yield "never"
                raise Exception("Rate limit exceeded. Retry after 1 second")
            yield "success"

        # Patch sleep to avoid actual waiting
        with patch("victor.agent.provider_coordinator.asyncio.sleep", new_callable=AsyncMock):
            chunks = []
            async for chunk in coordinator.stream_with_rate_limit_retry(mock_stream):
                chunks.append(chunk)

        assert chunks == ["success"]
        assert call_count == 2
        assert coordinator._rate_limit_count == 1

    @pytest.mark.asyncio
    async def test_stream_with_rate_limit_retry_max_retries_exceeded(self, coordinator):
        """Test stream_with_rate_limit_retry raises after max retries."""

        async def mock_stream():
            # Must yield at least once to be a proper async generator before raising
            if False:  # Never executed, but makes this a generator
                yield "never"
            raise Exception("Rate limit exceeded. 429 Too Many Requests")

        with patch("victor.agent.provider_coordinator.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="Rate limit"):
                async for _ in coordinator.stream_with_rate_limit_retry(mock_stream):
                    pass

        # 1 initial + 3 retries = 4 total rate limit hits
        assert coordinator._rate_limit_count == 4

    @pytest.mark.asyncio
    async def test_stream_with_rate_limit_retry_non_rate_limit_error(self, coordinator):
        """Test stream_with_rate_limit_retry raises on non-rate-limit error."""

        async def mock_stream():
            # Must yield at least once to be a proper async generator before raising
            if False:  # Never executed, but makes this a generator
                yield "never"
            raise ValueError("Some other error")

        with pytest.raises(ValueError, match="Some other error"):
            async for _ in coordinator.stream_with_rate_limit_retry(mock_stream):
                pass

        assert coordinator._rate_limit_count == 0

    def test_register_post_switch_hook(self, coordinator, mock_manager):
        """Test register_post_switch_hook."""
        callback = MagicMock()

        coordinator.register_post_switch_hook(callback)

        assert callback in coordinator._post_switch_hooks
        mock_manager.add_switch_callback.assert_called_once_with(callback)

    def test_notify_post_switch_hooks(self, coordinator, mock_manager, mock_state):
        """Test _notify_post_switch_hooks calls all hooks."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        coordinator._post_switch_hooks = [callback1, callback2]

        coordinator._notify_post_switch_hooks()

        callback1.assert_called_once_with(mock_state)
        callback2.assert_called_once_with(mock_state)

    def test_notify_post_switch_hooks_handles_exception(
        self, coordinator, mock_manager, mock_state
    ):
        """Test _notify_post_switch_hooks handles callback exceptions."""

        def bad_callback(state):
            raise ValueError("Callback error")

        good_callback = MagicMock()
        coordinator._post_switch_hooks = [bad_callback, good_callback]

        # Should not raise
        coordinator._notify_post_switch_hooks()

        # Good callback should still be called
        good_callback.assert_called_once_with(mock_state)

    @pytest.mark.asyncio
    async def test_close(self, coordinator, mock_manager):
        """Test close cleans up resources."""
        coordinator._is_monitoring = True

        await coordinator.close()

        mock_manager.stop_health_monitoring.assert_called_once()
        mock_manager.close.assert_called_once()
        assert coordinator._is_monitoring is False


class TestCreateProviderCoordinator:
    """Tests for create_provider_coordinator factory function."""

    @pytest.fixture
    def mock_manager(self):
        """Create mock ProviderManager."""
        manager = MagicMock(spec=ProviderManager)
        manager.provider = MagicMock()
        manager.model = "test-model"
        manager.provider_name = "test-provider"
        manager.tool_adapter = None
        manager.capabilities = None
        manager.switch_count = 0
        manager.add_switch_callback = MagicMock()
        return manager

    def test_create_basic(self, mock_manager):
        """Test basic factory creation."""
        coordinator = create_provider_coordinator(provider_manager=mock_manager)

        assert isinstance(coordinator, ProviderCoordinator)
        assert coordinator._manager is mock_manager
        assert coordinator.config.max_rate_limit_retries == 3

    def test_create_with_config(self, mock_manager):
        """Test factory creation with config."""
        config = ProviderCoordinatorConfig(max_rate_limit_retries=10)

        coordinator = create_provider_coordinator(
            provider_manager=mock_manager,
            config=config,
        )

        assert coordinator.config.max_rate_limit_retries == 10

    def test_create_with_overrides(self, mock_manager):
        """Test factory creation with tool adapter and capabilities overrides."""
        custom_adapter = MagicMock()
        custom_caps = ToolCallingCapabilities(
            native_tool_calls=True,
            tool_call_format=ToolCallFormat.OPENAI,
        )

        coordinator = create_provider_coordinator(
            provider_manager=mock_manager,
            tool_adapter=custom_adapter,
            capabilities=custom_caps,
        )

        assert coordinator.tool_adapter is custom_adapter
        assert coordinator.capabilities is custom_caps
