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

"""Tests for ProviderSwitchCoordinator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, call
from typing import Any, Dict, Optional

from victor.agent.provider.switcher import ProviderSwitcher, ProviderSwitcherState
from victor.agent.provider.health_monitor import ProviderHealthMonitor
from victor.core.events.protocols import Event


class TestProviderSwitchCoordinator:
    """Tests for ProviderSwitchCoordinator class."""

    @pytest.fixture
    def provider_switcher(self):
        """Create a mock provider switcher."""
        switcher = MagicMock(spec=ProviderSwitcher)
        switcher.get_current_state = MagicMock(return_value=None)
        switcher.switch_provider = AsyncMock(return_value=True)
        switcher.switch_model = AsyncMock(return_value=True)
        switcher.get_switch_history = MagicMock(return_value=[])
        return switcher

    @pytest.fixture
    def health_monitor(self):
        """Create a mock health monitor."""
        monitor = MagicMock(spec=ProviderHealthMonitor)
        monitor.check_health = AsyncMock(return_value=True)
        return monitor

    @pytest.fixture
    def coordinator(self, provider_switcher, health_monitor):
        """Create ProviderSwitchCoordinator with mocks."""
        from victor.agent.provider_switch_coordinator import ProviderSwitchCoordinator

        return ProviderSwitchCoordinator(
            provider_switcher=provider_switcher,
            health_monitor=health_monitor,
        )

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = MagicMock()
        provider.name = "test_provider"
        return provider

    def test_init(self, coordinator, provider_switcher, health_monitor):
        """Test coordinator initialization."""
        assert coordinator._provider_switcher is provider_switcher
        assert coordinator._health_monitor is health_monitor

    @pytest.mark.asyncio
    async def test_switch_provider_success(self, coordinator, provider_switcher, health_monitor):
        """Test successful provider switch."""
        # Setup switcher to return True
        provider_switcher.switch_provider = AsyncMock(return_value=True)

        # Switch provider
        result = await coordinator.switch_provider(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            reason="user_request",
        )

        # Verify switch succeeded
        assert result is True
        provider_switcher.switch_provider.assert_called_once_with(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            reason="user_request",
            settings=None,
        )

    @pytest.mark.asyncio
    async def test_switch_provider_with_health_check(
        self, coordinator, provider_switcher, health_monitor, mock_provider
    ):
        """Test provider switch with health verification."""
        # Setup mocks
        health_monitor.check_health = AsyncMock(return_value=True)
        provider_switcher.switch_provider = AsyncMock(return_value=True)

        # Mock provider creation to avoid real initialization
        from unittest.mock import patch

        with patch.object(coordinator, "_create_provider", return_value=mock_provider):
            # Switch with health check
            result = await coordinator.switch_provider(
                provider_name="anthropic",
                model="claude-sonnet-4-20250514",
                reason="user_request",
                verify_health=True,
            )

        # Verify health check was performed
        assert result is True
        health_monitor.check_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_provider_health_check_fails(
        self, coordinator, provider_switcher, health_monitor, mock_provider
    ):
        """Test provider switch when health check fails."""
        # Setup health monitor to fail
        health_monitor.check_health = AsyncMock(return_value=False)
        provider_switcher.switch_provider = AsyncMock(return_value=True)

        # Mock provider creation
        from unittest.mock import patch

        with patch.object(coordinator, "_create_provider", return_value=mock_provider):
            # Switch should still succeed but log warning
            result = await coordinator.switch_provider(
                provider_name="anthropic",
                model="claude-sonnet-4-20250514",
                reason="user_request",
                verify_health=True,
            )

        # Health check failed but switch proceeded
        assert result is True
        health_monitor.check_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_provider_failure(self, coordinator, provider_switcher):
        """Test provider switch failure."""
        # Setup switcher to return False
        provider_switcher.switch_provider = AsyncMock(return_value=False)

        # Switch provider
        result = await coordinator.switch_provider(
            provider_name="invalid_provider",
            model="invalid_model",
            reason="fallback",
        )

        # Verify switch failed
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_model_success(self, coordinator, provider_switcher):
        """Test successful model switch on same provider."""
        # Setup
        provider_switcher.switch_model = AsyncMock(return_value=True)

        # Switch model
        result = await coordinator.switch_model(
            model="claude-opus-4-20250514",
            reason="user_request",
        )

        # Verify switch succeeded
        assert result is True
        provider_switcher.switch_model.assert_called_once_with(
            model="claude-opus-4-20250514",
            reason="user_request",
        )

    @pytest.mark.asyncio
    async def test_switch_model_failure(self, coordinator, provider_switcher):
        """Test model switch failure."""
        # Setup
        provider_switcher.switch_model = AsyncMock(return_value=False)

        # Switch model
        result = await coordinator.switch_model(
            model="invalid_model",
            reason="fallback",
        )

        # Verify switch failed
        assert result is False

    def test_get_switch_history(self, coordinator, provider_switcher):
        """Test getting switch history."""
        # Setup history
        history = [
            {
                "from_provider": "anthropic",
                "from_model": "claude-sonnet-4-20250514",
                "to_provider": "openai",
                "to_model": "gpt-4",
                "reason": "user_request",
                "timestamp": "2025-01-07T12:00:00",
            }
        ]
        provider_switcher.get_switch_history = MagicMock(return_value=history)

        # Get history
        result = coordinator.get_switch_history()

        # Verify history returned
        assert result == history
        provider_switcher.get_switch_history.assert_called_once()

    def test_get_current_state(self, coordinator, provider_switcher, mock_provider):
        """Test getting current switcher state."""
        # Setup state
        state = ProviderSwitcherState(
            provider=mock_provider,
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            switch_count=2,
        )
        provider_switcher.get_current_state = MagicMock(return_value=state)

        # Get state
        result = coordinator.get_current_state()

        # Verify state returned
        assert result is state
        assert result.provider_name == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.switch_count == 2

    @pytest.mark.asyncio
    async def test_switch_with_callback(self, coordinator, provider_switcher):
        """Test provider switch with callback."""
        # Setup
        callback = MagicMock()
        provider_switcher.switch_provider = AsyncMock(return_value=True)

        # Register callback
        coordinator.on_switch(callback)

        # Switch provider
        await coordinator.switch_provider(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
        )

        # Verify callback was registered
        provider_switcher.on_switch.assert_called_once_with(callback)

    @pytest.mark.asyncio
    async def test_validate_switch_request_valid(self, coordinator):
        """Test validation of valid switch request."""
        # Valid request
        is_valid, error = coordinator.validate_switch_request(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
        )

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_switch_request_invalid(self, coordinator):
        """Test validation of invalid switch request."""
        # Invalid request (empty provider)
        is_valid, error = coordinator.validate_switch_request(
            provider_name="",
            model="claude-sonnet-4-20250514",
        )

        assert is_valid is False
        assert error is not None
        assert "provider" in error.lower()

    @pytest.mark.asyncio
    async def test_switch_with_post_switch_hooks(self, coordinator, provider_switcher):
        """Test provider switch with post-switch hooks."""
        # Setup
        hooks_called = []

        def post_switch_hook(state):
            hooks_called.append("post_switch")

        provider_switcher.switch_provider = AsyncMock(return_value=True)

        # Setup get_current_state to return a state
        mock_state = MagicMock()
        provider_switcher.get_current_state = MagicMock(return_value=mock_state)

        # Register hook
        coordinator.register_post_switch_hook(post_switch_hook)

        # Switch provider
        await coordinator.switch_provider(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
        )

        # Verify hook was called
        assert len(hooks_called) == 1
        assert "post_switch" in hooks_called


class TestProviderSwitchCoordinatorIntegration:
    """Integration tests for ProviderSwitchCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with real dependencies."""
        from victor.agent.provider_switch_coordinator import ProviderSwitchCoordinator

        # Use real ProviderSwitcher but with mocked dependencies
        switcher = MagicMock(spec=ProviderSwitcher)
        health_monitor = MagicMock(spec=ProviderHealthMonitor)

        return ProviderSwitchCoordinator(
            provider_switcher=switcher,
            health_monitor=health_monitor,
        )

    @pytest.mark.asyncio
    async def test_full_switch_lifecycle(self, coordinator):
        """Test complete switch lifecycle through coordinator."""
        # Setup
        mock_provider = MagicMock()
        coordinator._provider_switcher.switch_provider = AsyncMock(return_value=True)
        coordinator._health_monitor.check_health = AsyncMock(return_value=True)

        # Mock provider creation
        from unittest.mock import patch

        with patch.object(coordinator, "_create_provider", return_value=mock_provider):
            # Execute full switch
            result = await coordinator.switch_provider(
                provider_name="anthropic",
                model="claude-sonnet-4-20250514",
                reason="user_request",
                verify_health=True,
            )

        # Verify complete lifecycle
        assert result is True
        coordinator._health_monitor.check_health.assert_called_once()
        coordinator._provider_switcher.switch_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_with_fallback_on_failure(self, coordinator):
        """Test fallback handling during switch failure."""
        # Setup: first switch fails, second succeeds
        coordinator._provider_switcher.switch_provider = AsyncMock(side_effect=[False, True])

        # First switch fails
        result1 = await coordinator.switch_provider(
            provider_name="invalid_provider",
            model="invalid_model",
        )

        # Second switch succeeds
        result2 = await coordinator.switch_provider(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
        )

        # Verify results
        assert result1 is False
        assert result2 is True
        assert coordinator._provider_switcher.switch_provider.call_count == 2


class TestProviderSwitchCoordinatorErrorHandling:
    """Error handling tests for ProviderSwitchCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator."""
        from victor.agent.provider_switch_coordinator import ProviderSwitchCoordinator

        switcher = MagicMock(spec=ProviderSwitcher)
        health_monitor = MagicMock(spec=ProviderHealthMonitor)

        return ProviderSwitchCoordinator(
            provider_switcher=switcher,
            health_monitor=health_monitor,
        )

    @pytest.mark.asyncio
    async def test_switch_handles_exception(self, coordinator):
        """Test that switch exceptions are handled gracefully."""
        # Setup to raise exception
        coordinator._provider_switcher.switch_provider = AsyncMock(
            side_effect=RuntimeError("Provider unavailable")
        )

        # Switch should handle exception
        result = await coordinator.switch_provider(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
        )

        # Verify failure returned
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_handles_health_check_exception(self, coordinator):
        """Test that health check exceptions don't prevent switch."""
        # Setup health check to raise exception
        coordinator._health_monitor.check_health = AsyncMock(
            side_effect=RuntimeError("Health check failed")
        )
        coordinator._provider_switcher.switch_provider = AsyncMock(return_value=True)

        # Switch should still succeed
        result = await coordinator.switch_provider(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            verify_health=True,
        )

        # Verify switch succeeded despite health check error
        assert result is True

    @pytest.mark.asyncio
    async def test_switch_with_retry_on_transient_error(self, coordinator):
        """Test retry logic for transient errors."""
        # Setup: first attempt fails with transient error, second succeeds
        coordinator._provider_switcher.switch_provider = AsyncMock(
            side_effect=[
                RuntimeError("Rate limit"),
                True,  # Second attempt succeeds
            ]
        )

        # Switch with retry should eventually succeed
        result = await coordinator.switch_provider(
            provider_name="anthropic",
            model="claude-sonnet-4-20250514",
            max_retries=1,
        )

        # Verify retry occurred
        assert result is True
        assert coordinator._provider_switcher.switch_provider.call_count == 2
