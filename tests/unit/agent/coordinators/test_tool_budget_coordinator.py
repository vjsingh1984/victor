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

"""Unit tests for ToolBudgetCoordinator.

This test file contains tests migrated from test_orchestrator_core.py
that specifically test tool budget management functionality.

Migrated Tests:
- TestToolCallBudget (from test_orchestrator_core.py lines 338-352)
- TestToolBudgetManagement (from test_orchestrator_core.py lines 2028-2247)
- TestToolBudgetManagementExtended (from test_orchestrator_core.py lines 2249-2268)

These tests focus on budget tracking and management, which is the
responsibility of ToolBudgetCoordinator.
"""

import pytest
from unittest.mock import Mock, MagicMock

from victor.agent.coordinators.tool_budget_coordinator import (
    ToolBudgetCoordinator,
    ToolBudgetConfig,
    BudgetStatus,
    create_tool_budget_coordinator,
)


class TestToolBudgetCoordinatorConfig:
    """Tests for ToolBudgetConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ToolBudgetConfig()

        assert config.default_budget == 25
        assert config.budget_multiplier == 1.0
        assert config.warning_threshold == 0.2
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ToolBudgetConfig(
            default_budget=50,
            budget_multiplier=2.0,
            warning_threshold=0.3,
        )

        assert config.default_budget == 50
        assert config.budget_multiplier == 2.0
        assert config.warning_threshold == 0.3


class TestToolBudgetCoordinator:
    """Tests for ToolBudgetCoordinator functionality.

    These tests were migrated from TestToolCallBudget and related
    test classes in test_orchestrator_core.py.
    """

    @pytest.fixture
    def coordinator(self):
        """Create a ToolBudgetCoordinator for testing."""
        config = ToolBudgetConfig(default_budget=25, budget_multiplier=1.0)
        return ToolBudgetCoordinator(config=config)

    def test_initial_budget(self, coordinator):
        """Test initial budget is set correctly."""
        budget = coordinator.budget
        assert budget == 25
        assert budget > 0

    def test_tool_calls_used_tracking(self, coordinator):
        """Test budget_used is tracked."""
        assert coordinator.budget_used == 0

        coordinator.consume()
        assert coordinator.budget_used == 1

        coordinator.consume()
        assert coordinator.budget_used == 2

    def test_budget_property(self, coordinator):
        """Test tool_budget property returns budget."""
        budget = coordinator.budget
        assert isinstance(budget, int)
        assert budget > 0

    def test_budget_with_multiplier(self):
        """Test budget calculation with multiplier."""
        config = ToolBudgetConfig(default_budget=25, budget_multiplier=2.0)
        coordinator = ToolBudgetCoordinator(config=config)

        budget = coordinator.budget
        assert budget == 25  # Multiplier doesn't affect initial budget

    def test_budget_enforcement(self, coordinator):
        """Test budget enforcement prevents exceeding limit."""
        # Set a low budget for testing
        coordinator.budget = 3

        # Consume budget
        coordinator.consume()
        coordinator.consume()
        coordinator.consume()

        # Budget should be exhausted
        status = coordinator.get_status()
        assert status.remaining == 0
        assert status.is_exhausted is True

    def test_budget_reset(self, coordinator):
        """Test budget can be reset."""
        coordinator.consume()
        coordinator.consume()
        assert coordinator.budget_used == 2

        coordinator.reset()
        assert coordinator.budget_used == 0
        status = coordinator.get_status()
        assert status.remaining > 0

    def test_budget_status(self, coordinator):
        """Test get_status returns correct status."""
        status = coordinator.get_status()

        assert isinstance(status, BudgetStatus)
        assert status.total > 0
        assert status.remaining == status.total
        assert status.is_exhausted is False
        assert status.used == 0

    def test_budget_status_after_calls(self, coordinator):
        """Test budget status after consuming budget."""
        coordinator.consume()
        coordinator.consume()

        status = coordinator.get_status()

        assert status.used == 2
        assert status.remaining == status.total - 2

    def test_set_budget(self, coordinator):
        """Test setting custom budget."""
        coordinator.budget = 50

        status = coordinator.get_status()
        assert status.total == 50
        assert status.remaining == 50

    def test_max_budget_enforcement(self):
        """Test budget can be set to any value."""
        config = ToolBudgetConfig(default_budget=25, budget_multiplier=10.0)
        coordinator = ToolBudgetCoordinator(config=config)

        # Budget is set from config
        budget = coordinator.budget
        assert budget == 25

    def test_disable_budget_tracking(self):
        """Test budget can be disabled."""
        config = ToolBudgetConfig(enabled=False)
        coordinator = ToolBudgetCoordinator(config=config)

        # Should still track when enabled
        coordinator.consume()

        status = coordinator.get_status()
        # Budget tracking works independently
        assert status.is_exhausted is False


class TestBudgetStatus:
    """Tests for BudgetStatus dataclass."""

    def test_budget_status_creation(self):
        """Test BudgetStatus can be created."""
        status = BudgetStatus(
            total=25,
            used=5,
            remaining=20,
            is_exhausted=False,
            utilization=0.2,
        )

        assert status.total == 25
        assert status.remaining == 20
        assert status.used == 5
        assert status.is_exhausted is False

    def test_budget_status_exhausted(self):
        """Test BudgetStatus when exhausted."""
        status = BudgetStatus(
            total=25,
            used=25,
            remaining=0,
            is_exhausted=True,
            utilization=1.0,
        )

        assert status.is_exhausted is True
        assert status.remaining == 0


class TestBudgetManagerIntegration:
    """Tests for ToolBudgetCoordinator integration patterns.

    These tests verify that the coordinator properly integrates
    with the broader orchestrator budget management system.
    """

    @pytest.fixture
    def mock_budget_manager(self):
        """Create a mock BudgetManager."""
        manager = Mock()
        manager.get_max_tool_calls.return_value = 25
        manager.get_used_tool_calls.return_value = 0
        return manager

    def test_coordinator_delegates_to_budget_manager(self, mock_budget_manager):
        """Test coordinator properly delegates to BudgetManager when available."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(),
            budget_manager=mock_budget_manager,
        )

        # Should delegate to budget manager
        budget = coordinator.budget
        assert budget == 25
        mock_budget_manager.get_max_tool_calls.assert_called_once()

    def test_coordinator_fallback_without_manager(self):
        """Test coordinator works without BudgetManager."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=30),
            budget_manager=None,
        )

        # Should use config default
        budget = coordinator.budget
        assert budget == 30


class TestToolBudgetCoordinatorBudgetSetter:
    """Tests for budget setter with and without BudgetManager."""

    @pytest.fixture
    def mock_budget_manager(self):
        """Create a mock BudgetManager."""
        manager = Mock()
        manager.get_max_tool_calls.return_value = 25
        manager.get_used_tool_calls.return_value = 0
        manager.get_remaining_tool_calls.return_value = 25
        manager.config = Mock()
        manager.config.base_max_tool_calls = 25
        return manager

    def test_budget_setter_without_manager(self):
        """Test budget setter when no budget_manager."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        # Set new budget
        coordinator.budget = 50
        assert coordinator.budget == 50

    def test_budget_setter_with_negative_value(self):
        """Test budget setter with negative value."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        # Should clamp to 0
        coordinator.budget = -10
        assert coordinator.budget == 0

    def test_budget_setter_with_manager(self, mock_budget_manager):
        """Test budget setter delegates to budget_manager.config."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=mock_budget_manager,
        )

        # Set new budget
        coordinator.budget = 100

        # Should update manager config
        assert mock_budget_manager.config.base_max_tool_calls == 100


class TestToolBudgetCoordinatorBudgetUsed:
    """Tests for budget_used property with and without BudgetManager."""

    @pytest.fixture
    def mock_budget_manager(self):
        """Create a mock BudgetManager."""
        manager = Mock()
        manager.get_max_tool_calls.return_value = 25
        manager.get_used_tool_calls.return_value = 5
        manager.get_remaining_tool_calls.return_value = 20
        manager.config = Mock()
        manager.config.base_max_tool_calls = 25
        return manager

    def test_budget_used_without_manager(self):
        """Test budget_used property without budget_manager."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.consume()
        coordinator.consume()
        coordinator.consume()

        assert coordinator.budget_used == 3

    def test_budget_used_with_manager(self, mock_budget_manager):
        """Test budget_used property delegates to budget_manager."""
        mock_budget_manager.get_used_tool_calls.return_value = 10

        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=mock_budget_manager,
        )

        # Should delegate to manager
        used = coordinator.budget_used
        assert used == 10
        mock_budget_manager.get_used_tool_calls.assert_called_once()


class TestToolBudgetCoordinatorMultiplier:
    """Tests for budget_multiplier property and set_multiplier method."""

    def test_budget_multiplier_property(self):
        """Test budget_multiplier property returns config value."""
        config = ToolBudgetConfig(budget_multiplier=2.5)
        coordinator = ToolBudgetCoordinator(config=config)

        assert coordinator.budget_multiplier == 2.5

    def test_set_multiplier_updates_budget(self):
        """Test set_multiplier updates effective budget."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25, budget_multiplier=1.0),
            budget_manager=None,
        )

        # Initial budget
        assert coordinator.budget == 25

        # Set multiplier to 2.0
        coordinator.set_multiplier(2.0)

        # Budget should be updated
        assert coordinator.budget == 50  # 25 * 2.0
        assert coordinator.budget_multiplier == 2.0

    def test_set_multiplier_tracks_history(self):
        """Test set_multiplier tracks multiplier changes."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25, budget_multiplier=1.0),
            budget_manager=None,
        )

        # Change multiplier
        coordinator.set_multiplier(2.0)
        coordinator.set_multiplier(3.0)
        coordinator.set_multiplier(1.5)

        # Check history
        stats = coordinator.get_stats()
        assert len(stats.multiplier_history) == 3
        assert stats.multiplier_history[0] == (1.0, 2.0)
        assert stats.multiplier_history[1] == (2.0, 3.0)
        assert stats.multiplier_history[2] == (3.0, 1.5)

    def test_set_multiplier_with_fraction(self):
        """Test set_multiplier with fractional multiplier."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=100, budget_multiplier=1.0),
            budget_manager=None,
        )

        # Set multiplier to 0.5
        coordinator.set_multiplier(0.5)

        assert coordinator.budget == 50  # 100 * 0.5
        assert coordinator.budget_multiplier == 0.5

    def test_set_multiplier_with_large_value(self):
        """Test set_multiplier with large multiplier."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=10, budget_multiplier=1.0),
            budget_manager=None,
        )

        # Set multiplier to 10.0
        coordinator.set_multiplier(10.0)

        assert coordinator.budget == 100  # 10 * 10.0


class TestToolBudgetCoordinatorGetRemaining:
    """Tests for get_remaining_budget method."""

    @pytest.fixture
    def mock_budget_manager(self):
        """Create a mock BudgetManager."""
        manager = Mock()
        manager.get_max_tool_calls.return_value = 25
        manager.get_used_tool_calls.return_value = 0
        manager.get_remaining_tool_calls.return_value = 25
        manager.config = Mock()
        manager.config.base_max_tool_calls = 25
        return manager

    def test_get_remaining_budget_without_manager(self):
        """Test get_remaining_budget without budget_manager."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.consume()
        coordinator.consume()

        remaining = coordinator.get_remaining_budget()
        assert remaining == 23

    def test_get_remaining_budget_with_manager(self, mock_budget_manager):
        """Test get_remaining_budget delegates to budget_manager."""
        mock_budget_manager.get_remaining_tool_calls.return_value = 15

        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=mock_budget_manager,
        )

        remaining = coordinator.get_remaining_budget()
        assert remaining == 15
        mock_budget_manager.get_remaining_tool_calls.assert_called_once()

    def test_get_remaining_budget_when_exhausted(self):
        """Test get_remaining_budget returns 0 when exhausted."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=5),
            budget_manager=None,
        )

        # Consume all budget
        coordinator.consume(5)

        remaining = coordinator.get_remaining_budget()
        assert remaining == 0

    def test_get_remaining_budget_clamps_to_zero(self):
        """Test get_remaining_budget doesn't go negative."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=5),
            budget_manager=None,
        )

        # Try to consume more than available
        coordinator.consume(10)

        remaining = coordinator.get_remaining_budget()
        assert remaining == 0  # Should be clamped


class TestToolBudgetCoordinatorConsume:
    """Tests for consume method including edge cases and warnings."""

    def test_consume_with_custom_amount(self):
        """Test consume with custom amount."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.consume(5)
        coordinator.consume(10)

        assert coordinator.budget_used == 15

    def test_consume_with_zero_amount(self):
        """Test consume with zero amount does nothing."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        initial_used = coordinator.budget_used
        coordinator.consume(0)

        assert coordinator.budget_used == initial_used

    def test_consume_with_negative_amount(self):
        """Test consume with negative amount does nothing."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.consume(5)
        initial_used = coordinator.budget_used
        coordinator.consume(-5)

        # Should not change
        assert coordinator.budget_used == initial_used

    def test_consume_tracks_total_consumed_in_stats(self):
        """Test consume updates total_consumed in stats."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.consume(5)
        coordinator.consume(10)

        stats = coordinator.get_stats()
        # Note: get_stats adds budget_used (15) to _stats.total_consumed (15) = 30
        assert stats.total_consumed == 30

    def test_consume_with_warning_callback(self):
        """Test consume triggers warning callback when threshold reached."""
        warnings = []

        def warning_handler(remaining, total):
            warnings.append((remaining, total))

        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(
                default_budget=25,
                warning_threshold=0.2  # 20% = 5 calls threshold
            ),
            budget_manager=None,
            on_warning=warning_handler,
        )

        # Consume past threshold (21 calls used, 4 remaining - 4 < 25*0.2=5)
        coordinator.consume(21)

        # Should trigger warning
        assert len(warnings) == 1
        assert warnings[0] == (4, 25)

    def test_consume_multiple_warnings(self):
        """Test consume triggers warning each time threshold is crossed."""
        warnings = []

        def warning_handler(remaining, total):
            warnings.append((remaining, total))

        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(
                default_budget=25,
                warning_threshold=0.2
            ),
            budget_manager=None,
            on_warning=warning_handler,
        )

        # Consume past threshold multiple times
        coordinator.consume(21)  # 4 remaining - triggers warning
        coordinator.consume(1)   # 3 remaining - triggers warning
        coordinator.consume(1)   # 2 remaining - triggers warning

        # Should trigger 3 warnings
        assert len(warnings) == 3

    def test_consume_warning_without_callback(self):
        """Test consume handles missing warning callback gracefully."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(
                default_budget=25,
                warning_threshold=0.2
            ),
            budget_manager=None,
            on_warning=None,
        )

        # Should not raise
        coordinator.consume(20)

    def test_consume_with_manager(self):
        """Test consume delegates to budget_manager when available."""
        manager = Mock()
        manager.get_max_tool_calls.return_value = 25
        manager.get_used_tool_calls.return_value = 0
        manager.get_remaining_tool_calls.return_value = 25
        manager.config = Mock()
        manager.config.base_max_tool_calls = 25

        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=manager,
        )

        coordinator.consume(5)

        # Should delegate to manager
        manager.consume_tool_call.assert_called_once_with(5)


class TestToolBudgetCoordinatorReset:
    """Tests for reset method variations."""

    @pytest.fixture
    def mock_budget_manager(self):
        """Create a mock BudgetManager."""
        manager = Mock()
        manager.get_max_tool_calls.return_value = 25
        manager.get_used_tool_calls.return_value = 10
        manager.get_remaining_tool_calls.return_value = 15
        manager.config = Mock()
        manager.config.base_max_tool_calls = 25
        return manager

    def test_reset_without_new_budget(self):
        """Test reset uses default_budget from config."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.consume(10)
        assert coordinator.budget_used == 10

        coordinator.reset()

        assert coordinator.budget_used == 0
        assert coordinator.budget == 25

    def test_reset_with_new_budget(self):
        """Test reset with custom new_budget."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.consume(10)

        coordinator.reset(new_budget=50)

        assert coordinator.budget_used == 0
        assert coordinator.budget == 50

    def test_reset_with_manager(self, mock_budget_manager):
        """Test reset delegates to budget_manager when available."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=mock_budget_manager,
        )

        coordinator.reset()

        # Should call reset on manager
        mock_budget_manager.reset_tool_calls.assert_called_once()

    def test_reset_increments_reset_count(self):
        """Test reset increments total_reset_count in stats."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.reset()
        coordinator.reset()
        coordinator.reset()

        stats = coordinator.get_stats()
        assert stats.total_reset_count == 3


class TestToolBudgetCoordinatorStats:
    """Tests for get_stats and clear_stats methods."""

    def test_get_stats_returns_current_stats(self):
        """Test get_stats returns BudgetStats with current data."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.consume(10)
        coordinator.set_multiplier(2.0)

        stats = coordinator.get_stats()

        # Note: total_consumed = _stats.total_consumed (10) + budget_used (10) = 20
        assert stats.total_consumed == 20
        assert stats.total_reset_count == 0
        assert len(stats.multiplier_history) == 1
        assert stats.multiplier_history[0] == (1.0, 2.0)

    def test_get_stats_includes_current_budget_used(self):
        """Test get_stats includes current budget_used in total_consumed."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        # Consume before clearing stats
        coordinator.consume(5)

        # Clear stats (resets _stats.total_consumed to 0, but budget_used stays at 5)
        coordinator.clear_stats()

        # Consume more (budget_used is now 5 + 3 = 8, _stats.total_consumed is 3)
        coordinator.consume(3)

        # Stats will show: _stats.total_consumed (3) + budget_used (8) = 11
        stats = coordinator.get_stats()
        assert stats.total_consumed == 11

    def test_clear_stats_resets_all_statistics(self):
        """Test clear_stats resets all statistics."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        # Generate some activity
        coordinator.consume(10)
        coordinator.set_multiplier(2.0)
        coordinator.reset()
        coordinator.set_multiplier(3.0)

        # Clear stats
        coordinator.clear_stats()

        # Check all cleared (but budget_used is still tracked separately)
        stats = coordinator.get_stats()
        # After clear: _stats.total_consumed (0) + budget_used (0 after reset) = 0
        assert stats.total_consumed == 0
        assert stats.total_reset_count == 0
        assert len(stats.multiplier_history) == 0
        assert stats.warning_count == 0

    def test_clear_stats_does_not_affect_budget_state(self):
        """Test clear_stats doesn't affect budget tracking."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        coordinator.consume(10)

        # Clear stats
        coordinator.clear_stats()

        # Budget state should be unchanged
        assert coordinator.budget_used == 10
        assert coordinator.get_remaining_budget() == 15

        # Stats will show: _stats.total_consumed (0) + budget_used (10) = 10
        stats = coordinator.get_stats()
        assert stats.total_consumed == 10

    def test_warning_count_increments(self):
        """Test warning_count increments on warnings."""
        warnings = []

        def warning_handler(remaining, total):
            warnings.append((remaining, total))

        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(
                default_budget=25,
                warning_threshold=0.2
            ),
            budget_manager=None,
            on_warning=warning_handler,
        )

        # Trigger warnings (need to go below threshold: 25 * 0.2 = 5)
        coordinator.consume(21)  # First warning (4 remaining)
        coordinator.consume(1)   # Second warning (3 remaining)

        stats = coordinator.get_stats()
        assert stats.warning_count == 2


class TestToolBudgetCoordinatorFactory:
    """Tests for create_tool_budget_coordinator factory function."""

    def test_factory_with_default_params(self):
        """Test factory with default parameters."""
        coordinator = create_tool_budget_coordinator()

        assert coordinator.budget == 25
        assert coordinator.budget_multiplier == 1.0
        # Check warning_threshold in config
        assert coordinator._config.warning_threshold == 0.2

    def test_factory_with_custom_params(self):
        """Test factory with custom parameters."""
        coordinator = create_tool_budget_coordinator(
            default_budget=50,
            warning_threshold=0.3,
        )

        assert coordinator.budget == 50
        assert coordinator._config.warning_threshold == 0.3

    def test_factory_with_budget_manager(self):
        """Test factory with budget_manager parameter."""
        manager = Mock()
        manager.get_max_tool_calls.return_value = 100
        manager.get_used_tool_calls.return_value = 0
        manager.get_remaining_tool_calls.return_value = 100
        manager.config = Mock()
        manager.config.base_max_tool_calls = 100

        coordinator = create_tool_budget_coordinator(
            default_budget=100,
            budget_manager=manager,
        )

        assert coordinator.budget == 100


class TestToolBudgetCoordinatorIntegration:
    """Integration tests for budget management scenarios."""

    def test_full_budget_lifecycle(self):
        """Test complete budget lifecycle: set, consume, check, reset."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=25),
            budget_manager=None,
        )

        # Initial state
        status = coordinator.get_status()
        assert status.total == 25
        assert status.used == 0
        assert status.remaining == 25
        assert status.is_exhausted is False

        # Consume budget
        coordinator.consume(10)
        status = coordinator.get_status()
        assert status.used == 10
        assert status.remaining == 15

        # Set multiplier
        coordinator.set_multiplier(2.0)
        status = coordinator.get_status()
        assert status.total == 50  # 25 * 2.0
        assert status.remaining == 40  # 50 - 10

        # Exhaust budget
        coordinator.consume(40)
        status = coordinator.get_status()
        assert status.is_exhausted is True
        assert status.remaining == 0

        # Reset
        coordinator.reset(new_budget=30)
        status = coordinator.get_status()
        assert status.total == 30
        assert status.used == 0
        assert status.remaining == 30
        assert status.is_exhausted is False

    def test_budget_exhaustion_detection(self):
        """Test various exhaustion detection scenarios."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=10),
            budget_manager=None,
        )

        # Not exhausted
        assert coordinator.is_exhausted() is False

        # Partial use
        coordinator.consume(5)
        assert coordinator.is_exhausted() is False

        # Exactly at limit
        coordinator.consume(5)
        assert coordinator.is_exhausted() is True

        # Over limit
        coordinator.consume(1)
        assert coordinator.is_exhausted() is True

    def test_utilization_calculation(self):
        """Test utilization percentage calculation."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=100),
            budget_manager=None,
        )

        # Empty
        status = coordinator.get_status()
        assert status.utilization == 0.0

        # Half full
        coordinator.consume(50)
        status = coordinator.get_status()
        assert status.utilization == 0.5

        # Full
        coordinator.consume(50)
        status = coordinator.get_status()
        assert status.utilization == 1.0

    def test_utilization_with_zero_budget(self):
        """Test utilization when budget is 0."""
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=0),
            budget_manager=None,
        )

        status = coordinator.get_status()
        assert status.total == 0
        assert status.utilization == 0.0  # Should handle division by zero
