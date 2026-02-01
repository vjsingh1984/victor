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

"""Tests for ToolCoordinator.

Tests the tool coordination functionality including:
- Tool selection coordination
- Budget management
- Tool execution coordination
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.agent.coordinators import (
    ToolCoordinator,
    ToolCoordinatorConfig,
    TaskContext,
    create_tool_coordinator,
)


class TestToolCoordinatorConfig:
    """Tests for ToolCoordinatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ToolCoordinatorConfig()

        assert config.default_budget == 25
        assert config.budget_multiplier == 1.0
        assert config.enable_caching is True
        assert config.max_tools_per_selection == 15
        assert config.selection_threshold == 0.3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ToolCoordinatorConfig(
            default_budget=50,
            budget_multiplier=2.0,
            enable_caching=False,
            max_tools_per_selection=10,
        )

        assert config.default_budget == 50
        assert config.budget_multiplier == 2.0
        assert config.enable_caching is False
        assert config.max_tools_per_selection == 10


class TestTaskContext:
    """Tests for TaskContext."""

    def test_default_context(self):
        """Test default context values."""
        context = TaskContext(message="test message")

        assert context.message == "test message"
        assert context.task_type == "unknown"
        assert context.complexity == "medium"
        assert context.stage is None
        assert context.observed_files == set()
        assert context.executed_tools == set()

    def test_custom_context(self):
        """Test custom context values."""
        context = TaskContext(
            message="fix the bug",
            task_type="bugfix",
            complexity="high",
            stage="execution",
            observed_files={"file1.py", "file2.py"},
            executed_tools={"read", "search"},
        )

        assert context.message == "fix the bug"
        assert context.task_type == "bugfix"
        assert context.complexity == "high"
        assert context.stage == "execution"
        assert context.observed_files == {"file1.py", "file2.py"}
        assert context.executed_tools == {"read", "search"}


class TestToolCoordinator:
    """Tests for ToolCoordinator."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock tool pipeline."""
        pipeline = MagicMock()
        pipeline.execute_tool_calls = AsyncMock()
        return pipeline

    @pytest.fixture
    def mock_selector(self):
        """Create mock tool selector."""
        selector = MagicMock()
        selector.select_tools = AsyncMock(return_value=[])
        return selector

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        registry = MagicMock()
        registry.get_tool = MagicMock(return_value=None)
        registry.list_tools = MagicMock(return_value=[])
        return registry

    @pytest.fixture
    def coordinator(self, mock_pipeline, mock_registry, mock_selector):
        """Create coordinator with mocks."""
        return ToolCoordinator(
            tool_pipeline=mock_pipeline,
            tool_registry=mock_registry,
            tool_selector=mock_selector,
        )

    def test_init_default_config(self, mock_pipeline, mock_registry):
        """Test initialization with default config."""
        coordinator = ToolCoordinator(
            tool_pipeline=mock_pipeline,
            tool_registry=mock_registry,
        )

        assert coordinator.budget == 25
        assert coordinator.budget_used == 0
        assert coordinator.execution_count == 0

    def test_init_custom_config(self, mock_pipeline, mock_registry):
        """Test initialization with custom config."""
        config = ToolCoordinatorConfig(default_budget=50)
        coordinator = ToolCoordinator(
            tool_pipeline=mock_pipeline,
            tool_registry=mock_registry,
            config=config,
        )

        assert coordinator.budget == 50

    def test_get_remaining_budget(self, coordinator):
        """Test remaining budget calculation."""
        assert coordinator.get_remaining_budget() == 25

        coordinator.consume_budget(10)
        assert coordinator.get_remaining_budget() == 15

    def test_consume_budget(self, coordinator):
        """Test budget consumption."""
        coordinator.consume_budget(5)
        assert coordinator.budget_used == 5

        coordinator.consume_budget(3)
        assert coordinator.budget_used == 8

    def test_reset_budget(self, coordinator):
        """Test budget reset."""
        coordinator.consume_budget(20)
        assert coordinator.budget_used == 20

        coordinator.reset_budget()
        assert coordinator.budget_used == 0
        assert coordinator.budget == 25

    def test_reset_budget_with_new_value(self, coordinator):
        """Test budget reset with new value."""
        coordinator.consume_budget(20)
        coordinator.reset_budget(new_budget=50)

        assert coordinator.budget_used == 0
        assert coordinator.budget == 50

    def test_set_budget_multiplier(self, coordinator):
        """Test setting budget multiplier."""
        coordinator.set_budget_multiplier(2.0)

        # Default budget (25) * 2.0 = 50
        assert coordinator.budget == 50

    def test_is_budget_exhausted(self, coordinator):
        """Test budget exhaustion check."""
        assert coordinator.is_budget_exhausted() is False

        coordinator.consume_budget(25)
        assert coordinator.is_budget_exhausted() is True

    @pytest.mark.asyncio
    async def test_select_tools(self, coordinator, mock_selector):
        """Test tool selection."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_selector.select_tools.return_value = mock_tools

        context = TaskContext(message="search for file")
        result = await coordinator.select_tools(context)

        assert result == mock_tools
        mock_selector.select_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_tools_no_selector(self, mock_pipeline, mock_registry):
        """Test tool selection without selector configured."""
        coordinator = ToolCoordinator(
            tool_pipeline=mock_pipeline,
            tool_registry=mock_registry,
        )

        context = TaskContext(message="search for file")
        result = await coordinator.select_tools(context)

        assert result == []

    @pytest.mark.asyncio
    async def test_execute_tool_calls(self, coordinator, mock_pipeline):
        """Test tool execution."""
        mock_result = MagicMock()
        mock_result.successful_calls = 2
        mock_pipeline.execute_tool_calls.return_value = mock_result

        tool_calls = [{"name": "read", "args": {}}, {"name": "search", "args": {}}]
        result = await coordinator.execute_tool_calls(tool_calls)

        assert result == mock_result
        assert coordinator.budget_used == 2
        assert coordinator.execution_count == 2

    def test_get_selection_stats(self, coordinator):
        """Test selection statistics."""
        stats = coordinator.get_selection_stats()

        assert "total_selections" in stats
        assert "total_tools_selected" in stats
        assert "method_distribution" in stats
        assert stats["total_selections"] == 0

    def test_get_execution_stats(self, coordinator):
        """Test execution statistics."""
        coordinator.consume_budget(5)

        stats = coordinator.get_execution_stats()

        assert stats["total_executions"] == 0
        assert stats["budget_used"] == 5
        assert stats["budget_total"] == 25
        assert stats["budget_remaining"] == 20

    def test_budget_warning_callback(self, mock_pipeline, mock_registry):
        """Test budget warning callback."""
        warning_called = []

        def on_warning(remaining, total):
            warning_called.append((remaining, total))

        coordinator = ToolCoordinator(
            tool_pipeline=mock_pipeline,
            tool_registry=mock_registry,
            on_budget_warning=on_warning,
        )

        # Consume enough to trigger warning (< 20% remaining)
        coordinator.consume_budget(21)

        assert len(warning_called) == 1
        assert warning_called[0] == (4, 25)


class TestCreateToolCoordinator:
    """Tests for create_tool_coordinator factory function."""

    def test_create_basic(self):
        """Test basic factory creation."""
        mock_pipeline = MagicMock()
        mock_registry = MagicMock()
        coordinator = create_tool_coordinator(
            tool_pipeline=mock_pipeline,
            tool_registry=mock_registry,
        )

        assert isinstance(coordinator, ToolCoordinator)
        assert coordinator.budget == 25

    def test_create_with_config(self):
        """Test factory creation with config."""
        mock_pipeline = MagicMock()
        mock_registry = MagicMock()
        config = ToolCoordinatorConfig(default_budget=100)

        coordinator = create_tool_coordinator(
            tool_pipeline=mock_pipeline,
            tool_registry=mock_registry,
            config=config,
        )

        assert coordinator.budget == 100
