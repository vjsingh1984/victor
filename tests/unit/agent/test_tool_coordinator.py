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

"""Tests for ToolCoordinator (service-owned implementation).

Tests the tool coordination functionality including:
- Tool selection coordination
- Budget management
- Tool execution coordination
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.services.tool_compat import (
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
        pipeline.credit_tracking_service = MagicMock(pending_signals=0)
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

    def test_get_enabled_tools_delegates_to_bound_tool_service(self, coordinator):
        """Bound coordinators should delegate enabled-tool queries to ToolService."""
        service = MagicMock()
        service.get_enabled_tools.return_value = {"read", "grep"}
        coordinator.bind_tool_service(service)

        assert coordinator.get_enabled_tools() == {"read", "grep"}
        service.get_enabled_tools.assert_called_once_with()

    def test_bound_tool_service_budget_properties_use_live_totals(self, coordinator):
        """Bound coordinators should expose total and used budget from ToolService."""

        class BoundToolService:
            budget = 10
            budget_used = 4

            def get_tool_budget(self):
                return 6

            def get_remaining_budget(self):
                return 6

        coordinator.bind_tool_service(BoundToolService())

        assert coordinator.budget == 10
        assert coordinator.budget_used == 4
        assert coordinator.get_remaining_budget() == 6

    def test_bound_tool_service_budget_exhaustion_uses_live_remaining_budget(self, coordinator):
        """Bound coordinators should not rely on stale local exhaustion state."""

        class ExhaustedToolService:
            budget = 5
            budget_used = 5

            def get_tool_budget(self):
                return 0

            def get_remaining_budget(self):
                return 0

        coordinator.bind_tool_service(ExhaustedToolService())

        assert coordinator.is_budget_exhausted() is True

    def test_consume_budget(self, coordinator):
        """Test budget consumption."""
        coordinator.consume_budget(5)
        assert coordinator.budget_used == 5

        coordinator.consume_budget(3)
        assert coordinator.budget_used == 8

    @pytest.mark.asyncio
    async def test_execute_tool_with_retry_delegates_to_bound_tool_service(self, coordinator):
        """Bound coordinators should delegate retry execution to ToolService."""
        service = MagicMock()
        service.execute_tool_with_retry = AsyncMock(return_value=("result", True, None))
        coordinator.bind_tool_service(service)

        result = await coordinator.execute_tool_with_retry("read", {"path": "a.py"}, {})

        assert result == ("result", True, None)
        service.execute_tool_with_retry.assert_awaited_once()

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

    @pytest.mark.asyncio
    async def test_execute_tool_calls_forwards_credit_context(self, coordinator, mock_pipeline):
        """Additional context needed for credit attribution should reach the pipeline."""
        mock_result = MagicMock()
        mock_result.successful_calls = 1
        mock_pipeline.execute_tool_calls.return_value = mock_result

        tool_calls = [{"name": "read", "args": {}}]
        context = TaskContext(
            message="investigate failure",
            task_type="debug",
            additional_context={
                "agent_id": "researcher_1",
                "team_id": "team_debug",
                "session_id": "session-123",
            },
        )

        await coordinator.execute_tool_calls(tool_calls, context=context)

        mock_pipeline.execute_tool_calls.assert_awaited_once()
        forwarded_context = mock_pipeline.execute_tool_calls.await_args.kwargs["context"]
        assert forwarded_context["agent_id"] == "researcher_1"
        assert forwarded_context["team_id"] == "team_debug"
        assert forwarded_context["session_id"] == "session-123"

    def test_get_selection_stats(self, coordinator):
        """Test selection statistics via observability handler."""
        stats = coordinator._observability.get_selection_stats()

        assert "total_selections" in stats
        assert "total_tools_selected" in stats
        assert "method_distribution" in stats
        assert stats["total_selections"] == 0

    def test_get_execution_stats(self, coordinator):
        """Test execution statistics via observability handler."""
        coordinator.consume_budget(5)

        stats = coordinator._observability.get_execution_stats()

        assert stats["total_executions"] == 0
        assert stats["budget_used"] == 5
        assert stats["budget_total"] == 25
        assert stats["budget_remaining"] == 20

    def test_bound_tool_service_execution_stats_use_live_budget_values(self, coordinator):
        """Execution stats should reflect the bound ToolService budget state."""

        class BoundToolService:
            budget = 12
            budget_used = 5

            def get_tool_budget(self):
                return 7

            def get_remaining_budget(self):
                return 7

        coordinator.bind_tool_service(BoundToolService())

        stats = coordinator._observability.get_execution_stats()
        usage_stats = coordinator._observability.get_tool_usage_stats()

        assert stats["budget_used"] == 5
        assert stats["budget_total"] == 12
        assert stats["budget_remaining"] == 7
        assert stats["budget_utilization"] == pytest.approx(5 / 12)
        assert usage_stats["budget"] == {"total": 12, "used": 5, "remaining": 7}

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

    def test_get_available_tools_returns_tool_names(self, mock_pipeline, mock_selector):
        """ToolCoordinator should expose tool names, not tool instances."""
        read_tool = MagicMock()
        read_tool.name = "read"
        write_tool = MagicMock()
        write_tool.name = "write"
        registry = MagicMock()
        registry.list_tools.return_value = [read_tool, write_tool]

        coordinator = ToolCoordinator(
            tool_pipeline=mock_pipeline,
            tool_registry=registry,
            tool_selector=mock_selector,
        )

        assert coordinator.get_available_tools() == {"read", "write"}

    def test_is_tool_enabled_with_registry_tool_instances(self, mock_pipeline, mock_selector):
        """Enabled-tool fallback should work when registry.list_tools returns tool instances."""
        read_tool = MagicMock()
        read_tool.name = "read"
        write_tool = MagicMock()
        write_tool.name = "write"
        registry = MagicMock()
        registry.list_tools.return_value = [read_tool, write_tool]

        coordinator = ToolCoordinator(
            tool_pipeline=mock_pipeline,
            tool_registry=registry,
            tool_selector=mock_selector,
        )

        assert coordinator.is_tool_enabled("read") is True
        coordinator.set_enabled_tools({"read"})
        assert coordinator.is_tool_enabled("write") is False


class TestCreateToolCoordinator:
    """Tests for create_tool_coordinator factory function."""

    def test_package_export_warns(self):
        """Package-root ToolCoordinator exports are compatibility-only."""
        with pytest.warns(
            DeprecationWarning,
            match="victor.agent.coordinators.ToolCoordinator is deprecated compatibility surface",
        ):
            from victor.agent.coordinators import ToolCoordinator as package_tool_coordinator

        with pytest.warns(
            DeprecationWarning,
            match=(
                "victor.agent.coordinators.create_tool_coordinator is deprecated "
                "compatibility surface"
            ),
        ):
            from victor.agent.coordinators import (
                create_tool_coordinator as package_create_tool_coordinator,
            )

        assert package_tool_coordinator is ToolCoordinator
        assert package_create_tool_coordinator is create_tool_coordinator

    def test_create_basic(self):
        """Test basic factory creation."""
        mock_pipeline = MagicMock()
        mock_registry = MagicMock()
        with pytest.warns(
            DeprecationWarning,
            match="deprecated ToolCoordinator shim",
        ):
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

        with pytest.warns(
            DeprecationWarning,
            match="deprecated ToolCoordinator shim",
        ):
            coordinator = create_tool_coordinator(
                tool_pipeline=mock_pipeline,
                tool_registry=mock_registry,
                config=config,
            )

        assert coordinator.budget == 100

    def test_create_binds_mode_controller_and_tool_service(self):
        """Deprecated factory still binds runtime collaborators for compatibility."""
        mock_pipeline = MagicMock()
        mock_registry = MagicMock()
        mode_controller = MagicMock()
        tool_service = MagicMock()

        with pytest.warns(
            DeprecationWarning,
            match="deprecated ToolCoordinator shim",
        ):
            coordinator = create_tool_coordinator(
                tool_pipeline=mock_pipeline,
                tool_registry=mock_registry,
                mode_controller=mode_controller,
                tool_service=tool_service,
            )

        assert coordinator._mode_controller is mode_controller
        assert coordinator._tool_service is tool_service
