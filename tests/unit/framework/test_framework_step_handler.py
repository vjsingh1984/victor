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

"""Unit tests for FrameworkStepHandler apply methods."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any, List, Optional

from victor.core.verticals.base import VerticalBase
from victor.agent.vertical_context import VerticalContext


class MockVerticalWithHandlers(VerticalBase):
    """Mock vertical that provides handlers."""

    name = "test_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        return {
            "test_handler": MagicMock(),
            "another_handler": MagicMock(),
        }


class MockVerticalWithToolGraph(VerticalBase):
    """Mock vertical that provides tool graph."""

    name = "test_graph_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_tool_graph(cls) -> Any:
        return MagicMock(name="MockToolGraph")


class MockVerticalWithWorkflows(VerticalBase):
    """Mock vertical that provides workflows with auto triggers."""

    name = "test_workflow_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_workflow_provider(cls) -> Any:
        provider = MagicMock()
        provider.get_workflows.return_value = {"workflow1": MagicMock()}
        provider.get_auto_workflows.return_value = [
            (r"test.*pattern", "workflow1"),
        ]
        return provider


class MockVerticalWithTeams(VerticalBase):
    """Mock vertical that provides team specs."""

    name = "test_team_vertical"

    @classmethod
    def get_tools(cls) -> List[str]:
        return ["read"]

    @classmethod
    def get_system_prompt(cls) -> str:
        return "Test"

    @classmethod
    def get_team_specs(cls) -> Dict[str, Any]:
        return {
            "team1": MagicMock(),
            "team2": MagicMock(),
        }


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator."""
    orchestrator = MagicMock()
    orchestrator.set_team_specs = MagicMock()
    return orchestrator


@pytest.fixture
def mock_context():
    """Create a mock vertical context."""
    context = MagicMock(spec=VerticalContext)
    context.apply_workflows = MagicMock()
    context.apply_team_specs = MagicMock()
    return context


@pytest.fixture
def mock_result():
    """Create a mock integration result."""
    result = MagicMock()
    result.add_info = MagicMock()
    result.add_warning = MagicMock()
    result.workflows_count = 0
    result.team_specs_count = 0
    return result


@pytest.fixture
def reset_registries():
    """Reset all registries before and after test."""
    from victor.framework.handler_registry import HandlerRegistry

    HandlerRegistry.reset_instance()
    yield
    HandlerRegistry.reset_instance()


class TestApplyHandlers:
    """Tests for apply_handlers method."""

    def test_apply_handlers_registers_to_registry(
        self, mock_orchestrator, mock_context, mock_result, reset_registries
    ):
        """Test apply_handlers registers handlers to HandlerRegistry."""
        from victor.framework.step_handlers import FrameworkStepHandler
        from victor.framework.handler_registry import get_handler_registry

        handler = FrameworkStepHandler()
        handler.apply_handlers(
            mock_orchestrator,
            MockVerticalWithHandlers,
            mock_context,
            mock_result,
        )

        registry = get_handler_registry()
        assert registry.has("test_handler")
        assert registry.has("another_handler")
        assert registry.get_entry("test_handler").vertical == "test_vertical"

    def test_apply_handlers_adds_info_to_result(
        self, mock_orchestrator, mock_context, mock_result, reset_registries
    ):
        """Test apply_handlers adds info message to result."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()
        handler.apply_handlers(
            mock_orchestrator,
            MockVerticalWithHandlers,
            mock_context,
            mock_result,
        )

        mock_result.add_info.assert_called()
        call_args = str(mock_result.add_info.call_args)
        assert "handler" in call_args.lower()

    def test_apply_handlers_skips_vertical_without_handlers(
        self, mock_orchestrator, mock_context, mock_result, reset_registries
    ):
        """Test apply_handlers skips vertical without get_handlers."""
        from victor.framework.step_handlers import FrameworkStepHandler

        # Create a minimal vertical without get_handlers
        class MinimalVertical(VerticalBase):
            name = "minimal"

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "Test"

        handler = FrameworkStepHandler()
        # Should not raise
        handler.apply_handlers(
            mock_orchestrator,
            MinimalVertical,
            mock_context,
            mock_result,
        )


class TestApplyToolGraphs:
    """Tests for apply_tool_graphs method."""

    def test_apply_tool_graphs_registers_to_registry(
        self, mock_orchestrator, mock_context, mock_result
    ):
        """Test apply_tool_graphs registers graph to ToolGraphRegistry."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()

        with patch("victor.tools.tool_graph.ToolGraphRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.get_instance.return_value = mock_registry

            handler.apply_tool_graphs(
                mock_orchestrator,
                MockVerticalWithToolGraph,
                mock_context,
                mock_result,
            )

            mock_registry.register_graph.assert_called_once()
            call_args = mock_registry.register_graph.call_args
            assert call_args[0][0] == "test_graph_vertical"

    def test_apply_tool_graphs_skips_vertical_without_graph(
        self, mock_orchestrator, mock_context, mock_result
    ):
        """Test apply_tool_graphs skips vertical without tool graph."""
        from victor.framework.step_handlers import FrameworkStepHandler

        # VerticalBase returns None for get_tool_graph
        handler = FrameworkStepHandler()

        with patch("victor.tools.tool_graph.ToolGraphRegistry") as mock_registry_class:
            mock_registry = MagicMock()
            mock_registry_class.get_instance.return_value = mock_registry

            handler.apply_tool_graphs(
                mock_orchestrator,
                VerticalBase,  # Use base class which returns None
                mock_context,
                mock_result,
            )

            # Should not register since get_tool_graph returns None
            mock_registry.register_graph.assert_not_called()


class TestApplyWorkflowsWithTriggers:
    """Tests for workflow trigger registration in apply_workflows."""

    def test_apply_workflows_registers_triggers(self, mock_orchestrator, mock_context, mock_result):
        """Test apply_workflows registers triggers with WorkflowTriggerRegistry."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()

        # Use create=True since get_workflow_registry may not exist in the module
        with patch(
            "victor.workflows.registry.get_workflow_registry",
            create=True,
        ) as mock_wf_registry:
            mock_wf_registry.return_value = MagicMock()

            with patch(
                "victor.workflows.trigger_registry.get_trigger_registry",
                create=True,
            ) as mock_trigger:
                mock_trigger_registry = MagicMock()
                mock_trigger.return_value = mock_trigger_registry

                handler.apply_workflows(
                    mock_orchestrator,
                    MockVerticalWithWorkflows,
                    mock_context,
                    mock_result,
                )

                # Verify triggers were registered
                mock_trigger_registry.register_from_vertical.assert_called_once()
                call_args = mock_trigger_registry.register_from_vertical.call_args
                assert call_args[0][0] == "test_workflow_vertical"


class TestApplyTeamSpecsWithRegistry:
    """Tests for team spec registry registration in apply_team_specs."""

    def test_apply_team_specs_registers_to_global_registry(
        self, mock_orchestrator, mock_context, mock_result
    ):
        """Test apply_team_specs registers to TeamSpecRegistry."""
        from victor.framework.step_handlers import FrameworkStepHandler

        handler = FrameworkStepHandler()

        # Use create=True since get_team_registry may not exist
        with patch(
            "victor.framework.team_registry.get_team_registry",
            create=True,
        ) as mock_team:
            mock_team_registry = MagicMock()
            mock_team.return_value = mock_team_registry

            handler.apply_team_specs(
                mock_orchestrator,
                MockVerticalWithTeams,
                mock_context,
                mock_result,
            )

            # Verify teams were registered with global registry
            mock_team_registry.register_from_vertical.assert_called_once()
            call_args = mock_team_registry.register_from_vertical.call_args
            assert call_args[0][0] == "test_team_vertical"
