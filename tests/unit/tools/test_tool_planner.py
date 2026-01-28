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

"""Unit tests for ToolPlanner.

Tests tool planning, goal inference, and intent-based filtering.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Any

from victor.agent.tool_planner import ToolPlanner
from victor.config.settings import Settings
from victor.tools.auth_metadata import (
    ToolAuthMetadata,
    ToolAuthMetadataRegistry,
    ToolSafety,
)


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    return settings


@pytest.fixture(autouse=True)
def register_test_tool_metadata():
    """Register tool auth metadata for test tools.

    This fixture ensures that test tools like 'write_file', 'read_file', etc.
    have proper metadata registered for intent-based filtering tests.
    """
    registry = ToolAuthMetadataRegistry.get_instance()

    # Register common test tools with appropriate metadata
    test_tools_metadata = [
        ToolAuthMetadata(
            name="write_file",
            categories=["file_ops", "write"],
            capabilities=["file_write"],
            safety=ToolSafety.REQUIRES_CONFIRMATION,
            domain="coding",
        ),
        ToolAuthMetadata(
            name="edit_files",
            categories=["file_ops", "write", "refactor"],
            capabilities=["file_write"],
            safety=ToolSafety.REQUIRES_CONFIRMATION,
            domain="coding",
        ),
        ToolAuthMetadata(
            name="read_file",
            categories=["file_ops", "read"],
            capabilities=["file_read"],
            safety=ToolSafety.SAFE,
            domain="coding",
        ),
        ToolAuthMetadata(
            name="list_directory",
            categories=["file_ops", "read"],
            capabilities=["file_read"],
            safety=ToolSafety.SAFE,
            domain="coding",
        ),
        ToolAuthMetadata(
            name="execute_bash",
            categories=["execution", "shell"],
            capabilities=["shell", "execute"],
            safety=ToolSafety.DESTRUCTIVE,
            domain="coding",
        ),
    ]

    # Register all test tools
    for metadata in test_tools_metadata:
        registry.register(metadata)

    yield

    # Cleanup: clear test tool metadata from registry
    # Note: Registry doesn't have unregister(), so we just clear and re-register any existing tools
    registry._metadata.clear()


@pytest.fixture
def mock_tool_registrar():
    """Create mock ToolRegistrar."""
    registrar = MagicMock()

    # Configure return values
    registrar.plan_tools.return_value = [
        Mock(name="read_file", outputs=["file_contents"]),
        Mock(name="code_review", outputs=["review"]),
    ]
    registrar.infer_goals_from_message.return_value = ["summary", "documentation"]

    return registrar


@pytest.fixture
def tool_planner(mock_tool_registrar, mock_settings):
    """Create ToolPlanner with mocked dependencies."""
    return ToolPlanner(
        tool_registrar=mock_tool_registrar,
        settings=mock_settings,
    )


class TestToolPlanning:
    """Tests for tool planning operations."""

    def test_plan_tools(self, tool_planner, mock_tool_registrar):
        """Test planning tool sequence to achieve goals."""
        goals = ["summary", "documentation"]
        available_inputs = ["file_contents"]

        # Configure mocks with name attribute
        tool1 = Mock()
        tool1.name = "read_file"
        tool2 = Mock()
        tool2.name = "code_review"
        mock_tool_registrar.plan_tools.return_value = [tool1, tool2]

        result = tool_planner.plan_tools(goals, available_inputs)

        # Verify delegation to tool registrar
        mock_tool_registrar.plan_tools.assert_called_once_with(goals, available_inputs)

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].name == "read_file"
        assert result[1].name == "code_review"

    def test_plan_tools_no_available_inputs(self, tool_planner, mock_tool_registrar):
        """Test planning tools without available inputs."""
        goals = ["test_results"]

        result = tool_planner.plan_tools(goals)

        # Verify delegation with None for available_inputs
        mock_tool_registrar.plan_tools.assert_called_once_with(goals, None)
        assert isinstance(result, list)

    def test_plan_tools_empty_goals(self, tool_planner, mock_tool_registrar):
        """Test planning with empty goals list."""
        result = tool_planner.plan_tools([])

        mock_tool_registrar.plan_tools.assert_called_once_with([], None)

    def test_infer_goals_from_message(self, tool_planner, mock_tool_registrar):
        """Test inferring goals from user message."""
        message = "Please analyze this code and generate documentation"

        result = tool_planner.infer_goals_from_message(message)

        # Verify delegation
        mock_tool_registrar.infer_goals_from_message.assert_called_once_with(message)

        # Verify result
        assert isinstance(result, list)
        assert "summary" in result
        assert "documentation" in result

    def test_infer_goals_from_empty_message(self, tool_planner, mock_tool_registrar):
        """Test inferring goals from empty message."""
        mock_tool_registrar.infer_goals_from_message.return_value = []

        result = tool_planner.infer_goals_from_message("")

        mock_tool_registrar.infer_goals_from_message.assert_called_once_with("")
        assert result == []


class TestIntentFiltering:
    """Tests for intent-based tool filtering."""

    def test_filter_tools_no_intent(self, tool_planner):
        """Test filtering with no intent (no filtering)."""
        tools = [
            Mock(name="read_file"),
            Mock(name="write_file"),
            Mock(name="execute_bash"),
        ]

        result = tool_planner.filter_tools_by_intent(tools, None)

        # Should return all tools when no intent
        assert len(result) == 3
        assert result == tools

    def test_filter_tools_display_only_intent(self, tool_planner):
        """Test filtering with DISPLAY_ONLY intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Create mocks with name attribute properly set
        tool1 = Mock()
        tool1.name = "read_file"
        tool2 = Mock()
        tool2.name = "write_file"
        tool3 = Mock()
        tool3.name = "list_directory"
        tools = [tool1, tool2, tool3]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.DISPLAY_ONLY)

        # write_file should be filtered out
        assert len(result) < len(tools)
        tool_names = [t.name for t in result]
        assert "read_file" in tool_names
        assert "list_directory" in tool_names
        # write_file should be filtered
        assert "write_file" not in tool_names

    def test_filter_tools_read_only_intent(self, tool_planner):
        """Test filtering with READ_ONLY intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Create mocks with name attribute properly set
        tool1 = Mock()
        tool1.name = "read_file"
        tool2 = Mock()
        tool2.name = "write_file"
        tool3 = Mock()
        tool3.name = "edit_files"
        tools = [tool1, tool2, tool3]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.READ_ONLY)

        # write and edit tools should be filtered out
        assert len(result) < len(tools)
        tool_names = [t.name for t in result]
        assert "read_file" in tool_names
        assert "write_file" not in tool_names
        assert "edit_files" not in tool_names

    def test_filter_tools_write_allowed_intent(self, tool_planner):
        """Test filtering with WRITE_ALLOWED intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Create mocks with name attribute properly set
        tool1 = Mock()
        tool1.name = "read_file"
        tool2 = Mock()
        tool2.name = "write_file"
        tool3 = Mock()
        tool3.name = "execute_bash"
        tools = [tool1, tool2, tool3]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.WRITE_ALLOWED)

        # No filtering for WRITE_ALLOWED
        assert len(result) == len(tools)
        assert result == tools

    def test_filter_tools_with_dict_tools(self, tool_planner):
        """Test filtering with dict-based tool definitions."""
        from victor.agent.action_authorizer import ActionIntent

        tools = [
            {"name": "read_file"},
            {"name": "write_file"},
            {"name": "list_directory"},
        ]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.DISPLAY_ONLY)

        # write_file should be filtered out
        assert len(result) < len(tools)
        tool_names = [t["name"] for t in result]
        assert "read_file" in tool_names
        assert "write_file" not in tool_names

    def test_filter_tools_empty_list(self, tool_planner):
        """Test filtering with empty tools list."""
        from victor.agent.action_authorizer import ActionIntent

        result = tool_planner.filter_tools_by_intent([], ActionIntent.DISPLAY_ONLY)

        assert result == []

    def test_filter_tools_ambiguous_intent(self, tool_planner):
        """Test filtering with AMBIGUOUS intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Create mocks with name attribute properly set
        tool1 = Mock()
        tool1.name = "read_file"
        tool2 = Mock()
        tool2.name = "write_file"
        tools = [tool1, tool2]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.AMBIGUOUS)

        # AMBIGUOUS should not filter
        assert len(result) == len(tools)


class TestToolPlannerInitialization:
    """Tests for ToolPlanner initialization."""

    def test_initialization_with_valid_dependencies(self, mock_tool_registrar, mock_settings):
        """Test successful initialization with valid dependencies."""
        tool_planner = ToolPlanner(
            tool_registrar=mock_tool_registrar,
            settings=mock_settings,
        )

        assert tool_planner.tool_registrar is mock_tool_registrar
        assert tool_planner.settings is mock_settings

    def test_initialization_stores_dependencies(self, mock_tool_registrar, mock_settings):
        """Test that initialization stores all dependencies."""
        tool_planner = ToolPlanner(
            tool_registrar=mock_tool_registrar,
            settings=mock_settings,
        )

        # Verify dependencies are accessible
        assert hasattr(tool_planner, "tool_registrar")
        assert hasattr(tool_planner, "settings")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_tool_planner_with_none_registrar(self, mock_settings):
        """Test ToolPlanner handles None tool registrar gracefully."""
        tool_planner = ToolPlanner(
            tool_registrar=None,
            settings=mock_settings,
        )

        # Attempting to use it should fail
        with pytest.raises(AttributeError):
            tool_planner.plan_tools(["summary"])

    def test_filter_tools_with_invalid_tool_format(self, tool_planner):
        """Test filtering with tools that have invalid format."""
        from victor.agent.action_authorizer import ActionIntent

        # Tools without name attribute or key
        tools = [
            "invalid_tool",  # String instead of object/dict
            123,  # Number
            None,  # None
        ]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.DISPLAY_ONLY)

        # Should handle gracefully (filter out invalid tools)
        assert isinstance(result, list)
