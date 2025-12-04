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

"""Unit tests for workflows base module."""

import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

from victor.workflows.base import BaseWorkflow, WorkflowRegistry
from victor.workflows.new_feature_workflow import NewFeatureWorkflow
from victor.tools.base import ToolRegistry, ToolResult


class TestWorkflowBase(BaseWorkflow):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "test_workflow"

    @property
    def description(self) -> str:
        return "A test workflow"

    async def run(self, context: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return {"success": True, "message": "Test completed"}


class TestWorkflowRegistry:
    """Tests for WorkflowRegistry class."""

    def test_register_workflow(self):
        """Test registering a workflow."""
        registry = WorkflowRegistry()
        workflow = TestWorkflowBase()
        registry.register(workflow)
        assert registry.get("test_workflow") is workflow

    def test_register_duplicate_raises(self):
        """Test registering duplicate workflow raises ValueError."""
        registry = WorkflowRegistry()
        workflow1 = TestWorkflowBase()
        workflow2 = TestWorkflowBase()
        registry.register(workflow1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(workflow2)

    def test_get_unknown_returns_none(self):
        """Test getting unknown workflow returns None."""
        registry = WorkflowRegistry()
        assert registry.get("nonexistent") is None

    def test_list_workflows_empty(self):
        """Test listing workflows when empty."""
        registry = WorkflowRegistry()
        assert registry.list_workflows() == []

    def test_list_workflows_multiple(self):
        """Test listing multiple workflows."""
        registry = WorkflowRegistry()

        class AnotherWorkflow(BaseWorkflow):
            @property
            def name(self) -> str:
                return "another"

            @property
            def description(self) -> str:
                return "Another workflow"

            async def run(self, context: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
                return {}

        workflow1 = TestWorkflowBase()
        workflow2 = AnotherWorkflow()
        registry.register(workflow1)
        registry.register(workflow2)

        workflows = registry.list_workflows()
        assert len(workflows) == 2
        assert workflow1 in workflows
        assert workflow2 in workflows


class TestNewFeatureWorkflow:
    """Tests for NewFeatureWorkflow class."""

    def test_name_property(self):
        """Test workflow name."""
        workflow = NewFeatureWorkflow()
        assert workflow.name == "new_feature"

    def test_description_property(self):
        """Test workflow description."""
        workflow = NewFeatureWorkflow()
        assert "new feature" in workflow.description.lower()

    def test_sanitize_branch_name_basic(self):
        """Test basic branch name sanitization."""
        workflow = NewFeatureWorkflow()
        result = workflow._sanitize_branch_name("My Feature")
        assert result == "feature/my-feature"

    def test_sanitize_branch_name_special_chars(self):
        """Test sanitization removes special characters."""
        workflow = NewFeatureWorkflow()
        result = workflow._sanitize_branch_name("Feature@#$%Test")
        assert result == "feature/featuretest"

    def test_sanitize_branch_name_multiple_spaces(self):
        """Test sanitization handles multiple spaces."""
        workflow = NewFeatureWorkflow()
        result = workflow._sanitize_branch_name("Feature   With   Spaces")
        assert result == "feature/feature-with-spaces"

    @pytest.mark.asyncio
    async def test_run_missing_feature_name(self):
        """Test run without feature_name."""
        workflow = NewFeatureWorkflow()
        result = await workflow.run({})
        assert "error" in result
        assert "feature_name" in result["error"]

    @pytest.mark.asyncio
    async def test_run_missing_tool_registry(self):
        """Test run without ToolRegistry in context."""
        workflow = NewFeatureWorkflow()
        result = await workflow.run({}, feature_name="test")
        assert "error" in result
        assert "ToolRegistry" in result["error"]

    @pytest.mark.asyncio
    async def test_run_invalid_tool_registry(self):
        """Test run with invalid ToolRegistry type."""
        workflow = NewFeatureWorkflow()
        result = await workflow.run({"tool_registry": "not a registry"}, feature_name="test")
        assert "error" in result
        assert "ToolRegistry" in result["error"]

    @pytest.mark.asyncio
    async def test_run_branch_creation_fails(self):
        """Test run when branch creation fails."""
        workflow = NewFeatureWorkflow()
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.execute = AsyncMock(
            return_value=ToolResult(success=False, output="", error="Branch exists")
        )

        result = await workflow.run({"tool_registry": mock_registry}, feature_name="test feature")
        assert "error" in result
        assert "branch" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_source_file_creation_fails(self):
        """Test run when source file creation fails."""
        workflow = NewFeatureWorkflow()
        mock_registry = MagicMock(spec=ToolRegistry)

        # First call (branch) succeeds, second call (write_file) fails
        mock_registry.execute = AsyncMock(
            side_effect=[
                ToolResult(success=True, output="Branch created"),
                ToolResult(success=False, output="", error="Write failed"),
            ]
        )

        result = await workflow.run({"tool_registry": mock_registry}, feature_name="test feature")
        assert "error" in result
        assert "source file" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_test_file_creation_fails(self):
        """Test run when test file creation fails."""
        workflow = NewFeatureWorkflow()
        mock_registry = MagicMock(spec=ToolRegistry)

        # First two calls succeed, third call (test file) fails
        mock_registry.execute = AsyncMock(
            side_effect=[
                ToolResult(success=True, output="Branch created"),
                ToolResult(success=True, output="Source file created"),
                ToolResult(success=False, output="", error="Write failed"),
            ]
        )

        result = await workflow.run({"tool_registry": mock_registry}, feature_name="test feature")
        assert "error" in result
        assert "test file" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful workflow run."""
        workflow = NewFeatureWorkflow()
        mock_registry = MagicMock(spec=ToolRegistry)

        # All calls succeed
        mock_registry.execute = AsyncMock(return_value=ToolResult(success=True, output="Success"))

        result = await workflow.run({"tool_registry": mock_registry}, feature_name="my new feature")
        assert result["success"] is True
        assert result["branch_name"] == "feature/my-new-feature"
        assert result["source_file"] == "feature/my_new_feature.py"
        assert result["test_file"] == "tests/test_feature/my_new_feature.py"
