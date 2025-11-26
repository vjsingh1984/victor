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

"""Tests for workflows/new_feature_workflow.py module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.workflows.new_feature_workflow import NewFeatureWorkflow
from victor.tools.base import ToolRegistry, ToolResult


class TestNewFeatureWorkflow:
    """Tests for NewFeatureWorkflow class."""

    @pytest.fixture
    def workflow(self):
        """Create a NewFeatureWorkflow instance."""
        return NewFeatureWorkflow()

    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock ToolRegistry."""
        return MagicMock(spec=ToolRegistry)

    def test_workflow_name(self, workflow):
        """Test workflow name property."""
        assert workflow.name == "new_feature"

    def test_workflow_description(self, workflow):
        """Test workflow description property."""
        assert "git branch" in workflow.description
        assert "source" in workflow.description.lower()
        assert "test" in workflow.description.lower()

    def test_sanitize_branch_name(self, workflow):
        """Test branch name sanitization."""
        # Test with spaces
        assert workflow._sanitize_branch_name("My Feature") == "feature/my-feature"

        # Test with multiple spaces
        assert workflow._sanitize_branch_name("New   Feature   Name") == "feature/new-feature-name"

        # Test with special characters
        assert workflow._sanitize_branch_name("Feature@123!") == "feature/feature123"

        # Test with mixed case
        assert workflow._sanitize_branch_name("CamelCaseFeature") == "feature/camelcasefeature"

        # Test with numbers
        assert workflow._sanitize_branch_name("Feature 123") == "feature/feature-123"

    @pytest.mark.asyncio
    async def test_run_missing_feature_name(self, workflow, mock_tool_registry):
        """Test run with missing feature_name."""
        context = {"tool_registry": mock_tool_registry}

        result = await workflow.run(context)

        assert "error" in result
        assert "feature_name" in result["error"]

    @pytest.mark.asyncio
    async def test_run_missing_tool_registry(self, workflow):
        """Test run with missing tool_registry in context."""
        context = {}

        result = await workflow.run(context, feature_name="test")

        assert "error" in result
        assert "ToolRegistry" in result["error"]

    @pytest.mark.asyncio
    async def test_run_git_branch_failure(self, workflow, mock_tool_registry):
        """Test run when git branch creation fails."""
        context = {"tool_registry": mock_tool_registry}

        # Mock git branch creation to fail
        mock_tool_registry.execute = AsyncMock(
            return_value=ToolResult(
                success=False,
                output=None,
                error="Git error"
            )
        )

        result = await workflow.run(context, feature_name="test feature")

        assert "error" in result
        assert "git branch" in result["error"].lower()
        assert "details" in result

    @pytest.mark.asyncio
    async def test_run_source_file_failure(self, workflow, mock_tool_registry):
        """Test run when source file creation fails."""
        context = {"tool_registry": mock_tool_registry}

        call_count = [0]

        async def mock_execute(tool_name, ctx, **kwargs):
            call_count[0] += 1
            if tool_name == "git":
                return ToolResult(success=True, output="Branch created")
            elif tool_name == "write_file":
                # First write_file call (source) fails, second (test) succeeds
                if call_count[0] == 2:
                    return ToolResult(success=False, output=None, error="Write error")
                return ToolResult(success=True, output="Done")
            return ToolResult(success=True, output="Done")

        mock_tool_registry.execute = AsyncMock(side_effect=mock_execute)

        result = await workflow.run(context, feature_name="my feature")

        assert "error" in result
        assert "source file" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_test_file_failure(self, workflow, mock_tool_registry):
        """Test run when test file creation fails."""
        context = {"tool_registry": mock_tool_registry}

        async def mock_execute(tool, ctx, **kwargs):
            if tool == "git":
                return ToolResult(success=True, output="Branch created")
            elif tool == "write_file":
                path = kwargs.get("path", "")
                # Check for "tests/" prefix to identify test file
                if "tests/" in path:
                    return ToolResult(success=False, output=None, error="Write error")
                return ToolResult(success=True, output="File created")
            return ToolResult(success=True, output="Done")

        mock_tool_registry.execute = AsyncMock(side_effect=mock_execute)

        result = await workflow.run(context, feature_name="my feature")

        assert "error" in result
        assert "test file" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_run_success(self, workflow, mock_tool_registry):
        """Test successful workflow execution."""
        context = {"tool_registry": mock_tool_registry}

        # Mock all operations to succeed
        mock_tool_registry.execute = AsyncMock(
            return_value=ToolResult(success=True, output="Success")
        )

        result = await workflow.run(context, feature_name="My New Feature")

        assert result["success"] is True
        assert "message" in result
        assert result["branch_name"] == "feature/my-new-feature"
        assert result["source_file"] == "feature/my_new_feature.py"
        assert result["test_file"] == "tests/test_feature/my_new_feature.py"

    @pytest.mark.asyncio
    async def test_run_creates_correct_file_content(self, workflow, mock_tool_registry):
        """Test that correct content is written to files."""
        context = {"tool_registry": mock_tool_registry}

        captured_calls = []

        async def capture_execute(tool, ctx, **kwargs):
            captured_calls.append({"tool": tool, "kwargs": kwargs})
            return ToolResult(success=True, output="Success")

        mock_tool_registry.execute = AsyncMock(side_effect=capture_execute)

        result = await workflow.run(context, feature_name="Auth Feature")

        assert result["success"] is True
        assert len(captured_calls) == 3

        # Check git call
        assert captured_calls[0]["tool"] == "git"
        assert captured_calls[0]["kwargs"]["operation"] == "branch"
        assert captured_calls[0]["kwargs"]["branch"] == "feature/auth-feature"

        # Check source file call
        assert captured_calls[1]["tool"] == "write_file"
        assert captured_calls[1]["kwargs"]["path"] == "feature/auth_feature.py"
        assert "Auth Feature" in captured_calls[1]["kwargs"]["content"]

        # Check test file call
        assert captured_calls[2]["tool"] == "write_file"
        assert "tests/" in captured_calls[2]["kwargs"]["path"]
        assert "pytest" in captured_calls[2]["kwargs"]["content"]

    @pytest.mark.asyncio
    async def test_run_with_complex_feature_name(self, workflow, mock_tool_registry):
        """Test workflow with complex feature name."""
        context = {"tool_registry": mock_tool_registry}

        mock_tool_registry.execute = AsyncMock(
            return_value=ToolResult(success=True, output="Success")
        )

        result = await workflow.run(context, feature_name="User Authentication & OAuth2.0!")

        assert result["success"] is True
        assert result["branch_name"] == "feature/user-authentication--oauth20"
        assert result["source_file"] == "feature/user_authentication__oauth20.py"
        assert result["test_file"] == "tests/test_feature/user_authentication__oauth20.py"
