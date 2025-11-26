"""Tests for workflow_tool module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.tools.workflow_tool import run_workflow


class TestRunWorkflow:
    """Tests for run_workflow function."""

    @pytest.mark.asyncio
    async def test_run_workflow_success(self):
        """Test successful workflow execution."""
        # Mock workflow and registry
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value={"success": True, "result": "completed"})

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_workflow

        context = {"workflow_registry": mock_registry}
        workflow_args = {"arg1": "value1", "arg2": "value2"}

        result = await run_workflow(
            workflow_name="test_workflow",
            context=context,
            workflow_args=workflow_args
        )

        assert result == {"success": True, "result": "completed"}
        mock_registry.get.assert_called_once_with("test_workflow")
        mock_workflow.run.assert_called_once_with(context, arg1="value1", arg2="value2")

    @pytest.mark.asyncio
    async def test_run_workflow_no_registry(self):
        """Test handling of missing workflow registry."""
        context = {}
        result = await run_workflow(
            workflow_name="test_workflow",
            context=context,
            workflow_args={}
        )

        assert "error" in result
        assert "WorkflowRegistry not found" in result["error"]

    @pytest.mark.asyncio
    async def test_run_workflow_not_found(self):
        """Test handling of non-existent workflow."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        context = {"workflow_registry": mock_registry}
        result = await run_workflow(
            workflow_name="nonexistent_workflow",
            context=context,
            workflow_args={}
        )

        assert "error" in result
        assert "not found" in result["error"]
        assert "nonexistent_workflow" in result["error"]

    @pytest.mark.asyncio
    async def test_run_workflow_execution_error(self):
        """Test handling of workflow execution error."""
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(side_effect=Exception("Workflow failed"))

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_workflow

        context = {"workflow_registry": mock_registry}
        result = await run_workflow(
            workflow_name="failing_workflow",
            context=context,
            workflow_args={}
        )

        assert "error" in result
        assert "unexpected error" in result["error"]
        assert "Workflow failed" in result["error"]

    @pytest.mark.asyncio
    async def test_run_workflow_with_complex_args(self):
        """Test workflow with complex arguments."""
        mock_workflow = MagicMock()
        mock_workflow.run = AsyncMock(return_value={
            "success": True,
            "files_created": 5,
            "tests_passed": 10
        })

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_workflow

        context = {"workflow_registry": mock_registry}
        workflow_args = {
            "feature_name": "user_authentication",
            "include_tests": True,
            "database_migration": True,
            "options": {"strict_mode": True}
        }

        result = await run_workflow(
            workflow_name="new_feature",
            context=context,
            workflow_args=workflow_args
        )

        assert result["success"] is True
        assert result["files_created"] == 5
        assert result["tests_passed"] == 10
        mock_workflow.run.assert_called_once()
        # Verify all workflow_args were passed
        call_kwargs = mock_workflow.run.call_args[1]
        assert call_kwargs["feature_name"] == "user_authentication"
        assert call_kwargs["include_tests"] is True
        assert call_kwargs["database_migration"] is True
        assert call_kwargs["options"] == {"strict_mode": True}
