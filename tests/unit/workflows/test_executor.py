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

"""Tests for victor.workflows.executor module."""

from datetime import datetime, timezone
from enum import Enum
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass, field

import pytest

from victor.workflows.executor import (
    ExecutorNodeStatus,
    NodeResult,
    TemporalContext,
    WorkflowContext,
    WorkflowResult,
)


# =============================================================================
# ExecutorNodeStatus Tests
# =============================================================================


class TestExecutorNodeStatus:
    """Test ExecutorNodeStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert ExecutorNodeStatus.PENDING.value == "pending"
        assert ExecutorNodeStatus.RUNNING.value == "running"
        assert ExecutorNodeStatus.COMPLETED.value == "completed"
        assert ExecutorNodeStatus.FAILED.value == "failed"
        assert ExecutorNodeStatus.SKIPPED.value == "skipped"


# =============================================================================
# NodeResult Tests
# =============================================================================


class TestNodeResult:
    """Test NodeResult dataclass."""

    def test_initialization(self):
        """Test NodeResult initialization."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.COMPLETED,
            output={"result": "success"},
            error=None,
        )
        assert result.node_id == "test_node"
        assert result.status == ExecutorNodeStatus.COMPLETED
        assert result.output == {"result": "success"}
        assert result.error is None

    def test_initialization_with_duration(self):
        """Test NodeResult with duration and tool calls."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.COMPLETED,
            output="done",
            duration_seconds=1.5,
            tool_calls_used=5,
        )
        assert result.duration_seconds == 1.5
        assert result.tool_calls_used == 5

    def test_initialization_with_error(self):
        """Test NodeResult with error."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.FAILED,
            error="Execution failed",
        )
        assert result.status == ExecutorNodeStatus.FAILED
        assert result.error == "Execution failed"

    def test_success_true(self):
        """Test success property when True."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.COMPLETED,
        )
        assert result.success is True

    def test_success_false(self):
        """Test success property when False."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.RUNNING,
        )
        assert result.success is False

    def test_success_false_for_failed(self):
        """Test success property when failed."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.FAILED,
        )
        assert result.success is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.COMPLETED,
            output={"result": "success"},
            duration_seconds=2.5,
            tool_calls_used=3,
        )
        data = result.to_dict()
        assert data["node_id"] == "test_node"
        assert data["status"] == "completed"
        assert data["output"] == {"result": "success"}
        assert data["duration_seconds"] == 2.5
        assert data["tool_calls_used"] == 3


# =============================================================================
# TemporalContext Tests
# =============================================================================


class TestTemporalContext:
    """Test TemporalContext dataclass."""

    def test_initialization(self):
        """Test TemporalContext initialization."""
        temporal = TemporalContext(
            as_of_date="2025-01-15",
            lookback_periods=4,
        )
        assert temporal.as_of_date == "2025-01-15"
        assert temporal.lookback_periods == 4
        assert temporal.period_type == "quarters"
        assert temporal.include_end_date is True

    def test_default_values(self):
        """Test default values."""
        temporal = TemporalContext()
        assert temporal.as_of_date is None
        assert temporal.lookback_periods == 0
        assert temporal.period_type == "quarters"
        assert temporal.include_end_date is True

    def test_custom_period_type(self):
        """Test custom period type."""
        temporal = TemporalContext(
            as_of_date="2025-01-15",
            lookback_periods=30,
            period_type="days",
        )
        assert temporal.period_type == "days"

    def test_get_date_range_quarters(self):
        """Test date range calculation for quarters."""
        temporal = TemporalContext(
            as_of_date="2025-01-15",
            lookback_periods=2,
            period_type="quarters",
        )
        start, end = temporal.get_date_range()
        assert end == "2025-01-15"
        # Approximately 6 months back (2 quarters)
        assert start < "2025-01-15"

    def test_get_date_range_days(self):
        """Test date range calculation for days."""
        temporal = TemporalContext(
            as_of_date="2025-01-15",
            lookback_periods=7,
            period_type="days",
        )
        start, end = temporal.get_date_range()
        assert end == "2025-01-15"
        assert start < "2025-01-15"

    def test_is_valid_for_date_with_include(self):
        """Test date validation with include_end_date=True."""
        temporal = TemporalContext(
            as_of_date="2025-01-15",
            include_end_date=True,
        )
        assert temporal.is_valid_for_date("2025-01-15") is True
        assert temporal.is_valid_for_date("2025-01-10") is True
        assert temporal.is_valid_for_date("2025-01-20") is False

    def test_is_valid_for_date_without_include(self):
        """Test date validation with include_end_date=False."""
        temporal = TemporalContext(
            as_of_date="2025-01-15",
            include_end_date=False,
        )
        assert temporal.is_valid_for_date("2025-01-15") is False
        assert temporal.is_valid_for_date("2025-01-10") is True
        assert temporal.is_valid_for_date("2025-01-20") is False

    def test_is_valid_for_date_no_constraint(self):
        """Test date validation when no as_of_date is set."""
        temporal = TemporalContext()
        assert temporal.is_valid_for_date("2025-01-15") is True
        assert temporal.is_valid_for_date("2020-01-01") is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        temporal = TemporalContext(
            as_of_date="2025-01-15",
            lookback_periods=2,
            period_type="months",
            include_end_date=False,
        )
        data = temporal.to_dict()
        assert data["as_of_date"] == "2025-01-15"
        assert data["lookback_periods"] == 2
        assert data["period_type"] == "months"
        assert data["include_end_date"] is False

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "as_of_date": "2025-01-15",
            "lookback_periods": 3,
            "period_type": "weeks",
            "include_end_date": True,
        }
        temporal = TemporalContext.from_dict(data)
        assert temporal.as_of_date == "2025-01-15"
        assert temporal.lookback_periods == 3
        assert temporal.period_type == "weeks"
        assert temporal.include_end_date is True

    def test_from_dict_defaults(self):
        """Test from_dict with missing fields."""
        data = {"as_of_date": "2025-01-15"}
        temporal = TemporalContext.from_dict(data)
        assert temporal.as_of_date == "2025-01-15"
        assert temporal.lookback_periods == 0
        assert temporal.period_type == "quarters"
        assert temporal.include_end_date is True


# =============================================================================
# WorkflowContext Tests
# =============================================================================


class TestWorkflowContext:
    """Test WorkflowContext dataclass."""

    def test_initialization(self):
        """Test WorkflowContext initialization."""
        context = WorkflowContext(
            data={"key": "value"},
            metadata={"workflow_id": "workflow_1"},
        )
        assert context.data == {"key": "value"}
        assert context.metadata == {"workflow_id": "workflow_1"}
        assert context.node_results == {}
        assert context.temporal is None

    def test_get_state_value(self):
        """Test getting state value."""
        context = WorkflowContext(
            data={"user": "alice", "task": "test"},
        )
        assert context.get("user") == "alice"
        assert context.get("task") == "test"

    def test_get_state_default(self):
        """Test getting state value with default."""
        context = WorkflowContext(
            data={"user": "alice"},
        )
        assert context.get("nonexistent", default="default") == "default"

    def test_set_state_value(self):
        """Test setting state value."""
        context = WorkflowContext(data={})
        context.set("user", "bob")
        assert context.get("user") == "bob"

    def test_update_state(self):
        """Test updating state with dict."""
        context = WorkflowContext(
            data={"user": "alice"},
        )
        context.update({"task": "test", "priority": "high"})
        assert context.get("user") == "alice"
        assert context.get("task") == "test"
        assert context.get("priority") == "high"

    def test_add_node_result(self):
        """Test adding node result."""
        context = WorkflowContext()
        result = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.COMPLETED,
            output="done",
        )
        context.add_result(result)
        assert context.get_result("node1") is not None
        assert context.get_result("node1").output == "done"

    def test_get_result(self):
        """Test getting result for specific node."""
        context = WorkflowContext()
        result = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.COMPLETED,
            output={"data": "test"},
        )
        context.add_result(result)
        retrieved = context.get_result("node1")
        assert retrieved is not None
        assert retrieved.output == {"data": "test"}

    def test_get_result_not_found(self):
        """Test getting result for non-existent node."""
        context = WorkflowContext()
        result = context.get_result("nonexistent")
        assert result is None

    def test_has_failures_true(self):
        """Test has_failures when there are failures."""
        context = WorkflowContext()
        failed_result = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.FAILED,
            error="Failed",
        )
        context.add_result(failed_result)
        assert context.has_failures() is True

    def test_has_failures_false(self):
        """Test has_failures when there are no failures."""
        context = WorkflowContext()
        success_result = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.COMPLETED,
            output="done",
        )
        context.add_result(success_result)
        assert context.has_failures() is False

    def test_get_outputs(self):
        """Test getting all successful node outputs."""
        context = WorkflowContext()
        result1 = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.COMPLETED,
            output={"result": "success"},
        )
        result2 = NodeResult(
            node_id="node2",
            status=ExecutorNodeStatus.COMPLETED,
            output="done",
        )
        result3 = NodeResult(
            node_id="node3",
            status=ExecutorNodeStatus.FAILED,
            error="failed",
        )
        context.add_result(result1)
        context.add_result(result2)
        context.add_result(result3)

        outputs = context.get_outputs()
        assert "node1" in outputs
        assert "node2" in outputs
        assert "node3" not in outputs
        assert outputs["node1"] == {"result": "success"}
        assert outputs["node2"] == "done"

    def test_with_temporal_context(self):
        """Test WorkflowContext with temporal context."""
        temporal = TemporalContext(
            as_of_date="2025-01-15",
            lookback_periods=2,
        )
        context = WorkflowContext(
            data={"analysis": "test"},
            temporal=temporal,
        )
        assert context.temporal is not None
        assert context.temporal.as_of_date == "2025-01-15"
        assert context.get("analysis") == "test"


# =============================================================================
# WorkflowResult Tests
# =============================================================================


class TestWorkflowResult:
    """Test WorkflowResult dataclass."""

    def test_initialization_success(self):
        """Test WorkflowResult initialization for success."""
        context = WorkflowContext(
            data={"result": "success"},
        )
        result = WorkflowResult(
            workflow_name="test_workflow",
            success=True,
            context=context,
            total_duration=10.5,
            total_tool_calls=5,
        )
        assert result.workflow_name == "test_workflow"
        assert result.success is True
        assert result.context == context
        assert result.total_duration == 10.5
        assert result.total_tool_calls == 5
        assert result.error is None

    def test_initialization_failure(self):
        """Test WorkflowResult initialization for failure."""
        context = WorkflowContext(data={})
        result = WorkflowResult(
            workflow_name="test_workflow",
            success=False,
            context=context,
            error="Workflow execution failed",
        )
        assert result.success is False
        assert result.error == "Workflow execution failed"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        context = WorkflowContext(
            data={"result": "success"},
        )
        node_result = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.COMPLETED,
            output="done",
        )
        context.add_result(node_result)

        result = WorkflowResult(
            workflow_name="test_workflow",
            success=True,
            context=context,
            total_duration=5.0,
            total_tool_calls=3,
        )
        data = result.to_dict()
        assert data["workflow_name"] == "test_workflow"
        assert data["success"] is True
        assert data["total_duration"] == 5.0
        assert data["total_tool_calls"] == 3
        assert data["error"] is None
        assert "outputs" in data
        assert "node_results" in data

    def test_get_output(self):
        """Test getting output from specific node."""
        context = WorkflowContext()
        node_result = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.COMPLETED,
            output={"data": "test"},
        )
        context.add_result(node_result)

        result = WorkflowResult(
            workflow_name="test_workflow",
            success=True,
            context=context,
        )
        output = result.get_output("node1")
        assert output == {"data": "test"}

    def test_get_output_not_found(self):
        """Test getting output from non-existent node."""
        context = WorkflowContext()
        result = WorkflowResult(
            workflow_name="test_workflow",
            success=True,
            context=context,
        )
        output = result.get_output("nonexistent")
        assert output is None


# =============================================================================
# WorkflowExecutor Tests (Mock-based)
# =============================================================================


class TestWorkflowExecutor:
    """Test WorkflowExecutor class."""

    def test_initialization(self):
        """Test WorkflowExecutor initialization."""
        from victor.workflows.executor import WorkflowExecutor

        mock_orchestrator = MagicMock()
        mock_tool_registry = MagicMock()

        executor = WorkflowExecutor(
            orchestrator=mock_orchestrator,
            tool_registry=mock_tool_registry,
        )
        assert executor.orchestrator == mock_orchestrator
        assert executor.tool_registry == mock_tool_registry

    def test_initialization_with_cache(self):
        """Test WorkflowExecutor initialization with cache."""
        from victor.workflows.executor import WorkflowExecutor

        mock_orchestrator = MagicMock()
        mock_tool_registry = MagicMock()
        mock_cache = MagicMock()

        executor = WorkflowExecutor(
            orchestrator=mock_orchestrator,
            tool_registry=mock_tool_registry,
            cache=mock_cache,
        )
        assert executor.cache == mock_cache

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self):
        """Test executing a simple workflow."""
        from victor.workflows.executor import WorkflowExecutor
        from victor.workflows.definition import WorkflowDefinition, AgentNode

        mock_orchestrator = MagicMock()
        mock_tool_registry = MagicMock()

        executor = WorkflowExecutor(
            orchestrator=mock_orchestrator,
            tool_registry=mock_tool_registry,
        )

        # Create workflow with proper structure
        node1 = AgentNode(
            id="node1",
            name="Node 1",
            role="agent",
            goal="Test goal",
        )

        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            nodes={"node1": node1},
            start_node="node1",
        )

        # Mock the sub-agent execution
        with patch.object(executor, "_execute_agent_node") as mock_execute:
            mock_execute.return_value = NodeResult(
                node_id="node1",
                status=ExecutorNodeStatus.COMPLETED,
                output={"result": "success"},
            )

            result = await executor.execute(workflow)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_workflow_with_failure(self):
        """Test executing a workflow that fails."""
        from victor.workflows.executor import WorkflowExecutor
        from victor.workflows.definition import WorkflowDefinition, AgentNode

        mock_orchestrator = MagicMock()
        mock_tool_registry = MagicMock()

        executor = WorkflowExecutor(
            orchestrator=mock_orchestrator,
            tool_registry=mock_tool_registry,
        )

        # Create workflow with proper structure
        node1 = AgentNode(
            id="node1",
            name="Node 1",
            role="agent",
            goal="Test goal",
        )

        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            nodes={"node1": node1},
            start_node="node1",
        )

        # Mock the sub-agent execution to fail
        with patch.object(executor, "_execute_agent_node") as mock_execute:
            mock_execute.return_value = NodeResult(
                node_id="node1",
                status=ExecutorNodeStatus.FAILED,
                error="Execution failed",
            )

            result = await executor.execute(workflow)
            assert result.success is False

    def test_get_compute_handler(self):
        """Test getting compute handler."""
        from victor.workflows.executor import WorkflowExecutor, get_compute_handler, register_compute_handler

        # Create a custom handler
        async def custom_handler(node, context, tool_registry):
            return NodeResult(
                node_id=node.name,
                status=ExecutorNodeStatus.COMPLETED,
            )

        # Register handler
        register_compute_handler("custom", custom_handler)

        # Get handler
        handler = get_compute_handler("custom")
        assert handler is not None
        assert handler == custom_handler

    def test_get_compute_handler_not_found(self):
        """Test getting non-existent compute handler."""
        from victor.workflows.executor import get_compute_handler

        handler = get_compute_handler("nonexistent")
        # Should return None or raise
        assert handler is None
