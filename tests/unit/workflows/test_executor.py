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
        assert ExecutorNodeStatus.CANCELLED.value == "cancelled"


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
        assert result.metadata == {}

    def test_initialization_with_metadata(self):
        """Test NodeResult with metadata."""
        metadata = {
            "execution_time": 1.5,
            "tool_calls": 5,
        }
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.COMPLETED,
            metadata=metadata,
        )
        assert result.metadata == metadata

    def test_initialization_with_error(self):
        """Test NodeResult with error."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.FAILED,
            error="Execution failed",
        )
        assert result.status == ExecutorNodeStatus.FAILED
        assert result.error == "Execution failed"

    def test_is_complete_true(self):
        """Test is_complete property when True."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.COMPLETED,
        )
        assert result.is_complete is True

    def test_is_complete_false(self):
        """Test is_complete property when False."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.RUNNING,
        )
        assert result.is_complete is False

    def test_is_failed_true(self):
        """Test is_failed property when True."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.FAILED,
        )
        assert result.is_failed is True

    def test_is_failed_false(self):
        """Test is_failed property when False."""
        result = NodeResult(
            node_id="test_node",
            status=ExecutorNodeStatus.COMPLETED,
        )
        assert result.is_failed is False


# =============================================================================
# TemporalContext Tests
# =============================================================================


class TestTemporalContext:
    """Test TemporalContext dataclass."""

    def test_initialization(self):
        """Test TemporalContext initialization."""
        now = datetime.now(timezone.utc)
        temporal = TemporalContext(
            workflow_id="workflow_1",
            started_at=now,
            node_timeouts={"node1": 60},
        )
        assert temporal.workflow_id == "workflow_1"
        assert temporal.started_at == now
        assert temporal.node_timeouts == {"node1": 60}

    def test_default_timeouts(self):
        """Test default node timeouts."""
        temporal = TemporalContext(workflow_id="workflow_1")
        assert temporal.node_timeouts == {}

    def test_get_timeout_for_node(self):
        """Test getting timeout for a specific node."""
        temporal = TemporalContext(
            workflow_id="workflow_1",
            node_timeouts={"node1": 120, "node2": 60},
        )
        timeout = temporal.get_node_timeout("node1")
        assert timeout == 120

    def test_get_timeout_default(self):
        """Test getting timeout for node without specific timeout."""
        temporal = TemporalContext(
            workflow_id="workflow_1",
            node_timeouts={"node1": 120},
        )
        timeout = temporal.get_node_timeout("node2", default=30)
        assert timeout == 30


# =============================================================================
# WorkflowContext Tests
# =============================================================================


class TestWorkflowContext:
    """Test WorkflowContext dataclass."""

    def test_initialization(self):
        """Test WorkflowContext initialization."""
        context = WorkflowContext(
            workflow_id="workflow_1",
            execution_id="exec_1",
            state={"key": "value"},
        )
        assert context.workflow_id == "workflow_1"
        assert context.execution_id == "exec_1"
        assert context.state == {"key": "value"}

    def test_get_state_value(self):
        """Test getting state value."""
        context = WorkflowContext(
            workflow_id="workflow_1",
            execution_id="exec_1",
            state={"user": "alice", "task": "test"},
        )
        assert context.get("user") == "alice"
        assert context.get("task") == "test"

    def test_get_state_default(self):
        """Test getting state value with default."""
        context = WorkflowContext(
            workflow_id="workflow_1",
            execution_id="exec_1",
            state={"user": "alice"},
        )
        assert context.get("nonexistent", default="default") == "default"

    def test_set_state_value(self):
        """Test setting state value."""
        context = WorkflowContext(
            workflow_id="workflow_1",
            execution_id="exec_1",
            state={},
        )
        context.set("user", "bob")
        assert context.get("user") == "bob"

    def test_update_state(self):
        """Test updating state with dict."""
        context = WorkflowContext(
            workflow_id="workflow_1",
            execution_id="exec_1",
            state={"user": "alice"},
        )
        context.update({"task": "test", "priority": "high"})
        assert context.get("user") == "alice"
        assert context.get("task") == "test"
        assert context.get("priority") == "high"

    def test_delete_state_value(self):
        """Test deleting state value."""
        context = WorkflowContext(
            workflow_id="workflow_1",
            execution_id="exec_1",
            state={"user": "alice", "task": "test"},
        )
        context.delete("task")
        assert context.get("task") is None
        assert context.get("user") == "alice"

    def test_has_state_value(self):
        """Test checking if state has value."""
        context = WorkflowContext(
            workflow_id="workflow_1",
            execution_id="exec_1",
            state={"user": "alice"},
        )
        assert context.has("user") is True
        assert context.has("nonexistent") is False

    def test_keys(self):
        """Test getting state keys."""
        context = WorkflowContext(
            workflow_id="workflow_1",
            execution_id="exec_1",
            state={"user": "alice", "task": "test"},
        )
        keys = context.keys()
        assert "user" in keys
        assert "task" in keys

    def test_to_dict(self):
        """Test converting context to dict."""
        context = WorkflowContext(
            workflow_id="workflow_1",
            execution_id="exec_1",
            state={"user": "alice"},
        )
        data = context.to_dict()
        assert data["workflow_id"] == "workflow_1"
        assert data["execution_id"] == "exec_1"
        assert data["state"]["user"] == "alice"

    def test_from_dict(self):
        """Test creating context from dict."""
        data = {
            "workflow_id": "workflow_1",
            "execution_id": "exec_1",
            "state": {"user": "alice"},
        }
        context = WorkflowContext.from_dict(data)
        assert context.workflow_id == "workflow_1"
        assert context.state["user"] == "alice"


# =============================================================================
# WorkflowResult Tests
# =============================================================================


class TestWorkflowResult:
    """Test WorkflowResult dataclass."""

    def test_initialization_success(self):
        """Test WorkflowResult initialization for success."""
        now = datetime.now(timezone.utc)
        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            final_state={"result": "success"},
            node_results={},
        )
        assert result.workflow_id == "workflow_1"
        assert result.execution_id == "exec_1"
        assert result.status == ExecutorNodeStatus.COMPLETED
        assert result.final_state == {"result": "success"}
        assert result.error is None

    def test_initialization_failure(self):
        """Test WorkflowResult initialization for failure."""
        now = datetime.now(timezone.utc)
        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.FAILED,
            started_at=now,
            completed_at=now,
            error="Workflow execution failed",
            node_results={},
        )
        assert result.status == ExecutorNodeStatus.FAILED
        assert result.error == "Workflow failed"

    def test_is_successful_true(self):
        """Test is_successful property when True."""
        now = datetime.now(timezone.utc)
        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            node_results={},
        )
        assert result.is_successful is True

    def test_is_successful_false(self):
        """Test is_successful property when False."""
        now = datetime.now(timezone.utc)
        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.FAILED,
            started_at=now,
            completed_at=now,
            error="Failed",
            node_results={},
        )
        assert result.is_successful is False

    def test_execution_duration(self):
        """Test execution duration calculation."""
        started = datetime.now(timezone.utc)
        completed = datetime.fromtimestamp(
            started.timestamp() + 10, tz=timezone.utc
        )

        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.COMPLETED,
            started_at=started,
            completed_at=completed,
            node_results={},
        )
        duration = result.execution_duration
        assert duration >= 9.9 and duration <= 10.1  # Account for timing

    def test_execution_duration_none(self):
        """Test execution duration when completed_at is None."""
        now = datetime.now(timezone.utc)
        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.RUNNING,
            started_at=now,
            completed_at=None,
            node_results={},
        )
        assert result.execution_duration is None

    def test_get_node_result(self):
        """Test getting result for specific node."""
        now = datetime.now(timezone.utc)
        node_result = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.COMPLETED,
            output={"data": "test"},
        )
        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            node_results={"node1": node_result},
        )
        retrieved = result.get_node_result("node1")
        assert retrieved is not None
        assert retrieved.output == {"data": "test"}

    def test_get_node_result_not_found(self):
        """Test getting result for non-existent node."""
        now = datetime.now(timezone.utc)
        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            node_results={},
        )
        retrieved = result.get_node_result("nonexistent")
        assert retrieved is None

    def test_get_failed_nodes(self):
        """Test getting list of failed nodes."""
        now = datetime.now(timezone.utc)
        node1 = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.COMPLETED,
        )
        node2 = NodeResult(
            node_id="node2",
            status=ExecutorNodeStatus.FAILED,
        )
        node3 = NodeResult(
            node_id="node3",
            status=ExecutorNodeStatus.FAILED,
        )

        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.FAILED,
            started_at=now,
            completed_at=now,
            node_results={"node1": node1, "node2": node2, "node3": node3},
        )

        failed = result.get_failed_nodes()
        assert "node2" in failed
        assert "node3" in failed
        assert "node1" not in failed
        assert len(failed) == 2

    def test_to_dict(self):
        """Test converting result to dict."""
        now = datetime.now(timezone.utc)
        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.COMPLETED,
            started_at=now,
            completed_at=now,
            final_state={"result": "success"},
            node_results={},
        )
        data = result.to_dict()
        assert data["workflow_id"] == "workflow_1"
        assert data["status"] == "completed"
        assert data["final_state"]["result"] == "success"

    def test_from_dict(self):
        """Test creating result from dict."""
        now = datetime.now(timezone.utc)
        data = {
            "workflow_id": "workflow_1",
            "execution_id": "exec_1",
            "status": "completed",
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "final_state": {"result": "success"},
            "node_results": {},
        }
        result = WorkflowResult.from_dict(data)
        assert result.workflow_id == "workflow_1"
        assert result.status == ExecutorNodeStatus.COMPLETED

    def test_summary(self):
        """Test getting result summary."""
        now = datetime.now(timezone.utc)
        node1 = NodeResult(node_id="node1", status=ExecutorNodeStatus.COMPLETED)
        node2 = NodeResult(node_id="node2", status=ExecutorNodeStatus.FAILED)

        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.FAILED,
            started_at=now,
            completed_at=now,
            node_results={"node1": node1, "node2": node2},
        )

        summary = result.summary()
        assert "workflow_id" in summary
        assert "status" in summary
        assert "total_nodes" in summary
        assert summary["total_nodes"] == 2

    def test_merge_node_results(self):
        """Test merging node results."""
        now = datetime.now(timezone.utc)
        result = WorkflowResult(
            workflow_id="workflow_1",
            execution_id="exec_1",
            status=ExecutorNodeStatus.RUNNING,
            started_at=now,
            completed_at=None,
            node_results={},
        )

        node_result = NodeResult(
            node_id="node1",
            status=ExecutorNodeStatus.COMPLETED,
            output={"data": "test"},
        )

        result.merge_node_result(node_result)
        assert "node1" in result.node_results
        assert result.node_results["node1"].output == {"data": "test"}


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
        from victor.workflows.definition import WorkflowDefinition, WorkflowNode, WorkflowNodeType

        mock_orchestrator = MagicMock()
        mock_tool_registry = MagicMock()

        executor = WorkflowExecutor(
            orchestrator=mock_orchestrator,
            tool_registry=mock_tool_registry,
        )

        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            nodes=[
                WorkflowNode(
                    name="node1",
                    node_type=WorkflowNodeType.AGENT,
                    role="agent",
                    goal="Test goal",
                    next_nodes=[],
                ),
            ],
        )

        # Mock the sub-agent execution
        with patch.object(executor, "_execute_agent_node") as mock_execute:
            mock_execute.return_value = NodeResult(
                node_id="node1",
                status=ExecutorNodeStatus.COMPLETED,
                output={"result": "success"},
            )

            context = WorkflowContext(
                workflow_id="test_workflow",
                execution_id="exec_1",
                state={},
            )

            result = await executor.execute(workflow, context)
            assert result.status == ExecutorNodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_workflow_with_failure(self):
        """Test executing a workflow that fails."""
        from victor.workflows.executor import WorkflowExecutor
        from victor.workflows.definition import WorkflowDefinition, WorkflowNode, WorkflowNodeType

        mock_orchestrator = MagicMock()
        mock_tool_registry = MagicMock()

        executor = WorkflowExecutor(
            orchestrator=mock_orchestrator,
            tool_registry=mock_tool_registry,
        )

        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            nodes=[
                WorkflowNode(
                    name="node1",
                    node_type=WorkflowNodeType.AGENT,
                    role="agent",
                    goal="Test goal",
                    next_nodes=[],
                ),
            ],
        )

        # Mock the sub-agent execution to fail
        with patch.object(executor, "_execute_agent_node") as mock_execute:
            mock_execute.return_value = NodeResult(
                node_id="node1",
                status=ExecutorNodeStatus.FAILED,
                error="Execution failed",
            )

            context = WorkflowContext(
                workflow_id="test_workflow",
                execution_id="exec_1",
                state={},
            )

            result = await executor.execute(workflow, context)
            assert result.status == ExecutorNodeStatus.FAILED

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
