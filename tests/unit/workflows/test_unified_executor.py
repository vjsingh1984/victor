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

"""Tests for StateGraphExecutor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.workflows.definition import (
    AgentNode,
    ComputeNode,
    ConditionNode,
    ParallelNode,
    TaskConstraints,
    TransformNode,
    WorkflowBuilder,
    WorkflowDefinition,
)
from victor.workflows.unified_executor import (
    ExecutorConfig,
    ExecutorResult,
    StateGraphExecutor,
    execute_workflow,
    get_executor,
)


class TestExecutorConfig:
    """Tests for ExecutorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExecutorConfig()

        assert config.enable_checkpointing is True
        assert config.max_iterations == 25
        assert config.timeout is None
        assert config.interrupt_nodes == []

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutorConfig(
            enable_checkpointing=False,
            max_iterations=100,
            timeout=300.0,
            interrupt_nodes=["approval"],
        )

        assert config.enable_checkpointing is False
        assert config.max_iterations == 100
        assert config.timeout == 300.0
        assert config.interrupt_nodes == ["approval"]


class TestExecutorResult:
    """Tests for ExecutorResult."""

    def test_successful_result(self):
        """Test successful result creation."""
        result = ExecutorResult(
            success=True,
            state={"output": "data", "computed": 42},
            duration_seconds=1.5,
            nodes_executed=["node1", "node2"],
            iterations=3,
        )

        assert result.success is True
        assert result.get("output") == "data"
        assert result.get("computed") == 42
        assert result.get("missing", "default") == "default"

    def test_failed_result(self):
        """Test failed result creation."""
        result = ExecutorResult(
            success=False,
            state={},
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_interrupted_result(self):
        """Test interrupted result for HITL."""
        result = ExecutorResult(
            success=True,
            state={"pending": "approval"},
            interrupted=True,
            interrupt_node="approval_node",
        )

        assert result.success is True
        assert result.interrupted is True
        assert result.interrupt_node == "approval_node"

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ExecutorResult(
            success=True,
            state={"key": "value"},
            duration_seconds=2.5,
            nodes_executed=["a", "b"],
            iterations=5,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["duration_seconds"] == 2.5
        assert d["nodes_executed"] == ["a", "b"]
        assert d["iterations"] == 5
        assert d["state_keys"] == ["key"]


class TestStateGraphExecutor:
    """Tests for StateGraphExecutor."""

    def test_initialization(self):
        """Test executor initialization."""
        executor = StateGraphExecutor()

        assert executor.orchestrator is None
        assert executor.config.enable_checkpointing is True
        assert executor.config.max_iterations == 25

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        config = ExecutorConfig(
            max_iterations=50,
            timeout=120.0,
        )
        executor = StateGraphExecutor(config=config)

        assert executor.config.max_iterations == 50
        assert executor.config.timeout == 120.0

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self):
        """Test execution of simple workflow."""
        executor = StateGraphExecutor()

        workflow = (
            WorkflowBuilder("test")
            .add_transform("init", lambda ctx: {**ctx, "initialized": True})
            .add_transform("process", lambda ctx: {**ctx, "processed": True})
            .build()
        )

        result = await executor.execute(
            workflow,
            {"input": "data"},
        )

        assert result.success is True
        assert result.get("initialized") is True
        assert result.get("processed") is True
        assert "init" in result.nodes_executed
        assert "process" in result.nodes_executed

    @pytest.mark.asyncio
    async def test_execute_with_checkpointing(self):
        """Test execution with checkpointing enabled."""
        executor = StateGraphExecutor(
            config=ExecutorConfig(enable_checkpointing=True)
        )

        workflow = (
            WorkflowBuilder("test")
            .add_transform("step", lambda ctx: {**ctx, "done": True})
            .build()
        )

        result = await executor.execute(
            workflow,
            {"input": "data"},
            thread_id="session-123",
        )

        assert result.success is True
        assert result.get("done") is True

    @pytest.mark.asyncio
    async def test_execute_handles_errors(self):
        """Test that execution handles errors gracefully.

        Transform nodes catch exceptions internally and record them
        in node_results, but the overall execution may still succeed
        (the node ran, it just had an error).
        """
        executor = StateGraphExecutor()

        def failing_transform(ctx):
            raise ValueError("Intentional failure")

        workflow = (
            WorkflowBuilder("failing")
            .add_transform("fail", failing_transform)
            .build()
        )

        result = await executor.execute(workflow, {})

        # Execution completes (doesn't crash), node was executed
        assert "fail" in result.nodes_executed

    @pytest.mark.asyncio
    async def test_execute_with_condition(self):
        """Test execution with condition node."""
        executor = StateGraphExecutor()

        workflow = WorkflowDefinition(
            name="conditional",
            description="Conditional workflow",
            nodes={
                "start": TransformNode(
                    id="start",
                    name="Start",
                    next_nodes=["check"],
                    transform=lambda ctx: {**ctx, "started": True},
                ),
                "check": ConditionNode(
                    id="check",
                    name="Check",
                    condition=lambda ctx: "yes" if ctx.get("flag") else "no",
                    branches={"yes": "good_path", "no": "bad_path"},
                ),
                "good_path": TransformNode(
                    id="good_path",
                    name="Good Path",
                    transform=lambda ctx: {**ctx, "path": "good"},
                ),
                "bad_path": TransformNode(
                    id="bad_path",
                    name="Bad Path",
                    transform=lambda ctx: {**ctx, "path": "bad"},
                ),
            },
            start_node="start",
        )

        # Test with flag=True
        result_yes = await executor.execute(workflow, {"flag": True})
        assert result_yes.success is True
        assert result_yes.get("path") == "good"

        # Test with flag=False
        result_no = await executor.execute(workflow, {"flag": False})
        assert result_no.success is True
        assert result_no.get("path") == "bad"

    @pytest.mark.asyncio
    async def test_stream_execution(self):
        """Test streaming execution."""
        executor = StateGraphExecutor()

        workflow = (
            WorkflowBuilder("stream_test")
            .add_transform("step1", lambda ctx: {**ctx, "step1": True})
            .add_transform("step2", lambda ctx: {**ctx, "step2": True})
            .build()
        )

        states = []
        async for node_id, state in executor.stream(workflow, {"input": "data"}):
            states.append((node_id, state))

        assert len(states) >= 2
        # Last state should have both steps
        final_state = states[-1][1]
        assert final_state.get("step1") is True
        assert final_state.get("step2") is True


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_executor(self):
        """Test getting default executor."""
        executor = get_executor()

        assert isinstance(executor, StateGraphExecutor)

        # Should return same instance
        executor2 = get_executor()
        assert executor is executor2

    @pytest.mark.asyncio
    async def test_execute_workflow_function(self):
        """Test execute_workflow convenience function."""
        workflow = (
            WorkflowBuilder("convenience")
            .add_transform("step", lambda ctx: {**ctx, "done": True})
            .build()
        )

        result = await execute_workflow(
            workflow,
            {"input": "data"},
        )

        assert result.success is True
        assert result.get("done") is True


class TestWorkflowPatterns:
    """Tests for common workflow patterns."""

    @pytest.mark.asyncio
    async def test_linear_workflow(self):
        """Test simple linear workflow: A -> B -> C."""
        executor = StateGraphExecutor()

        workflow = (
            WorkflowBuilder("linear")
            .add_transform("a", lambda ctx: {**ctx, "a": True})
            .add_transform("b", lambda ctx: {**ctx, "b": True})
            .add_transform("c", lambda ctx: {**ctx, "c": True})
            .build()
        )

        result = await executor.execute(workflow, {})

        assert result.success is True
        assert result.get("a") is True
        assert result.get("b") is True
        assert result.get("c") is True

    @pytest.mark.asyncio
    async def test_computation_workflow(self):
        """Test workflow that computes values."""
        executor = StateGraphExecutor()

        workflow = (
            WorkflowBuilder("compute")
            .add_transform("double", lambda ctx: {**ctx, "value": ctx.get("value", 0) * 2})
            .add_transform("add_ten", lambda ctx: {**ctx, "value": ctx.get("value", 0) + 10})
            .build()
        )

        result = await executor.execute(workflow, {"value": 5})

        assert result.success is True
        # (5 * 2) + 10 = 20
        assert result.get("value") == 20
