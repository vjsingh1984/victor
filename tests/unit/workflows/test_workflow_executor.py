"""Tests for WorkflowExecutor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.executor import (
    ExecutorNodeStatus,
    NodeResult,
    WorkflowContext,
    WorkflowExecutor,
    WorkflowResult,
)


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.settings = MagicMock()
    return orch


@pytest.fixture
def executor(mock_orchestrator):
    return WorkflowExecutor(mock_orchestrator, max_parallel=2, default_timeout=60.0)


class TestNodeResult:
    def test_success_property(self):
        r = NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED, output="ok")
        assert r.success is True

    def test_failed_not_success(self):
        r = NodeResult(node_id="n1", status=ExecutorNodeStatus.FAILED, error="err")
        assert r.success is False

    def test_to_dict(self):
        r = NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED, output="data")
        d = r.to_dict()
        assert d["node_id"] == "n1"
        assert d["status"] == "completed"
        assert d["output"] == "data"


class TestWorkflowContext:
    def test_get_set(self):
        ctx = WorkflowContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_get_default(self):
        ctx = WorkflowContext()
        assert ctx.get("missing", "default") == "default"

    def test_update(self):
        ctx = WorkflowContext()
        ctx.update({"a": 1, "b": 2})
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2

    def test_add_and_get_result(self):
        ctx = WorkflowContext()
        result = NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED, output="x")
        ctx.add_result(result)
        assert ctx.get_result("n1") is result

    def test_has_failures(self):
        ctx = WorkflowContext()
        ctx.add_result(NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED))
        assert ctx.has_failures() is False
        ctx.add_result(NodeResult(node_id="n2", status=ExecutorNodeStatus.FAILED, error="e"))
        assert ctx.has_failures() is True

    def test_get_outputs(self):
        ctx = WorkflowContext()
        ctx.add_result(
            NodeResult(node_id="n1", status=ExecutorNodeStatus.COMPLETED, output="data1")
        )
        ctx.add_result(NodeResult(node_id="n2", status=ExecutorNodeStatus.FAILED, error="e"))
        outputs = ctx.get_outputs()
        assert "n1" in outputs
        assert "n2" not in outputs


class TestWorkflowResult:
    def test_get_output(self):
        ctx = WorkflowContext()
        ctx.add_result(
            NodeResult(node_id="analyze", status=ExecutorNodeStatus.COMPLETED, output="result")
        )
        wr = WorkflowResult(workflow_name="test", success=True, context=ctx)
        assert wr.get_output("analyze") == "result"
        assert wr.get_output("missing") is None

    def test_to_dict(self):
        ctx = WorkflowContext()
        wr = WorkflowResult(workflow_name="wf", success=True, context=ctx, total_duration=1.5)
        d = wr.to_dict()
        assert d["workflow_name"] == "wf"
        assert d["success"] is True
        assert d["total_duration"] == 1.5


class TestWorkflowExecutorInit:
    def test_default_attributes(self, executor):
        assert executor.max_parallel == 2
        assert executor.default_timeout == 60.0

    async def test_execute_empty_workflow_raises(self, executor):
        workflow = MagicMock()
        workflow.name = "empty"
        workflow.start_node = None
        workflow.metadata = {}

        result = await executor.execute(workflow, {})
        # An empty workflow with no start_node should error
        assert result.success is False
        assert result.error is not None

    async def test_execute_passes_initial_context(self, executor):
        """Verify initial_context data is available in WorkflowContext."""
        workflow = MagicMock()
        workflow.name = "test_wf"
        workflow.start_node = "step1"
        workflow.metadata = {}

        node = MagicMock()
        node.id = "step1"
        node.next_nodes = []
        node.condition = None
        workflow.get_node.return_value = node

        # Mock _execute_node to capture context
        captured_ctx = {}

        async def mock_execute_node(n, ctx):
            captured_ctx.update(ctx.data)
            return NodeResult(node_id=n.id, status=ExecutorNodeStatus.COMPLETED, output="done")

        with patch.object(executor, "_execute_node", side_effect=mock_execute_node):
            with patch.object(executor, "_get_next_nodes", return_value=[]):
                with patch.object(executor, "_emit_workflow_completed_event"):
                    with patch.object(executor, "_emit_workflow_step_event"):
                        result = await executor.execute(
                            workflow, initial_context={"files": ["main.py"]}
                        )

        assert captured_ctx.get("files") == ["main.py"]


class TestWorkflowExecutorChainHandlers:
    @pytest.mark.asyncio
    async def test_execute_chain_handler_uses_asyncio_to_thread_for_sync_invoke(self, executor):
        node = MagicMock()
        node.id = "compute_sync_invoke"
        node.output_key = "chain_output"
        node.input_mapping = {"payload": "value"}
        context = WorkflowContext(data={"value": 42})

        chain_obj = MagicMock()
        chain_obj.invoke = MagicMock(return_value={"result": "ok"})
        registry = MagicMock()
        registry.create.return_value = chain_obj

        async def call_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("victor.workflows.executor.get_chain_registry", return_value=registry),
            patch(
                "victor.workflows.executor.asyncio.to_thread", side_effect=call_to_thread
            ) as mock_to_thread,
        ):
            result = await executor._execute_chain_handler(node, context, "analysis_chain", 0.0)

        assert result.status is ExecutorNodeStatus.COMPLETED
        assert result.output == {"result": "ok"}
        assert context.get("chain_output") == {"result": "ok"}
        mock_to_thread.assert_awaited_once()
        called = mock_to_thread.await_args
        assert called.args[0] is chain_obj.invoke
        assert called.args[1] == {"payload": 42}

    @pytest.mark.asyncio
    async def test_execute_chain_handler_uses_asyncio_to_thread_for_sync_callable(self, executor):
        node = MagicMock()
        node.id = "compute_sync_callable"
        node.output_key = "callable_output"
        node.input_mapping = {"payload": "value"}
        context = WorkflowContext(data={"value": "repo"})

        def sync_chain(**kwargs):
            return {"seen": kwargs}

        registry = MagicMock()
        registry.create.return_value = sync_chain

        async def call_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("victor.workflows.executor.get_chain_registry", return_value=registry),
            patch(
                "victor.workflows.executor.asyncio.to_thread", side_effect=call_to_thread
            ) as mock_to_thread,
        ):
            result = await executor._execute_chain_handler(node, context, "callable_chain", 0.0)

        assert result.status is ExecutorNodeStatus.COMPLETED
        assert result.output == {"seen": {"payload": "repo"}}
        assert context.get("callable_output") == {"seen": {"payload": "repo"}}
        mock_to_thread.assert_awaited_once()
        called = mock_to_thread.await_args
        assert called.args[0] is sync_chain
        assert called.kwargs == {"payload": "repo"}
