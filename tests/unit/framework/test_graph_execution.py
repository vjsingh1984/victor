"""TDD tests for NodeExecutionResult typed error propagation — Wave D.

Verifies: NodeExecutionResult dataclass, ok/fail factories, error_type preserved,
and NodeExecutor.execute_typed() uses NodeExecutionResult.
"""

from __future__ import annotations

import asyncio

import pytest


class TestNodeExecutionResult:
    def test_importable(self):
        from victor.framework.graph_execution import NodeExecutionResult

        result = NodeExecutionResult.ok(state={"x": 1})
        assert result.success is True
        assert result.state == {"x": 1}
        assert result.error is None
        assert result.error_type == ""

    def test_ok_sets_success_true(self):
        from victor.framework.graph_execution import NodeExecutionResult

        result = NodeExecutionResult.ok(state={"a": 1})
        assert result.success is True
        assert result.error is None

    def test_fail_preserves_exception_instance(self):
        from victor.framework.graph_execution import NodeExecutionResult

        exc = ValueError("something broke")
        result = NodeExecutionResult.fail(error=exc, state={})
        assert result.success is False
        assert result.error is exc

    def test_fail_sets_error_type_to_class_name(self):
        from victor.framework.graph_execution import NodeExecutionResult

        result = NodeExecutionResult.fail(error=ValueError("oops"), state={})
        assert result.error_type == "ValueError"

    def test_fail_sets_error_message_to_str_error(self):
        from victor.framework.graph_execution import NodeExecutionResult

        result = NodeExecutionResult.fail(error=RuntimeError("bad state"), state={})
        assert result.error_message == "bad state"

    def test_timeout_error_classified_correctly(self):
        from victor.framework.graph_execution import NodeExecutionResult

        exc = asyncio.TimeoutError()
        result = NodeExecutionResult.fail(error=exc, state={})
        assert result.error_type == "TimeoutError"

    def test_keyerror_classified_correctly(self):
        from victor.framework.graph_execution import NodeExecutionResult

        result = NodeExecutionResult.fail(error=KeyError("missing"), state={})
        assert result.error_type == "KeyError"

    def test_state_preserved_on_failure(self):
        from victor.framework.graph_execution import NodeExecutionResult

        state = {"progress": 0.5}
        result = NodeExecutionResult.fail(error=ValueError("fail"), state=state)
        assert result.state is state


class TestNodeExecutorTyped:
    async def test_execute_typed_returns_ok_on_success(self):
        from victor.framework.graph_execution import NodeExecutor, TimeoutManager

        class MockNode:
            async def execute(self, state):
                return {"result": "done"}

        executor = NodeExecutor(nodes={"test": MockNode()}, use_copy_on_write=False)
        timeout = TimeoutManager(timeout=None)

        result = await executor.execute_typed(
            node_id="test", state={"input": 1}, timeout_manager=timeout
        )
        assert result.success is True
        assert result.error is None

    async def test_execute_typed_returns_fail_on_exception(self):
        from victor.framework.graph_execution import NodeExecutor, TimeoutManager

        class FailingNode:
            async def execute(self, state):
                raise ValueError("node failed")

        executor = NodeExecutor(nodes={"fail": FailingNode()}, use_copy_on_write=False)
        timeout = TimeoutManager(timeout=None)

        result = await executor.execute_typed(node_id="fail", state={}, timeout_manager=timeout)
        assert result.success is False
        assert result.error_type == "ValueError"
        assert "node failed" in result.error_message

    async def test_execute_typed_classifies_timeout(self):
        from victor.framework.graph_execution import NodeExecutor, TimeoutManager

        class SlowNode:
            async def execute(self, state):
                await asyncio.sleep(10)
                return state

        executor = NodeExecutor(nodes={"slow": SlowNode()}, use_copy_on_write=False)
        timeout = TimeoutManager(timeout=0.01)
        timeout.start()  # must start timer before get_remaining() returns a value

        result = await executor.execute_typed(node_id="slow", state={}, timeout_manager=timeout)
        assert result.success is False
        assert result.error_type == "TimeoutError"

    def test_node_execution_result_in_graph_execution_module(self):
        import victor.framework.graph_execution as mod

        assert hasattr(
            mod, "NodeExecutionResult"
        ), "NodeExecutionResult must be defined in victor.framework.graph_execution"
