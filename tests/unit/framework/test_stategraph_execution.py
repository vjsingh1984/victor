# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Comprehensive tests for StateGraph execution.

Target: 60%+ coverage for victor/framework/graph.py
"""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.framework.graph import (
    StateGraph,
    CompiledGraph,
    CopyOnWriteState,
    WorkflowCheckpoint,
    CheckpointerProtocol,
    MemoryCheckpointer,
    GraphExecutionResult,
    END,
    IterationController,
    TimeoutManager,
)


@pytest.fixture
def memory_checkpointer():
    return MemoryCheckpointer()


class TestNodeCreation:
    def test_add_single_node(self):
        graph = StateGraph()
        result = graph.add_node("test_node", lambda s: s)
        assert result is graph
        assert "test_node" in graph._nodes

    def test_add_duplicate_node_raises_error(self):
        graph = StateGraph()
        graph.add_node("test_node", lambda s: s)
        with pytest.raises(ValueError):
            graph.add_node("test_node", lambda s: s)


class TestEdges:
    def test_add_normal_edge(self):
        graph = StateGraph()
        graph.add_node("a", lambda s: s)
        graph.add_node("b", lambda s: s)
        result = graph.add_edge("a", "b")
        assert result is graph

    def test_add_conditional_edge(self):
        graph = StateGraph()
        graph.add_node("router", lambda s: s)

        def route_func(state):
            return state.get("branch", "a")

        graph.add_conditional_edge("router", route_func, {"a": "branch_a"})
        assert "router" in graph._edges


class TestGraphCompilation:
    def test_compile_valid_graph(self):
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.add_node("end", lambda s: s)
        graph.add_edge("start", "end")
        graph.set_entry_point("start")
        compiled = graph.compile()
        assert isinstance(compiled, CompiledGraph)

    def test_compile_without_entry_point_fails(self):
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        with pytest.raises(ValueError, match="entry"):
            graph.compile()


class TestSimpleExecution:
    @pytest.mark.asyncio
    async def test_linear_execution(self):
        graph = StateGraph()

        def node_a(state):
            state["steps"] = state.get("steps", "") + "->a"
            return state

        def node_b(state):
            state["steps"] += "->b"
            return state

        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        graph.set_entry_point("a")

        compiled = graph.compile()
        result = await compiled.invoke({"steps": ""})

        assert result.success
        assert result.state["steps"] == "->a->b"


class TestConditionalEdges:
    @pytest.mark.asyncio
    async def test_conditional_routing(self):
        graph = StateGraph()

        def router(state):
            return "a"

        def branch_a(state):
            state["result"] = "A"
            return state

        graph.add_node("router", lambda s: s)
        graph.add_node("branch_a", branch_a)
        graph.add_conditional_edge("router", router, {"a": "branch_a"})
        graph.add_edge("branch_a", END)
        graph.set_entry_point("router")

        compiled = graph.compile()
        result = await compiled.invoke({})

        assert result.success
        assert result.state["result"] == "A"


class TestCyclesAndLoops:
    @pytest.mark.asyncio
    async def test_simple_cycle(self):
        graph = StateGraph()

        def counter(state):
            count = state.get("count", 0) + 1
            state["count"] = count
            return state

        def should_continue(state):
            return "continue" if state["count"] < 3 else "stop"

        graph.add_node("counter", counter)
        graph.add_node("stop", lambda s: s)
        graph.add_conditional_edge(
            "counter", should_continue, {"continue": "counter", "stop": "stop"}
        )
        graph.add_edge("stop", END)
        graph.set_entry_point("counter")

        compiled = graph.compile()
        result = await compiled.invoke({"count": 0})

        assert result.success
        assert result.state["count"] == 3


class TestStateManagement:
    def test_copy_on_write_read(self):
        original = {"key": "value"}
        cow = CopyOnWriteState(original)
        value = cow["key"]
        assert value == "value"
        assert not cow.was_modified

    def test_copy_on_write_write(self):
        original = {"key": "value"}
        cow = CopyOnWriteState(original)
        cow["key"] = "new_value"
        assert cow.was_modified
        assert cow["key"] == "new_value"


class TestCheckpointing:
    @pytest.mark.asyncio
    async def test_save_checkpoint(self, memory_checkpointer):
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp1",
            thread_id="thread1",
            node_id="node1",
            state={"value": "test"},
            timestamp=1234567890.0,
        )
        await memory_checkpointer.save(checkpoint)
        checkpoints = await memory_checkpointer.list("thread1")
        assert len(checkpoints) == 1


class TestHelperClasses:
    def test_iteration_controller(self):
        controller = IterationController(max_iterations=5, recursion_limit=10)
        for i in range(5):
            should_continue, _ = controller.should_continue("node1")
            assert should_continue is True
        should_continue, _ = controller.should_continue("node1")
        assert should_continue is False

    def test_timeout_manager_no_timeout(self):
        manager = TimeoutManager(timeout=None)
        manager.start()
        assert manager.get_remaining() is None
        assert manager.is_expired() is False


class TestStreamingExecution:
    @pytest.mark.asyncio
    async def test_stream_linear_graph(self):
        graph = StateGraph()

        graph.add_node("a", lambda s: {**s, "step": "a"})
        graph.add_node("b", lambda s: {**s, "step": "b"})
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        graph.set_entry_point("a")

        compiled = graph.compile()
        results = []
        async for node_id, state in compiled.stream({}):
            results.append(node_id)

        assert len(results) == 2


class TestIntegration:
    @pytest.mark.asyncio
    async def test_complex_workflow(self):
        graph = StateGraph()

        def process(state):
            count = state.get("count", 0) + 1
            state["count"] = count
            return state

        def should_retry(state):
            return "retry" if state["count"] < 3 else "done"

        graph.add_node("process", process)
        graph.add_node("done", lambda s: {**s, "status": "complete"})
        graph.add_conditional_edge("process", should_retry, {"retry": "process", "done": "done"})
        graph.add_edge("done", END)
        graph.set_entry_point("process")

        compiled = graph.compile()
        result = await compiled.invoke({"count": 0})

        assert result.success
        assert result.state["count"] == 3
        assert result.state["status"] == "complete"
