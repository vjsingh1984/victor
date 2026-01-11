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

"""Tests for victor.framework.graph module (StateGraph).

These tests verify the LangGraph-compatible StateGraph implementation
including cyclic workflows, checkpointing, and typed state management.
"""

import pytest
from typing import TypedDict, Optional, List
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.graph import (
    StateGraph,
    CompiledGraph,
    Node,
    Edge,
    EdgeType,
    FrameworkNodeStatus,
    GraphExecutionResult,
    GraphConfig,
    WorkflowCheckpoint,
    CheckpointerProtocol,
    MemoryCheckpointer,
    RLCheckpointerAdapter,
    END,
    START,
    create_graph,
)

from victor.framework.config import (
    ExecutionConfig,
    CheckpointConfig,
    InterruptConfig,
    PerformanceConfig,
    ObservabilityConfig,
)


# =============================================================================
# Test State Types
# =============================================================================


class SimpleState(TypedDict):
    """Simple state for testing."""

    value: int
    history: List[str]


class TaskState(TypedDict, total=False):
    """Task state with optional fields."""

    task: str
    result: Optional[str]
    iteration: int
    complete: bool


# =============================================================================
# Node Functions for Testing
# =============================================================================


async def increment_node(state: SimpleState) -> SimpleState:
    """Node that increments value."""
    state["value"] += 1
    state["history"].append("increment")
    return state


async def double_node(state: SimpleState) -> SimpleState:
    """Node that doubles value."""
    state["value"] *= 2
    state["history"].append("double")
    return state


def sync_node(state: SimpleState) -> SimpleState:
    """Synchronous node for testing."""
    state["value"] += 10
    state["history"].append("sync")
    return state


async def task_process_node(state: TaskState) -> TaskState:
    """Process task and increment iteration."""
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def task_check_node(state: TaskState) -> TaskState:
    """Check if task is complete."""
    if state.get("iteration", 0) >= 3:
        state["complete"] = True
        state["result"] = "Completed"
    return state


def should_continue(state: TaskState) -> str:
    """Condition: continue processing or end."""
    if state.get("complete", False):
        return "done"
    return "continue"


# =============================================================================
# Node and Edge Tests
# =============================================================================


class TestNode:
    """Tests for Node class."""

    def test_node_creation(self):
        """Node should store id, func, and metadata."""
        node = Node(id="test", func=increment_node)

        assert node.id == "test"
        assert node.func == increment_node
        assert node.metadata == {}

    def test_node_with_metadata(self):
        """Node should store metadata."""
        node = Node(id="test", func=increment_node, metadata={"key": "value"})

        assert node.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_node_execute_async(self):
        """execute should run async function."""
        node = Node(id="inc", func=increment_node)
        state: SimpleState = {"value": 5, "history": []}

        result = await node.execute(state)

        assert result["value"] == 6
        assert result["history"] == ["increment"]

    @pytest.mark.asyncio
    async def test_node_execute_sync(self):
        """execute should handle sync function."""
        node = Node(id="sync", func=sync_node)
        state: SimpleState = {"value": 5, "history": []}

        result = await node.execute(state)

        assert result["value"] == 15
        assert result["history"] == ["sync"]


class TestEdge:
    """Tests for Edge class."""

    def test_normal_edge(self):
        """Normal edge should store source and target."""
        edge = Edge(source="a", target="b")

        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.edge_type == EdgeType.NORMAL
        assert edge.condition is None

    def test_conditional_edge(self):
        """Conditional edge should store branches and condition."""
        edge = Edge(
            source="a",
            target={"yes": "b", "no": "c"},
            edge_type=EdgeType.CONDITIONAL,
            condition=lambda s: "yes" if s.get("flag") else "no",
        )

        assert edge.edge_type == EdgeType.CONDITIONAL
        assert edge.target == {"yes": "b", "no": "c"}

    def test_get_target_normal(self):
        """get_target should return target for normal edge."""
        edge = Edge(source="a", target="b")

        assert edge.get_target({}) == "b"

    def test_get_target_conditional(self):
        """get_target should evaluate condition for conditional edge."""
        edge = Edge(
            source="a",
            target={"yes": "b", "no": "c"},
            edge_type=EdgeType.CONDITIONAL,
            condition=lambda s: "yes" if s.get("flag") else "no",
        )

        assert edge.get_target({"flag": True}) == "b"
        assert edge.get_target({"flag": False}) == "c"


# =============================================================================
# WorkflowCheckpoint Tests
# =============================================================================


class TestCheckpoint:
    """Tests for WorkflowCheckpoint class."""

    def test_checkpoint_creation(self):
        """WorkflowCheckpoint should store all fields."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp1",
            thread_id="t1",
            node_id="node1",
            state={"value": 42},
            timestamp=1234567890.0,
            metadata={"key": "value"},
        )

        assert checkpoint.checkpoint_id == "cp1"
        assert checkpoint.thread_id == "t1"
        assert checkpoint.node_id == "node1"
        assert checkpoint.state == {"value": 42}
        assert checkpoint.timestamp == 1234567890.0
        assert checkpoint.metadata == {"key": "value"}

    def test_to_dict(self):
        """to_dict should serialize checkpoint."""
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp1",
            thread_id="t1",
            node_id="node1",
            state={"value": 42},
            timestamp=1234567890.0,
        )

        d = checkpoint.to_dict()

        assert d["checkpoint_id"] == "cp1"
        assert d["thread_id"] == "t1"
        assert d["state"] == {"value": 42}

    def test_from_dict(self):
        """from_dict should deserialize checkpoint."""
        data = {
            "checkpoint_id": "cp1",
            "thread_id": "t1",
            "node_id": "node1",
            "state": {"value": 42},
            "timestamp": 1234567890.0,
            "metadata": {"key": "value"},
        }

        checkpoint = WorkflowCheckpoint.from_dict(data)

        assert checkpoint.checkpoint_id == "cp1"
        assert checkpoint.metadata == {"key": "value"}


class TestMemoryCheckpointer:
    """Tests for MemoryCheckpointer."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """save and load should work together."""
        checkpointer = MemoryCheckpointer()

        checkpoint = WorkflowCheckpoint(
            checkpoint_id="cp1",
            thread_id="t1",
            node_id="node1",
            state={"value": 42},
            timestamp=1234567890.0,
        )

        await checkpointer.save(checkpoint)
        loaded = await checkpointer.load("t1")

        assert loaded is not None
        assert loaded.checkpoint_id == "cp1"
        assert loaded.state == {"value": 42}

    @pytest.mark.asyncio
    async def test_load_returns_latest(self):
        """load should return latest checkpoint."""
        checkpointer = MemoryCheckpointer()

        await checkpointer.save(
            WorkflowCheckpoint(
                checkpoint_id="cp1",
                thread_id="t1",
                node_id="node1",
                state={"value": 1},
                timestamp=1.0,
            )
        )
        await checkpointer.save(
            WorkflowCheckpoint(
                checkpoint_id="cp2",
                thread_id="t1",
                node_id="node2",
                state={"value": 2},
                timestamp=2.0,
            )
        )

        loaded = await checkpointer.load("t1")

        assert loaded.checkpoint_id == "cp2"
        assert loaded.state == {"value": 2}

    @pytest.mark.asyncio
    async def test_load_unknown_thread(self):
        """load should return None for unknown thread."""
        checkpointer = MemoryCheckpointer()

        loaded = await checkpointer.load("unknown")

        assert loaded is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """list should return all checkpoints for thread."""
        checkpointer = MemoryCheckpointer()

        await checkpointer.save(
            WorkflowCheckpoint(
                checkpoint_id="cp1",
                thread_id="t1",
                node_id="n1",
                state={},
                timestamp=1.0,
            )
        )
        await checkpointer.save(
            WorkflowCheckpoint(
                checkpoint_id="cp2",
                thread_id="t1",
                node_id="n2",
                state={},
                timestamp=2.0,
            )
        )

        checkpoints = await checkpointer.list("t1")

        assert len(checkpoints) == 2


# =============================================================================
# StateGraph Tests
# =============================================================================


class TestStateGraph:
    """Tests for StateGraph class."""

    def test_create_graph(self):
        """StateGraph should be creatable."""
        graph = StateGraph(SimpleState)

        assert graph._state_schema == SimpleState
        assert graph._nodes == {}
        assert graph._edges == {}

    def test_add_node(self):
        """add_node should register node."""
        graph = StateGraph(SimpleState)

        graph.add_node("inc", increment_node)

        assert "inc" in graph._nodes
        assert graph._nodes["inc"].func == increment_node

    def test_add_node_chaining(self):
        """add_node should return self for chaining."""
        graph = StateGraph(SimpleState)

        result = graph.add_node("a", increment_node).add_node("b", double_node)

        assert result is graph
        assert len(graph._nodes) == 2

    def test_add_duplicate_node_raises(self):
        """add_node should raise for duplicate nodes."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)

        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("a", double_node)

    def test_add_edge(self):
        """add_edge should register edge."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.add_node("b", double_node)

        graph.add_edge("a", "b")

        assert len(graph._edges["a"]) == 1
        assert graph._edges["a"][0].target == "b"

    def test_add_conditional_edge(self):
        """add_conditional_edge should register conditional edge."""
        graph = StateGraph(TaskState)
        graph.add_node("a", task_process_node)
        graph.add_node("b", task_check_node)

        graph.add_conditional_edge("a", should_continue, {"continue": "a", "done": "b"})

        edge = graph._edges["a"][0]
        assert edge.edge_type == EdgeType.CONDITIONAL
        assert edge.target == {"continue": "a", "done": "b"}

    def test_set_entry_point(self):
        """set_entry_point should set entry."""
        graph = StateGraph(SimpleState)
        graph.add_node("start", increment_node)

        graph.set_entry_point("start")

        assert graph._entry_point == "start"

    def test_set_entry_point_invalid(self):
        """set_entry_point should raise for unknown node."""
        graph = StateGraph(SimpleState)

        with pytest.raises(ValueError, match="not found"):
            graph.set_entry_point("unknown")

    def test_set_finish_point(self):
        """set_finish_point should add edge to END."""
        graph = StateGraph(SimpleState)
        graph.add_node("last", increment_node)

        graph.set_finish_point("last")

        assert graph._edges["last"][0].target == END


class TestStateGraphCompile:
    """Tests for StateGraph compilation."""

    def test_compile_returns_compiled_graph(self):
        """compile should return CompiledGraph."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.set_entry_point("a")
        graph.add_edge("a", END)

        compiled = graph.compile()

        assert isinstance(compiled, CompiledGraph)

    def test_compile_validates_nodes(self):
        """compile should fail if no nodes."""
        graph = StateGraph(SimpleState)

        with pytest.raises(ValueError, match="no nodes"):
            graph.compile()

    def test_compile_validates_entry_point(self):
        """compile should fail if no entry point."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)

        with pytest.raises(ValueError, match="No entry point"):
            graph.compile()

    def test_compile_validates_edge_targets(self):
        """compile should fail for invalid edge targets."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.set_entry_point("a")
        graph.add_edge("a", "nonexistent")  # Invalid target

        with pytest.raises(ValueError, match="not found"):
            graph.compile()


# =============================================================================
# CompiledGraph Execution Tests
# =============================================================================


class TestCompiledGraphExecution:
    """Tests for CompiledGraph execution."""

    @pytest.fixture
    def simple_graph(self):
        """Create simple linear graph."""
        graph = StateGraph(SimpleState)
        graph.add_node("inc", increment_node)
        graph.add_node("double", double_node)
        graph.add_edge("inc", "double")
        graph.add_edge("double", END)
        graph.set_entry_point("inc")
        return graph.compile()

    @pytest.mark.asyncio
    async def test_invoke_linear_graph(self, simple_graph):
        """invoke should execute nodes in order."""
        state: SimpleState = {"value": 5, "history": []}

        result = await simple_graph.invoke(state)

        assert result.success is True
        assert result.state["value"] == 12  # (5+1)*2
        assert result.state["history"] == ["increment", "double"]
        assert result.node_history == ["inc", "double"]

    @pytest.mark.asyncio
    async def test_invoke_tracks_iterations(self, simple_graph):
        """invoke should track iterations."""
        state: SimpleState = {"value": 0, "history": []}

        result = await simple_graph.invoke(state)

        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_invoke_measures_duration(self, simple_graph):
        """invoke should measure duration."""
        state: SimpleState = {"value": 0, "history": []}

        result = await simple_graph.invoke(state)

        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_invoke_with_cycle(self):
        """invoke should handle cyclic graphs."""
        graph = StateGraph(TaskState)
        graph.add_node("process", task_process_node)
        graph.add_node("check", task_check_node)

        graph.add_edge("process", "check")
        graph.add_conditional_edge(
            "check",
            should_continue,
            {"continue": "process", "done": END},
        )
        graph.set_entry_point("process")
        compiled = graph.compile()

        state: TaskState = {"task": "test", "iteration": 0, "complete": False}
        result = await compiled.invoke(state)

        assert result.success is True
        assert result.state["complete"] is True
        assert result.state["iteration"] == 3

    @pytest.mark.asyncio
    async def test_invoke_respects_max_iterations(self):
        """invoke should stop at max_iterations."""

        async def infinite_node(state: SimpleState) -> SimpleState:
            state["value"] += 1
            return state

        graph = StateGraph(SimpleState)
        graph.add_node("loop", infinite_node)
        graph.add_edge("loop", "loop")  # Infinite cycle
        graph.set_entry_point("loop")
        compiled = graph.compile(max_iterations=5)

        state: SimpleState = {"value": 0, "history": []}
        result = await compiled.invoke(state)

        assert result.success is False
        assert "Max iterations" in result.error
        assert result.state["value"] == 5  # Ran 5 times

    @pytest.mark.asyncio
    async def test_invoke_with_checkpointer(self):
        """invoke should save checkpoints."""
        checkpointer = MemoryCheckpointer()

        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.add_node("b", double_node)
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        graph.set_entry_point("a")
        compiled = graph.compile(checkpointer=checkpointer)

        state: SimpleState = {"value": 5, "history": []}
        await compiled.invoke(state, thread_id="test_thread")

        checkpoints = await checkpointer.list("test_thread")
        assert len(checkpoints) == 2  # One per node


class TestCompiledGraphStream:
    """Tests for CompiledGraph streaming."""

    @pytest.mark.asyncio
    async def test_stream_yields_states(self):
        """stream should yield state after each node."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.add_node("b", double_node)
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        graph.set_entry_point("a")
        compiled = graph.compile()

        state: SimpleState = {"value": 5, "history": []}
        results = []

        async for node_id, node_state in compiled.stream(state):
            results.append((node_id, node_state["value"]))

        assert results == [("a", 6), ("b", 12)]


class TestCompiledGraphSchema:
    """Tests for graph schema extraction."""

    def test_get_graph_schema(self):
        """get_graph_schema should return graph structure."""
        graph = StateGraph(SimpleState)
        graph.add_node("a", increment_node)
        graph.add_node("b", double_node)
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        graph.set_entry_point("a")
        compiled = graph.compile()

        schema = compiled.get_graph_schema()

        assert set(schema["nodes"]) == {"a", "b"}
        assert schema["entry_point"] == "a"
        assert "a" in schema["edges"]


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateGraph:
    """Tests for create_graph factory."""

    def test_create_graph_returns_state_graph(self):
        """create_graph should return StateGraph."""
        graph = create_graph(SimpleState)

        assert isinstance(graph, StateGraph)
        assert graph._state_schema == SimpleState


# =============================================================================
# GraphConfig Tests
# =============================================================================


class TestGraphConfig:
    """Tests for GraphConfig."""

    def test_defaults(self):
        """GraphConfig should have sensible defaults."""
        config = GraphConfig()

        # Test focused config structure (ISP compliance)
        assert config.execution.max_iterations == 25
        assert config.execution.timeout is None
        assert config.execution.recursion_limit == 100
        assert config.checkpoint.checkpointer is None
        assert config.interrupt.interrupt_before == []
        assert config.interrupt.interrupt_after == []

    def test_custom_values(self):
        """GraphConfig should accept custom values via focused configs (ISP compliant)."""
        checkpointer = MemoryCheckpointer()
        config = GraphConfig(
            execution=ExecutionConfig(max_iterations=50, timeout=300.0),
            checkpoint=CheckpointConfig(checkpointer=checkpointer),
            interrupt=InterruptConfig(interrupt_before=["review"]),
        )

        assert config.execution.max_iterations == 50
        assert config.execution.timeout == 300.0
        assert config.checkpoint.checkpointer == checkpointer
        assert config.interrupt.interrupt_before == ["review"]

    def test_from_legacy_migration(self):
        """GraphConfig.from_legacy() should migrate from legacy format."""
        checkpointer = MemoryCheckpointer()
        config = GraphConfig.from_legacy(
            max_iterations=50,
            timeout=300.0,
            checkpointer=checkpointer,
            interrupt_before=["review"],
        )

        assert config.execution.max_iterations == 50
        assert config.execution.timeout == 300.0
        assert config.checkpoint.checkpointer == checkpointer
        assert config.interrupt.interrupt_before == ["review"]


# =============================================================================
# Framework Export Tests
# =============================================================================


class TestFrameworkExports:
    """Tests for framework module exports."""

    def test_graph_exported_from_framework(self):
        """Graph types should be exported from victor.framework."""
        from victor.framework import (
            StateGraph,
            CompiledGraph,
            Node,
            Edge,
            EdgeType,
            GraphExecutionResult,
            GraphConfig,
            WorkflowCheckpoint,
            MemoryCheckpointer,
            RLCheckpointerAdapter,
            END,
            START,
            create_graph,
        )

        assert StateGraph is not None
        assert CompiledGraph is not None
        assert END == "__end__"
        assert START == "__start__"

    def test_edge_types(self):
        """EdgeType enum should have correct values."""
        from victor.framework import EdgeType

        assert EdgeType.NORMAL.value == "normal"
        assert EdgeType.CONDITIONAL.value == "conditional"

    def test_node_status(self):
        """FrameworkNodeStatus enum should have correct values."""
        from victor.framework import FrameworkNodeStatus

        assert FrameworkNodeStatus.PENDING.value == "pending"
        assert FrameworkNodeStatus.RUNNING.value == "running"
        assert FrameworkNodeStatus.COMPLETED.value == "completed"
        assert FrameworkNodeStatus.FAILED.value == "failed"
        assert FrameworkNodeStatus.SKIPPED.value == "skipped"
