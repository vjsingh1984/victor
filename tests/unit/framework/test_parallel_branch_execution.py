"""Tests for graph-level parallel branch execution.

Tests the ParallelBranchExecutor and StateGraph.add_parallel_edges functionality.
"""

import asyncio
import time
from typing import Any, Dict, List

import pytest

from victor.framework.graph import (
    END,
    StateGraph,
    Edge,
    EdgeType,
    ParallelBranchExecutor,
    ParallelBranchResult,
    CopyOnWriteState,
)


class TestEdgeParallelSupport:
    """Tests for Edge class parallel support."""

    def test_edge_type_parallel_exists(self):
        """Test EdgeType.PARALLEL is defined."""
        assert EdgeType.PARALLEL.value == "parallel"

    def test_edge_is_parallel(self):
        """Test Edge.is_parallel() method."""
        normal_edge = Edge(source="a", target="b", edge_type=EdgeType.NORMAL)
        parallel_edge = Edge(
            source="a",
            target=["b", "c", "d"],
            edge_type=EdgeType.PARALLEL,
        )

        assert not normal_edge.is_parallel()
        assert parallel_edge.is_parallel()

    def test_edge_get_parallel_targets(self):
        """Test Edge.get_parallel_targets() method."""
        parallel_edge = Edge(
            source="a",
            target=["b", "c", "d"],
            edge_type=EdgeType.PARALLEL,
        )
        normal_edge = Edge(source="a", target="b", edge_type=EdgeType.NORMAL)

        assert parallel_edge.get_parallel_targets() == ["b", "c", "d"]
        assert normal_edge.get_parallel_targets() == []

    def test_parallel_edge_get_target_returns_none(self):
        """Test that get_target returns None for parallel edges."""
        parallel_edge = Edge(
            source="a",
            target=["b", "c"],
            edge_type=EdgeType.PARALLEL,
        )
        assert parallel_edge.get_target({}) is None


class TestStateGraphParallelEdges:
    """Tests for StateGraph.add_parallel_edges method."""

    def test_add_parallel_edges_basic(self):
        """Test adding parallel edges to a graph."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.add_node("branch_a", lambda s: s)
        graph.add_node("branch_b", lambda s: s)
        graph.add_node("join", lambda s: s)

        graph.add_parallel_edges(
            "start",
            ["branch_a", "branch_b"],
            join_node="join",
        )

        assert "start" in graph._edges
        edges = graph._edges["start"]
        assert len(edges) == 1
        assert edges[0].is_parallel()
        assert edges[0].get_parallel_targets() == ["branch_a", "branch_b"]
        assert edges[0].join_node == "join"

    def test_add_parallel_edges_requires_two_targets(self):
        """Test that parallel edges require at least 2 targets."""
        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.add_node("only_one", lambda s: s)

        with pytest.raises(ValueError, match="at least 2 targets"):
            graph.add_parallel_edges("start", ["only_one"])

    def test_add_parallel_edges_with_merge_func(self):
        """Test parallel edges with custom merge function."""

        def custom_merge(states: List[Dict]) -> Dict:
            return {"merged": [s.get("value") for s in states]}

        graph = StateGraph()
        graph.add_node("start", lambda s: s)
        graph.add_node("a", lambda s: s)
        graph.add_node("b", lambda s: s)

        graph.add_parallel_edges(
            "start",
            ["a", "b"],
            merge_func=custom_merge,
        )

        edge = graph._edges["start"][0]
        assert edge.merge_func is custom_merge

    def test_add_parallel_edges_chaining(self):
        """Test method chaining with add_parallel_edges."""
        graph = (
            StateGraph()
            .add_node("start", lambda s: s)
            .add_node("a", lambda s: s)
            .add_node("b", lambda s: s)
            .add_parallel_edges("start", ["a", "b"])
        )

        assert "start" in graph._edges


class TestParallelBranchResult:
    """Tests for ParallelBranchResult dataclass."""

    def test_parallel_branch_result_creation(self):
        """Test creating ParallelBranchResult."""
        result = ParallelBranchResult(
            branch_id="branch_a",
            success=True,
            state={"key": "value"},
            node_history=["node1", "node2"],
        )

        assert result.branch_id == "branch_a"
        assert result.success is True
        assert result.state == {"key": "value"}
        assert result.error is None
        assert result.node_history == ["node1", "node2"]

    def test_parallel_branch_result_with_error(self):
        """Test ParallelBranchResult with error."""
        result = ParallelBranchResult(
            branch_id="branch_a",
            success=False,
            state={},
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"


class TestParallelBranchExecutor:
    """Tests for ParallelBranchExecutor class."""

    @pytest.mark.asyncio
    async def test_execute_parallel_branches_basic(self):
        """Test basic parallel branch execution."""
        from victor.framework.graph import Node, TimeoutManager

        # Create simple nodes
        async def branch_a(state):
            state = dict(state)
            state["branch_a"] = True
            return state

        async def branch_b(state):
            state = dict(state)
            state["branch_b"] = True
            return state

        nodes = {
            "branch_a": Node(id="branch_a", func=branch_a),
            "branch_b": Node(id="branch_b", func=branch_b),
        }
        edges: Dict[str, List[Edge]] = {}

        executor = ParallelBranchExecutor(
            nodes=nodes,
            edges=edges,
            use_copy_on_write=True,
        )

        timeout_manager = TimeoutManager(timeout=30.0)
        timeout_manager.start()

        success, error, merged_state, history = await executor.execute_parallel_branches(
            branch_targets=["branch_a", "branch_b"],
            state={"initial": True},
            join_node=None,
            merge_func=None,
            timeout_manager=timeout_manager,
        )

        assert success is True
        assert error is None
        assert merged_state["initial"] is True
        assert merged_state["branch_a"] is True
        assert merged_state["branch_b"] is True
        assert "branch_a" in history
        assert "branch_b" in history

    @pytest.mark.asyncio
    async def test_execute_parallel_branches_with_delay(self):
        """Test that branches execute concurrently (not sequentially)."""
        from victor.framework.graph import Node, TimeoutManager

        async def slow_branch_a(state):
            await asyncio.sleep(0.1)
            state = dict(state)
            state["branch_a_time"] = time.time()
            return state

        async def slow_branch_b(state):
            await asyncio.sleep(0.1)
            state = dict(state)
            state["branch_b_time"] = time.time()
            return state

        nodes = {
            "branch_a": Node(id="branch_a", func=slow_branch_a),
            "branch_b": Node(id="branch_b", func=slow_branch_b),
        }
        edges: Dict[str, List[Edge]] = {}

        executor = ParallelBranchExecutor(
            nodes=nodes,
            edges=edges,
            use_copy_on_write=True,
        )

        timeout_manager = TimeoutManager(timeout=30.0)
        timeout_manager.start()

        start_time = time.time()
        success, error, merged_state, _ = await executor.execute_parallel_branches(
            branch_targets=["branch_a", "branch_b"],
            state={},
            join_node=None,
            merge_func=None,
            timeout_manager=timeout_manager,
        )
        elapsed = time.time() - start_time

        assert success is True
        # If sequential, would take ~0.2s. Parallel should take ~0.1s
        assert elapsed < 0.18, f"Branches should run in parallel, but took {elapsed}s"

    @pytest.mark.asyncio
    async def test_execute_parallel_branches_with_custom_merge(self):
        """Test parallel execution with custom merge function."""
        from victor.framework.graph import Node, TimeoutManager

        async def branch_a(state):
            return {"result": "a"}

        async def branch_b(state):
            return {"result": "b"}

        def custom_merge(states: List[Dict]) -> Dict:
            return {"results": [s["result"] for s in states]}

        nodes = {
            "branch_a": Node(id="branch_a", func=branch_a),
            "branch_b": Node(id="branch_b", func=branch_b),
        }
        edges: Dict[str, List[Edge]] = {}

        executor = ParallelBranchExecutor(
            nodes=nodes,
            edges=edges,
            use_copy_on_write=True,
        )

        timeout_manager = TimeoutManager(timeout=30.0)
        timeout_manager.start()

        success, error, merged_state, _ = await executor.execute_parallel_branches(
            branch_targets=["branch_a", "branch_b"],
            state={},
            join_node=None,
            merge_func=custom_merge,
            timeout_manager=timeout_manager,
        )

        assert success is True
        assert "results" in merged_state
        assert set(merged_state["results"]) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_execute_parallel_branches_failure(self):
        """Test parallel execution with branch failure."""
        from victor.framework.graph import Node, TimeoutManager

        async def branch_a(state):
            return {"branch_a": True}

        async def branch_b(state):
            raise ValueError("Branch B failed")

        nodes = {
            "branch_a": Node(id="branch_a", func=branch_a),
            "branch_b": Node(id="branch_b", func=branch_b),
        }
        edges: Dict[str, List[Edge]] = {}

        executor = ParallelBranchExecutor(
            nodes=nodes,
            edges=edges,
            use_copy_on_write=True,
        )

        timeout_manager = TimeoutManager(timeout=30.0)
        timeout_manager.start()

        success, error, _, _ = await executor.execute_parallel_branches(
            branch_targets=["branch_a", "branch_b"],
            state={},
            join_node=None,
            merge_func=None,
            timeout_manager=timeout_manager,
        )

        assert success is False
        assert "branch_b" in error
        assert "Branch B failed" in error

    @pytest.mark.asyncio
    async def test_execute_parallel_branches_state_isolation(self):
        """Test that branches have isolated state copies."""
        from victor.framework.graph import Node, TimeoutManager

        async def branch_a(state):
            state = dict(state)
            state["shared_list"] = state.get("shared_list", []) + ["a"]
            return state

        async def branch_b(state):
            state = dict(state)
            state["shared_list"] = state.get("shared_list", []) + ["b"]
            return state

        nodes = {
            "branch_a": Node(id="branch_a", func=branch_a),
            "branch_b": Node(id="branch_b", func=branch_b),
        }
        edges: Dict[str, List[Edge]] = {}

        executor = ParallelBranchExecutor(
            nodes=nodes,
            edges=edges,
            use_copy_on_write=True,
        )

        timeout_manager = TimeoutManager(timeout=30.0)
        timeout_manager.start()

        initial_state = {"shared_list": ["initial"]}
        success, _, merged_state, _ = await executor.execute_parallel_branches(
            branch_targets=["branch_a", "branch_b"],
            state=initial_state,
            join_node=None,
            merge_func=None,
            timeout_manager=timeout_manager,
        )

        assert success is True
        # Default merge concatenates lists
        merged_list = merged_state["shared_list"]
        assert "initial" in merged_list
        assert "a" in merged_list
        assert "b" in merged_list


class TestCompiledGraphParallelExecution:
    """Integration tests for parallel execution in CompiledGraph."""

    @pytest.mark.asyncio
    async def test_graph_with_parallel_branches(self):
        """Test full graph execution with parallel branches."""

        async def start_node(state):
            state["started"] = True
            return state

        async def branch_a(state):
            state = dict(state)
            state["branch_a_done"] = True
            return state

        async def branch_b(state):
            state = dict(state)
            state["branch_b_done"] = True
            return state

        async def join_node(state):
            state["joined"] = True
            return state

        graph = StateGraph()
        graph.add_node("start", start_node)
        graph.add_node("branch_a", branch_a)
        graph.add_node("branch_b", branch_b)
        graph.add_node("join", join_node)

        graph.set_entry_point("start")
        graph.add_parallel_edges(
            "start",
            ["branch_a", "branch_b"],
            join_node="join",
        )
        graph.add_edge("join", END)

        compiled = graph.compile()
        result = await compiled.invoke({"initial": True})

        assert result.success is True
        assert result.state["started"] is True
        assert result.state["branch_a_done"] is True
        assert result.state["branch_b_done"] is True
        assert result.state["joined"] is True

    @pytest.mark.asyncio
    async def test_graph_parallel_without_join(self):
        """Test parallel branches without explicit join node."""

        async def start_node(state):
            state["started"] = True
            return state

        async def branch_a(state):
            state = dict(state)
            state["branch_a"] = True
            return state

        async def branch_b(state):
            state = dict(state)
            state["branch_b"] = True
            return state

        graph = StateGraph()
        graph.add_node("start", start_node)
        graph.add_node("branch_a", branch_a)
        graph.add_node("branch_b", branch_b)

        graph.set_entry_point("start")
        graph.add_parallel_edges("start", ["branch_a", "branch_b"])

        compiled = graph.compile()
        result = await compiled.invoke({})

        assert result.success is True
        assert result.state["started"] is True
        assert result.state["branch_a"] is True
        assert result.state["branch_b"] is True

    @pytest.mark.asyncio
    async def test_graph_three_parallel_branches(self):
        """Test graph with three parallel branches."""

        async def start(state):
            return {"count": 0}

        async def add_one(state):
            state = dict(state)
            state["count"] = state.get("count", 0) + 1
            state["branch1"] = True
            return state

        async def add_two(state):
            state = dict(state)
            state["count"] = state.get("count", 0) + 2
            state["branch2"] = True
            return state

        async def add_three(state):
            state = dict(state)
            state["count"] = state.get("count", 0) + 3
            state["branch3"] = True
            return state

        async def summarize(state):
            state["summarized"] = True
            return state

        def sum_counts(states: List[Dict]) -> Dict:
            total = sum(s.get("count", 0) for s in states)
            merged = {}
            for s in states:
                merged.update(s)
            merged["count"] = total
            return merged

        graph = StateGraph()
        graph.add_node("start", start)
        graph.add_node("add_one", add_one)
        graph.add_node("add_two", add_two)
        graph.add_node("add_three", add_three)
        graph.add_node("summarize", summarize)

        graph.set_entry_point("start")
        graph.add_parallel_edges(
            "start",
            ["add_one", "add_two", "add_three"],
            join_node="summarize",
            merge_func=sum_counts,
        )
        graph.add_edge("summarize", END)

        compiled = graph.compile()
        result = await compiled.invoke({})

        assert result.success is True
        assert result.state["count"] == 6  # 1 + 2 + 3
        assert result.state["branch1"] is True
        assert result.state["branch2"] is True
        assert result.state["branch3"] is True
        assert result.state["summarized"] is True

    @pytest.mark.asyncio
    async def test_parallel_branch_with_multi_node_chains(self):
        """Test parallel branches where each branch has multiple nodes."""

        async def start(state):
            return {"started": True}

        async def a1(state):
            state = dict(state)
            state["a1"] = True
            return state

        async def a2(state):
            state = dict(state)
            state["a2"] = True
            return state

        async def b1(state):
            state = dict(state)
            state["b1"] = True
            return state

        async def b2(state):
            state = dict(state)
            state["b2"] = True
            return state

        async def join(state):
            state["joined"] = True
            return state

        graph = StateGraph()
        graph.add_node("start", start)
        graph.add_node("a1", a1)
        graph.add_node("a2", a2)
        graph.add_node("b1", b1)
        graph.add_node("b2", b2)
        graph.add_node("join", join)

        graph.set_entry_point("start")
        # Fork into two branches
        graph.add_parallel_edges("start", ["a1", "b1"], join_node="join")
        # Each branch has a chain
        graph.add_edge("a1", "a2")
        graph.add_edge("a2", "join")  # Implicit stop at join
        graph.add_edge("b1", "b2")
        graph.add_edge("b2", "join")  # Implicit stop at join
        graph.add_edge("join", END)

        compiled = graph.compile()
        result = await compiled.invoke({})

        assert result.success is True
        assert result.state["started"] is True
        assert result.state["a1"] is True
        assert result.state["a2"] is True
        assert result.state["b1"] is True
        assert result.state["b2"] is True
        assert result.state["joined"] is True
