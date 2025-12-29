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

"""Tests for StateGraph-based coding workflows.

These tests verify the LangGraph-compatible workflows for coding tasks
including TDD cycles, bug fix loops, and feature implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.verticals.coding.workflows.graph_workflows import (
    # State types
    CodingState,
    TestState,
    BugFixState,
    # Node functions
    research_node,
    plan_node,
    implement_node,
    execute_tests_node,
    review_node,
    finalize_node,
    write_test_node,
    implement_feature_node,
    run_tests_node,
    refactor_node,
    investigate_node,
    apply_fix_node,
    verify_fix_node,
    # Condition functions
    should_retry_implementation,
    should_continue_tdd,
    check_fix_verified,
    # Workflow factories
    create_feature_workflow,
    create_tdd_workflow,
    create_bugfix_workflow,
    create_code_review_workflow,
    # Executor
    GraphWorkflowExecutor,
)
from victor.framework.graph import (
    StateGraph,
    CompiledGraph,
    END,
    MemoryCheckpointer,
    GraphConfig,
)


# =============================================================================
# State Type Tests
# =============================================================================


class TestCodingState:
    """Tests for CodingState TypedDict."""

    def test_coding_state_defaults(self):
        """CodingState should work with minimal fields."""
        state: CodingState = {
            "task": "Add feature",
            "messages": [],
        }

        assert state["task"] == "Add feature"
        assert state["messages"] == []

    def test_coding_state_all_fields(self):
        """CodingState should accept all fields."""
        state: CodingState = {
            "task": "Add feature",
            "messages": ["Started"],
            "research_findings": {"patterns_found": []},
            "implementation_plan": "Plan here",
            "code_changes": ["file1.py"],
            "test_results": {"passed": True},
            "review_feedback": None,
            "iteration_count": 1,
            "max_iterations": 3,
            "error": None,
            "success": True,
        }

        assert state["task"] == "Add feature"
        assert state["success"] is True


class TestTestState:
    """Tests for TestState TypedDict."""

    def test_test_state_defaults(self):
        """TestState should work with minimal fields."""
        state: TestState = {
            "feature_description": "Add auth",
        }

        assert state["feature_description"] == "Add auth"

    def test_test_state_all_fields(self):
        """TestState should accept all fields."""
        state: TestState = {
            "feature_description": "Add auth",
            "test_code": "def test_auth(): pass",
            "implementation_code": "def auth(): pass",
            "test_passed": True,
            "test_output": "OK",
            "iteration": 2,
            "max_tdd_cycles": 5,
        }

        assert state["test_passed"] is True
        assert state["iteration"] == 2


class TestBugFixState:
    """Tests for BugFixState TypedDict."""

    def test_bugfix_state_defaults(self):
        """BugFixState should work with minimal fields."""
        state: BugFixState = {
            "bug_description": "Login fails",
        }

        assert state["bug_description"] == "Login fails"

    def test_bugfix_state_all_fields(self):
        """BugFixState should accept all fields."""
        state: BugFixState = {
            "bug_description": "Login fails",
            "stack_trace": "Error at line 42",
            "root_cause": "Missing null check",
            "fix_applied": "Added null check",
            "regression_test": "def test_login(): pass",
            "verified": True,
            "attempts": 2,
            "max_attempts": 3,
        }

        assert state["verified"] is True
        assert state["attempts"] == 2


# =============================================================================
# Node Function Tests
# =============================================================================


class TestCodingNodeFunctions:
    """Tests for coding workflow node functions."""

    @pytest.mark.asyncio
    async def test_research_node(self):
        """research_node should populate research_findings."""
        state: CodingState = {
            "task": "Add feature",
            "messages": [],
        }

        result = await research_node(state)

        assert "research_findings" in result
        assert "Research phase completed" in result["messages"]

    @pytest.mark.asyncio
    async def test_plan_node(self):
        """plan_node should create implementation_plan."""
        state: CodingState = {
            "task": "Add feature",
            "messages": [],
            "research_findings": {"patterns_found": ["pattern1"]},
        }

        result = await plan_node(state)

        assert result["implementation_plan"] is not None
        assert "Planning phase completed" in result["messages"]

    @pytest.mark.asyncio
    async def test_implement_node_increments_iteration(self):
        """implement_node should increment iteration_count."""
        state: CodingState = {
            "task": "Add feature",
            "messages": [],
            "iteration_count": 0,
        }

        result = await implement_node(state)

        assert result["iteration_count"] == 1
        assert "iteration 1" in result["messages"][-1]

    @pytest.mark.asyncio
    async def test_implement_node_initializes_iteration(self):
        """implement_node should initialize iteration_count if missing."""
        state: CodingState = {
            "task": "Add feature",
            "messages": [],
        }

        result = await implement_node(state)

        assert result["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_execute_tests_node(self):
        """execute_tests_node should populate test_results."""
        state: CodingState = {
            "task": "Add feature",
            "messages": [],
        }

        result = await execute_tests_node(state)

        assert "test_results" in result
        assert "Tests executed" in result["messages"]

    @pytest.mark.asyncio
    async def test_review_node(self):
        """review_node should set review_feedback."""
        state: CodingState = {
            "task": "Add feature",
            "messages": [],
        }

        result = await review_node(state)

        assert "review_feedback" in result
        assert "Review completed" in result["messages"]

    @pytest.mark.asyncio
    async def test_finalize_node(self):
        """finalize_node should set success flag."""
        state: CodingState = {
            "task": "Add feature",
            "messages": [],
        }

        result = await finalize_node(state)

        assert result["success"] is True
        assert "Changes finalized" in result["messages"]


class TestTDDNodeFunctions:
    """Tests for TDD workflow node functions."""

    @pytest.mark.asyncio
    async def test_write_test_node(self):
        """write_test_node should create test code."""
        state: TestState = {
            "feature_description": "Add auth",
        }

        result = await write_test_node(state)

        assert "test_code" in result
        assert result["test_passed"] is False
        assert result["iteration"] == 1

    @pytest.mark.asyncio
    async def test_write_test_node_increments_iteration(self):
        """write_test_node should increment iteration."""
        state: TestState = {
            "feature_description": "Add auth",
            "iteration": 2,
        }

        result = await write_test_node(state)

        assert result["iteration"] == 3

    @pytest.mark.asyncio
    async def test_implement_feature_node(self):
        """implement_feature_node should create implementation code."""
        state: TestState = {
            "feature_description": "Add auth",
        }

        result = await implement_feature_node(state)

        assert "implementation_code" in result

    @pytest.mark.asyncio
    async def test_run_tests_node(self):
        """run_tests_node should set test_passed and output."""
        state: TestState = {
            "feature_description": "Add auth",
        }

        result = await run_tests_node(state)

        assert result["test_passed"] is True
        assert result["test_output"] == "All tests passed"

    @pytest.mark.asyncio
    async def test_refactor_node(self):
        """refactor_node should preserve state."""
        state: TestState = {
            "feature_description": "Add auth",
            "test_passed": True,
        }

        result = await refactor_node(state)

        assert result["test_passed"] is True


class TestBugFixNodeFunctions:
    """Tests for bug fix workflow node functions."""

    @pytest.mark.asyncio
    async def test_investigate_node(self):
        """investigate_node should identify root cause."""
        state: BugFixState = {
            "bug_description": "Login fails",
        }

        result = await investigate_node(state)

        assert result["root_cause"] is not None
        assert result["attempts"] == 1

    @pytest.mark.asyncio
    async def test_investigate_node_increments_attempts(self):
        """investigate_node should increment attempts."""
        state: BugFixState = {
            "bug_description": "Login fails",
            "attempts": 1,
        }

        result = await investigate_node(state)

        assert result["attempts"] == 2

    @pytest.mark.asyncio
    async def test_apply_fix_node(self):
        """apply_fix_node should record fix."""
        state: BugFixState = {
            "bug_description": "Login fails",
            "root_cause": "Null check missing",
        }

        result = await apply_fix_node(state)

        assert result["fix_applied"] is not None

    @pytest.mark.asyncio
    async def test_verify_fix_node(self):
        """verify_fix_node should verify and add test."""
        state: BugFixState = {
            "bug_description": "Login fails",
            "fix_applied": "Added null check",
        }

        result = await verify_fix_node(state)

        assert result["verified"] is True
        assert result["regression_test"] is not None


# =============================================================================
# Condition Function Tests
# =============================================================================


class TestConditionFunctions:
    """Tests for workflow condition functions."""

    def test_should_retry_implementation_done_on_pass(self):
        """should_retry_implementation returns 'done' when tests pass."""
        state: CodingState = {
            "test_results": {"passed": True},
            "iteration_count": 1,
            "max_iterations": 3,
        }

        assert should_retry_implementation(state) == "done"

    def test_should_retry_implementation_retry_on_fail(self):
        """should_retry_implementation returns 'retry' when tests fail."""
        state: CodingState = {
            "test_results": {"passed": False},
            "iteration_count": 1,
            "max_iterations": 3,
        }

        assert should_retry_implementation(state) == "retry"

    def test_should_retry_implementation_done_at_max(self):
        """should_retry_implementation returns 'done' at max iterations."""
        state: CodingState = {
            "test_results": {"passed": False},
            "iteration_count": 3,
            "max_iterations": 3,
        }

        assert should_retry_implementation(state) == "done"

    def test_should_continue_tdd_implement_when_fail(self):
        """should_continue_tdd returns 'implement' when tests fail."""
        state: TestState = {
            "test_passed": False,
            "iteration": 1,
            "max_tdd_cycles": 5,
        }

        assert should_continue_tdd(state) == "implement"

    def test_should_continue_tdd_refactor_when_pass(self):
        """should_continue_tdd returns 'refactor' when tests pass."""
        state: TestState = {
            "test_passed": True,
            "iteration": 1,
            "max_tdd_cycles": 5,
        }

        assert should_continue_tdd(state) == "refactor"

    def test_should_continue_tdd_finish_at_max(self):
        """should_continue_tdd returns 'finish' at max cycles."""
        state: TestState = {
            "test_passed": False,
            "iteration": 5,
            "max_tdd_cycles": 5,
        }

        assert should_continue_tdd(state) == "finish"

    def test_check_fix_verified_done_when_verified(self):
        """check_fix_verified returns 'done' when verified."""
        state: BugFixState = {
            "verified": True,
            "attempts": 1,
            "max_attempts": 3,
        }

        assert check_fix_verified(state) == "done"

    def test_check_fix_verified_retry_when_not_verified(self):
        """check_fix_verified returns 'retry' when not verified."""
        state: BugFixState = {
            "verified": False,
            "attempts": 1,
            "max_attempts": 3,
        }

        assert check_fix_verified(state) == "retry"

    def test_check_fix_verified_done_at_max_attempts(self):
        """check_fix_verified returns 'done' at max attempts."""
        state: BugFixState = {
            "verified": False,
            "attempts": 3,
            "max_attempts": 3,
        }

        assert check_fix_verified(state) == "done"


# =============================================================================
# Workflow Factory Tests
# =============================================================================


class TestWorkflowFactories:
    """Tests for workflow factory functions."""

    def test_create_feature_workflow_returns_graph(self):
        """create_feature_workflow should return StateGraph."""
        graph = create_feature_workflow()

        assert isinstance(graph, StateGraph)

    def test_create_feature_workflow_has_nodes(self):
        """create_feature_workflow should have required nodes."""
        graph = create_feature_workflow()

        assert "research" in graph._nodes
        assert "plan" in graph._nodes
        assert "implement" in graph._nodes
        assert "test" in graph._nodes
        assert "review" in graph._nodes
        assert "finalize" in graph._nodes

    def test_create_feature_workflow_has_entry_point(self):
        """create_feature_workflow should have entry point."""
        graph = create_feature_workflow()

        assert graph._entry_point == "research"

    def test_create_feature_workflow_compiles(self):
        """create_feature_workflow should compile successfully."""
        graph = create_feature_workflow()

        compiled = graph.compile()

        assert isinstance(compiled, CompiledGraph)

    def test_create_tdd_workflow_returns_graph(self):
        """create_tdd_workflow should return StateGraph."""
        graph = create_tdd_workflow()

        assert isinstance(graph, StateGraph)

    def test_create_tdd_workflow_has_nodes(self):
        """create_tdd_workflow should have TDD nodes."""
        graph = create_tdd_workflow()

        assert "write_test" in graph._nodes
        assert "implement" in graph._nodes
        assert "run_tests" in graph._nodes
        assert "refactor" in graph._nodes

    def test_create_tdd_workflow_entry_point(self):
        """create_tdd_workflow should start with write_test."""
        graph = create_tdd_workflow()

        assert graph._entry_point == "write_test"

    def test_create_bugfix_workflow_returns_graph(self):
        """create_bugfix_workflow should return StateGraph."""
        graph = create_bugfix_workflow()

        assert isinstance(graph, StateGraph)

    def test_create_bugfix_workflow_has_nodes(self):
        """create_bugfix_workflow should have bug fix nodes."""
        graph = create_bugfix_workflow()

        assert "investigate" in graph._nodes
        assert "apply_fix" in graph._nodes
        assert "verify" in graph._nodes

    def test_create_bugfix_workflow_entry_point(self):
        """create_bugfix_workflow should start with investigate."""
        graph = create_bugfix_workflow()

        assert graph._entry_point == "investigate"

    def test_create_code_review_workflow_returns_graph(self):
        """create_code_review_workflow should return StateGraph."""
        graph = create_code_review_workflow()

        assert isinstance(graph, StateGraph)

    def test_create_code_review_workflow_has_nodes(self):
        """create_code_review_workflow should have review nodes."""
        graph = create_code_review_workflow()

        assert "review" in graph._nodes
        assert "revise" in graph._nodes
        assert "finalize" in graph._nodes


# =============================================================================
# Workflow Execution Tests
# =============================================================================


class TestWorkflowExecution:
    """Tests for workflow execution."""

    @pytest.mark.asyncio
    async def test_feature_workflow_executes(self):
        """Feature workflow should execute successfully."""
        graph = create_feature_workflow()
        compiled = graph.compile()

        state: CodingState = {
            "task": "Add feature",
            "messages": [],
            "max_iterations": 3,
        }

        result = await compiled.invoke(state)

        assert result.success is True
        assert result.state["success"] is True

    @pytest.mark.asyncio
    async def test_tdd_workflow_executes(self):
        """TDD workflow should execute with cycles."""
        graph = create_tdd_workflow()
        compiled = graph.compile(max_iterations=20)

        state: TestState = {
            "feature_description": "Add auth",
            "iteration": 0,
            "max_tdd_cycles": 3,
        }

        result = await compiled.invoke(state)

        # Should complete within max_tdd_cycles
        assert result.state["iteration"] <= 3

    @pytest.mark.asyncio
    async def test_bugfix_workflow_executes(self):
        """Bug fix workflow should execute."""
        graph = create_bugfix_workflow()
        compiled = graph.compile()

        state: BugFixState = {
            "bug_description": "Login fails",
            "attempts": 0,
            "max_attempts": 3,
        }

        result = await compiled.invoke(state)

        assert result.success is True
        assert result.state["verified"] is True

    @pytest.mark.asyncio
    async def test_feature_workflow_with_checkpointing(self):
        """Feature workflow should work with checkpointing."""
        checkpointer = MemoryCheckpointer()
        graph = create_feature_workflow()
        compiled = graph.compile(checkpointer=checkpointer)

        state: CodingState = {
            "task": "Add feature",
            "messages": [],
            "max_iterations": 3,
        }

        await compiled.invoke(state, thread_id="test_thread")

        checkpoints = await checkpointer.list("test_thread")
        assert len(checkpoints) > 0


# =============================================================================
# GraphWorkflowExecutor Tests
# =============================================================================


class TestGraphWorkflowExecutor:
    """Tests for GraphWorkflowExecutor."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        return MagicMock()

    def test_executor_creation(self, mock_orchestrator):
        """Executor should store orchestrator."""
        executor = GraphWorkflowExecutor(mock_orchestrator)

        assert executor._orchestrator == mock_orchestrator
        assert executor._checkpointer is None

    def test_executor_with_checkpointer(self, mock_orchestrator):
        """Executor should store checkpointer."""
        checkpointer = MemoryCheckpointer()
        executor = GraphWorkflowExecutor(mock_orchestrator, checkpointer=checkpointer)

        assert executor._checkpointer == checkpointer

    @pytest.mark.asyncio
    async def test_executor_run(self, mock_orchestrator):
        """Executor run should execute graph."""
        executor = GraphWorkflowExecutor(mock_orchestrator)
        graph = create_feature_workflow()

        state: CodingState = {
            "task": "Add feature",
            "messages": [],
            "max_iterations": 3,
        }

        result = await executor.run(graph, state)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_executor_run_with_thread_id(self, mock_orchestrator):
        """Executor run should use thread_id."""
        checkpointer = MemoryCheckpointer()
        executor = GraphWorkflowExecutor(mock_orchestrator, checkpointer=checkpointer)
        graph = create_feature_workflow()

        state: CodingState = {
            "task": "Add feature",
            "messages": [],
            "max_iterations": 3,
        }

        await executor.run(graph, state, thread_id="my_thread")

        checkpoints = await checkpointer.list("my_thread")
        assert len(checkpoints) > 0

    @pytest.mark.asyncio
    async def test_executor_stream(self, mock_orchestrator):
        """Executor stream should yield states."""
        executor = GraphWorkflowExecutor(mock_orchestrator)
        graph = create_feature_workflow()

        state: CodingState = {
            "task": "Add feature",
            "messages": [],
            "max_iterations": 3,
        }

        node_ids = []
        async for node_id, node_state in executor.stream(graph, state):
            node_ids.append(node_id)

        assert "research" in node_ids
        assert "plan" in node_ids
        assert "implement" in node_ids


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_workflows_exported_from_package(self):
        """Graph workflows should be exported from workflows package."""
        from victor.verticals.coding.workflows import (
            CodingState,
            TestState,
            BugFixState,
            create_feature_workflow,
            create_tdd_workflow,
            create_bugfix_workflow,
            create_code_review_workflow,
            GraphWorkflowExecutor,
        )

        assert CodingState is not None
        assert create_feature_workflow is not None
        assert GraphWorkflowExecutor is not None

    def test_all_state_types_exported(self):
        """All state types should be exported."""
        from victor.verticals.coding.workflows import (
            CodingState,
            TestState,
            BugFixState,
        )

        # Verify they are TypedDicts
        assert hasattr(CodingState, "__annotations__")
        assert hasattr(TestState, "__annotations__")
        assert hasattr(BugFixState, "__annotations__")

    def test_all_workflow_factories_exported(self):
        """All workflow factories should be exported."""
        from victor.verticals.coding.workflows import (
            create_feature_workflow,
            create_tdd_workflow,
            create_bugfix_workflow,
            create_code_review_workflow,
        )

        # Verify they are callable
        assert callable(create_feature_workflow)
        assert callable(create_tdd_workflow)
        assert callable(create_bugfix_workflow)
        assert callable(create_code_review_workflow)
