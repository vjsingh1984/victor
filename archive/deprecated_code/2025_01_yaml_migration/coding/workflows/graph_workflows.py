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

"""StateGraph-based workflows for coding vertical.

Provides LangGraph-compatible StateGraph workflows for complex coding tasks
that benefit from:
- Typed state management
- Cyclic execution (test-fix loops, validation cycles)
- Explicit retry limits
- Checkpoint/resume semantics
- Human-in-the-loop interrupts

These workflows complement the WorkflowBuilder DSL, offering more control
for complex multi-iteration tasks.

Example:
    from victor.coding.workflows.graph_workflows import (
        create_tdd_workflow,
        CodingState,
    )
    from victor.framework.graph import RLCheckpointerAdapter

    # Create workflow with checkpointing
    graph = create_tdd_workflow()
    checkpointer = RLCheckpointerAdapter("tdd_workflow")
    app = graph.compile(checkpointer=checkpointer)

    # Execute with typed state
    result = await app.invoke(CodingState(
        task="Add user authentication",
        messages=[],
    ))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, TypedDict

from victor.framework.graph import (
    END,
    StateGraph,
    GraphConfig,
)

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Typed State Definitions
# =============================================================================


class CodingState(TypedDict, total=False):
    """Typed state for coding workflows.

    Attributes:
        task: The task description
        messages: Conversation history
        research_findings: Code analysis results
        implementation_plan: Planned approach
        code_changes: Files modified
        test_results: Test execution output
        review_feedback: Review comments
        iteration_count: Current iteration
        max_iterations: Maximum allowed iterations
        error: Error message if failed
        success: Whether task succeeded
    """

    task: str
    messages: List[str]
    research_findings: Optional[Dict[str, Any]]
    implementation_plan: Optional[str]
    code_changes: Optional[List[str]]
    test_results: Optional[Dict[str, Any]]
    review_feedback: Optional[str]
    iteration_count: int
    max_iterations: int
    error: Optional[str]
    success: bool


class TestState(TypedDict, total=False):
    """Typed state for TDD workflows.

    Attributes:
        feature_description: What to implement
        test_code: Written test code
        implementation_code: Implementation code
        test_passed: Whether tests pass
        test_output: Test execution output
        iteration: Current TDD iteration
        max_tdd_cycles: Maximum red-green-refactor cycles
    """

    feature_description: str
    test_code: Optional[str]
    implementation_code: Optional[str]
    test_passed: bool
    test_output: Optional[str]
    iteration: int
    max_tdd_cycles: int


class BugFixState(TypedDict, total=False):
    """Typed state for bug fix workflows.

    Attributes:
        bug_description: Bug report or symptoms
        stack_trace: Error stack trace if available
        root_cause: Identified root cause
        fix_applied: Description of fix
        regression_test: Test to prevent recurrence
        verified: Whether fix is verified
        attempts: Number of fix attempts
        max_attempts: Maximum fix attempts allowed
    """

    bug_description: str
    stack_trace: Optional[str]
    root_cause: Optional[str]
    fix_applied: Optional[str]
    regression_test: Optional[str]
    verified: bool
    attempts: int
    max_attempts: int


# =============================================================================
# Node Functions
# =============================================================================


async def research_node(state: CodingState) -> CodingState:
    """Research codebase for relevant patterns.

    This node analyzes the codebase to find:
    - Related code patterns
    - Dependencies
    - Existing implementations to follow
    """
    # In actual implementation, this would use the orchestrator
    # Here we define the structure for type safety
    state["research_findings"] = {
        "patterns_found": [],
        "dependencies": [],
        "related_files": [],
    }
    state["messages"].append("Research phase completed")
    return state


async def plan_node(state: CodingState) -> CodingState:
    """Create implementation plan based on research."""
    findings = state.get("research_findings", {})
    state["implementation_plan"] = (
        f"Plan based on {len(findings.get('patterns_found', []))} patterns"
    )
    state["messages"].append("Planning phase completed")
    return state


async def implement_node(state: CodingState) -> CodingState:
    """Implement the feature according to plan."""
    state["code_changes"] = []
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["messages"].append(f"Implementation iteration {state['iteration_count']}")
    return state


async def execute_tests_node(state: CodingState) -> CodingState:
    """Run tests and capture results."""
    state["test_results"] = {
        "passed": True,  # Would be actual test results
        "failures": [],
        "coverage": 0.0,
    }
    state["messages"].append("Tests executed")
    return state


async def review_node(state: CodingState) -> CodingState:
    """Review implementation for issues."""
    state["review_feedback"] = None  # No issues found
    state["messages"].append("Review completed")
    return state


async def finalize_node(state: CodingState) -> CodingState:
    """Finalize and commit changes."""
    state["success"] = True
    state["messages"].append("Changes finalized")
    return state


# TDD Node Functions


async def write_test_node(state: TestState) -> TestState:
    """Write failing test for feature (Red phase)."""
    state["test_code"] = f"# Test for: {state.get('feature_description', 'unknown')}"
    state["test_passed"] = False
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def implement_feature_node(state: TestState) -> TestState:
    """Implement just enough to pass test (Green phase)."""
    state["implementation_code"] = "# Implementation"
    return state


async def run_tests_node(state: TestState) -> TestState:
    """Run tests and check if they pass."""
    # Would actually run tests
    state["test_passed"] = True
    state["test_output"] = "All tests passed"
    return state


async def refactor_node(state: TestState) -> TestState:
    """Refactor while keeping tests green."""
    # Would perform refactoring
    return state


# Bug Fix Node Functions


async def investigate_node(state: BugFixState) -> BugFixState:
    """Investigate bug to find root cause."""
    state["attempts"] = state.get("attempts", 0) + 1
    state["root_cause"] = "Identified root cause"
    return state


async def apply_fix_node(state: BugFixState) -> BugFixState:
    """Apply fix for the identified root cause."""
    state["fix_applied"] = "Fix applied"
    return state


async def verify_fix_node(state: BugFixState) -> BugFixState:
    """Verify fix works and add regression test."""
    state["verified"] = True
    state["regression_test"] = "def test_regression(): pass"
    return state


# =============================================================================
# Condition Functions
# =============================================================================


def should_retry_implementation(state: CodingState) -> str:
    """Determine if implementation should be retried.

    Returns:
        'retry' if tests failed and under limit, 'done' otherwise
    """
    test_results = state.get("test_results", {})
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if test_results.get("passed", False):
        return "done"
    if iteration >= max_iter:
        return "done"  # Give up after max iterations
    return "retry"


def should_continue_tdd(state: TestState) -> str:
    """Determine if TDD cycle should continue.

    Returns:
        'continue' if more cycles needed, 'finish' if done
    """
    iteration = state.get("iteration", 0)
    max_cycles = state.get("max_tdd_cycles", 5)

    if iteration >= max_cycles:
        return "finish"
    if state.get("test_passed", False):
        return "refactor"
    return "implement"


def check_fix_verified(state: BugFixState) -> str:
    """Check if bug fix is verified.

    Returns:
        'done' if verified, 'retry' if not and under limit
    """
    if state.get("verified", False):
        return "done"
    if state.get("attempts", 0) >= state.get("max_attempts", 3):
        return "done"  # Give up
    return "retry"


# =============================================================================
# Workflow Factories
# =============================================================================


def create_feature_workflow() -> StateGraph[CodingState]:
    """Create a feature implementation workflow with test-fix cycle.

    This workflow implements:
    1. Research -> Plan -> Implement -> Test
    2. If tests fail, retry implementation (up to max_iterations)
    3. Review -> Finalize

    The cyclic test-fix loop allows iterative refinement.

    Returns:
        StateGraph for feature implementation
    """
    graph = StateGraph(CodingState)

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("plan", plan_node)
    graph.add_node("implement", implement_node)
    graph.add_node("test", execute_tests_node)
    graph.add_node("review", review_node)
    graph.add_node("finalize", finalize_node)

    # Add edges (including cycle)
    graph.add_edge("research", "plan")
    graph.add_edge("plan", "implement")
    graph.add_edge("implement", "test")

    # Conditional: retry or proceed to review
    graph.add_conditional_edge(
        "test",
        should_retry_implementation,
        {"retry": "implement", "done": "review"},
    )

    graph.add_edge("review", "finalize")
    graph.add_edge("finalize", END)

    # Set entry point
    graph.set_entry_point("research")

    return graph


def create_tdd_workflow() -> StateGraph[TestState]:
    """Create a Test-Driven Development workflow.

    Implements the Red-Green-Refactor cycle:
    1. Write failing test (Red)
    2. Implement to pass (Green)
    3. Refactor while green
    4. Repeat for next feature

    Returns:
        StateGraph for TDD workflow
    """
    graph = StateGraph(TestState)

    # Add nodes
    graph.add_node("write_test", write_test_node)
    graph.add_node("implement", implement_feature_node)
    graph.add_node("run_tests", run_tests_node)
    graph.add_node("refactor", refactor_node)

    # Red -> Green -> Refactor cycle
    graph.add_edge("write_test", "implement")
    graph.add_edge("implement", "run_tests")

    # Conditional: if tests pass, refactor; if fail, keep implementing
    graph.add_conditional_edge(
        "run_tests",
        should_continue_tdd,
        {
            "implement": "implement",  # Tests still failing
            "refactor": "refactor",  # Tests pass, refactor
            "finish": END,  # Max cycles reached
        },
    )

    # After refactor, write next test (cycle) or finish
    graph.add_edge("refactor", "write_test")

    # Set entry point
    graph.set_entry_point("write_test")

    return graph


def create_bugfix_workflow() -> StateGraph[BugFixState]:
    """Create a bug fix workflow with verification loop.

    Implements:
    1. Investigate -> Apply Fix -> Verify
    2. If not verified, re-investigate (up to max_attempts)
    3. Add regression test on success

    Returns:
        StateGraph for bug fix workflow
    """
    graph = StateGraph(BugFixState)

    # Add nodes
    graph.add_node("investigate", investigate_node)
    graph.add_node("apply_fix", apply_fix_node)
    graph.add_node("verify", verify_fix_node)

    # Linear flow with retry loop
    graph.add_edge("investigate", "apply_fix")
    graph.add_edge("apply_fix", "verify")

    # Conditional: retry investigation or complete
    graph.add_conditional_edge(
        "verify",
        check_fix_verified,
        {"retry": "investigate", "done": END},
    )

    # Set entry point
    graph.set_entry_point("investigate")

    return graph


def create_code_review_workflow() -> StateGraph[CodingState]:
    """Create a code review workflow with revision cycle.

    Implements:
    1. Review code
    2. If issues found, revise implementation
    3. Re-review until approved or max iterations

    Returns:
        StateGraph for code review workflow
    """
    graph = StateGraph(CodingState)

    # Add nodes
    graph.add_node("review", review_node)
    graph.add_node("revise", implement_node)  # Reuse implement for revisions
    graph.add_node("finalize", finalize_node)

    # Review -> Revise cycle
    def check_review(state: CodingState) -> str:
        feedback = state.get("review_feedback")
        iteration = state.get("iteration_count", 0)
        if feedback is None or iteration >= state.get("max_iterations", 3):
            return "approve"
        return "revise"

    graph.add_conditional_edge(
        "review",
        check_review,
        {"revise": "revise", "approve": "finalize"},
    )

    graph.add_edge("revise", "review")  # Back to review after revision
    graph.add_edge("finalize", END)

    # Set entry point
    graph.set_entry_point("review")

    return graph


# =============================================================================
# Orchestrator Integration
# =============================================================================


class GraphWorkflowExecutor:
    """Executor that integrates StateGraph with AgentOrchestrator.

    Bridges the StateGraph execution with the actual agent orchestrator,
    allowing node functions to use the full agent capabilities.

    Example:
        executor = GraphWorkflowExecutor(orchestrator)
        graph = create_feature_workflow()
        result = await executor.run(graph, initial_state)
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        checkpointer: Optional[Any] = None,
    ):
        """Initialize executor.

        Args:
            orchestrator: AgentOrchestrator for agent execution
            checkpointer: Optional checkpointer for persistence
        """
        self._orchestrator = orchestrator
        self._checkpointer = checkpointer

    async def run(
        self,
        graph: StateGraph,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        config: Optional[GraphConfig] = None,
    ):
        """Execute a StateGraph workflow.

        Args:
            graph: StateGraph to execute
            initial_state: Initial state
            thread_id: Optional thread ID for checkpointing
            config: Optional execution config

        Returns:
            ExecutionResult with final state
        """
        # Compile with checkpointer
        compiled = graph.compile(checkpointer=self._checkpointer)

        # Merge config if provided
        exec_config = config or GraphConfig()
        if self._checkpointer:
            exec_config.checkpointer = self._checkpointer

        # Execute
        return await compiled.invoke(
            initial_state,
            config=exec_config,
            thread_id=thread_id,
        )

    async def stream(
        self,
        graph: StateGraph,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
    ):
        """Stream execution yielding state after each node.

        Args:
            graph: StateGraph to execute
            initial_state: Initial state
            thread_id: Optional thread ID

        Yields:
            Tuple of (node_id, state) after each node
        """
        compiled = graph.compile(checkpointer=self._checkpointer)

        async for node_id, state in compiled.stream(
            initial_state,
            thread_id=thread_id,
        ):
            yield node_id, state


__all__ = [
    # State types
    "CodingState",
    "TestState",
    "BugFixState",
    # Workflow factories
    "create_feature_workflow",
    "create_tdd_workflow",
    "create_bugfix_workflow",
    "create_code_review_workflow",
    # Executor
    "GraphWorkflowExecutor",
]
