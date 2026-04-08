"""Fixtures for integration/verticals tests."""

from dataclasses import dataclass

import pytest

# =====================================================================
# StateGraph / Workflow fixtures for test_competitive_features.py
# =====================================================================


@pytest.fixture
def empty_workflow_graph():
    """Create an empty WorkflowGraph for testing."""
    from victor.workflows.graph_dsl import WorkflowGraph, State

    @dataclass
    class TestState(State):
        value: str = ""

    return WorkflowGraph(TestState, name="test_workflow")


@pytest.fixture
def branching_workflow_graph():
    """Create a WorkflowGraph with conditional branching."""
    from victor.workflows.graph_dsl import WorkflowGraph, State

    @dataclass
    class BranchState(State):
        value: str = ""
        branch: str = "a"

    graph = WorkflowGraph(BranchState, name="branching_workflow")
    graph.add_node("start", lambda s: s)
    graph.add_node("branch_a", lambda s: s)
    graph.add_node("branch_b", lambda s: s)
    graph.add_node("merge", lambda s: s)

    graph.add_conditional_edges(
        "start",
        lambda s: "branch_a" if s.branch == "a" else "branch_b",
        {"branch_a": "branch_a", "branch_b": "branch_b"},
    )
    graph.add_edge("branch_a", "merge")
    graph.add_edge("branch_b", "merge")
    graph.set_entry_point("start")
    graph.set_finish_point("merge")

    return graph


@pytest.fixture
def linear_workflow_graph():
    """Create a linear 3-node WorkflowGraph."""
    from victor.workflows.graph_dsl import WorkflowGraph, State

    @dataclass
    class LinearState(State):
        value: str = ""

    graph = WorkflowGraph(LinearState, name="linear_workflow")
    graph.add_node("step_1", lambda s: s)
    graph.add_node("step_2", lambda s: s)
    graph.add_node("step_3", lambda s: s)
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "step_3")
    graph.set_entry_point("step_1")
    graph.set_finish_point("step_3")

    return graph


# =====================================================================
# Team / Agent fixtures for test_competitive_features.py
# =====================================================================


@pytest.fixture
def team_member_specs():
    """Create a list of TeamMemberSpec for team formation tests."""
    from victor.framework.teams import TeamMemberSpec

    return [
        TeamMemberSpec(
            role="researcher",
            goal="Research information",
            name="Researcher",
            tool_budget=15,
        ),
        TeamMemberSpec(
            role="writer",
            goal="Write content",
            name="Writer",
            tool_budget=10,
        ),
        TeamMemberSpec(
            role="reviewer",
            goal="Review work",
            name="Reviewer",
            tool_budget=5,
        ),
    ]


@pytest.fixture
def mock_team_member():
    """Create a mock team member with executor role."""
    from victor.teams import TeamMember
    from victor.agent.subagents.base import SubAgentRole

    return TeamMember(
        id="test_member_1",
        role=SubAgentRole.EXECUTOR,
        name="Test Executor",
        goal="Execute tasks",
        tool_budget=15,
        is_manager=False,
    )


# =====================================================================
# HITL fixtures for test_competitive_features.py
# =====================================================================


@pytest.fixture
def hitl_approval_node():
    """Create an HITL approval node."""
    from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

    return HITLNode(
        id="approval_test",
        name="Approval Gate",
        hitl_type=HITLNodeType.APPROVAL,
        prompt="Approve this action?",
        timeout=30.0,
        fallback=HITLFallback.ABORT,
    )


@pytest.fixture
def auto_approve_handler():
    """Create an auto-approve HITL handler."""
    from victor.workflows.hitl import HITLResponse, HITLStatus

    async def handler(request):
        return HITLResponse(
            request_id=request.request_id,
            status=HITLStatus.APPROVED,
            approved=True,
        )

    return handler


@pytest.fixture
def auto_reject_handler():
    """Create an auto-reject HITL handler."""
    from victor.workflows.hitl import HITLResponse, HITLStatus

    async def handler(request):
        return HITLResponse(
            request_id=request.request_id,
            status=HITLStatus.REJECTED,
            approved=False,
            reason="Auto-rejected for testing",
        )

    return handler


# =====================================================================
# ModeConfig fixtures for test_competitive_features.py
# =====================================================================


@pytest.fixture
def mode_config_registry():
    """Create a ModeConfigRegistry with default modes."""
    from victor.core.mode_config import ModeConfigRegistry, ModeDefinition

    registry = ModeConfigRegistry()
    registry.register_vertical(
        "default",
        modes={
            "quick": ModeDefinition(name="quick", tool_budget=5, max_iterations=10),
            "standard": ModeDefinition(
                name="standard", tool_budget=15, max_iterations=30
            ),
        },
    )
    return registry


@pytest.fixture
def registered_mode_registry(mode_config_registry):
    """Create a ModeConfigRegistry with vertical-specific overrides."""
    from victor.core.mode_config import ModeDefinition

    registry = mode_config_registry

    # Register vertical with custom mode and task budgets
    registry.register_vertical(
        "test_vertical",
        modes={
            "custom": ModeDefinition(name="custom", tool_budget=25, max_iterations=50),
        },
        task_budgets={
            "test_task": 15,
            "complex_task": 30,
        },
        default_budget=12,
    )
    return registry
