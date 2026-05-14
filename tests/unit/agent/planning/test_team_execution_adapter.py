from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter
from victor.teams.types import MemberResult, TeamFormation, TeamResult


class RecordingCoordinator:
    def __init__(self):
        self.members = []
        self.manager = None
        self.formation = None
        self.context = None

    def add_member(self, member):
        self.members.append(member)
        return self

    def set_manager(self, manager):
        self.manager = manager
        if manager not in self.members:
            self.members.insert(0, manager)
        return self

    def set_formation(self, formation):
        self.formation = formation
        return self

    async def execute_task(self, task, context):
        self.context = context
        return TeamResult(
            success=True,
            final_output="team complete",
            formation=self.formation,
            total_tool_calls=3,
            member_results={
                member.id: MemberResult(
                    member_id=member.id,
                    success=True,
                    output=f"{member.id} output",
                    metadata={"member_id": member.id},
                    tool_calls_used=1,
                )
                for member in self.members
            },
        )


class FallbackWorkerCoordinator(RecordingCoordinator):
    async def execute_task(self, task, context):
        self.context = context
        return TeamResult(
            success=False,
            final_output="inventory complete",
            formation=self.formation,
            total_tool_calls=1,
            member_results={
                "plan_manager": MemberResult(
                    member_id="plan_manager",
                    success=False,
                    output="",
                    error="Unknown or disabled tool: shell",
                ),
                "step_2_researcher": MemberResult(
                    member_id="step_2_researcher",
                    success=True,
                    output="inventory complete",
                    tool_calls_used=1,
                ),
            },
        )


def _complex_plan() -> ReadableTaskPlan:
    return ReadableTaskPlan(
        name="Rust Arc Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust Arc and readonly usage by workspace",
        steps=[
            ["1", "research", "Map Rust workspaces", "ls,read"],
            ["2", "review", "Review Arc usage", "grep,read", ["1"]],
        ],
    )


@pytest.mark.asyncio
async def test_adapter_executes_plan_step_with_direct_worker():
    coordinator = RecordingCoordinator()
    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary="Read Cargo.toml and found 2 workspace crates.",
            error=None,
            details={"full_response": "Read Cargo.toml and clients/rust/Cargo.toml."},
            tool_calls_used=2,
            duration_seconds=0.1,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
        coordinator_factory=lambda _orchestrator: coordinator,
    )
    plan = _complex_plan()
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="session_root",
    )

    assert result.success is True
    assert result.output == "Read Cargo.toml and found 2 workspace crates."
    assert coordinator.formation is None
    assert coordinator.manager is None
    spawn_kwargs = subagents.spawn.await_args.kwargs
    assert spawn_kwargs["member_id"] == "step_1_researcher"
    assert spawn_kwargs["parent_session_id"] == "session_root"
    assert spawn_kwargs["child_session_id"] == (
        f"session_root:team_{execution_plan.id}_1:step_1_researcher"
    )
    assert result.metadata["execution_mode"] == "direct_step_worker"
    assert result.artifacts == []


@pytest.mark.asyncio
async def test_adapter_member_executor_spawns_isolated_child_session_with_handoff_metadata():
    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary="bounded child handoff",
            error=None,
            details={
                "agent_id": "team_plan_1_step_1_step_1_researcher",
                "parent_handoff": {"summary": "bounded child handoff"},
            },
            tool_calls_used=2,
            duration_seconds=0.2,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    plan = _complex_plan()
    execution_plan = plan.to_execution_plan()
    team_id = "team_plan_1_step_1"
    members = adapter._build_members(execution_plan, team_id)

    payload = await members["step_1_researcher"].execute_task(
        "Map Rust workspaces",
        {
            "root_session_id": "session_root",
            "parent_session_id": "session_root",
            "plan_step_id": "1",
        },
    )

    assert payload["success"] is True
    assert payload["metadata"]["parent_handoff"]["summary"] == "bounded child handoff"
    spawn_kwargs = subagents.spawn.await_args.kwargs
    assert spawn_kwargs["member_id"] == "step_1_researcher"
    assert spawn_kwargs["agent_id"] == "team_plan_1_step_1_step_1_researcher"
    assert spawn_kwargs["team_id"] == team_id
    assert spawn_kwargs["parent_session_id"] == "session_root"
    assert spawn_kwargs["child_session_id"] == "session_root:team_plan_1_step_1:step_1_researcher"
    assert spawn_kwargs["plan_id"] == execution_plan.id
    assert spawn_kwargs["plan_step_id"] == "1"
    assert spawn_kwargs["allowed_tools"] == ["read", "ls", "grep"]


@pytest.mark.asyncio
async def test_adapter_preserves_shell_from_plan_step_tool_hints():
    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary="inventory complete",
            error=None,
            details={},
            tool_calls_used=1,
            duration_seconds=0.1,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Rust Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust source files",
        steps=[["3", "analyze", "Inventory all Rust source files", "grep,shell"]],
    )
    execution_plan = plan.to_execution_plan()
    members = adapter._build_members(execution_plan, "team_plan_step")

    await members["step_3_researcher"].execute_task(
        "Inventory all Rust source files",
        {"root_session_id": "session_root", "parent_session_id": "session_root"},
    )

    spawn_kwargs = subagents.spawn.await_args.kwargs
    assert spawn_kwargs["role"].value == "researcher"
    assert spawn_kwargs["allowed_tools"] == ["read", "ls", "grep", "shell"]


@pytest.mark.asyncio
async def test_adapter_executes_compute_node_step_without_provider():
    """Explicit exec='compute' steps run deterministically — no subagent, no model call."""
    from victor.agent.planning.base import StepResult

    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock()

    # Register a language-specific compute node (simulates what victor-coding does at
    # plugin init — the framework itself ships with no language-specific content).
    PlanningTeamExecutionAdapter.register_compute_node(
        "rust_best_practices_checklist",
        lambda step: StepResult(
            success=True,
            output="Rust best-practices checklist:\n1. Ownership ...\n2. Arc ...",
            tool_calls_used=0,
            metadata={
                "execution_mode": "compute_node",
                "compute_node": "rust_best_practices_checklist",
                "node_type": "deterministic_planning_step",
            },
        ),
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=parent,
        sub_agent_orchestrator=subagents,
    )
    # Use the rich dict format with explicit exec + node so dispatch is unambiguous.
    plan = ReadableTaskPlan(
        name="Rust Checklist",
        complexity=TaskComplexity.COMPLEX,
        desc="Create Rust review checklist",
        steps=[
            {
                "id": "4",
                "type": "doc",
                "desc": "Create a master Rust best practices checklist",
                "tools": [],
                "deps": [],
                "exec": "compute",
                "node": "rust_best_practices_checklist",
                "exit": ["checklist covers Arc and immutability"],
            }
        ],
    )
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="session_root",
    )

    assert result.success is True
    assert "Rust best-practices checklist" in result.output
    assert "Arc" in result.output
    assert result.metadata["execution_mode"] == "compute_node"
    assert result.metadata["compute_node"] == "rust_best_practices_checklist"
    assert result.metadata["node_type"] == "deterministic_planning_step"
    subagents.spawn.assert_not_awaited()

    # Clean up registry so other tests are not affected.
    PlanningTeamExecutionAdapter._COMPUTE_NODES.pop("rust_best_practices_checklist", None)


def test_adapter_gives_step_tools_to_hierarchical_manager():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Rust Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust source files",
        steps=[["2", "analyze", "Enumerate all Rust source files", "shell,read"]],
    )
    execution_plan = plan.to_execution_plan()

    members = adapter._build_members(
        execution_plan,
        "team_plan_step",
        current_step=execution_plan.steps[0],
    )

    assert members["plan_manager"].member.allowed_tools == ["read", "ls", "grep", "shell"]


def test_adapter_treats_successful_fallback_worker_as_step_success():
    result = TeamResult(
        success=False,
        final_output="worker output",
        formation=TeamFormation.HIERARCHICAL,
        total_tool_calls=4,
        member_results={
            "plan_manager": MemberResult(
                member_id="plan_manager",
                success=False,
                output="",
                error="Unknown or disabled tool: shell",
                tool_calls_used=0,
            ),
            "step_2_researcher": MemberResult(
                member_id="step_2_researcher",
                success=True,
                output="inventory complete",
                tool_calls_used=1,
            ),
        },
    )

    step_result = PlanningTeamExecutionAdapter._team_result_to_step_result(result)

    assert step_result.success is True
    assert step_result.output == "inventory complete"
    assert step_result.error is None
    assert step_result.tool_calls_used == 4


def test_adapter_treats_dict_shaped_successful_fallback_worker_as_step_success():
    result = {
        "success": False,
        "final_output": "worker output",
        "error": "manager failed",
        "total_tool_calls": 1,
        "member_results": {
            "plan_manager": {"success": False, "output": "", "error": "shell unavailable"},
            "step_2_researcher": {"success": True, "output": "inventory complete"},
        },
    }

    step_result = PlanningTeamExecutionAdapter._team_result_to_step_result(result)

    assert step_result.success is True
    assert step_result.output == "inventory complete"
    assert step_result.error is None


@pytest.mark.asyncio
async def test_adapter_execute_step_does_not_start_hierarchical_manager_for_single_step():
    coordinator = FallbackWorkerCoordinator()
    subagents = MagicMock()
    subagents.spawn = AsyncMock(
        return_value=SimpleNamespace(
            success=True,
            summary="Enumerated 192 Rust files including src/lib.rs.",
            error=None,
            details={},
            tool_calls_used=1,
            duration_seconds=0.1,
        )
    )
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="session_root"),
        sub_agent_orchestrator=subagents,
        coordinator_factory=lambda _orchestrator: coordinator,
    )
    plan = ReadableTaskPlan(
        name="Rust Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust source files",
        steps=[["2", "analyze", "Enumerate all Rust source files", "shell,read"]],
    )
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="session_root",
    )

    assert result.success is True
    assert result.output == "Enumerated 192 Rust files including src/lib.rs."
    assert coordinator.manager is None
    spawn_kwargs = subagents.spawn.await_args.kwargs
    assert spawn_kwargs["allowed_tools"] == ["read", "ls", "grep", "shell"]


def test_adapter_only_uses_team_for_complex_exploratory_plans():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    complex_plan = _complex_plan()
    simple_plan = ReadableTaskPlan(
        name="Small Fix",
        complexity=TaskComplexity.SIMPLE,
        desc="Fix one typo",
        steps=[["1", "feature", "Fix typo", "edit"]],
    )

    assert adapter.should_use_team(complex_plan) is True
    assert adapter.should_use_team(simple_plan) is False
