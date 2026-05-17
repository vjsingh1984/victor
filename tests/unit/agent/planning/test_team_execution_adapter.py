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


@pytest.mark.asyncio
async def test_compute_node_receives_plan_state():
    """Compute nodes registered with fn(step, plan_state) can read prior step outputs."""
    from victor.agent.planning.base import StepResult

    received_plan_state: dict = {}

    def _checklist_with_context(step, plan_state: dict) -> StepResult:
        received_plan_state.update(plan_state)
        members = plan_state.get("workspace_members", [])
        return StepResult(
            success=True,
            output=f"Checklist for {len(members)} crate(s)",
            tool_calls_used=0,
            metadata={"execution_mode": "compute_node", "compute_node": "lang_checklist"},
        )

    PlanningTeamExecutionAdapter.register_compute_node("lang_checklist", _checklist_with_context)
    parent = SimpleNamespace(active_session_id="session_root")
    subagents = MagicMock()
    subagents.spawn = AsyncMock()
    adapter = PlanningTeamExecutionAdapter(orchestrator=parent, sub_agent_orchestrator=subagents)

    plan = ReadableTaskPlan(
        name="Checklist test",
        complexity=TaskComplexity.MODERATE,
        desc="Test plan_state pass-through to compute node",
        steps=[
            {
                "id": "1",
                "type": "analyze",
                "desc": "Checklist",
                "exec": "compute",
                "node": "lang_checklist",
            }
        ],
    )
    exec_plan = plan.to_execution_plan()
    step = exec_plan.steps[0]

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=exec_plan,
        step=step,
        plan_state={"workspace_members": ["core", "util", "cli"]},
    )

    assert result.success is True
    assert "3 crate(s)" in result.output
    assert received_plan_state.get("workspace_members") == ["core", "util", "cli"]
    subagents.spawn.assert_not_awaited()

    PlanningTeamExecutionAdapter._COMPUTE_NODES.pop("lang_checklist", None)


# ---------------------------------------------------------------------------
# Built-in _workspace_members compute node
# ---------------------------------------------------------------------------


def test_builtin_workspace_members_parses_cargo_toml(tmp_path):
    """_builtin_parse_workspace_members reads Cargo.toml and returns member paths."""
    cargo = tmp_path / "rust" / "Cargo.toml"
    cargo.parent.mkdir(parents=True)
    cargo.write_text(
        '[workspace]\nmembers = [\n    "crates/protocol",\n    "crates/state",\n]\n'
    )
    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        step = SimpleNamespace(description="Inventory workspace members from rust/Cargo.toml", context={})
        result = PlanningTeamExecutionAdapter._builtin_parse_workspace_members(step, {})
        assert result.success is True
        assert "rust/crates/protocol" in result.output
        assert "rust/crates/state" in result.output
        assert result.metadata["member_count"] == 2
    finally:
        os.chdir(old_cwd)


def test_builtin_workspace_members_returns_none_when_no_cargo_toml(tmp_path):
    """Returns (none) sentinel when no Cargo.toml exists."""
    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        step = SimpleNamespace(description="Inventory workspace members", context={})
        result = PlanningTeamExecutionAdapter._builtin_parse_workspace_members(step, {})
        assert result.success is True
        assert result.output == "(none)"
    finally:
        os.chdir(old_cwd)


def test_compute_node_for_step_auto_detects_workspace_members():
    """_compute_node_for_step returns '_workspace_members' when description mentions cargo.toml."""
    step = SimpleNamespace(
        description="Inventory all workspace members from rust/Cargo.toml",
        execution=None,
        context={"produces": "workspace_members"},
    )
    # _workspace_members is registered at module load
    result = PlanningTeamExecutionAdapter._compute_node_for_step(step)
    assert result == "_workspace_members"


def test_compute_node_for_step_no_auto_detect_without_cargo_toml():
    """Auto-detection must NOT fire when description lacks 'cargo.toml' — other workspace formats fall through to LLM."""
    step = SimpleNamespace(
        description="Produce a list of all Python source files",
        execution=None,
        context={"produces": "workspace_members"},
    )
    result = PlanningTeamExecutionAdapter._compute_node_for_step(step)
    assert result is None  # No cargo.toml mention → falls through to LLM


def test_should_use_team_handles_dict_format_steps():
    """should_use_team must not crash on rich dict steps (KeyError: 1 guard)."""
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())

    # Dict-format steps with exploratory types → exploratory count >= 2
    plan_moderate = ReadableTaskPlan(
        name="Dict plan",
        complexity=TaskComplexity.MODERATE,
        desc="Dict step team detection",
        steps=[
            {"id": "1", "type": "analyze", "desc": "Analyze codebase"},
            {"id": "2", "type": "review", "desc": "Review patterns"},
            {"id": "3", "type": "feature", "desc": "Apply fixes"},
        ],
    )
    assert adapter.should_use_team(plan_moderate) is True

    # Complex plans always use team regardless of step format
    plan_complex = ReadableTaskPlan(
        name="Complex plan",
        complexity=TaskComplexity.COMPLEX,
        desc="Complex",
        steps=[{"id": "1", "type": "analyze", "desc": "step"}],
    )
    assert adapter.should_use_team(plan_complex) is True

    # Mixed list + dict steps
    plan_mixed = ReadableTaskPlan(
        name="Mixed",
        complexity=TaskComplexity.MODERATE,
        desc="Mixed step formats",
        steps=[
            ["1", "analyze", "List step"],
            {"id": "2", "type": "analyze", "desc": "Dict step"},
        ],
    )
    assert adapter.should_use_team(plan_mixed) is True


def test_should_use_team_forces_team_for_advanced_execution_types():
    """Any step with compute/loop/conditional/approval forces team routing."""
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())

    for exec_type in ("compute", "loop", "conditional", "approval", "checkpoint", "tool"):
        plan = ReadableTaskPlan(
            name=f"Plan with {exec_type}",
            complexity=TaskComplexity.SIMPLE,
            desc=f"Single {exec_type} step",
            steps=[{"id": "1", "type": "feature", "desc": "Step", "exec": exec_type}],
        )
        assert adapter.should_use_team(plan) is True, f"Expected True for exec={exec_type!r}"

    # Compact list format with 6th element
    plan_list = ReadableTaskPlan(
        name="List format compute",
        complexity=TaskComplexity.SIMPLE,
        desc="List step with exec",
        steps=[["1", "feature", "Compute step", "read", [], "compute"]],
    )
    assert adapter.should_use_team(plan_list) is True

    # Plain agent step on a SIMPLE plan → only 1 exploratory step → False
    plan_simple = ReadableTaskPlan(
        name="Simple plan",
        complexity=TaskComplexity.SIMPLE,
        desc="Simple single step",
        steps=[{"id": "1", "type": "feature", "desc": "Write code"}],
    )
    assert adapter.should_use_team(plan_simple) is False


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


# ---------------------------------------------------------------------------
# Loop node tests
# ---------------------------------------------------------------------------


def _spawn_result(item: str, success: bool = True):
    from types import SimpleNamespace as NS

    return NS(
        success=success,
        summary=f"Reviewed {item}.",
        error=None if success else f"Failed on {item}",
        tool_calls_used=3,
        duration_seconds=1.0,
        details={},
    )


def _make_loop_plan(items=None, loop_over=None):
    step_dict: dict = {
        "id": "5",
        "type": "analyze",
        "desc": "Review each workspace member",
        "tools": ["read", "grep"],
        "deps": [],
        "exec": "loop",
    }
    if items is not None:
        step_dict["items"] = items
    if loop_over is not None:
        step_dict["loop_over"] = loop_over

    return ReadableTaskPlan(
        name="Loop test",
        complexity=TaskComplexity.COMPLEX,
        desc="Loop over crates",
        steps=[step_dict],
    )


@pytest.mark.asyncio
async def test_loop_node_iterates_over_static_items():
    """Loop node spawns one subagent per static item, aggregates output."""
    calls = []

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        calls.append(item)
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(items=["protocol", "state", "tools"])
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={},
    )

    assert result.success is True
    assert calls == ["protocol", "state", "tools"]
    assert result.metadata["execution_mode"] == "loop_node"
    assert result.metadata["loop_items_count"] == 3
    assert "[protocol]" in result.output
    assert "[state]" in result.output
    assert "[tools]" in result.output


@pytest.mark.asyncio
async def test_loop_node_resolves_items_from_plan_state():
    """Loop node reads item list from plan_state when loop_over is set."""
    calls = []

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        calls.append(item)
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(loop_over="workspace_members")
    execution_plan = plan.to_execution_plan()

    plan_state = {"workspace_members": ["alpha", "beta"]}
    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state=plan_state,
    )

    assert result.success is True
    assert calls == ["alpha", "beta"]
    assert result.metadata["loop_items_count"] == 2


@pytest.mark.asyncio
async def test_loop_node_no_items_returns_success_with_skip_message():
    """Loop node with no items or empty plan_state succeeds with a skip notice."""
    subagents = MagicMock()
    subagents.spawn = AsyncMock()
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(loop_over="missing_key")
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={},
    )

    assert result.success is True
    assert "no items" in result.output.lower()
    subagents.spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_loop_node_partial_failure_marks_result_failed():
    """Loop node is failed when any item fails; failed items listed in error."""

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        return _spawn_result(item, success=(item != "beta"))

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(items=["alpha", "beta", "gamma"])
    execution_plan = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        plan_state={},
    )

    assert result.success is False
    assert "beta" in result.metadata["failed_items"]
    assert result.metadata["loop_items_count"] == 3


# ---------------------------------------------------------------------------
# Plan state / produces key tests
# ---------------------------------------------------------------------------


def test_resolve_loop_items_falls_back_to_newline_string():
    """When plan_state value is a raw string, split it into lines."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="5",
        description="loop step",
        step_type=StepType.RESEARCH,
        execution="loop",
        context={"loop_over": "members"},
    )
    plan_state = {"members": "protocol\nstate\ntools"}
    items = PlanningTeamExecutionAdapter._resolve_loop_items(step, plan_state)
    assert items == ["protocol", "state", "tools"]


def test_step_dict_with_produces_and_loop_over_parsed_into_context():
    """Rich dict steps carry loop_over and produces through to PlanStep.context."""
    plan = ReadableTaskPlan(
        name="state passing",
        complexity=TaskComplexity.COMPLEX,
        desc="state test",
        steps=[
            {
                "id": 2,
                "type": "analyze",
                "desc": "Discover crates",
                "tools": ["shell"],
                "deps": [],
                "exec": "tool",
                "produces": "workspace_members",
            },
            {
                "id": 3,
                "type": "analyze",
                "desc": "Review each crate",
                "tools": ["grep"],
                "deps": [2],
                "exec": "loop",
                "loop_over": "workspace_members",
            },
        ],
    )
    ep = plan.to_execution_plan()
    producer = ep.steps[0]
    looper = ep.steps[1]

    assert producer.context["produces"] == "workspace_members"
    assert producer.execution == "tool"
    assert looper.context["loop_over"] == "workspace_members"
    assert looper.execution == "loop"


# ---------------------------------------------------------------------------
# Conditional node tests
# ---------------------------------------------------------------------------


def _conditional_step(condition_on, condition="non_empty", branches=None, produces=""):
    from victor.agent.planning.base import PlanStep, StepType

    ctx = {
        "condition_on": condition_on,
        "condition": condition,
        "execution": "conditional",
    }
    if branches:
        ctx["branches"] = branches
    if produces:
        ctx["produces"] = produces
    return PlanStep(
        id="cond",
        description="conditional test step",
        step_type=StepType.RESEARCH,
        execution="conditional",
        context=ctx,
    )


def test_conditional_node_non_empty_true():
    # branches["true"] = steps to RUN when True → skip branches["false"] when True
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    step = _conditional_step(
        "members", "non_empty", branches={"true": ["yes_step"], "false": ["7"]}
    )
    result = adapter._execute_conditional_node(step, {"members": ["a", "b"]})

    assert result.success is True
    assert result.metadata["condition_result"] is True
    assert result.metadata["active_branch"] == "true"
    assert result.metadata["skip_step_ids"] == ["7"]  # skip the "false" branch


def test_conditional_node_non_empty_false_skips_true_branch():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    # branches["true"] = loop step; when condition is False, skip "true" branch (loop)
    step = _conditional_step("members", "non_empty", branches={"true": ["6a"], "false": ["6b"]})
    result = adapter._execute_conditional_node(step, {"members": []})

    assert result.metadata["condition_result"] is False
    assert result.metadata["active_branch"] == "false"
    assert result.metadata["skip_step_ids"] == ["6a"]  # skip the "true" branch


def test_conditional_node_multiple_condition():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    # True (multiple crates) → run loop "6a", skip single-agent "6b"
    # branches["true"] = loop step "6a"; branches["false"] = single-agent "6b"
    # When True: inactive="false" → skip branches["false"] = ["6b"] ✓
    step = _conditional_step(
        "crates",
        "multiple",
        branches={"true": ["6a"], "false": ["6b"]},
        produces="is_workspace",
    )
    plan_state: dict = {"crates": ["alpha", "beta", "gamma"]}
    result = adapter._execute_conditional_node(step, plan_state)

    assert result.metadata["condition_result"] is True
    assert result.metadata["skip_step_ids"] == ["6b"]  # skip single-agent when multi-crate
    assert plan_state["is_workspace"] is True


def test_conditional_node_single_item_multiple_false():
    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    # branches["true"] = loop "6a"; branches["false"] = single-agent "6b"
    # single item → multiple=False → inactive="true" → skip ["6a"] (loop)
    step = _conditional_step("crates", "multiple", branches={"true": ["6a"], "false": ["6b"]})
    result = adapter._execute_conditional_node(step, {"crates": ["only_one"]})

    assert result.metadata["condition_result"] is False
    assert result.metadata["skip_step_ids"] == ["6a"]  # skip loop when single crate


def test_conditional_node_applies_skip_via_runtime(tmp_path):
    """_skip_specific_steps marks PENDING steps SKIPPED; already-running steps are unaffected."""
    from victor.agent.planning.base import PlanStep, StepStatus, StepType, ExecutionPlan
    from victor.agent.services.planning_runtime import PlanningRuntimeService

    steps = [
        PlanStep(id="loop_step", description="loop", step_type=StepType.RESEARCH),
        PlanStep(id="single_step", description="single", step_type=StepType.RESEARCH),
        PlanStep(
            id="synth_step",
            description="synth",
            step_type=StepType.RESEARCH,
            status=StepStatus.IN_PROGRESS,
        ),
    ]
    plan = ExecutionPlan(id="p", goal="test", steps=steps)

    PlanningRuntimeService._skip_specific_steps(plan, ["loop_step"])

    assert steps[0].status == StepStatus.SKIPPED
    assert steps[1].status == StepStatus.PENDING
    assert steps[2].status == StepStatus.IN_PROGRESS  # not touched


def test_conditional_step_dict_parses_branches_and_condition_on():
    plan = ReadableTaskPlan(
        name="cond test",
        complexity=TaskComplexity.COMPLEX,
        desc="branch routing",
        steps=[
            {
                "id": 5,
                "type": "analyze",
                "desc": "Route by workspace size",
                "deps": [],
                "exec": "conditional",
                "condition_on": "workspace_members",
                "condition": "multiple",
                "produces": "is_workspace",
                "branches": {"true": ["6b"], "false": ["6a"]},
            }
        ],
    )
    ep = plan.to_execution_plan()
    step = ep.steps[0]

    assert step.execution == "conditional"
    assert step.context["condition_on"] == "workspace_members"
    assert step.context["condition"] == "multiple"
    assert step.context["produces"] == "is_workspace"
    assert step.context["branches"] == {"true": ["6b"], "false": ["6a"]}


# ---------------------------------------------------------------------------
# Exit criteria enforcement tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_node_stops_early_when_exit_criteria_met():
    """Sequential loop stops after the first iteration that satisfies exit_criteria."""
    calls = []

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        calls.append(item)
        summary = "done reviewing" if item == "beta" else "still working"
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )

    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="L",
        description="review each crate",
        step_type=StepType.RESEARCH,
        execution="loop",
        context={"items": ["alpha", "beta", "gamma"]},
        exit_criteria=["reviewed alpha"],  # satisfied after first item's output
    )
    plan = _make_loop_plan(items=["alpha", "beta", "gamma"])
    ep = plan.to_execution_plan()
    # Inject exit_criteria into the step
    ep.steps[0].exit_criteria = ["Reviewed alpha"]

    result = await adapter.execute_step(
        plan=plan,
        execution_plan=ep,
        step=ep.steps[0],
        plan_state={},
    )

    assert result.success is True
    assert result.metadata["early_stopped"] is True
    assert len(calls) == 1  # stopped after "alpha"


@pytest.mark.asyncio
async def test_loop_node_no_early_stop_without_exit_criteria():
    """Loop runs all items when no exit_criteria are set."""
    calls = []

    async def fake_spawn(**kwargs):
        item = kwargs["display_name"].split(": ", 1)[-1]
        calls.append(item)
        return _spawn_result(item)

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = _make_loop_plan(items=["x", "y", "z"])
    ep = plan.to_execution_plan()

    result = await adapter.execute_step(
        plan=plan, execution_plan=ep, step=ep.steps[0], plan_state={}
    )
    assert calls == ["x", "y", "z"]
    assert result.metadata.get("early_stopped") is False


# ---------------------------------------------------------------------------
# Approval node tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_node_auto_approves_when_no_callback():
    """Approval node succeeds immediately (auto-approve) when no callback is set."""
    from victor.agent.planning.base import PlanStep, StepType

    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    step = PlanStep(
        id="A",
        description="Review findings before applying fixes",
        step_type=StepType.REVIEW,
        execution="approval",
    )
    plan = ReadableTaskPlan(
        name="ap",
        complexity=TaskComplexity.SIMPLE,
        desc="d",
        steps=[["A", "review", "Review findings before applying fixes", ""]],
    )
    ep = plan.to_execution_plan()

    result = await adapter._execute_approval_node(step, {})

    assert result.success is True
    assert result.metadata["approved"] is True
    assert result.metadata["auto_approved"] is True


@pytest.mark.asyncio
async def test_approval_node_calls_callback_and_uses_decision():
    """Approval node respects callback decision: rejected → failure."""
    from victor.agent.planning.base import PlanStep, StepType

    async def reject_callback(step, context):
        return False, "Not ready yet"

    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(),
        approval_callback=reject_callback,
    )
    step = PlanStep(
        id="A", description="Apply patches", step_type=StepType.REVIEW, execution="approval"
    )
    result = await adapter._execute_approval_node(step, {})

    assert result.success is False
    assert result.metadata["approved"] is False
    assert "Not ready yet" in (result.metadata["feedback"] or "")


@pytest.mark.asyncio
async def test_approval_node_approved_via_callback():
    from victor.agent.planning.base import PlanStep, StepType

    async def approve_callback(step, context):
        return True, "Looks good"

    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(),
        approval_callback=approve_callback,
    )
    step = PlanStep(
        id="A", description="Deploy", step_type=StepType.DEPLOYMENT, execution="approval"
    )
    result = await adapter._execute_approval_node(step, {})

    assert result.success is True
    assert result.metadata["approved"] is True
    assert "Looks good" in result.output


# ---------------------------------------------------------------------------
# Fallback resolution tests
# ---------------------------------------------------------------------------


def test_conditional_node_falls_back_to_word_overlap_when_exact_key_missing():
    """When condition_on key is absent, the node searches plan_state by word overlap."""
    from types import SimpleNamespace
    from victor.agent.planning.base import PlanStep, StepType

    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    step = PlanStep(
        id="7",
        description="Route: multi vs single",
        step_type=StepType.RESEARCH,
        execution="conditional",
        context={
            "condition_on": "workspace_members",  # key is absent from plan_state
            "condition": "multiple",
            "branches": {"true": ["8a"], "false": ["8b"]},
        },
    )
    # plan_state uses a different key but with overlapping words
    plan_state = {"workspace_member_crates": ["crates/protocol", "crates/state", "crates/tools"]}

    result = adapter._execute_conditional_node(step, plan_state)

    # Should resolve the list via fallback and evaluate True (multiple items)
    assert result.success is True
    assert result.metadata["condition_result"] is True
    assert "8b" in result.metadata["skip_step_ids"]  # false branch skipped


def test_resolve_loop_items_falls_back_when_key_name_differs():
    """_resolve_loop_items resolves via word-overlap when loop_over key is absent."""
    from types import SimpleNamespace
    from victor.agent.planning.base import PlanStep, StepType

    adapter = PlanningTeamExecutionAdapter(orchestrator=SimpleNamespace())
    step = PlanStep(
        id="8a",
        description="Loop over each workspace member crate",
        step_type=StepType.RESEARCH,
        execution="loop",
        context={"loop_over": "workspace_members"},  # exact key absent
    )
    # plan_state uses a slightly different key
    plan_state = {"workspace_member_crates": ["crates/protocol", "crates/state"]}

    items = adapter._resolve_loop_items(step, plan_state)
    assert items == ["crates/protocol", "crates/state"]


# ---------------------------------------------------------------------------
# Regression: _task_description_for_step output-format injection
# ---------------------------------------------------------------------------


def test_task_description_for_step_appends_format_instruction_when_produces_key_present():
    """Regression: steps with a 'produces' key must carry explicit output-format instructions.

    Bug: LLM sub-agents for list-producing steps returned prose summaries instead of
    structured lists, causing _extract_list_from_output to return [] due to its prose
    guard, which in turn left plan_state unpopulated and broke conditional routing.
    Fix: _task_description_for_step appends a required output-format block whenever
    the step carries a 'produces' context key.
    """
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="2",
        description="Discover Rust workspace members",
        step_type=StepType.RESEARCH,
        context={"produces": "workspace_members"},
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step)

    assert task.startswith("Discover Rust workspace members")
    assert "OUTPUT FORMAT" in task
    assert "workspace_members" in task
    assert "(none)" in task


def test_task_description_for_step_unchanged_without_produces_key():
    """Steps without a 'produces' key are not augmented — no spurious instructions."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="3",
        description="Review Arc usage across all crates",
        step_type=StepType.RESEARCH,
        context={},
    )

    task = PlanningTeamExecutionAdapter._task_description_for_step(step)

    assert task == "Review Arc usage across all crates"
    assert "OUTPUT FORMAT" not in task


@pytest.mark.asyncio
async def test_execute_step_passes_format_instruction_to_subagent_for_produces_step():
    """The format instruction reaches the sub-agent spawn call when produces is set."""
    captured_task: list[str] = []

    async def fake_spawn(**kwargs):
        captured_task.append(kwargs.get("task", ""))
        return SimpleNamespace(
            success=True,
            summary="crates/protocol\ncrates/state\ncrates/tools",
            error=None,
            details={},
            tool_calls_used=1,
            duration_seconds=0.1,
        )

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Workspace discovery",
        complexity=TaskComplexity.COMPLEX,
        desc="Discover crates",
        steps=[
            {
                "id": "2",
                "type": "analyze",
                "desc": "Discover Rust workspace members",
                "tools": ["shell"],
                "deps": [],
                "exec": "tool",
                "produces": "workspace_members",
            }
        ],
    )
    execution_plan = plan.to_execution_plan()

    await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
    )

    assert len(captured_task) == 1
    assert "OUTPUT FORMAT" in captured_task[0]
    assert "workspace_members" in captured_task[0]
    assert "(none)" in captured_task[0]


def test_task_description_injects_plan_state_list_context():
    """plan_state list values appear as 'Context from prior steps' in the task."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="3",
        description="Inventory Rust source files in each crate",
        step_type=StepType.RESEARCH,
        context={"produces": "crate_file_inventory"},
    )
    plan_state = {
        "workspace_members": [
            "rust/crates/protocol",
            "rust/crates/state",
            "rust/crates/tools",
            "rust/crates/edge-runtime",
            "rust/crates/python-bindings",
        ]
    }

    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state)

    assert "Context from prior steps" in task
    assert "workspace_members: rust/crates/protocol" in task
    assert "rust/crates/edge-runtime" in task
    assert "OUTPUT FORMAT" in task
    assert task.index("Context from prior steps") < task.index("OUTPUT FORMAT")


def test_task_description_excludes_raw_step_dumps_from_plan_state():
    """step_N keys (raw step output text) must not bleed into the task description."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="4",
        description="Analyze crate dependencies",
        step_type=StepType.RESEARCH,
        context={},
    )
    plan_state = {
        "step_1": "verified git repo at /Users/vijaysingh/code/codingagent",
        "step_2": "rust/crates/protocol\nrust/crates/state",
        "workspace_members": ["rust/crates/protocol", "rust/crates/state"],
    }

    task = PlanningTeamExecutionAdapter._task_description_for_step(step, plan_state)

    assert "step_1" not in task
    assert "step_2" not in task
    assert "workspace_members" in task


def test_task_description_unchanged_when_plan_state_empty():
    """An empty plan_state dict must not alter the task description."""
    from victor.agent.planning.base import PlanStep, StepType

    step = PlanStep(
        id="2",
        description="Discover workspace members",
        step_type=StepType.RESEARCH,
        context={"produces": "workspace_members"},
    )

    task_without = PlanningTeamExecutionAdapter._task_description_for_step(step)
    task_with_empty = PlanningTeamExecutionAdapter._task_description_for_step(step, {})

    assert task_without == task_with_empty


@pytest.mark.asyncio
async def test_execute_step_tool_node_injects_plan_state_into_task():
    """Tool node path must forward plan_state context to the sub-agent task string."""
    captured_task: list[str] = []

    async def fake_spawn(**kwargs):
        captured_task.append(kwargs.get("task", ""))
        return SimpleNamespace(
            success=True,
            summary="rust/crates/protocol/src/lib.rs\nrust/crates/state/src/lib.rs",
            error=None,
            details={},
            tool_calls_used=5,
            duration_seconds=1.0,
        )

    subagents = MagicMock()
    subagents.spawn = AsyncMock(side_effect=fake_spawn)
    adapter = PlanningTeamExecutionAdapter(
        orchestrator=SimpleNamespace(active_session_id="sess"),
        sub_agent_orchestrator=subagents,
    )
    plan = ReadableTaskPlan(
        name="Crate inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Map Rust crate files",
        steps=[
            {
                "id": "3",
                "type": "analyze",
                "desc": "Inventory all Rust source files in each workspace crate",
                "tools": ["shell"],
                "deps": [],
                "exec": "tool",
                "produces": "crate_file_inventory",
            }
        ],
    )
    execution_plan = plan.to_execution_plan()

    await adapter.execute_step(
        plan=plan,
        execution_plan=execution_plan,
        step=execution_plan.steps[0],
        root_session_id="sess",
        plan_state={
            "workspace_members": [
                "rust/crates/protocol",
                "rust/crates/state",
                "rust/crates/tools",
            ]
        },
    )

    assert len(captured_task) == 1
    assert "Context from prior steps" in captured_task[0]
    assert "workspace_members" in captured_task[0]
    assert "rust/crates/protocol" in captured_task[0]
    assert "OUTPUT FORMAT" in captured_task[0]
