import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.planning.base import StepResult
from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
from victor.agent.services.planning_runtime import PlanningConfig, PlanningRuntimeService


@pytest.mark.asyncio
async def test_execute_plan_routes_complex_plan_through_team_adapter():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Arc Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust Arc usage by workspace",
        steps=[
            ["1", "research", "Map Rust workspaces", "ls,read"],
            ["2", "review", "Review Arc usage", "grep,read", ["1"]],
        ],
    )
    adapter = MagicMock()
    adapter.should_use_team.return_value = True
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(success=True, output="mapped", tool_calls_used=2),
            StepResult(success=True, output="reviewed", tool_calls_used=3),
        ]
    )

    with (
        patch(
            "victor.agent.planning.team_execution.PlanningTeamExecutionAdapter",
            return_value=adapter,
        ),
        patch("victor.agent.planning.autonomous.AutonomousPlanner") as planner_cls,
    ):
        result = await service._execute_plan(plan, user_approved=True)

    planner_cls.assert_not_called()
    assert adapter.execute_step.await_count == 2
    first_call = adapter.execute_step.await_args_list[0].kwargs
    assert first_call["root_session_id"] == "session_root"
    assert first_call["step"].id == "1"
    assert result.success is True
    assert result.steps_completed == 2
    assert result.total_tool_calls == 5
    assert result.final_output == "mapped\n\nreviewed"


@pytest.mark.asyncio
async def test_team_plan_execution_bounds_independent_step_concurrency():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(
        orchestrator,
        config=PlanningConfig(max_parallel_steps=2),
    )
    plan = ReadableTaskPlan(
        name="Large Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review many independent areas",
        steps=[
            ["1", "research", "Map workspace A", "read"],
            ["2", "research", "Map workspace B", "read"],
            ["3", "research", "Map workspace C", "read"],
            ["4", "research", "Map workspace D", "read"],
        ],
    )
    adapter = MagicMock()
    active = 0
    max_active = 0

    async def execute_step(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return StepResult(
            success=True,
            output=f"done {kwargs['step'].id}",
            tool_calls_used=1,
        )

    adapter.execute_step = AsyncMock(side_effect=execute_step)

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    assert result.steps_completed == 4
    assert adapter.execute_step.await_count == 4
    assert max_active == 2


@pytest.mark.asyncio
async def test_team_plan_continues_after_shell_inventory_fallback_success():
    orchestrator = SimpleNamespace(active_session_id="session_root")
    service = PlanningRuntimeService(orchestrator)
    plan = ReadableTaskPlan(
        name="Rust Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust source after inventory",
        steps=[
            ["1", "analyze", "Enumerate all Rust source files", "shell,read"],
            ["2", "review", "Review Arc usage in inventoried files", "grep,read", [1]],
        ],
    )
    adapter = MagicMock()
    adapter.execute_step = AsyncMock(
        side_effect=[
            StepResult(success=True, output="inventory complete", tool_calls_used=1),
            StepResult(success=True, output="review complete", tool_calls_used=2),
        ]
    )

    result = await service._execute_plan_via_team_adapter(plan, adapter)

    assert result.success is True
    assert result.steps_completed == 2
    assert result.steps_failed == 0
    assert adapter.execute_step.await_count == 2
    assert [call.kwargs["step"].id for call in adapter.execute_step.await_args_list] == [
        "1",
        "2",
    ]
    assert result.final_output == "inventory complete\n\nreview complete"


def test_read_only_plan_does_not_require_execution_approval():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Read-only Review",
        complexity=TaskComplexity.COMPLEX,
        desc="Review Rust code without changing files",
        steps=[
            ["1", "analyze", "Read Cargo.toml", "read"],
            ["2", "review", "Search Arc usages", "grep,code_search,overview"],
            ["3", "doc", "Summarize findings in chat", "read"],
        ],
    )

    assert service._plan_requires_execution_approval(plan) is False


def test_shell_plan_requires_execution_approval():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Inventory",
        complexity=TaskComplexity.COMPLEX,
        desc="Inventory Rust files using a command",
        steps=[
            ["1", "analyze", "Enumerate Rust files", "shell"],
        ],
    )

    assert service._plan_requires_execution_approval(plan) is True


def test_write_plan_requires_execution_approval():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Write Report",
        complexity=TaskComplexity.MODERATE,
        desc="Write findings to disk",
        steps=[
            ["1", "doc", "Write findings report", "write"],
        ],
    )

    assert service._plan_requires_execution_approval(plan) is True


@pytest.mark.asyncio
async def test_plan_approval_prompt_explains_enter_and_reject_options():
    service = PlanningRuntimeService(SimpleNamespace())
    plan = ReadableTaskPlan(
        name="Effectful Plan",
        complexity=TaskComplexity.MODERATE,
        desc="Run tests",
        steps=[["1", "testing", "Run tests", "shell"]],
    )
    console = MagicMock()

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("asyncio.to_thread", new=AsyncMock(return_value=True)),
    ):
        approved = await service._request_plan_approval(plan, console)

    assert approved is True
    rendered = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
    assert "Press Enter to execute" in rendered
    assert "type y then Enter" in rendered
    assert "Type n then Enter to reject" in rendered
