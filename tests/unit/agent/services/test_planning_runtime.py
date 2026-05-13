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
