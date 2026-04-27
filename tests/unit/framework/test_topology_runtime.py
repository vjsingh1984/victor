"""Tests for shared topology runtime preparation helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from victor.agent.topology_contract import TopologyAction, TopologyKind
from victor.agent.topology_grounder import GroundedTopologyPlan
from victor.framework.team_runtime import ResolvedTeamExecutionPlan
from victor.framework.topology_runtime import (
    derive_topology_task_context,
    prepare_topology_runtime_contract,
)
from victor.teams.types import TeamFormation


def test_derive_topology_task_context_normalizes_classification():
    task_type, complexity = derive_topology_task_context(
        SimpleNamespace(task_type="feature", complexity=SimpleNamespace(value="complex"))
    )

    assert task_type == "feature"
    assert complexity == "complex"


def test_prepare_topology_runtime_contract_parallel_exploration_builds_request():
    topology_plan = GroundedTopologyPlan(
        action=TopologyAction.PARALLEL_EXPLORATION,
        topology=TopologyKind.PARALLEL_EXPLORATION,
        execution_mode="parallel_exploration",
        tool_budget=4,
        iteration_budget=2,
    )

    prepared = prepare_topology_runtime_contract(topology_plan)

    assert prepared.action == "parallel_exploration"
    assert prepared.execution_mode == "parallel_exploration"
    assert prepared.parallel_exploration == {"force": True, "max_results_override": 4}
    assert prepared.runtime_context_overrides["topology_action"] == "parallel_exploration"
    assert prepared.runtime_context_overrides["tool_budget"] == 4


def test_prepare_topology_runtime_contract_resolves_team_plan():
    topology_plan = GroundedTopologyPlan(
        action=TopologyAction.TEAM_PLAN,
        topology=TopologyKind.TEAM,
        execution_mode="team_execution",
        formation="parallel",
        max_workers=2,
        tool_budget=6,
        iteration_budget=2,
    )
    orchestrator = SimpleNamespace()

    with patch(
        "victor.framework.topology_runtime.resolve_configured_team",
        return_value=ResolvedTeamExecutionPlan(
            team_name="feature_team",
            display_name="Feature Team",
            formation=TeamFormation.PARALLEL,
            member_count=2,
            total_tool_budget=6,
            max_iterations=20,
            max_workers=2,
        ),
    ) as resolve_team:
        prepared = prepare_topology_runtime_contract(
            topology_plan,
            orchestrator=orchestrator,
            task_type="feature",
            complexity="complex",
        )

    resolve_team.assert_called_once_with(
        orchestrator,
        task_type="feature",
        complexity="complex",
        preferred_team=None,
        preferred_formation="parallel",
        max_workers=2,
        tool_budget=6,
    )
    assert prepared.team_plan is not None
    assert prepared.runtime_context_overrides["team_name"] == "feature_team"
    assert prepared.runtime_context_overrides["formation_hint"] == "parallel"
    assert prepared.runtime_context_overrides["execution_mode"] == "team_execution"
