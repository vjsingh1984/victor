"""Tests for framework coordination recommendation helpers."""

from __future__ import annotations

from types import SimpleNamespace

from victor.framework.coordination_runtime import (
    RuleBasedTeamSelector,
    RuleBasedWorkflowSelector,
    build_coordination_suggestion,
    get_action_for_complexity,
    recommend_teams_for_catalog,
    recommend_workflows_for_catalog,
    resolve_default_workflow_for_mode,
)
from victor.framework.team_runtime import resolve_vertical_coordination_catalog
from victor.agent.vertical_context import create_vertical_context
from victor.protocols.coordination import TeamSuggestionAction


def test_build_coordination_suggestion_uses_shared_coordination_catalog() -> None:
    context = create_vertical_context("coding")
    context.apply_team_specs(
        {
            "feature_team": SimpleNamespace(
                name="Feature Team",
                formation="parallel",
                members=[],
                description="Handles feature implementation",
            )
        }
    )
    context.apply_workflows(
        {
            "feature_implementation": SimpleNamespace(
                name="feature_implementation",
                description="Implement features end-to-end",
            )
        }
    )
    catalog = resolve_vertical_coordination_catalog(context)

    suggestion = build_coordination_suggestion(
        task_type="feature",
        complexity="high",
        mode="build",
        coordination_catalog=catalog,
        team_selector=RuleBasedTeamSelector(),
        workflow_selector=RuleBasedWorkflowSelector(),
    )

    assert suggestion.action == TeamSuggestionAction.AUTO_SPAWN
    assert suggestion.primary_team is not None
    assert suggestion.primary_team.team_name == "feature_team"
    assert suggestion.primary_workflow is not None
    assert suggestion.primary_workflow.workflow_name == "feature_implementation"


def test_recommendation_helpers_return_empty_when_catalog_is_empty() -> None:
    empty_catalog = resolve_vertical_coordination_catalog(create_vertical_context("coding"))

    assert (
        recommend_teams_for_catalog(
            task_type="feature",
            complexity="high",
            coordination_catalog=empty_catalog,
            team_selector=RuleBasedTeamSelector(),
        )
        == []
    )
    assert (
        recommend_workflows_for_catalog(
            task_type="feature",
            mode="build",
            coordination_catalog=empty_catalog,
            workflow_selector=RuleBasedWorkflowSelector(),
        )
        == []
    )


def test_resolve_default_workflow_for_mode_prefers_available_shared_workflow() -> None:
    context = create_vertical_context("coding")
    context.apply_workflows({"planning_workflow": SimpleNamespace(name="planning_workflow")})
    catalog = resolve_vertical_coordination_catalog(context)

    assert resolve_default_workflow_for_mode("plan", coordination_catalog=catalog) == (
        "planning_workflow"
    )


def test_get_action_for_complexity_uses_shared_mode_policy() -> None:
    assert get_action_for_complexity("high", "build") == TeamSuggestionAction.AUTO_SPAWN
    assert get_action_for_complexity("low", "build") == TeamSuggestionAction.NONE
