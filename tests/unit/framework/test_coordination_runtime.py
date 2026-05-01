"""Tests for framework coordination recommendation helpers."""

from __future__ import annotations

from unittest.mock import patch
from types import SimpleNamespace

from victor.framework.coordination_runtime import (
    VerticalCoordinationAdvisor,
    build_registered_coordination_suggestions,
    RuleBasedTeamSelector,
    RuleBasedWorkflowSelector,
    build_coordination_suggestion,
    build_runtime_coordination_suggestion,
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


def test_build_runtime_coordination_suggestion_uses_runtime_mode_and_vertical_context() -> None:
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
    runtime = SimpleNamespace(
        get_vertical_context=lambda: context,
        mode_controller=SimpleNamespace(current_mode=SimpleNamespace(value="build")),
        coordination=SimpleNamespace(
            _team_selector=RuleBasedTeamSelector(),
            _workflow_selector=RuleBasedWorkflowSelector(),
        ),
    )

    suggestion = build_runtime_coordination_suggestion(
        runtime_subject=runtime,
        task_type="feature",
        complexity="high",
    )

    assert suggestion.action == TeamSuggestionAction.AUTO_SPAWN
    assert suggestion.primary_team is not None
    assert suggestion.primary_team.team_name == "feature_team"
    assert suggestion.primary_workflow is not None
    assert suggestion.primary_workflow.workflow_name == "feature_implementation"


def test_build_runtime_coordination_suggestion_prefers_coordination_advisor_surface() -> None:
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
    advisor = VerticalCoordinationAdvisor(
        vertical_context=context,
        team_selector=RuleBasedTeamSelector(),
        workflow_selector=RuleBasedWorkflowSelector(),
    )
    runtime = SimpleNamespace(
        get_vertical_context=lambda: context,
        mode_controller=SimpleNamespace(current_mode=SimpleNamespace(value="build")),
        coordination_advisor=advisor,
        coordination=SimpleNamespace(suggest_for_task=lambda **_: None),
    )

    suggestion = build_runtime_coordination_suggestion(
        runtime_subject=runtime,
        task_type="feature",
        complexity="high",
    )

    assert suggestion.primary_team is not None
    assert suggestion.primary_team.team_name == "feature_team"
    assert suggestion.primary_workflow is not None
    assert suggestion.primary_workflow.workflow_name == "feature_implementation"


def test_build_registered_coordination_suggestions_uses_shared_registered_catalogs() -> None:
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

    with patch(
        "victor.framework.team_runtime.resolve_registered_coordination_catalogs",
        return_value={"coding": resolve_vertical_coordination_catalog(context)},
    ):
        suggestions = build_registered_coordination_suggestions(
            task_type="feature",
            complexity="high",
            mode="build",
        )

    assert [suggestion.vertical for suggestion in suggestions] == ["coding"]
    payload = suggestions[0].to_dict()
    assert payload["primary_team"]["team_name"] == "feature_team"
    assert payload["primary_workflow"]["workflow_name"] == "feature_implementation"
    assert payload["default_workflow"] == "feature_implementation"


def test_build_registered_coordination_suggestions_keeps_requested_vertical_even_without_match() -> (
    None
):
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

    with patch(
        "victor.framework.team_runtime.resolve_registered_coordination_catalogs",
        return_value={"coding": resolve_vertical_coordination_catalog(context)},
    ):
        suggestions = build_registered_coordination_suggestions(
            task_type="documentation",
            complexity="low",
            mode="build",
            vertical="coding",
        )

    assert len(suggestions) == 1
    payload = suggestions[0].to_dict()
    assert payload["vertical"] == "coding"
    assert payload["primary_team"] is None
    assert payload["primary_workflow"] is None


def test_vertical_coordination_advisor_uses_shared_framework_catalogs() -> None:
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
    advisor = VerticalCoordinationAdvisor(
        vertical_context=context,
        team_selector=RuleBasedTeamSelector(),
        workflow_selector=RuleBasedWorkflowSelector(),
    )

    suggestion = advisor.suggest_for_task("feature", "high", "build")

    assert suggestion.primary_team is not None
    assert suggestion.primary_team.team_name == "feature_team"
    assert suggestion.primary_workflow is not None
    assert suggestion.primary_workflow.workflow_name == "feature_implementation"
    assert advisor.get_default_workflow("build") == "feature_implementation"
