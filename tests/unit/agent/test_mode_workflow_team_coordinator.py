"""Tests for ModeWorkflowTeamCoordinator framework-first catalog reuse."""

from __future__ import annotations

from types import SimpleNamespace

from victor.agent.mode_workflow_team_coordinator import (
    ModeWorkflowTeamCoordinator,
    RuleBasedTeamSelector,
    RuleBasedWorkflowSelector,
)
from victor.agent.vertical_context import create_vertical_context


def test_suggest_for_task_uses_framework_catalogs_from_context_config_provider():
    """Provider-backed vertical context should drive coordinator suggestions."""
    vertical = SimpleNamespace(
        get_team_spec_provider=lambda: SimpleNamespace(
            get_team_specs=lambda: {
                "feature_team": SimpleNamespace(
                    name="Feature Team",
                    formation="parallel",
                    members=[],
                    description="Handles feature implementation",
                )
            }
        ),
        get_workflow_provider=lambda: SimpleNamespace(
            get_workflows=lambda: {
                "feature_implementation": SimpleNamespace(
                    name="feature_implementation",
                    description="Implement features end-to-end",
                )
            }
        ),
    )
    context = create_vertical_context("coding", vertical)
    coordinator = ModeWorkflowTeamCoordinator(
        vertical_context=context,
        team_selector=RuleBasedTeamSelector(),
        workflow_selector=RuleBasedWorkflowSelector(),
    )

    suggestion = coordinator.suggest_for_task(
        task_type="feature",
        complexity="high",
        mode="build",
    )

    assert suggestion.primary_team is not None
    assert suggestion.primary_team.team_name == "feature_team"
    assert suggestion.workflow_recommendations
    assert suggestion.workflow_recommendations[0].workflow_name == "feature_implementation"


def test_get_default_workflow_prefers_framework_catalog_discovery():
    """Default workflow lookup should honor provider-backed catalogs."""
    vertical = SimpleNamespace(
        get_workflow_provider=lambda: SimpleNamespace(
            get_workflows=lambda: {
                "planning_workflow": SimpleNamespace(name="planning_workflow"),
            }
        )
    )
    context = create_vertical_context("coding", vertical)
    coordinator = ModeWorkflowTeamCoordinator(vertical_context=context)

    assert coordinator.get_default_workflow("plan") == "planning_workflow"
