"""Tests for ModeWorkflowTeamCoordinator framework-first catalog reuse."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.mode_workflow_team_coordinator import (
    ModeWorkflowTeamCoordinator,
    RuleBasedTeamSelector,
    RuleBasedWorkflowSelector,
    create_coordinator,
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
    with pytest.warns(
        DeprecationWarning,
        match="ModeWorkflowTeamCoordinator is deprecated",
    ):
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
    with pytest.warns(
        DeprecationWarning,
        match="ModeWorkflowTeamCoordinator is deprecated",
    ):
        coordinator = ModeWorkflowTeamCoordinator(vertical_context=context)

    assert coordinator.get_default_workflow("plan") == "planning_workflow"


def test_create_coordinator_warns_and_wraps_framework_advisor():
    """Legacy coordinator factory should warn and wrap the framework advisor."""
    context = MagicMock(name="vertical_context")
    advisor = MagicMock(name="coordination_advisor")

    with (
        patch(
            "victor.agent.mode_workflow_team_coordinator.create_vertical_coordination_advisor",
            return_value=advisor,
        ) as create_advisor,
        pytest.warns(DeprecationWarning) as recorded,
    ):
        coordinator = create_coordinator(vertical_context=context)

    assert isinstance(coordinator, ModeWorkflowTeamCoordinator)
    assert coordinator._advisor is advisor
    messages = [str(item.message) for item in recorded]
    assert any("create_coordinator(...) is deprecated" in message for message in messages)
    assert any("ModeWorkflowTeamCoordinator is deprecated" in message for message in messages)
    create_advisor.assert_called_once_with(
        vertical_context=context,
        team_learner=None,
        selection_strategy="hybrid",
    )
