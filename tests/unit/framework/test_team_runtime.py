"""Tests for framework team runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from victor.agent.vertical_context import create_vertical_context
from victor.framework.team_runtime import (
    execute_resolved_team,
    list_registered_team_names,
    list_registered_workflow_names,
    resolve_configured_team,
    resolve_named_team,
    resolve_registered_coordination_catalogs,
    resolve_registered_team_catalogs,
    resolve_registered_workflow_catalogs,
    resolve_vertical_coordination_catalog,
    resolve_vertical_team_catalog,
    resolve_vertical_workflow_catalog,
    run_named_team,
    VerticalWorkflowCatalog,
)
from victor.teams.types import TeamFormation, TeamResult


def _make_team_spec(
    *,
    name: str = "Feature Team",
    formation: TeamFormation = TeamFormation.SEQUENTIAL,
    members: list[object] | None = None,
    total_tool_budget: int = 20,
    max_iterations: int = 12,
):
    return SimpleNamespace(
        name=name,
        formation=formation,
        members=members or [SimpleNamespace(name="Researcher"), SimpleNamespace(name="Executor")],
        total_tool_budget=total_tool_budget,
        max_iterations=max_iterations,
    )


def test_resolve_configured_team_prefers_coordination_recommendation():
    feature_team = _make_team_spec()
    review_team = _make_team_spec(name="Review Team")
    orchestrator = SimpleNamespace(
        get_team_specs=lambda: {
            "feature_team": feature_team,
            "review_team": review_team,
        },
        get_coordination_suggestion=lambda *_, **__: SimpleNamespace(
            primary_team=SimpleNamespace(
                team_name="feature_team",
                formation="parallel",
                reason="Feature work needs parallel research and execution",
                source="hybrid",
            )
        ),
        mode_controller=SimpleNamespace(current_mode=SimpleNamespace(value="build")),
    )

    resolved = resolve_configured_team(
        orchestrator,
        task_type="feature",
        complexity="high",
        max_workers=2,
        tool_budget=6,
    )

    assert resolved is not None
    assert resolved.team_name == "feature_team"
    assert resolved.formation == TeamFormation.PARALLEL
    assert resolved.member_count == 2
    assert resolved.total_tool_budget == 6
    assert resolved.recommendation_source == "hybrid"


def test_resolve_named_team_prefers_explicit_team_and_reuses_plan_logic():
    feature_team = _make_team_spec(name="Feature Team", formation=TeamFormation.PARALLEL)
    orchestrator = SimpleNamespace(
        get_team_specs=lambda: {
            "feature_team": feature_team,
        }
    )

    resolved = resolve_named_team(
        orchestrator,
        team_name="feature_team",
        max_workers=1,
        tool_budget=7,
    )

    assert resolved is not None
    assert resolved.team_name == "feature_team"
    assert resolved.display_name == "Feature Team"
    assert resolved.formation == TeamFormation.PARALLEL
    assert resolved.member_count == 1
    assert resolved.total_tool_budget == 7
    assert resolved.recommendation_source == "explicit"


@pytest.mark.asyncio
async def test_execute_resolved_team_limits_members_and_passes_budget():
    members = [
        SimpleNamespace(name="Planner", is_manager=True),
        SimpleNamespace(name="Researcher", is_manager=False),
        SimpleNamespace(name="Executor", is_manager=False),
    ]
    orchestrator = SimpleNamespace(
        get_team_specs=lambda: {
            "feature_team": _make_team_spec(
                name="Feature Team",
                formation=TeamFormation.HIERARCHICAL,
                members=members,
                total_tool_budget=18,
                max_iterations=9,
            )
        },
        get_coordination_suggestion=lambda *_, **__: SimpleNamespace(
            primary_team=SimpleNamespace(
                team_name="feature_team",
                formation="hierarchical",
                reason="Hierarchical execution fits the task",
                source="rule",
            )
        ),
        mode_controller=SimpleNamespace(current_mode=SimpleNamespace(value="build")),
    )
    resolved = resolve_configured_team(
        orchestrator,
        task_type="feature",
        complexity="high",
        max_workers=2,
        tool_budget=5,
    )
    assert resolved is not None

    mock_team = SimpleNamespace(
        run=AsyncMock(
            return_value=TeamResult(
                success=True,
                final_output="done",
                member_results={},
                formation=TeamFormation.HIERARCHICAL,
                total_tool_calls=2,
            )
        )
    )

    with patch(
        "victor.framework.team_runtime.AgentTeam.create", new=AsyncMock(return_value=mock_team)
    ) as create_team:
        result = await execute_resolved_team(
            orchestrator,
            goal="Implement the feature",
            resolved_plan=resolved,
            context={"query": "Implement the feature"},
        )

    assert result.success is True
    create_team.assert_awaited_once()
    kwargs = create_team.await_args.kwargs
    assert kwargs["formation"] == TeamFormation.HIERARCHICAL
    assert kwargs["total_tool_budget"] == 5
    assert len(kwargs["members"]) == 2
    assert kwargs["members"][0].name == "Planner"


@pytest.mark.asyncio
async def test_run_named_team_executes_explicit_team_plan():
    feature_team = _make_team_spec(name="Feature Team", formation=TeamFormation.PARALLEL)
    orchestrator = SimpleNamespace(
        get_team_specs=lambda: {"feature_team": feature_team},
    )
    expected_result = TeamResult(
        success=True,
        final_output="done",
        member_results={},
        formation=TeamFormation.PARALLEL,
        total_tool_calls=1,
    )

    with patch(
        "victor.framework.team_runtime.execute_resolved_team",
        new=AsyncMock(return_value=expected_result),
    ) as execute_team:
        team_execution = await run_named_team(
            orchestrator,
            team_name="feature_team",
            goal="Ship the feature",
            context={"query": "Ship the feature"},
            timeout_seconds=30,
        )

    assert team_execution is not None
    resolved_plan, result = team_execution
    assert resolved_plan.team_name == "feature_team"
    assert result is expected_result
    execute_team.assert_awaited_once_with(
        orchestrator,
        goal="Ship the feature",
        resolved_plan=resolved_plan,
        context={"query": "Ship the feature"},
        timeout_seconds=30,
    )


def test_resolve_vertical_team_catalog_reads_provider_specs():
    feature_team = _make_team_spec()
    vertical = SimpleNamespace(
        get_team_spec_provider=lambda: SimpleNamespace(
            get_team_specs=lambda: {"feature_team": feature_team}
        )
    )

    catalog = resolve_vertical_team_catalog(vertical)

    assert catalog.supported is True
    assert catalog.provider_available is True
    assert catalog.has_team_specs is True
    assert catalog.get("feature_team") is feature_team
    assert catalog.list_names() == ["feature_team"]


def test_resolve_vertical_team_catalog_reads_context_snapshot():
    feature_team = _make_team_spec()
    context = create_vertical_context("coding")
    context.apply_team_specs({"feature_team": feature_team})

    catalog = resolve_vertical_team_catalog(context)

    assert catalog.supported is True
    assert catalog.provider_available is True
    assert catalog.has_team_specs is True
    assert catalog.get("feature_team") is feature_team


def test_resolve_vertical_team_catalog_reads_context_config_provider():
    feature_team = _make_team_spec()
    context = create_vertical_context(
        "coding",
        SimpleNamespace(
            get_team_spec_provider=lambda: SimpleNamespace(
                get_team_specs=lambda: {"feature_team": feature_team}
            )
        ),
    )

    catalog = resolve_vertical_team_catalog(context)

    assert catalog.supported is True
    assert catalog.provider_available is True
    assert catalog.has_team_specs is True
    assert catalog.get("feature_team") is feature_team


def test_resolve_vertical_workflow_catalog_with_get_workflows():
    """Test workflow catalog resolution when provider has get_workflows()."""
    workflow_spec = SimpleNamespace(name="feature_workflow", description="Implement features")
    vertical = SimpleNamespace(
        get_workflow_provider=lambda: SimpleNamespace(
            get_workflows=lambda: {"feature_workflow": workflow_spec}
        )
    )

    catalog = resolve_vertical_workflow_catalog(vertical)

    assert catalog.supported is True
    assert catalog.provider_available is True
    assert catalog.has_workflow_specs is True
    assert catalog.get("feature_workflow") is workflow_spec
    assert catalog.list_names() == ["feature_workflow"]


def test_resolve_vertical_workflow_catalog_reads_context_snapshot():
    workflow_spec = SimpleNamespace(name="feature_workflow", description="Implement features")
    context = create_vertical_context("coding")
    context.apply_workflows({"feature_workflow": workflow_spec})

    catalog = resolve_vertical_workflow_catalog(context)

    assert catalog.supported is True
    assert catalog.provider_available is True
    assert catalog.has_workflow_specs is True
    assert catalog.get("feature_workflow") is workflow_spec


def test_resolve_vertical_workflow_catalog_reads_context_config_provider():
    workflow_spec = SimpleNamespace(name="feature_workflow", description="Implement features")
    context = create_vertical_context(
        "coding",
        SimpleNamespace(
            get_workflow_provider=lambda: SimpleNamespace(
                get_workflows=lambda: {"feature_workflow": workflow_spec}
            )
        ),
    )

    catalog = resolve_vertical_workflow_catalog(context)

    assert catalog.supported is True
    assert catalog.provider_available is True
    assert catalog.has_workflow_specs is True
    assert catalog.get("feature_workflow") is workflow_spec


def test_resolve_vertical_workflow_catalog_with_get_workflow_names():
    """Test workflow catalog resolution when provider has get_workflow_names()."""
    vertical = SimpleNamespace(
        get_workflow_provider=lambda: SimpleNamespace(
            get_workflow_names=lambda: ["feature_workflow", "bug_fix_workflow"]
        )
    )

    catalog = resolve_vertical_workflow_catalog(vertical)

    assert catalog.supported is True
    assert catalog.provider_available is True
    assert catalog.has_workflow_specs is True
    # When only names are available, specs are None
    assert catalog.get("feature_workflow") is None
    assert set(catalog.list_names()) == {"feature_workflow", "bug_fix_workflow"}


def test_resolve_vertical_workflow_catalog_no_provider_method():
    """Test workflow catalog resolution when provider has no callable methods."""
    vertical = SimpleNamespace(
        get_workflow_provider=lambda: SimpleNamespace()  # No get_workflows or get_workflow_names
    )

    catalog = resolve_vertical_workflow_catalog(vertical)

    assert catalog.supported is True
    assert catalog.provider_available is False
    assert catalog.has_workflow_specs is False
    assert catalog.list_names() == []


def test_resolve_vertical_workflow_catalog_no_vertical():
    """Test workflow catalog resolution when vertical is None."""
    catalog = resolve_vertical_workflow_catalog(None)

    assert catalog.supported is False
    assert catalog.provider_available is False
    assert catalog.has_workflow_specs is False
    assert catalog.list_names() == []


def test_resolve_vertical_workflow_catalog_no_provider():
    """Test workflow catalog resolution when vertical has no provider."""
    vertical = SimpleNamespace(get_workflow_provider=lambda: None)

    catalog = resolve_vertical_workflow_catalog(vertical)

    assert catalog.supported is True
    assert catalog.provider_available is False
    assert catalog.has_workflow_specs is False
    assert catalog.list_names() == []


def test_resolve_vertical_workflow_catalog_not_supported():
    """Test workflow catalog resolution when vertical doesn't support workflows."""
    vertical = SimpleNamespace()  # No get_workflow_provider method

    catalog = resolve_vertical_workflow_catalog(vertical)

    assert catalog.supported is False
    assert catalog.provider_available is False
    assert catalog.has_workflow_specs is False
    assert catalog.list_names() == []


def test_resolve_registered_team_catalogs_wraps_provider_registry_payload():
    feature_team = _make_team_spec()
    provider_registry = SimpleNamespace(
        get_all_team_specs=lambda: {"coding": {"feature_team": feature_team}}
    )

    with patch(
        "victor.framework.providers.protocol.get_provider_registry",
        return_value=provider_registry,
    ):
        catalogs = resolve_registered_team_catalogs()
        listed_names = list_registered_team_names()
        listed_vertical_names = list_registered_team_names(
            vertical="coding",
            qualify_with_vertical=False,
        )

    assert set(catalogs.keys()) == {"coding"}
    assert catalogs["coding"].has_team_specs is True
    assert catalogs["coding"].get("feature_team") is feature_team
    assert listed_names == ["coding:feature_team"]
    assert listed_vertical_names == ["feature_team"]


def test_resolve_registered_workflow_catalogs_wraps_provider_registry_payload():
    workflow_spec = SimpleNamespace(name="feature_workflow")
    provider_registry = SimpleNamespace(
        get_all_workflows=lambda: {"coding": {"feature_workflow": workflow_spec}}
    )

    with patch(
        "victor.framework.providers.protocol.get_provider_registry",
        return_value=provider_registry,
    ):
        catalogs = resolve_registered_workflow_catalogs()
        listed_names = list_registered_workflow_names()
        listed_vertical_names = list_registered_workflow_names(
            vertical="coding",
            qualify_with_vertical=False,
        )

    assert set(catalogs.keys()) == {"coding"}
    assert catalogs["coding"].has_workflow_specs is True
    assert catalogs["coding"].get("feature_workflow") is workflow_spec
    assert listed_names == ["coding:feature_workflow"]
    assert listed_vertical_names == ["feature_workflow"]


def test_resolve_registered_catalogs_normalizes_unnamespaced_payload_to_default():
    provider_registry = SimpleNamespace(
        get_all_team_specs=lambda: {"feature_team": _make_team_spec()},
        get_all_workflows=lambda: {"feature_workflow": SimpleNamespace(name="feature_workflow")},
    )

    with patch(
        "victor.framework.providers.protocol.get_provider_registry",
        return_value=provider_registry,
    ):
        team_catalogs = resolve_registered_team_catalogs()
        workflow_catalogs = resolve_registered_workflow_catalogs()

    assert set(team_catalogs.keys()) == {"default"}
    assert team_catalogs["default"].list_names() == ["feature_team"]
    assert set(workflow_catalogs.keys()) == {"default"}
    assert workflow_catalogs["default"].list_names() == ["feature_workflow"]


def test_resolve_vertical_coordination_catalog_combines_team_and_workflow_sources():
    feature_team = _make_team_spec()
    workflow_spec = SimpleNamespace(name="feature_workflow")
    context = create_vertical_context("coding")
    context.apply_team_specs({"feature_team": feature_team})
    context.apply_workflows({"feature_workflow": workflow_spec})

    catalog = resolve_vertical_coordination_catalog(context)

    assert catalog.has_team_specs is True
    assert catalog.has_workflow_specs is True
    assert catalog.list_team_names() == ["feature_team"]
    assert catalog.list_workflow_names() == ["feature_workflow"]


def test_resolve_registered_coordination_catalogs_merges_team_and_workflow_catalogs():
    feature_team = _make_team_spec()
    workflow_spec = SimpleNamespace(name="feature_workflow")
    provider_registry = SimpleNamespace(
        get_all_team_specs=lambda: {"coding": {"feature_team": feature_team}},
        get_all_workflows=lambda: {"coding": {"feature_workflow": workflow_spec}},
    )

    with patch(
        "victor.framework.providers.protocol.get_provider_registry",
        return_value=provider_registry,
    ):
        catalogs = resolve_registered_coordination_catalogs()

    assert set(catalogs.keys()) == {"coding"}
    assert catalogs["coding"].list_team_names() == ["feature_team"]
    assert catalogs["coding"].list_workflow_names() == ["feature_workflow"]


def test_vertical_workflow_catalog_frozen():
    """Test that VerticalWorkflowCatalog is a frozen dataclass."""
    workflow_spec = SimpleNamespace(name="test_workflow")
    catalog = VerticalWorkflowCatalog(
        supported=True,
        provider_available=True,
        workflow_specs={"test_workflow": workflow_spec},
    )

    # Should be frozen - attempting to modify should raise an error
    with pytest.raises(Exception):  # FrozenInstanceError from dataclasses
        catalog.supported = False

    with pytest.raises(Exception):
        catalog.workflow_specs = {}


def test_vertical_workflow_catalog_helper_methods():
    """Test VerticalWorkflowCatalog helper methods."""
    workflow_specs = {
        "feature": SimpleNamespace(name="feature"),
        "bugfix": SimpleNamespace(name="bugfix"),
        "review": SimpleNamespace(name="review"),
    }
    catalog = VerticalWorkflowCatalog(
        supported=True,
        provider_available=True,
        workflow_specs=workflow_specs,
    )

    assert catalog.has_workflow_specs is True
    assert catalog.get("feature") is workflow_specs["feature"]
    assert catalog.get("nonexistent") is None
    assert set(catalog.list_names()) == {"feature", "bugfix", "review"}
