"""Tests for shared capability discovery surfaces."""

from __future__ import annotations

from unittest.mock import patch

from victor.ui.commands.capabilities import CapabilityDiscovery


def test_discover_all_uses_shared_team_and_workflow_catalog_surfaces() -> None:
    discovery = CapabilityDiscovery()

    with (
        patch.object(discovery, "_discover_tools", return_value=[]),
        patch.object(discovery, "_discover_tool_categories", return_value=[]),
        patch.object(discovery, "_discover_verticals", return_value=["coding"]),
        patch.object(discovery, "_discover_personas", return_value=[]),
        patch.object(discovery, "_discover_chains", return_value=[]),
        patch.object(discovery, "_discover_handlers", return_value=[]),
        patch.object(discovery, "_discover_task_types", return_value=[]),
        patch.object(discovery, "_discover_providers", return_value=[]),
        patch.object(discovery, "_discover_events", return_value=[]),
        patch(
            "victor.framework.team_runtime.list_registered_team_names",
            return_value=["coding:feature_team"],
        ) as list_teams,
        patch(
            "victor.framework.team_runtime.list_registered_workflow_names",
            return_value=["coding:feature_workflow"],
        ) as list_workflows,
    ):
        manifest = discovery.discover_all()

    assert manifest.teams == ["coding:feature_team"]
    assert manifest.workflows == ["coding:feature_workflow"]
    list_teams.assert_called_once_with()
    list_workflows.assert_called_once_with()


def test_discover_by_vertical_uses_shared_team_and_workflow_catalog_surfaces() -> None:
    discovery = CapabilityDiscovery()
    coordination_catalog = type(
        "Catalog",
        (),
        {
            "list_team_names": lambda self: ["feature_team"],
            "list_workflow_names": lambda self: ["feature_workflow"],
        },
    )()

    with (
        patch(
            "victor.framework.team_runtime.resolve_registered_coordination_catalogs",
            return_value={"coding": coordination_catalog},
        ) as resolve_catalogs,
        patch("victor.framework.persona_registry.get_persona_registry", side_effect=Exception),
        patch("victor.framework.chain_registry.get_chain_registry", side_effect=Exception),
        patch("victor.framework.handler_registry.get_handler_registry", side_effect=Exception),
        patch("victor.framework.task_types.get_task_type_registry", side_effect=Exception),
    ):
        manifest = discovery.discover_by_vertical("coding")

    assert manifest["teams"] == ["feature_team"]
    assert manifest["workflows"] == ["feature_workflow"]
    resolve_catalogs.assert_called_once_with()
