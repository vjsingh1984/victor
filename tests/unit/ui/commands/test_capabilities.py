"""Tests for shared capability discovery surfaces."""

from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from victor.ui.commands.capabilities import CapabilityDiscovery, capabilities_app


runner = CliRunner()


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


def test_recommend_for_task_uses_shared_framework_coordination_runtime() -> None:
    discovery = CapabilityDiscovery()
    serialized = [{"vertical": "coding", "primary_team": {"team_name": "feature_team"}}]

    with (
        patch(
            "victor.framework.coordination_runtime.build_registered_coordination_suggestions",
            return_value=["raw-suggestion"],
        ) as build_suggestions,
        patch(
            "victor.framework.coordination_runtime.serialize_catalog_coordination_suggestions",
            return_value=serialized,
        ) as serialize_suggestions,
    ):
        payload = discovery.recommend_for_task(
            task_type="feature",
            complexity="high",
            mode="build",
            vertical="coding",
        )

    assert payload == {
        "task_type": "feature",
        "complexity": "high",
        "mode": "build",
        "vertical": "coding",
        "count": 1,
        "recommendations": serialized,
    }
    build_suggestions.assert_called_once_with(
        task_type="feature",
        complexity="high",
        mode="build",
        vertical="coding",
    )
    serialize_suggestions.assert_called_once_with(["raw-suggestion"])


def test_recommend_command_json_uses_shared_framework_recommendation_surface() -> None:
    payload = {
        "task_type": "feature",
        "complexity": "high",
        "mode": "build",
        "vertical": "coding",
        "count": 1,
        "recommendations": [{"vertical": "coding", "action": "auto_spawn"}],
    }

    with patch(
        "victor.ui.commands.capabilities.get_capability_discovery",
        return_value=type(
            "Discovery",
            (),
            {
                "recommend_for_task": lambda self, **_: payload,
            },
        )(),
    ):
        result = runner.invoke(
            capabilities_app,
            ["recommend", "feature", "high", "--mode", "build", "--vertical", "coding", "--json"],
        )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == payload
