from pathlib import Path
from unittest.mock import Mock

import tomllib

from victor_sdk import VictorPlugin, VerticalBase

from victor_research.assistant import ResearchAssistant
from victor_research.plugin import ResearchPlugin, plugin

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _entry_points() -> dict:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["entry-points"]


def test_pyproject_registers_plugin_instance_entry_point() -> None:
    entry_points = _entry_points()

    assert entry_points["victor.plugins"]["research"] == "victor_research.plugin:plugin"


def test_pyproject_registers_canonical_runtime_extension_entry_points() -> None:
    entry_points = _entry_points()

    assert entry_points["victor.prompt_contributors"]["research"] == (
        "victor_research.prompts:ResearchPromptContributor"
    )
    assert entry_points["victor.mode_configs"]["research"] == (
        "victor_research.mode_config:ResearchModeConfigProvider"
    )
    assert entry_points["victor.workflow_providers"]["research"] == (
        "victor_research.workflows:ResearchWorkflowProvider"
    )
    assert entry_points["victor.capability_providers"]["research"] == (
        "victor_research.capabilities:ResearchCapabilityProvider"
    )
    assert entry_points["victor.team_spec_providers"]["research"] == (
        "victor_research.teams:ResearchTeamSpecProvider"
    )


def test_pyproject_keeps_sdk_in_base_dependencies_and_victor_runtime_optional() -> None:
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert any(dependency.startswith("victor-sdk") for dependency in project["dependencies"])
    assert all("victor-ai" not in dependency for dependency in project["dependencies"])
    assert any(
        dependency.startswith("victor-ai>=")
        for dependency in project["optional-dependencies"]["runtime"]
    )


def test_plugin_implements_protocol_and_registers_vertical() -> None:
    context = Mock()

    assert isinstance(plugin, VictorPlugin)
    assert isinstance(plugin, ResearchPlugin)
    assert plugin.name == "research"

    plugin.register(context)

    context.register_vertical.assert_called_once_with(ResearchAssistant)


def test_assistant_inherits_sdk_vertical_base() -> None:
    assert issubclass(ResearchAssistant, VerticalBase)
