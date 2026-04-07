"""Integrity checks for the SDK-only external vertical example."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "external_vertical"
EXAMPLE_SRC_DIR = EXAMPLE_DIR / "src"
README_PATH = EXAMPLE_DIR / "README.md"
PYPROJECT_PATH = EXAMPLE_DIR / "pyproject.toml"


def _example_project_data() -> dict[str, object]:
    """Return parsed example package metadata."""

    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_external_vertical_example_metadata_matches_security_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package metadata, entry points, and the example assistant should stay aligned."""

    project_data = _example_project_data()
    project = project_data["project"]
    entry_points = project["entry-points"]["victor.plugins"]

    monkeypatch.syspath_prepend(str(EXAMPLE_SRC_DIR))
    from victor_security import SecurityAssistant, plugin

    definition = SecurityAssistant.get_definition()

    assert project["name"] == "victor-security"
    assert project["version"] == SecurityAssistant.version
    assert "victor-sdk>=0.6.0" in project["dependencies"]
    assert "victor-ai>=0.3.0" in project["optional-dependencies"]["runtime"]
    assert entry_points["security"] == "victor_security:plugin"
    assert plugin.name == "security"

    assert SecurityAssistant.get_name() == "security"
    assert definition.name == "security"
    assert definition.workflow_metadata.initial_stage == "reconnaissance"
    assert definition.workflow_metadata.workflow_spec == {
        "stage_order": ["reconnaissance", "analysis", "reporting"]
    }
    assert definition.team_metadata.default_team == "security_review_team"


def test_external_vertical_readme_documents_current_install_and_entry_point_flow() -> None:
    """README examples should stay aligned with the package metadata contract."""

    readme = README_PATH.read_text(encoding="utf-8")

    required_snippets = [
        "pip install -e .",
        'pip install -e ".[runtime]"',
        'security = "victor_security:plugin"',
        "get_definition()",
        "`victor-sdk`",
        "`victor-ai`",
        "SecurityAssistant",
        "SDK-only package dependency for authoring",
        "`victor.plugins`",
    ]

    missing = sorted(snippet for snippet in required_snippets if snippet not in readme)
    assert not missing, f"External vertical README is missing required snippets: {missing}"
