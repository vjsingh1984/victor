"""Integrity checks for the minimal contract-only vertical example."""

from __future__ import annotations

from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "victor-contracts" / "examples" / "minimal_vertical"
PYPROJECT_PATH = EXAMPLE_DIR / "pyproject.toml"
README_PATH = EXAMPLE_DIR / "README.md"
INIT_PATH = EXAMPLE_DIR / "__init__.py"
PROTOCOLS_PATH = EXAMPLE_DIR / "protocols.py"


def test_minimal_vertical_example_uses_semantic_extension_entry_points() -> None:
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    project = data["project"]
    entry_points = data["project"]["entry-points"]

    assert "victor-contracts>=0.7.0" in project["dependencies"]
    assert "victor.extension.protocols" in entry_points
    assert "victor.extension.capabilities" in entry_points
    assert "victor.extension.validators" in entry_points
    assert "victor.sdk.protocols" not in entry_points
    assert "victor.sdk.capabilities" not in entry_points
    assert "victor.sdk.validators" not in entry_points


def test_minimal_vertical_example_uses_contract_import_namespace() -> None:
    init_source = INIT_PATH.read_text(encoding="utf-8")
    protocols_source = PROTOCOLS_PATH.read_text(encoding="utf-8")

    assert "from victor_contracts import PluginContext, VictorPlugin" in init_source
    assert (
        "from victor_contracts.verticals.protocols.base import VerticalBase"
        in init_source
    )
    assert (
        "from victor_contracts.verticals.protocols import ToolProvider, SafetyProvider"
        in (init_source)
    )
    assert (
        "from victor_contracts.verticals.protocols import ToolProvider, SafetyProvider"
        in (protocols_source)
    )
    legacy_namespace = "victor" "_sdk"
    assert legacy_namespace not in init_source
    assert legacy_namespace not in protocols_source


def test_minimal_vertical_readme_documents_contract_surface() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    required_snippets = [
        "`victor_contracts`",
        "from victor_contracts.discovery import get_global_registry",
        "victor.extension.protocols",
        "victor.extension.capabilities",
        "victor.extension.validators",
    ]

    missing = sorted(snippet for snippet in required_snippets if snippet not in readme)
    assert (
        not missing
    ), f"Minimal vertical README is missing required snippets: {missing}"
