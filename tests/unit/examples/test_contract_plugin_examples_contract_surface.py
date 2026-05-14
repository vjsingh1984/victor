"""Integrity checks for plugin examples that should use the contract namespace."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "contract_plugins"
PYTHON_EXAMPLES = (
    EXAMPLE_DIR / "basic_plugin.py",
    EXAMPLE_DIR / "factory_plugin.py",
    EXAMPLE_DIR / "complete_plugin.py",
)
README_PATH = EXAMPLE_DIR / "README.md"


def test_contract_plugin_examples_import_contract_namespace() -> None:
    for path in PYTHON_EXAMPLES:
        source = path.read_text(encoding="utf-8")

        assert "victor_contracts" in source, f"{path.name} should import contract symbols"
        assert "victor_sdk" not in source, f"{path.name} should not import legacy SDK symbols"


def test_contract_plugin_readme_documents_contract_namespace() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    required_snippets = [
        "victor_contracts.core.plugins",
        "from victor_contracts.core.plugins import VictorPlugin, PluginContext",
        "victor_contracts.verticals.protocols",
    ]

    missing = sorted(snippet for snippet in required_snippets if snippet not in readme)
    assert not missing, f"Contract plugin README is missing contract snippets: {missing}"
    assert "victor_sdk" not in readme
