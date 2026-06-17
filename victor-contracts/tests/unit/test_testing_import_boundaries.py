"""Tests for SDK import-boundary helper behavior."""

from pathlib import Path

from victor_contracts.testing import assert_import_boundaries


def test_assert_import_boundaries_flags_static_forbidden_import(tmp_path: Path) -> None:
    package_dir = tmp_path / "sample_vertical"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "assistant.py").write_text(
        "from victor.agent.orchestrator import AgentOrchestrator\n",
        encoding="utf-8",
    )

    violations = assert_import_boundaries(str(package_dir))

    assert len(violations) == 1
    assert "from victor.agent.orchestrator import ..." in violations[0]


def test_assert_import_boundaries_flags_dynamic_forbidden_imports(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "sample_vertical"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "\n".join(
            [
                "import importlib",
                "importlib.import_module('victor.agent.orchestrator')",
                "__import__('victor.core.container')",
            ]
        ),
        encoding="utf-8",
    )

    violations = assert_import_boundaries(str(package_dir))

    assert len(violations) == 2
    assert any("dynamic import victor.agent.orchestrator" in item for item in violations)
    assert any("dynamic import victor.core.container" in item for item in violations)
