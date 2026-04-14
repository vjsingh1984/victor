"""Tests for extracted-vertical contract auditing."""

from pathlib import Path

from victor.core.verticals.contract_audit import VerticalContractAuditor


def _write_pyproject(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_audit_reports_missing_plugin_entry_point(tmp_path: Path) -> None:
    repo = tmp_path / "victor-sample"
    package_dir = repo / "victor_sample"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-sample"
version = "0.1.0"
dependencies = ["victor-sdk>=0.1.0"]
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is False
    assert any(issue.code == "missing_plugin_entry_point" for issue in report.issues)


def test_audit_reports_forbidden_core_runtime_imports(tmp_path: Path) -> None:
    repo = tmp_path / "victor-leaky"
    package_dir = repo / "victor_leaky"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-leaky"
version = "0.1.0"
dependencies = ["victor-sdk>=0.1.0"]

[project.entry-points."victor.plugins"]
coding = "victor_leaky.plugin:get_plugin"
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "from victor.framework.agent import Agent\n",
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is False
    assert any(issue.code == "forbidden_runtime_import" for issue in report.issues)


def test_audit_passes_sdk_pure_vertical_repo(tmp_path: Path) -> None:
    repo = tmp_path / "victor-clean"
    package_dir = repo / "victor_clean"
    package_dir.mkdir(parents=True)
    _write_pyproject(
        repo / "pyproject.toml",
        """
[project]
name = "victor-clean"
version = "0.1.0"
dependencies = ["victor-sdk>=0.1.0"]

[project.entry-points."victor.plugins"]
clean = "victor_clean.plugin:get_plugin"
""".strip(),
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        "from victor_sdk.core.plugins import VictorPlugin\n",
        encoding="utf-8",
    )

    report = VerticalContractAuditor().audit_path(repo)

    assert report.passed is True
    assert report.error_count == 0
