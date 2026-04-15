from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys


def load_script_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "ci"
        / "check_extracted_vertical_boundaries.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_extracted_vertical_boundaries",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_extracted_vertical_boundaries = load_script_module()


def _write_vertical_repo(root: Path, name: str, *, leaky: bool) -> Path:
    repo = root / name
    package_dir = repo / name.replace("-", "_")
    package_dir.mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        f"""
[project]
name = "{name}"
version = "0.1.0"
dependencies = ["victor-sdk>=0.1.0"]

[project.entry-points."victor.plugins"]
{name.split('-')[-1]} = "{name.replace('-', '_')}.plugin:get_plugin"
""".strip(),
        encoding="utf-8",
    )
    plugin_import = (
        "from victor.framework.agent import Agent\n"
        if leaky
        else "from victor_sdk.core.plugins import VictorPlugin\n"
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(plugin_import, encoding="utf-8")
    return repo


def test_discover_default_paths_only_returns_existing_repos(tmp_path: Path) -> None:
    core_repo = tmp_path / "victor"
    core_repo.mkdir()
    (tmp_path / "victor-coding").mkdir()
    (tmp_path / "victor-devops").mkdir()

    discovered = check_extracted_vertical_boundaries.discover_default_paths(repo_root=core_repo)

    assert discovered == [
        (tmp_path / "victor-coding").resolve(),
        (tmp_path / "victor-devops").resolve(),
    ]


def test_main_returns_zero_when_no_default_repos_exist(tmp_path: Path) -> None:
    core_repo = tmp_path / "victor"
    core_repo.mkdir()
    output = io.StringIO()

    exit_code = check_extracted_vertical_boundaries.main([], repo_root=core_repo, stdout=output)

    assert exit_code == 0
    assert "No extracted vertical repositories found" in output.getvalue()


def test_main_fails_for_leaky_vertical_repo(tmp_path: Path) -> None:
    core_repo = tmp_path / "victor"
    core_repo.mkdir()
    leaky_repo = _write_vertical_repo(tmp_path, "victor-coding", leaky=True)
    output = io.StringIO()

    exit_code = check_extracted_vertical_boundaries.main(
        [str(leaky_repo)],
        repo_root=core_repo,
        stdout=output,
    )

    assert exit_code == 1
    assert "FAILED" in output.getvalue()
    assert "forbidden_runtime_import" in output.getvalue()


def test_main_passes_for_sdk_pure_vertical_repo(tmp_path: Path) -> None:
    core_repo = tmp_path / "victor"
    core_repo.mkdir()
    clean_repo = _write_vertical_repo(tmp_path, "victor-coding", leaky=False)
    output = io.StringIO()

    exit_code = check_extracted_vertical_boundaries.main(
        [str(clean_repo)],
        repo_root=core_repo,
        stdout=output,
    )

    assert exit_code == 0
    assert "PASSED" in output.getvalue()
