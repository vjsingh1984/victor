"""Integration smoke test for wheel packaging of vertical assets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import zipfile

import pytest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Unable to locate repository root")


@pytest.mark.integration
@pytest.mark.slow
def test_wheel_omits_extracted_vertical_runtime_assets(tmp_path: Path) -> None:
    """Core wheel should not bundle runtime assets for extracted vertical repos."""
    repo_root = _repo_root()

    build_attempts = [
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(tmp_path),
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "-w",
            str(tmp_path),
        ],
    ]
    build_errors = []
    for command in build_attempts:
        try:
            subprocess.run(
                command,
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            break
        except subprocess.CalledProcessError as exc:
            build_errors.append(exc.stderr or exc.stdout or str(exc))
    else:
        pytest.skip(
            "Wheel build unavailable in this environment: "
            + " | ".join(error[:160] for error in build_errors)
        )

    wheels = sorted(tmp_path.glob("victor_ai-*.whl"))
    assert wheels, "Expected victor_ai wheel artifact"
    wheel_path = wheels[-1]

    legacy_vertical_assets = {
        "victor/verticals/contrib/coding/tool_dependencies.yaml",
        "victor/verticals/contrib/devops/tool_dependencies.yaml",
        "victor/verticals/contrib/research/tool_dependencies.yaml",
        "victor/verticals/contrib/rag/tool_dependencies.yaml",
        "victor/verticals/contrib/dataanalysis/tool_dependencies.yaml",
    }

    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        assert "victor/verticals/contrib/__init__.py" in names
        bundled_legacy_assets = sorted(legacy_vertical_assets & names)
        assert not bundled_legacy_assets, (
            "Core wheel should not bundle extracted vertical runtime assets: "
            f"{bundled_legacy_assets}"
        )

        entry_point_files = [name for name in names if name.endswith("entry_points.txt")]
        assert entry_point_files, "Expected wheel entry_points metadata"
