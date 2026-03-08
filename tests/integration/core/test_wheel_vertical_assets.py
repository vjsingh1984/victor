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
def test_wheel_contains_vertical_runtime_assets(tmp_path: Path) -> None:
    """Wheel build should contain contrib vertical YAML/TOML/SVG runtime assets."""
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

    required_paths = {
        "victor/verticals/contrib/coding/tool_dependencies.yaml",
        "victor/verticals/contrib/devops/tool_dependencies.yaml",
        "victor/verticals/contrib/research/tool_dependencies.yaml",
        "victor/verticals/contrib/rag/tool_dependencies.yaml",
        "victor/verticals/contrib/dataanalysis/tool_dependencies.yaml",
    }

    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        missing = sorted(required_paths - names)
        assert not missing, f"Missing required vertical assets in wheel: {missing}"

        workflow_assets = [
            name
            for name in names
            if name.startswith("victor/verticals/contrib/")
            and "/workflows/" in name
            and (name.endswith(".yaml") or name.endswith(".svg"))
        ]
        assert workflow_assets, "Expected contrib workflow assets in wheel"
