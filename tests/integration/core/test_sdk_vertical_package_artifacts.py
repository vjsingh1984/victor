"""Artifact verification for SDK and external vertical packaging."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
VICTOR_SDK_DIR = REPO_ROOT / "victor-sdk"
EXTERNAL_EXAMPLE_DIR = REPO_ROOT / "examples" / "external_vertical"


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and include stdout/stderr on failure."""

    completed = subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            f"{' '.join(args)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _build_wheel(package_dir: Path, output_dir: Path) -> Path:
    """Build a wheel artifact for a local package without network access."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(package_dir),
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
    )
    wheels = sorted(output_dir.glob("*.whl"))
    assert wheels, f"No wheel built for {package_dir}"
    return wheels[-1]


def _build_sdist(package_dir: Path, output_dir: Path) -> Path:
    """Build an sdist artifact using the local setuptools backend."""

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = _run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from setuptools.build_meta import build_sdist; "
                f"name = build_sdist(r'{output_dir}'); "
                "print(json.dumps({'artifact': name}))"
            ),
        ],
        cwd=package_dir,
    )
    artifact_name = json.loads(completed.stdout.strip().splitlines()[-1])["artifact"]
    artifact_path = output_dir / artifact_name
    assert artifact_path.exists(), f"Expected sdist artifact at {artifact_path}"
    return artifact_path


def _assert_archive_contains_suffixes(names: set[str], required_suffixes: set[str]) -> None:
    """Assert an archive contains paths ending in the required suffixes."""

    missing = sorted(
        suffix
        for suffix in required_suffixes
        if not any(name.endswith(suffix) for name in names)
    )
    assert not missing, f"Missing required archive paths: {missing}"


@pytest.mark.integration
def test_victor_sdk_wheel_and_sdist_include_contract_files(tmp_path: Path) -> None:
    """victor-sdk artifacts should ship the contract surface used by vertical authors."""

    wheel_path = _build_wheel(VICTOR_SDK_DIR, tmp_path / "sdk-wheel")
    sdist_path = _build_sdist(VICTOR_SDK_DIR, tmp_path / "sdk-sdist")

    wheel_required_suffixes = {
        "victor_sdk/__init__.py",
        "victor_sdk/constants/tool_names.py",
        "victor_sdk/constants/capability_ids.py",
        "victor_sdk/core/types.py",
        "victor_sdk/verticals/protocols/base.py",
        "METADATA",
    }
    sdist_required_suffixes = {
        "victor_sdk/__init__.py",
        "victor_sdk/constants/tool_names.py",
        "victor_sdk/constants/capability_ids.py",
        "victor_sdk/core/types.py",
        "victor_sdk/verticals/protocols/base.py",
        "README.md",
        "pyproject.toml",
    }

    with zipfile.ZipFile(wheel_path) as archive:
        _assert_archive_contains_suffixes(set(archive.namelist()), wheel_required_suffixes)

    with tarfile.open(sdist_path, "r:gz") as archive:
        _assert_archive_contains_suffixes(set(archive.getnames()), sdist_required_suffixes)


@pytest.mark.integration
def test_external_vertical_wheel_and_sdist_include_entry_point_and_sources(tmp_path: Path) -> None:
    """The SDK-only example should package its entry point and source files correctly."""

    wheel_path = _build_wheel(EXTERNAL_EXAMPLE_DIR, tmp_path / "example-wheel")
    sdist_path = _build_sdist(EXTERNAL_EXAMPLE_DIR, tmp_path / "example-sdist")

    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        _assert_archive_contains_suffixes(
            names,
            {
                "victor_security/__init__.py",
                "victor_security/assistant.py",
                "entry_points.txt",
                "METADATA",
            },
        )
        entry_points_name = next(name for name in names if name.endswith("entry_points.txt"))
        entry_points = archive.read(entry_points_name).decode("utf-8")
        assert "[victor.verticals]" in entry_points
        assert "security = victor_security:SecurityAssistant" in entry_points

    with tarfile.open(sdist_path, "r:gz") as archive:
        _assert_archive_contains_suffixes(
            set(archive.getnames()),
            {
                "src/victor_security/__init__.py",
                "src/victor_security/assistant.py",
                "pyproject.toml",
                "README.md",
            },
        )
