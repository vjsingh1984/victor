"""Regression tests for unit-test working-directory isolation."""

from __future__ import annotations

import os
from pathlib import Path


def test_deleted_working_directory_can_exist_during_test(tmp_path) -> None:
    """Simulate a test leaving the process in a deleted working directory."""
    doomed = tmp_path / "doomed-cwd"
    doomed.mkdir()
    os.chdir(doomed)
    doomed.rmdir()

    try:
        Path.cwd()
    except FileNotFoundError:
        return

    raise AssertionError(
        "expected deleted cwd to raise FileNotFoundError inside the test"
    )


def test_working_directory_is_restored_after_deleted_cwd() -> None:
    """The autouse fixture should recover to a valid directory before the next test."""
    cwd = Path.cwd()
    repo_root = Path(__file__).resolve().parents[2]

    assert cwd.exists()
    assert cwd == repo_root
