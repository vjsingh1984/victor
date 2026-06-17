# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for workspace setup utilities."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from victor.context.workspace_setup import (
    detect_project_name,
    ensure_project_importable,
)


class TestDetectProjectName:
    """Tests for project name detection."""

    def test_detect_from_pyproject(self, tmp_path):
        """Detect name from pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "my-project"\n')
        result = detect_project_name(tmp_path)
        assert result == "my_project"

    def test_detect_from_directory_name(self, tmp_path):
        """Fallback to directory name."""
        result = detect_project_name(tmp_path)
        # tmp_path name varies, just check it returns something
        assert result is not None

    def test_detect_handles_missing_files(self, tmp_path):
        """Works even with no config files."""
        result = detect_project_name(tmp_path)
        assert isinstance(result, str)


class TestEnsureProjectImportable:
    """Tests for ensure_project_importable."""

    @pytest.mark.asyncio
    async def test_already_importable_noop(self, tmp_path):
        """If project imports fine, no installation needed."""
        # 'os' is always importable
        result = await ensure_project_importable("os", tmp_path)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_build_system_skips(self, tmp_path):
        """No setup.py/pyproject.toml → skip install, return False."""
        result = await ensure_project_importable("nonexistent_pkg_xyz", tmp_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_install_with_setup_py(self, tmp_path):
        """Project with setup.py triggers pip install."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            result = await ensure_project_importable("nonexistent_pkg_xyz", tmp_path)

        # pip install was called
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0]
        assert "-m" in call_args
        assert "pip" in call_args
        assert "install" in call_args

    @pytest.mark.asyncio
    async def test_install_failure_returns_false(self, tmp_path):
        """Failed pip install returns False."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"error: bad setup"))
            mock_exec.return_value = mock_proc

            result = await ensure_project_importable("nonexistent_pkg_xyz", tmp_path)
            assert result is False
