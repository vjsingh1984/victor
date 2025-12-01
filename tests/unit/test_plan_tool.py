# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for plan_tool module."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.plan_tool import plan_files, _safe_walk


class TestSafeWalk:
    """Tests for _safe_walk function."""

    def test_safe_walk_basic(self):
        """Test basic directory walk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(f"{tmpdir}/file1.py").write_text("test")
            Path(f"{tmpdir}/file2.txt").write_text("test")

            files = _safe_walk(tmpdir)
            assert len(files) == 2

    def test_safe_walk_excludes_git(self):
        """Test that .git directories are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .git dir and file inside
            git_dir = Path(f"{tmpdir}/.git")
            git_dir.mkdir()
            Path(f"{git_dir}/config").write_text("test")

            # Create normal file
            Path(f"{tmpdir}/file.py").write_text("test")

            files = _safe_walk(tmpdir)
            assert len(files) == 1
            assert ".git" not in files[0]

    def test_safe_walk_excludes_pycache(self):
        """Test that __pycache__ is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create __pycache__ dir
            pycache = Path(f"{tmpdir}/__pycache__")
            pycache.mkdir()
            Path(f"{pycache}/module.pyc").write_text("test")

            Path(f"{tmpdir}/file.py").write_text("test")

            files = _safe_walk(tmpdir)
            assert len(files) == 1
            assert "__pycache__" not in files[0]


class TestPlanFiles:
    """Tests for plan_files function."""

    @pytest.mark.asyncio
    async def test_plan_files_no_patterns(self):
        """Test plan_files without patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/file1.py").write_text("test")
            Path(f"{tmpdir}/file2.txt").write_text("test")

            result = await plan_files(root=tmpdir)
            assert result["success"] is True
            assert len(result["files"]) == 2

    @pytest.mark.asyncio
    async def test_plan_files_with_patterns(self):
        """Test plan_files with patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/test_file.py").write_text("test")
            Path(f"{tmpdir}/other.txt").write_text("test")

            result = await plan_files(root=tmpdir, patterns=["test"])
            assert result["success"] is True
            assert len(result["files"]) == 1
            assert "test_file.py" in result["files"][0]

    @pytest.mark.asyncio
    async def test_plan_files_with_limit(self):
        """Test plan_files respects limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                Path(f"{tmpdir}/file{i}.py").write_text("test")

            result = await plan_files(root=tmpdir, limit=3)
            assert result["success"] is True
            assert len(result["files"]) <= 3

    @pytest.mark.asyncio
    async def test_plan_files_limit_capped(self):
        """Test plan_files limit is capped at 10."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(15):
                Path(f"{tmpdir}/file{i}.py").write_text("test")

            result = await plan_files(root=tmpdir, limit=20)
            assert result["success"] is True
            assert len(result["files"]) <= 10

    @pytest.mark.asyncio
    async def test_plan_files_patterns_as_string(self):
        """Test plan_files with patterns as comma-separated string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/test_foo.py").write_text("test")
            Path(f"{tmpdir}/test_bar.py").write_text("test")
            Path(f"{tmpdir}/other.txt").write_text("test")

            result = await plan_files(root=tmpdir, patterns="test,foo")
            assert result["success"] is True
            # Should match files containing "test" OR "foo"

    @pytest.mark.asyncio
    async def test_plan_files_nonexistent_root(self):
        """Test plan_files with nonexistent root returns empty files."""
        result = await plan_files(root="/nonexistent/path")
        # Function returns success=True with empty files list for nonexistent paths
        assert result["success"] is True
        assert result["files"] == []
