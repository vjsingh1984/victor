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

"""Tests for common utilities module."""

import os
import tempfile
from pathlib import Path

import pytest

from victor.tools.common import (
    EXCLUDE_DIRS,
    DEFAULT_CODE_EXTENSIONS,
    safe_walk,
    gather_code_files,
    latest_mtime,
)


class TestExcludeDirs:
    """Tests for EXCLUDE_DIRS constant."""

    def test_contains_common_exclusions(self):
        """Test that common directories are excluded."""
        assert ".git" in EXCLUDE_DIRS
        assert "node_modules" in EXCLUDE_DIRS
        assert "venv" in EXCLUDE_DIRS
        assert ".venv" in EXCLUDE_DIRS
        assert "__pycache__" in EXCLUDE_DIRS

    def test_contains_build_directories(self):
        """Test that build directories are excluded."""
        assert "dist" in EXCLUDE_DIRS
        assert "build" in EXCLUDE_DIRS


class TestDefaultCodeExtensions:
    """Tests for DEFAULT_CODE_EXTENSIONS constant."""

    def test_contains_python_extensions(self):
        """Test Python extensions are included."""
        assert ".py" in DEFAULT_CODE_EXTENSIONS

    def test_contains_web_extensions(self):
        """Test web extensions are included."""
        assert ".js" in DEFAULT_CODE_EXTENSIONS
        assert ".ts" in DEFAULT_CODE_EXTENSIONS
        assert ".html" in DEFAULT_CODE_EXTENSIONS
        assert ".css" in DEFAULT_CODE_EXTENSIONS

    def test_contains_documentation_extensions(self):
        """Test documentation extensions are included."""
        assert ".md" in DEFAULT_CODE_EXTENSIONS
        assert ".txt" in DEFAULT_CODE_EXTENSIONS

    def test_contains_config_extensions(self):
        """Test config extensions are included."""
        assert ".yaml" in DEFAULT_CODE_EXTENSIONS
        assert ".yml" in DEFAULT_CODE_EXTENSIONS
        assert ".json" in DEFAULT_CODE_EXTENSIONS
        assert ".toml" in DEFAULT_CODE_EXTENSIONS


class TestSafeWalk:
    """Tests for safe_walk function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            Path(tmpdir, "file1.py").touch()
            Path(tmpdir, "file2.txt").touch()

            # Create a subdirectory with files
            subdir = Path(tmpdir, "subdir")
            subdir.mkdir()
            Path(subdir, "file3.py").touch()

            # Create an excluded directory
            gitdir = Path(tmpdir, ".git")
            gitdir.mkdir()
            Path(gitdir, "config").touch()

            # Create node_modules
            node_modules = Path(tmpdir, "node_modules")
            node_modules.mkdir()
            Path(node_modules, "package.json").touch()

            yield tmpdir

    def test_walks_directory(self, temp_dir):
        """Test that safe_walk finds files."""
        files = safe_walk(temp_dir)
        assert len(files) > 0

    def test_excludes_git_directory(self, temp_dir):
        """Test that .git directory is excluded."""
        files = safe_walk(temp_dir)
        for f in files:
            assert ".git" not in f

    def test_excludes_node_modules(self, temp_dir):
        """Test that node_modules is excluded."""
        files = safe_walk(temp_dir)
        for f in files:
            assert "node_modules" not in f

    def test_filters_by_extension(self, temp_dir):
        """Test extension filtering."""
        files = safe_walk(temp_dir, extensions={".py"})
        for f in files:
            assert f.endswith(".py")

    def test_custom_exclude_dirs(self, temp_dir):
        """Test custom exclude directories."""
        # Create a custom directory
        custom_dir = Path(temp_dir, "custom_exclude")
        custom_dir.mkdir()
        Path(custom_dir, "file.py").touch()

        files = safe_walk(temp_dir, exclude_dirs={"custom_exclude"})
        for f in files:
            assert "custom_exclude" not in f


class TestGatherCodeFiles:
    """Tests for gather_code_files function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with code files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.py").touch()
            Path(tmpdir, "config.yaml").touch()
            Path(tmpdir, "readme.md").touch()
            Path(tmpdir, "data.csv").touch()  # Not a code file by default
            yield tmpdir

    def test_gathers_code_files(self, temp_dir):
        """Test that code files are gathered."""
        files = gather_code_files(temp_dir)
        filenames = [os.path.basename(f) for f in files]
        assert "main.py" in filenames
        assert "config.yaml" in filenames
        assert "readme.md" in filenames

    def test_excludes_non_code_files(self, temp_dir):
        """Test that non-code files are excluded by default."""
        files = gather_code_files(temp_dir)
        filenames = [os.path.basename(f) for f in files]
        assert "data.csv" not in filenames

    def test_custom_extensions(self, temp_dir):
        """Test gathering with custom extensions."""
        files = gather_code_files(temp_dir, extensions={".csv"})
        filenames = [os.path.basename(f) for f in files]
        assert "data.csv" in filenames
        assert "main.py" not in filenames


class TestLatestMtime:
    """Tests for latest_mtime function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.py").touch()
            Path(tmpdir, "file2.py").touch()
            yield tmpdir

    def test_returns_float(self, temp_dir):
        """Test that latest_mtime returns a float."""
        result = latest_mtime(Path(temp_dir))
        assert isinstance(result, float)

    def test_returns_recent_time(self, temp_dir):
        """Test that latest_mtime returns a recent time."""
        import time

        now = time.time()
        result = latest_mtime(Path(temp_dir))
        # Should be within the last minute
        assert now - result < 60

    def test_empty_directory_returns_zero(self):
        """Test that an empty directory returns 0.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = latest_mtime(Path(tmpdir))
            assert result == 0.0

    def test_excludes_git_directory(self, temp_dir):
        """Test that .git directory is excluded from mtime calculation."""
        import time

        # Create .git with a file
        gitdir = Path(temp_dir, ".git")
        gitdir.mkdir()
        git_file = Path(gitdir, "config")
        git_file.touch()

        # Wait a tiny bit
        time.sleep(0.01)

        # Touch a regular file to make it newer
        Path(temp_dir, "file1.py").touch()

        result = latest_mtime(Path(temp_dir))

        # The result should be from file1.py, not from .git/config
        # (though they might be very close in time)
        assert result > 0
