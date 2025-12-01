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

"""Tests for code_search_tool module."""

import tempfile
from pathlib import Path

from victor.tools.code_search_tool import (
    _latest_mtime,
    _gather_files,
    _keyword_score,
)


class TestLatestMtime:
    """Tests for _latest_mtime function."""

    def test_latest_mtime(self):
        """Test getting latest modification time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/file1.py").write_text("test")
            mtime = _latest_mtime(Path(tmpdir))
            assert mtime > 0

    def test_latest_mtime_empty_dir(self):
        """Test latest mtime on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mtime = _latest_mtime(Path(tmpdir))
            assert mtime == 0.0


class TestGatherFiles:
    """Tests for _gather_files function."""

    def test_gather_files_default_extensions(self):
        """Test gathering files with default extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/file.py").write_text("test")
            Path(f"{tmpdir}/file.txt").write_text("test")
            Path(f"{tmpdir}/file.xyz").write_text("test")

            files = _gather_files(tmpdir, exts=None, max_files=100)
            # Should include .py and .txt but not .xyz
            assert any(f.endswith(".py") for f in files)
            assert any(f.endswith(".txt") for f in files)
            assert not any(f.endswith(".xyz") for f in files)

    def test_gather_files_custom_extensions(self):
        """Test gathering files with custom extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(f"{tmpdir}/file.py").write_text("test")
            Path(f"{tmpdir}/file.js").write_text("test")

            files = _gather_files(tmpdir, exts=[".js"], max_files=100)
            assert len(files) == 1
            assert files[0].endswith(".js")

    def test_gather_files_max_limit(self):
        """Test gathering files respects max limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(20):
                Path(f"{tmpdir}/file{i}.py").write_text("test")

            files = _gather_files(tmpdir, exts=None, max_files=5)
            assert len(files) == 5

    def test_gather_files_excludes_git(self):
        """Test that .git directories are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = Path(f"{tmpdir}/.git")
            git_dir.mkdir()
            Path(f"{git_dir}/config.py").write_text("test")
            Path(f"{tmpdir}/file.py").write_text("test")

            files = _gather_files(tmpdir, exts=None, max_files=100)
            assert len(files) == 1
            assert ".git" not in files[0]


class TestKeywordScore:
    """Tests for _keyword_score function."""

    def test_single_keyword(self):
        """Test scoring with single keyword."""
        score = _keyword_score("hello world hello", "hello")
        assert score == 2  # "hello" appears twice

    def test_multiple_keywords(self):
        """Test scoring with multiple keywords."""
        score = _keyword_score("hello world foo bar", "hello world")
        assert score >= 2  # Each keyword counted

    def test_case_insensitive(self):
        """Test scoring is case insensitive."""
        score = _keyword_score("HELLO World", "hello")
        assert score == 1

    def test_no_match(self):
        """Test scoring with no matches."""
        score = _keyword_score("foo bar", "xyz")
        assert score == 0
