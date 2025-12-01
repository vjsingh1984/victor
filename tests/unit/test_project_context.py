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

"""Tests for project_context module."""

import tempfile
from pathlib import Path
from victor.context.project_context import ProjectContext, CONTEXT_FILE_NAMES


class TestProjectContext:
    """Tests for ProjectContext class."""

    def test_init_defaults(self):
        """Test ProjectContext initialization with defaults."""
        pc = ProjectContext()
        assert pc.root_path == Path.cwd()
        assert pc._context_file is None
        assert pc._content is None

    def test_init_with_path(self):
        """Test ProjectContext with custom path."""
        pc = ProjectContext("/tmp")
        assert pc.root_path == Path("/tmp")

    def test_find_context_file_not_found(self):
        """Test finding context file when none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pc = ProjectContext(tmpdir)
            result = pc.find_context_file()
            assert result is None

    def test_find_context_file_victor_md(self):
        """Test finding .victor.md file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .victor.md file
            context_file = Path(tmpdir) / ".victor.md"
            context_file.write_text("# Project Context\n\nThis is a test.")

            pc = ProjectContext(tmpdir)
            result = pc.find_context_file()
            assert result is not None
            assert result.name == ".victor.md"

    def test_find_context_file_victor_md_uppercase(self):
        """Test finding VICTOR.md file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create VICTOR.md file
            context_file = Path(tmpdir) / "VICTOR.md"
            context_file.write_text("# Project Context\n\nThis is a test.")

            pc = ProjectContext(tmpdir)
            result = pc.find_context_file()
            assert result is not None
            assert result.name == "VICTOR.md"

    def test_find_context_file_priority(self):
        """Test that .victor.md takes priority over VICTOR.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create both files
            (Path(tmpdir) / ".victor.md").write_text("# Lower case")
            (Path(tmpdir) / "VICTOR.md").write_text("# Upper case")

            pc = ProjectContext(tmpdir)
            result = pc.find_context_file()
            assert result is not None
            # .victor.md should take priority
            assert result.name == ".victor.md"

    def test_load_no_context_file(self):
        """Test loading when no context file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pc = ProjectContext(tmpdir)
            result = pc.load()
            assert result is False
            assert pc._content is None

    def test_load_with_context_file(self):
        """Test loading a context file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create context file
            context_file = Path(tmpdir) / ".victor.md"
            context_file.write_text("# Test Project\n\nProject instructions here.")

            pc = ProjectContext(tmpdir)
            result = pc.load()
            assert result is True
            assert pc._content is not None
            assert "Test Project" in pc._content

    def test_content_after_load(self):
        """Test content is accessible after loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_file = Path(tmpdir) / ".victor.md"
            context_file.write_text("# Content Test\n\nSome content.")

            pc = ProjectContext(tmpdir)
            pc.load()
            # Access _content directly as no getter method
            assert pc._content is not None
            assert "Content Test" in pc._content
            assert "Some content" in pc._content

    def test_content_before_load(self):
        """Test content is None before loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pc = ProjectContext(tmpdir)
            assert pc._content is None

    def test_context_file_names_constant(self):
        """Test that CONTEXT_FILE_NAMES has expected values."""
        assert ".victor.md" in CONTEXT_FILE_NAMES
        assert "VICTOR.md" in CONTEXT_FILE_NAMES

    def test_find_context_file_in_parent_dir(self):
        """Test finding context file in parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .victor.md in parent
            parent = Path(tmpdir)
            (parent / ".victor.md").write_text("# Parent context")

            # Create subdirectory
            subdir = parent / "subdir"
            subdir.mkdir()

            pc = ProjectContext(str(subdir))
            result = pc.find_context_file()
            assert result is not None
            assert result.name == ".victor.md"
            assert result.parent == parent
