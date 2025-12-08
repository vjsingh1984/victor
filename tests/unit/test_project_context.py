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

"""Tests for project_context module.

The project context module now uses settings-driven paths:
- Default location: .victor/init.md (from VICTOR_DIR_NAME, VICTOR_CONTEXT_FILE)
"""

import tempfile
from pathlib import Path
from victor.context.project_context import (
    ProjectContext,
    CONTEXT_FILE_PATH,
)
from victor.config.settings import VICTOR_DIR_NAME, VICTOR_CONTEXT_FILE


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

    def test_find_context_file_victor_init_md(self):
        """Test finding .victor/init.md file (settings-based path)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .victor/init.md file using settings
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            context_file = victor_dir / VICTOR_CONTEXT_FILE
            context_file.write_text("# Project Context\n\nThis is a test.")

            pc = ProjectContext(tmpdir)
            result = pc.find_context_file()
            assert result is not None
            assert result.name == VICTOR_CONTEXT_FILE
            assert result.parent.name == VICTOR_DIR_NAME

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
            # Create .victor/init.md file
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            context_file = victor_dir / VICTOR_CONTEXT_FILE
            context_file.write_text("# Test Project\n\nProject instructions here.")

            pc = ProjectContext(tmpdir)
            result = pc.load()
            assert result is True
            assert pc._content is not None
            assert "Test Project" in pc._content

    def test_content_after_load(self):
        """Test content is accessible after loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            context_file = victor_dir / VICTOR_CONTEXT_FILE
            context_file.write_text("# Content Test\n\nSome content.")

            pc = ProjectContext(tmpdir)
            pc.load()
            assert pc._content is not None
            assert "Content Test" in pc._content
            assert "Some content" in pc._content

    def test_content_before_load(self):
        """Test content is None before loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pc = ProjectContext(tmpdir)
            assert pc._content is None

    def test_context_file_path_constant(self):
        """Test that CONTEXT_FILE_PATH has expected value from settings."""
        expected_path = f"{VICTOR_DIR_NAME}/{VICTOR_CONTEXT_FILE}"
        assert CONTEXT_FILE_PATH == expected_path

    def test_find_context_file_in_parent_dir(self):
        """Test finding context file in parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .victor/init.md in parent
            parent = Path(tmpdir)
            victor_dir = parent / VICTOR_DIR_NAME
            victor_dir.mkdir()
            context_file = victor_dir / VICTOR_CONTEXT_FILE
            context_file.write_text("# Parent context")

            # Create subdirectory
            subdir = parent / "subdir"
            subdir.mkdir()

            pc = ProjectContext(str(subdir))
            result = pc.find_context_file()
            assert result is not None
            assert result.name == VICTOR_CONTEXT_FILE
            assert result.parent.name == VICTOR_DIR_NAME
            assert result.parent.parent == parent

    def test_find_context_file_stops_at_git_root(self):
        """Test that search stops at git root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            # Create git marker in parent
            (parent / ".git").mkdir()

            # Create subdir without .victor/
            subdir = parent / "subdir"
            subdir.mkdir()

            pc = ProjectContext(str(subdir))
            result = pc.find_context_file()
            # Should not find anything since we stop at .git
            assert result is None

    def test_content_property(self):
        """Test content property returns empty string when not loaded."""
        pc = ProjectContext()
        assert pc.content == ""

    def test_content_property_after_load(self):
        """Test content property returns content after load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text("# Test Content")
            pc = ProjectContext(tmpdir)
            pc.load()
            assert pc.content == "# Test Content"

    def test_context_file_property(self):
        """Test context_file property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            context_path = victor_dir / VICTOR_CONTEXT_FILE
            context_path.write_text("# Test")
            pc = ProjectContext(tmpdir)
            pc.load()
            assert pc.context_file == context_path

    def test_context_file_property_not_loaded(self):
        """Test context_file property returns None when not loaded."""
        pc = ProjectContext()
        assert pc.context_file is None


class TestParseSections:
    """Tests for section parsing."""

    def test_parse_sections_basic(self):
        """Test parsing sections from markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """# Project
Overview content here.

## Commands
Run pytest.

## Architecture
The architecture is simple.
"""
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text(content)
            pc = ProjectContext(tmpdir)
            pc.load()

            assert "overview" in pc._parsed_sections
            assert "commands" in pc._parsed_sections
            assert "architecture" in pc._parsed_sections

    def test_parse_sections_empty_content(self):
        """Test parsing empty content."""
        pc = ProjectContext()
        pc._content = None
        pc._parse_sections()
        assert pc._parsed_sections == {}

    def test_get_section(self):
        """Test get_section method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """# Project
Overview content here.

## Commands
Run pytest.
"""
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text(content)
            pc = ProjectContext(tmpdir)
            pc.load()

            assert "Run pytest" in pc.get_section("commands")
            assert pc.get_section("nonexistent") == ""

    def test_get_section_case_insensitive(self):
        """Test get_section is case insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """## Commands
Run pytest.
"""
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text(content)
            pc = ProjectContext(tmpdir)
            pc.load()

            assert "Run pytest" in pc.get_section("COMMANDS")
            assert "Run pytest" in pc.get_section("Commands")

    def test_get_section_with_spaces(self):
        """Test get_section converts spaces to underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """## Package Layout
Layout info here.
"""
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text(content)
            pc = ProjectContext(tmpdir)
            pc.load()

            assert "Layout info" in pc.get_section("package layout")
            assert "Layout info" in pc.get_section("package_layout")


class TestGetSystemPromptAddition:
    """Tests for get_system_prompt_addition method."""

    def test_system_prompt_addition_no_content(self):
        """Test system prompt addition when no content."""
        pc = ProjectContext()
        assert pc.get_system_prompt_addition() == ""

    def test_system_prompt_addition_with_content(self):
        """Test system prompt addition with content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text("# My Project\nSome instructions.")
            pc = ProjectContext(tmpdir)
            pc.load()

            result = pc.get_system_prompt_addition()
            assert "<project-context>" in result
            assert "</project-context>" in result
            assert "My Project" in result
            # The filename in the system prompt should be VICTOR_CONTEXT_FILE
            assert VICTOR_CONTEXT_FILE in result


class TestGetPackageLayoutHint:
    """Tests for get_package_layout_hint method."""

    def test_package_layout_hint_not_found(self):
        """Test when no layout hint is found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text("# Project\nNo layout info.")
            pc = ProjectContext(tmpdir)
            pc.load()
            assert pc.get_package_layout_hint() == ""

    def test_package_layout_hint_found(self):
        """Test finding package_layout section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """# Project
## Package Layout
- victor/
- tests/
"""
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text(content)
            pc = ProjectContext(tmpdir)
            pc.load()
            result = pc.get_package_layout_hint()
            assert "victor/" in result

    def test_directory_structure_hint(self):
        """Test finding directory_structure section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """## Directory Structure
src/ contains source code.
"""
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text(content)
            pc = ProjectContext(tmpdir)
            pc.load()
            result = pc.get_package_layout_hint()
            assert "src/" in result

    def test_architecture_hint(self):
        """Test finding architecture section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = """## Architecture
MVC pattern.
"""
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            (victor_dir / VICTOR_CONTEXT_FILE).write_text(content)
            pc = ProjectContext(tmpdir)
            pc.load()
            result = pc.get_package_layout_hint()
            assert "MVC" in result


class TestGenerateVictorMd:
    """Tests for generate_victor_md function."""

    def test_generate_basic(self):
        """Test basic generation."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_victor_md(tmpdir)
            # Should generate with the context file name from settings
            assert f"# {VICTOR_CONTEXT_FILE}" in result
            assert "## Project Overview" in result
            assert "## Package Layout" in result
            assert "## Common Commands" in result

    def test_generate_with_readme(self):
        """Test generation with README.md."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create README with badges first, then content
            readme_content = """# My Project

[![License](https://badge.svg)](url)

This is my awesome project that does cool things.
"""
            (Path(tmpdir) / "README.md").write_text(readme_content)
            result = generate_victor_md(tmpdir)
            assert "awesome project" in result

    def test_generate_with_pyproject(self):
        """Test generation with pyproject.toml."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text("[project]")
            result = generate_victor_md(tmpdir)
            assert "pip install" in result
            assert "pytest" in result
            assert "black" in result

    def test_generate_with_setup_py(self):
        """Test generation with setup.py."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "setup.py").write_text("from setuptools import setup")
            result = generate_victor_md(tmpdir)
            assert "pip install -e" in result

    def test_generate_with_package_json(self):
        """Test generation with package.json."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text("{}")
            result = generate_victor_md(tmpdir)
            assert "npm install" in result
            assert "npm test" in result

    def test_generate_with_cargo_toml(self):
        """Test generation with Cargo.toml."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Cargo.toml").write_text("[package]")
            result = generate_victor_md(tmpdir)
            assert "cargo build" in result
            assert "cargo test" in result

    def test_generate_with_go_mod(self):
        """Test generation with go.mod."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "go.mod").write_text("module example")
            result = generate_victor_md(tmpdir)
            assert "go build" in result
            assert "go test" in result

    def test_generate_with_python_package(self):
        """Test generation with Python package directory."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir) / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")
            result = generate_victor_md(tmpdir)
            assert "mypackage" in result

    def test_generate_with_src_layout(self):
        """Test generation with src layout."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            pkg_dir = src_dir / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")
            result = generate_victor_md(tmpdir)
            assert "src/" in result

    def test_generate_with_tests_dir(self):
        """Test generation with tests directory."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "tests").mkdir()
            result = generate_victor_md(tmpdir)
            assert "tests/" in result

    def test_generate_with_docs_dir(self):
        """Test generation with docs directory."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "docs").mkdir()
            result = generate_victor_md(tmpdir)
            assert "docs/" in result

    def test_generate_default_cwd(self):
        """Test generation with default cwd."""
        from victor.context.project_context import generate_victor_md

        result = generate_victor_md()
        assert f"# {VICTOR_CONTEXT_FILE}" in result

    def test_generate_no_readme_content(self):
        """Test generation when README exists but has no usable content."""
        from victor.context.project_context import generate_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            # README with only headers and images
            (Path(tmpdir) / "README.md").write_text("# Title\n\n![image](url)")
            result = generate_victor_md(tmpdir)
            assert "[Add project description here]" in result


class TestInitVictorMd:
    """Tests for init_victor_md function."""

    def test_init_creates_file(self):
        """Test that init creates .victor/init.md file."""
        from victor.context.project_context import init_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            result = init_victor_md(tmpdir)
            assert result is not None
            assert result.exists()
            assert result.name == VICTOR_CONTEXT_FILE
            assert result.parent.name == VICTOR_DIR_NAME

    def test_init_does_not_overwrite(self):
        """Test that init doesn't overwrite existing file."""
        from victor.context.project_context import init_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            existing = victor_dir / VICTOR_CONTEXT_FILE
            existing.write_text("Existing content")

            result = init_victor_md(tmpdir)
            assert result is None
            # Original content should be preserved
            assert existing.read_text() == "Existing content"

    def test_init_force_overwrite(self):
        """Test that init with force overwrites existing file."""
        from victor.context.project_context import init_victor_md

        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            existing = victor_dir / VICTOR_CONTEXT_FILE
            existing.write_text("Existing content")

            result = init_victor_md(tmpdir, force=True)
            assert result is not None
            # Content should be replaced
            assert f"# {VICTOR_CONTEXT_FILE}" in result.read_text()

    def test_init_default_cwd(self):
        """Test init with default cwd."""
        from victor.context.project_context import init_victor_md
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = init_victor_md()
                assert result is not None
                assert result.exists()
            finally:
                os.chdir(old_cwd)

    def test_init_write_error(self):
        """Test init handles write errors."""
        from victor.context.project_context import init_victor_md
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, "write_text", side_effect=PermissionError("No access")):
                result = init_victor_md(tmpdir)
                assert result is None


class TestLoadError:
    """Tests for error handling during load."""

    def test_load_read_error(self):
        """Test load handles read errors."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / VICTOR_DIR_NAME
            victor_dir.mkdir()
            context_file = victor_dir / VICTOR_CONTEXT_FILE
            context_file.write_text("content")

            pc = ProjectContext(tmpdir)
            with patch.object(Path, "read_text", side_effect=PermissionError("No access")):
                result = pc.load()
                assert result is False
