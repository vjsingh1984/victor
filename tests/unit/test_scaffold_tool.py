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

"""Tests for scaffold_tool module."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.scaffold_tool import scaffold, TEMPLATES


class TestScaffold:
    """Tests for scaffold function."""

    @pytest.mark.asyncio
    async def test_scaffold_list_templates(self):
        """Test listing available templates."""
        result = await scaffold(operation="list")
        assert result["success"] is True
        assert "templates" in result
        assert len(result["templates"]) > 0

    @pytest.mark.asyncio
    async def test_scaffold_invalid_operation(self):
        """Test invalid operation."""
        result = await scaffold(operation="invalid_op")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

    @pytest.mark.asyncio
    async def test_scaffold_create_requires_template(self):
        """Test create requires template."""
        result = await scaffold(operation="create")
        assert result["success"] is False
        # Should mention template or name

    @pytest.mark.asyncio
    async def test_scaffold_create_invalid_template(self):
        """Test create with invalid template."""
        result = await scaffold(operation="create", template="nonexistent_template")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_scaffold_create_fastapi(self):
        """Test creating FastAPI project."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory for project creation
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = await scaffold(operation="create", template="fastapi", name="test_project")
                assert result["success"] is True
                # Check project directory was created
                project_dir = Path(tmpdir) / "test_project"
                assert project_dir.exists()
            finally:
                os.chdir(old_cwd)

    @pytest.mark.asyncio
    async def test_scaffold_create_requires_name(self):
        """Test creating project requires name."""
        result = await scaffold(operation="create", template="fastapi")
        assert result["success"] is False
        assert "name" in result["error"].lower()


class TestTemplates:
    """Tests for TEMPLATES constant."""

    def test_templates_has_fastapi(self):
        """Test TEMPLATES includes FastAPI."""
        assert "fastapi" in TEMPLATES
        assert "name" in TEMPLATES["fastapi"]
        assert "files" in TEMPLATES["fastapi"]

    def test_templates_has_flask(self):
        """Test TEMPLATES includes Flask."""
        assert "flask" in TEMPLATES

    def test_templates_has_python_cli(self):
        """Test TEMPLATES includes python-cli."""
        assert "python-cli" in TEMPLATES

    def test_templates_have_required_fields(self):
        """Test all templates have required fields."""
        for name, template in TEMPLATES.items():
            assert "name" in template, f"{name} missing 'name'"
            assert "description" in template, f"{name} missing 'description'"
            assert "files" in template, f"{name} missing 'files'"
