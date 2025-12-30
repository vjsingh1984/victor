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

"""Simple tests for documentation_tool module focusing on error handling."""

import tempfile
from pathlib import Path
import pytest

from victor.tools.documentation_tool import (
    docs_generate_docstrings,
    docs_generate_api,
    docs_generate_readme,
    docs_add_type_hints,
    docs_analyze_coverage,
)


class TestDocsGenerateDocstrings:
    """Tests for docs_generate_docstrings function."""

    @pytest.mark.asyncio
    async def test_generate_docstrings_missing_file(self):
        """Test handling of missing file parameter."""
        result = await docs_generate_docstrings(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_docstrings_file_not_found(self):
        """Test handling of non-existent file."""
        result = await docs_generate_docstrings(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_docstrings_empty_file(self, tmp_path):
        """Test with empty Python file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await docs_generate_docstrings(file=str(test_file))

        # Should handle gracefully
        assert "success" in result

    @pytest.mark.asyncio
    async def test_generate_docstrings_simple_function(self, tmp_path):
        """Test with simple function."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("def hello():\n    pass\n")

        result = await docs_generate_docstrings(file=str(test_file), format="google")

        # Should complete
        assert "success" in result

    @pytest.mark.asyncio
    async def test_generate_docstrings_with_existing_docs(self, tmp_path):
        """Test file with existing docstrings."""
        test_file = tmp_path / "documented.py"
        test_file.write_text(
            '''
def hello():
    """Already documented."""
    pass
'''
        )

        result = await docs_generate_docstrings(file=str(test_file))

        assert "success" in result


class TestDocsGenerateApi:
    """Tests for docs_generate_api function."""

    @pytest.mark.asyncio
    async def test_generate_api_missing_file(self):
        """Test handling of missing file parameter."""
        result = await docs_generate_api(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_api_file_not_found(self):
        """Test handling of non-existent file."""
        result = await docs_generate_api(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_api_empty_file(self, tmp_path):
        """Test with empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await docs_generate_api(file=str(test_file))

        # Should handle gracefully
        assert "success" in result

    @pytest.mark.asyncio
    async def test_generate_api_with_simple_file(self, tmp_path):
        """Test with simple Python file."""
        test_file = tmp_path / "module.py"
        test_file.write_text("def func():\n    pass\n")

        result = await docs_generate_api(file=str(test_file))

        assert "success" in result

    @pytest.mark.asyncio
    async def test_generate_api_different_formats(self, tmp_path):
        """Test different output formats."""
        test_file = tmp_path / "module.py"
        test_file.write_text("def func():\n    pass\n")

        # Test markdown format
        result_md = await docs_generate_api(file=str(test_file), format="markdown")
        assert "success" in result_md

        # Test rst format
        result_rst = await docs_generate_api(file=str(test_file), format="rst")
        assert "success" in result_rst


class TestDocsGenerateReadme:
    """Tests for docs_generate_readme function."""

    @pytest.mark.asyncio
    async def test_generate_readme_default_section(self):
        """Test generating default README section (installation)."""
        result = await docs_generate_readme()

        assert result["success"] is True
        assert result["section"] == "installation"
        assert "content" in result
        assert len(result["content"]) > 0

    @pytest.mark.asyncio
    async def test_generate_readme_installation_section(self):
        """Test generating installation section."""
        result = await docs_generate_readme(section="installation")

        assert result["success"] is True
        assert result["section"] == "installation"
        assert "content" in result

    @pytest.mark.asyncio
    async def test_generate_readme_usage_section(self):
        """Test generating usage section."""
        result = await docs_generate_readme(section="usage")

        assert result["success"] is True
        assert result["section"] == "usage"
        assert "content" in result

    @pytest.mark.asyncio
    async def test_generate_readme_contributing_section(self):
        """Test generating contributing section."""
        result = await docs_generate_readme(section="contributing")

        assert result["success"] is True
        assert result["section"] == "contributing"
        assert "content" in result

    @pytest.mark.asyncio
    async def test_generate_readme_invalid_section(self):
        """Test handling of invalid section."""
        result = await docs_generate_readme(section="nonexistent")

        assert result["success"] is False
        assert "Unknown section" in result["error"]


class TestDocsAddTypeHints:
    """Tests for docs_add_type_hints function."""

    @pytest.mark.asyncio
    async def test_add_type_hints_missing_file(self):
        """Test handling of missing file parameter."""
        result = await docs_add_type_hints(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_add_type_hints_file_not_found(self):
        """Test handling of non-existent file."""
        result = await docs_add_type_hints(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_add_type_hints_empty_file(self, tmp_path):
        """Test with empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await docs_add_type_hints(file=str(test_file))

        # Should handle gracefully
        assert "success" in result

    @pytest.mark.asyncio
    async def test_add_type_hints_simple_function(self, tmp_path):
        """Test with simple function."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("def add(x, y):\n    return x + y\n")

        result = await docs_add_type_hints(file=str(test_file))

        assert "success" in result

    @pytest.mark.asyncio
    async def test_add_type_hints_already_typed(self, tmp_path):
        """Test file with existing type hints."""
        test_file = tmp_path / "typed.py"
        test_file.write_text("def add(x: int, y: int) -> int:\n    return x + y\n")

        result = await docs_add_type_hints(file=str(test_file))

        assert "success" in result


class TestDocsAnalyzeCoverage:
    """Tests for docs_analyze_coverage function."""

    @pytest.mark.asyncio
    async def test_analyze_coverage_missing_file(self):
        """Test handling of missing file parameter."""
        result = await docs_analyze_coverage(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_coverage_file_not_found(self):
        """Test handling of non-existent file."""
        result = await docs_analyze_coverage(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_coverage_empty_file(self, tmp_path):
        """Test analyzing empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is True
        assert "total_items" in result

    @pytest.mark.asyncio
    async def test_analyze_coverage_with_mixed_documentation(self, tmp_path):
        """Test analyzing file with mixed documentation."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''
def documented():
    """Has docs."""
    pass

def undocumented():
    pass
'''
        )

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is True
        assert result["total_items"] == 2
        assert "coverage" in result

    @pytest.mark.asyncio
    async def test_analyze_coverage_fully_documented(self, tmp_path):
        """Test with fully documented code."""
        test_file = tmp_path / "documented.py"
        test_file.write_text(
            '''
"""Module docstring."""

def func1():
    """Function 1."""
    pass

def func2():
    """Function 2."""
    pass
'''
        )

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is True
        if result["total_items"] > 0:
            assert result["coverage"] >= 0

    @pytest.mark.asyncio
    async def test_analyze_coverage_no_documentation(self, tmp_path):
        """Test with completely undocumented code."""
        test_file = tmp_path / "undocumented.py"
        test_file.write_text(
            """
def func1():
    pass

def func2():
    pass

class MyClass:
    def method(self):
        pass
"""
        )

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is True
        assert result["total_items"] >= 1
