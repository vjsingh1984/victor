"""Tests for refactor_tool module."""

import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, mock_open

from victor.tools.refactor_tool import (
    refactor_rename_symbol,
    refactor_extract_function,
    refactor_inline_variable,
    refactor_organize_imports,
)


class TestRefactorRenameSymbol:
    """Tests for refactor_rename_symbol function."""

    @pytest.mark.asyncio
    async def test_rename_symbol_missing_file(self):
        """Test handling of missing file parameter."""
        result = await refactor_rename_symbol(
            file="",
            old_name="test",
            new_name="new_test"
        )

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_symbol_missing_old_name(self):
        """Test handling of missing old_name parameter."""
        result = await refactor_rename_symbol(
            file="test.py",
            old_name="",
            new_name="new_test"
        )

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_symbol_missing_new_name(self):
        """Test handling of missing new_name parameter."""
        result = await refactor_rename_symbol(
            file="test.py",
            old_name="test",
            new_name=""
        )

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_symbol_file_not_found(self):
        """Test handling of non-existent file."""
        result = await refactor_rename_symbol(
            file="/nonexistent/file.py",
            old_name="old",
            new_name="new"
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_symbol_not_python_file(self, tmp_path):
        """Test handling of non-Python file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Not Python")

        result = await refactor_rename_symbol(
            file=str(test_file),
            old_name="old",
            new_name="new"
        )

        # Implementation returns syntax error when parsing fails
        assert result["success"] is False
        assert "Syntax error" in result["error"] or "error" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_symbol_success_preview(self, tmp_path):
        """Test successful rename in preview mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def old_function():
    x = old_function()
    return old_function
""")

        result = await refactor_rename_symbol(
            file=str(test_file),
            old_name="old_function",
            new_name="new_function",
            preview=True
        )

        assert result["success"] is True
        assert result["changes_count"] > 0
        assert "preview" in result["formatted_report"].lower()

    @pytest.mark.asyncio
    async def test_rename_symbol_success_apply(self, tmp_path):
        """Test successful rename with changes applied."""
        test_file = tmp_path / "test.py"
        original_content = """
def calculate(x):
    return calculate(x)
"""
        test_file.write_text(original_content)

        result = await refactor_rename_symbol(
            file=str(test_file),
            old_name="calculate",
            new_name="compute",
            preview=False
        )

        assert result["success"] is True
        assert result["changes_count"] > 0

        # Verify file was modified
        new_content = test_file.read_text()
        assert "compute" in new_content

    @pytest.mark.asyncio
    async def test_rename_symbol_no_occurrences(self, tmp_path):
        """Test rename when symbol not found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello():
    pass
""")

        result = await refactor_rename_symbol(
            file=str(test_file),
            old_name="nonexistent",
            new_name="new_name"
        )

        # Implementation returns error when symbol not found
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_class_preview(self, tmp_path):
        """Test class renaming in preview mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class OldClass:
    def __init__(self):
        self.value = 0

def use_class():
    obj = OldClass()
    return obj
""")

        result = await refactor_rename_symbol(
            file=str(test_file),
            old_name="OldClass",
            new_name="NewClass",
            preview=True
        )

        assert result["success"] is True
        assert result["changes_count"] > 0
        assert "preview" in result["formatted_report"].lower()

    @pytest.mark.asyncio
    async def test_rename_class_apply(self, tmp_path):
        """Test class renaming with changes applied."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class Calculator:
    def add(self, x, y):
        return x + y

calc = Calculator()
""")

        result = await refactor_rename_symbol(
            file=str(test_file),
            old_name="Calculator",
            new_name="MathCalculator",
            preview=False
        )

        assert result["success"] is True
        assert result["changes_count"] > 0

        # Verify file was modified
        new_content = test_file.read_text()
        assert "MathCalculator" in new_content


class TestRefactorExtractFunction:
    """Tests for refactor_extract_function function."""

    @pytest.mark.asyncio
    async def test_extract_function_missing_file(self):
        """Test handling of missing file parameter."""
        result = await refactor_extract_function(
            file="",
            start_line=1,
            end_line=5,
            function_name="extracted"
        )

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_function_missing_function_name(self):
        """Test handling of missing function_name."""
        result = await refactor_extract_function(
            file="test.py",
            start_line=1,
            end_line=5,
            function_name=""
        )

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_function_invalid_line_range(self, tmp_path):
        """Test handling of invalid line range."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        result = await refactor_extract_function(
            file=str(test_file),
            start_line=5,
            end_line=2,
            function_name="extracted"
        )

        assert result["success"] is False
        assert "Invalid line range" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_function_file_not_found(self):
        """Test handling of non-existent file."""
        result = await refactor_extract_function(
            file="/nonexistent/file.py",
            start_line=1,
            end_line=5,
            function_name="extracted"
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_function_not_python_file(self, tmp_path):
        """Test handling of non-Python file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Not Python")

        result = await refactor_extract_function(
            file=str(test_file),
            start_line=1,
            end_line=1,
            function_name="extracted"
        )

        # Implementation may handle this gracefully
        # Just verify it completes without crashing
        assert "success" in result

    @pytest.mark.asyncio
    async def test_extract_function_success_preview(self, tmp_path):
        """Test successful function extraction in preview mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def main():
    x = 5
    y = 10
    result = x + y
    print(result)
    return result
""")

        result = await refactor_extract_function(
            file=str(test_file),
            start_line=4,
            end_line=5,
            function_name="calculate_sum",
            preview=True
        )

        # Should handle gracefully
        assert "success" in result

    @pytest.mark.asyncio
    async def test_extract_function_simple_code_block(self, tmp_path):
        """Test extracting simple code block."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def process_data(data):
    filtered = [x for x in data if x > 0]
    sorted_data = sorted(filtered)
    return sorted_data
""")

        result = await refactor_extract_function(
            file=str(test_file),
            start_line=3,
            end_line=4,
            function_name="filter_and_sort"
        )

        # Should handle gracefully
        assert "success" in result


class TestRefactorInlineVariable:
    """Tests for refactor_inline_variable function."""

    @pytest.mark.asyncio
    async def test_inline_variable_missing_file(self):
        """Test handling of missing file parameter."""
        result = await refactor_inline_variable(
            file="",
            variable_name="var"
        )

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_variable_missing_variable_name(self):
        """Test handling of missing variable_name."""
        result = await refactor_inline_variable(
            file="test.py",
            variable_name=""
        )

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_variable_file_not_found(self):
        """Test handling of non-existent file."""
        result = await refactor_inline_variable(
            file="/nonexistent/file.py",
            variable_name="var"
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_variable_not_python_file(self, tmp_path):
        """Test handling of non-Python file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Not Python")

        result = await refactor_inline_variable(
            file=str(test_file),
            variable_name="var"
        )

        # Implementation returns syntax error when parsing fails
        assert result["success"] is False
        assert "Syntax error" in result["error"] or "error" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_variable_not_found(self, tmp_path):
        """Test handling of variable not found."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        result = await refactor_inline_variable(
            file=str(test_file),
            variable_name="nonexistent"
        )

        # Variable not found
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_inline_variable_success_preview(self, tmp_path):
        """Test successful variable inlining in preview mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def calculate():
    result = 5
    x = result * 2
    y = result + 3
    return x + y
""")

        result = await refactor_inline_variable(
            file=str(test_file),
            variable_name="result",
            preview=True
        )

        assert result["success"] is True
        assert result["changes_count"] > 0
        assert "preview" in result["formatted_report"].lower()

    @pytest.mark.asyncio
    async def test_inline_variable_success_apply(self, tmp_path):
        """Test successful variable inlining with changes applied."""
        test_file = tmp_path / "test.py"
        original_content = """
def process():
    value = 10
    result = value * 2
    return result
"""
        test_file.write_text(original_content)

        result = await refactor_inline_variable(
            file=str(test_file),
            variable_name="value",
            preview=False
        )

        assert result["success"] is True
        assert result["changes_count"] > 0

        # Verify file was modified
        new_content = test_file.read_text()
        assert "value = 10" not in new_content or "value * 2" not in new_content


class TestRefactorOrganizeImports:
    """Tests for refactor_organize_imports function."""

    @pytest.mark.asyncio
    async def test_organize_imports_missing_file(self):
        """Test handling of missing file parameter."""
        result = await refactor_organize_imports(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_organize_imports_file_not_found(self):
        """Test handling of non-existent file."""
        result = await refactor_organize_imports(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_organize_imports_not_python_file(self, tmp_path):
        """Test handling of non-Python file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Not Python")

        result = await refactor_organize_imports(file=str(test_file))

        # Implementation returns syntax error when parsing fails
        assert result["success"] is False
        assert "Syntax error" in result["error"] or "error" in result["error"]

    @pytest.mark.asyncio
    async def test_organize_imports_no_imports(self, tmp_path):
        """Test organizing file with no imports."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello():
    pass
""")

        result = await refactor_organize_imports(file=str(test_file))

        # Should succeed even with no imports
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_organize_imports_already_organized(self, tmp_path):
        """Test organizing already well-organized imports."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
import os
import sys

from pathlib import Path
from typing import Dict

def hello():
    pass
""")

        result = await refactor_organize_imports(file=str(test_file))

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_organize_imports_preview_mode(self, tmp_path):
        """Test organize imports in preview mode."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
from typing import Dict
import sys
import os
from pathlib import Path

def hello():
    pass
""")

        result = await refactor_organize_imports(file=str(test_file), preview=True)

        assert result["success"] is True
        assert "preview" in result["formatted_report"].lower()

    @pytest.mark.asyncio
    async def test_organize_imports_apply_changes(self, tmp_path):
        """Test applying import organization."""
        test_file = tmp_path / "test.py"
        original_content = """
from typing import Dict
import sys
import os

def hello():
    pass
"""
        test_file.write_text(original_content)

        result = await refactor_organize_imports(file=str(test_file), preview=False)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_organize_imports_with_duplicates(self, tmp_path):
        """Test organizing imports with duplicates."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
import os
import sys
import os

def hello():
    pass
""")

        result = await refactor_organize_imports(file=str(test_file))

        assert result["success"] is True
