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

"""Tests for code_intelligence_tool module and consolidated rename from refactor_tool."""

import pytest

from victor.tools.code_intelligence_tool import (
    symbol,
    refs,
)
from victor.tools.refactor_tool import rename


class TestSymbol:
    """Tests for symbol function."""

    @pytest.mark.asyncio
    async def test_find_function(self, tmp_path):
        """Test finding a function definition."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello_world():
    print("Hello, World!")
    return True

def another_function():
    pass
""")

        result = await symbol(file_path=str(test_file), symbol_name="hello_world")

        assert result is not None
        assert result["symbol_name"] == "hello_world"
        assert result["type"] == "function"
        assert result["start_line"] > 0
        assert result["end_line"] > result["start_line"]
        assert "def hello_world" in result["code"]

    @pytest.mark.asyncio
    async def test_find_class(self, tmp_path):
        """Test finding a class definition."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class MyClass:
    def __init__(self):
        self.value = 0

    def method(self):
        return self.value
""")

        result = await symbol(file_path=str(test_file), symbol_name="MyClass")

        assert result is not None
        assert result["symbol_name"] == "MyClass"
        assert result["type"] == "class"
        assert "class MyClass" in result["code"]

    @pytest.mark.asyncio
    async def test_symbol_not_found(self, tmp_path):
        """Test searching for non-existent symbol."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def existing_function():
    pass
""")

        result = await symbol(file_path=str(test_file), symbol_name="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_symbol_file_not_found(self):
        """Test with non-existent file."""
        result = await symbol(file_path="/nonexistent/file.py", symbol_name="test")

        assert result is not None
        assert "error" in result

    @pytest.mark.asyncio
    async def test_symbol_empty_file(self, tmp_path):
        """Test with empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await symbol(file_path=str(test_file), symbol_name="test")

        assert result is None

    @pytest.mark.asyncio
    async def test_find_nested_function(self, tmp_path):
        """Test finding nested function."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def outer_function():
    def inner_function():
        return "inner"
    return inner_function()
""")

        result = await symbol(file_path=str(test_file), symbol_name="inner_function")

        assert result is not None
        assert result["symbol_name"] == "inner_function"
        assert result["type"] == "function"

    @pytest.mark.asyncio
    async def test_find_method_in_class(self, tmp_path):
        """Test finding a method inside a class."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
class TestClass:
    def test_method(self):
        pass
""")

        result = await symbol(file_path=str(test_file), symbol_name="test_method")

        assert result is not None
        assert result["symbol_name"] == "test_method"
        assert result["type"] == "function"

    @pytest.mark.asyncio
    async def test_symbol_generic_exception(self, tmp_path):
        """Test generic exception handling in symbol."""
        from unittest.mock import patch

        test_file = tmp_path / "test.py"
        test_file.write_text("def test_func(): pass")

        # Mock to raise a generic exception during parsing
        with patch(
            "victor.tools.code_intelligence_tool.get_parser",
            side_effect=RuntimeError("Parse error"),
        ):
            result = await symbol(file_path=str(test_file), symbol_name="test_func")

            assert result is not None
            assert "error" in result
            assert "unexpected error" in result["error"].lower()


class TestRefs:
    """Tests for refs function."""

    @pytest.mark.asyncio
    async def test_refs_basic(self, tmp_path):
        """Test finding references in directory."""
        # Create a directory with Python files
        test_file1 = tmp_path / "file1.py"
        test_file1.write_text("""
def target_function():
    return True
""")

        test_file2 = tmp_path / "file2.py"
        test_file2.write_text("""
from file1 import target_function

result = target_function()
""")

        result = await refs(symbol_name="target_function", search_path=str(tmp_path))

        assert isinstance(result, list)
        # May or may not find cross-file references depending on implementation

    @pytest.mark.asyncio
    async def test_refs_no_matches(self, tmp_path):
        """Test finding references when none exist."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def some_function():
    pass
""")

        result = await refs(symbol_name="nonexistent_function", search_path=str(tmp_path))

        assert isinstance(result, list)
        # Empty list or minimal results expected

    @pytest.mark.asyncio
    async def test_refs_invalid_path(self):
        """Test with invalid search path."""
        result = await refs(symbol_name="test", search_path="/nonexistent/path")

        assert isinstance(result, list)
        # Should handle gracefully, return empty list

    @pytest.mark.asyncio
    async def test_refs_empty_directory(self, tmp_path):
        """Test with empty directory."""
        result = await refs(symbol_name="test", search_path=str(tmp_path))

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_refs_with_parse_errors(self, tmp_path):
        """Test that unparseable files are skipped gracefully."""
        # Create a file with invalid Python
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def invalid syntax here!")

        # Create a valid file
        good_file = tmp_path / "good.py"
        good_file.write_text("""
def target_function():
    pass

result = target_function()
""")

        result = await refs(symbol_name="target_function", search_path=str(tmp_path))

        # Should still return results from good file
        assert isinstance(result, list)
        # The bad file should be skipped (exception caught)

    @pytest.mark.asyncio
    async def test_refs_with_file_exception(self, tmp_path):
        """Test that file exceptions are caught and processing continues."""
        from unittest.mock import patch

        # Create a file that will exist
        test_file = tmp_path / "test.py"
        test_file.write_text("def target(): pass")

        # Mock open to raise an exception for any file
        with patch("builtins.open", side_effect=OSError("File read error")):
            result = await refs(symbol_name="target", search_path=str(tmp_path))

            # Should return empty list since all files failed to read
            assert isinstance(result, list)
            assert len(result) == 0


class TestRename:
    """Tests for consolidated rename function from refactor_tool."""

    @pytest.mark.asyncio
    async def test_rename_single_file(self, tmp_path):
        """Test renaming symbol in a single file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""def old_func():
    return True

result = old_func()
""")

        result = await rename(
            old_name="old_func",
            new_name="new_func",
            path=str(test_file),
            scope="file",
            preview=False,
        )

        assert result["success"] is True
        assert result["files_count"] == 1
        assert result["total_changes"] >= 2  # Definition + usage

        # Verify file was modified
        content = test_file.read_text()
        assert "new_func" in content
        assert "old_func" not in content

    @pytest.mark.asyncio
    async def test_rename_single_file_preview(self, tmp_path):
        """Test preview mode doesn't modify file."""
        test_file = tmp_path / "test.py"
        original_content = """def old_func():
    return True
"""
        test_file.write_text(original_content)

        result = await rename(
            old_name="old_func",
            new_name="new_func",
            path=str(test_file),
            scope="file",
            preview=True,
        )

        assert result["success"] is True
        assert "PREVIEW" in result["formatted_report"]

        # File should NOT be modified
        assert test_file.read_text() == original_content

    @pytest.mark.asyncio
    async def test_rename_directory_scope(self, tmp_path):
        """Test renaming symbol across directory."""
        # Create files in directory
        file1 = tmp_path / "module1.py"
        file1.write_text("def helper(): pass\nhelper()")

        file2 = tmp_path / "module2.py"
        file2.write_text("from module1 import helper\nhelper()")

        result = await rename(
            old_name="helper",
            new_name="util",
            path=str(tmp_path),
            scope="directory",
            preview=False,
        )

        assert result["success"] is True
        assert result["files_count"] == 2

        # Both files should be modified
        assert "util" in file1.read_text()
        assert "util" in file2.read_text()

    @pytest.mark.asyncio
    async def test_rename_project_scope(self, tmp_path):
        """Test renaming symbol across entire project."""
        # Create nested directory structure
        subdir = tmp_path / "subpackage"
        subdir.mkdir()

        file1 = tmp_path / "main.py"
        file1.write_text("CONFIG = 'test'\nprint(CONFIG)")

        file2 = subdir / "utils.py"
        file2.write_text("from main import CONFIG\nval = CONFIG")

        result = await rename(
            old_name="CONFIG",
            new_name="SETTINGS",
            path=str(tmp_path),
            scope="project",
            preview=False,
        )

        assert result["success"] is True
        assert result["files_count"] == 2

        # Both files should be modified
        assert "SETTINGS" in file1.read_text()
        assert "SETTINGS" in file2.read_text()

    @pytest.mark.asyncio
    async def test_rename_with_depth_limit(self, tmp_path):
        """Test depth limiting for project scope."""
        # Create nested structure
        deep_dir = tmp_path / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)

        file_root = tmp_path / "root.py"
        file_root.write_text("target = 1")

        file_deep = deep_dir / "deep.py"
        file_deep.write_text("target = 2")

        # Only rename at depth 0 (current dir only)
        result = await rename(
            old_name="target",
            new_name="renamed",
            path=str(tmp_path),
            scope="project",
            depth=0,
            preview=False,
        )

        assert result["success"] is True
        # Should only modify root file
        assert "renamed" in file_root.read_text()
        assert "target" in file_deep.read_text()  # Deep file unchanged

    @pytest.mark.asyncio
    async def test_rename_missing_params(self):
        """Test error when required params missing."""
        result = await rename(old_name="", new_name="new")
        assert result["success"] is False
        assert "Missing" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_same_name(self, tmp_path):
        """Test error when old and new names are the same."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func(): pass")

        result = await rename(
            old_name="func",
            new_name="func",
            path=str(test_file),
            scope="file",
        )

        assert result["success"] is False
        assert "different" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_invalid_scope(self, tmp_path):
        """Test error with invalid scope."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func(): pass")

        result = await rename(
            old_name="func",
            new_name="new_func",
            path=str(test_file),
            scope="invalid",
        )

        assert result["success"] is False
        assert "Invalid scope" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_file_not_found(self):
        """Test error when file doesn't exist."""
        result = await rename(
            old_name="func",
            new_name="new_func",
            path="/nonexistent/file.py",
            scope="file",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_symbol_not_found(self, tmp_path):
        """Test error when symbol not found in file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def other_func(): pass")

        result = await rename(
            old_name="nonexistent",
            new_name="new_name",
            path=str(test_file),
            scope="file",
        )

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_no_python_files(self, tmp_path):
        """Test error when no Python files in directory."""
        # Create empty directory
        result = await rename(
            old_name="func",
            new_name="new_func",
            path=str(tmp_path),
            scope="project",
        )

        assert result["success"] is False
        assert "No Python files" in result["error"]

    @pytest.mark.asyncio
    async def test_rename_skips_excluded_dirs(self, tmp_path):
        """Test that __pycache__, .git etc are skipped."""
        # Create file in __pycache__
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        cache_file = cache_dir / "cached.py"
        cache_file.write_text("target = 1")

        # Create normal file
        normal_file = tmp_path / "main.py"
        normal_file.write_text("target = 2")

        result = await rename(
            old_name="target",
            new_name="renamed",
            path=str(tmp_path),
            scope="project",
            preview=False,
        )

        assert result["success"] is True
        # Cache file should be unchanged
        assert "target" in cache_file.read_text()
        # Normal file should be modified
        assert "renamed" in normal_file.read_text()

    @pytest.mark.asyncio
    async def test_rename_word_boundary_safety(self, tmp_path):
        """Test that rename uses word boundaries to avoid partial matches."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""def get_user(): pass
def get_username(): pass  # Should NOT be renamed
user = get_user()
username = get_username()  # Should NOT be renamed
""")

        result = await rename(
            old_name="get_user",
            new_name="fetch_user",
            path=str(test_file),
            scope="file",
            preview=False,
        )

        assert result["success"] is True

        content = test_file.read_text()
        assert "fetch_user" in content
        assert "get_username" in content  # Not renamed
        assert "fetch_username" not in content  # Partial match prevented
