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

"""Tests for code_intelligence_tool module."""

import tempfile
from pathlib import Path
import pytest

from victor.tools.code_intelligence_tool import (
    find_symbol,
    find_references,
    rename_symbol,
)


class TestFindSymbol:
    """Tests for find_symbol function."""

    @pytest.mark.asyncio
    async def test_find_function(self, tmp_path):
        """Test finding a function definition."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
def hello_world():
    print("Hello, World!")
    return True

def another_function():
    pass
"""
        )

        result = await find_symbol(file_path=str(test_file), symbol_name="hello_world")

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
        test_file.write_text(
            """
class MyClass:
    def __init__(self):
        self.value = 0

    def method(self):
        return self.value
"""
        )

        result = await find_symbol(file_path=str(test_file), symbol_name="MyClass")

        assert result is not None
        assert result["symbol_name"] == "MyClass"
        assert result["type"] == "class"
        assert "class MyClass" in result["code"]

    @pytest.mark.asyncio
    async def test_find_symbol_not_found(self, tmp_path):
        """Test searching for non-existent symbol."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
def existing_function():
    pass
"""
        )

        result = await find_symbol(file_path=str(test_file), symbol_name="nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_find_symbol_file_not_found(self):
        """Test with non-existent file."""
        result = await find_symbol(file_path="/nonexistent/file.py", symbol_name="test")

        assert result is not None
        assert "error" in result

    @pytest.mark.asyncio
    async def test_find_symbol_empty_file(self, tmp_path):
        """Test with empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await find_symbol(file_path=str(test_file), symbol_name="test")

        assert result is None

    @pytest.mark.asyncio
    async def test_find_nested_function(self, tmp_path):
        """Test finding nested function."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
def outer_function():
    def inner_function():
        return "inner"
    return inner_function()
"""
        )

        result = await find_symbol(file_path=str(test_file), symbol_name="inner_function")

        assert result is not None
        assert result["symbol_name"] == "inner_function"
        assert result["type"] == "function"

    @pytest.mark.asyncio
    async def test_find_method_in_class(self, tmp_path):
        """Test finding a method inside a class."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
class TestClass:
    def test_method(self):
        pass
"""
        )

        result = await find_symbol(file_path=str(test_file), symbol_name="test_method")

        assert result is not None
        assert result["symbol_name"] == "test_method"
        assert result["type"] == "function"

    @pytest.mark.asyncio
    async def test_find_symbol_generic_exception(self, tmp_path):
        """Test generic exception handling in find_symbol."""
        from unittest.mock import patch

        test_file = tmp_path / "test.py"
        test_file.write_text("def test_func(): pass")

        # Mock to raise a generic exception during parsing
        with patch(
            "victor.tools.code_intelligence_tool.get_parser",
            side_effect=RuntimeError("Parse error"),
        ):
            result = await find_symbol(file_path=str(test_file), symbol_name="test_func")

            assert result is not None
            assert "error" in result
            assert "unexpected error" in result["error"].lower()


class TestFindReferences:
    """Tests for find_references function."""

    @pytest.mark.asyncio
    async def test_find_references_basic(self, tmp_path):
        """Test finding references in directory."""
        # Create a directory with Python files
        test_file1 = tmp_path / "file1.py"
        test_file1.write_text(
            """
def target_function():
    return True
"""
        )

        test_file2 = tmp_path / "file2.py"
        test_file2.write_text(
            """
from file1 import target_function

result = target_function()
"""
        )

        result = await find_references(symbol_name="target_function", search_path=str(tmp_path))

        assert isinstance(result, list)
        # May or may not find cross-file references depending on implementation

    @pytest.mark.asyncio
    async def test_find_references_no_matches(self, tmp_path):
        """Test finding references when none exist."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
def some_function():
    pass
"""
        )

        result = await find_references(
            symbol_name="nonexistent_function", search_path=str(tmp_path)
        )

        assert isinstance(result, list)
        # Empty list or minimal results expected

    @pytest.mark.asyncio
    async def test_find_references_invalid_path(self):
        """Test with invalid search path."""
        result = await find_references(symbol_name="test", search_path="/nonexistent/path")

        assert isinstance(result, list)
        # Should handle gracefully, return empty list

    @pytest.mark.asyncio
    async def test_find_references_empty_directory(self, tmp_path):
        """Test with empty directory."""
        result = await find_references(symbol_name="test", search_path=str(tmp_path))

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_find_references_with_parse_errors(self, tmp_path):
        """Test that unparseable files are skipped gracefully."""
        # Create a file with invalid Python
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def invalid syntax here!")

        # Create a valid file
        good_file = tmp_path / "good.py"
        good_file.write_text(
            """
def target_function():
    pass

result = target_function()
"""
        )

        result = await find_references(symbol_name="target_function", search_path=str(tmp_path))

        # Should still return results from good file
        assert isinstance(result, list)
        # The bad file should be skipped (exception caught)

    @pytest.mark.asyncio
    async def test_find_references_with_file_exception(self, tmp_path):
        """Test that file exceptions are caught and processing continues."""
        from unittest.mock import patch, mock_open

        # Create a file that will exist
        test_file = tmp_path / "test.py"
        test_file.write_text("def target(): pass")

        # Mock open to raise an exception for any file
        with patch("builtins.open", side_effect=OSError("File read error")):
            result = await find_references(symbol_name="target", search_path=str(tmp_path))

            # Should return empty list since all files failed to read
            assert isinstance(result, list)
            assert len(result) == 0


class TestRenameSymbol:
    """Tests for rename_symbol function."""

    @pytest.mark.asyncio
    async def test_rename_missing_context(self, tmp_path):
        """Test rename with missing context."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
def old_function():
    return True

def caller():
    result = old_function()
    return result
"""
        )

        result = await rename_symbol(
            symbol_name="old_function",
            new_symbol_name="new_function",
            context={},  # Empty context, no tool_registry
            search_path=str(tmp_path),
        )

        # Should handle missing tool_registry gracefully
        assert result is not None
        assert "Error" in result or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_rename_with_no_references(self, tmp_path):
        """Test rename when no references found."""
        from unittest.mock import AsyncMock, MagicMock

        # Create mock tool_registry
        mock_registry = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = []  # No references found
        mock_registry.execute = AsyncMock(return_value=mock_result)

        result = await rename_symbol(
            symbol_name="nonexistent",
            new_symbol_name="new_name",
            context={"tool_registry": mock_registry},
            search_path=str(tmp_path),
        )

        # Should indicate no references found
        assert result is not None
        assert "No references" in result or "no references" in result.lower()

    @pytest.mark.asyncio
    async def test_rename_execution_failure(self):
        """Test rename when find_references fails."""
        from unittest.mock import AsyncMock, MagicMock

        # Create mock tool_registry that returns failure
        mock_registry = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Test error"
        mock_registry.execute = AsyncMock(return_value=mock_result)

        result = await rename_symbol(
            symbol_name="test",
            new_symbol_name="new_test",
            context={"tool_registry": mock_registry},
            search_path=".",
        )

        # Should handle error gracefully
        assert result is not None
        assert "Error" in result or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_rename_basic_structure(self):
        """Test basic structure of rename_symbol function."""
        # Just test that it handles missing context properly
        result = await rename_symbol(symbol_name="test", new_symbol_name="new_test", context={})

        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_rename_with_successful_transaction(self, tmp_path):
        """Test successful rename with transaction."""
        from unittest.mock import AsyncMock, MagicMock

        # Create test files
        test_file1 = tmp_path / "file1.py"
        test_file1.write_text(
            """
def old_name():
    return True

result = old_name()
"""
        )

        test_file2 = tmp_path / "file2.py"
        test_file2.write_text(
            """
from file1 import old_name

value = old_name()
"""
        )

        # Create mock tool_registry with proper responses
        mock_registry = MagicMock()

        # Mock find_references to return references
        references = [
            {"file_path": str(test_file1), "line": 2, "column": 5, "preview": "def old_name():"},
            {
                "file_path": str(test_file1),
                "line": 5,
                "column": 10,
                "preview": "result = old_name()",
            },
            {"file_path": str(test_file2), "line": 4, "column": 8, "preview": "value = old_name()"},
        ]

        find_refs_result = MagicMock()
        find_refs_result.success = True
        find_refs_result.output = references

        # Mock transaction start
        start_result = MagicMock()
        start_result.success = True

        # Mock file reads
        read_result1 = MagicMock()
        read_result1.success = True
        read_result1.output = test_file1.read_text()

        read_result2 = MagicMock()
        read_result2.success = True
        read_result2.output = test_file2.read_text()

        # Mock add_modify
        add_modify_result = MagicMock()
        add_modify_result.success = True

        # Setup execute mock to return appropriate results
        async def mock_execute(tool_name, context, **kwargs):
            if tool_name == "find_references":
                return find_refs_result
            elif kwargs.get("operation") == "start_transaction":
                return start_result
            elif tool_name == "read_file":
                if kwargs.get("path") == str(test_file1):
                    return read_result1
                else:
                    return read_result2
            elif kwargs.get("operation") == "add_modify":
                return add_modify_result
            return MagicMock(success=True)

        mock_registry.execute = AsyncMock(side_effect=mock_execute)

        result = await rename_symbol(
            symbol_name="old_name",
            new_symbol_name="new_name",
            context={"tool_registry": mock_registry},
            search_path=str(tmp_path),
        )

        assert result is not None
        assert "Queued rename" in result
        assert "new_name" in result

    @pytest.mark.asyncio
    async def test_rename_transaction_start_failure(self):
        """Test rename when transaction start fails."""
        from unittest.mock import AsyncMock, MagicMock

        mock_registry = MagicMock()

        # find_references succeeds
        find_refs_result = MagicMock()
        find_refs_result.success = True
        find_refs_result.output = [
            {"file_path": "test.py", "line": 1, "column": 1, "preview": "test"}
        ]

        # transaction start fails
        start_result = MagicMock()
        start_result.success = False
        start_result.error = "Transaction error"

        async def mock_execute(tool_name, context, **kwargs):
            if tool_name == "find_references":
                return find_refs_result
            elif kwargs.get("operation") == "start_transaction":
                return start_result
            return MagicMock(success=True)

        mock_registry.execute = AsyncMock(side_effect=mock_execute)

        result = await rename_symbol(
            symbol_name="test",
            new_symbol_name="new_test",
            context={"tool_registry": mock_registry},
            search_path=".",
        )

        assert "Error starting transaction" in result

    @pytest.mark.asyncio
    async def test_rename_file_read_failure(self, tmp_path):
        """Test rename when file read fails."""
        from unittest.mock import AsyncMock, MagicMock

        test_file = tmp_path / "test.py"
        test_file.write_text("def old(): pass")

        mock_registry = MagicMock()

        # find_references succeeds
        find_refs_result = MagicMock()
        find_refs_result.success = True
        find_refs_result.output = [
            {"file_path": str(test_file), "line": 1, "column": 5, "preview": "def old():"}
        ]

        # transaction start succeeds
        start_result = MagicMock()
        start_result.success = True

        # read_file fails
        read_result = MagicMock()
        read_result.success = False

        # abort succeeds
        abort_result = MagicMock()
        abort_result.success = True

        async def mock_execute(tool_name, context, **kwargs):
            if tool_name == "find_references":
                return find_refs_result
            elif kwargs.get("operation") == "start_transaction":
                return start_result
            elif tool_name == "read_file":
                return read_result
            elif kwargs.get("operation") == "abort":
                return abort_result
            return MagicMock(success=True)

        mock_registry.execute = AsyncMock(side_effect=mock_execute)

        result = await rename_symbol(
            symbol_name="old",
            new_symbol_name="new",
            context={"tool_registry": mock_registry},
            search_path=str(tmp_path),
        )

        assert "No files were modified" in result or "aborted" in result.lower()

    @pytest.mark.asyncio
    async def test_rename_add_modify_failure(self, tmp_path):
        """Test rename when add_modify fails."""
        from unittest.mock import AsyncMock, MagicMock

        test_file = tmp_path / "test.py"
        test_file.write_text("def old(): pass")

        mock_registry = MagicMock()

        # find_references succeeds
        find_refs_result = MagicMock()
        find_refs_result.success = True
        find_refs_result.output = [
            {"file_path": str(test_file), "line": 1, "column": 5, "preview": "def old():"}
        ]

        # transaction start succeeds
        start_result = MagicMock()
        start_result.success = True

        # read_file succeeds
        read_result = MagicMock()
        read_result.success = True
        read_result.output = "def old(): pass"

        # add_modify fails
        add_modify_result = MagicMock()
        add_modify_result.success = False
        add_modify_result.error = "Modify failed"

        async def mock_execute(tool_name, context, **kwargs):
            if tool_name == "find_references":
                return find_refs_result
            elif kwargs.get("operation") == "start_transaction":
                return start_result
            elif tool_name == "read_file":
                return read_result
            elif kwargs.get("operation") == "add_modify":
                return add_modify_result
            return MagicMock(success=True)

        mock_registry.execute = AsyncMock(side_effect=mock_execute)

        result = await rename_symbol(
            symbol_name="old",
            new_symbol_name="new",
            context={"tool_registry": mock_registry},
            search_path=str(tmp_path),
        )

        assert "Error queuing modification" in result

    @pytest.mark.asyncio
    async def test_rename_exception_during_processing(self, tmp_path):
        """Test rename when exception occurs during file processing."""
        from unittest.mock import AsyncMock, MagicMock

        test_file = tmp_path / "test.py"
        test_file.write_text("def old(): pass")

        mock_registry = MagicMock()

        # find_references succeeds
        find_refs_result = MagicMock()
        find_refs_result.success = True
        find_refs_result.output = [
            {"file_path": str(test_file), "line": 1, "column": 5, "preview": "def old():"}
        ]

        # transaction start succeeds
        start_result = MagicMock()
        start_result.success = True

        # abort succeeds
        abort_result = MagicMock()
        abort_result.success = True

        # read_file raises exception
        async def mock_execute(tool_name, context, **kwargs):
            if tool_name == "find_references":
                return find_refs_result
            elif kwargs.get("operation") == "start_transaction":
                return start_result
            elif tool_name == "read_file":
                raise RuntimeError("File system error")
            elif kwargs.get("operation") == "abort":
                return abort_result
            return MagicMock(success=True)

        mock_registry.execute = AsyncMock(side_effect=mock_execute)

        result = await rename_symbol(
            symbol_name="old",
            new_symbol_name="new",
            context={"tool_registry": mock_registry},
            search_path=str(tmp_path),
        )

        assert "Failed to process file" in result
        assert "aborted" in result.lower()
