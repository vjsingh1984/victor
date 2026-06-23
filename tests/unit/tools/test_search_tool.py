import pytest
from unittest.mock import patch, AsyncMock
import argparse

# We will create this module in the next step
from victor.tools.unified.search_tool import search_tool


@pytest.mark.asyncio
async def test_search_tool_grep():
    """Test `search grep` subcommand."""
    with patch("victor.tools.unified.search_tool.grep_search", new_callable=AsyncMock) as mock_grep:
        mock_grep.return_value = [
            {"file": "app.py", "line": 10, "content": "def foo():"},
            {"file": "app.py", "line": 12, "content": "    return 'bar'"},
        ]

        result = await search_tool('search grep "def foo" app.py')

        mock_grep.assert_called_once_with(
            query="def foo", path="app.py", regex=False, case_sensitive=False
        )
        assert "app.py:10: def foo():" in result


@pytest.mark.asyncio
async def test_search_tool_grep_truncation_hint():
    """Test truncation hints for huge grep results."""
    with patch("victor.tools.unified.search_tool.grep_search", new_callable=AsyncMock) as mock_grep:
        # Create 150 fake matches
        mock_grep.return_value = [
            {"file": f"file{i}.py", "line": 1, "content": "foo"} for i in range(150)
        ]

        result = await search_tool('search grep "foo" .')

        assert "### 💡 SYSTEM HINT" in result
        assert "Too many matches" in result
        # Ensure it truncated the output
        assert result.count("foo") < 150


@pytest.mark.asyncio
async def test_search_tool_files():
    """Test `search files` subcommand."""
    with patch("victor.tools.unified.search_tool.find", new_callable=AsyncMock) as mock_find:
        mock_find.return_value = [
            {"path": "src/app.py", "type": "file", "size": "10KB"},
            {"path": "tests/test_app.py", "type": "file", "size": "2KB"},
        ]

        result = await search_tool('search files "*.py" .')

        mock_find.assert_called_once_with(name="*.py", path=".")
        assert "src/app.py" in result
        assert "tests/test_app.py" in result
