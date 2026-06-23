import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from victor.tools.unified.code_tool import code_tool


@pytest.mark.asyncio
async def test_code_tool_test():
    """Test `code test` subcommand."""
    with patch("victor.tools.unified.code_tool.run_tests", new_callable=AsyncMock) as mock_test:
        mock_test.return_value = {"passed": 5, "failed": 0, "output": "5 passed"}

        result = await code_tool("code test pytest tests/")

        mock_test.assert_called_once_with(runner="pytest", path="tests/")
        assert "5 passed" in result


@pytest.mark.asyncio
async def test_code_tool_execute():
    """Test `code execute` subcommand."""
    with patch(
        "victor.tools.unified.code_tool.execute_python", new_callable=AsyncMock
    ) as mock_exec:
        mock_exec.return_value = "hello world\n"

        result = await code_tool("code execute \"print('hello world')\"")

        mock_exec.assert_called_once_with("print('hello world')")
        assert "hello world" in result


@pytest.mark.asyncio
async def test_code_tool_metrics():
    """Test `code metrics` subcommand formatting."""
    with patch(
        "victor.tools.unified.code_tool.analyze_metrics", new_callable=AsyncMock
    ) as mock_metrics:
        mock_metrics.return_value = {"complexity": 10, "loc": 500}

        result = await code_tool("code metrics src/")

        assert "**complexity**: 10" in result
        assert "**loc**: 500" in result
