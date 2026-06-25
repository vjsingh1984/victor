import warnings
from unittest.mock import AsyncMock, patch

import pytest

from victor.tools.unified.search_tool import search_tool


@pytest.mark.asyncio
async def test_search_grep_forwards_to_code_grep():
    """``search grep`` is a shim that forwards to ``code grep``."""
    mock_code = AsyncMock(return_value="code-grep-output")
    with patch("victor.tools.unified.code_tool.code_tool", mock_code):
        result = await search_tool('search grep "def foo" app.py --case-sensitive')

    mock_code.assert_awaited_once()
    forwarded = mock_code.call_args.args[0]
    assert forwarded.startswith("code grep")
    assert "def foo" in forwarded
    assert "app.py" in forwarded
    assert result == "code-grep-output"


@pytest.mark.asyncio
async def test_search_files_forwards_to_fs_search():
    """``search files`` is a shim that forwards to ``fs search``."""
    mock_fs = AsyncMock(return_value="fs-search-output")
    with patch("victor.tools.unified.fs_tool.fs_tool", mock_fs):
        result = await search_tool('search files "*.py" src')

    mock_fs.assert_awaited_once()
    forwarded = mock_fs.call_args.args[0]
    assert forwarded.startswith("fs search")
    assert "*.py" in forwarded
    assert "src" in forwarded
    assert result == "fs-search-output"


@pytest.mark.asyncio
async def test_search_emits_deprecation_warning():
    """Calling the deprecated tool emits a DeprecationWarning."""
    with patch("victor.tools.unified.code_tool.code_tool", AsyncMock(return_value="ok")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await search_tool('search grep "x" .')

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


@pytest.mark.asyncio
async def test_search_unknown_subcommand_returns_error():
    result = await search_tool("search frobnicate")
    assert "### ❌ ERROR" in result
    assert "Unknown search subcommand" in result


@pytest.mark.asyncio
async def test_search_no_subcommand_returns_error():
    result = await search_tool("search")
    assert "### ❌ ERROR" in result
    assert "deprecated" in result
