# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for the unified ``code`` tool's ``grep`` subcommand.

``code grep`` is the canonical literal content-search surface and delegates to
the shared ``grep_search`` helper.
"""

from unittest.mock import AsyncMock, patch

import pytest

from victor.tools.unified.code_tool import code_tool


class TestCodeGrep:
    """``code grep {query} {path}`` -> ``grep_search``."""

    @pytest.mark.asyncio
    async def test_grep_delegates_to_grep_search(self):
        mock_grep = AsyncMock(
            return_value=[
                {"file": "a.py", "line": 3, "content": "def foo():"},
                {"file": "b.py", "line": 7, "content": "def bar():"},
            ]
        )
        with patch("victor.tools.unified.code_tool.grep_search", mock_grep, create=True):
            # grep_search is imported lazily inside the handler; patch the source.
            with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
                result = await code_tool('code grep "def foo" src')
        mock_grep.assert_awaited_once_with(
            query="def foo", path="src", regex=False, case_sensitive=False
        )
        assert "a.py:3" in result
        assert "def foo():" in result

    @pytest.mark.asyncio
    async def test_grep_regex_and_case_sensitive_flags_forwarded(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep "^def " src --regex --case-sensitive')
        mock_grep.assert_awaited_once_with(
            query="^def ", path="src", regex=True, case_sensitive=True
        )
        assert "No matches found." in result

    @pytest.mark.asyncio
    async def test_grep_short_case_sensitive_flag(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            await code_tool('code grep "Foo" src -C')
        assert mock_grep.call_args.kwargs["case_sensitive"] is True

    @pytest.mark.asyncio
    async def test_grep_defaults_path_to_cwd(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            await code_tool('code grep "foo"')
        assert mock_grep.call_args.kwargs["path"] == "."

    @pytest.mark.asyncio
    async def test_grep_no_matches_message(self):
        mock_grep = AsyncMock(return_value=[])
        with patch("victor.tools.unified._search_helpers.grep_search", mock_grep):
            result = await code_tool('code grep "zzz" src')
        assert result == "No matches found."

    @pytest.mark.asyncio
    async def test_existing_code_subcommands_still_work(self):
        """Regression: metrics subcommand must still parse after adding grep/search."""
        with patch("victor.tools.unified.code_tool.analyze_metrics", AsyncMock()):
            # Just confirm it routes to metrics (not grep/search) without error.
            result = await code_tool("code metrics .")
            assert "### ❌ ERROR" not in result or "Metrics" in result
