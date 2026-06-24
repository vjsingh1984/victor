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

"""Tests for the unified ``fs`` tool's ``search`` and ``edit`` subcommands.

``fs search`` delegates to the granular ``find()`` (file-name/metadata search);
``fs edit`` delegates to the structured, atomic ``edit()`` tool.
"""

from unittest.mock import AsyncMock, patch

import pytest

from victor.tools.unified.fs_tool import fs_tool


class TestFsSearch:
    """``fs search {pattern}`` -> ``find()``."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_find(self):
        mock_find = AsyncMock(
            return_value=[
                {"path": "a.py", "type": "file"},
                {"path": "b.py", "type": "file"},
            ]
        )
        with patch("victor.tools.unified.fs_tool.find", mock_find):
            result = await fs_tool('fs search "*.py" src --type file --limit 10')
        mock_find.assert_awaited_once_with(name="*.py", path="src", type="file", limit=10)
        assert "Found 2 match(es)" in result
        assert "a.py" in result and "b.py" in result

    @pytest.mark.asyncio
    async def test_search_defaults_path_and_type(self):
        mock_find = AsyncMock(return_value=[])
        with patch("victor.tools.unified.fs_tool.find", mock_find):
            result = await fs_tool('fs search "README*"')
        mock_find.assert_awaited_once_with(name="README*", path=".", type="all", limit=50)
        assert "No files matching" in result

    @pytest.mark.asyncio
    async def test_search_rejects_bad_type(self):
        # argparse choices enforce file/dir/all; an invalid type is a parse error.
        result = await fs_tool('fs search "*.py" --type directory')
        assert "invalid choice" in result


class TestFsEdit:
    """``fs edit {path} --old/--new`` -> ``edit(ops=[{type: replace, ...}])``."""

    @pytest.mark.asyncio
    async def test_edit_replace_builds_ops_and_delegates(self):
        mock_edit = AsyncMock(return_value={"success": True, "summary": "Edited", "files": ["x.py"]})
        with patch("victor.tools.file_editor_tool.edit", mock_edit):
            result = await fs_tool('fs edit x.py --old "DEBUG = True" --new "DEBUG = False"')
        mock_edit.assert_awaited_once()
        ops = mock_edit.call_args.kwargs["ops"]
        assert ops == [
            {
                "type": "replace",
                "path": "x.py",
                "old_str": "DEBUG = True",
                "new_str": "DEBUG = False",
            }
        ]
        assert "Edited" in result
        assert "x.py" in result

    @pytest.mark.asyncio
    async def test_edit_requires_old_and_new(self):
        result = await fs_tool("fs edit x.py --old only")
        assert "### ❌ ERROR" in result
        assert "--old/--new" in result

    @pytest.mark.asyncio
    async def test_edit_accepts_raw_ops_json(self):
        mock_edit = AsyncMock(return_value={"success": True, "summary": "done"})
        with patch("victor.tools.file_editor_tool.edit", mock_edit):
            await fs_tool('fs edit x.py --ops \'[{"type": "create", "path": "n.py", "content": "x"}]\'')
        ops = mock_edit.call_args.kwargs["ops"]
        assert ops[0]["type"] == "create"

    @pytest.mark.asyncio
    async def test_edit_invalid_json_ops_errors(self):
        result = await fs_tool("fs edit x.py --ops not-json")
        assert "### ❌ ERROR" in result
        assert "JSON" in result

    @pytest.mark.asyncio
    async def test_edit_surfaces_failure(self):
        mock_edit = AsyncMock(return_value={"success": False, "error": "old_str not found"})
        with patch("victor.tools.file_editor_tool.edit", mock_edit):
            result = await fs_tool('fs edit x.py --old a --new b')
        assert "### ❌ ERROR" in result
        assert "old_str not found" in result
