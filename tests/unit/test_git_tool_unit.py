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

"""Tests for git_tool module."""

import pytest
from unittest.mock import patch

from victor.tools.git_tool import git


class TestGitTool:
    """Tests for git function."""

    @pytest.mark.asyncio
    async def test_git_invalid_operation(self):
        """Test git with invalid operation."""
        result = await git(operation="invalid_op")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

    @pytest.mark.asyncio
    async def test_git_status(self):
        """Test git status operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "clean", "")
            result = await git(operation="status")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_diff(self):
        """Test git diff operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "diff output", "")
            result = await git(operation="diff")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_log(self):
        """Test git log operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "commit logs", "")
            result = await git(operation="log", limit=5)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_git_branch(self):
        """Test git branch operation."""
        with patch("victor.tools.git_tool._run_git") as mock:
            mock.return_value = (True, "* main", "")
            result = await git(operation="branch")
            assert result["success"] is True
