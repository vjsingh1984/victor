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

"""Tests for bash tool."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.tools.bash import shell


@pytest.mark.asyncio
async def test_shell_simple_command():
    """Test executing a simple bash command."""
    result = await shell(cmd="echo 'hello'")

    assert result["success"] is True
    assert "hello" in result["stdout"]
    assert result["return_code"] == 0


@pytest.mark.asyncio
async def test_shell_missing_command():
    """Test bash tool with missing command."""
    result = await shell(cmd="")

    assert result["success"] is False
    assert "Missing required parameter" in result["error"]


@pytest.mark.asyncio
async def test_shell_dangerous_command():
    """Test bash tool blocks dangerous commands."""
    result = await shell(cmd="rm -rf /", dangerous=False)

    assert result["success"] is False
    assert "Dangerous command blocked" in result["error"]


@pytest.mark.asyncio
async def test_shell_allow_dangerous():
    """Test bash tool allows dangerous commands when explicitly allowed."""
    # This won't actually execute, just test the allow flag works
    with patch("asyncio.create_subprocess_shell") as mock_subprocess:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_subprocess.return_value = mock_process

        await shell(cmd="rm -rf test", dangerous=True)

        # Should attempt to execute
        mock_subprocess.assert_called_once()


@pytest.mark.asyncio
async def test_shell_with_working_dir():
    """Test bash command with working directory."""
    result = await shell(cmd="pwd", cwd="/tmp")

    assert result["success"] is True
    # The working dir might be in stdout or as a field
    assert "/tmp" in result.get("stdout", "") or result.get("cwd") == "/tmp"


@pytest.mark.asyncio
async def test_shell_timeout():
    """Test bash command timeout."""
    with patch("asyncio.create_subprocess_shell") as mock_subprocess:
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=TimeoutError())
        mock_subprocess.return_value = mock_process

        result = await shell(cmd="sleep 100", timeout=1)

        assert result["success"] is False
        assert "timed out" in result["error"] or "Failed to execute" in result["error"]


@pytest.mark.asyncio
async def test_shell_working_dir_not_found():
    """Test bash command with non-existent working directory."""
    result = await shell(cmd="pwd", cwd="/nonexistent/directory")

    assert result["success"] is False
    assert "Working directory does not exist" in result["error"]
    assert result["return_code"] == -1


@pytest.mark.asyncio
async def test_shell_general_exception():
    """Test bash command general exception handling."""
    with patch("asyncio.create_subprocess_shell") as mock_subprocess:
        mock_subprocess.side_effect = RuntimeError("Unexpected error")

        result = await shell(cmd="echo test")

        assert result["success"] is False
        assert "Failed to execute command" in result["error"]
        assert result["return_code"] == -1
