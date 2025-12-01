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

"""Tests for docker_tool module."""

import pytest
from unittest.mock import patch, MagicMock

from victor.tools.docker_tool import docker, _check_docker, _run_docker_command


class TestDockerTool:
    """Tests for docker function."""

    @pytest.mark.asyncio
    async def test_docker_invalid_operation(self):
        """Test docker with invalid operation."""
        with patch("victor.tools.docker_tool._check_docker", return_value=True):
            result = await docker(operation="invalid_op")
            assert result["success"] is False
            assert "Unknown operation" in result["error"]

    @pytest.mark.asyncio
    async def test_docker_ps(self):
        """Test docker ps operation."""
        with patch("victor.tools.docker_tool._check_docker", return_value=True):
            with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
                mock_run.return_value = (
                    True,
                    '[{"ID":"abc123","Names":"test","State":"running"}]',
                    "",
                )
                result = await docker(operation="ps")
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_images(self):
        """Test docker images operation."""
        with patch("victor.tools.docker_tool._check_docker", return_value=True):
            with patch("victor.tools.docker_tool._run_docker_command") as mock_run:
                mock_run.return_value = (
                    True,
                    '[{"ID":"sha256:abc123","Repository":"test","Tag":"latest"}]',
                    "",
                )
                result = await docker(operation="images")
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_docker_not_available(self):
        """Test docker when Docker is not available."""
        with patch("victor.tools.docker_tool._check_docker", return_value=False):
            result = await docker(operation="ps")
            assert result["success"] is False
            assert "Docker" in result["error"]


class TestCheckDocker:
    """Tests for _check_docker function."""

    def test_check_docker_available(self):
        """Test _check_docker when Docker is available."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _check_docker() is True

    def test_check_docker_not_available(self):
        """Test _check_docker when Docker is not available."""

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert _check_docker() is False


class TestRunDockerCommand:
    """Tests for _run_docker_command function."""

    def test_run_docker_command_success(self):
        """Test successful Docker command."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "output"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            success, stdout, stderr = _run_docker_command(["ps"])
            assert success is True
            assert stdout == "output"

    def test_run_docker_command_timeout(self):
        """Test Docker command timeout."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=30)
            success, stdout, stderr = _run_docker_command(["ps"])
            assert success is False
            assert "timed out" in stderr.lower()
