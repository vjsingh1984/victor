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

"""Tests for code_executor_tool module."""

import io
import tarfile
import pytest
from unittest.mock import MagicMock, Mock, patch

from victor.tools.code_executor_tool import (
    CodeExecutionManager,
    execute_python_in_sandbox,
    upload_files_to_sandbox,
)


class TestCodeExecutionManager:
    """Tests for CodeExecutionManager class."""

    def test_init_success(self):
        """Test successful initialization with Docker available."""
        with patch("docker.from_env") as mock_docker:
            mock_docker.return_value = MagicMock()
            manager = CodeExecutionManager()

            assert manager.docker_image == "python:3.11-slim"
            assert manager.container is None
            assert manager.working_dir == "/app"
            mock_docker.assert_called_once()

    def test_init_docker_not_available(self):
        """Test initialization fails when Docker is not available."""
        from docker.errors import DockerException

        with patch("docker.from_env", side_effect=DockerException("Docker not found")):
            with pytest.raises(RuntimeError, match="Docker is not running"):
                CodeExecutionManager()

    def test_start_container(self):
        """Test starting a new container."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()
            mock_client.containers.run.return_value = mock_container

            manager = CodeExecutionManager()
            manager.start()

            mock_client.images.pull.assert_called_once_with("python:3.11-slim")
            mock_client.containers.run.assert_called_once()
            assert manager.container == mock_container

    def test_start_already_started(self):
        """Test that start() doesn't create a new container if already started."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()

            manager = CodeExecutionManager()
            manager.container = mock_container
            manager.start()

            # Should not call run again
            mock_client.containers.run.assert_not_called()

    def test_start_failure(self):
        """Test handling of container start failure."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_client.containers.run.side_effect = Exception("Container start failed")

            manager = CodeExecutionManager()
            with pytest.raises(RuntimeError, match="Failed to start Docker container"):
                manager.start()

            assert manager.container is None

    def test_stop_container(self):
        """Test stopping and removing a container."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()

            manager = CodeExecutionManager()
            manager.container = mock_container
            manager.stop()

            mock_container.remove.assert_called_once_with(force=True)
            assert manager.container is None

    def test_stop_no_container(self):
        """Test stopping when no container is running."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client

            manager = CodeExecutionManager()
            manager.stop()  # Should not raise

            assert manager.container is None

    def test_stop_container_not_found(self):
        """Test stopping when container is already gone."""
        with patch("docker.from_env") as mock_docker:
            import docker.errors

            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()
            mock_container.remove.side_effect = docker.errors.NotFound("Container not found")

            manager = CodeExecutionManager()
            manager.container = mock_container
            manager.stop()  # Should handle gracefully

            assert manager.container is None

    def test_execute_success(self):
        """Test successful code execution."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()

            # Mock exec_run response
            mock_exec_result = MagicMock()
            mock_exec_result.exit_code = 0
            mock_exec_result.output = (b"Hello, World!\n", b"")
            mock_container.exec_run.return_value = mock_exec_result

            manager = CodeExecutionManager()
            manager.container = mock_container

            result = manager.execute("print('Hello, World!')")

            assert result["exit_code"] == 0
            assert result["stdout"] == "Hello, World!\n"
            assert result["stderr"] == ""
            mock_container.exec_run.assert_called_once()

    def test_execute_with_error(self):
        """Test code execution with stderr output."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()

            mock_exec_result = MagicMock()
            mock_exec_result.exit_code = 1
            mock_exec_result.output = (b"", b"NameError: name 'x' is not defined\n")
            mock_container.exec_run.return_value = mock_exec_result

            manager = CodeExecutionManager()
            manager.container = mock_container

            result = manager.execute("print(x)")

            assert result["exit_code"] == 1
            assert result["stdout"] == ""
            assert "NameError" in result["stderr"]

    def test_execute_no_container(self):
        """Test executing without starting container first."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client

            manager = CodeExecutionManager()

            with pytest.raises(RuntimeError, match="Execution session not started"):
                manager.execute("print('hello')")

    def test_put_files_success(self, tmp_path):
        """Test uploading files to container."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()

            manager = CodeExecutionManager()
            manager.container = mock_container

            manager.put_files([str(test_file)])

            mock_container.put_archive.assert_called_once()

    def test_put_files_not_found(self):
        """Test uploading non-existent file."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()

            manager = CodeExecutionManager()
            manager.container = mock_container

            with pytest.raises(FileNotFoundError, match="Local file not found"):
                manager.put_files(["/nonexistent/file.txt"])

    def test_put_files_no_container(self):
        """Test uploading files without container."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client

            manager = CodeExecutionManager()

            with pytest.raises(RuntimeError, match="Execution session not started"):
                manager.put_files(["test.txt"])

    def test_get_file_success(self):
        """Test retrieving a file from container."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()

            # Create a tar archive with test content
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                content = b"Test file content"
                tarinfo = tarfile.TarInfo(name="test.txt")
                tarinfo.size = len(content)
                tar.addfile(tarinfo, io.BytesIO(content))

            tar_stream.seek(0)
            mock_container.get_archive.return_value = ([tar_stream.read()], {})

            manager = CodeExecutionManager()
            manager.container = mock_container

            result = manager.get_file("/app/test.txt")

            assert result == b"Test file content"
            mock_container.get_archive.assert_called_once_with("/app/test.txt")

    def test_get_file_no_container(self):
        """Test getting file without container."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client

            manager = CodeExecutionManager()

            with pytest.raises(RuntimeError, match="Execution session not started"):
                manager.get_file("/app/test.txt")

    def test_get_file_not_regular_file(self):
        """Test retrieving a non-regular file (e.g., directory) from container."""
        with patch("docker.from_env") as mock_docker:
            mock_client = MagicMock()
            mock_docker.return_value = mock_client
            mock_container = MagicMock()

            # Create a tar archive with a directory (non-regular file)
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tarinfo = tarfile.TarInfo(name="testdir")
                tarinfo.type = tarfile.DIRTYPE  # Directory type
                tar.addfile(tarinfo)

            tar_stream.seek(0)
            mock_container.get_archive.return_value = ([tar_stream.read()], {})

            manager = CodeExecutionManager()
            manager.container = mock_container

            with pytest.raises(FileNotFoundError, match="File not found in container"):
                manager.get_file("/app/testdir")


class TestExecutePythonInSandbox:
    """Tests for execute_python_in_sandbox tool function."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful code execution through tool."""
        mock_manager = MagicMock()
        mock_manager.execute.return_value = {
            "exit_code": 0,
            "stdout": "Hello\n",
            "stderr": "",
        }

        context = {"code_manager": mock_manager}
        result = await execute_python_in_sandbox(code="print('Hello')", context=context)

        assert "Exit Code: 0" in result
        assert "Hello" in result
        mock_manager.execute.assert_called_once_with("print('Hello')")

    @pytest.mark.asyncio
    async def test_execute_with_stderr(self):
        """Test code execution with error output."""
        mock_manager = MagicMock()
        mock_manager.execute.return_value = {
            "exit_code": 1,
            "stdout": "",
            "stderr": "Error: something went wrong\n",
        }

        context = {"code_manager": mock_manager}
        result = await execute_python_in_sandbox(code="raise Exception('test')", context=context)

        assert "Exit Code: 1" in result
        assert "STDERR" in result
        assert "something went wrong" in result

    @pytest.mark.asyncio
    async def test_execute_no_manager(self):
        """Test execution without code manager in context."""
        context = {}
        result = await execute_python_in_sandbox(code="print('test')", context=context)

        assert "Error" in result
        assert "CodeExecutionManager not found" in result


class TestUploadFilesToSandbox:
    """Tests for upload_files_to_sandbox tool function."""

    @pytest.mark.asyncio
    async def test_upload_success(self):
        """Test successful file upload."""
        mock_manager = MagicMock()
        mock_manager.put_files = MagicMock()

        context = {"code_manager": mock_manager}
        result = await upload_files_to_sandbox(
            file_paths=["file1.txt", "file2.txt"], context=context
        )

        assert "Successfully uploaded 2 files" in result
        mock_manager.put_files.assert_called_once_with(["file1.txt", "file2.txt"])

    @pytest.mark.asyncio
    async def test_upload_failure(self):
        """Test file upload failure."""
        mock_manager = MagicMock()
        mock_manager.put_files.side_effect = Exception("Upload failed")

        context = {"code_manager": mock_manager}
        result = await upload_files_to_sandbox(file_paths=["file.txt"], context=context)

        assert "Error uploading files" in result
        assert "Upload failed" in result

    @pytest.mark.asyncio
    async def test_upload_no_manager(self):
        """Test upload without code manager in context."""
        context = {}
        result = await upload_files_to_sandbox(file_paths=["file.txt"], context=context)

        assert "Error" in result
        assert "CodeExecutionManager not found" in result
