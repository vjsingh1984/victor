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

"""
A tool for executing Python code in a secure, stateful Docker container.
"""

import io
import tarfile
from pathlib import Path
from typing import List

# Optional docker import
try:
    import docker
    from docker.models.containers import Container
    from docker.errors import DockerException

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None  # type: ignore
    Container = None  # type: ignore
    DockerException = Exception  # Fallback

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool


class CodeSandbox:
    """Manages a persistent, isolated Docker container for stateful code execution.

    Note: Previously named `CodeExecutionManager`. Alias kept for backward compatibility.
    """

    def __init__(self, docker_image: str = "python:3.11-slim", require_docker: bool = False):
        self.docker_image = docker_image
        self.container: Container | None = None
        self.working_dir = "/app"
        self.docker_available = False
        self.docker_client = None

        if not DOCKER_AVAILABLE:
            if require_docker:
                raise RuntimeError("Docker package not installed. Install with: pip install docker")
            # Docker package not available - continue without it
            return

        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except DockerException as e:
            if require_docker:
                raise RuntimeError(
                    "Docker is not running or not installed. This feature requires Docker."
                ) from e
            # Docker not available, but not required - continue without it
            self.docker_client = None

    def start(self) -> None:
        """Starts the persistent Docker container.

        If Docker fails to start, the code executor will continue in
        degraded mode (no container execution available).
        """
        if not self.docker_available:
            # Docker not available - skip container startup
            return

        if self.container:
            return  # Already started

        try:
            self.docker_client.images.pull(self.docker_image)
            self.container = self.docker_client.containers.run(
                self.docker_image,
                command="sleep infinity",  # Keep the container running
                detach=True,
                working_dir=self.working_dir,
                # No volume mounting needed - we execute code directly
            )
        except Exception as e:
            # Log the error but don't crash Victor
            # Docker container execution will be unavailable
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Docker container startup failed: {e}. "
                "Code execution in containers will be unavailable."
            )
            self.container = None
            self.docker_available = False  # Mark Docker as unavailable

    def stop(self) -> None:
        """Stops and removes the Docker container."""
        if not self.docker_available:
            # Docker not available - nothing to stop
            return

        if self.container:
            try:
                self.container.remove(force=True)
            except Exception:
                pass  # Container already gone or other error
            self.container = None

    def execute(self, code: str, timeout: int = 60) -> dict:
        """Executes a block of Python code inside the running container."""
        if not self.docker_available:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "Docker is not available. Code execution in containers is disabled.",
            }

        if not self.container:
            raise RuntimeError("Execution session not started. Call start() first.")

        exec_result = self.container.exec_run(
            ["python", "-c", code],
            demux=True,  # Separate stdout and stderr
        )

        stdout = exec_result.output[0].decode("utf-8") if exec_result.output[0] else ""
        stderr = exec_result.output[1].decode("utf-8") if exec_result.output[1] else ""

        return {
            "exit_code": exec_result.exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

    def put_files(self, file_paths: List[str]):
        """Copies files from the local filesystem into the container's working dir."""
        if not self.docker_available:
            # Docker not available - skip file operations
            return

        if not self.container:
            raise RuntimeError("Execution session not started. Call start() first.")

        # Create a tarball in memory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for file_path in file_paths:
                p = Path(file_path)
                if not p.exists():
                    raise FileNotFoundError(f"Local file not found: {file_path}")
                tar.add(str(p), arcname=p.name)

        tar_stream.seek(0)
        self.container.put_archive(self.working_dir, tar_stream)

    def get_file(self, remote_path: str) -> bytes:
        """Retrieves a single file from the container."""
        if not self.docker_available:
            # Docker not available - return empty bytes
            return b""

        if not self.container:
            raise RuntimeError("Execution session not started. Call start() first.")

        bits, _ = self.container.get_archive(remote_path)

        # Read the tar archive from the bits
        with io.BytesIO(b"".join(bits)) as tar_stream:
            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                # Assuming the tar contains one file
                member = tar.getmembers()[0]
                file_obj = tar.extractfile(member)
                if file_obj:
                    return file_obj.read()
        raise FileNotFoundError(f"File not found in container: {remote_path}")


# Backward compatibility alias
CodeExecutionManager = CodeSandbox


async def _execute_code(sandbox_instance: CodeSandbox, code: str) -> str:
    """Internal: Execute Python code in sandbox."""
    result = sandbox_instance.execute(code)
    output = f"Exit Code: {result['exit_code']}\n"
    if result["stdout"]:
        output += f"--- STDOUT ---\n{result['stdout']}\n"
    if result["stderr"]:
        output += f"--- STDERR ---\n{result['stderr']}\n"
    return output


async def _upload_files(sandbox_instance: CodeSandbox, file_paths: List[str]) -> str:
    """Internal: Upload files to sandbox."""
    try:
        sandbox_instance.put_files(file_paths)
        return f"Successfully uploaded {len(file_paths)} files to the sandbox."
    except Exception as e:
        return f"Error uploading files: {e}"


@tool(
    category="execution",
    priority=Priority.MEDIUM,  # Task-specific code execution
    access_mode=AccessMode.EXECUTE,  # Runs code in container
    danger_level=DangerLevel.HIGH,  # Arbitrary code execution
    keywords=["sandbox", "execute", "python", "container", "docker", "upload", "code"],
)
async def sandbox(
    operation: str,
    code: str = "",
    file_paths: List[str] = None,
    context: dict = None,
) -> str:
    """Unified sandbox operations for code execution in isolated Docker container.

    Operations:
    - "execute": Run Python code in the sandbox
    - "upload": Upload local files to the sandbox

    Args:
        operation: Operation to perform - "execute" or "upload"
        code: Python code to execute (for "execute" operation)
        file_paths: List of local file paths to upload (for "upload" operation)
        context: Tool context provided by orchestrator

    Returns:
        Operation result string
    """
    if context is None:
        return "Error: Context not provided."

    sandbox_instance: CodeSandbox = context.get("code_manager")
    if not sandbox_instance:
        return "Error: CodeSandbox not found in context."

    op = operation.lower().strip()

    if op == "execute":
        if not code:
            return "Error: 'code' parameter required for execute operation."
        return await _execute_code(sandbox_instance, code)

    elif op == "upload":
        if not file_paths:
            return "Error: 'file_paths' parameter required for upload operation."
        return await _upload_files(sandbox_instance, file_paths)

    else:
        return f"Error: Unknown operation '{operation}'. Use 'execute' or 'upload'."


