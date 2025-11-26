"""
A tool for executing Python code in a secure, stateful Docker container.
"""

import io
import os
import tarfile
from pathlib import Path
from typing import List

import docker
from docker.models.containers import Container
from docker.errors import DockerException

from victor.tools.decorators import tool


class CodeExecutionManager:
    """
    Manages a persistent, isolated Docker container for stateful code execution.
    """

    def __init__(self, docker_image: str = "python:3.11-slim"):
        self.docker_image = docker_image
        self.container: Container | None = None
        self.working_dir = "/app"
        try:
            self.docker_client = docker.from_env()
        except DockerException as e:
            raise RuntimeError(
                "Docker is not running or not installed. This feature requires Docker."
            ) from e

    def start(self):
        """Starts the persistent Docker container."""
        if self.container:
            return  # Already started

        try:
            self.docker_client.images.pull(self.docker_image)
            self.container = self.docker_client.containers.run(
                self.docker_image,
                command="sleep infinity",  # Keep the container running
                detach=True,
                working_dir=self.working_dir,
                volumes={
                    os.path.abspath(self.working_dir): {
                        "bind": self.working_dir,
                        "mode": "rw",
                    }
                }
            )
        except Exception as e:
            self.container = None
            raise RuntimeError(f"Failed to start Docker container: {e}")

    def stop(self):
        """Stops and removes the Docker container."""
        if self.container:
            try:
                self.container.remove(force=True)
            except docker.errors.NotFound:
                pass  # Container already gone
            self.container = None

    def execute(self, code: str, timeout: int = 60) -> dict:
        """Executes a block of Python code inside the running container."""
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
        if not self.container:
            raise RuntimeError("Execution session not started. Call start() first.")

        bits, _ = self.container.get_archive(remote_path)
        
        # Read the tar archive from the bits
        with io.BytesIO(b"".join(bits)) as tar_stream:
            with tarfile.open(fileobj=tar_stream, mode='r') as tar:
                # Assuming the tar contains one file
                member = tar.getmembers()[0]
                file_obj = tar.extractfile(member)
                if file_obj:
                    return file_obj.read()
        raise FileNotFoundError(f"File not found in container: {remote_path}")


@tool
async def execute_python_in_sandbox(code: str, context: dict) -> str:
    """
    Executes a block of Python code in a stateful, sandboxed environment.
    Files can be uploaded to the sandbox using `upload_files_to_sandbox`.

    Args:
        code: The Python code to execute.
        context: The tool context, provided by the orchestrator.

    Returns:
        A string containing the exit code, stdout, and stderr.
    """
    manager: CodeExecutionManager = context.get("code_manager")
    if not manager:
        return "Error: CodeExecutionManager not found in context."
    
    result = manager.execute(code)
    
    output = f"Exit Code: {result['exit_code']}\n"
    if result['stdout']:
        output += f"--- STDOUT ---\n{result['stdout']}\n"
    if result['stderr']:
        output += f"--- STDERR ---\n{result['stderr']}\n"
    return output


@tool
async def upload_files_to_sandbox(file_paths: List[str], context: dict) -> str:
    """
    Uploads one or more local files to the code execution sandbox.
    The files will be placed in the root of the execution environment.

    Args:
        file_paths: A list of local file paths to upload.
        context: The tool context, provided by the orchestrator.

    Returns:
        A confirmation message.
    """
    manager: CodeExecutionManager = context.get("code_manager")
    if not manager:
        return "Error: CodeExecutionManager not found in context."
    
    try:
        manager.put_files(file_paths)
        return f"Successfully uploaded {len(file_paths)} files to the sandbox."
    except Exception as e:
        return f"Error uploading files: {e}"

