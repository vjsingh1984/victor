"""Docker tool for container and image management.

Provides Docker operations without requiring docker-py library,
using the Docker CLI instead for maximum compatibility.

Features:
- Container management (list, start, stop, remove)
- Image operations (list, pull, build, remove)
- Logs and stats
- Network inspection
- Volume management
"""

import json
import subprocess
from typing import Any, Dict, List

from victor.tools.base import BaseTool, ToolParameter, ToolResult


class DockerTool(BaseTool):
    """Tool for Docker container and image management."""

    def __init__(self):
        """Initialize Docker tool."""
        super().__init__()

    @property
    def name(self) -> str:
        """Get tool name."""
        return "docker"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Docker container and image management.

Provides Docker operations using Docker CLI.

Operations:
- ps: List containers
- images: List images
- pull: Pull image from registry
- run: Run a container
- stop: Stop container(s)
- start: Start container(s)
- restart: Restart container(s)
- rm: Remove container(s)
- rmi: Remove image(s)
- logs: Get container logs
- stats: Get container stats
- inspect: Inspect container or image
- networks: List networks
- volumes: List volumes
- exec: Execute command in container

Example workflows:
1. Run container:
   docker(operation="pull", image="nginx:latest")
   docker(operation="run", image="nginx:latest", name="my-nginx", ports=["80:80"])

2. Container management:
   docker(operation="ps")
   docker(operation="logs", container="my-nginx")
   docker(operation="stop", container="my-nginx")

3. Image management:
   docker(operation="images")
   docker(operation="rmi", image="nginx:latest")

Safety:
- Requires Docker CLI installed
- No dangerous operations by default
- Operations are logged
"""

    @property
    def parameters(self) -> List[ToolParameter]:
        """Get tool parameters."""
        return [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: ps, images, pull, run, stop, start, restart, rm, rmi, logs, stats, inspect, networks, volumes, exec",
                required=True,
            ),
            ToolParameter(
                name="container",
                type="string",
                description="Container name or ID",
                required=False,
            ),
            ToolParameter(
                name="image",
                type="string",
                description="Image name (e.g., nginx:latest)",
                required=False,
            ),
            ToolParameter(
                name="name",
                type="string",
                description="Container name (for run operation)",
                required=False,
            ),
            ToolParameter(
                name="ports",
                type="array",
                description="Port mappings (e.g., ['80:80', '443:443'])",
                required=False,
            ),
            ToolParameter(
                name="env",
                type="array",
                description="Environment variables (e.g., ['KEY=value'])",
                required=False,
            ),
            ToolParameter(
                name="volumes",
                type="array",
                description="Volume mappings (e.g., ['/host:/container'])",
                required=False,
            ),
            ToolParameter(
                name="command",
                type="string",
                description="Command to execute (for run or exec)",
                required=False,
            ),
            ToolParameter(
                name="detach",
                type="boolean",
                description="Run container in background (default: true)",
                required=False,
            ),
            ToolParameter(
                name="all",
                type="boolean",
                description="Include stopped containers (for ps)",
                required=False,
            ),
            ToolParameter(
                name="tail",
                type="integer",
                description="Number of log lines to show (default: 100)",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute Docker operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with output or error
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        # Check if Docker is installed
        if not self._check_docker():
            return ToolResult(
                success=False,
                output="",
                error="Docker CLI not found. Please install Docker.",
            )

        try:
            if operation == "ps":
                return await self._ps(kwargs)
            elif operation == "images":
                return await self._images(kwargs)
            elif operation == "pull":
                return await self._pull(kwargs)
            elif operation == "run":
                return await self._run(kwargs)
            elif operation == "stop":
                return await self._stop(kwargs)
            elif operation == "start":
                return await self._start(kwargs)
            elif operation == "restart":
                return await self._restart(kwargs)
            elif operation == "rm":
                return await self._rm(kwargs)
            elif operation == "rmi":
                return await self._rmi(kwargs)
            elif operation == "logs":
                return await self._logs(kwargs)
            elif operation == "stats":
                return await self._stats(kwargs)
            elif operation == "inspect":
                return await self._inspect(kwargs)
            elif operation == "networks":
                return await self._networks(kwargs)
            elif operation == "volumes":
                return await self._volumes(kwargs)
            elif operation == "exec":
                return await self._exec(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            return ToolResult(
                success=False, output="", error=f"Docker error: {str(e)}"
            )

    def _check_docker(self) -> bool:
        """Check if Docker CLI is available."""
        try:
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_docker_command(self, args: List[str], timeout: int = 30) -> tuple[bool, str, str]:
        """Run Docker command.

        Args:
            args: Docker command arguments
            timeout: Command timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                ["docker"] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    async def _ps(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List containers."""
        show_all = kwargs.get("all", False)

        args = ["ps", "--format", "json"]
        if show_all:
            args.append("--all")

        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        # Parse JSON lines
        containers = []
        for line in stdout.strip().split("\n"):
            if line:
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return ToolResult(
            success=True,
            output=json.dumps({"containers": containers, "count": len(containers)}, indent=2),
            error="",
        )

    async def _images(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List images."""
        args = ["images", "--format", "json"]

        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        # Parse JSON lines
        images = []
        for line in stdout.strip().split("\n"):
            if line:
                try:
                    images.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return ToolResult(
            success=True,
            output=json.dumps({"images": images, "count": len(images)}, indent=2),
            error="",
        )

    async def _pull(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Pull image from registry."""
        image = kwargs.get("image")

        if not image:
            return ToolResult(
                success=False, output="", error="Missing required parameter: image"
            )

        args = ["pull", image]
        success, stdout, stderr = self._run_docker_command(args, timeout=300)  # 5 minutes

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=stdout, error="")

    async def _run(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Run a container."""
        image = kwargs.get("image")

        if not image:
            return ToolResult(
                success=False, output="", error="Missing required parameter: image"
            )

        args = ["run"]

        # Detached mode (default)
        if kwargs.get("detach", True):
            args.append("-d")

        # Container name
        if kwargs.get("name"):
            args.extend(["--name", kwargs["name"]])

        # Port mappings
        for port in kwargs.get("ports", []):
            args.extend(["-p", port])

        # Environment variables
        for env in kwargs.get("env", []):
            args.extend(["-e", env])

        # Volume mappings
        for volume in kwargs.get("volumes", []):
            args.extend(["-v", volume])

        # Image
        args.append(image)

        # Command (if provided)
        if kwargs.get("command"):
            args.extend(kwargs["command"].split())

        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(
            success=True,
            output=f"Container started: {stdout.strip()}",
            error="",
        )

    async def _stop(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Stop container(s)."""
        container = kwargs.get("container")

        if not container:
            return ToolResult(
                success=False, output="", error="Missing required parameter: container"
            )

        args = ["stop", container]
        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=f"Container stopped: {container}", error="")

    async def _start(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Start container(s)."""
        container = kwargs.get("container")

        if not container:
            return ToolResult(
                success=False, output="", error="Missing required parameter: container"
            )

        args = ["start", container]
        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=f"Container started: {container}", error="")

    async def _restart(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Restart container(s)."""
        container = kwargs.get("container")

        if not container:
            return ToolResult(
                success=False, output="", error="Missing required parameter: container"
            )

        args = ["restart", container]
        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=f"Container restarted: {container}", error="")

    async def _rm(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Remove container(s)."""
        container = kwargs.get("container")

        if not container:
            return ToolResult(
                success=False, output="", error="Missing required parameter: container"
            )

        args = ["rm", "-f", container]  # Force remove
        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=f"Container removed: {container}", error="")

    async def _rmi(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Remove image(s)."""
        image = kwargs.get("image")

        if not image:
            return ToolResult(
                success=False, output="", error="Missing required parameter: image"
            )

        args = ["rmi", image]
        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=f"Image removed: {image}", error="")

    async def _logs(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Get container logs."""
        container = kwargs.get("container")
        tail = kwargs.get("tail", 100)

        if not container:
            return ToolResult(
                success=False, output="", error="Missing required parameter: container"
            )

        args = ["logs", "--tail", str(tail), container]
        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=stdout, error="")

    async def _stats(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Get container stats."""
        container = kwargs.get("container")

        args = ["stats", "--no-stream", "--format", "json"]
        if container:
            args.append(container)

        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=stdout, error="")

    async def _inspect(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Inspect container or image."""
        target = kwargs.get("container") or kwargs.get("image")

        if not target:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: container or image",
            )

        args = ["inspect", target]
        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=stdout, error="")

    async def _networks(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List networks."""
        args = ["network", "ls", "--format", "json"]

        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=stdout, error="")

    async def _volumes(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List volumes."""
        args = ["volume", "ls", "--format", "json"]

        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=stdout, error="")

    async def _exec(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Execute command in container."""
        container = kwargs.get("container")
        command = kwargs.get("command")

        if not container:
            return ToolResult(
                success=False, output="", error="Missing required parameter: container"
            )

        if not command:
            return ToolResult(
                success=False, output="", error="Missing required parameter: command"
            )

        args = ["exec", container] + command.split()
        success, stdout, stderr = self._run_docker_command(args)

        if not success:
            return ToolResult(success=False, output="", error=stderr)

        return ToolResult(success=True, output=stdout, error="")
