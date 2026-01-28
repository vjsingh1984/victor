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

"""Consolidated Docker tool for container and image management.

Provides unified Docker operations using the Docker CLI for maximum compatibility.

Features:
- Container management (list, start, stop, restart, remove, exec)
- Image operations (list, pull, remove)
- Logs and stats
- Network and volume inspection
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.subprocess_executor import run_command_async, check_docker_available


async def _run_docker_command_async(args: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
    """Run Docker command asynchronously.

    Args:
        args: Docker command arguments
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success, stdout, stderr)
    """
    command = "docker " + " ".join(args)
    result = await run_command_async(
        command,
        timeout=timeout,
        check_dangerous=False,
    )
    return result.success, result.stdout, result.stderr


@tool(
    cost_tier=CostTier.MEDIUM,
    category="docker",
    priority=Priority.MEDIUM,  # Task-specific containerization tool
    access_mode=AccessMode.EXECUTE,  # Runs Docker commands
    danger_level=DangerLevel.HIGH,  # Container operations can have significant effects
    keywords=["docker", "container", "image", "run", "build"],
)
async def docker(
    operation: str,
    resource_id: Optional[str] = None,
    resource_type: str = "container",
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Unified Docker operations for container and image management.

    Provides a single interface for all Docker operations, mirroring the
    Docker CLI structure. Consolidates 15+ separate Docker tools into one.

    Args:
        operation: Docker operation to perform. Options:
            Container ops: "ps", "start", "stop", "restart", "rm", "logs", "stats", "exec", "inspect"
            Image ops: "images", "pull", "build", "rmi"
            Network ops: "networks"
            Volume ops: "volumes"
        resource_id: Container ID, image name, or resource identifier
        resource_type: Type of resource: "container", "image", "network", "volume"
        options: Additional options as dictionary:
            For ps: {"all": True} to show all containers
            For logs: {"tail": 100, "follow": False}
            For exec: {"command": "ls -la"}
            For pull: {"platform": "linux/amd64"}
            For build: {"path": ".", "dockerfile": "Dockerfile", "build_args": {}, "target": "stage"}
            For run: {"image": "nginx", "detach": True, "ports": {"80": "8080"}}

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - result: Operation-specific result data
        - message: Human-readable message
        - error: Error message if failed

    Examples:
        # List running containers
        docker("ps")

        # List all containers including stopped
        docker("ps", options={"all": True})

        # Start a container
        docker("start", resource_id="my-container")

        # Stop a container
        docker("stop", resource_id="container-id")

        # Get container logs
        docker("logs", resource_id="my-container", options={"tail": 100})

        # List images
        docker("images")

        # Pull an image
        docker("pull", resource_id="nginx:latest")

        # Build an image
        docker("build", resource_id="myapp:1.0", options={"path": "."})

        # Build with custom Dockerfile
        docker("build", resource_id="myapp:1.0", options={"path": ".", "dockerfile": "Dockerfile.prod"})

        # Build with arguments
        docker("build", resource_id="myapp:1.0", options={"path": ".", "build_args": {"VERSION": "1.0"}})

        # Execute command in container
        docker("exec", resource_id="my-container", options={"command": "ls -la"})

        # Inspect container
        docker("inspect", resource_id="my-container")

        # List networks
        docker("networks")

        # List volumes
        docker("volumes")
    """
    if not check_docker_available():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if options is None:
        options = {}

    operation = operation.lower()

    # Container operations
    if operation == "ps":
        args = ["ps", "--format", "json"]
        if options.get("all"):
            args.append("--all")

        success, stdout, stderr = await _run_docker_command_async(args)
        if not success:
            return {"success": False, "error": stderr}

        containers = []
        for line in stdout.strip().split("\n"):
            if line:
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return {
            "success": True,
            "result": {"containers": containers, "count": len(containers)},
            "message": f"Found {len(containers)} container(s)",
        }

    elif operation in ["start", "stop", "restart", "rm"]:
        if not resource_id:
            return {"success": False, "error": f"resource_id required for {operation} operation"}

        args = [operation, resource_id]
        if operation == "rm" and options.get("force"):
            args.insert(1, "-f")

        success, stdout, stderr = await _run_docker_command_async(args)
        if not success:
            return {"success": False, "error": stderr}

        return {
            "success": True,
            "result": {"container_id": resource_id},
            "message": f"Container {operation} successful: {resource_id}",
        }

    elif operation == "logs":
        if not resource_id:
            return {"success": False, "error": "resource_id required for logs operation"}

        args = ["logs", resource_id]
        if "tail" in options:
            args.extend(["--tail", str(options["tail"])])
        if options.get("follow"):
            args.append("--follow")

        timeout = 60 if options.get("follow") else 30
        success, stdout, stderr = await _run_docker_command_async(args, timeout=timeout)

        if not success:
            return {"success": False, "error": stderr}

        return {
            "success": True,
            "result": {"logs": stdout, "stderr": stderr},
            "message": f"Retrieved logs for {resource_id}",
        }

    elif operation == "stats":
        args = ["stats", "--no-stream", "--format", "json"]
        if resource_id:
            args.append(resource_id)

        success, stdout, stderr = await _run_docker_command_async(args)
        if not success:
            return {"success": False, "error": stderr}

        stats = []
        for line in stdout.strip().split("\n"):
            if line:
                try:
                    stats.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return {
            "success": True,
            "result": {"stats": stats, "count": len(stats)},
            "message": f"Retrieved stats for {len(stats)} container(s)",
        }

    elif operation == "exec":
        if not resource_id:
            return {"success": False, "error": "resource_id required for exec operation"}
        if "command" not in options:
            return {"success": False, "error": "command required in options for exec"}

        cmd = options["command"]
        if isinstance(cmd, str):
            cmd = cmd.split()

        args = ["exec", resource_id] + cmd

        success, stdout, stderr = await _run_docker_command_async(args)

        return {
            "success": success,
            "result": {"stdout": stdout, "stderr": stderr},
            "message": (
                f"Executed command in {resource_id}" if success else f"Exec failed: {stderr}"
            ),
        }

    elif operation == "inspect":
        if not resource_id:
            return {"success": False, "error": "resource_id required for inspect operation"}

        args = ["inspect", resource_id]
        success, stdout, stderr = await _run_docker_command_async(args)

        if not success:
            return {"success": False, "error": stderr}

        try:
            inspect_data = json.loads(stdout)
            return {
                "success": True,
                "result": {"data": inspect_data},
                "message": f"Inspected {resource_id}",
            }
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse inspect output"}

    # Image operations
    elif operation == "images":
        args = ["images", "--format", "json"]

        success, stdout, stderr = await _run_docker_command_async(args)
        if not success:
            return {"success": False, "error": stderr}

        images = []
        for line in stdout.strip().split("\n"):
            if line:
                try:
                    images.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return {
            "success": True,
            "result": {"images": images, "count": len(images)},
            "message": f"Found {len(images)} image(s)",
        }

    elif operation == "pull":
        if not resource_id:
            return {
                "success": False,
                "error": "resource_id (image name) required for pull operation",
            }

        args = ["pull", resource_id]
        if "platform" in options:
            args.extend(["--platform", options["platform"]])

        success, stdout, stderr = await _run_docker_command_async(args, timeout=300)

        if not success:
            return {"success": False, "error": stderr}

        return {
            "success": True,
            "result": {"image": resource_id},
            "message": f"Pulled image: {resource_id}",
        }

    elif operation == "build":
        """Build a Docker image from a Dockerfile.

        Args:
            resource_id: Image name and tag (e.g., "myapp:1.0")
            options:
                path: Build context path (default: ".")
                dockerfile: Alternative Dockerfile path (default: "Dockerfile")
                build_args: Dict of build arguments
                target: Target stage for multi-stage builds
        """
        if not resource_id:
            return {
                "success": False,
                "error": "resource_id (image name:tag) required for build operation",
            }

        build_path = options.get("path", ".")
        args = ["build", "-t", resource_id, build_path]

        # Add optional Dockerfile path
        if "dockerfile" in options:
            args.extend(["-f", options["dockerfile"]])

        # Add build arguments
        if "build_args" in options:
            for key, value in options["build_args"].items():
                args.extend(["--build-arg", f"{key}={value}"])

        # Add target stage for multi-stage builds
        if "target" in options:
            args.extend(["--target", options["target"]])

        # Build operations can take longer, use extended timeout
        success, stdout, stderr = await _run_docker_command_async(args, timeout=600)

        if not success:
            return {"success": False, "error": stderr}

        return {
            "success": True,
            "result": {"image": resource_id, "build_path": build_path},
            "message": f"Built image: {resource_id}",
        }

    elif operation == "rmi":
        if not resource_id:
            return {
                "success": False,
                "error": "resource_id (image name) required for rmi operation",
            }

        args = ["rmi", resource_id]
        if options.get("force"):
            args.insert(1, "-f")

        success, stdout, stderr = await _run_docker_command_async(args)

        if not success:
            return {"success": False, "error": stderr}

        return {
            "success": True,
            "result": {"image": resource_id},
            "message": f"Removed image: {resource_id}",
        }

    # Network operations
    elif operation == "networks":
        args = ["network", "ls", "--format", "json"]

        success, stdout, stderr = await _run_docker_command_async(args)
        if not success:
            return {"success": False, "error": stderr}

        networks = []
        for line in stdout.strip().split("\n"):
            if line:
                try:
                    networks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return {
            "success": True,
            "result": {"networks": networks, "count": len(networks)},
            "message": f"Found {len(networks)} network(s)",
        }

    # Volume operations
    elif operation == "volumes":
        args = ["volume", "ls", "--format", "json"]

        success, stdout, stderr = await _run_docker_command_async(args)
        if not success:
            return {"success": False, "error": stderr}

        volumes = []
        for line in stdout.strip().split("\n"):
            if line:
                try:
                    volumes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return {
            "success": True,
            "result": {"volumes": volumes, "count": len(volumes)},
            "message": f"Found {len(volumes)} volume(s)",
        }

    else:
        return {
            "success": False,
            "error": f"Unknown operation: {operation}. Supported: ps, start, stop, restart, rm, logs, stats, exec, inspect, images, pull, build, rmi, networks, volumes",
        }
