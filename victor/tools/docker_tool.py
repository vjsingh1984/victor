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
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.decorators import tool


def _check_docker() -> bool:
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


def _run_docker_command(args: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
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


@tool
async def docker_ps(all: bool = False) -> Dict[str, Any]:
    """
    List Docker containers.

    Shows running containers by default, or all containers
    including stopped ones if all=True.

    Args:
        all: Include stopped containers (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - containers: List of container objects
        - count: Number of containers
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    args = ["ps", "--format", "json"]
    if all:
        args.append("--all")

    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    # Parse JSON lines
    containers = []
    for line in stdout.strip().split("\n"):
        if line:
            try:
                containers.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return {
        "success": True,
        "containers": containers,
        "count": len(containers)
    }


@tool
async def docker_images() -> Dict[str, Any]:
    """
    List Docker images.

    Returns all Docker images available on the local system.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - images: List of image objects
        - count: Number of images
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    args = ["images", "--format", "json"]
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    # Parse JSON lines
    images = []
    for line in stdout.strip().split("\n"):
        if line:
            try:
                images.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return {
        "success": True,
        "images": images,
        "count": len(images)
    }


@tool
async def docker_pull(image: str) -> Dict[str, Any]:
    """
    Pull Docker image from registry.

    Downloads the specified image from Docker Hub or another
    configured registry.

    Args:
        image: Image name (e.g., nginx:latest, ubuntu:22.04).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Pull operation output
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if not image:
        return {"success": False, "error": "Missing required parameter: image"}

    args = ["pull", image]
    success, stdout, stderr = _run_docker_command(args, timeout=300)  # 5 minutes

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "output": stdout,
        "message": f"Successfully pulled image: {image}"
    }


@tool
async def docker_run(
    image: str,
    name: Optional[str] = None,
    ports: Optional[List[str]] = None,
    env: Optional[List[str]] = None,
    volumes: Optional[List[str]] = None,
    command: Optional[str] = None,
    detach: bool = True,
) -> Dict[str, Any]:
    """
    Run a Docker container.

    Creates and starts a new container from the specified image.

    Args:
        image: Image name to run (required).
        name: Container name (optional).
        ports: Port mappings like ['80:80', '443:443'] (optional).
        env: Environment variables like ['KEY=value'] (optional).
        volumes: Volume mappings like ['/host:/container'] (optional).
        command: Command to run in container (optional).
        detach: Run in background (default: True).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - container_id: ID of created container
        - message: Success message
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if not image:
        return {"success": False, "error": "Missing required parameter: image"}

    args = ["run"]

    # Detached mode (default)
    if detach:
        args.append("-d")

    # Container name
    if name:
        args.extend(["--name", name])

    # Port mappings
    for port in (ports or []):
        args.extend(["-p", port])

    # Environment variables
    for e in (env or []):
        args.extend(["-e", e])

    # Volume mappings
    for volume in (volumes or []):
        args.extend(["-v", volume])

    # Image
    args.append(image)

    # Command (if provided)
    if command:
        args.extend(command.split())

    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "container_id": stdout.strip(),
        "message": f"Container started: {stdout.strip()}"
    }


@tool
async def docker_stop(container: str) -> Dict[str, Any]:
    """
    Stop running Docker container.

    Args:
        container: Container name or ID.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - message: Success message
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if not container:
        return {"success": False, "error": "Missing required parameter: container"}

    args = ["stop", container]
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "message": f"Container stopped: {container}"
    }


@tool
async def docker_start(container: str) -> Dict[str, Any]:
    """
    Start stopped Docker container.

    Args:
        container: Container name or ID.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - message: Success message
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if not container:
        return {"success": False, "error": "Missing required parameter: container"}

    args = ["start", container]
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "message": f"Container started: {container}"
    }


@tool
async def docker_restart(container: str) -> Dict[str, Any]:
    """
    Restart Docker container.

    Args:
        container: Container name or ID.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - message: Success message
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if not container:
        return {"success": False, "error": "Missing required parameter: container"}

    args = ["restart", container]
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "message": f"Container restarted: {container}"
    }


@tool
async def docker_rm(container: str) -> Dict[str, Any]:
    """
    Remove Docker container.

    Forcefully removes the specified container.

    Args:
        container: Container name or ID.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - message: Success message
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if not container:
        return {"success": False, "error": "Missing required parameter: container"}

    args = ["rm", "-f", container]  # Force remove
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "message": f"Container removed: {container}"
    }


@tool
async def docker_rmi(image: str) -> Dict[str, Any]:
    """
    Remove Docker image.

    Args:
        image: Image name or ID.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - message: Success message
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if not image:
        return {"success": False, "error": "Missing required parameter: image"}

    args = ["rmi", image]
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "message": f"Image removed: {image}"
    }


@tool
async def docker_logs(container: str, tail: int = 100) -> Dict[str, Any]:
    """
    Get Docker container logs.

    Args:
        container: Container name or ID.
        tail: Number of log lines to show (default: 100).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - logs: Container log output
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if not container:
        return {"success": False, "error": "Missing required parameter: container"}

    args = ["logs", "--tail", str(tail), container]
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "logs": stdout
    }


@tool
async def docker_stats(container: Optional[str] = None) -> Dict[str, Any]:
    """
    Get Docker container resource usage statistics.

    Args:
        container: Container name or ID (optional, shows all if not specified).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - stats: Resource usage statistics
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    args = ["stats", "--no-stream", "--format", "json"]
    if container:
        args.append(container)

    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "stats": stdout
    }


@tool
async def docker_inspect(container: Optional[str] = None, image: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspect Docker container or image.

    Returns detailed information about a container or image.

    Args:
        container: Container name or ID (one of container/image required).
        image: Image name or ID (one of container/image required).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - details: Inspection details as JSON
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    target = container or image

    if not target:
        return {
            "success": False,
            "error": "Missing required parameter: container or image"
        }

    args = ["inspect", target]
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "details": stdout
    }


@tool
async def docker_networks() -> Dict[str, Any]:
    """
    List Docker networks.

    Returns all Docker networks.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - networks: Network listing
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    args = ["network", "ls", "--format", "json"]
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "networks": stdout
    }


@tool
async def docker_volumes() -> Dict[str, Any]:
    """
    List Docker volumes.

    Returns all Docker volumes.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - volumes: Volume listing
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    args = ["volume", "ls", "--format", "json"]
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "volumes": stdout
    }


@tool
async def docker_exec(container: str, command: str) -> Dict[str, Any]:
    """
    Execute command in running Docker container.

    Args:
        container: Container name or ID.
        command: Command to execute.

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - output: Command output
        - error: Error message if failed
    """
    if not _check_docker():
        return {"success": False, "error": "Docker CLI not found. Please install Docker."}

    if not container:
        return {"success": False, "error": "Missing required parameter: container"}

    if not command:
        return {"success": False, "error": "Missing required parameter: command"}

    args = ["exec", container] + command.split()
    success, stdout, stderr = _run_docker_command(args)

    if not success:
        return {"success": False, "error": stderr}

    return {
        "success": True,
        "output": stdout
    }


# Keep class for backward compatibility
class DockerTool:
    """Deprecated: Use individual docker_* functions instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "DockerTool class is deprecated. Use docker_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
