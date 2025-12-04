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

"""Shared subprocess execution utilities.

This module provides consolidated subprocess execution functions used across
multiple tools (bash, git, docker, cicd, testing, etc.). Consolidating these
utilities reduces code duplication and ensures consistent error handling.

Features:
- Both sync and async subprocess execution
- Command safety checks with dangerous command blocking
- Specialized runners for git, docker, npm, pip
- Proper timeout handling with configurable defaults
- Output capture and structured result format
- Error categorization and logging
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes and Enums
# =============================================================================


class CommandErrorType(Enum):
    """Types of command execution errors."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"  # Command not found
    PERMISSION_DENIED = "permission_denied"
    WORKING_DIR_NOT_FOUND = "working_dir_not_found"
    DANGEROUS_COMMAND = "dangerous_command"
    EXECUTION_ERROR = "execution_error"
    UNKNOWN = "unknown"


@dataclass
class CommandResult:
    """Result of command execution."""

    success: bool
    stdout: str
    stderr: str
    return_code: int
    error_type: CommandErrorType
    error_message: Optional[str] = None
    command: Optional[str] = None
    working_dir: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "command": self.command,
            "working_dir": self.working_dir,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# Safety Configuration
# =============================================================================


# Commands that should never be executed
DANGEROUS_COMMANDS = frozenset(
    {
        "rm -rf /",
        "rm -rf /*",
        "dd",
        "mkfs",
        ":(){ :|:& };:",  # Fork bomb
        "> /dev/sda",
    }
)

# Patterns that indicate dangerous commands
DANGEROUS_PATTERNS = (
    "rm -rf /",
    "rm -rf /*",
    "rm -rf $HOME",
    "rm -rf ~",
    "dd if=/dev/",
    "dd of=/dev/",
    "mkfs.",
    "> /dev/sd",
    "wget | sh",
    "wget | bash",
    "curl | sh",
    "curl | bash",
    ":(){",  # Fork bomb variant
    "chmod 777 /",
    "chown root /",
)


def is_dangerous_command(command: str) -> bool:
    """Check if a command is potentially dangerous.

    Args:
        command: Command string to check.

    Returns:
        True if the command is dangerous, False otherwise.
    """
    command_lower = command.lower().strip()

    # Check exact matches
    if command_lower in DANGEROUS_COMMANDS:
        return True

    # Check for dangerous patterns
    return any(pattern in command_lower for pattern in DANGEROUS_PATTERNS)


# =============================================================================
# Tool Availability Checking
# =============================================================================


def is_tool_available(tool_name: str) -> bool:
    """Check if a command-line tool is available.

    Args:
        tool_name: Name of the tool to check (e.g., 'git', 'docker').

    Returns:
        True if the tool is available, False otherwise.
    """
    return shutil.which(tool_name) is not None


def check_git_available() -> bool:
    """Check if git is available."""
    if not is_tool_available("git"):
        return False
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_docker_available() -> bool:
    """Check if docker is available."""
    if not is_tool_available("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_npm_available() -> bool:
    """Check if npm is available."""
    if not is_tool_available("npm"):
        return False
    try:
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_pip_available() -> bool:
    """Check if pip is available."""
    try:
        result = subprocess.run(
            ["pip", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# =============================================================================
# Synchronous Execution
# =============================================================================


def run_command(
    args: Union[str, List[str]],
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 60,
    check_dangerous: bool = True,
    env: Optional[Dict[str, str]] = None,
    shell: bool = False,
) -> CommandResult:
    """Execute a command synchronously and return structured result.

    Args:
        args: Command to execute. Either a string (requires shell=True) or list of args.
        working_dir: Working directory for command execution.
        timeout: Timeout in seconds (default: 60).
        check_dangerous: Whether to check for dangerous commands (default: True).
        env: Environment variables to set.
        shell: Whether to use shell execution (default: False).

    Returns:
        CommandResult with execution details.
    """
    import time

    start_time = time.time()

    # Convert args to string for safety check
    cmd_str = args if isinstance(args, str) else " ".join(args)

    # Safety check
    if check_dangerous and is_dangerous_command(cmd_str):
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.DANGEROUS_COMMAND,
            error_message=f"Dangerous command blocked: {cmd_str}",
            command=cmd_str,
        )

    # Validate working directory
    if working_dir:
        working_dir = Path(working_dir)
        if not working_dir.exists():
            return CommandResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error_type=CommandErrorType.WORKING_DIR_NOT_FOUND,
                error_message=f"Working directory not found: {working_dir}",
                command=cmd_str,
                working_dir=str(working_dir),
            )

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
            env=env,
            shell=shell,
        )

        duration_ms = (time.time() - start_time) * 1000

        return CommandResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            error_type=(
                CommandErrorType.SUCCESS
                if result.returncode == 0
                else CommandErrorType.EXECUTION_ERROR
            ),
            error_message=result.stderr if result.returncode != 0 else None,
            command=cmd_str,
            working_dir=str(working_dir) if working_dir else None,
            duration_ms=duration_ms,
        )

    except subprocess.TimeoutExpired:
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.TIMEOUT,
            error_message=f"Command timed out after {timeout} seconds",
            command=cmd_str,
            working_dir=str(working_dir) if working_dir else None,
        )

    except FileNotFoundError as e:
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.NOT_FOUND,
            error_message=f"Command not found: {e}",
            command=cmd_str,
        )

    except PermissionError as e:
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.PERMISSION_DENIED,
            error_message=f"Permission denied: {e}",
            command=cmd_str,
        )

    except Exception as e:
        logger.exception("Unexpected error executing command: %s", cmd_str)
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.UNKNOWN,
            error_message=str(e),
            command=cmd_str,
        )


# =============================================================================
# Asynchronous Execution
# =============================================================================


async def run_command_async(
    command: str,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 60,
    check_dangerous: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    """Execute a shell command asynchronously and return structured result.

    Args:
        command: Shell command to execute.
        working_dir: Working directory for command execution.
        timeout: Timeout in seconds (default: 60).
        check_dangerous: Whether to check for dangerous commands (default: True).
        env: Environment variables to set.

    Returns:
        CommandResult with execution details.
    """
    import time

    start_time = time.time()

    # Safety check
    if check_dangerous and is_dangerous_command(command):
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.DANGEROUS_COMMAND,
            error_message=f"Dangerous command blocked: {command}",
            command=command,
        )

    # Validate working directory
    cwd = None
    if working_dir:
        cwd = Path(working_dir)
        if not cwd.exists():
            return CommandResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error_type=CommandErrorType.WORKING_DIR_NOT_FOUND,
                error_message=f"Working directory not found: {working_dir}",
                command=command,
                working_dir=str(working_dir),
            )

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return CommandResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                error_type=CommandErrorType.TIMEOUT,
                error_message=f"Command timed out after {timeout} seconds",
                command=command,
                working_dir=str(working_dir) if working_dir else None,
            )

        duration_ms = (time.time() - start_time) * 1000
        stdout_str = stdout.decode("utf-8") if stdout else ""
        stderr_str = stderr.decode("utf-8") if stderr else ""

        return CommandResult(
            success=process.returncode == 0,
            stdout=stdout_str,
            stderr=stderr_str,
            return_code=process.returncode or 0,
            error_type=(
                CommandErrorType.SUCCESS
                if process.returncode == 0
                else CommandErrorType.EXECUTION_ERROR
            ),
            error_message=stderr_str if process.returncode != 0 else None,
            command=command,
            working_dir=str(working_dir) if working_dir else None,
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.exception("Unexpected error executing async command: %s", command)
        return CommandResult(
            success=False,
            stdout="",
            stderr="",
            return_code=-1,
            error_type=CommandErrorType.UNKNOWN,
            error_message=str(e),
            command=command,
        )


# =============================================================================
# Specialized Runners
# =============================================================================


def run_git(
    *args: str,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 30,
) -> Tuple[bool, str, str]:
    """Execute a git command.

    Args:
        *args: Git command arguments (e.g., "status", "-s").
        working_dir: Repository directory.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_git_available():
        return False, "", "Git is not available"

    result = run_command(
        ["git", *args],
        working_dir=working_dir,
        timeout=timeout,
        check_dangerous=False,  # Git commands are generally safe
    )

    return result.success, result.stdout, result.stderr


def run_docker(
    *args: str,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute a docker command.

    Args:
        *args: Docker command arguments.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_docker_available():
        return False, "", "Docker is not available"

    result = run_command(
        ["docker", *args],
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


def run_npm(
    *args: str,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute an npm command.

    Args:
        *args: npm command arguments.
        working_dir: Project directory.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_npm_available():
        return False, "", "npm is not available"

    result = run_command(
        ["npm", *args],
        working_dir=working_dir,
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


def run_pip(
    *args: str,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute a pip command.

    Args:
        *args: pip command arguments.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    result = run_command(
        ["pip", *args],
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


async def run_git_async(
    *args: str,
    working_dir: Optional[Union[str, Path]] = None,
    timeout: int = 30,
) -> Tuple[bool, str, str]:
    """Execute a git command asynchronously.

    Args:
        *args: Git command arguments.
        working_dir: Repository directory.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_git_available():
        return False, "", "Git is not available"

    command = "git " + " ".join(args)
    result = await run_command_async(
        command,
        working_dir=working_dir,
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


async def run_docker_async(
    *args: str,
    timeout: int = 300,
) -> Tuple[bool, str, str]:
    """Execute a docker command asynchronously.

    Args:
        *args: Docker command arguments.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, stdout, stderr).
    """
    if not check_docker_available():
        return False, "", "Docker is not available"

    command = "docker " + " ".join(args)
    result = await run_command_async(
        command,
        timeout=timeout,
        check_dangerous=False,
    )

    return result.success, result.stdout, result.stderr


# =============================================================================
# Utility Functions
# =============================================================================


def parse_git_status(stdout: str) -> Dict[str, List[str]]:
    """Parse git status output into categorized files.

    Args:
        stdout: Output from 'git status --porcelain'.

    Returns:
        Dictionary with 'staged', 'modified', 'untracked' file lists.
    """
    result: Dict[str, List[str]] = {
        "staged": [],
        "modified": [],
        "untracked": [],
        "deleted": [],
    }

    for line in stdout.strip().split("\n"):
        if not line:
            continue

        status = line[:2]
        filename = line[3:].strip()

        if status[0] in ("A", "M", "D", "R"):
            result["staged"].append(filename)
        if status[1] == "M":
            result["modified"].append(filename)
        if status == "??":
            result["untracked"].append(filename)
        if status[1] == "D" or status[0] == "D":
            result["deleted"].append(filename)

    return result


def parse_docker_ps(stdout: str) -> List[Dict[str, str]]:
    """Parse docker ps output into container information.

    Args:
        stdout: Output from 'docker ps'.

    Returns:
        List of container info dictionaries.
    """
    containers = []
    lines = stdout.strip().split("\n")

    if len(lines) <= 1:
        return containers

    # Skip header line, parse data lines
    for line in lines[1:]:
        if not line.strip():
            continue

        # Simple parsing - assumes standard docker ps format
        parts = line.split()
        if len(parts) >= 2:
            containers.append(
                {
                    "id": parts[0],
                    "image": parts[1] if len(parts) > 1 else "",
                    "status": " ".join(parts[4:-2]) if len(parts) > 4 else "",
                    "name": parts[-1] if len(parts) > 0 else "",
                }
            )

    return containers
