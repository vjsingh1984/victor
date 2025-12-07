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

"""Bash command execution tool."""

import asyncio
from typing import Dict, Any, Optional

from victor.config.timeouts import ProcessTimeouts
from victor.tools.decorators import tool


# Dangerous commands that should be blocked
DANGEROUS_COMMANDS = {
    "rm -rf /",
    "dd",
    "mkfs",
    ":(){ :|:& };:",  # Fork bomb
    "> /dev/sda",
}

DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "dd if=",
    "mkfs.",
    "> /dev/sd",
    "wget | sh",
    "curl | sh",
    "curl | bash",
]


def _is_dangerous(command: str) -> bool:
    """Check if command is potentially dangerous.

    Args:
        command: Command to check

    Returns:
        True if dangerous, False otherwise
    """
    command_lower = command.lower().strip()

    # Check exact matches
    if command_lower in DANGEROUS_COMMANDS:
        return True

    # Check for dangerous patterns
    return any(pattern in command_lower for pattern in DANGEROUS_PATTERNS)


@tool
async def execute_bash(
    command: str,
    working_dir: Optional[str] = None,
    timeout: Optional[int] = None,
    allow_dangerous: bool = False,
) -> Dict[str, Any]:
    """
    Execute a bash command and return its output.

    This tool allows executing shell commands with safety checks to prevent
    dangerous operations. Commands are executed with a configurable timeout
    and working directory.

    Args:
        command: The bash command to execute.
        working_dir: Working directory for command execution (optional).
        timeout: Command timeout in seconds (default: ProcessTimeouts.BASH_DEFAULT).
        allow_dangerous: Whether to allow potentially dangerous commands (default: False).

    Returns:
        A dictionary containing:
        - stdout: Standard output from the command
        - stderr: Standard error from the command
        - return_code: Exit code of the command
        - success: Whether the command succeeded (return_code == 0)

    Examples:
        Run tests:
            await execute_bash("pytest tests/", timeout=120)

        Install dependencies:
            await execute_bash("pip install -r requirements.txt")

        Check git status:
            await execute_bash("git status", working_dir="/path/to/repo")

        List Python files:
            await execute_bash("find . -name '*.py' | head -10")

        Run with longer timeout:
            await execute_bash("npm install", timeout=300)
    """
    if not command:
        return {
            "success": False,
            "error": "Missing required parameter: command",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    # Apply default timeout from centralized config
    if timeout is None:
        timeout = ProcessTimeouts.BASH_DEFAULT

    # Check for dangerous commands
    if not allow_dangerous and _is_dangerous(command):
        return {
            "success": False,
            "error": f"Dangerous command blocked: {command}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    # Validate working directory exists before execution
    if working_dir:
        import os

        if not os.path.isdir(working_dir):
            return {
                "success": False,
                "error": f"Working directory does not exist: {working_dir}",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

    try:
        # Create subprocess
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        # Wait for completion with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

        stdout_str = stdout.decode("utf-8") if stdout else ""
        stderr_str = stderr.decode("utf-8") if stderr else ""

        result = {
            "success": process.returncode == 0,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "return_code": process.returncode,
            "command": command,
            "working_dir": working_dir,
        }

        # Include informative error message when command fails
        if process.returncode != 0:
            error_parts = []
            error_parts.append(f"Command failed with exit code {process.returncode}")
            if stderr_str.strip():
                # Truncate stderr if too long, keeping first and last parts
                stderr_preview = stderr_str.strip()
                if len(stderr_preview) > 500:
                    stderr_preview = stderr_preview[:250] + "\n...\n" + stderr_preview[-250:]
                error_parts.append(f"stderr: {stderr_preview}")
            elif stdout_str.strip():
                # Some commands output errors to stdout
                stdout_preview = stdout_str.strip()
                if len(stdout_preview) > 300:
                    stdout_preview = stdout_preview[:150] + "..." + stdout_preview[-150:]
                error_parts.append(f"output: {stdout_preview}")
            result["error"] = "\n".join(error_parts)

        return result

    except FileNotFoundError:
        return {
            "success": False,
            "error": f"Working directory not found: {working_dir}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to execute command: {str(e)}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }
