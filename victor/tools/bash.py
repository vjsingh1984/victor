"""Bash command execution tool."""

import asyncio
from typing import Dict, Any, Optional

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
    timeout: int = 60,
    allow_dangerous: bool = False
) -> Dict[str, Any]:
    """
    Execute a bash command and return its output.

    This tool allows executing shell commands with safety checks to prevent
    dangerous operations. Commands are executed with a configurable timeout
    and working directory.

    Args:
        command: The bash command to execute.
        working_dir: Working directory for command execution (optional).
        timeout: Command timeout in seconds (default: 60).
        allow_dangerous: Whether to allow potentially dangerous commands (default: False).

    Returns:
        A dictionary containing:
        - stdout: Standard output from the command
        - stderr: Standard error from the command
        - return_code: Exit code of the command
        - success: Whether the command succeeded (return_code == 0)
    """
    if not command:
        return {
            "success": False,
            "error": "Missing required parameter: command",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    # Check for dangerous commands
    if not allow_dangerous and _is_dangerous(command):
        return {
            "success": False,
            "error": f"Dangerous command blocked: {command}",
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

        return {
            "success": process.returncode == 0,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "return_code": process.returncode,
            "command": command,
            "working_dir": working_dir,
        }

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
