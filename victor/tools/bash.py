"""Bash command execution tool."""

import asyncio
import shlex
from typing import Any, Dict

from victor.tools.base import BaseTool, ToolResult


class BashTool(BaseTool):
    """Tool for executing bash commands."""

    # Dangerous commands that should be blocked or require confirmation
    DANGEROUS_COMMANDS = {
        "rm -rf /",
        "dd",
        "mkfs",
        ":(){ :|:& };:",  # Fork bomb
        "> /dev/sda",
    }

    def __init__(self, timeout: int = 60, allow_dangerous: bool = False):
        """Initialize bash tool.

        Args:
            timeout: Command timeout in seconds
            allow_dangerous: Whether to allow dangerous commands
        """
        self.timeout = timeout
        self.allow_dangerous = allow_dangerous

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return "Execute a bash command and return its output"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for command execution (optional)",
                },
            },
            "required": ["command"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute bash command.

        Args:
            command: Command to execute
            working_dir: Working directory (optional)

        Returns:
            ToolResult with command output
        """
        command = kwargs.get("command")
        working_dir = kwargs.get("working_dir")

        if not command:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: command",
            )

        # Check for dangerous commands
        if not self.allow_dangerous and self._is_dangerous(command):
            return ToolResult(
                success=False,
                output=None,
                error=f"Dangerous command blocked: {command}",
            )

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
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Command timed out after {self.timeout} seconds",
                )

            stdout_str = stdout.decode("utf-8") if stdout else ""
            stderr_str = stderr.decode("utf-8") if stderr else ""

            return ToolResult(
                success=process.returncode == 0,
                output={
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "return_code": process.returncode,
                },
                metadata={
                    "command": command,
                    "working_dir": working_dir,
                },
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Working directory not found: {working_dir}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to execute command: {str(e)}",
            )

    def _is_dangerous(self, command: str) -> bool:
        """Check if command is potentially dangerous.

        Args:
            command: Command to check

        Returns:
            True if dangerous, False otherwise
        """
        command_lower = command.lower().strip()

        # Check exact matches
        if command_lower in self.DANGEROUS_COMMANDS:
            return True

        # Check for dangerous patterns
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf /*",
            "dd if=",
            "mkfs.",
            "> /dev/sd",
            "wget | sh",
            "curl | sh",
            "curl | bash",
        ]

        return any(pattern in command_lower for pattern in dangerous_patterns)
