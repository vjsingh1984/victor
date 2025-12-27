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

"""Bash command execution tool with readonly mode support."""

import asyncio
import platform
import shlex
from typing import Any, Dict, List, Optional, Set

from victor.config.timeouts import ProcessTimeouts
from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
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


# Platform-specific readonly commands - safe for exploration/analysis
# These commands cannot modify state, only read
READONLY_COMMANDS_UNIX: Set[str] = {
    # Navigation & listing
    "pwd",
    "ls",
    "ll",
    "la",
    "tree",
    "find",
    "locate",
    "which",
    "whereis",
    "type",
    # File content viewing
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",
    "file",
    "stat",
    "md5sum",
    "sha256sum",
    # Text search & processing (readonly)
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ag",
    "awk",
    "sed",  # Only with -n (no in-place edit)
    "cut",
    "sort",
    "uniq",
    "diff",
    "cmp",
    # System info
    "uname",
    "hostname",
    "whoami",
    "id",
    "date",
    "uptime",
    "df",
    "du",
    "free",
    "top",
    "ps",
    "pgrep",
    "lsof",
    "env",
    "printenv",
    "echo",
    "printf",
    # Git (readonly commands)
    "git",  # Will filter subcommands
    # Package info (readonly)
    "pip",  # Will filter subcommands
    "npm",  # Will filter subcommands
    "cargo",  # Will filter subcommands
    "go",  # Will filter subcommands
    # Development
    "python",  # Will filter for -c, --version, etc
    "node",  # Will filter for -e, --version, etc
}

READONLY_COMMANDS_WINDOWS: Set[str] = {
    # Navigation & listing
    "cd",
    "dir",
    "tree",
    "where",
    "type",
    # File content viewing
    "more",
    "find",
    "findstr",
    # System info
    "hostname",
    "whoami",
    "date",
    "time",
    "ver",
    "systeminfo",
    "set",
    "echo",
    # Git (readonly)
    "git",
}

# Git subcommands that are readonly
GIT_READONLY_SUBCOMMANDS: Set[str] = {
    "status",
    "log",
    "show",
    "diff",
    "branch",
    "tag",
    "remote",
    "ls-files",
    "ls-tree",
    "rev-parse",
    "rev-list",
    "describe",
    "shortlog",
    "blame",
    "grep",
    "config",  # readonly by default (no --global, --unset)
    "reflog",
    "stash",  # list only
    "worktree",  # list only
    "cat-file",
    "ls-remote",
    "check-ignore",
    "check-attr",
    "name-rev",
    "verify-commit",
    "verify-tag",
    "for-each-ref",
}

# pip subcommands that are readonly
PIP_READONLY_SUBCOMMANDS: Set[str] = {
    "list",
    "show",
    "freeze",
    "check",
    "config",  # readonly by default
    "debug",
    "help",
    "search",
    "index",
}

# npm subcommands that are readonly
NPM_READONLY_SUBCOMMANDS: Set[str] = {
    "list",
    "ls",
    "view",
    "show",
    "info",
    "search",
    "help",
    "config",  # readonly by default
    "outdated",
    "audit",
    "explain",
    "pkg",
    "query",
    "version",
    "why",
}


def _get_readonly_commands() -> Set[str]:
    """Get platform-specific readonly commands."""
    if platform.system() == "Windows":
        return READONLY_COMMANDS_WINDOWS
    return READONLY_COMMANDS_UNIX


def _extract_base_command(cmd: str) -> str:
    """Extract the base command from a command string."""
    try:
        parts = shlex.split(cmd.strip())
        if parts:
            return parts[0].lower()
    except ValueError:
        # Handle shlex parsing errors (unbalanced quotes, etc)
        parts = cmd.strip().split()
        if parts:
            return parts[0].lower()
    return ""


def _extract_subcommand(cmd: str, base_cmd: str) -> Optional[str]:
    """Extract subcommand for commands like git, pip, npm."""
    try:
        parts = shlex.split(cmd.strip())
        if len(parts) >= 2 and parts[0].lower() == base_cmd:
            # Skip options to find subcommand
            for part in parts[1:]:
                if not part.startswith("-"):
                    return part.lower()
    except ValueError:
        pass
    return None


def _is_readonly_command(cmd: str) -> bool:
    """Check if command is a readonly command.

    Returns True if the command is safe for read-only operations.
    """
    readonly_commands = _get_readonly_commands()
    base_cmd = _extract_base_command(cmd)

    if not base_cmd:
        return False

    # Check if base command is in readonly set
    if base_cmd not in readonly_commands:
        return False

    # Special handling for commands with subcommands
    if base_cmd == "git":
        subcommand = _extract_subcommand(cmd, "git")
        return subcommand in GIT_READONLY_SUBCOMMANDS if subcommand else False

    if base_cmd == "pip" or base_cmd == "pip3":
        subcommand = _extract_subcommand(cmd, base_cmd)
        return subcommand in PIP_READONLY_SUBCOMMANDS if subcommand else False

    if base_cmd == "npm":
        subcommand = _extract_subcommand(cmd, "npm")
        return subcommand in NPM_READONLY_SUBCOMMANDS if subcommand else False

    # Check for sed with -i (in-place edit)
    if base_cmd == "sed" and "-i" in cmd:
        return False

    # Check for dangerous redirect patterns in readonly mode
    if ">" in cmd or ">>" in cmd:
        return False

    # Check for pipe to shell
    if "| sh" in cmd or "| bash" in cmd or "|sh" in cmd or "|bash" in cmd:
        return False

    return True


def get_allowed_readonly_commands() -> List[str]:
    """Return list of allowed readonly commands for LLM reference."""
    commands = list(_get_readonly_commands())
    commands.sort()
    return commands


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


@tool(
    category="execution",
    priority=Priority.CRITICAL,  # Always available
    access_mode=AccessMode.EXECUTE,  # Executes external commands
    danger_level=DangerLevel.HIGH,  # Arbitrary command execution is risky
    # Registry-driven metadata for tool selection and loop detection
    progress_params=["cmd"],  # Different commands indicate progress, not loops
    stages=["execution", "verification"],  # Conversation stages where relevant
    task_types=["action", "analysis"],  # Task types for classification-aware selection
    execution_category=ExecutionCategory.EXECUTE,  # Cannot run safely in parallel
    mandatory_keywords=[
        "run command",
        "execute",
        "shell",
        # Git diff/compare operations (from MANDATORY_TOOL_KEYWORDS)
        "diff",
        "show changes",
        "git diff",
        "show diff",
        "compare",
        # Running/executing (from MANDATORY_TOOL_KEYWORDS)
        "run",
        "install",
        "test",
        # Count operations (from MANDATORY_TOOL_KEYWORDS)
        "count",
        "how many",
    ],  # Force inclusion
    keywords=["bash", "shell", "command", "run", "execute", "terminal", "cli"],
)
async def shell(
    cmd: str,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    dangerous: bool = False,
) -> Dict[str, Any]:
    """Run shell command with safety checks. Returns stdout/stderr/return_code.

    Args:
        cmd: Shell command to run
        cwd: Working directory
        timeout: Seconds before timeout
        dangerous: Allow risky commands
    """
    if not cmd:
        return {
            "success": False,
            "error": "Missing required parameter: cmd",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    # Apply default timeout from centralized config
    if timeout is None:
        timeout = ProcessTimeouts.BASH_DEFAULT

    # Check for dangerous commands
    if not dangerous and _is_dangerous(cmd):
        return {
            "success": False,
            "error": f"Dangerous command blocked: {cmd}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    # Validate working directory exists before execution
    if cwd:
        import os

        if not os.path.isdir(cwd):
            return {
                "success": False,
                "error": f"Working directory does not exist: {cwd}",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

    try:
        # Create subprocess
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
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
            "command": cmd,
            "working_dir": cwd,
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
            "error": f"Working directory not found: {cwd}",
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


@tool(
    category="exploration",
    priority=Priority.HIGH,  # Available for exploration
    access_mode=AccessMode.READONLY,  # Only read operations
    danger_level=DangerLevel.SAFE,  # Readonly commands are safe
    # Registry-driven metadata for tool selection
    stages=["initial", "exploring", "analyzing"],  # Available in exploration stages
    task_types=["exploration", "analysis", "understanding"],
    execution_category=ExecutionCategory.READ_ONLY,  # No state changes
    keywords=[
        "pwd",
        "cd",
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "tree",
        "file",
        "wc",
        "directory",
        "navigate",
        "explore",
        "git status",
        "git log",
        "git diff",
    ],
    use_cases=[
        "navigating directories",
        "viewing file contents",
        "exploring codebase structure",
        "checking git status",
        "viewing git history",
        "finding files",
    ],
    examples=[
        "pwd - show current directory",
        "ls -la - list files with details",
        "cat file.txt - view file contents",
        "git status - check git status",
        "git log --oneline -10 - view recent commits",
        "find . -name '*.py' - find Python files",
    ],
)
async def shell_readonly(
    cmd: str,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute readonly shell commands for exploration (pwd, ls, cat, grep, git status, etc).

    This tool only allows safe, readonly commands that cannot modify files or state.
    Use this for navigating directories, viewing files, and exploring the codebase.

    Allowed commands include: pwd, ls, cat, head, tail, grep, find, tree, file, wc,
    git status/log/diff/branch, and similar readonly operations.

    Args:
        cmd: Readonly shell command to run
        cwd: Working directory (optional, defaults to current)
        timeout: Seconds before timeout (optional)

    Returns:
        Dict with stdout, stderr, return_code, cwd context
    """
    import os
    from pathlib import Path

    if not cmd:
        return {
            "success": False,
            "error": "Missing required parameter: cmd",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "cwd": os.getcwd(),
        }

    # Validate command is readonly
    if not _is_readonly_command(cmd):
        base_cmd = _extract_base_command(cmd)
        allowed = get_allowed_readonly_commands()
        return {
            "success": False,
            "error": (
                f"Command '{base_cmd}' is not allowed in readonly mode. "
                f"Allowed commands: {', '.join(allowed[:15])}... "
                "Use 'shell' tool for other commands."
            ),
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "cwd": os.getcwd(),
        }

    # Apply default timeout from centralized config
    if timeout is None:
        timeout = ProcessTimeouts.BASH_DEFAULT

    # Validate working directory exists before execution
    effective_cwd = cwd or os.getcwd()
    if not os.path.isdir(effective_cwd):
        return {
            "success": False,
            "error": f"Working directory does not exist: {effective_cwd}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "cwd": os.getcwd(),
        }

    try:
        # Create subprocess
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=effective_cwd,
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
                "cwd": effective_cwd,
            }

        stdout_str = stdout.decode("utf-8") if stdout else ""
        stderr_str = stderr.decode("utf-8") if stderr else ""

        # Compute relative path from current working directory
        try:
            relative_cwd = str(Path(effective_cwd).relative_to(Path.cwd()))
        except ValueError:
            relative_cwd = effective_cwd

        result = {
            "success": process.returncode == 0,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "return_code": process.returncode,
            "command": cmd,
            "cwd": os.getcwd(),  # Root cwd for context
            "working_dir": effective_cwd,
            "relative_working_dir": relative_cwd if relative_cwd != "." else ".",
        }

        # Include informative error message when command fails
        if process.returncode != 0:
            error_parts = []
            error_parts.append(f"Command failed with exit code {process.returncode}")
            if stderr_str.strip():
                stderr_preview = stderr_str.strip()
                if len(stderr_preview) > 500:
                    stderr_preview = stderr_preview[:250] + "\n...\n" + stderr_preview[-250:]
                error_parts.append(f"stderr: {stderr_preview}")
            result["error"] = "\n".join(error_parts)

        return result

    except FileNotFoundError:
        return {
            "success": False,
            "error": f"Working directory not found: {effective_cwd}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "cwd": os.getcwd(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to execute command: {str(e)}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
            "cwd": os.getcwd(),
        }
