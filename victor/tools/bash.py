from __future__ import annotations

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
import logging
import os
import platform
import re
import shlex
import shutil
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

from victor.config.timeouts import ProcessTimeouts
from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool

# Dangerous commands that should be blocked
# Consolidated dangerous command detection — single source of truth.
from victor.security.command_safety import (
    is_dangerous_command as _is_dangerous_consolidated,
)

# Platform-specific readonly commands - safe for exploration/analysis
# These commands cannot modify state, only read
READONLY_COMMANDS_UNIX: Set[str] = {
    # Navigation & listing
    "pwd",
    "ls",
    "ll",
    "la",
    "cd",  # Directory navigation (read-only, doesn't modify files)
    "source",  # Shell environment (venv activation)
    ".",       # Shell environment (venv activation)
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
    "pip3",
    "npm",  # Will filter subcommands
    "cargo",  # Will filter subcommands
    "go",  # Will filter subcommands
    "make",  # Build/Verify
    # Development
    "python",  # Will filter for -m pytest, -c, --version, etc
    "python3",
    "pytest",  # Testing (read-heavy, usually safe for verification)
    "tox",
    "node",  # Will filter for -e, --version, etc
    # CLI tools (readonly subcommands)
    "gh",  # GitHub CLI
    "az",  # Azure CLI
    "kubectl",  # Kubernetes CLI
    # Research tools
    "arxiv",  # arXiv CLI
    "web_search",  # Web search tool
}

READONLY_COMMANDS_WINDOWS: Set[str] = {
    # Navigation & listing
    "cd",
    "dir",
    "tree",
    "where",
    "type",
    "source",
    ".",
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
    "wheel",   # Building wheels is usually safe (readonly source)
    "build",   # Building is usually safe
    "inspect",
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

# GitHub CLI (gh) readonly subcommands
GH_READONLY_SUBCOMMANDS: Set[str] = {
    "view",
    "list",
    "search",
    "repo",
    "issue",
    "pr",
    "release",
    "workflow",
    "run",
    "actions",
    "auth",
    "config",
    "secret",
    "variable",
    "environment",
}

# Azure CLI (az) readonly subcommands
AZ_READONLY_SUBCOMMANDS: Set[str] = {
    "list",
    "show",
    "find",
    "account",
    "config",
    "monitor",
    "log",
    "metrics",
}

# Kubernetes (kubectl) readonly subcommands
KUBECTL_READONLY_SUBCOMMANDS: Set[str] = {
    "get",
    "describe",
    "logs",
    "top",
    "api-resources",
    "api-versions",
    "cluster-info",
    "version",
    "auth",
    "certificate",
    "cp",
    "diff",
    "explain",
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


def _split_compound_command(cmd: str) -> List[str]:
    """Split a compound command into individual components.

    Handles &&, ||, and ; operators while respecting quoted strings.
    Returns a list of command strings.
    """
    components = []
    current = []
    i = 0
    in_quote = None  # None, '"', or "'"
    in_escape = False

    # Use variables to avoid quote escaping issues
    single_quote = "'"
    double_quote = '"'
    quote_chars = {single_quote, double_quote}

    while i < len(cmd):
        char = cmd[i]

        if in_escape:
            current.append(char)
            in_escape = False
            i += 1
            continue

        if char == "\\":
            current.append(char)
            in_escape = True
            i += 1
            continue

        if char in quote_chars and (not in_quote or in_quote == char):
            if in_quote == char:
                in_quote = None
            else:
                in_quote = char
            current.append(char)
            i += 1
            continue

        # Check for compound operators (only when not in quotes)
        if not in_quote:
            # Check for && (must be & followed by &, not part of other text)
            if char == "&" and i + 1 < len(cmd) and cmd[i + 1] == "&":
                component = "".join(current).strip()
                if component:
                    components.append(component)
                current = []
                i += 2
                continue
            # Check for ||
            if char == "|" and i + 1 < len(cmd) and cmd[i + 1] == "|":
                component = "".join(current).strip()
                if component:
                    components.append(component)
                current = []
                i += 2
                continue
            # Check for ; (not part of ;; or other constructs)
            if char == ";":
                # Skip if it's ;; (used in some shells like case statements)
                if i + 1 < len(cmd) and cmd[i + 1] == ";":
                    current.append(char)
                    i += 1
                else:
                    component = "".join(current).strip()
                    if component:
                        components.append(component)
                    current = []
                i += 1
                continue

        current.append(char)
        i += 1

    # Add the last component
    component = "".join(current).strip()
    if component:
        components.append(component)

    return components if components else [cmd.strip()]


def _validate_readonly_command(cmd: str) -> tuple[bool, str]:
    """Validate if a command is readonly and return (is_valid, failing_cmd)."""
    # Split compound commands and validate each component
    components = _split_compound_command(cmd)

    if len(components) > 1:
        # Compound command: all components must be readonly
        for comp in components:
            valid, failing = _validate_readonly_command(comp.strip())
            if not valid:
                return False, failing
        return True, ""

    readonly_commands = _get_readonly_commands()
    base_cmd = _extract_base_command(cmd)

    if not base_cmd:
        return False, cmd

    # Check if base command is in readonly set
    if base_cmd not in readonly_commands:
        return False, base_cmd

    # Special handling for commands with subcommands
    if base_cmd == "git":
        subcommand = _extract_subcommand(cmd, "git")
        if not subcommand or subcommand not in GIT_READONLY_SUBCOMMANDS:
            return False, f"git {subcommand or ''}"
        return True, ""

    if base_cmd in {"pip", "pip3"}:
        subcommand = _extract_subcommand(cmd, base_cmd)
        if not subcommand or subcommand not in PIP_READONLY_SUBCOMMANDS:
            return False, f"{base_cmd} {subcommand or ''}"
        return True, ""

    if base_cmd == "npm":
        subcommand = _extract_subcommand(cmd, "npm")
        # Allow test, list, view, show, etc.
        if subcommand in NPM_READONLY_SUBCOMMANDS or subcommand == "test":
            return True, ""
        return False, f"npm {subcommand or ''}"

    if base_cmd == "cargo":
        subcommand = _extract_subcommand(cmd, "cargo")
        # Allow test, list, metadata, check, verify, etc.
        if subcommand in {"test", "check", "metadata", "list", "verify", "tree"}:
            return True, ""
        return False, f"cargo {subcommand or ''}"

    if base_cmd == "go":
        subcommand = _extract_subcommand(cmd, "go")
        # Allow test, list, version, etc.
        if subcommand in {"test", "list", "version", "help", "doc"}:
            return True, ""
        return False, f"go {subcommand or ''}"

    if base_cmd == "make":
        # Allow make test, make check, make help
        if any(target in cmd for target in ["test", "check", "help", "--version"]):
            return True, ""
        # 'make' alone often defaults to a build/test target, but we'll be slightly restrictive
        return False, "make (non-test target)"

    if base_cmd == "gh":
        subcommand = _extract_subcommand(cmd, "gh")
        if not subcommand or subcommand not in GH_READONLY_SUBCOMMANDS:
            return False, f"gh {subcommand or ''}"
        return True, ""

    if base_cmd == "az":
        subcommand = _extract_subcommand(cmd, "az")
        if not subcommand or subcommand not in AZ_READONLY_SUBCOMMANDS:
            return False, f"az {subcommand or ''}"
        return True, ""

    if base_cmd == "kubectl":
        subcommand = _extract_subcommand(cmd, "kubectl")
        if not subcommand or subcommand not in KUBECTL_READONLY_SUBCOMMANDS:
            return False, f"kubectl {subcommand or ''}"
        return True, ""

    # Python handling: allow -m pytest, -c, --version, -V, -h, --help
    if base_cmd in {"python", "python3"}:
        # Check for dangerous flags like -i (interactive) or -u
        if "-m pytest" in cmd:
            return True, ""
        if "-c" in cmd or "--version" in cmd or "-V" in cmd or "-h" in cmd or "--help" in cmd:
            return True, ""
        # If it's just 'python script.py', it depends on the script, but we allow it
        # as python is in the readonly set. We could be stricter here.
        return True, ""

    # Check for sed with -i (in-place edit)
    if base_cmd == "sed" and "-i" in cmd:
        return False, "sed -i"

    # Check for dangerous redirect patterns in readonly mode.
    redirect_check = re.sub(r"\s*\d?>\s*/dev/null\b", "", cmd)
    redirect_check = re.sub(r"\s*\d?>&\d\b", "", redirect_check)
    if ">" in redirect_check or ">>" in redirect_check:
        return False, "redirection (>)"

    # Check for pipe to shell
    if any(p in cmd for p in ["| sh", "| bash", "|sh", "|bash"]):
        return False, "pipe to shell"

    return True, ""


def _is_readonly_command(cmd: str) -> bool:
    """Check if command is a readonly command.

    Returns True if the command is safe for read-only operations.
    """
    valid, _ = _validate_readonly_command(cmd)
    return valid


def get_allowed_readonly_commands() -> List[str]:
    """Return list of allowed readonly commands for LLM reference."""
    commands = list(_get_readonly_commands())
    commands.sort()
    return commands


# =========================================================================
# Command Optimizer Pipeline
# =========================================================================
# Pluggable chain of command optimizers applied before execution.
# Each optimizer is a callable (str) -> str that may rewrite the command
# for better performance. Optimizers are applied in registration order.
# =========================================================================

_command_optimizers: List[Any] = []


def register_command_optimizer(optimizer: Any) -> Any:
    """Register a command optimizer. Can be used as a decorator."""
    _command_optimizers.append(optimizer)
    return optimizer


def optimize_command(cmd: str) -> str:
    """Apply all registered command optimizers to a shell command."""
    for opt in _command_optimizers:
        cmd = opt(cmd)
    return cmd


@register_command_optimizer
def _optimize_grep_to_rg(cmd: str) -> str:
    """Replace slow recursive grep with ripgrep (rg) when available.

    grep -r/-R on large repos can hang for minutes. ripgrep is 10-100x faster
    because it respects .gitignore, uses memory-mapped I/O, and parallelizes.
    Basic grep flags (-n, -i, -l, -c, -w, -e) are compatible with rg.
    """
    if not re.match(r"^grep\s+.*-[rR]", cmd) and not re.match(r"^grep\s+-[a-zA-Z]*[rR]", cmd):
        return cmd

    if not shutil.which("rg"):
        return cmd

    # Replace 'grep' with 'rg' and remove -r/-R (rg is recursive by default)
    optimized = re.sub(r"^grep\b", "rg", cmd)
    optimized = re.sub(
        r"\s-([a-zA-Z]*)r([a-zA-Z]*)",
        lambda m: (f" -{m.group(1)}{m.group(2)}" if m.group(1) or m.group(2) else ""),
        optimized,
    )
    optimized = re.sub(
        r"\s-([a-zA-Z]*)R([a-zA-Z]*)",
        lambda m: (f" -{m.group(1)}{m.group(2)}" if m.group(1) or m.group(2) else ""),
        optimized,
    )
    # Clean up empty flag groups and extra whitespace
    optimized = re.sub(r"\s-\s", " ", optimized)
    optimized = re.sub(r"\s+", " ", optimized).strip()

    if optimized != cmd:
        logger.info(f"Shell optimizer: grep→rg rewrite: {cmd!r} → {optimized!r}")

    return optimized


def _is_dangerous(command: str) -> bool:
    """Check if command is potentially dangerous.

    Delegates to the consolidated command safety module.

    Args:
        command: Command to check

    Returns:
        True if dangerous, False otherwise
    """
    return _is_dangerous_consolidated(command)


@tool(
    category="execution",
    priority=Priority.CRITICAL,  # Always available
    access_mode=AccessMode.EXECUTE,  # Executes external commands
    danger_level=DangerLevel.HIGH,  # Arbitrary command execution is risky
    # Registry-driven metadata for tool selection and loop detection
    signature_params=["cmd"],  # Different commands indicate progress, not loops
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
        # Database operations
        "database",
        "sqlite",
        "query",
        "sql",
    ],  # Force inclusion
    keywords=[
        "bash",
        "shell",
        "command",
        "run",
        "execute",
        "terminal",
        "cli",
        "sqlite3",
        "database",
        "sql",
    ],
)
async def shell(
    cmd: str,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    dangerous: bool = False,
    readonly: bool = True,
    stdout_limit: Optional[int] = None,
    stderr_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a shell command. The `cmd` parameter is required.

    Examples:
        shell(cmd="ls -la")
        shell(cmd="git status")
        shell(cmd='sqlite3 data.db ".tables"')  # Database operations
        shell(cmd='sqlite3 data.db "SELECT * FROM users LIMIT 10"')

    For database files (SQLite, PostgreSQL, MySQL):
    - Use shell(cmd='sqlite3 file.db ".tables"') to list tables
    - Use shell(cmd='sqlite3 file.db "SELECT * FROM table LIMIT 10"') to query
    - Use shell(cmd='psql -h localhost -U user -d db -c "SELECT * FROM table"') for PostgreSQL
    - Use shell(cmd='mysql -u user -p db -e "SHOW TABLES"') for MySQL

    For multiline scripts or quote-heavy payloads, prefer a heredoc inside
    `cmd` instead of deeply nested shell escaping. Example:
        shell(cmd="python - <<'PY'\\nprint('hello')\\nPY")

    Args:
        cmd: The shell command string to execute (required)
        cwd: Working directory for the command
        timeout: Maximum seconds before timeout
        dangerous: Set true only for destructive commands (rm, kill, etc.)
        readonly: Defaults to True. Set False only when the command must mutate state
            or invoke non-readonly subcommands.
        stdout_limit: Max lines for stdout (None=unlimited, default: 10000)
        stderr_limit: Max lines for stderr (None=unlimited, default: 2000)

    Returns:
        Dict with stdout, stderr, return_code keys
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

    # Apply command optimizer pipeline (grep→rg, etc.)
    cmd = optimize_command(cmd)

    # Redirect broad recursive search commands to code_search (when available).
    # Models bypass the semantic index by calling shell("rg ...") directly.
    # Allow targeted single-file searches (grep -n "pattern" specific_file.py)
    # since those are precise and don't benefit from semantic search.
    import re as _re

    _base_cmd = cmd.strip().split("|")[0].strip()
    if _re.match(r"^\s*(rg|grep|ag|ack)\s+", _base_cmd, _re.IGNORECASE):
        # Allow targeted searches: grep on a specific file path (not recursive)
        _is_recursive = bool(_re.search(r"\s-[a-zA-Z]*r[a-zA-Z]*\s", _base_cmd))
        _targets_file = bool(
            _re.search(r"\s[\w./~\-]+\.\w{1,10}\s*$", _base_cmd)
            or _re.search(r"\s[\w./~\-]+\.\w{1,10}\s*\|", cmd.strip())
        )
        # Always allow grep targeting library/venv files — code_search only covers project code
        _targets_external = bool(
            _re.search(r"(\.venv|site-packages|/lib/python\d|/usr/lib|/usr/local/lib)", _base_cmd)
        )
        # Detect command substitution in the file path (e.g. "$(python -c '...')/file.py")
        _has_cmd_sub = bool(_re.search(r"\$\(", _base_cmd))

        if not _targets_external and (not _targets_file or _is_recursive):
            if _has_cmd_sub:
                error_msg = (
                    "Command substitution in file paths is not supported for grep. "
                    "Resolve the path first, then use read() or grep with the literal path. "
                    "Example: shell(cmd='python -c \"import arxiv; print(arxiv.__file__)\"') "
                    "then read(path='<result>')."
                )
            else:
                error_msg = (
                    "Use code_search(query='...') instead of shell search commands for project code. "
                    "code_search uses the semantic index and is more reliable. "
                    "For library/venv files use read(path='...') directly. "
                    "Example: code_search(query='FilePathField', mode='semantic')"
                )
            return {
                "success": False,
                "error": error_msg,
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

    # Check for dangerous commands
    if not dangerous and _is_dangerous(cmd):
        return {
            "success": False,
            "error": f"Dangerous command blocked: {cmd}",
            "stdout": "",
            "stderr": "",
            "return_code": -1,
        }

    # Check readonly mode restrictions
    if readonly:
        is_valid, failing_cmd = _validate_readonly_command(cmd)
        if not is_valid:
            return {
                "success": False,
                "error": (
                    f"Command '{failing_cmd}' is not allowed in readonly mode. "
                    f"Allowed commands: {', '.join(sorted(get_allowed_readonly_commands())[:15])}... "
                    "Use 'shell' tool without readonly=True for other commands."
                ),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "cwd": os.getcwd(),
            }

    # Validate working directory exists before execution
    if cwd:
        if not os.path.isdir(cwd):
            return {
                "success": False,
                "error": f"Working directory does not exist: {cwd}",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

    try:
        # Check cache for read-only commands (CI/CD queries, git log, etc.)
        from victor.tools.shell_command_cache import get_shell_cache, execute_with_cache

        # Use cache for read-only commands
        if not dangerous and _is_readonly_command(cmd):
            try:
                returncode, stdout_str, stderr_str = execute_with_cache(
                    cmd,
                    cwd=cwd,
                    shell=True,
                    timeout=timeout,
                )

                # Apply truncation to cached results too
                final_stdout_limit = stdout_limit if stdout_limit is not None else 10000
                final_stderr_limit = stderr_limit if stderr_limit is not None else 2000

                from victor.tools.subprocess_executor import _truncate_output_by_lines

                stdout_str, stdout_truncated, stdout_lines = _truncate_output_by_lines(
                    stdout_str, final_stdout_limit, max_bytes=None, stream_name="stdout"
                )

                stderr_str, stderr_truncated, stderr_lines = _truncate_output_by_lines(
                    stderr_str, final_stderr_limit, max_bytes=None, stream_name="stderr"
                )

                return {
                    "success": returncode == 0,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "return_code": returncode,
                    "command": cmd,
                    "working_dir": cwd,
                    "cached": True,
                    "truncated": stdout_truncated or stderr_truncated,
                    "stdout_lines": stdout_lines,
                    "stderr_lines": stderr_lines,
                }
            except Exception as cache_error:
                # If caching fails, fall through to normal execution
                logger.warning(f"Cache lookup failed, executing directly: {cache_error}")

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

        # Apply defaults: 10K stdout lines, 2K stderr lines (None=unlimited)
        final_stdout_limit = stdout_limit if stdout_limit is not None else 10000
        final_stderr_limit = stderr_limit if stderr_limit is not None else 2000

        # Truncate stdout
        from victor.tools.subprocess_executor import _truncate_output_by_lines

        stdout_str, stdout_truncated, stdout_lines = _truncate_output_by_lines(
            stdout_str,
            final_stdout_limit,
            max_bytes=None,  # Use internal 1MB default
            stream_name="stdout",
        )

        # Truncate stderr
        stderr_str, stderr_truncated, stderr_lines = _truncate_output_by_lines(
            stderr_str,
            final_stderr_limit,
            max_bytes=None,  # Use internal 1MB default
            stream_name="stderr",
        )

        was_truncated = stdout_truncated or stderr_truncated

        # Invalidate file content cache if command was NOT readonly (may have modified files)
        if not readonly:
            try:
                from victor.tools.filesystem import (
                    clear_file_content_cache,
                    is_file_cache_enabled,
                )

                if is_file_cache_enabled():
                    clear_file_content_cache(reset_stats=False)
                    logger.debug("Cleared file content cache after non-readonly shell command")
            except (ImportError, Exception):
                pass

        result = {
            "success": process.returncode == 0,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "return_code": process.returncode,
            "command": cmd,
            "working_dir": cwd,
            "truncated": was_truncated,
            "stdout_lines": stdout_lines,
            "stderr_lines": stderr_lines,
        }

        # Cache successful read-only command results
        if process.returncode == 0 and not dangerous and _is_readonly_command(cmd):
            try:
                cache = get_shell_cache()
                cache.set(cmd, (process.returncode, stdout_str, stderr_str), cwd)
            except Exception as cache_error:
                logger.warning(f"Failed to cache command result: {cache_error}")

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
