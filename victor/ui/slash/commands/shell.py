"""Shell command: execute local shell commands from the interactive REPL.
Usage:
    !ls -la
The ``!`` prefix runs the command directly in your shell, displays the output,
and injects it into the conversation so the agent can act on it. This matches
the Claude Code bang-prefix convention.
This module also registers a ``/shell`` slash command as an alias.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Optional
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)
_SHELL_TIMEOUT = 300


async def _run_shell_command(
    command: str,
    *,
    timeout: int = _SHELL_TIMEOUT,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _format_display(
    command: str,
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    truncated: bool = False,
) -> str:
    parts: list[str] = []
    status = chr(0x2713) if returncode == 0 else chr(0x2717)
    parts.append(f"[bold]`$ {command}`[/] [dim](exit {returncode})[/] [green]{status}[/]")
    if stdout:
        parts.append("")
        parts.append(stdout.rstrip("\n"))
    if stderr:
        parts.append("")
        parts.append(f"[yellow]stderr:[/]\n{stderr.rstrip()}")
    if truncated:
        parts.append("")
        parts.append("[dim]... (output truncated)[/]")
    return "\n".join(parts)


def _format_for_agent(
    command: str,
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    truncated: bool = False,
) -> str:
    parts: list[str] = []
    parts.append(f"Shell command `{command}` completed with exit code {returncode}.")
    if stdout:
        parts.append("")
        parts.append("```")
        parts.append(stdout.rstrip("\n"))
        parts.append("```")
    if stderr:
        parts.append("")
        parts.append("stderr:")
        parts.append("```")
        parts.append(stderr.rstrip("\n"))
        parts.append("```")
    if truncated:
        parts.append("")
        parts.append("(output was truncated)")
    return "\n".join(parts)


@register_command
class ShellCommand(BaseSlashCommand):
    """Execute a local shell command from the interactive REPL.
    ``!<command>`` runs the command, displays output, and sends it to the
    agent so the assistant can act on the result.
    """

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="shell",
            description="Execute a local shell command (!cmd) -- output sent to agent",
            usage="!<command>  or  /shell <command>",
            aliases=["sh", "bash", "run"],
            category="tools",
            requires_agent=False,
        )

    def execute(self, ctx: CommandContext) -> None:
        command = " ".join(ctx.args) if ctx.args else ""
        if not command:
            ctx.console.print("[yellow]Usage:[/] /shell <command>")
            return
        ctx.console.print(f"[dim]Running:[/] $ {command}")
        try:
            import subprocess

            result = subprocess.run(
                command,
                shell=True,  # nosec B602 — intentional: /shell REPL runs user-typed shell input
                capture_output=True,
                text=True,
                timeout=_SHELL_TIMEOUT,
            )
            output = _format_display(
                command,
                result.returncode,
                result.stdout,
                result.stderr,
            )
            ctx.console.print(output)
        except subprocess.TimeoutExpired:
            ctx.console.print(f"[red]Command timed out after {_SHELL_TIMEOUT}s:[/] {command}")
        except Exception as e:
            ctx.console.print(f"[red]Error running command:[/] {e}")


def is_shell_command(user_input: str) -> bool:
    return user_input.strip().startswith("!")


def parse_shell_command(user_input: str) -> str:
    stripped = user_input.strip()
    if stripped.startswith("!"):
        return stripped[1:].strip()
    return stripped


async def execute_shell_command(
    user_input: str,
    console: object,
    agent: Optional[object] = None,
) -> Optional[str]:
    """Execute a ``!`` command from the interactive REPL.
    Displays output to the user AND returns a formatted string for
    injection into the agent conversation. Returns None only on empty
    input or command failure.
    """
    command = parse_shell_command(user_input)
    if not command:
        console.print("[yellow]Usage:[/] !<command>")
        return None
    console.print(f"[dim]Running:[/] $ {command}")
    try:
        returncode, stdout, stderr = await _run_shell_command(command)
    except asyncio.TimeoutError:
        console.print(f"[red]Command timed out after {_SHELL_TIMEOUT}s:[/] {command}")
        return None
    except Exception as e:
        console.print(f"[red]Error running command:[/] {e}")
        return None
    MAX_OUTPUT = 50 * 1024
    truncated = False
    if len(stdout) > MAX_OUTPUT:
        stdout = stdout[-MAX_OUTPUT:]
        truncated = True
    if len(stderr) > MAX_OUTPUT:
        stderr = stderr[-MAX_OUTPUT:]
        truncated = True
    display = _format_display(command, returncode, stdout, stderr, truncated=truncated)
    console.print(display)
    return _format_for_agent(command, returncode, stdout, stderr, truncated=truncated)
