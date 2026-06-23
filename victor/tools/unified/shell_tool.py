import asyncio
import argparse
import sys
from io import StringIO

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.agent.background_tasks import BackgroundTaskDef
from victor.tools.unified.parser import split_command


async def execute_bash(cmd: str, sandbox: bool = False, readonly: bool = False):
    """Execute through the production shell tool implementation."""
    from victor.tools.bash import shell

    return await shell(
        cmd=cmd,
        readonly=readonly,
        action="read" if readonly else "write",
    )


class UnifiedShellParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_shell_parser() -> UnifiedShellParser:
    parser = UnifiedShellParser(
        prog="shell", description="Unified shell operations.", exit_on_error=False
    )
    # The shell wrapper is simpler: it takes a command string and optional flags
    parser.add_argument("cmd", help="The bash command to run")
    parser.add_argument("--sandbox", action="store_true", help="Run the command inside a sandbox")
    parser.add_argument("--readonly", action="store_true", help="Enforce readonly limits")
    parser.add_argument(
        "--timebound-sync",
        type=int,
        default=None,
        help="Max time (in seconds) to wait synchronously before backgrounding or timing out.",
    )

    return parser


@tool(
    name="shell",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.HIGH,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.HIGH,
)
async def shell_tool(cmd: str) -> str:
    """Unified shell tool.
    Example commands:
      shell "ls -la"
      shell "npm start" --timebound-sync 300
    """
    parser = create_shell_parser()

    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "shell":
            args_list = args_list[1:]
        parsed_args = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error: {e}"

    try:
        coro = execute_bash(
            cmd=parsed_args.cmd, sandbox=parsed_args.sandbox, readonly=parsed_args.readonly
        )

        # Determine if we should enforce a timebound sync
        if parsed_args.timebound_sync is not None:
            task = asyncio.create_task(coro)
            done, pending = await asyncio.wait([task], timeout=parsed_args.timebound_sync)

            if pending:
                # Task is still running after timeout
                # If we are in an agent environment (watcher exists), we background it.
                # For this prototype, we'll assume the watcher is active and return the Def.
                has_watcher = True
                if has_watcher:
                    return BackgroundTaskDef(task=task, context=f"shell_tool: {parsed_args.cmd}")
                else:
                    task.cancel()
                    return f"### ❌ ERROR\nCommand timed out after {parsed_args.timebound_sync}s"

            # Task completed
            results = task.result()
        else:
            # Traditional synchronous wait
            results = await coro

        if not isinstance(results, dict):
            return str(results)

        stdout = results.get("stdout", "").strip()
        stderr = results.get("stderr", "").strip()

        out = []
        if stdout:
            out.append(f"### STDOUT\n```text\n{stdout}\n```")
        if stderr:
            out.append(f"### STDERR\n```text\n{stderr}\n```")

        if not out:
            return "Command executed successfully with no output."
        return "\n\n".join(out)
    except Exception as e:
        return f"### ❌ ERROR\nShell execution failed: {e}"
