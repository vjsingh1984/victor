import shlex
import warnings

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified.parser import split_command

_SEARCH_DEPRECATED_MSG = (
    "The 'search' tool is deprecated; slated for removal in 0.9.0. "
    "Use 'code grep' for file-content search and 'fs search' for file-name search."
)


@tool(
    name="search",
    access_mode=AccessMode.READONLY,
    danger_level=DangerLevel.SAFE,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.LOW,
)
async def search_tool(cmd: str) -> str:
    """DEPRECATED back-compat shim for the unified search tool.

    The ``search`` domain has been retired: file-content search moved to
    ``code grep`` and file-name search moved to ``fs search``. This shim
    forwards old ``search grep ...`` / ``search files ...`` calls to their new
    homes so existing prompts, traces, and tool selections keep working.

    Forwarding map:
        search grep "def foo" /path   ->  code grep "def foo" /path
        search files "*.py" /path     ->  fs search "*.py" /path

    Args:
        cmd: A bash-style ``search`` command (e.g. ``search grep "x" src``).
    """
    warnings.warn(_SEARCH_DEPRECATED_MSG, DeprecationWarning, stacklevel=2)

    args_list = split_command(cmd)
    if args_list and args_list[0] == "search":
        args_list = args_list[1:]
    if not args_list:
        return f"### ❌ ERROR\n{_SEARCH_DEPRECATED_MSG}"

    subcommand = args_list[0]
    rest = args_list[1:]

    if subcommand == "grep":
        from victor.tools.unified.code_tool import code_tool

        return await code_tool(_join(["code", "grep", *rest]))

    if subcommand == "files":
        from victor.tools.unified.fs_tool import fs_tool

        # `search files <pattern> [path]` maps to `fs search <pattern> [path]`.
        return await fs_tool(_join(["fs", "search", *rest]))

    return f"### ❌ ERROR\nUnknown search subcommand '{subcommand}'. {_SEARCH_DEPRECATED_MSG}"


def _join(tokens: list[str]) -> str:
    """Re-serialize command tokens into a single shell-safe command string."""
    return " ".join(shlex.quote(str(t)) for t in tokens)


__all__ = ["search_tool"]
