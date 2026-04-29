"""Canonicalization helpers for Victor's core file and shell tools.

This intentionally normalizes only the compact internal runtime surface:
`read`, `write`, `edit`, `ls`, and `shell`.

It does not collapse broader families like `git_*` or `search` because those
remain distinct runtime tools with different semantics.
"""

from __future__ import annotations

CORE_TOOL_ALIASES = {
    "read_file": "read",
    "list_directory": "ls",
    "write_file": "write",
    "create_file": "write",
    "edit_files": "edit",
    "edit_file": "edit",
    "patch_file": "edit",
    "execute_bash": "shell",
    "bash": "shell",
    "shell_readonly": "shell",  # Legacy alias retained for backward compatibility
}


def canonicalize_core_tool_name(name: str, preserve_variants: bool = False) -> str:
    """Resolve known core-tool aliases to Victor's canonical internal names.

    Args:
        name: Tool name to normalize.
        preserve_variants: When True, keep intent/runtime variants like
            ``shell_readonly`` distinct instead of collapsing them to ``shell``.
    """
    if preserve_variants and name == "shell_readonly":
        return name
    return CORE_TOOL_ALIASES.get(name, name)
