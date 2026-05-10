"""Canonicalization helpers for Victor's core file and shell tools.

This intentionally normalizes only the compact internal runtime surface:
`read`, `write`, `edit`, `ls`, and `shell`.

It does not collapse broader families like `git_*` or `search` because those
remain distinct runtime tools with different semantics.
"""

from __future__ import annotations

import re

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
}


def normalize_model_tool_name(name: str) -> str:
    """Normalize provider-emitted tool names before alias lookup.

    Some providers emit camelCase or mixed-case names even when the advertised
    runtime tool names are snake_case. Normalize those spellings before
    validation so aliases and availability checks all see the same surface.
    """
    cleaned = str(name or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("-", "_").replace(".", "_")
    cleaned = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", cleaned)
    cleaned = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.lower()


def canonicalize_core_tool_name(name: str) -> str:
    """Resolve known core-tool aliases to Victor's canonical internal names."""
    normalized = normalize_model_tool_name(name)
    return CORE_TOOL_ALIASES.get(normalized, normalized)
