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

"""Tool detection and resolution utilities.

This module provides utility functions for:
- Resolving tool name aliases to canonical names
- Detecting shell-related tool variants
- Mapping tool names based on mode and availability

These utilities were extracted from AgentOrchestrator to improve
testability and reusability across the codebase.
"""

from __future__ import annotations

import logging
from typing import Any, Set, cast

from victor.tools.tool_names import ToolNames

logger = logging.getLogger(__name__)

# Shell-related aliases that should resolve intelligently
# LLMs often hallucinate these tool names, so we map them appropriately
SHELL_ALIASES: Set[str] = {
    "run",
    "bash",
    "execute",
    "cmd",
    "execute_bash",
    "shell_readonly",
    "shell",
}


def get_shell_aliases() -> Set[str]:
    """Get the set of all shell-related tool aliases.

    Returns:
        Set of tool name strings that are shell aliases
    """
    return SHELL_ALIASES.copy()


def is_shell_alias(tool_name: str) -> bool:
    """Check if a tool name is a shell alias.

    This function determines whether a given tool name is one of the
    recognized shell aliases. This is useful for tool resolution logic
    that needs to identify shell-related tools.

    Args:
        tool_name: The tool name to check

    Returns:
        True if the tool name is a shell alias, False otherwise

    Example:
        >>> is_shell_alias("bash")
        True
        >>> is_shell_alias("read_file")
        False
    """
    return tool_name in SHELL_ALIASES


def resolve_shell_variant(
    tool_name: str,
    mode_coordinator: Any = None,
) -> str:
    """Resolve shell aliases to the appropriate enabled shell variant.

    LLMs often hallucinate shell tool names like 'run', 'bash', 'execute'.
    These map to 'shell' canonically, but in INITIAL stage only 'shell_readonly'
    may be enabled. This method resolves to whichever shell variant is available.

    Args:
        tool_name: Original tool name (may be alias like 'run')
        mode_coordinator: Optional ModeCoordinator for mode-aware resolution.
                         If None, returns the canonical ToolNames.SHELL.

    Returns:
        The appropriate enabled shell tool name, or original if not a shell alias

    Example:
        >>> resolve_shell_variant("bash")
        'shell'
        >>> resolve_shell_variant("read_file")
        'read_file'

    Note:
        When mode_coordinator is provided, this delegates to the coordinator's
        resolve_shell_variant method for mode-aware resolution (e.g., choosing
        between 'shell' and 'shell_readonly' based on agent mode).
    """
    # If not a shell alias, return as-is
    if not is_shell_alias(tool_name):
        return tool_name

    # If mode coordinator is available, delegate for mode-aware resolution
    if mode_coordinator is not None:
        result = mode_coordinator.resolve_shell_variant(tool_name)
        return cast(str, result)

    # Default to canonical shell name
    return ToolNames.SHELL


def detect_mentioned_tools(
    text: str,
    available_tools: Set[str],
    tool_aliases: dict[str, str] | None = None,
) -> Set[str]:
    """Detect which tools are mentioned in a text string.

    This function scans text for mentions of tool names or their aliases,
    returning the set of canonical tool names that were mentioned.

    Args:
        text: The text to scan for tool mentions
        available_tools: Set of available tool names
        tool_aliases: Optional mapping of aliases to canonical tool names

    Returns:
        Set of canonical tool names that were mentioned in the text

    Example:
        >>> available = {"read_file", "write_file", "execute_bash"}
        >>> detect_mentioned_tools("Use read_file to view code", available)
        {"read_file"}
        >>> detect_mentioned_tools("Run bash command", available, {"bash": "execute_bash"})
        {"execute_bash"}
    """
    if not text:
        return set()

    mentioned = set()
    text_lower = text.lower()

    # Check for direct tool name mentions
    for tool in available_tools:
        if tool.lower() in text_lower:
            mentioned.add(tool)

    # Check for alias mentions
    if tool_aliases:
        for alias, canonical in tool_aliases.items():
            if alias.lower() in text_lower and canonical in available_tools:
                mentioned.add(canonical)

    return mentioned


__all__ = [
    "get_shell_aliases",
    "is_shell_alias",
    "resolve_shell_variant",
    "detect_mentioned_tools",
]
