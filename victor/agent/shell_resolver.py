"""Shell alias resolution for orchestrator.

Extracted from AgentOrchestrator to reduce orchestrator LOC.
Handles resolution of LLM-hallucinated shell tool names (run, bash, execute)
to the appropriate enabled shell variant (shell, shell_readonly).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from victor.tools.alias_resolver import get_alias_resolver
from victor.tools.tool_names import ToolNames

if TYPE_CHECKING:
    from victor.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# Shell-related aliases that should resolve intelligently
SHELL_ALIASES = {
    "run",
    "bash",
    "execute",
    "cmd",
    "execute_bash",
    "shell_readonly",
    "shell",
}


def resolve_shell_variant(
    tool_name: str,
    tools: "ToolRegistry",
    mode_controller: Optional[object] = None,
) -> str:
    """Resolve shell aliases to the appropriate enabled shell variant.

    LLMs often hallucinate shell tool names like 'run', 'bash', 'execute'.
    These map to 'shell' canonically, but in INITIAL stage only 'shell_readonly'
    may be enabled. This resolves to whichever shell variant is available.

    Args:
        tool_name: Original tool name (may be alias like 'run')
        tools: Tool registry to check enabled status
        mode_controller: Mode controller for checking BUILD mode

    Returns:
        The appropriate enabled shell tool name, or original if not a shell alias
    """
    if tool_name not in SHELL_ALIASES:
        return tool_name

    resolver = get_alias_resolver()
    resolver.register(
        ToolNames.SHELL,
        aliases=list(SHELL_ALIASES - {ToolNames.SHELL}),
        resolver=lambda name: _shell_alias_resolver(name, tools, mode_controller),
    )
    return resolver.resolve(tool_name, enabled_tools=[])


def _shell_alias_resolver(
    tool_name: str,
    tools: "ToolRegistry",
    mode_controller: Optional[object],
) -> str:
    """Custom resolver for shell aliases that checks mode and tool availability."""
    from victor.tools.tool_names import get_canonical_name

    mc = mode_controller
    if mc is not None:
        config = mc.config
        if config.allow_all_tools and "shell" not in config.disallowed_tools:
            logger.debug(
                f"Resolved '{tool_name}' to 'shell' (BUILD mode allows all tools)"
            )
            return ToolNames.SHELL

    if tools.is_tool_enabled(ToolNames.SHELL):
        logger.debug(f"Resolved '{tool_name}' to 'shell' (shell enabled)")
        return ToolNames.SHELL

    if tools.is_tool_enabled(ToolNames.SHELL_READONLY):
        logger.debug(f"Resolved '{tool_name}' to 'shell_readonly' (readonly mode)")
        return ToolNames.SHELL_READONLY

    canonical = get_canonical_name(tool_name)
    logger.debug(
        f"No shell variant enabled for '{tool_name}', using canonical '{canonical}'"
    )
    return canonical
