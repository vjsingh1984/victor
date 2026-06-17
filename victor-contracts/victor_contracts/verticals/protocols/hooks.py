"""Hook system protocol definitions.

These protocols enable external verticals to declare pre/post tool
execution hooks for validation, auditing, and gating.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class HookProvider(Protocol):
    """Protocol for providing tool execution hooks.

    Verticals implement this to run custom logic before and after
    tool calls. Hooks can allow, deny, or warn.
    """

    def get_pre_tool_hooks(self) -> List[str]:
        """Return shell commands to run before tool execution.

        Each command receives tool context via environment variables
        (HOOK_EVENT, HOOK_TOOL_NAME, HOOK_TOOL_INPUT) and stdin JSON.

        Exit codes:
            0 = allow (stdout as optional message)
            2 = deny (stdout as denial reason)
            other = warn but allow

        Returns:
            List of shell command strings.
        """
        ...

    def get_post_tool_hooks(self) -> List[str]:
        """Return shell commands to run after tool execution.

        Receives additional HOOK_TOOL_OUTPUT and HOOK_TOOL_IS_ERROR.
        Post-hooks are informational — denial does not block.

        Returns:
            List of shell command strings.
        """
        ...


@runtime_checkable
class HookConfigProvider(Protocol):
    """Protocol for providing hook configuration.

    Verticals can declare which hooks should be active and how
    they should be configured.
    """

    def get_hook_config(self) -> Dict[str, Any]:
        """Return hook configuration for this vertical.

        Returns:
            Dict with optional keys:
            - 'enabled': bool (default True)
            - 'pre_tool_use': List[str] - shell commands
            - 'post_tool_use': List[str] - shell commands
            - 'timeout_ms': int - per-hook timeout
        """
        ...
