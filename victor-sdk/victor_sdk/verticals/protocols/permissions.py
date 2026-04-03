"""Permission hierarchy protocol definitions.

These protocols enable external verticals to declare tool permission
requirements using the three-tier model:
    ReadOnly → WorkspaceWrite → DangerFullAccess
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class PermissionProvider(Protocol):
    """Protocol for providing tool permission requirements.

    Verticals implement this to declare what permission level
    each of their tools requires.
    """

    def get_permission_mode(self) -> str:
        """Return the default permission mode for this vertical.

        Returns:
            One of: "read-only", "workspace-write", "danger-full-access"
        """
        ...

    def get_tool_permissions(self) -> Dict[str, str]:
        """Return per-tool permission requirements.

        Returns:
            Dict mapping tool names to required permission level strings.
            Tools not listed default to "danger-full-access".
        """
        ...

    def get_permission_escalation_rules(self) -> List[Dict[str, Any]]:
        """Return rules for when to prompt for permission escalation.

        Returns:
            List of rule dicts, each with:
            - 'tool_pattern': str - glob pattern for tool names
            - 'from_mode': str - current mode
            - 'to_mode': str - required mode
            - 'auto_approve': bool - skip user prompt if True
        """
        ...
