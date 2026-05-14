"""Sandbox/isolation protocol definitions.

These protocols enable external verticals to declare sandbox requirements
for safe tool execution.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class SandboxProvider(Protocol):
    """Protocol for providing sandbox configuration requirements.

    Verticals implement this to declare what level of isolation
    their tools require during execution.
    """

    def get_sandbox_config(self) -> Dict[str, Any]:
        """Return sandbox configuration for this vertical.

        Returns:
            Dict with optional keys:
            - 'enabled': bool (default True)
            - 'filesystem_mode': "off" | "workspace-only" | "allow-list"
            - 'namespace_restrictions': bool
            - 'network_isolation': bool
            - 'allowed_mounts': List[str] - additional paths to allow
        """
        ...

    def get_tool_sandbox_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Return per-tool sandbox configuration overrides.

        Some tools may need different sandbox settings (e.g., docker
        tools need network access, database tools need mount access).

        Returns:
            Dict mapping tool names to sandbox config overrides.
        """
        ...
