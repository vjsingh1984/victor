"""External plugin protocol definitions.

These protocols enable external verticals to declare plugin requirements
and provide plugin-specific configurations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class ExternalPluginProvider(Protocol):
    """Protocol for declaring external plugin requirements.

    Verticals implement this to specify which external plugins
    they depend on or recommend.
    """

    def get_required_plugins(self) -> List[str]:
        """Return list of required plugin IDs.

        These plugins must be installed and enabled for the
        vertical to function correctly.

        Returns:
            List of plugin ID strings (e.g., "my-plugin@external").
        """
        ...

    def get_recommended_plugins(self) -> List[str]:
        """Return list of recommended plugin IDs.

        These plugins enhance the vertical but are not required.

        Returns:
            List of plugin ID strings.
        """
        ...

    def get_plugin_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return per-plugin configuration overrides.

        Returns:
            Dict mapping plugin IDs to config override dicts.
        """
        ...
