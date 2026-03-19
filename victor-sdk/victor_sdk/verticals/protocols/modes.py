"""Mode-related protocol definitions.

These protocols define how verticals provide mode configurations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional


@runtime_checkable
class ModeConfigProvider(Protocol):
    """Protocol for providing mode-specific configurations.

    Modes allow verticals to behave differently based on context
    or user preference.
    """

    def get_available_modes(self) -> List[str]:
        """Return list of available modes.

        Examples: "interactive", "batch", "safe", "aggressive"

        Returns:
            List of mode identifiers
        """
        ...

    def get_default_mode(self) -> str:
        """Return the default mode.

        Returns:
            Default mode identifier
        """
        ...

    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """Return configuration for a specific mode.

        Args:
            mode: Mode identifier

        Returns:
            Configuration dictionary for this mode
        """
        ...

    def is_mode_available(self, mode: str) -> bool:
        """Check if a mode is available.

        Args:
            mode: Mode identifier

        Returns:
            True if mode is available
        """
        ...
