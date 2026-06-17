"""RL (Reinforcement Learning) mixin for VerticalBase."""

from __future__ import annotations

from typing import Any, List, Optional


class RLMixin:
    """Opt-in mixin providing RL configuration hooks.

    Methods:
        get_rl_config_provider: Return the RL config provider, if any.
        get_rl_hooks: Return RL hooks for outcome recording.
    """

    @classmethod
    def get_rl_config_provider(cls) -> Optional[Any]:
        """Return the RL config provider for this vertical, if any."""
        return None

    @classmethod
    def get_rl_hooks(cls) -> List[Any]:
        """Return RL hooks for this vertical, if any."""
        return []
