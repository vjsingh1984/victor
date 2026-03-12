"""Compatibility shim for Research runtime RL helpers."""

from victor.verticals.contrib.research.runtime.rl import (
    ResearchRLConfig,
    ResearchRLHooks,
    get_default_config,
    get_research_rl_hooks,
)

__all__ = [
    "ResearchRLConfig",
    "ResearchRLHooks",
    "get_default_config",
    "get_research_rl_hooks",
]
