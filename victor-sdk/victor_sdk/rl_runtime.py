"""SDK host adapters for RL runtime helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.rl import (
        RLCoordinator,
        RLManager,
        create_prompt_rollout_experiment,
        create_prompt_rollout_experiment_async,
        get_rl_coordinator,
        get_rl_coordinator_async,
    )

__all__ = [
    "RLCoordinator",
    "RLManager",
    "create_prompt_rollout_experiment",
    "create_prompt_rollout_experiment_async",
    "get_rl_coordinator",
    "get_rl_coordinator_async",
]

_LAZY_IMPORTS = {
    "RLCoordinator": "victor.framework.rl",
    "RLManager": "victor.framework.rl",
    "create_prompt_rollout_experiment": "victor.framework.rl",
    "create_prompt_rollout_experiment_async": "victor.framework.rl",
    "get_rl_coordinator": "victor.framework.rl",
    "get_rl_coordinator_async": "victor.framework.rl",
}


def __getattr__(name: str):
    """Resolve RL helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.rl_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
