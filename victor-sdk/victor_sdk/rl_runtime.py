"""SDK host adapters for RL runtime helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.framework.rl import (
        RLCoordinator,
        RLManager,
        analyze_prompt_rollout_experiment,
        analyze_prompt_rollout_experiment_async,
        apply_prompt_rollout_recommendation,
        apply_prompt_rollout_recommendation_async,
        process_prompt_candidate_evaluation_suite,
        process_prompt_candidate_evaluation_suite_async,
        create_prompt_rollout_experiment,
        create_prompt_rollout_experiment_async,
        get_rl_coordinator,
        get_rl_coordinator_async,
    )

__all__ = [
    "RLCoordinator",
    "RLManager",
    "analyze_prompt_rollout_experiment",
    "analyze_prompt_rollout_experiment_async",
    "apply_prompt_rollout_recommendation",
    "apply_prompt_rollout_recommendation_async",
    "process_prompt_candidate_evaluation_suite",
    "process_prompt_candidate_evaluation_suite_async",
    "create_prompt_rollout_experiment",
    "create_prompt_rollout_experiment_async",
    "get_rl_coordinator",
    "get_rl_coordinator_async",
]

_LAZY_IMPORTS = {
    "RLCoordinator": "victor.framework.rl",
    "RLManager": "victor.framework.rl",
    "analyze_prompt_rollout_experiment": "victor.framework.rl",
    "analyze_prompt_rollout_experiment_async": "victor.framework.rl",
    "apply_prompt_rollout_recommendation": "victor.framework.rl",
    "apply_prompt_rollout_recommendation_async": "victor.framework.rl",
    "process_prompt_candidate_evaluation_suite": "victor.framework.rl",
    "process_prompt_candidate_evaluation_suite_async": "victor.framework.rl",
    "create_prompt_rollout_experiment": "victor.framework.rl",
    "create_prompt_rollout_experiment_async": "victor.framework.rl",
    "get_rl_coordinator": "victor.framework.rl",
    "get_rl_coordinator_async": "victor.framework.rl",
}


def __getattr__(name: str) -> Any:
    """Resolve RL helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.rl_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
