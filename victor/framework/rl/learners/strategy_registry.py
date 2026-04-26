"""Factory helpers for prompt-optimization strategies.

Keeps strategy-name resolution out of the learner so config-driven selection
can expand without growing more conditionals in PromptOptimizerLearner.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_prompt_strategy(
    strategy_name: str,
    *,
    settings: Optional[Any] = None,
    gepa_strategy: Optional[Any] = None,
) -> Optional[Any]:
    """Instantiate a prompt-optimization strategy by config name."""
    normalized = (strategy_name or "").strip().lower()

    if normalized == "gepa":
        if gepa_strategy is not None:
            return gepa_strategy
        from victor.framework.rl.learners.prompt_optimizer import GEPAStrategy

        return GEPAStrategy()

    if normalized == "miprov2":
        from victor.framework.rl.learners.strategies.miprov2_strategy import (
            MIPROv2Strategy,
        )

        mipro_settings = getattr(settings, "miprov2", None)
        return MIPROv2Strategy(
            max_examples=getattr(mipro_settings, "max_examples", 3),
        )

    if normalized == "cot_distillation":
        from victor.framework.rl.learners.strategies.cot_distillation_strategy import (
            CoTDistillationStrategy,
        )

        cot_settings = getattr(settings, "cot_distillation", None)
        return CoTDistillationStrategy(
            min_source_score=getattr(cot_settings, "min_source_score", 0.7),
            max_steps=getattr(cot_settings, "max_steps", 5),
            min_score_gap=getattr(cot_settings, "min_score_gap", 0.15),
        )

    if normalized == "prefpo":
        from victor.framework.rl.learners.strategies.prefpo_strategy import (
            PrefPOStrategy,
        )

        prefpo_settings = getattr(settings, "prefpo", None)
        return PrefPOStrategy(
            max_guidance_items=getattr(prefpo_settings, "max_guidance_items", 2),
            min_failure_count=getattr(prefpo_settings, "min_failure_count", 1),
            max_prompt_growth_chars=getattr(prefpo_settings, "max_prompt_growth_chars", 240),
        )

    logger.warning("Unknown prompt optimization strategy '%s' - skipping", strategy_name)
    return None
