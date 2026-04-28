# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Model-specific post-compaction continuation bonuses (P1 feature).

Based on AgentSwing research findings:
- DeepSeek models prefer Summary strategy, need more continuation help after compaction
- GPT models prefer Discard-All, are moderately resilient
- Other models get minimal bonus

This module provides the compaction continuation bonus calculator that
is used by the continuation strategy to determine how many additional
continuation prompts to allow after context compaction occurs.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompactionBonusConfig:
    """Configuration for model-specific compaction continuation bonuses.

    Attributes:
        model_bonuses: Map of model name patterns to bonus values.
            Longer keys are matched first (e.g., "deepseek-coder" before "deepseek")
        default_bonus: Default bonus for models not in the map.
    """
    model_bonuses: Dict[str, int]
    default_bonus: int = 1

    def __post_init__(self):
        """Validate configuration values."""
        if self.default_bonus < 0:
            raise ValueError("default_bonus must be non-negative")
        for key, value in self.model_bonuses.items():
            if value < 0:
                raise ValueError(f"model_bonuses['{key}'] must be non-negative")


class CompactionContinuationBonus:
    """Calculate continuation bonus after compaction based on model.

    Based on AgentSwing findings (arXiv:2603.27490):
    - DeepSeek prefers Summary strategy (0.49), needs more continuation help (+3)
    - GPT models prefer Discard-All (0.55), moderately resilient (+2)
    - Other models get minimal bonus (+1)

    The bonus scales based on compaction severity (messages removed) to provide
    more help when more context was lost.
    """

    # Default model bonuses based on research findings
    DEFAULT_MODEL_BONUSES: Dict[str, int] = {
        # DeepSeek models - highest bonus (prefer Summary, most affected by compaction)
        "deepseek-chat": 3,
        "deepseek": 3,
        "deepseek-coder": 2,
        "deepseek-r1": 2,
        # GPT models - moderate bonus (prefer Discard-All, more resilient)
        "gpt-4o": 2,
        "gpt-4o-mini": 2,
        "gpt-4-turbo": 2,
        "gpt-4": 2,
        "chatgpt-4o": 2,
        "gpt-3.5": 2,
        "claude-3.5": 2,  # Also resilient
        "claude-3": 2,
        # Local models - minimal bonus (variable behavior)
        "llama": 1,
        "mistral": 1,
        "qwen": 1,
        "gemma": 1,
    }

    def __init__(self, config: Optional[CompactionBonusConfig] = None):
        """Initialize the bonus calculator.

        Args:
            config: Optional custom configuration. If None, uses defaults.
        """
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> CompactionBonusConfig:
        """Create default configuration with research-based bonuses."""
        return CompactionBonusConfig(
            model_bonuses=CompactionContinuationBonus.DEFAULT_MODEL_BONUSES.copy(),
            default_bonus=1,
        )

    def get_bonus(
        self,
        provider: str,
        model: str,
        compaction_occurred: bool,
        messages_removed: int = 0,
    ) -> int:
        """Calculate continuation bonus after compaction.

        Args:
            provider: Provider name (e.g., "deepseek", "openai", "anthropic")
            model: Model name (e.g., "chat", "gpt-4o", "claude-3.5-sonnet")
            compaction_occurred: Whether compaction just happened
            messages_removed: Number of messages removed (for scaling)

        Returns:
            Additional continuation prompts to allow (0 if no compaction)
        """
        if not compaction_occurred:
            return 0

        # Check for model-specific bonus
        # Sort by key length descending to match longer, more specific keys first
        provider_lower = provider.lower()
        model_lower = model.lower()
        combined = f"{provider_lower}:{model_lower}"

        # Sort keys by length (descending) to match "deepseek-coder" before "deepseek"
        sorted_keys = sorted(
            self.config.model_bonuses.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for model_key, bonus in sorted_keys:
            if model_key in provider_lower or model_key in model_lower or model_key in combined:
                # Scale bonus based on severity (more messages removed = more help needed)
                scale = min(2.0, messages_removed / 50)  # Max 2x multiplier at 50+ messages
                result = int(bonus * (1 + scale))
                logger.debug(
                    f"Compaction bonus for {provider}:{model}: {result} "
                    f"(base={bonus}, scale={scale:.2f}, messages_removed={messages_removed})"
                )
                return result

        # Default bonus with scaling
        scale = min(2.0, messages_removed / 50)
        result = int(self.config.default_bonus * (1 + scale))
        logger.debug(
            f"Default compaction bonus for {provider}:{model}: {result} "
            f"(base={self.config.default_bonus}, scale={scale:.2f}, messages_removed={messages_removed})"
        )
        return result

    def get_adjusted_continuation_budget(
        self,
        base_budget: int,
        provider: str,
        model: str,
        compaction_occurred: bool,
        messages_removed: int = 0,
    ) -> int:
        """Get the total continuation budget including compaction bonus.

        Args:
            base_budget: Base continuation budget from settings
            provider: Provider name
            model: Model name
            compaction_occurred: Whether compaction just happened
            messages_removed: Number of messages removed

        Returns:
            Total continuation budget (base + bonus if compaction occurred)
        """
        bonus = self.get_bonus(provider, model, compaction_occurred, messages_removed)
        return base_budget + bonus


# Singleton instance for convenience
_default_instance: Optional[CompactionContinuationBonus] = None


def get_compaction_bonus() -> CompactionContinuationBonus:
    """Get the singleton compaction bonus calculator.

    Returns:
        The default CompactionContinuationBonus instance
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = CompactionContinuationBonus()
    return _default_instance
