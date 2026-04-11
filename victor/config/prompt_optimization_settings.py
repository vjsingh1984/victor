"""Prompt optimization configuration.

Strategy-agnostic settings for evolving system prompt sections at runtime.
GEPA is the default (and currently only) strategy, but the design supports
future strategies via the section_strategies mapping.

Configuration hierarchy:
  prompt_optimization.enabled          → master switch (all strategies)
  prompt_optimization.default_strategies → ["gepa"] applied to all sections
  prompt_optimization.section_strategies → per-section overrides
  prompt_optimization.gepa.*           → GEPA-specific config (tiers, bloat, etc.)

Section strategy resolution:
  1. If section listed in section_strategies → use that list
  2. Else → use default_strategies
  3. Empty list [] → skip optimization for that section
  4. List with entries → apply strategies in order (layered)
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from victor.config.gepa_settings import GEPASettings


class MIPROv2Settings(BaseModel):
    """MIPROv2 few-shot demonstration mining configuration."""

    max_examples: int = 3
    min_completion_score: float = 0.7
    example_diversity: bool = True
    max_example_chars: int = 400


class CoTDistillationSettings(BaseModel):
    """Chain-of-Thought distillation configuration.

    Source and target providers are NOT hardcoded — they are determined
    dynamically at runtime from benchmark data:
    - source: auto-detected as the highest-scoring provider
    - target: all providers scoring below source
    - Can be overridden per call via evolve(source_provider=..., target_provider=...)
    """

    auto_detect_source: bool = True  # Auto-pick best provider as source
    min_source_score: float = 0.7
    max_steps: int = 5
    min_score_gap: float = 0.15  # Only distill if source-target gap >= 15pp


class PromptOptimizationSettings(BaseModel):
    """Top-level prompt optimization configuration."""

    # Master switch — gates ALL prompt optimization
    enabled: bool = False

    # Default strategy list applied to all evolvable sections.
    # Strategies are applied in order (layered).
    default_strategies: List[str] = Field(default_factory=lambda: ["gepa"])

    # Per-section overrides. Key = section name, value = strategy list.
    # Empty list [] = skip optimization for that section.
    # Missing key = use default_strategies.
    section_strategies: Dict[str, List[str]] = Field(default_factory=dict)

    # Strategy-specific configurations (nested)
    gepa: GEPASettings = Field(default_factory=GEPASettings)
    miprov2: MIPROv2Settings = Field(default_factory=MIPROv2Settings)
    cot_distillation: CoTDistillationSettings = Field(default_factory=CoTDistillationSettings)

    def get_strategies_for_section(self, section_name: str) -> List[str]:
        """Resolve which strategies apply to a given section.

        Returns:
            List of strategy names. Empty list means skip.
        """
        if not self.enabled:
            return []
        if section_name in self.section_strategies:
            return self.section_strategies[section_name]
        return self.default_strategies

    def is_strategy_active(self, strategy_name: str) -> bool:
        """Check if a strategy is used by any section."""
        if not self.enabled:
            return False
        if strategy_name in self.default_strategies:
            return True
        for strategies in self.section_strategies.values():
            if strategy_name in strategies:
                return True
        return False
