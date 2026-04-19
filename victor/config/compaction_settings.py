"""Session compaction configuration for managing context window limits."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from victor.config.compaction_strategy_settings import (
    CompactionStrategySettings,
    CompactionFeatureFlags,
)


class CompactionSettings(BaseModel):
    """Session compaction configuration for managing context window limits.

    When conversation history exceeds token thresholds, older messages
    are summarized to free context space while preserving recent history.

    Supports three compaction strategies:
    - Rule-based: Fast, deterministic, sub-100ms (Claudecode-style)
    - LLM-based: Rich, intelligent summaries (Victor-style)
    - Hybrid: Combined approach with best of both worlds
    """

    # Legacy compaction settings (backward compatibility)
    compaction_enabled: bool = True
    compaction_preserve_recent: int = 4
    compaction_max_estimated_tokens: int = 10000
    compaction_auto_compact: bool = False

    # Hybrid compaction strategy settings (new)
    strategy: Optional[CompactionStrategySettings] = Field(
        default=None,
        description="Hybrid compaction strategy settings for intelligent "
        "strategy selection between rule-based, LLM-based, and hybrid approaches.",
    )

    # Feature flags for compaction (renamed from 'feature_flags' to avoid collision
    # with Settings.feature_flags which is a different type)
    compaction_feature_flags: Optional[CompactionFeatureFlags] = Field(
        default=None,
        description="Feature flags for compaction enhancements. "
        "Controls availability of different compaction strategies.",
    )

    def get_strategy_settings(self) -> CompactionStrategySettings:
        """Get compaction strategy settings with defaults.

        Returns:
            CompactionStrategySettings instance
        """
        if self.strategy is None:
            # Create defaults from legacy settings
            return CompactionStrategySettings(
                rule_preserve_recent=self.compaction_preserve_recent,
                rule_max_estimated_tokens=self.compaction_max_estimated_tokens,
            )
        return self.strategy

    def get_compaction_feature_flags(self) -> CompactionFeatureFlags:
        """Get compaction feature flags with defaults.

        Returns:
            CompactionFeatureFlags instance
        """
        if self.compaction_feature_flags is None:
            # Enable all features by default
            return CompactionFeatureFlags()
        return self.compaction_feature_flags

    # Backward compatibility alias
    def get_feature_flags(self) -> CompactionFeatureFlags:
        """Get compaction feature flags with defaults.

        Deprecated: Use get_compaction_feature_flags() instead.
        This method name collides with Settings.feature_flags which is a different type.

        Returns:
            CompactionFeatureFlags instance
        """
        return self.get_compaction_feature_flags()
