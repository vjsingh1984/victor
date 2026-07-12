"""Session compaction configuration for managing context window limits."""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field


class AdaptiveCompactionSettings(BaseModel):
    """Settings for adaptive compaction threshold based on conversation patterns.

    The adaptive system analyzes conversation patterns to dynamically adjust
    the compaction threshold:
    - Rapid topic switches → lower threshold (35-40%)
    - Deep reasoning on one topic → higher threshold (65-70%)
    - Q&A style → lower threshold (35-40%)
    - Multi-step problem solving → higher threshold (65-70%)
    """

    enabled: bool = Field(
        default_factory=lambda: os.getenv("VICTOR_ADAPTIVE_THRESHOLD_ENABLED", "false").lower()
        == "true",
        description="Enable adaptive threshold based on conversation pattern analysis. "
        "Env: VICTOR_ADAPTIVE_THRESHOLD_ENABLED",
    )

    min_threshold: float = Field(
        default_factory=lambda: float(os.getenv("VICTOR_COMPACTION_MIN_THRESHOLD", "0.35")),
        ge=0.10,
        le=0.80,
        description="Minimum adaptive threshold (for rapid topic switches). "
        "Default: 0.35 (35%), Env: VICTOR_COMPACTION_MIN_THRESHOLD",
    )

    max_threshold: float = Field(
        default_factory=lambda: float(os.getenv("VICTOR_COMPACTION_MAX_THRESHOLD", "0.70")),
        ge=0.20,
        le=0.95,
        description="Maximum adaptive threshold (for deep reasoning). "
        "Default: 0.70 (70%), Env: VICTOR_COMPACTION_MAX_THRESHOLD",
    )

    analysis_window: int = Field(
        default_factory=lambda: int(os.getenv("VICTOR_COMPACTION_ANALYSIS_WINDOW", "20")),
        ge=5,
        le=50,
        description="Number of recent messages to analyze for pattern detection. "
        "Default: 20, Env: VICTOR_COMPACTION_ANALYSIS_WINDOW",
    )

    update_frequency: int = Field(
        default_factory=lambda: int(os.getenv("VICTOR_COMPACTION_UPDATE_FREQUENCY", "5")),
        ge=1,
        le=20,
        description="Re-analyze conversation pattern every N turns. "
        "Default: 5, Env: VICTOR_COMPACTION_UPDATE_FREQUENCY",
    )

    def validate_settings(self) -> None:
        """Validate adaptive threshold settings.

        Raises:
            ValueError: If settings are invalid
        """
        if self.min_threshold >= self.max_threshold:
            raise ValueError(
                f"min_threshold ({self.min_threshold}) must be less than "
                f"max_threshold ({self.max_threshold})"
            )

        if self.min_threshold < 0.10:
            raise ValueError("min_threshold must be at least 0.10 (10%)")

        if self.max_threshold > 0.95:
            raise ValueError("max_threshold must be at most 0.95 (95%)")


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

    # Adaptive threshold settings
    adaptive_threshold: Optional[AdaptiveCompactionSettings] = Field(
        default=None,
        description="Adaptive threshold settings for pattern-based compaction. "
        "When enabled, adjusts compaction threshold dynamically based on "
        "conversation patterns (Q&A, deep reasoning, topic switching, etc.).",
    )
