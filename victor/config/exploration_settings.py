"""Exploration iteration limits, recovery, and continuation prompts."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field

from victor.core.loop_thresholds import (
    DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
    DEFAULT_BLOCKED_TOTAL_THRESHOLD,
)


class ExplorationSettings(BaseModel):
    """Exploration iteration limits, recovery, and continuation prompts."""

    max_exploration_iterations: int = 8
    max_exploration_iterations_action: int = 12
    max_exploration_iterations_analysis: int = 50
    chat_max_iterations: int = 50
    max_consecutive_tool_calls: int = 20
    max_research_iterations: int = 6
    min_content_threshold: int = 150
    recovery_empty_response_threshold: int = 5
    recovery_blocked_consecutive_threshold: int = DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
    recovery_blocked_total_threshold: int = DEFAULT_BLOCKED_TOTAL_THRESHOLD
    max_continuation_prompts_default: int = 3
    max_continuation_prompts_action: int = 5
    max_continuation_prompts_analysis: int = 6
    continuation_prompt_overrides: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    enable_continuation_rl_learning: bool = True
