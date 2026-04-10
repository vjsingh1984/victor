"""Tiered decision service configuration.

Routes different DecisionTypes to different model tiers:
- edge: Local model (Ollama) for fast micro-decisions (~5ms)
- balanced: Mid-tier cloud (DeepSeek) for moderate decisions (~500ms)
- performance: Frontier model (Anthropic) for complex decisions (~2s)

Follows the same pattern as GEPASettings/GEPATierManager.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class DecisionModelSpec(BaseModel):
    """Provider/model specification for one decision tier."""

    provider: str = "ollama"
    model: str = "qwen3.5:2b"
    timeout_ms: int = 4000
    max_tokens: int = 50


class DecisionServiceSettings(BaseModel):
    """Tiered decision service configuration.

    Each tier defines a provider/model pair. Decision types are
    routed to tiers via the tier_routing map. Fallback chain:
    performance → balanced → edge → heuristic.
    """

    enabled: bool = True

    # Tier definitions (mirrors GEPASettings pattern)
    edge: DecisionModelSpec = Field(
        default_factory=lambda: DecisionModelSpec(
            provider="ollama", model="qwen3.5:2b", timeout_ms=4000, max_tokens=50
        )
    )
    balanced: DecisionModelSpec = Field(
        default_factory=lambda: DecisionModelSpec(
            provider="deepseek", model="deepseek-chat", timeout_ms=8000, max_tokens=200
        )
    )
    performance: DecisionModelSpec = Field(
        default_factory=lambda: DecisionModelSpec(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            timeout_ms=15000,
            max_tokens=500,
        )
    )

    # Decision type → tier mapping
    tier_routing: Dict[str, str] = Field(
        default_factory=lambda: {
            "tool_selection": "edge",
            "skill_selection": "edge",
            "stage_detection": "edge",
            "intent_classification": "edge",
            "task_completion": "edge",
            "error_classification": "edge",
            "continuation_action": "edge",
            "loop_detection": "edge",
            "prompt_focus": "edge",
            "question_classification": "edge",
            "task_type_classification": "balanced",
            "multi_skill_decomposition": "balanced",
        }
    )
