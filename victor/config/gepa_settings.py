"""GEPA v2 prompt optimizer configuration.

Configures trace enrichment (ASI), tiered model selection for
reflection/mutation, Pareto-based candidate selection, and
prompt bloat control.

Three model tiers:
- economic: Local model (Ollama qwen3:8b) — post-convergence maintenance
- balanced: Mid-tier cloud (GPT-4.1-mini) — default
- performance: Frontier (Sonnet) — initial convergence
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GEPAModelSpec(BaseModel):
    """A (provider, model) specification for one GEPA tier."""

    provider: str = "ollama"
    model: str = "qwen3:8b"
    timeout_s: float = 30.0
    max_tokens: int = 1000


class GEPASettings(BaseModel):
    """All GEPA prompt optimizer configuration."""

    enabled: bool = True

    # --- Trace enrichment (ASI) ---
    capture_reasoning: bool = True
    capture_tool_args: bool = True
    capture_tool_output: bool = True
    max_output_chars: int = 2000
    max_reasoning_chars: int = 1000

    # --- Tiered models for reflection/mutation ---
    economic_model: GEPAModelSpec = Field(
        default_factory=lambda: GEPAModelSpec(
            provider="ollama", model="gemma4"
        )
    )
    balanced_model: GEPAModelSpec = Field(
        default_factory=lambda: GEPAModelSpec(
            provider="openai", model="gpt-4.1-mini"
        )
    )
    performance_model: GEPAModelSpec = Field(
        default_factory=lambda: GEPAModelSpec(
            provider="anthropic", model="claude-sonnet-4-20250514"
        )
    )
    default_tier: str = "balanced"

    # --- Auto-tier switching ---
    auto_tier_switch: bool = True
    convergence_window: int = 10
    convergence_threshold: float = 0.02

    # --- Prompt bloat control ---
    max_prompt_chars: int = 1500

    # --- Evolution thresholds ---
    min_traces_for_evolution: int = 5
    max_training_traces: int = 50
    max_candidates_per_section: int = 20
