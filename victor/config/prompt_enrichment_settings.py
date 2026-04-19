"""Prompt enrichment and optimization configuration."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PromptEnrichmentSettings(BaseModel):
    """Prompt enrichment and optimization configuration."""

    prompt_enrichment_enabled: bool = True
    prompt_enrichment_max_tokens: int = 2000
    prompt_enrichment_timeout_ms: float = 500.0
    prompt_enrichment_cache_enabled: bool = True
    prompt_enrichment_cache_ttl: int = 300
    prompt_enrichment_strategies: List[str] = Field(
        default_factory=lambda: [
            "knowledge_graph",
            "conversation",
            "web_search",
        ],
    )
    prompt_enrichment_coding: bool = True
    prompt_enrichment_research: bool = True
    prompt_enrichment_devops: bool = True
    prompt_enrichment_data_analysis: bool = True
