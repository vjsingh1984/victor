# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compaction strategy settings for hybrid LLM + rule-based compaction system.

This module defines the settings architecture for the hybrid compaction system
that combines Claudecode's deterministic rule-based approach with Victor's
sophisticated LLM-based approach.

Key design principles:
1. Fast path common cases: 80% of compactions use rules (sub-100ms)
2. LLM for complex cases: 20% get rich summaries when complexity warrants
3. Dual storage: Store both XML (machine-readable) and natural language summaries
4. Graceful degradation: LLM failures fall back to rules automatically
5. Zero breaking changes: Full backward compatibility with existing system
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class CompactionStrategySettings(BaseModel):
    """Settings for compaction strategy selection and behavior.

    Controls the hybrid compaction system that intelligently chooses between
    rule-based (fast, deterministic) and LLM-based (rich, intelligent) compaction
    strategies based on message complexity, token count, and settings thresholds.
    """

    # Strategy selection thresholds
    llm_min_complexity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum complexity score (0.0-1.0) to use LLM compaction. "
        "Lower values use LLM more frequently; higher values prefer rules.",
    )
    llm_min_tokens: int = Field(
        default=5000,
        ge=1000,
        le=100000,
        description="Minimum estimated token count to consider LLM compaction. "
        "Smaller conversations use rules or hybrid approach.",
    )
    llm_min_messages: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Minimum message count to consider LLM compaction. "
        "Shorter conversations prefer faster rule-based compaction.",
    )

    # Rule-based settings (claudecode-style)
    rule_preserve_recent: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of recent messages to preserve verbatim in rule-based compaction. "
        "Recent context is most valuable for continuation.",
    )
    rule_max_estimated_tokens: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Token threshold for rule-based compaction. "
        "Conversations below this threshold can use fast rule-based compaction.",
    )
    rule_xml_format: bool = Field(
        default=True,
        description="Generate XML format summaries (machine-readable). "
        "XML format enables structured parsing and better context preservation.",
    )

    # LLM-based settings
    llm_provider: Optional[str] = Field(
        default=None,
        description="Provider to use for LLM compaction (e.g., 'anthropic', 'openai'). "
        "None = auto-detect from main session provider.",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Model to use for LLM compaction (e.g., 'claude-3-haiku'). "
        "None = auto-detect based on provider and complexity.",
    )
    llm_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Timeout for LLM summarization calls. "
        "LLM failures fall back to rule-based compaction automatically.",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retries for LLM compaction on transient failures.",
    )
    llm_max_input_chars: int = Field(
        default=8000,
        ge=1000,
        le=100000,
        description="Maximum input characters to send to LLM for summarization.",
    )
    llm_max_summary_tokens: int = Field(
        default=300,
        ge=100,
        le=1000,
        description="Maximum tokens to generate for LLM-based summary.",
    )

    # Hybrid settings
    hybrid_llm_enhancement: bool = Field(
        default=True,
        description="Whether to enhance rule-based summaries with LLM in hybrid mode. "
        "When True, hybrid mode uses rules for structure + LLM for key sections.",
    )
    hybrid_llm_sections: List[str] = Field(
        default=["pending_work", "current_work"],
        description="Which summary sections to enhance with LLM in hybrid mode. "
        "Options: 'pending_work', 'current_work', 'tools_mentioned', 'key_files_referenced'.",
    )

    # Storage settings
    store_both_formats: bool = Field(
        default=True,
        description="Store both XML and natural language summaries in database. "
        "Enables dual-format retrieval for different use cases.",
    )
    store_compaction_history: bool = Field(
        default=True,
        description="Log compaction events to analytics table for monitoring and optimization.",
    )

    # Performance settings
    enable_async_compaction: bool = Field(
        default=False,
        description="Enable asynchronous compaction (experimental). "
        "When True, compaction runs in background to avoid blocking main thread.",
    )
    compaction_queue_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max number of pending compaction jobs when async is enabled.",
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            # Encode frozenset as list for JSON serialization
            frozenset: lambda v: list(v),
        }


class CompactionFeatureFlags(BaseModel):
    """Feature flags for compaction enhancements.

    Controls the availability of different compaction strategies.
    Used for gradual rollout and A/B testing.
    """

    enable_rule_based: bool = Field(
        default=True,
        description="Enable rule-based compaction (claudecode-style). "
        "Fast, deterministic, sub-100ms compaction using structured rules.",
    )
    enable_llm_based: bool = Field(
        default=True,
        description="Enable LLM-based compaction. "
        "Rich, intelligent summaries using fast LLM calls with fallback.",
    )
    enable_hybrid: bool = Field(
        default=True,
        description="Enable hybrid compaction combining rules + LLM. "
        "Best of both worlds: fast base with rich enhancements.",
    )
    enable_json_storage: bool = Field(
        default=True,
        description="Enable JSON1 extension usage for structured data storage. "
        "Improves query performance and enables complex JSON operations.",
    )
    enable_compaction_analytics: bool = Field(
        default=True,
        description="Enable compaction history tracking for monitoring and optimization.",
    )

    class Config:
        """Pydantic config."""

        validate_assignment = True
        extra = "forbid"
