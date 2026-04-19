"""Tool execution, selection, and retry configuration."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from victor.config.model_capabilities import _load_tool_capable_patterns_from_yaml
from victor.config.orchestrator_constants import BUDGET_LIMITS


class ToolSettings(BaseModel):
    """Tool execution, selection, and retry configuration."""

    tool_call_budget: int = Field(default_factory=lambda: BUDGET_LIMITS.max_session_budget)
    tool_call_budget_warning_threshold: int = Field(
        default_factory=lambda: int(
            BUDGET_LIMITS.max_session_budget * BUDGET_LIMITS.warning_threshold_pct
        )
    )
    tool_calling_models: Dict[str, list[str]] = Field(
        default_factory=_load_tool_capable_patterns_from_yaml
    )
    tool_retry_enabled: bool = True
    tool_retry_max_attempts: int = 3
    tool_retry_base_delay: float = 1.0
    tool_retry_max_delay: float = 10.0
    fallback_max_tools: int = 8
    enable_tool_deduplication: bool = True
    tool_deduplication_window_size: int = 20
    use_semantic_tool_selection: bool = True
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    tool_cache_enabled: bool = True
    tool_cache_ttl: int = 600
    tool_cache_allowlist: List[str] = Field(
        default_factory=lambda: [
            "read",
            "ls",
            "grep",
            "overview",
            "diff",
            "code_search",
            "semantic_code_search",
            "list_directory",
            "plan_files",
        ]
    )
    generic_result_cache_enabled: bool = False
    generic_result_cache_ttl: int = 300
    tool_selection_cache_enabled: bool = True
    tool_selection_cache_ttl: int = 300
    tool_validation_mode: str = "lenient"
    # Token budget for tool schemas broadcast to LLM.
    # When total estimated tokens exceed this budget, lowest-ranked tools
    # are demoted (COMPACT→STUB) or dropped (tail STUBs removed).
    # Default 4000 is a safe ceiling: with fallback_max_tools=8, the tiered
    # selection typically uses ~650 tokens, so this only triggers if someone
    # raises fallback_max_tools significantly. Set 0 to disable enforcement.
    max_tool_schema_tokens: int = 4000
    # Promote high-confidence semantic matches from STUB→COMPACT.
    # Tools with semantic similarity >= this threshold get richer schemas.
    schema_promotion_threshold: float = 0.8
    # Maximum MCP tools to broadcast per turn (when relevance filtering is active)
    max_mcp_tools_per_turn: int = 12
    # Enable cross-turn tool result deduplication (session-scoped cache)
    cross_turn_dedup_enabled: bool = True
    cross_turn_dedup_ttl: int = 300  # seconds
