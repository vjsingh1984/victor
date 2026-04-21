"""Tool execution, selection, and retry configuration."""

from __future__ import annotations

import os
from typing import Dict, List, Set

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

    # Tool output preview configuration (user display only, does NOT affect LLM)
    tool_output_preview_enabled: bool = Field(
        default=True,
        description="Show tool output preview to user (default: yes for transparency)",
    )
    tool_output_preview_lines: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of lines to show in tool output preview (default: 3)",
    )
    tool_output_pruning_enabled: bool = Field(
        default=True,
        description="[DEPRECATED - Now only affects user preview, NOT LLM input] Enable tool output pruning for user display",
    )
    tool_output_pruning_safe_only: bool = Field(
        default=True,
        description="Prune only safe verbose read-heavy tool outputs by default; set false for broader opt-in pruning",
    )
    tool_output_show_transparency: bool = Field(
        default=True,
        description="Show preview status to user when output was truncated for display (default: yes)",
    )
    tool_output_expand_hotkey: str = Field(
        default="^O",
        description="Hotkey to expand tool output preview (default: Ctrl+O, format: ^X for Ctrl+X)",
    )

    # Tool output byte limit for display
    # This is separate from pruning and affects what users see in terminal/UI
    # When pruning is disabled (accuracy-first), byte limit should be higher
    tool_output_byte_limit_mb: float = Field(
        default=10.0,  # 10MB default (much higher when accuracy matters)
        ge=1.0,  # Minimum 1MB
        le=100.0,  # Maximum 100MB
        description="Maximum tool output size in MB for display (default: 10MB, higher when pruning disabled)",
    )

    # Embedding-intensive tool concurrency configuration
    # These tools use embedding models and need lower concurrency limits
    # to prevent resource exhaustion (memory, CPU, embedding model contention)
    max_embedding_concurrent: int = Field(
        default_factory=lambda: int(os.getenv("VICTOR_MAX_EMBEDDING_CONCURRENT", "2")),
        ge=1,
        le=10,
        description="Maximum concurrent embedding-intensive tool executions (default: 2, env: VICTOR_MAX_EMBEDDING_CONCURRENT)",
    )
    embedding_intensive_tools: Set[str] = Field(
        default_factory=lambda: set(
            os.getenv(
                "VICTOR_EMBEDDING_INTENSIVE_TOOLS",
                "code_search,semantic_code_search",
            ).split(",")
        ),
        description="Tool names that require embedding concurrency limits (default: code_search,semantic_code_search, env: VICTOR_EMBEDDING_INTENSIVE_TOOLS)",
    )


def get_tool_settings() -> ToolSettings:
    """Get the current tool settings from the global settings singleton.

    Returns:
        ToolSettings instance with current configuration

    Example:
        >>> from victor.config.tool_settings import get_tool_settings
        >>> settings = get_tool_settings()
        >>> if settings.tool_output_preview_enabled:
        ...     show_preview()
    """
    from victor.config.settings import get_settings

    global_settings = get_settings()
    if global_settings.tool_settings is None:
        # Return default ToolSettings if not configured
        return ToolSettings()
    return global_settings.tool_settings
