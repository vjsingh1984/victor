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
