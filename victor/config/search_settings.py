"""Codebase search and semantic configuration."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchSettings(BaseModel):
    """Codebase search and semantic configuration."""

    unified_embedding_model: str = "BAAI/bge-small-en-v1.5"
    codebase_vector_store: str = "lancedb"
    codebase_embedding_provider: str = "sentence-transformers"
    codebase_embedding_model: str = "BAAI/bge-small-en-v1.5"
    codebase_persist_directory: Optional[str] = None
    codebase_dimension: int = 384
    codebase_batch_size: int = 32
    codebase_graph_store: str = "sqlite"
    codebase_graph_path: Optional[str] = None
    core_readonly_tools: Optional[List[str]] = None
    semantic_similarity_threshold: float = 0.25
    semantic_query_expansion_enabled: bool = True
    semantic_max_query_expansions: int = 5
    enable_hybrid_search: bool = False
    hybrid_search_semantic_weight: float = 0.6
    hybrid_search_keyword_weight: float = 0.4
    enable_semantic_threshold_rl_learning: bool = False
    semantic_threshold_overrides: dict = Field(default_factory=dict)
    # RL mode: "full" (write all), "selective" (skip unchanged), "none" (read-only)
    rl_mode: str = "selective"
