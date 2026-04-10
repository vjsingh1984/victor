"""Context window management and conversation memory."""

from __future__ import annotations

from pydantic import BaseModel


class ContextSettings(BaseModel):
    """Context window management and conversation memory."""

    context_compaction_strategy: str = "tiered"
    context_min_messages_to_keep: int = 6
    context_tool_retention_weight: float = 1.5
    context_recency_weight: float = 2.0
    context_semantic_threshold: float = 0.3
    max_context_tokens: int = 100000
    response_token_reserve: int = 4096
    conversation_memory_enabled: bool = True
    conversation_embeddings_enabled: bool = True
