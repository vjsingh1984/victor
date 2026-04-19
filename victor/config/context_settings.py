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

    # Prefix cache optimization: lock tools + system prompt at session start
    # so the provider can cache them at 90% discount (Anthropic, OpenAI, Google).
    # When enabled: full tool set sent every call, reminders via user messages,
    # system prompt frozen after first build.
    cache_optimization_enabled: bool = True

    # KV prefix cache optimization: freeze system prompt and sort tools for
    # prefix stability. Independent from cache_optimization_enabled (API billing).
    # Applies to local providers (Ollama, LMStudio, llama.cpp, MLX, vLLM) that
    # benefit from stable KV prefixes across turns.
    kv_optimization_enabled: bool = True

    # KV tool selection strategy for providers with KV prefix caching (Ollama, etc.)
    # Controls how tools are managed across turns for KV cache stability:
    #   'per_turn'       — Fresh semantic selection each turn (max relevance, breaks prefix)
    #   'session_stable' — Lock semantic selection after first query (stable prefix, may miss tools)
    # Note: API-caching providers always use session-locked full tool set regardless.
    kv_tool_strategy: str = "per_turn"

    # Tiered schema broadcasting: use FULL/COMPACT/STUB schema levels based on
    # TieredToolConfig tier membership. Reduces tool token cost by 50-65% while
    # preserving cache stability (FULL+COMPACT prefix cached, STUB suffix dynamic).
    tiered_schema_enabled: bool = True

    # HITL tool approval mode: controls which tools require human approval.
    #   'auto'      — All tools auto-approved (default, backward compatible)
    #   'dangerous'  — MEDIUM+ danger level tools require approval
    #   'all'        — Every tool call requires approval
    tool_approval_mode: str = "auto"

    # System prompt strategy: controls whether system prompt can vary during session
    #   'static'   — Freeze at session start for cache optimization (default, 50-90% discount)
    #   'dynamic'  — Rebuild per-turn based on context/tool calls (no cache benefit)
    #   'hybrid'   — Static for API providers (cache), dynamic for local providers (no cache benefit)
    system_prompt_strategy: str = "static"
