"""Conversation persistence and history limits."""

from __future__ import annotations

from pydantic import BaseModel


class ConversationSettings(BaseModel):
    """Conversation persistence and history limits."""

    conversation_memory_enabled: bool = True
    conversation_embeddings_enabled: bool = True
    max_conversation_history: int = 100
    session_idle_timeout: int = 180
