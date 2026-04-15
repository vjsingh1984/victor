# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Conversation management package.

Canonical modules:
- types: ConversationMessage, MessageRole, MessagePriority
- scoring: score_messages(), ScoringWeights, weight presets
- controller: ConversationController (runtime state)
- store: ConversationStore (SQLite persistence)
- state_machine: ConversationStateMachine (stage detection)
- assembler: TurnBoundaryContextAssembler (context budget)
"""

from victor.agent.conversation.types import (
    ConversationMessage,
    MessagePriority,
    MessageRole,
)
from victor.agent.conversation.scoring import (
    CONTROLLER_WEIGHTS,
    DEFAULT_WEIGHTS,
    STORE_WEIGHTS,
    ScoringWeights,
    score_messages,
)

# Legacy re-exports from SOLID extraction (will be absorbed in later phases)
from victor.agent.conversation.message_store import MessageStore
from victor.agent.conversation.context_handler import ContextOverflowHandler
from victor.agent.conversation.session_manager import SessionManager
from victor.agent.conversation.embedding_manager import EmbeddingManager

__all__ = [
    # Canonical types
    "ConversationMessage",
    "MessagePriority",
    "MessageRole",
    # Scoring
    "CONTROLLER_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "STORE_WEIGHTS",
    "ScoringWeights",
    "score_messages",
    # Legacy (to be absorbed)
    "MessageStore",
    "ContextOverflowHandler",
    "SessionManager",
    "EmbeddingManager",
]
