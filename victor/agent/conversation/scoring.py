# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Unified message scoring for context management.

Single canonical scorer used by both ConversationController (runtime)
and ConversationStore (persistence). Replaces two separate scoring
implementations with one configurable function.

Supports Rust-accelerated batch scoring via _NATIVE_AVAILABLE pattern.

Usage:
    from victor.agent.conversation.scoring import score_messages, CONTROLLER_WEIGHTS

    scored = score_messages(messages, current_query="fix the bug", weights=CONTROLLER_WEIGHTS)
    # Returns: List[Tuple[ConversationMessage, float]] sorted by score desc
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from victor.agent.conversation.types import ConversationMessage, MessagePriority

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringWeights:
    """Configurable weights for message scoring factors.

    All weights should sum to ~1.0. Each factor produces a 0.0-1.0 score
    that is multiplied by its weight and summed.

    Presets:
        STORE_WEIGHTS: priority 40% + recency 60% (original ConversationStore)
        CONTROLLER_WEIGHTS: role 30% + recency 30% + length 10% + semantic 30%
        DEFAULT_WEIGHTS: balanced across all factors
    """

    priority: float = 0.2
    recency: float = 0.4
    role: float = 0.2
    length: float = 0.1
    semantic: float = 0.1


# Presets matching the two original algorithms
STORE_WEIGHTS = ScoringWeights(priority=0.4, recency=0.6, role=0.0, length=0.0, semantic=0.0)
CONTROLLER_WEIGHTS = ScoringWeights(priority=0.0, recency=0.3, role=0.3, length=0.1, semantic=0.3)
DEFAULT_WEIGHTS = ScoringWeights()

# Role importance scores (from ConversationController's scoring)
# Adjusted to prevent system message hoarding (was 1.0 for system)
# 'tool' matches MessageRole.TOOL (OpenAI spec for tool results)
_ROLE_SCORES: Dict[str, float] = {
    "system": 0.7,        # High but not absolute
    "user": 0.8,          # Higher than system to ensure user intent survives
    "assistant": 0.6,
    "tool": 0.9,          # Tool results (role=tool) are critical for task state
    "tool_call": 0.7,     # Internal role for assistant tool requests
}


def score_messages(
    messages: List[ConversationMessage],
    current_query: Optional[str] = None,
    *,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
    embedding_fn: Optional[Callable[[List[str], str], List[float]]] = None,
) -> List[Tuple[ConversationMessage, float]]:
    """Score messages for context selection.

    Canonical scorer for the entire conversation system. Uses configurable
    weights to balance priority, recency, role importance, content length,
    and semantic similarity factors.

    Attempts Rust-accelerated batch scoring when available (4.8-9.9x speedup)
    for the priority+recency factors. Falls back to Python for all factors.

    Args:
        messages: Messages to score
        current_query: Optional query for semantic relevance scoring
        weights: Scoring weight configuration
        embedding_fn: Optional function(texts, query) -> similarity scores.
            When provided and weights.semantic > 0, computes semantic
            similarity between messages and the current query.

    Returns:
        List of (message, score) tuples, sorted by score descending
    """
    if not messages:
        return []

    # Fast path: pure priority+recency scoring with Rust acceleration
    if (
        weights.role == 0
        and weights.length == 0
        and weights.semantic == 0
        and weights.priority > 0
        and weights.recency > 0
    ):
        result = _score_priority_recency_fast(messages, weights)
        if result is not None:
            return result

    # Full scoring path (Python)
    now = datetime.now(tz=timezone.utc)
    max_age = _max_age_seconds(messages, now)

    # Pre-compute semantic similarities if needed
    similarities: Optional[List[float]] = None
    if weights.semantic > 0 and current_query and embedding_fn:
        try:
            texts = [m.content for m in messages]
            similarities = embedding_fn(texts, current_query)
        except Exception as e:
            logger.debug("Semantic scoring failed: %s", e)

    scored: List[Tuple[ConversationMessage, float]] = []
    for i, msg in enumerate(messages):
        score = 0.0

        # Priority factor
        if weights.priority > 0:
            priority_val = msg.priority.value if isinstance(msg.priority, MessagePriority) else 50
            score += (priority_val / 100.0) * weights.priority

        # Recency factor
        if weights.recency > 0:
            age = (now - msg.timestamp).total_seconds()
            recency = 1.0 - (age / max_age) if max_age > 0 else 1.0
            score += recency * weights.recency

        # Role importance factor
        if weights.role > 0:
            role_score = _ROLE_SCORES.get(msg.role, 0.5)
            score += role_score * weights.role

        # Content length factor (substantive messages score higher)
        if weights.length > 0:
            length_score = min(len(msg.content) / 500.0, 1.0)
            score += length_score * weights.length

        # Semantic similarity factor
        if weights.semantic > 0 and similarities is not None:
            score += similarities[i] * weights.semantic

        scored.append((msg, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _score_priority_recency_fast(
    messages: List[ConversationMessage],
    weights: ScoringWeights,
) -> Optional[List[Tuple[ConversationMessage, float]]]:
    """Try Rust-accelerated scoring for priority+recency only.

    Returns None if Rust is unavailable, triggering Python fallback.
    """
    try:
        from victor.processing.native.context_fitter import batch_score_messages

        priorities = [
            int(msg.priority.value) if isinstance(msg.priority, MessagePriority) else 50
            for msg in messages
        ]
        timestamps = [msg.timestamp.timestamp() for msg in messages]
        scored_indices = batch_score_messages(priorities, timestamps)
        return [(messages[idx], score) for idx, score in scored_indices]
    except (ImportError, Exception):
        return None


def _max_age_seconds(messages: List[ConversationMessage], now: datetime) -> float:
    """Calculate max age across messages for normalization."""
    if not messages:
        return 1.0
    ages = [(now - m.timestamp).total_seconds() for m in messages]
    return max(ages) or 1.0


__all__ = [
    "CONTROLLER_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "STORE_WEIGHTS",
    "ScoringWeights",
    "score_messages",
]
