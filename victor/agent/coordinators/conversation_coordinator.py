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

"""Conversation coordinator for agent orchestration.

This module provides the ConversationCoordinator which extracts
conversation management responsibilities from the orchestrator.

Design Pattern: Coordinator (SRP Compliance)
- Message history management
- Turn tracking and state
- Context window management
- Conversation summarization
- Message deduplication

Extracted from AgentOrchestrator to improve modularity and testability
as part of the SOLID refactoring initiative (Phase 2).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from victor.core.messages import Message

logger = logging.getLogger(__name__)


class TurnType(Enum):
    """Types of conversation turns."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ConversationTurn:
    """A single turn in the conversation.

    Attributes:
        turn_id: Unique identifier for this turn
        turn_type: Type of turn (user, assistant, system, tool)
        content: Message content
        timestamp: When this turn occurred
        metadata: Additional turn metadata
        tool_calls: Tool calls made in this turn (if any)
    """

    turn_id: str
    turn_type: TurnType
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "turn_type": self.turn_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
        }


@dataclass
class ConversationStats:
    """Statistics about the conversation.

    Attributes:
        total_turns: Total number of turns
        user_turns: Number of user turns
        assistant_turns: Number of assistant turns
        tool_calls: Total tool calls made
        total_tokens: Estimated total tokens
        start_time: Conversation start time
        last_activity: Last activity timestamp
    """

    total_turns: int = 0
    user_turns: int = 0
    assistant_turns: int = 0
    tool_turns: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    start_time: float = 0.0
    last_activity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_turns": self.total_turns,
            "user_turns": self.user_turns,
            "assistant_turns": self.assistant_turns,
            "tool_turns": self.tool_turns,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_tokens,
            "start_time": self.start_time,
            "last_activity": self.last_activity,
            "duration_seconds": self.last_activity - self.start_time if self.start_time else 0,
        }


@dataclass
class ConversationContext:
    """Context for the current conversation state.

    Attributes:
        turn_count: Current turn number
        context_window_size: Size of context window in tokens
        max_history_turns: Maximum turns to keep in history
        needs_summarization: Whether conversation needs summarization
        summarization_threshold: Turns before triggering summarization
    """

    turn_count: int = 0
    context_window_size: int = 128000
    max_history_turns: int = 50
    needs_summarization: bool = False
    summarization_threshold: int = 40

    def should_summarize(self) -> bool:
        """Check if conversation should be summarized.

        Returns:
            True if summarization is needed
        """
        return self.turn_count >= self.summarization_threshold


class ConversationCoordinator:
    """Coordinates conversation management for the orchestrator.

    This class extracts conversation-related responsibilities from the
    orchestrator, providing a focused interface for:

    1. Message History: Track and manage conversation turns
    2. Turn Tracking: Track turn count, types, and metadata
    3. Context Window: Manage context window size and summarization
    4. Deduplication: Detect and handle duplicate messages
    5. Statistics: Track conversation metrics

    Example:
        coordinator = ConversationCoordinator(
            max_history_turns=50,
            summarization_threshold=40,
        )

        # Add a message
        coordinator.add_message(
            role="user",
            content="Hello, world!",
            turn_type=TurnType.USER,
        )

        # Get conversation history
        history = coordinator.get_history()

        # Get stats
        stats = coordinator.get_stats()
    """

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
        context_window_size: int = 128000,
        enable_deduplication: bool = True,
        enable_statistics: bool = True,
    ):
        """Initialize the conversation coordinator.

        Args:
            max_history_turns: Maximum turns to keep in history
            summarization_threshold: Turns before triggering summarization
            context_window_size: Size of context window in tokens
            enable_deduplication: Whether to enable message deduplication
            enable_statistics: Whether to track conversation statistics
        """
        self._max_history_turns = max_history_turns
        self._summarization_threshold = summarization_threshold
        self._context_window_size = context_window_size
        self._enable_deduplication = enable_deduplication
        self._enable_statistics = enable_statistics

        # Conversation storage
        self._turns: deque[ConversationTurn] = deque(maxlen=max_history_turns)
        self._turn_counter: int = 0
        self._message_hashes: Set[str] = set()

        # Conversation state
        self._context: ConversationContext = ConversationContext(
            turn_count=0,
            context_window_size=context_window_size,
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
        )

        # Statistics
        self._stats: ConversationStats = ConversationStats()
        self._stats.start_time = time.time()
        self._stats.last_activity = self._stats.start_time

        # Summarization state
        self._summaries: List[str] = []
        self._last_summary_turn: int = 0

        logger.debug(
            f"ConversationCoordinator initialized with max_turns={max_history_turns}, "
            f"summarization_threshold={summarization_threshold}"
        )

    # ========================================================================
    # Message Management
    # ========================================================================

    def add_message(
        self,
        role: str,
        content: str,
        turn_type: TurnType,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            turn_type: Type of turn
            metadata: Optional metadata
            tool_calls: Optional tool calls made in this turn

        Returns:
            Turn ID for the added message
        """
        self._turn_counter += 1
        turn_id = f"turn_{self._turn_counter}"

        # Create turn
        turn = ConversationTurn(
            turn_id=turn_id,
            turn_type=turn_type,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
            tool_calls=tool_calls or [],
        )

        # Check for duplicates
        if self._enable_deduplication:
            content_hash = self._hash_content(content)
            if content_hash in self._message_hashes:
                logger.debug(f"Duplicate message detected: {turn_id}")
                return turn_id
            self._message_hashes.add(content_hash)

        # Add to history
        self._turns.append(turn)
        self._context.turn_count = self._turn_counter

        # Update statistics
        if self._enable_statistics:
            self._update_stats(turn)

        # Check if summarization is needed
        self._context.needs_summarization = self._context.should_summarize()

        logger.debug(f"Added turn {turn_id}, total turns: {self._turn_counter}")

        return turn_id

    def get_history(
        self,
        max_turns: Optional[int] = None,
        include_system: bool = True,
        include_tool: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get conversation history.

        Args:
            max_turns: Maximum number of turns to return
            include_system: Whether to include system messages
            include_tool: Whether to include tool messages

        Returns:
            List of message dictionaries
        """
        turns = list(self._turns)

        # Filter by type
        if not include_system:
            turns = [t for t in turns if t.turn_type != TurnType.SYSTEM]
        if not include_tool:
            turns = [t for t in turns if t.turn_type != TurnType.TOOL]

        # Limit turns
        if max_turns:
            turns = turns[-max_turns:]

        return [self._turn_to_message(t) for t in turns]

    def get_last_n_turns(self, n: int) -> List[ConversationTurn]:
        """Get the last n turns from the conversation.

        Args:
            n: Number of turns to retrieve

        Returns:
            List of the last n turns
        """
        return list(self._turns)[-n:]

    def clear_history(self, keep_summaries: bool = True) -> None:
        """Clear conversation history.

        Args:
            keep_summaries: Whether to keep conversation summaries
        """
        self._turns.clear()
        self._message_hashes.clear()
        self._context.turn_count = 0
        self._context.needs_summarization = False

        if not keep_summaries:
            self._summaries.clear()
            self._last_summary_turn = 0

        logger.info("Conversation history cleared")

    def remove_turn(self, turn_id: str) -> bool:
        """Remove a specific turn from history.

        Args:
            turn_id: ID of the turn to remove

        Returns:
            True if turn was removed, False if not found
        """
        for i, turn in enumerate(self._turns):
            if turn.turn_id == turn_id:
                removed = self._turns[i]
                del self._turns[i]

                # Update stats
                if self._enable_statistics and removed.turn_type == TurnType.USER:
                    self._stats.user_turns = max(0, self._stats.user_turns - 1)
                elif self._enable_statistics and removed.turn_type == TurnType.ASSISTANT:
                    self._stats.assistant_turns = max(0, self._stats.assistant_turns - 1)

                logger.debug(f"Removed turn {turn_id}")
                return True

        return False

    # ========================================================================
    # Context Window Management
    # ========================================================================

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def get_context_usage(self) -> Tuple[int, int]:
        """Get current context window usage.

        Returns:
            Tuple of (used_tokens, total_tokens)
        """
        used = sum(self.estimate_tokens(turn.content) for turn in self._turns)
        return used, self._context_window_size

    def get_context_utilization(self) -> float:
        """Get context window utilization as a percentage.

        Returns:
            Utilization percentage (0.0 to 1.0)
        """
        used, total = self.get_context_usage()
        return used / total if total > 0 else 0.0

    def truncate_history_if_needed(self, max_tokens: Optional[int] = None) -> int:
        """Truncate history if context window is exceeded.

        Args:
            max_tokens: Maximum tokens to keep (uses context_window_size if None)

        Returns:
            Number of turns removed
        """
        max_tok = max_tokens or self._context_window_size
        removed = 0

        while self.get_context_usage()[0] > max_tok and len(self._turns) > 2:
            # Remove oldest non-system turn
            for i, turn in enumerate(self._turns):
                if turn.turn_type != TurnType.SYSTEM:
                    del self._turns[i]
                    removed += 1
                    break

        if removed > 0:
            logger.info(f"Truncated {removed} turns to fit context window")

        return removed

    # ========================================================================
    # Summarization
    # ========================================================================

    def needs_summarization(self) -> bool:
        """Check if conversation needs summarization.

        Returns:
            True if summarization is recommended
        """
        return self._context.needs_summarization

    def add_summary(self, summary: str) -> None:
        """Add a conversation summary.

        Args:
            summary: Summary text
        """
        self._summaries.append(summary)
        self._last_summary_turn = self._turn_counter

        # Clear old turns after summarization
        turns_to_keep = self._max_history_turns // 2
        removed = 0
        if len(self._turns) > turns_to_keep:
            removed = len(self._turns) - turns_to_keep
            for _ in range(removed):
                self._turns.popleft()

        self._context.needs_summarization = False
        if removed > 0:
            logger.info(f"Added summary, cleared {removed} old turns")

    def get_summaries(self) -> List[str]:
        """Get all conversation summaries.

        Returns:
            List of summary strings
        """
        return self._summaries.copy()

    def get_full_context(self) -> str:
        """Get full conversation context including summaries.

        Returns:
            Formatted context string
        """
        parts = []

        # Add summaries
        if self._summaries:
            parts.append("# Previous Conversation Summaries")
            parts.extend(self._summaries)
            parts.append("")

        # Add recent turns
        parts.append("# Recent Conversation")
        for turn in self._turns:
            role = turn.turn_type.value.upper()
            parts.append(f"{role}: {turn.content}")

        return "\n".join(parts)

    # ========================================================================
    # Deduplication
    # ========================================================================

    def is_duplicate(self, content: str) -> bool:
        """Check if content is a duplicate.

        Args:
            content: Content to check

        Returns:
            True if duplicate detected
        """
        content_hash = self._hash_content(content)
        return content_hash in self._message_hashes

    def _hash_content(self, content: str) -> str:
        """Hash content for deduplication.

        Args:
            content: Content to hash

        Returns:
            Hash string
        """
        import hashlib

        return hashlib.md5(content.encode()).hexdigest()

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_stats(self) -> ConversationStats:
        """Get conversation statistics.

        Returns:
            ConversationStats object
        """
        self._stats.total_turns = len(self._turns)
        self._stats.last_activity = time.time()
        return self._stats

    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary.

        Returns:
            Dictionary of statistics
        """
        stats = self.get_stats()
        return stats.to_dict()

    def _update_stats(self, turn: ConversationTurn) -> None:
        """Update statistics for a turn.

        Args:
            turn: Turn to update stats for
        """
        self._stats.total_turns = len(self._turns)
        self._stats.last_activity = turn.timestamp

        if turn.turn_type == TurnType.USER:
            self._stats.user_turns += 1
        elif turn.turn_type == TurnType.ASSISTANT:
            self._stats.assistant_turns += 1
        elif turn.turn_type == TurnType.TOOL:
            self._stats.tool_turns += 1
            self._stats.tool_calls += len(turn.tool_calls)

        # Estimate tokens
        self._stats.total_tokens += self.estimate_tokens(turn.content)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _turn_to_message(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Convert a turn to a message dictionary.

        Args:
            turn: Turn to convert

        Returns:
            Message dictionary
        """
        return {
            "role": turn.turn_type.value,
            "content": turn.content,
            "timestamp": turn.timestamp,
            "turn_id": turn.turn_id,
            "metadata": turn.metadata,
        }

    def get_turn_count(self) -> int:
        """Get current turn count.

        Returns:
            Number of turns in conversation
        """
        return self._turn_counter

    def is_empty(self) -> bool:
        """Check if conversation is empty.

        Returns:
            True if no turns have been added
        """
        return len(self._turns) == 0

    def reset(self) -> None:
        """Reset conversation state."""
        self._turns.clear()
        self._message_hashes.clear()
        self._turn_counter = 0
        self._context.turn_count = 0
        self._context.needs_summarization = False
        self._summaries.clear()
        self._last_summary_turn = 0

        self._stats = ConversationStats()
        self._stats.start_time = time.time()
        self._stats.last_activity = self._stats.start_time

        logger.info("ConversationCoordinator reset")

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data for dashboard integration.

        Returns:
            Dictionary with observability data
        """
        stats = self.get_stats()
        used_tokens, total_tokens = self.get_context_usage()

        return {
            "source_id": f"coordinator:conversation:{id(self)}",
            "source_type": "coordinator",
            "coordinator_type": "conversation",
            "stats": stats.to_dict(),
            "context": {
                "turn_count": self._turn_counter,
                "context_usage": {
                    "used_tokens": used_tokens,
                    "total_tokens": total_tokens,
                    "utilization_percent": self.get_context_utilization() * 100,
                },
                "needs_summarization": self._context.needs_summarization,
                "summary_count": len(self._summaries),
            },
        }


__all__ = [
    "ConversationCoordinator",
    "TurnType",
    "ConversationTurn",
    "ConversationStats",
    "ConversationContext",
]
