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

"""Context compaction strategy protocol for dependency inversion.

This module defines the ICompactionStrategy protocol that enables
dependency injection for context compaction, following the
Dependency Inversion Principle (DIP).

Design Principles:
    - DIP: ContextCoordinator depends on this protocol, not concrete strategies
    - OCP: New compaction strategies can be added without modifying existing code
    - SRP: Each strategy handles one compaction approach
    - Strategy Pattern: Different algorithms for context reduction

Usage:
    class TruncationCompactionStrategy(ICompactionStrategy):
        async def can_apply(self, context: ConversationContext, budget: ContextBudget) -> bool:
            return context.token_count > budget.max_tokens

        async def compact(self, context: ConversationContext, budget: ContextBudget) -> CompactionResult:
            # Remove oldest messages until within budget
            ...

        def estimated_savings(self) -> int:
            return 5000  # Estimated token savings

    class SummarizationCompactionStrategy(ICompactionStrategy):
        async def compact(self, context: ConversationContext, budget: ContextBudget) -> CompactionResult:
            # Summarize old messages using LLM
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Forward reference - ConversationContext is defined in victor/framework/conversations/
    # This avoids circular imports
    class ConversationContext:
        """Placeholder for type checking."""
        token_count: int


@dataclass
class CompactionResult:
    """Result from context compaction operation.

    Attributes:
        compacted_context: The compacted conversation context
        tokens_saved: Number of tokens saved by compaction
        messages_removed: Number of messages removed
        strategy_used: Name of the strategy that was applied
        metadata: Additional metadata about the compaction
    """

    compacted_context: "ConversationContext"
    tokens_saved: int
    messages_removed: int
    strategy_used: str
    metadata: Dict[str, Any] | None = None


@runtime_checkable
class ICompactionStrategy(Protocol):
    """Protocol for context compaction strategies.

    Implementations provide different algorithms for reducing
    conversation context size when it exceeds the token budget.

    Common strategies:
    - Truncation: Remove oldest messages
    - Summarization: Use LLM to summarize old messages
    - Semantic: Remove semantically similar messages
    - Priority-based: Remove low-priority messages first
    - Hybrid: Combine multiple approaches

    The ContextCoordinator selects the best strategy based on
    context characteristics and budget constraints.
    """

    async def can_apply(
        self,
        context: "ConversationContext",
        budget: "ContextBudget",
    ) -> bool:
        """Check if this strategy can be applied to the context.

        Args:
            context: Current conversation context
            budget: Context budget with token limits

        Returns:
            True if strategy can be applied, False otherwise

        Example:
            async def can_apply(self, context, budget) -> bool:
                # Only apply if context exceeds budget by > 20%
                return context.token_count > budget.max_tokens * 1.2
        """
        ...

    async def compact(
        self,
        context: "ConversationContext",
        budget: "ContextBudget",
    ) -> CompactionResult:
        """Apply compaction strategy to reduce context size.

        Args:
            context: Current conversation context to compact
            budget: Context budget with token limits and constraints

        Returns:
            CompactionResult with compacted context and metadata

        Example:
            async def compact(self, context, budget) -> CompactionResult:
                # Remove oldest messages until within budget
                compacted = self._remove_oldest(context, budget)
                tokens_saved = context.token_count - compacted.token_count
                return CompactionResult(
                    compacted_context=compacted,
                    tokens_saved=tokens_saved,
                    messages_removed=len(context.messages) - len(compacted.messages),
                    strategy_used="truncation",
                )
        """
        ...

    def estimated_savings(self) -> int:
        """Estimated token savings for this strategy.

        Used by ContextCoordinator to select the best strategy
        when multiple strategies can be applied.

        Returns:
            Estimated number of tokens that would be saved

        Example:
            def estimated_savings(self) -> int:
                return 5000  # Estimates saving ~5000 tokens
        """
        ...


# Dict-based models for context-related types (instantiable for tests).
class CompactionContext(dict):
    """Context for context compaction operations.

    This is distinct from ConversationContext in victor/framework/conversations/
    which is used for multi-agent conversations. This type is specifically
    for context compaction strategies.

    Common keys:
        - messages: List[Dict[str, str]] - Conversation messages
        - token_count: int - Total token count
        - metadata: Dict[str, Any] - Additional context metadata
    """


class ContextBudget(dict):
    """Context budget with token limits.

    Common keys:
        - max_tokens: int - Maximum token limit
        - reserve_tokens: int - Tokens to reserve for response
        - min_messages: int - Minimum messages to keep
    """


__all__ = ["ICompactionStrategy", "CompactionResult", "CompactionContext", "ContextBudget"]
