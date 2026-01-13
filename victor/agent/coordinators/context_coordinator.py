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

"""Context coordinator for context management and compaction.

This module implements the ContextCoordinator which manages conversation
context and applies compaction strategies when context exceeds budget.

Design Patterns:
    - Strategy Pattern: Multiple compaction strategies via ICompactionStrategy
    - Chain of Responsibility: Try strategies in order
    - SRP: Focused only on context management and compaction

Usage:
    from victor.agent.coordinators.context_coordinator import ContextCoordinator
    from victor.protocols import ICompactionStrategy, CompactionContext, ContextBudget

    # Create coordinator with compaction strategies
    coordinator = ContextCoordinator(strategies=[truncation_strategy, summarization_strategy])

    # Compact context
    result = await coordinator.compact_context(
        context=CompactionContext({"messages": [...], "token_count": 10000}),
        budget=ContextBudget({"max_tokens": 8000})
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.protocols import ICompactionStrategy, CompactionResult, CompactionContext, ContextBudget


logger = logging.getLogger(__name__)


class ContextCoordinator:
    """Context management and compaction coordination.

    This coordinator manages conversation context and applies compaction
    strategies when context size exceeds the token budget.

    Responsibilities:
    - Monitor context size against budget
    - Select appropriate compaction strategy
    - Apply compaction strategies
    - Track compaction history and statistics
    - Support context rebuilding after failures

    Strategies are applied in order, with the first applicable strategy
    being used. Strategies can define their own applicability criteria.
    """

    def __init__(
        self,
        strategies: Optional[List[ICompactionStrategy]] = None,
        enable_auto_compaction: bool = True,
    ) -> None:
        """Initialize the context coordinator.

        Args:
            strategies: List of compaction strategies (ordered by priority)
            enable_auto_compaction: Enable automatic compaction when budget exceeded
        """
        self._strategies = strategies or []
        self._enable_auto_compaction = enable_auto_compaction
        self._compaction_history: List[Dict[str, Any]] = []

    async def compact_context(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> CompactionResult:
        """Compact context to fit within budget.

        Selects and applies the first applicable compaction strategy.

        Args:
            context: Current conversation context
            budget: Context budget with token limits

        Returns:
            CompactionResult with compacted context and metadata

        Raises:
            ContextCompactionError: If no applicable strategy found

        Example:
            result = await coordinator.compact_context(
                context={"messages": [...], "token_count": 10000},
                budget={"max_tokens": 8000}
            )
            if result.tokens_saved > 0:
                print(f"Compacted context, saved {result.tokens_saved} tokens")
        """
        # Check if compaction is needed
        token_count = context.get("token_count", 0)
        max_tokens = budget.get("max_tokens", 4096)

        if token_count <= max_tokens:
            # No compaction needed
            return CompactionResult(
                compacted_context=context,
                tokens_saved=0,
                messages_removed=0,
                strategy_used="none",
                metadata={"reason": "within_budget"},
            )

        # Find applicable strategy
        for strategy in self._strategies:
            try:
                # Check if strategy can be applied
                can_apply = await strategy.can_apply(context, budget)
                if can_apply:
                    # Apply the strategy
                    result = await strategy.compact(context, budget)
                    self._record_compaction(result)
                    return result
            except Exception as e:
                logger.warning(
                    f"Compaction strategy {strategy.__class__.__name__} failed: {e}"
                )
                continue

        # No strategy found
        raise ContextCompactionError(
            f"No applicable compaction strategy found for context with {token_count} tokens "
            f"and budget {max_tokens}"
        )

    async def rebuild_context(
        self,
        session_id: str,
        strategy: str = "truncate",
    ) -> CompactionContext:
        """Rebuild context for a session after failure.

        Rebuilds conversation context from persisted state using the
        specified strategy.

        Args:
            session_id: Session to rebuild context for
            strategy: Rebuild strategy (truncate, summarize, etc.)

        Returns:
            Rebuilt context

        Example:
            context = await coordinator.rebuild_context(
                session_id="abc123",
                strategy="truncate"
            )
        """
        # This would integrate with persistence layer in a full implementation
        # For now, return empty context
        logger.info(f"Rebuilding context for session {session_id} using {strategy} strategy")
        return {
            "messages": [],
            "token_count": 0,
            "rebuilt": True,
            "rebuild_strategy": strategy,
        }

    def estimate_token_count(
        self,
        context: CompactionContext,
    ) -> int:
        """Estimate token count for context.

        Provides a rough estimate of token count for the given context.
        This is useful for budget checking before actual token counting.

        Args:
            context: Conversation context

        Returns:
            Estimated token count

        Example:
            tokens = coordinator.estimate_token_count(context)
            if tokens > budget["max_tokens"]:
                await coordinator.compact_context(context, budget)
        """
        # Use provided token count if available
        if "token_count" in context:
            return context["token_count"]

        # Otherwise estimate from messages
        messages = context.get("messages", [])
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        # Rough estimate: 1 token ≈ 4 characters for English text
        return total_chars // 4

    def is_within_budget(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> bool:
        """Check if context is within budget.

        Args:
            context: Conversation context
            budget: Context budget

        Returns:
            True if context is within budget, False otherwise

        Example:
            if not coordinator.is_within_budget(context, budget):
                result = await coordinator.compact_context(context, budget)
        """
        token_count = self.estimate_token_count(context)
        max_tokens = budget.get("max_tokens", 4096)
        return token_count <= max_tokens

    def add_strategy(
        self,
        strategy: ICompactionStrategy,
    ) -> None:
        """Add a compaction strategy.

        Args:
            strategy: Compaction strategy to add

        Example:
            strategy = SummarizationCompactionStrategy()
            coordinator.add_strategy(strategy)
        """
        self._strategies.append(strategy)

    def remove_strategy(
        self,
        strategy: ICompactionStrategy,
    ) -> None:
        """Remove a compaction strategy.

        Args:
            strategy: Compaction strategy to remove
        """
        if strategy in self._strategies:
            self._strategies.remove(strategy)

    def get_compaction_history(self) -> List[Dict[str, Any]]:
        """Get history of compaction operations.

        Returns:
            List of compaction records

        Example:
            history = coordinator.get_compaction_history()
            for record in history:
                print(f"{record['strategy']}: {record['tokens_saved']} tokens saved")
        """
        return self._compaction_history.copy()

    def clear_compaction_history(self) -> None:
        """Clear compaction history."""
        self._compaction_history.clear()

    def _record_compaction(
        self,
        result: CompactionResult,
    ) -> None:
        """Record compaction operation in history.

        Args:
            result: Compaction result to record
        """
        self._compaction_history.append(
            {
                "strategy": result.strategy_used,
                "tokens_saved": result.tokens_saved,
                "messages_removed": result.messages_removed,
                "metadata": result.metadata,
            }
        )


class ContextCompactionError(Exception):
    """Exception raised when context compaction fails."""

    pass


# Built-in compaction strategies

class BaseCompactionStrategy(ICompactionStrategy):
    """Base class for compaction strategies.

    Provides default implementation for ICompactionStrategy protocol
    that subclasses can override.

    Attributes:
        _name: Strategy name
    """

    def __init__(self, name: str):
        """Initialize the compaction strategy.

        Args:
            name: Strategy name
        """
        self._name = name

    async def can_apply(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> bool:
        """Check if strategy can be applied.

        Default implementation returns True if context exceeds budget.
        Subclasses can override for more specific criteria.

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            True if strategy can be applied, False otherwise
        """
        token_count = context.get("token_count", 0)
        max_tokens = budget.get("max_tokens", 4096)
        return token_count > max_tokens

    async def compact(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> CompactionResult:
        """Apply compaction strategy.

        Subclasses must implement this method.

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            CompactionResult with compacted context
        """
        raise NotImplementedError("Subclasses must implement compact()")

    def estimated_savings(self) -> int:
        """Estimated token savings for this strategy.

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            Estimated number of tokens that would be saved
        """
        return 1000  # Default estimate


class TruncationCompactionStrategy(BaseCompactionStrategy):
    """Compaction strategy that removes oldest messages.

    This strategy removes the oldest messages from the conversation
    until the context is within the token budget.
    """

    def __init__(self, reserve_messages: int = 10):
        """Initialize the truncation strategy.

        Args:
            reserve_messages: Minimum number of recent messages to keep
        """
        super().__init__("truncation")
        self._reserve_messages = reserve_messages

    async def compact(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> CompactionResult:
        """Apply truncation compaction.

        Removes oldest messages until within budget, reserving the
        most recent messages.

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            CompactionResult with truncated context
        """
        messages = context.get("messages", [])
        original_count = len(messages)
        original_tokens = context.get("token_count", 0)

        # Remove oldest messages while keeping reserved ones
        if len(messages) > self._reserve_messages:
            compacted_messages = messages[-self._reserve_messages :]
        else:
            compacted_messages = messages.copy()

        # Estimate new token count
        # Rough estimate: assume uniform message size
        if original_count > 0:
            new_tokens = int(original_tokens * (len(compacted_messages) / original_count))
        else:
            new_tokens = 0

        compacted_context: CompactionContext = {
            **context,
            "messages": compacted_messages,
            "token_count": new_tokens,
        }

        return CompactionResult(
            compacted_context=compacted_context,
            tokens_saved=original_tokens - new_tokens,
            messages_removed=original_count - len(compacted_messages),
            strategy_used=self._name,
            metadata={
                "reserve_messages": self._reserve_messages,
                "original_tokens": original_tokens,
                "new_tokens": new_tokens,
            },
        )

    def estimated_savings(self) -> int:
        """Estimated token savings.

        Returns estimated savings based on reserve_messages setting.
        """
        return 2000  # Estimate for removing older messages


__all__ = [
    "ContextCoordinator",
    "ContextCompactionError",
    "BaseCompactionStrategy",
    "TruncationCompactionStrategy",
]
