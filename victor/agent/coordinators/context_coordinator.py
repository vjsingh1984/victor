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
                logger.warning(f"Compaction strategy {strategy.__class__.__name__} failed: {e}")
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
        return CompactionContext({
            "messages": [],
            "token_count": 0,
            "rebuilt": True,
            "rebuild_strategy": strategy,
        })

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
            return int(context["token_count"])

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
        max_tokens = int(budget.get("max_tokens", 4096))
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
        token_count = int(context.get("token_count", 0))
        max_tokens = int(budget.get("max_tokens", 4096))
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

        compacted_context: CompactionContext = CompactionContext({
            **context,
            "messages": compacted_messages,
            "token_count": new_tokens,
        })

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


class SummarizationCompactionStrategy(BaseCompactionStrategy):
    """Compaction strategy that summarizes old messages using LLM.

    This strategy uses an LLM to summarize older messages, preserving
    important information while reducing token count.
    """

    def __init__(
        self,
        reserve_messages: int = 5,
        summarize_threshold: int = 1000,
    ):
        """Initialize the summarization strategy.

        Args:
            reserve_messages: Minimum number of recent messages to keep
            summarize_threshold: Minimum token count to trigger summarization
        """
        super().__init__("summarization")
        self._reserve_messages = reserve_messages
        self._summarize_threshold = summarize_threshold

    async def can_apply(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> bool:
        """Check if summarization can be applied.

        Only applies if context significantly exceeds budget.

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            True if summarization should be applied
        """
        token_count: int = context.get("token_count", 0)
        max_tokens: int = budget.get("max_tokens", 4096)

        # Only apply if significantly over budget (by at least threshold)
        return token_count > max_tokens + self._summarize_threshold

    async def compact(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> CompactionResult:
        """Apply summarization compaction.

        Summarizes older messages and keeps recent messages intact.
        For testing purposes, creates a simple summary placeholder.

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            CompactionResult with summarized context
        """
        messages = context.get("messages", [])
        original_count = len(messages)
        original_tokens = context.get("token_count", 0)

        if len(messages) <= self._reserve_messages:
            # Not enough messages to summarize
            return CompactionResult(
                compacted_context=context,
                tokens_saved=0,
                messages_removed=0,
                strategy_used=self._name,
                metadata={"reason": "not_enough_messages"},
            )

        # Split into old and recent messages
        old_messages = messages[: -self._reserve_messages]
        recent_messages = messages[-self._reserve_messages :]

        # Create a summary placeholder
        # In production, this would use an LLM to summarize
        summary_message = {
            "role": "system",
            "content": f"[Summary of {len(old_messages)} earlier messages compacted]",
        }

        # Estimate tokens saved (assume 50% reduction for old messages)
        tokens_per_message = original_tokens / original_count if original_count > 0 else 100
        old_tokens = int(len(old_messages) * tokens_per_message)
        summary_tokens = 100  # Rough estimate for summary
        tokens_saved = old_tokens - summary_tokens

        # Build compacted context
        compacted_messages = [summary_message] + recent_messages
        new_tokens = original_tokens - tokens_saved

        compacted_context: CompactionContext = CompactionContext({
            **context,
            "messages": compacted_messages,
            "token_count": new_tokens,
        })

        return CompactionResult(
            compacted_context=compacted_context,
            tokens_saved=tokens_saved,
            messages_removed=len(old_messages),
            strategy_used=self._name,
            metadata={
                "reserve_messages": self._reserve_messages,
                "summarized_count": len(old_messages),
                "original_tokens": original_tokens,
                "new_tokens": new_tokens,
            },
        )

    def estimated_savings(self) -> int:
        """Estimated token savings.

        Returns higher estimate since summarization is more effective.
        """
        return 5000  # Estimate for summarization


class SemanticCompactionStrategy(BaseCompactionStrategy):
    """Compaction strategy that uses semantic similarity to remove redundant messages.

    This strategy removes messages that are semantically similar to each other,
    preserving diverse content while reducing redundancy.
    """

    def __init__(
        self,
        reserve_messages: int = 8,
        similarity_threshold: float = 0.85,
    ):
        """Initialize the semantic compaction strategy.

        Args:
            reserve_messages: Minimum number of recent messages to keep
            similarity_threshold: Similarity threshold for considering messages redundant
        """
        super().__init__("semantic")
        self._reserve_messages = reserve_messages
        self._similarity_threshold = similarity_threshold

    async def can_apply(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> bool:
        """Check if semantic compaction can be applied.

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            True if semantic compaction should be applied
        """
        token_count: int = context.get("token_count", 0)
        max_tokens: int = budget.get("max_tokens", 4096)

        # Apply if context exceeds budget
        return token_count > max_tokens

    async def compact(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> CompactionResult:
        """Apply semantic compaction.

        Removes semantically similar messages while preserving diverse content.
        For testing purposes, uses simple length-based deduplication.

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            CompactionResult with semantically compacted context
        """
        messages = context.get("messages", [])
        original_count = len(messages)
        original_tokens = context.get("token_count", 0)

        if len(messages) <= self._reserve_messages:
            return CompactionResult(
                compacted_context=context,
                tokens_saved=0,
                messages_removed=0,
                strategy_used=self._name,
                metadata={"reason": "not_enough_messages"},
            )

        # Simple semantic deduplication based on content length similarity
        # In production, this would use embeddings
        seen_signatures = set()
        compacted_messages = []

        # Always keep recent messages
        recent_messages = messages[-self._reserve_messages :]
        compacted_messages.extend(recent_messages)

        # Process older messages, removing similar ones
        old_messages = messages[: -self._reserve_messages]
        for msg in old_messages:
            # Create a simple signature based on content length and first few chars
            content = msg.get("content", "")
            signature = (len(content), content[:50] if content else "")

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                compacted_messages.insert(0, msg)  # Insert at beginning

        messages_removed = original_count - len(compacted_messages)

        # Estimate new token count
        if original_count > 0:
            new_tokens = int(original_tokens * (len(compacted_messages) / original_count))
        else:
            new_tokens = 0

        tokens_saved = original_tokens - new_tokens

        compacted_context: CompactionContext = CompactionContext({
            **context,
            "messages": compacted_messages,
            "token_count": new_tokens,
        })

        return CompactionResult(
            compacted_context=compacted_context,
            tokens_saved=tokens_saved,
            messages_removed=messages_removed,
            strategy_used=self._name,
            metadata={
                "reserve_messages": self._reserve_messages,
                "similarity_threshold": self._similarity_threshold,
                "original_tokens": original_tokens,
                "new_tokens": new_tokens,
            },
        )

    def estimated_savings(self) -> int:
        """Estimated token savings.

        Returns moderate estimate for semantic deduplication.
        """
        return 3500  # Estimate for semantic compaction


class HybridCompactionStrategy(BaseCompactionStrategy):
    """Compaction strategy that combines multiple approaches.

    This strategy first applies semantic compaction to remove redundant messages,
    then applies summarization if still over budget, and finally truncation as
    a last resort.
    """

    def __init__(
        self,
        reserve_messages: int = 6,
        summarize_threshold: int = 1000,
    ):
        """Initialize the hybrid compaction strategy.

        Args:
            reserve_messages: Minimum number of recent messages to keep
            summarize_threshold: Minimum token count to trigger summarization
        """
        super().__init__("hybrid")
        self._reserve_messages = reserve_messages
        self._summarize_threshold = summarize_threshold
        # Initialize sub-strategies
        self._semantic_strategy = SemanticCompactionStrategy(reserve_messages=reserve_messages + 2)
        self._summarization_strategy = SummarizationCompactionStrategy(
            reserve_messages=reserve_messages,
            summarize_threshold=summarize_threshold,
        )
        self._truncation_strategy = TruncationCompactionStrategy(reserve_messages=reserve_messages)

    async def can_apply(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> bool:
        """Check if hybrid compaction can be applied.

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            True if hybrid compaction should be applied
        """
        token_count: int = context.get("token_count", 0)
        max_tokens: int = budget.get("max_tokens", 4096)

        # Apply if context exceeds budget
        return token_count > max_tokens

    async def compact(
        self,
        context: CompactionContext,
        budget: ContextBudget,
    ) -> CompactionResult:
        """Apply hybrid compaction.

        Tries multiple strategies in sequence:
        1. Semantic compaction (remove redundant messages)
        2. Summarization (summarize older messages)
        3. Truncation (remove oldest messages)

        Args:
            context: Current conversation context
            budget: Context budget

        Returns:
            CompactionResult with compacted context
        """
        original_tokens = context.get("token_count", 0)
        max_tokens = budget.get("max_tokens", 4096)

        # Step 1: Try semantic compaction
        context_after_semantic = context
        result = await self._semantic_strategy.compact(context, budget)

        compacted_dict = result.compacted_context if hasattr(result.compacted_context, 'get') else {"token_count": 0}
        if compacted_dict.get("token_count", 0) <= max_tokens:
            result.strategy_used = f"{self._name}_semantic"
            return result

        context_after_semantic = result.compacted_context

        # Step 2: Try summarization
        result = await self._summarization_strategy.compact(context_after_semantic, budget)

        compacted_dict = result.compacted_context if hasattr(result.compacted_context, 'get') else {"token_count": 0}
        if compacted_dict.get("token_count", 0) <= max_tokens:
            result.strategy_used = f"{self._name}_summarization"
            return result

        context_after_summarization = result.compacted_context

        # Step 3: Fall back to truncation
        result = await self._truncation_strategy.compact(context_after_summarization, budget)

        result.strategy_used = f"{self._name}_truncation"

        # Calculate total tokens saved
        final_tokens = result.compacted_context if hasattr(result.compacted_context, 'get') else {"token_count": 0}
        final_token_count = final_tokens.get("token_count", 0)
        total_tokens_saved = original_tokens - final_token_count

        # Update metadata to reflect hybrid approach
        existing_metadata = result.metadata if isinstance(result.metadata, dict) else {}
        result.metadata = {
            **existing_metadata,
            "hybrid_steps": ["semantic", "summarization", "truncation"],
            "original_tokens": original_tokens,
            "final_tokens": final_token_count,
        }

        # Override tokens_saved with total
        result = CompactionResult(
            compacted_context=result.compacted_context,
            tokens_saved=total_tokens_saved,
            messages_removed=result.messages_removed,
            strategy_used=result.strategy_used,
            metadata=result.metadata,
        )

        return result

    def estimated_savings(self) -> int:
        """Estimated token savings.

        Returns highest estimate since hybrid combines multiple strategies.
        """
        return 7000  # Estimate for hybrid approach


__all__ = [
    "ContextCoordinator",
    "ContextCompactionError",
    "BaseCompactionStrategy",
    "TruncationCompactionStrategy",
    "SummarizationCompactionStrategy",
    "SemanticCompactionStrategy",
    "HybridCompactionStrategy",
]
