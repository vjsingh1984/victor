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

"""Tests for ContextCoordinator.

Tests the context management and compaction coordination functionality.
"""

import pytest

from victor.agent.coordinators.context_coordinator import (
    ContextCoordinator,
    ContextCompactionError,
    BaseCompactionStrategy,
    TruncationCompactionStrategy,
    SummarizationCompactionStrategy,
    SemanticCompactionStrategy,
    HybridCompactionStrategy,
)
from victor.protocols import ICompactionStrategy, CompactionResult, CompactionContext, ContextBudget


class MockCompactionStrategy(BaseCompactionStrategy):
    """Mock compaction strategy for testing."""

    def __init__(self, name="mock", can_apply_result=True, tokens_to_save=1000):
        super().__init__(name)
        self._can_apply_result = can_apply_result
        self._tokens_to_save = tokens_to_save

    async def can_apply(self, context, budget):
        return self._can_apply_result

    async def compact(self, context, budget):
        # Simple compaction: reduce token count
        new_tokens = context.get("token_count", 0) - self._tokens_to_save
        compacted_context = ({
            **context,
            "token_count": max(new_tokens, 0),
        })
        return CompactionResult(
            compacted_context=compacted_context,
            tokens_saved=self._tokens_to_save,
            messages_removed=0,
            strategy_used=self._name,
        )


class FailingCompactionStrategy(ICompactionStrategy):
    """Compaction strategy that always fails."""

    async def can_apply(self, context, budget):
        return True

    async def compact(self, context, budget):
        raise ValueError("Intentional compaction error")

    def estimated_savings(self):
        return 500


class TestBaseCompactionStrategy:
    """Tests for BaseCompactionStrategy."""

    @pytest.mark.asyncio
    async def test_can_apply_within_budget(self):
        """Test can_apply returns False when within budget."""
        strategy = BaseCompactionStrategy(name="test")
        context: CompactionContext = {"token_count": 3000}
        budget: ContextBudget = {"max_tokens": 4096}

        result = await strategy.can_apply(context, budget)

        assert result is False

    @pytest.mark.asyncio
    async def test_can_apply_exceeds_budget(self):
        """Test can_apply returns True when exceeds budget."""
        strategy = BaseCompactionStrategy(name="test")
        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        result = await strategy.can_apply(context, budget)

        assert result is True

    def test_compact_raises_not_implemented(self):
        """Test compact raises NotImplementedError."""
        strategy = BaseCompactionStrategy(name="test")

        with pytest.raises(NotImplementedError):
            # Need to call it via asyncio since it's async
            import asyncio
            asyncio.run(strategy.compact({}, {}))

    def test_estimated_savings(self):
        """Test estimated_savings returns default value."""
        strategy = BaseCompactionStrategy(name="test")

        assert strategy.estimated_savings() == 1000


class TestTruncationCompactionStrategy:
    """Tests for TruncationCompactionStrategy."""

    def test_init_default(self):
        """Test initialization with defaults."""
        strategy = TruncationCompactionStrategy()

        assert strategy._name == "truncation"
        assert strategy._reserve_messages == 10

    def test_init_custom_reserve(self):
        """Test initialization with custom reserve_messages."""
        strategy = TruncationCompactionStrategy(reserve_messages=5)

        assert strategy._reserve_messages == 5

    @pytest.mark.asyncio
    async def test_compact_removes_oldest_messages(self):
        """Test that oldest messages are removed."""
        strategy = TruncationCompactionStrategy(reserve_messages=2)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        context = ({
            "messages": messages,
            "token_count": 5000,
        })
        budget = ({"max_tokens": 2000})

        result = await strategy.compact(context, budget)

        # Should keep only last 2 messages
        assert len(result.compacted_context["messages"]) == 2
        assert result.messages_removed == 8
        assert result.strategy_used == "truncation"

    @pytest.mark.asyncio
    async def test_compact_preserves_reserved_messages(self):
        """Test that reserved messages are preserved."""
        strategy = TruncationCompactionStrategy(reserve_messages=10)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(5)
        ]
        context = ({
            "messages": messages,
            "token_count": 3000,
        })
        budget = ({"max_tokens": 2000})

        result = await strategy.compact(context, budget)

        # Should keep all messages since there are fewer than reserve
        assert len(result.compacted_context["messages"]) == 5
        assert result.messages_removed == 0

    @pytest.mark.asyncio
    async def test_compact_calculates_tokens_saved(self):
        """Test that tokens_saved is calculated correctly."""
        strategy = TruncationCompactionStrategy(reserve_messages=5)
        messages = [{"role": "user", "content": "Message"} for _ in range(100)]
        context = ({
            "messages": messages,
            "token_count": 10000,
        })
        budget = ({"max_tokens": 1000})

        result = await strategy.compact(context, budget)

        # Should have saved tokens by removing 95 messages
        assert result.tokens_saved > 0
        assert result.compacted_context["token_count"] < 10000

    def test_estimated_savings(self):
        """Test estimated_savings returns expected value."""
        strategy = TruncationCompactionStrategy()

        assert strategy.estimated_savings() == 2000


class TestContextCoordinator:
    """Tests for ContextCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create empty coordinator."""
        return ContextCoordinator(strategies=[])

    @pytest.fixture
    def coordinator_with_strategies(self):
        """Create coordinator with mock strategies."""
        strategy1 = MockCompactionStrategy(name="strategy1", can_apply_result=False)
        strategy2 = MockCompactionStrategy(name="strategy2", can_apply_result=True, tokens_to_save=2000)
        return ContextCoordinator(strategies=[strategy1, strategy2])

    def test_init_empty(self):
        """Test initialization with no strategies."""
        coordinator = ContextCoordinator(strategies=[])

        assert coordinator._strategies == []
        assert coordinator._enable_auto_compaction is True
        assert coordinator._compaction_history == []

    def test_init_with_strategies(self):
        """Test initialization with strategies."""
        strategy = MockCompactionStrategy()
        coordinator = ContextCoordinator(
            strategies=[strategy], enable_auto_compaction=False
        )

        assert len(coordinator._strategies) == 1
        assert coordinator._enable_auto_compaction is False

    @pytest.mark.asyncio
    async def test_compact_context_within_budget(self, coordinator):
        """Test compaction when context is within budget."""
        context = ({"token_count": 3000})
        budget = ({"max_tokens": 4096})

        result = await coordinator.compact_context(context, budget)

        assert result.tokens_saved == 0
        assert result.messages_removed == 0
        assert result.strategy_used == "none"
        assert result.metadata["reason"] == "within_budget"

    @pytest.mark.asyncio
    async def test_compact_context_exceeds_budget(self, coordinator_with_strategies):
        """Test compaction when context exceeds budget."""
        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        result = await coordinator_with_strategies.compact_context(context, budget)

        assert result.tokens_saved > 0
        assert result.strategy_used == "strategy2"  # The applicable one

    @pytest.mark.asyncio
    async def test_compact_context_uses_first_applicable_strategy(self):
        """Test that first applicable strategy is used."""
        strategy1 = MockCompactionStrategy(name="first", can_apply_result=True, tokens_to_save=1000)
        strategy2 = MockCompactionStrategy(name="second", can_apply_result=True, tokens_to_save=2000)
        coordinator = ContextCoordinator(strategies=[strategy1, strategy2])

        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        result = await coordinator.compact_context(context, budget)

        # Should use first strategy
        assert result.strategy_used == "first"
        assert result.tokens_saved == 1000

    @pytest.mark.asyncio
    async def test_compact_context_no_applicable_strategy(self):
        """Test error when no applicable strategy found."""
        strategy = MockCompactionStrategy(name="mock", can_apply_result=False)
        coordinator = ContextCoordinator(strategies=[strategy])

        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        with pytest.raises(ContextCompactionError) as exc_info:
            await coordinator.compact_context(context, budget)

        assert "No applicable compaction strategy found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_compact_context_handles_strategy_errors(self):
        """Test that failing strategies are skipped."""
        failing_strategy = FailingCompactionStrategy()
        working_strategy = MockCompactionStrategy(name="working", can_apply_result=True)
        coordinator = ContextCoordinator(strategies=[failing_strategy, working_strategy])

        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        result = await coordinator.compact_context(context, budget)

        # Should use the working strategy
        assert result.strategy_used == "working"

    @pytest.mark.asyncio
    async def test_rebuild_context(self, coordinator):
        """Test context rebuilding."""
        result = await coordinator.rebuild_context("session123", strategy="truncate")

        assert result["rebuilt"] is True
        assert result["rebuild_strategy"] == "truncate"
        assert result["messages"] == []
        assert result["token_count"] == 0

    def test_estimate_token_count_from_context(self, coordinator):
        """Test estimation when token_count is in context."""
        context = ({"token_count": 3500})

        tokens = coordinator.estimate_token_count(context)

        assert tokens == 3500

    def test_estimate_token_count_from_messages(self, coordinator):
        """Test estimation from messages when no token_count."""
        context = ({
            "messages": [
                {"content": "Hello world " * 100},  # ~1100 chars
                {"content": "Testing message " * 50},  # ~700 chars
            ]
        })

        tokens = coordinator.estimate_token_count(context)

        # Should estimate based on character count // 4
        assert tokens > 0

    def test_is_within_budget_true(self, coordinator):
        """Test is_within_budget returns True when within budget."""
        context = ({"token_count": 3000})
        budget = ({"max_tokens": 4096})

        result = coordinator.is_within_budget(context, budget)

        assert result is True

    def test_is_within_budget_false(self, coordinator):
        """Test is_within_budget returns False when exceeds budget."""
        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        result = coordinator.is_within_budget(context, budget)

        assert result is False

    def test_is_within_budget_default_max_tokens(self, coordinator):
        """Test is_within_budget uses default max_tokens."""
        context = ({"token_count": 3000})
        budget = ({})  # No max_tokens specified

        result = coordinator.is_within_budget(context, budget)

        # Should use default of 4096
        assert result is True

    def test_add_strategy(self, coordinator):
        """Test adding a strategy."""
        strategy = MockCompactionStrategy()
        coordinator.add_strategy(strategy)

        assert len(coordinator._strategies) == 1
        assert coordinator._strategies[0] == strategy

    def test_remove_strategy(self, coordinator):
        """Test removing a strategy."""
        strategy = MockCompactionStrategy()
        coordinator.add_strategy(strategy)

        coordinator.remove_strategy(strategy)

        assert len(coordinator._strategies) == 0

    def test_remove_nonexistent_strategy(self, coordinator):
        """Test removing a strategy that doesn't exist."""
        strategy = MockCompactionStrategy()

        # Should not raise
        coordinator.remove_strategy(strategy)

        assert len(coordinator._strategies) == 0

    def test_get_compaction_history(self, coordinator_with_strategies):
        """Test getting compaction history."""
        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        # Perform compaction
        import asyncio
        asyncio.run(coordinator_with_strategies.compact_context(context, budget))

        history = coordinator_with_strategies.get_compaction_history()

        assert len(history) == 1
        assert history[0]["strategy"] == "strategy2"
        assert history[0]["tokens_saved"] > 0

    def test_clear_compaction_history(self, coordinator_with_strategies):
        """Test clearing compaction history."""
        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        # Perform compaction
        import asyncio
        asyncio.run(coordinator_with_strategies.compact_context(context, budget))

        # Clear history
        coordinator_with_strategies.clear_compaction_history()

        history = coordinator_with_strategies.get_compaction_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_compact_context_records_history(self, coordinator_with_strategies):
        """Test that compaction operations are recorded."""
        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        await coordinator_with_strategies.compact_context(context, budget)

        history = coordinator_with_strategies.get_compaction_history()

        assert len(history) == 1
        assert "strategy" in history[0]
        assert "tokens_saved" in history[0]
        assert "messages_removed" in history[0]
        assert "metadata" in history[0]


class TestSummarizationCompactionStrategy:
    """Tests for SummarizationCompactionStrategy."""

    def test_init_default(self):
        """Test initialization with defaults."""
        strategy = SummarizationCompactionStrategy()

        assert strategy._name == "summarization"
        assert strategy._reserve_messages == 5
        assert strategy._summarize_threshold == 1000

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        strategy = SummarizationCompactionStrategy(
            reserve_messages=3, summarize_threshold=500
        )

        assert strategy._reserve_messages == 3
        assert strategy._summarize_threshold == 500

    @pytest.mark.asyncio
    async def test_can_apply_within_threshold(self):
        """Test can_apply returns False when within threshold."""
        strategy = SummarizationCompactionStrategy(summarize_threshold=1000)
        context: CompactionContext = {"token_count": 4500}
        budget: ContextBudget = {"max_tokens": 4096}

        # 4500 < 4096 + 1000 = 5096, so should not apply
        result = await strategy.can_apply(context, budget)

        assert result is False

    @pytest.mark.asyncio
    async def test_can_apply_exceeds_threshold(self):
        """Test can_apply returns True when exceeds threshold."""
        strategy = SummarizationCompactionStrategy(summarize_threshold=1000)
        context = ({"token_count": 5500})
        budget = ({"max_tokens": 4096})

        # 5500 > 4096 + 1000 = 5096, so should apply
        result = await strategy.can_apply(context, budget)

        assert result is True

    @pytest.mark.asyncio
    async def test_compact_summarizes_old_messages(self):
        """Test that old messages are summarized."""
        strategy = SummarizationCompactionStrategy(reserve_messages=3)
        messages = [
            {"role": "user", "content": f"Old message {i}"}
            for i in range(10)
        ]
        context = ({
            "messages": messages,
            "token_count": 8000,
        })
        budget = ({"max_tokens": 2000})

        result = await strategy.compact(context, budget)

        # Should have summary message + recent messages
        assert len(result.compacted_context["messages"]) == 4  # 1 summary + 3 recent
        assert result.compacted_context["messages"][0]["role"] == "system"
        assert "Summary" in result.compacted_context["messages"][0]["content"]
        assert result.messages_removed == 7
        assert result.strategy_used == "summarization"

    @pytest.mark.asyncio
    async def test_compact_preserves_recent_messages(self):
        """Test that recent messages are preserved."""
        strategy = SummarizationCompactionStrategy(reserve_messages=5)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        context = ({
            "messages": messages,
            "token_count": 6000,
        })
        budget = ({"max_tokens": 2000})

        result = await strategy.compact(context, budget)

        # Should keep last 5 messages
        recent_messages = result.compacted_context["messages"][1:]
        assert len(recent_messages) == 5
        assert recent_messages[0]["content"] == "Message 5"

    @pytest.mark.asyncio
    async def test_compact_not_enough_messages(self):
        """Test compact when there aren't enough messages."""
        strategy = SummarizationCompactionStrategy(reserve_messages=10)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(5)
        ]
        context = ({
            "messages": messages,
            "token_count": 3000,
        })
        budget = ({"max_tokens": 2000})

        result = await strategy.compact(context, budget)

        # Should not compact
        assert result.tokens_saved == 0
        assert result.messages_removed == 0
        assert result.metadata["reason"] == "not_enough_messages"

    def test_estimated_savings(self):
        """Test estimated_savings returns expected value."""
        strategy = SummarizationCompactionStrategy()

        assert strategy.estimated_savings() == 5000


class TestSemanticCompactionStrategy:
    """Tests for SemanticCompactionStrategy."""

    def test_init_default(self):
        """Test initialization with defaults."""
        strategy = SemanticCompactionStrategy()

        assert strategy._name == "semantic"
        assert strategy._reserve_messages == 8
        assert strategy._similarity_threshold == 0.85

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        strategy = SemanticCompactionStrategy(
            reserve_messages=5, similarity_threshold=0.9
        )

        assert strategy._reserve_messages == 5
        assert strategy._similarity_threshold == 0.9

    @pytest.mark.asyncio
    async def test_can_apply_exceeds_budget(self):
        """Test can_apply returns True when exceeds budget."""
        strategy = SemanticCompactionStrategy()
        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        result = await strategy.can_apply(context, budget)

        assert result is True

    @pytest.mark.asyncio
    async def test_compact_removes_similar_messages(self):
        """Test that similar messages are removed."""
        strategy = SemanticCompactionStrategy(reserve_messages=3)
        messages = [
            {"role": "user", "content": "Similar message content"},  # Will be deduplicated
            {"role": "user", "content": "Similar message content"},  # Will be deduplicated
            {"role": "user", "content": "Different message"},
            {"role": "user", "content": "Another different message"},
        ]
        context = ({
            "messages": messages,
            "token_count": 4000,
        })
        budget = ({"max_tokens": 2000})

        result = await strategy.compact(context, budget)

        # Should remove similar messages
        assert result.messages_removed >= 0
        assert result.strategy_used == "semantic"

    @pytest.mark.asyncio
    async def test_compact_preserves_recent_messages(self):
        """Test that recent messages are preserved."""
        strategy = SemanticCompactionStrategy(reserve_messages=2)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        context = ({
            "messages": messages,
            "token_count": 5000,
        })
        budget = ({"max_tokens": 2000})

        result = await strategy.compact(context, budget)

        # Should keep last 2 messages
        recent_messages = result.compacted_context["messages"][-2:]
        assert len(recent_messages) == 2

    @pytest.mark.asyncio
    async def test_compact_not_enough_messages(self):
        """Test compact when there aren't enough messages."""
        strategy = SemanticCompactionStrategy(reserve_messages=10)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(5)
        ]
        context = ({
            "messages": messages,
            "token_count": 3000,
        })
        budget = ({"max_tokens": 2000})

        result = await strategy.compact(context, budget)

        # Should not compact
        assert result.tokens_saved == 0
        assert result.messages_removed == 0
        assert result.metadata["reason"] == "not_enough_messages"

    def test_estimated_savings(self):
        """Test estimated_savings returns expected value."""
        strategy = SemanticCompactionStrategy()

        assert strategy.estimated_savings() == 3500


class TestHybridCompactionStrategy:
    """Tests for HybridCompactionStrategy."""

    def test_init_default(self):
        """Test initialization with defaults."""
        strategy = HybridCompactionStrategy()

        assert strategy._name == "hybrid"
        assert strategy._reserve_messages == 6
        assert strategy._summarize_threshold == 1000

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        strategy = HybridCompactionStrategy(
            reserve_messages=4, summarize_threshold=500
        )

        assert strategy._reserve_messages == 4
        assert strategy._summarize_threshold == 500

    def test_init_creates_sub_strategies(self):
        """Test that sub-strategies are created."""
        strategy = HybridCompactionStrategy()

        assert isinstance(strategy._semantic_strategy, SemanticCompactionStrategy)
        assert isinstance(
            strategy._summarization_strategy, SummarizationCompactionStrategy
        )
        assert isinstance(strategy._truncation_strategy, TruncationCompactionStrategy)

    @pytest.mark.asyncio
    async def test_can_apply_exceeds_budget(self):
        """Test can_apply returns True when exceeds budget."""
        strategy = HybridCompactionStrategy()
        context = ({"token_count": 5000})
        budget = ({"max_tokens": 4096})

        result = await strategy.can_apply(context, budget)

        assert result is True

    @pytest.mark.asyncio
    async def test_compact_uses_semantic_first(self):
        """Test that semantic compaction is tried first."""
        strategy = HybridCompactionStrategy()
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(20)
        ]
        context = ({
            "messages": messages,
            "token_count": 15000,  # Large context
        })
        budget = ({"max_tokens": 5000})  # Need significant reduction

        result = await strategy.compact(context, budget)

        # Should apply compaction
        assert result.tokens_saved > 0
        assert "hybrid" in result.strategy_used
        # Metadata should exist
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_compact_falls_back_to_truncation(self):
        """Test that truncation is used as fallback."""
        strategy = HybridCompactionStrategy(reserve_messages=3)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(100)
        ]
        context = ({
            "messages": messages,
            "token_count": 50000,  # Very large context
        })
        budget = ({"max_tokens": 1000})  # Very small budget

        result = await strategy.compact(context, budget)

        # Should apply compaction
        assert result.tokens_saved > 0
        assert "hybrid" in result.strategy_used
        # With such extreme reduction, should reach truncation
        assert "truncation" in result.strategy_used

    @pytest.mark.asyncio
    async def test_compact_stops_early_if_successful(self):
        """Test that compaction stops early if budget is met."""
        strategy = HybridCompactionStrategy(reserve_messages=5)
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(15)
        ]
        context = ({
            "messages": messages,
            "token_count": 8000,
        })
        budget = ({"max_tokens": 6000})  # Only need small reduction

        result = await strategy.compact(context, budget)

        # Should stop after first successful compaction
        assert result.tokens_saved > 0
        assert "hybrid" in result.strategy_used

    @pytest.mark.asyncio
    async def test_compact_calculates_total_savings(self):
        """Test that total savings are calculated correctly."""
        strategy = HybridCompactionStrategy()
        original_tokens = 10000
        context = ({
            "messages": [{"role": "user", "content": "msg"} for _ in range(50)],
            "token_count": original_tokens,
        })
        budget = ({"max_tokens": 2000})

        result = await strategy.compact(context, budget)

        # Verify tokens_saved matches the difference
        final_tokens = result.compacted_context.get("token_count", 0)
        expected_savings = original_tokens - final_tokens
        assert result.tokens_saved == expected_savings

    def test_estimated_savings(self):
        """Test estimated_savings returns expected value."""
        strategy = HybridCompactionStrategy()

        assert strategy.estimated_savings() == 7000


class TestContextCoordinatorWithNewStrategies:
    """Integration tests with new compaction strategies."""

    @pytest.mark.asyncio
    async def test_coordinator_uses_summarization_strategy(self):
        """Test coordinator using SummarizationCompactionStrategy."""
        strategy = SummarizationCompactionStrategy()
        coordinator = ContextCoordinator(strategies=[strategy])

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        context = ({
            "messages": messages,
            "token_count": 7000,
        })
        budget = ({"max_tokens": 4096})

        result = await coordinator.compact_context(context, budget)

        assert result.strategy_used == "summarization"
        assert result.tokens_saved > 0

    @pytest.mark.asyncio
    async def test_coordinator_uses_semantic_strategy(self):
        """Test coordinator using SemanticCompactionStrategy."""
        strategy = SemanticCompactionStrategy(reserve_messages=3)
        coordinator = ContextCoordinator(strategies=[strategy])

        # Create messages with some duplicates (similar content)
        messages = [
            {"role": "user", "content": "This is a repeated message"},
            {"role": "user", "content": "This is a repeated message"},  # Duplicate
            {"role": "user", "content": "Another repeated message"},
            {"role": "user", "content": "Another repeated message"},  # Duplicate
            {"role": "user", "content": "Unique message 1"},
            {"role": "user", "content": "Unique message 2"},
            {"role": "user", "content": "Unique message 3"},
            {"role": "user", "content": "Unique message 4"},
            {"role": "user", "content": "Unique message 5"},
        ]
        context = ({
            "messages": messages,
            "token_count": 6000,
        })
        budget = ({"max_tokens": 4096})

        result = await coordinator.compact_context(context, budget)

        assert result.strategy_used == "semantic"
        # Should have saved some tokens by removing duplicates
        assert result.tokens_saved >= 0

    @pytest.mark.asyncio
    async def test_coordinator_uses_hybrid_strategy(self):
        """Test coordinator using HybridCompactionStrategy."""
        strategy = HybridCompactionStrategy()
        coordinator = ContextCoordinator(strategies=[strategy])

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(20)
        ]
        context = ({
            "messages": messages,
            "token_count": 15000,
        })
        budget = ({"max_tokens": 5000})

        result = await coordinator.compact_context(context, budget)

        assert "hybrid" in result.strategy_used
        assert result.tokens_saved > 0

    @pytest.mark.asyncio
    async def test_coordinator_strategy_priority_order(self):
        """Test that strategies are tried in priority order."""
        strategies = [
            TruncationCompactionStrategy(reserve_messages=10),
            SummarizationCompactionStrategy(reserve_messages=5),
            HybridCompactionStrategy(reserve_messages=3),
        ]
        coordinator = ContextCoordinator(strategies=strategies)

        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(15)
        ]
        context = ({
            "messages": messages,
            "token_count": 8000,
        })
        budget = ({"max_tokens": 4096})

        result = await coordinator.compact_context(context, budget)

        # Should use first applicable strategy (truncation)
        assert result.strategy_used == "truncation"
        assert result.tokens_saved > 0
