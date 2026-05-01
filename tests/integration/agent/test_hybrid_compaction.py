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

"""Integration tests for hybrid compaction system."""

import pytest
from unittest.mock import AsyncMock, Mock

from victor.agent.compaction_router import CompactionRouter, CompactionType
from victor.agent.compaction_rule_based import RuleBasedCompactionSummarizer
from victor.agent.compaction_hybrid import HybridCompactionSummarizer
from victor.agent.llm_compaction_summarizer import LLMCompactionSummarizer
from victor.agent.conversation.controller import ConversationController, ConversationConfig
from victor.config.compaction_strategy_settings import (
    CompactionStrategySettings,
    CompactionFeatureFlags,
)
from victor.providers.base import Message


@pytest.fixture
def settings():
    """Create compaction strategy settings for testing."""
    return CompactionStrategySettings(
        llm_min_complexity=0.7,
        llm_min_tokens=5000,
        llm_min_messages=20,
        hybrid_llm_enhancement=True,
        hybrid_llm_sections=["pending_work", "current_work"],
        llm_timeout_seconds=5.0,
        store_compaction_history=True,
    )


@pytest.fixture
def feature_flags():
    """Create feature flags for testing."""
    return CompactionFeatureFlags(
        enable_rule_based=True,
        enable_llm_based=True,
        enable_hybrid=True,
        enable_json_storage=True,
        enable_compaction_analytics=True,
    )


@pytest.fixture
def mock_provider():
    """Create mock provider for LLM summarizer."""
    provider = Mock()
    # Return a longer summary to meet test expectations (> 100 chars)
    provider.chat = Mock(
        return_value=(
            "Earlier conversation covered authentication bug fixes, "
            "file structure analysis, and pending unit test implementation. "
            "User requested fixing login issues in auth service."
        )
    )
    return provider


@pytest.fixture
def rule_summarizer(settings):
    """Create rule-based summarizer."""
    return RuleBasedCompactionSummarizer(settings)


@pytest.fixture
def llm_summarizer(settings, mock_provider):
    """Create LLM-based summarizer."""
    return LLMCompactionSummarizer(
        provider=mock_provider,
        max_summary_tokens=300,
        timeout_seconds=5.0,
    )


@pytest.fixture
def hybrid_summarizer(settings, rule_summarizer, llm_summarizer):
    """Create hybrid summarizer."""
    return HybridCompactionSummarizer(
        config=settings,
        rule_summarizer=rule_summarizer,
        llm_summarizer=llm_summarizer,
    )


@pytest.fixture
def router(settings, feature_flags, rule_summarizer, llm_summarizer, hybrid_summarizer):
    """Create compaction router with all summarizers."""
    return CompactionRouter(
        settings=settings,
        feature_flags=feature_flags,
        rule_summarizer=rule_summarizer,
        llm_summarizer=llm_summarizer,
        hybrid_summarizer=hybrid_summarizer,
    )


@pytest.fixture
def sample_conversation():
    """Create a realistic sample conversation."""
    return [
        Message(role="user", content="I need help fixing a bug in the authentication system"),
        Message(
            role="assistant",
            content="I'll help you fix the authentication bug. Let me start by examining the authentication module.",
            tool_calls=[{"name": "read_file", "id": "call_1"}],
        ),
        Message(
            role="tool",
            tool_call_id="call_1",
            tool_name="read_file",
            content="""File: src/auth/login.py

def login(username, password):
    # TODO: Add input validation
    if not username or not password:
        return None
    return authenticate(username, password)
""",
        ),
        Message(
            role="assistant",
            content="I found the issue. The login function is missing proper input validation. Let me add validation for empty strings and whitespace.",
            tool_calls=[{"name": "write_file", "id": "call_2"}],
        ),
        Message(
            role="tool",
            tool_call_id="call_2",
            tool_name="write_file",
            content="Successfully updated src/auth/login.py with input validation",
        ),
        Message(
            role="assistant",
            content="I've added input validation to the login function. The changes include:\n"
            "1. Check for None values\n"
            "2. Strip whitespace from username and password\n"
            "3. Validate that both fields are non-empty after stripping\n\n"
            "TODO: We should add unit tests for this function later.",
        ),
        Message(
            role="user",
            content="Great! Now can you also add error handling for invalid credentials?",
        ),
    ]


@pytest.fixture
def large_conversation():
    """Create a large conversation for testing compaction."""
    messages = []
    for i in range(50):
        messages.append(
            Message(role="user", content=f"Task {i}: Fix bug in src/module{i}/file{i}.py")
        )
        messages.append(
            Message(
                role="assistant",
                content=f"I'll fix the bug in file{i}.py",
                tool_calls=[{"name": "read_file", "id": f"call_{i}"}],
            )
        )
        messages.append(
            Message(
                role="tool",
                tool_call_id=f"call_{i}",
                tool_name="read_file",
                content=f"Content of file{i}.py...",
            )
        )
        messages.append(
            Message(
                role="assistant",
                content=f"Fixed the bug. Next, we need to add tests for module{i}.",
                tool_calls=[{"name": "write_file", "id": f"call_write_{i}"}],
            )
        )
        messages.append(
            Message(
                role="tool",
                tool_call_id=f"call_write_{i}",
                tool_name="write_file",
                content=f"Successfully updated file{i}.py",
            )
        )
    return messages


class TestHybridCompactionIntegration:
    """Integration tests for hybrid compaction system."""

    @pytest.mark.asyncio
    async def test_full_hybrid_compaction_flow(self, router, sample_conversation):
        """Test complete hybrid compaction flow with all components."""
        result = await router.compact(
            messages=sample_conversation,
            current_query="add error handling",
            session_id="test-session-123",
        )

        # Verify result
        assert result.success is True
        assert result.summary
        assert result.removed_count == len(sample_conversation)
        # tokens_saved can be 0 for small conversations where summary is not much shorter
        assert result.tokens_saved >= 0
        assert result.duration_ms >= 0
        assert result.session_id == "test-session-123"

        # Verify summary format (JSON for rule-based, XML or "Compacted context" for LLM-based)
        assert (
            "{" in result.summary  # JSON format (rule-based)
            or "<summary>" in result.summary  # XML format
            or "Compacted context" in result.summary  # LLM-based format
        )

    @pytest.mark.asyncio
    async def test_strategy_selection_based_on_complexity(
        self, router, sample_conversation, large_conversation
    ):
        """Test that router selects appropriate strategy based on complexity."""
        # Small conversation should use rule-based
        small_result = await router.compact(sample_conversation)
        assert small_result.strategy_used in [CompactionType.RULE_BASED, CompactionType.HYBRID]

        # Large conversation should use LLM-based
        large_result = await router.compact(large_conversation)
        assert large_result.strategy_used == CompactionType.LLM_BASED

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_llm_failure(
        self, settings, feature_flags, rule_summarizer
    ):
        """Test graceful degradation when LLM fails."""
        # Create LLM summarizer that fails
        mock_provider = Mock()
        mock_provider.chat = Mock(side_effect=Exception("LLM service unavailable"))
        llm_summarizer = LLMCompactionSummarizer(
            provider=mock_provider,
            timeout_seconds=1.0,
        )

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
            hybrid_summarizer=None,
        )

        messages = [Message(role="user", content="Test message")]
        result = await router.compact(messages)

        # Should fall back to rule-based
        assert result.success is True
        assert result.summary

    @pytest.mark.asyncio
    async def test_concurrent_compaction_requests(self, router, sample_conversation):
        """Test handling concurrent compaction requests."""
        import asyncio

        # Create multiple concurrent compaction requests
        tasks = [router.compact(sample_conversation, session_id=f"session-{i}") for i in range(5)]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        assert all(r.success for r in results)
        assert all(r.summary for r in results)


class TestConversationControllerIntegration:
    """Test integration with ConversationController."""

    def test_controller_with_hybrid_compaction(self, settings):
        """Test ConversationController with hybrid compaction configuration."""
        config = ConversationConfig(
            compaction_strategy="hybrid",
            min_messages_to_keep=settings.rule_preserve_recent,
        )

        controller = ConversationController(config=config)

        # Add messages
        for i in range(20):
            controller.add_user_message(f"Message {i}")
            controller.add_assistant_message(f"Response {i}")

        # Check compaction works (use force=True to bypass threshold checks for testing)
        initial_count = controller.message_count
        removed = controller.smart_compact_history(target_messages=10, force=True)

        assert removed > 0
        assert controller.message_count < initial_count

    def test_controller_with_rule_based_compaction(self, settings):
        """Test ConversationController with rule-based compaction."""
        config = ConversationConfig(
            compaction_strategy="simple",
            min_messages_to_keep=settings.rule_preserve_recent,
        )

        controller = ConversationController(config=config)

        # Add messages
        for i in range(20):
            controller.add_user_message(f"Message {i}")
            controller.add_assistant_message(f"Response {i}")

        # Check compaction works
        initial_count = controller.message_count
        removed = controller.compact_history(keep_recent=10)

        assert removed > 0
        assert controller.message_count < initial_count


class TestSettingsDrivenBehavior:
    """Test that settings drive compaction behavior correctly."""

    @pytest.mark.asyncio
    async def test_low_complexity_threshold_uses_rules(self, feature_flags, rule_summarizer):
        """Test that low complexity threshold favors rule-based."""
        settings = CompactionStrategySettings(
            llm_min_complexity=0.9,  # Very high threshold
        )

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=None,
            hybrid_summarizer=None,
        )

        messages = [
            Message(role="user", content="Simple message"),
            Message(role="assistant", content="Simple response"),
        ]

        result = await router.compact(messages)

        # Should use rule-based
        assert result.strategy_used == CompactionType.RULE_BASED

    @pytest.mark.asyncio
    async def test_high_complexity_threshold_uses_llm(
        self, settings, feature_flags, rule_summarizer, mock_provider
    ):
        """Test that high complexity threshold favors LLM-based."""
        settings.llm_min_complexity = 0.3  # Very low threshold

        llm_summarizer = LLMCompactionSummarizer(
            provider=mock_provider,
            timeout_seconds=5.0,
        )

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
            hybrid_summarizer=None,
        )

        messages = [
            Message(role="user", content="Complex message about fixing bugs in multiple files"),
            Message(role="assistant", content="Complex response"),
        ]

        result = await router.compact(messages)

        # Should use LLM-based
        assert result.strategy_used == CompactionType.LLM_BASED

    @pytest.mark.asyncio
    async def test_feature_flags_disable_strategies(self, settings, rule_summarizer):
        """Test that feature flags control strategy availability."""
        feature_flags = CompactionFeatureFlags(
            enable_rule_based=True,
            enable_llm_based=False,  # Disable LLM
            enable_hybrid=False,  # Disable hybrid
        )

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=None,
            hybrid_summarizer=None,
        )

        messages = [Message(role="user", content="Test")]
        result = await router.compact(messages)

        # Should only use rule-based
        assert result.strategy_used == CompactionType.RULE_BASED


class TestPerformanceAndQuality:
    """Test performance and quality metrics."""

    @pytest.mark.asyncio
    async def test_rule_based_performance(self, router, sample_conversation):
        """Test that rule-based compaction is fast."""
        import time

        start = time.time()
        result = await router.compact(sample_conversation)
        duration = (time.time() - start) * 1000  # Convert to ms

        # Rule-based should be fast
        if result.strategy_used == CompactionType.RULE_BASED:
            assert duration < 500, f"Rule-based too slow: {duration}ms"

    @pytest.mark.asyncio
    async def test_llm_based_quality(self, router, large_conversation):
        """Test that LLM-based compaction produces high-quality summaries."""
        result = await router.compact(large_conversation)

        if result.strategy_used == CompactionType.LLM_BASED:
            # LLM summary should be substantial
            assert len(result.summary) > 100
            # Should save significant tokens
            assert result.tokens_saved > 1000

    @pytest.mark.asyncio
    async def test_hybrid_balance(self, router, sample_conversation):
        """Test that hybrid balances speed and quality."""
        result = await router.compact(sample_conversation)

        if result.strategy_used == CompactionType.HYBRID:
            # Hybrid should be reasonably fast
            assert result.duration_ms < 2000
            # But still produce good summary
            assert len(result.summary) > 50


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_debugging_session_compaction(self, router):
        """Test compaction during a debugging session."""
        messages = []
        debugging_steps = [
            "Read the error logs",
            "Identify the root cause",
            "Fix the bug in src/auth.py",
            "Add error handling",
            "Test the fix",
            "TODO: Add regression tests",
        ]

        for step in debugging_steps:
            messages.append(Message(role="user", content=step))
            messages.append(
                Message(
                    role="assistant",
                    content=f"I'll {step.lower()}",
                    tool_calls=[{"name": "read_file", "id": f"call_{len(messages)}"}],
                )
            )

        result = await router.compact(messages, current_query="continue debugging")

        assert result.success
        # Should capture pending work
        assert "TODO" in result.summary or "pending" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_multi_file_refactoring_session(self, router):
        """Test compaction during multi-file refactoring."""
        messages = []
        files = ["src/auth.py", "src/user.py", "src/models.py", "tests/test_auth.py"]

        for file_path in files:
            messages.append(Message(role="user", content=f"Refactor {file_path}"))
            messages.append(
                Message(
                    role="assistant",
                    content=f"I'll refactor {file_path}",
                    tool_calls=[{"name": "read_file", "id": f"call_{len(messages)}"}],
                )
            )
            messages.append(
                Message(
                    role="tool",
                    tool_call_id=f"call_{len(messages)}",
                    tool_name="read_file",
                    content=f"Content of {file_path}...",
                )
            )

        result = await router.compact(messages)

        assert result.success
        # Should capture multiple files
        assert any(f in result.summary for f in files)


class TestErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_timeout_recovery(self, settings, feature_flags, rule_summarizer, mock_provider):
        """Test recovery from LLM timeout."""
        # Mock slow LLM
        import asyncio

        async def slow_chat(*args, **kwargs):
            await asyncio.sleep(10)
            return "Slow response"

        mock_provider.chat = Mock(side_effect=slow_chat)

        llm_summarizer = LLMCompactionSummarizer(
            provider=mock_provider,
            timeout_seconds=1.0,  # Short timeout
        )

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
            hybrid_summarizer=None,
        )

        messages = [Message(role="user", content="Test")]
        result = await router.compact(messages)

        # Should recover and return summary
        assert result.success
        assert result.summary

    @pytest.mark.asyncio
    async def test_partial_llm_failure(
        self, settings, feature_flags, rule_summarizer, hybrid_summarizer
    ):
        """Test handling partial LLM failure in hybrid mode."""
        # Mock hybrid that fails on some sections
        original_summarize_async = hybrid_summarizer.summarize_async

        call_count = [0]

        async def failing_summarize(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:  # Fail every other call
                raise Exception("LLM enhancement failed")
            return await original_summarize_async(*args, **kwargs)

        hybrid_summarizer.summarize_async = failing_summarize

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=None,
            hybrid_summarizer=hybrid_summarizer,
        )

        messages = [Message(role="user", content="Test")]
        result = await router.compact(messages)

        # Should handle partial failures gracefully
        assert result.success
        assert result.summary
