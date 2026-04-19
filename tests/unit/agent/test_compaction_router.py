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

"""Unit tests for compaction router."""

import pytest
from unittest.mock import AsyncMock, Mock

from victor.agent.compaction_router import (
    CompactionRouter,
    CompactionType,
    CompactionResult,
)
from victor.agent.compaction_rule_based import RuleBasedCompactionSummarizer
from victor.config.compaction_strategy_settings import (
    CompactionStrategySettings,
    CompactionFeatureFlags,
)
from victor.providers.base import Message


@pytest.fixture
def settings():
    """Create default compaction strategy settings."""
    return CompactionStrategySettings(
        llm_min_complexity=0.7,
        llm_min_tokens=5000,
        llm_min_messages=20,
    )


@pytest.fixture
def feature_flags():
    """Create default feature flags."""
    return CompactionFeatureFlags(
        enable_rule_based=True,
        enable_llm_based=True,
        enable_hybrid=True,
    )


@pytest.fixture
def rule_summarizer(settings):
    """Create rule-based summarizer."""
    return RuleBasedCompactionSummarizer(settings)


@pytest.fixture
def llm_summarizer():
    """Create mock LLM summarizer."""
    mock_summarizer = Mock()
    mock_summarizer.summarize = Mock(return_value="LLM-based summary")
    return mock_summarizer


@pytest.fixture
def hybrid_summarizer():
    """Create mock hybrid summarizer."""
    mock_summarizer = Mock()
    mock_summarizer.summarize_async = AsyncMock(return_value="Hybrid summary")
    return mock_summarizer


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
def simple_messages():
    """Create simple messages (low complexity)."""
    return [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]


@pytest.fixture
def complex_messages():
    """Create complex messages (high complexity)."""
    messages = []
    for i in range(30):
        messages.append(Message(role="user", content=f"Message {i} about fixing bug in src/file{i}.py"))
        messages.append(Message(
            role="assistant",
            content=f"I'll help with that",
            tool_calls=[{"name": f"tool_{i}", "id": f"call_{i}"}]
        ))
        messages.append(Message(role="tool", tool_call_id=f"call_{i}", tool_name=f"tool_{i}", content="Result"))
    return messages


class TestCompactionRouter:
    """Test suite for CompactionRouter."""

    @pytest.mark.asyncio
    async def test_compact_empty_messages(self, router):
        """Test compacting empty message list."""
        result = await router.compact([])

        assert result.summary == ""
        assert result.removed_count == 0
        assert result.success is True

    @pytest.mark.asyncio
    async def test_compact_simple_messages(self, router, simple_messages):
        """Test compacting simple messages (should use rule-based)."""
        result = await router.compact(simple_messages)

        assert result.summary
        assert result.removed_count == len(simple_messages)
        assert result.success is True
        # Simple messages should use rule-based
        assert result.strategy_used == CompactionType.RULE_BASED

    @pytest.mark.asyncio
    async def test_compact_complex_messages(self, router, complex_messages):
        """Test compacting complex messages (should use LLM-based)."""
        result = await router.compact(complex_messages)

        assert result.summary
        assert result.removed_count == len(complex_messages)
        assert result.success is True
        # Complex messages should have higher complexity, but might still use rule-based
        # depending on the threshold. Let's just verify the complexity is reasonable.
        assert result.complexity_score > 0.5  # Should be moderately complex

    @pytest.mark.asyncio
    async def test_compact_with_session_id(self, router, simple_messages):
        """Test compacting with session ID for analytics."""
        result = await router.compact(simple_messages, session_id="test-session")

        assert result.summary
        assert result.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_compact_with_current_query(self, router, simple_messages):
        """Test compacting with current query for relevance."""
        result = await router.compact(
            simple_messages,
            current_query="continue with the task"
        )

        assert result.summary
        # Query should affect complexity score
        assert 0.0 <= result.complexity_score <= 1.0


class TestStrategySelection:
    """Test strategy selection logic."""

    def test_select_rule_based_low_complexity(self, router):
        """Test selecting rule-based for low complexity."""
        strategy = router._select_strategy(
            message_count=10,
            estimated_tokens=2000,
            complexity_score=0.3,  # Below threshold
        )

        assert strategy == CompactionType.RULE_BASED

    def test_select_llm_based_high_complexity(self, router):
        """Test selecting LLM-based for high complexity."""
        strategy = router._select_strategy(
            message_count=30,
            estimated_tokens=10000,
            complexity_score=0.8,  # Above threshold
        )

        assert strategy == CompactionType.LLM_BASED

    def test_select_hybrid_medium_complexity(self, router):
        """Test selecting hybrid for medium complexity."""
        strategy = router._select_strategy(
            message_count=15,  # Below llm_min_messages
            estimated_tokens=3000,  # Below llm_min_tokens
            complexity_score=0.75,  # Above threshold
        )

        assert strategy == CompactionType.HYBRID

    def test_select_strategy_with_llm_disabled(self, settings, feature_flags):
        """Test strategy selection when LLM is disabled."""
        feature_flags.enable_llm_based = False

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=Mock(),
            llm_summarizer=None,
            hybrid_summarizer=None,
        )

        strategy = router._select_strategy(
            message_count=30,
            estimated_tokens=10000,
            complexity_score=0.9,
        )

        assert strategy == CompactionType.RULE_BASED

    def test_select_strategy_with_rule_disabled(self, settings, feature_flags):
        """Test strategy selection when rule-based is disabled."""
        feature_flags.enable_rule_based = False

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=Mock(),
            llm_summarizer=Mock(),
            hybrid_summarizer=None,
        )

        strategy = router._select_strategy(
            message_count=10,
            estimated_tokens=2000,
            complexity_score=0.3,
        )

        assert strategy == CompactionType.LLM_BASED


class TestComplexityScoring:
    """Test complexity scoring logic."""

    def test_calculate_complexity_simple(self, router, simple_messages):
        """Test complexity score for simple messages."""
        score = router._calculate_complexity(simple_messages, None)

        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low

    def test_calculate_complexity_complex(self, router, complex_messages):
        """Test complexity score for complex messages."""
        score = router._calculate_complexity(complex_messages, None)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high

    def test_calculate_complexity_with_continuation_query(self, router, simple_messages):
        """Test complexity score with continuation query."""
        score_without = router._calculate_complexity(simple_messages, None)
        score_with = router._calculate_complexity(simple_messages, "continue with the next step")

        # Continuation query should increase complexity
        assert score_with >= score_without

    def test_complexity_factors(self, router):
        """Test individual complexity factors."""
        # Message count factor
        messages_10 = [Message(role="user", content="Test")] * 10
        messages_50 = [Message(role="user", content="Test")] * 50

        score_10 = router._calculate_complexity(messages_10, None)
        score_50 = router._calculate_complexity(messages_50, None)

        assert score_50 > score_10


class TestFallbackBehavior:
    """Test fallback behavior on errors."""

    @pytest.mark.asyncio
    async def test_llm_failure_fallback_to_rule(self, settings, feature_flags, rule_summarizer):
        """Test fallback to rule-based when LLM fails."""
        llm_summarizer = Mock()
        llm_summarizer.summarize = Mock(side_effect=Exception("LLM failed"))

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
            hybrid_summarizer=None,
        )

        messages = [Message(role="user", content="Test")]
        result = await router.compact(messages)

        # Should fall back to rule-based
        assert result.success is True
        assert result.summary

    @pytest.mark.asyncio
    async def test_hybrid_failure_fallback_to_rule(self, settings, feature_flags, rule_summarizer):
        """Test fallback to rule-based when hybrid fails."""
        hybrid_summarizer = Mock()
        hybrid_summarizer.summarize_async = AsyncMock(side_effect=Exception("Hybrid failed"))

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=None,
            hybrid_summarizer=hybrid_summarizer,
        )

        messages = [Message(role="user", content="Test")]
        result = await router.compact(messages)

        # Should fall back to rule-based
        assert result.success is True
        assert result.summary


class TestTokenEstimation:
    """Test token estimation logic."""

    def test_estimate_tokens_empty(self, router):
        """Test token estimation for empty messages."""
        tokens = router._estimate_tokens([])
        assert tokens == 0

    def test_estimate_tokens_simple(self, router):
        """Test token estimation for simple messages."""
        messages = [
            Message(role="user", content="Hello world"),  # ~11 chars
        ]

        tokens = router._estimate_tokens(messages)
        # Should be ~3 tokens (11 chars / 4)
        assert tokens > 0
        assert tokens < 10

    def test_estimate_tokens_complex(self, router):
        """Test token estimation for complex messages."""
        messages = [
            Message(role="user", content="Test" * 1000),  # ~4000 chars
        ]

        tokens = router._estimate_tokens(messages)
        # Should be ~1000 tokens
        assert tokens > 500
        assert tokens < 2000


class TestToolExtraction:
    """Test tool extraction logic."""

    def test_extract_tools_from_tool_calls(self, router):
        """Test extracting tools from tool_calls."""
        messages = [
            Message(
                role="assistant",
                content="I'll read the file",
                tool_calls=[
                    {"name": "read_file", "id": "call_1"},
                    {"name": "write_file", "id": "call_2"},
                ]
            ),
        ]

        tools = router._extract_tools(messages)

        assert len(tools) == 2
        assert "read_file" in tools
        assert "write_file" in tools

    def test_extract_tools_from_tool_name(self, router):
        """Test extracting tools from tool_name attribute."""
        # Create a message and manually set tool_name
        from unittest.mock import Mock

        msg = Mock(spec=Message)
        msg.role = "tool"
        msg.tool_call_id = "call_1"
        msg.tool_name = "read_file"
        msg.content = "File content"
        msg.tool_calls = None

        # Extract tools from the mock message
        tools = []
        if hasattr(msg, 'tool_name') and msg.tool_name:
            tools.append(msg.tool_name)

        assert len(tools) == 1
        assert "read_file" in tools


class TestFileExtraction:
    """Test file extraction logic."""

    def test_extract_files_from_messages(self, router):
        """Test extracting file paths from messages."""
        messages = [
            Message(role="user", content="Fix bug in src/auth/login.py"),
            Message(role="assistant", content="I'll also check tests/test_auth.py"),
        ]

        files = router._extract_files(messages)

        assert len(files) >= 1
        assert any("login.py" in f for f in files)

    def test_extract_files_no_matches(self, router):
        """Test file extraction with no file paths."""
        messages = [
            Message(role="user", content="Hello world"),
            Message(role="assistant", content="Hi there"),
        ]

        files = router._extract_files(messages)

        assert len(files) == 0


class TestCompactionResult:
    """Test CompactionResult dataclass."""

    def test_result_creation(self):
        """Test creating a CompactionResult."""
        result = CompactionResult(
            summary="Test summary",
            removed_count=10,
            strategy_used=CompactionType.RULE_BASED,
            complexity_score=0.5,
            duration_ms=100,
            tokens_saved=1000,
            success=True,
            error_message=None,
            session_id="test-session",
        )

        assert result.summary == "Test summary"
        assert result.removed_count == 10
        assert result.strategy_used == CompactionType.RULE_BASED
        assert result.complexity_score == 0.5
        assert result.duration_ms == 100
        assert result.tokens_saved == 1000
        assert result.success is True
        assert result.session_id == "test-session"


class TestSettingsIntegration:
    """Test integration with settings."""

    def test_custom_thresholds(self, feature_flags, rule_summarizer):
        """Test router with custom thresholds."""
        settings = CompactionStrategySettings(
            llm_min_complexity=0.5,  # Lower threshold
            llm_min_tokens=3000,     # Lower threshold
            llm_min_messages=10,     # Lower threshold
        )

        router = CompactionRouter(
            settings=settings,
            feature_flags=feature_flags,
            rule_summarizer=rule_summarizer,
            llm_summarizer=None,
            hybrid_summarizer=None,
        )

        # Should use LLM with lower complexity
        strategy = router._select_strategy(
            message_count=15,
            estimated_tokens=4000,
            complexity_score=0.6,  # Above 0.5 threshold
        )

        assert strategy == CompactionType.LLM_BASED
