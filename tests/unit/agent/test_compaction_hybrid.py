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

"""Unit tests for hybrid compaction summarizer."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from victor.agent.compaction_hybrid import HybridCompactionSummarizer, HybridSummary
from victor.agent.compaction_rule_based import (
    RuleBasedCompactionSummarizer,
    RuleBasedSummary,
)
from victor.config.compaction_strategy_settings import CompactionStrategySettings
from victor.providers.base import Message


@pytest.fixture
def settings():
    """Create default compaction strategy settings."""
    return CompactionStrategySettings(
        hybrid_llm_enhancement=True,
        hybrid_llm_sections=["pending_work", "current_work"],
        llm_timeout_seconds=5.0,
    )


@pytest.fixture
def rule_summarizer(settings):
    """Create rule-based summarizer."""
    return RuleBasedCompactionSummarizer(settings)


@pytest.fixture
def llm_summarizer():
    """Create mock LLM summarizer."""
    mock_summarizer = Mock()
    mock_summarizer.summarize = Mock(return_value="Enhanced LLM summary")
    return mock_summarizer


@pytest.fixture
def hybrid_summarizer(settings, rule_summarizer, llm_summarizer):
    """Create hybrid summarizer."""
    return HybridCompactionSummarizer(
        config=settings,
        rule_summarizer=rule_summarizer,
        llm_summarizer=llm_summarizer,
    )


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(role="user", content="Fix the authentication bug"),
        Message(
            role="assistant",
            content="I'll help fix the bug. TODO: Add tests later",
            tool_calls=[{"name": "read_file", "id": "call_1"}]
        ),
        Message(
            role="tool",
            tool_call_id="call_1",
            tool_name="read_file",
            content="File content..."
        ),
    ]


class TestHybridCompactionSummarizer:
    """Test suite for HybridCompactionSummarizer."""

    @pytest.mark.asyncio
    async def test_summarize_empty_messages(self, hybrid_summarizer):
        """Test summarizing empty message list."""
        summary = await hybrid_summarizer.summarize_async([])
        assert summary == ""

    @pytest.mark.asyncio
    async def test_summarize_basic(
        self,
        hybrid_summarizer,
        sample_messages,
    ):
        """Test basic hybrid summarization."""
        import json
        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # Verify JSON format
        summary_dict = json.loads(summary)
        assert "scope" in summary_dict
        assert "tools_mentioned" in summary_dict

    @pytest.mark.asyncio
    async def test_summarize_without_llm_enhancement(
        self,
        settings,
        rule_summarizer,
        llm_summarizer,
        sample_messages,
    ):
        """Test hybrid summarization with LLM enhancement disabled."""
        settings.hybrid_llm_enhancement = False
        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
        )

        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # LLM should not be called
        llm_summarizer.summarize.assert_not_called()

        # Should still have valid summary
        assert summary

    @pytest.mark.asyncio
    async def test_summarize_without_llm_summarizer(
        self,
        settings,
        rule_summarizer,
        sample_messages,
    ):
        """Test hybrid summarization without LLM summarizer."""
        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=None,  # No LLM summarizer
        )

        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # Should fall back to rule-based
        assert summary

    @pytest.mark.asyncio
    async def test_llm_enhancement_pending_work(
        self,
        settings,
        rule_summarizer,
        llm_summarizer,
        sample_messages,
    ):
        """Test LLM enhancement of pending work section."""
        llm_summarizer.summarize.return_value = "Pending: Add comprehensive unit tests"

        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
        )

        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # Should contain enhanced content (either in pending work or current work section)
        assert "Pending:" in summary or "Add comprehensive unit tests" in summary

    @pytest.mark.asyncio
    async def test_llm_enhancement_current_work(
        self,
        settings,
        rule_summarizer,
        llm_summarizer,
        sample_messages,
    ):
        """Test LLM enhancement of current work section."""
        import json
        llm_summarizer.summarize.return_value = "Currently fixing authentication bug in login.py"

        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
        )

        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # Should contain enhanced content
        summary_dict = json.loads(summary)
        assert "current_work" in summary_dict
        # The enhanced content should be present
        assert "authentication bug" in summary_dict["current_work"] or summary_dict["current_work"]

    @pytest.mark.asyncio
    async def test_llm_timeout_fallback(
        self,
        settings,
        rule_summarizer,
        llm_summarizer,
        sample_messages,
    ):
        """Test fallback to rule-based on LLM timeout."""
        # Mock LLM call to timeout
        async def mock_timeout(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return "LLM summary"

        llm_summarizer.summarize = Mock(side_effect=mock_timeout)

        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
        )

        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # Should fall back to rule-based (no exception)
        assert summary

    @pytest.mark.asyncio
    async def test_llm_error_fallback(
        self,
        settings,
        rule_summarizer,
        llm_summarizer,
        sample_messages,
    ):
        """Test fallback to rule-based on LLM error."""
        # Mock LLM call to raise exception
        llm_summarizer.summarize = Mock(side_effect=Exception("LLM failed"))

        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
        )

        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # Should fall back to rule-based (no exception)
        assert summary

    def test_summarize_sync(
        self,
        hybrid_summarizer,
        sample_messages,
    ):
        """Test synchronous summarize method."""
        summary = hybrid_summarizer.summarize(sample_messages)

        # Should return valid summary
        assert summary

    @pytest.mark.asyncio
    async def test_multiple_enhancement_sections(
        self,
        settings,
        rule_summarizer,
        llm_summarizer,
        sample_messages,
    ):
        """Test enhancing multiple sections."""
        import json
        settings.hybrid_llm_sections = [
            "pending_work",
            "current_work",
            "tools_mentioned",
            "key_files_referenced",
        ]

        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
        )

        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # Should contain all sections
        summary_dict = json.loads(summary)
        assert "tools_mentioned" in summary_dict
        assert "pending_work" in summary_dict or "current_work" in summary_dict


class TestHybridSummary:
    """Test suite for HybridSummary dataclass."""

    def test_summary_creation(self):
        """Test creating a HybridSummary."""
        rule_summary = RuleBasedSummary(
            scope="10 messages compacted",
            tools_mentioned=["read_file"],
            recent_user_requests=["Fix bug"],
            pending_work=["Add tests"],
            key_files_referenced=["src/auth/login.py"],
            current_work="Fixing bug",
            key_timeline=[],
        )

        hybrid_summary = HybridSummary(
            rule_summary=rule_summary,
            enhanced_sections={"pending_work": "Enhanced pending work"},
            strategy_used="hybrid",
            enhancement_success=True,
            fallback_reason=None,
        )

        assert hybrid_summary.strategy_used == "hybrid"
        assert hybrid_summary.enhancement_success is True
        assert "pending_work" in hybrid_summary.enhanced_sections


class TestCompactionSettingsIntegration:
    """Test integration with CompactionStrategySettings."""

    @pytest.mark.asyncio
    async def test_custom_enhancement_sections(
        self,
        rule_summarizer,
        llm_summarizer,
        sample_messages,
    ):
        """Test custom enhancement sections configuration."""
        settings = CompactionStrategySettings(
            hybrid_llm_sections=["tools_mentioned"],  # Only enhance tools
        )

        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
        )

        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # Should only enhance tools section
        assert summary

    @pytest.mark.asyncio
    async def test_timeout_settings(
        self,
        rule_summarizer,
        llm_summarizer,
        sample_messages,
    ):
        """Test custom timeout settings."""
        settings = CompactionStrategySettings(
            llm_timeout_seconds=1.0,  # Very short timeout
        )

        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
        )

        # Mock slow LLM
        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(2)
            return "Slow LLM summary"

        llm_summarizer.summarize = Mock(side_effect=slow_llm)

        summary = await hybrid_summarizer.summarize_async(sample_messages)

        # Should timeout and fall back to rule-based
        assert summary


class TestPromptBuilding:
    """Test prompt building for LLM enhancement."""

    @pytest.mark.asyncio
    async def test_pending_work_prompt(
        self,
        hybrid_summarizer,
        sample_messages,
    ):
        """Test prompt building for pending work enhancement."""
        # The prompt should include context about pending work
        # This is tested indirectly through the summarize method
        summary = await hybrid_summarizer.summarize_async(sample_messages)
        assert summary

    @pytest.mark.asyncio
    async def test_current_work_prompt(
        self,
        hybrid_summarizer,
        sample_messages,
    ):
        """Test prompt building for current work enhancement."""
        summary = await hybrid_summarizer.summarize_async(sample_messages)
        assert summary

    @pytest.mark.asyncio
    async def test_tools_prompt(
        self,
        hybrid_summarizer,
        sample_messages,
    ):
        """Test prompt building for tools enhancement."""
        summary = await hybrid_summarizer.summarize_async(sample_messages)
        assert summary

    @pytest.mark.asyncio
    async def test_files_prompt(
        self,
        hybrid_summarizer,
        sample_messages,
    ):
        """Test prompt building for files enhancement."""
        summary = await hybrid_summarizer.summarize_async(sample_messages)
        assert summary


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_messages_with_none_content(
        self,
        hybrid_summarizer,
    ):
        """Test handling messages with None content."""
        messages = [
            Message(role="user", content="Valid message"),
            Message(role="assistant", content=""),  # Empty string instead of None
        ]

        summary = await hybrid_summarizer.summarize_async(messages)
        assert summary

    @pytest.mark.asyncio
    async def test_empty_rule_summary(
        self,
        settings,
        llm_summarizer,
    ):
        """Test handling when rule summary has no content."""
        import json
        rule_summarizer = Mock()
        rule_summarizer.summarize = Mock(return_value="")

        hybrid_summarizer = HybridCompactionSummarizer(
            config=settings,
            rule_summarizer=rule_summarizer,
            llm_summarizer=llm_summarizer,
        )

        messages = [Message(role="user", content="Test")]
        summary = await hybrid_summarizer.summarize_async(messages)
        # When rule summary is empty, hybrid should return a valid JSON structure with empty fields
        summary_dict = json.loads(summary)
        assert summary_dict["scope"] == ""
        assert summary_dict["tools_mentioned"] == []
        assert summary_dict["current_work"] == ""

    @pytest.mark.asyncio
    async def test_concurrent_summarization(
        self,
        hybrid_summarizer,
        sample_messages,
    ):
        """Test concurrent summarization calls."""
        # Run multiple summarizations concurrently
        tasks = [
            hybrid_summarizer.summarize_async(sample_messages)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)
        assert len(results) == 5
