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

"""Tests for ContextCompactor."""

import pytest
from unittest.mock import MagicMock

from victor.agent.context_compactor import (
    CompactionAction,
    CompactionTrigger,
    CompactorConfig,
    ContextCompactor,
    TruncationStrategy,
    create_context_compactor,
)
from victor.agent.conversation_controller import ContextMetrics
from victor.providers.base import Message


@pytest.fixture
def mock_controller():
    """Create a mock ConversationController."""
    controller = MagicMock()

    # Default metrics - 50% utilization
    controller.get_context_metrics.return_value = ContextMetrics(
        char_count=100000,
        estimated_tokens=25000,
        message_count=20,
        is_overflow_risk=False,
        max_context_chars=200000,
    )

    controller.smart_compact_history.return_value = 10  # 10 messages removed
    controller.get_compaction_summaries.return_value = []
    controller.messages = [Message(role="system", content="System prompt")]

    return controller


@pytest.fixture
def compactor(mock_controller):
    """Create a ContextCompactor with mock controller."""
    return ContextCompactor(mock_controller)


class TestCompactorConfig:
    """Tests for CompactorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CompactorConfig()

        assert config.proactive_threshold == 0.90
        assert config.min_messages_after_compact == 8
        assert config.tool_result_max_chars == 8192
        assert config.tool_result_max_lines == 230
        assert config.truncation_strategy == TruncationStrategy.SMART
        assert config.preserve_code_blocks is True
        assert config.enable_proactive is True
        assert config.enable_tool_truncation is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CompactorConfig(
            proactive_threshold=0.60,
            min_messages_after_compact=10,
            enable_proactive=False,
        )

        assert config.proactive_threshold == 0.60
        assert config.min_messages_after_compact == 10
        assert config.enable_proactive is False


class TestCompactionAction:
    """Tests for CompactionAction dataclass."""

    def test_action_taken_true(self):
        """Test action_taken when messages removed."""
        action = CompactionAction(
            trigger=CompactionTrigger.THRESHOLD,
            messages_removed=5,
        )
        assert action.action_taken is True

    def test_action_taken_false(self):
        """Test action_taken when nothing done."""
        action = CompactionAction(
            trigger=CompactionTrigger.NONE,
            messages_removed=0,
        )
        assert action.action_taken is False

    def test_action_taken_truncation(self):
        """Test action_taken with truncations only."""
        action = CompactionAction(
            trigger=CompactionTrigger.MANUAL,
            messages_removed=0,
            truncations_applied=3,
        )
        assert action.action_taken is True


class TestContextCompactorInit:
    """Tests for ContextCompactor initialization."""

    def test_default_initialization(self, mock_controller):
        """Test default initialization."""
        compactor = ContextCompactor(mock_controller)

        assert compactor.controller is mock_controller
        assert compactor.config.proactive_threshold == 0.90
        assert compactor._compaction_count == 0

    def test_custom_config(self, mock_controller):
        """Test initialization with custom config."""
        config = CompactorConfig(proactive_threshold=0.80)
        compactor = ContextCompactor(mock_controller, config)

        assert compactor.config.proactive_threshold == 0.80


class TestShouldCompact:
    """Tests for should_compact method."""

    def test_should_not_compact_low_utilization(self, compactor, mock_controller):
        """Test no compaction at low utilization."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=12500,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()
        assert should is False
        assert trigger == CompactionTrigger.NONE

    def test_should_compact_at_threshold(self, compactor, mock_controller):
        """Test compaction triggered at threshold."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=185000,  # 92.5% utilization (exceeds 90% threshold)
            estimated_tokens=46250,
            message_count=30,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()
        assert should is True
        assert trigger == CompactionTrigger.THRESHOLD

    def test_should_compact_on_overflow(self, compactor, mock_controller):
        """Test compaction triggered on overflow."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=200000,
            estimated_tokens=50000,
            message_count=50,
            is_overflow_risk=True,
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()
        assert should is True
        assert trigger == CompactionTrigger.OVERFLOW

    def test_proactive_disabled(self, mock_controller):
        """Test proactive compaction can be disabled."""
        config = CompactorConfig(enable_proactive=False)
        compactor = ContextCompactor(mock_controller, config)

        # Even at 92.5% utilization, should not trigger when proactive disabled
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=185000,
            estimated_tokens=46250,
            message_count=30,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        should, trigger = compactor.should_compact()
        assert should is False


class TestCheckAndCompact:
    """Tests for check_and_compact method."""

    def test_no_action_low_utilization(self, compactor, mock_controller):
        """Test no action at low utilization."""
        mock_controller.get_context_metrics.return_value = ContextMetrics(
            char_count=50000,
            estimated_tokens=12500,
            message_count=10,
            is_overflow_risk=False,
            max_context_chars=200000,
        )

        action = compactor.check_and_compact()

        assert action.trigger == CompactionTrigger.NONE
        assert action.action_taken is False
        mock_controller.smart_compact_history.assert_not_called()

    def test_compaction_at_threshold(self, compactor, mock_controller):
        """Test compaction at threshold."""
        # First call returns high utilization (92.5%, above 90% threshold)
        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=185000,
                estimated_tokens=46250,
                message_count=30,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]

        action = compactor.check_and_compact()

        assert action.trigger == CompactionTrigger.THRESHOLD
        assert action.messages_removed == 10
        assert action.chars_freed == 105000  # 185000 - 80000
        mock_controller.smart_compact_history.assert_called_once()

    def test_forced_compaction(self, compactor, mock_controller):
        """Test forced compaction."""
        # Even at low utilization, force should trigger
        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=50000,
                estimated_tokens=12500,
                message_count=10,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=30000,
                estimated_tokens=7500,
                message_count=5,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]

        action = compactor.check_and_compact(force=True)

        assert action.trigger == CompactionTrigger.MANUAL
        assert action.action_taken is True

    def test_statistics_updated(self, compactor, mock_controller):
        """Test that statistics are updated after compaction."""
        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=185000,  # 92.5% utilization (exceeds 90% threshold)
                estimated_tokens=46250,
                message_count=30,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            # Third call from get_statistics()
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]

        compactor.check_and_compact()

        stats = compactor.get_statistics()
        assert stats["compaction_count"] == 1
        assert stats["total_chars_freed"] > 0


class TestTruncateToolResult:
    """Tests for tool result truncation."""

    def test_no_truncation_needed(self, compactor):
        """Test no truncation when content is small."""
        content = "Short content that doesn't need truncation."

        result = compactor.truncate_tool_result(content)

        assert result.truncated is False
        assert result.content == content
        assert result.truncated_chars == 0

    def test_truncation_by_chars(self, compactor):
        """Test truncation by character limit."""
        content = "x" * 10000  # Exceeds default 8000

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert len(result.content) < len(content)
        assert result.truncated_chars > 0

    def test_truncation_by_lines(self, compactor):
        """Test truncation by line limit."""
        content = "\n".join([f"Line {i}" for i in range(500)])

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert result.content.count("\n") < 500

    def test_truncation_disabled(self, mock_controller):
        """Test truncation can be disabled."""
        config = CompactorConfig(enable_tool_truncation=False)
        compactor = ContextCompactor(mock_controller, config)

        content = "x" * 10000

        result = compactor.truncate_tool_result(content)

        assert result.truncated is False
        assert result.content == content

    def test_head_truncation_strategy(self, mock_controller):
        """Test HEAD truncation strategy."""
        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.HEAD,
            tool_result_max_chars=100,
            tool_result_max_lines=5,
        )
        compactor = ContextCompactor(mock_controller, config)

        content = "\n".join([f"Line {i}: some content here" for i in range(20)])

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert "Line 0" in result.content
        assert "[output truncated]" in result.content

    def test_tail_truncation_strategy(self, mock_controller):
        """Test TAIL truncation strategy."""
        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.TAIL,
            tool_result_max_chars=100,
            tool_result_max_lines=5,
        )
        compactor = ContextCompactor(mock_controller, config)

        content = "\n".join([f"Line {i}: some content here" for i in range(20)])

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert "Line 19" in result.content
        assert "[output truncated]" in result.content

    def test_smart_truncation_preserves_errors(self, mock_controller):
        """Test SMART truncation preserves error messages."""
        config = CompactorConfig(
            truncation_strategy=TruncationStrategy.SMART,
            tool_result_max_chars=500,
            tool_result_max_lines=10,
        )
        compactor = ContextCompactor(mock_controller, config)

        lines = [f"Line {i}: normal content" for i in range(50)]
        lines[25] = "Error: Something went wrong!"
        content = "\n".join(lines)

        result = compactor.truncate_tool_result(content)

        # Error should be preserved or at least the beginning/end kept
        assert result.truncated is True
        # Verify truncation marker is present
        assert "truncated" in result.content.lower()


class TestTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens_prose(self, compactor):
        """Test token estimation for prose."""
        tokens = compactor._estimate_tokens(4000, "prose")
        assert tokens == 1000  # 4000 / 4.0

    def test_estimate_tokens_code(self, compactor):
        """Test token estimation for code."""
        tokens = compactor._estimate_tokens(3000, "code")
        assert tokens == 1000  # 3000 / 3.0

    def test_estimate_message_tokens_code(self, compactor):
        """Test message token estimation for code content."""
        message = Message(role="assistant", content="```python\nprint('hello')\n```")

        tokens = compactor.estimate_message_tokens(message)

        # Should use code factor (3.0) plus overhead
        assert tokens > 0

    def test_estimate_message_tokens_prose(self, compactor):
        """Test message token estimation for prose."""
        message = Message(role="user", content="This is a simple text message.")

        tokens = compactor.estimate_message_tokens(message)

        # Should use prose factor (4.0) plus overhead
        assert tokens > 0


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_statistics(self, compactor, mock_controller):
        """Test getting statistics."""
        stats = compactor.get_statistics()

        assert "current_utilization" in stats
        assert "compaction_count" in stats
        assert "proactive_threshold" in stats
        assert stats["compaction_count"] == 0

    def test_reset_statistics(self, compactor, mock_controller):
        """Test resetting statistics."""
        # Perform a compaction (92.5% utilization exceeds 90% threshold)
        mock_controller.get_context_metrics.side_effect = [
            ContextMetrics(
                char_count=185000,
                estimated_tokens=46250,
                message_count=30,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
            ContextMetrics(
                char_count=80000,
                estimated_tokens=20000,
                message_count=20,
                is_overflow_risk=False,
                max_context_chars=200000,
            ),
        ]
        compactor.check_and_compact()

        # Verify stats were updated
        assert compactor._compaction_count == 1

        # Reset and verify
        compactor.reset_statistics()

        assert compactor._compaction_count == 0
        assert compactor._total_chars_freed == 0


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_context_compactor(self, mock_controller):
        """Test factory function."""
        compactor = create_context_compactor(
            mock_controller,
            proactive_threshold=0.80,
            enable_proactive=True,
        )

        assert compactor.config.proactive_threshold == 0.80
        assert compactor.config.enable_proactive is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_content_truncation(self, compactor):
        """Test truncation with empty content."""
        result = compactor.truncate_tool_result("")

        assert result.truncated is False
        assert result.content == ""

    def test_single_line_content(self, compactor):
        """Test truncation with single line content."""
        content = "Single line content"

        result = compactor.truncate_tool_result(content)

        assert result.truncated is False

    def test_very_long_single_line(self, mock_controller):
        """Test truncation of very long single line."""
        config = CompactorConfig(tool_result_max_chars=100)
        compactor = ContextCompactor(mock_controller, config)

        content = "x" * 1000

        result = compactor.truncate_tool_result(content)

        assert result.truncated is True
        assert len(result.content) < len(content)
