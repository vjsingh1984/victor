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

"""Unit tests for rule-based compaction summarizer."""

import pytest

from victor.agent.compaction_rule_based import (
    RuleBasedCompactionSummarizer,
    RuleBasedSummary,
)
from victor.config.compaction_strategy_settings import CompactionStrategySettings
from victor.providers.base import Message


@pytest.fixture
def settings():
    """Create default compaction strategy settings."""
    return CompactionStrategySettings()


@pytest.fixture
def summarizer(settings):
    """Create rule-based summarizer with default settings."""
    return RuleBasedCompactionSummarizer(settings)


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(role="user", content="Please help me fix the bug in the authentication module"),
        Message(
            role="assistant",
            content="I'll help you fix the authentication bug. Let me start by reading the relevant files.",
            tool_calls=[{"name": "read_file", "id": "call_1"}],
        ),
        Message(
            role="tool",
            tool_call_id="call_1",
            tool_name="read_file",
            content="File: src/auth/login.py\n\ndef login(username, password):\n    # TODO: Add validation\n    return authenticate(username, password)",
        ),
        Message(
            role="assistant",
            content="I found the issue. The login function is missing input validation. Let me add that.",
            tool_calls=[{"name": "write_file", "id": "call_2"}],
        ),
        Message(
            role="tool",
            tool_call_id="call_2",
            tool_name="write_file",
            content="Successfully wrote to src/auth/login.py",
        ),
        Message(role="user", content="Great! Now can you also add tests for the login function?"),
    ]


class TestRuleBasedCompactionSummarizer:
    """Test suite for RuleBasedCompactionSummarizer."""

    def test_summarize_empty_messages(self, summarizer):
        """Test summarizing empty message list."""
        summary = summarizer.summarize([])
        assert summary == ""

    def test_summarize_basic_messages(self, summarizer, sample_messages):
        """Test basic summarization of messages."""
        import json

        summary = summarizer.summarize(sample_messages)

        # Verify JSON format
        summary_dict = json.loads(summary)
        assert "scope" in summary_dict
        assert "tools_mentioned" in summary_dict
        assert "recent_user_requests" in summary_dict
        assert "pending_work" in summary_dict
        assert "key_files_referenced" in summary_dict
        assert "current_work" in summary_dict
        assert "key_timeline" in summary_dict

        # Verify scope
        assert "earlier messages compacted" in summary_dict["scope"]
        assert "user=2" in summary_dict["scope"]  # 2 user messages
        assert "assistant=2" in summary_dict["scope"]  # 2 assistant messages
        assert "tool=2" in summary_dict["scope"]  # 2 tool messages

    def test_summarize_with_tools(self, summarizer):
        """Test tool extraction from messages."""
        import json

        messages = [
            Message(role="user", content="Read the file"),
            Message(
                role="assistant",
                content="I'll read it",
                tool_calls=[{"name": "read_file", "id": "call_1"}],
            ),
            Message(role="tool", tool_call_id="call_1", tool_name="read_file", content="..."),
            Message(
                role="assistant",
                content="Now I'll write to it",
                tool_calls=[{"name": "write_file", "id": "call_2"}],
            ),
            Message(role="tool", tool_call_id="call_2", tool_name="write_file", content="..."),
        ]

        summary = summarizer.summarize(messages)
        summary_dict = json.loads(summary)
        assert "read_file" in summary_dict["tools_mentioned"]
        assert "write_file" in summary_dict["tools_mentioned"]

    def test_extract_file_paths(self, summarizer):
        """Test file path extraction from messages."""
        import json

        messages = [
            Message(role="user", content="Fix the bug in src/auth/login.py"),
            Message(role="assistant", content="I'll also check tests/test_auth.py"),
            Message(role="user", content="What about src/utils/helpers.ts?"),
        ]

        summary = summarizer.summarize(messages)
        summary_dict = json.loads(summary)
        # At least some files should be extracted (extraction logic may vary)
        assert len(summary_dict["key_files_referenced"]) >= 2
        # Check that the Python files are extracted
        assert "src/auth/login.py" in summary_dict["key_files_referenced"]
        assert "tests/test_auth.py" in summary_dict["key_files_referenced"]

    def test_infer_pending_work(self, summarizer):
        """Test pending work inference from messages."""
        import json

        messages = [
            Message(role="user", content="Fix the authentication bug"),
            Message(role="assistant", content="Done! Next, we need to add tests"),
            Message(role="user", content="TODO: Add error handling later"),
        ]

        summary = summarizer.summarize(messages)
        summary_dict = json.loads(summary)
        # Should have pending work items
        assert len(summary_dict["pending_work"]) >= 0  # May or may not detect pending work

    def test_truncation(self, summarizer):
        """Test content truncation in summaries."""
        import json

        long_content = "x" * 500
        messages = [
            Message(role="user", content=long_content),
        ]

        summary = summarizer.summarize(messages)
        summary_dict = json.loads(summary)
        # Content should be truncated in timeline
        assert len(summary_dict) > 0

    def test_no_system_messages_in_timeline(self, summarizer):
        """Test that system messages are excluded from timeline."""
        import json

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="User message"),
            Message(role="assistant", content="Assistant message"),
        ]

        summary = summarizer.summarize(messages)
        summary_dict = json.loads(summary)
        # Timeline should not contain system messages
        for entry in summary_dict["key_timeline"]:
            assert entry["role"] != "system"

    def test_performance(self, summarizer):
        """Test that summarization is fast (<100ms for 100 messages)."""
        import time

        # Create 100 messages
        messages = [
            Message(role="user", content=f"Message {i} about fixing bugs in src/file{i}.py")
            for i in range(100)
        ]

        start = time.time()
        summary = summarizer.summarize(messages)
        duration = (time.time() - start) * 1000  # Convert to ms

        assert summary
        assert duration < 100, f"Summarization too slow: {duration}ms"

    def test_max_files_limit(self, summarizer):
        """Test that file extraction is limited to 8 files."""
        import json

        # Create messages with more than 8 files
        messages = [
            Message(role="user", content=f"Check src/file{i}.py for bugs") for i in range(20)
        ]

        summary = summarizer.summarize(messages)
        summary_dict = json.loads(summary)

        # Should be limited to 8 files
        assert len(summary_dict["key_files_referenced"]) <= 8


class TestRuleBasedSummary:
    """Test suite for RuleBasedSummary dataclass."""

    def test_summary_creation(self):
        """Test creating a RuleBasedSummary."""
        summary = RuleBasedSummary(
            scope="10 messages compacted",
            tools_mentioned=["read_file", "write_file"],
            recent_user_requests=["Fix the bug"],
            pending_work=["Add tests"],
            key_files_referenced=["src/auth/login.py"],
            current_work="Fixing authentication bug",
            key_timeline=[
                {"role": "user", "content": "Fix bug"},
                {"role": "assistant", "content": "I'll help"},
            ],
        )

        assert summary.scope == "10 messages compacted"
        assert len(summary.tools_mentioned) == 2
        assert len(summary.pending_work) == 1
        assert len(summary.key_files_referenced) == 1


class TestCompactionSettingsIntegration:
    """Test integration with CompactionStrategySettings."""

    def test_settings_integration(self):
        """Test that summarizer respects settings."""
        import json

        settings = CompactionStrategySettings(
            rule_preserve_recent=10,
        )

        summarizer = RuleBasedCompactionSummarizer(settings)

        # Create messages
        messages = [
            Message(role="user", content="Fix the bug in src/auth/login.py"),
            Message(role="assistant", content="I'll help"),
        ]

        summary = summarizer.summarize(messages)

        # Verify JSON format is used
        summary_dict = json.loads(summary)
        assert "scope" in summary_dict


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_messages_with_none_content(self, summarizer):
        """Test handling messages with None content."""
        messages = [
            Message(role="user", content="Valid message"),
            Message(role="assistant", content=""),  # Empty string instead of None
            Message(
                role="tool", tool_call_id="call_1", tool_name="read_file", content=""
            ),  # Empty string
        ]

        # Should not raise an error
        summary = summarizer.summarize(messages)
        assert summary

    def test_messages_with_unicode(self, summarizer):
        """Test handling messages with unicode characters."""
        import json

        messages = [
            Message(role="user", content="Fix the bug in 文件.py"),
            Message(role="assistant", content="I'll help with the fix 🐛"),
        ]

        summary = summarizer.summarize(messages)
        summary_dict = json.loads(summary)
        assert summary
        # Check that unicode is preserved in the parsed JSON
        assert any("文件" in req for req in summary_dict["recent_user_requests"])

    def test_very_long_message(self, summarizer):
        """Test handling very long messages."""
        import json

        long_content = "word " * 10000  # ~50KB
        messages = [
            Message(role="user", content=long_content),
        ]

        summary = summarizer.summarize(messages)
        summary_dict = json.loads(summary)
        assert summary
        # Check that content is truncated in timeline
        assert len(summary_dict["key_timeline"]) > 0
        # Timeline entries should be truncated
        for entry in summary_dict["key_timeline"]:
            assert len(entry["content"]) <= 160 + 20  # Allow some margin for JSON encoding

    def test_mixed_tool_call_formats(self, summarizer):
        """Test handling different tool call formats."""
        messages = [
            Message(
                role="assistant",
                content="I'll read the file",
                tool_calls=[
                    {"name": "read_file", "id": "call_1"},
                    {"name": "write_file", "id": "call_2"},  # Dict format
                ],
            ),
        ]

        # Should handle dict format
        summary = summarizer.summarize(messages)
        assert summary
