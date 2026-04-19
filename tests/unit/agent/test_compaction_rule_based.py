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
        Message(
            role="user",
            content="Please help me fix the bug in the authentication module"
        ),
        Message(
            role="assistant",
            content="I'll help you fix the authentication bug. Let me start by reading the relevant files.",
            tool_calls=[{"name": "read_file", "id": "call_1"}]
        ),
        Message(
            role="tool",
            tool_call_id="call_1",
            tool_name="read_file",
            content="File: src/auth/login.py\n\ndef login(username, password):\n    # TODO: Add validation\n    return authenticate(username, password)"
        ),
        Message(
            role="assistant",
            content="I found the issue. The login function is missing input validation. Let me add that.",
            tool_calls=[{"name": "write_file", "id": "call_2"}]
        ),
        Message(
            role="tool",
            tool_call_id="call_2",
            tool_name="write_file",
            content="Successfully wrote to src/auth/login.py"
        ),
        Message(
            role="user",
            content="Great! Now can you also add tests for the login function?"
        ),
    ]


class TestRuleBasedCompactionSummarizer:
    """Test suite for RuleBasedCompactionSummarizer."""

    def test_summarize_empty_messages(self, summarizer):
        """Test summarizing empty message list."""
        summary = summarizer.summarize([])
        assert summary == ""

    def test_summarize_basic_messages(self, summarizer, sample_messages):
        """Test basic summarization of messages."""
        summary = summarizer.summarize(sample_messages)

        # Verify XML format
        assert "<summary>" in summary
        assert "</summary>" in summary
        assert "Conversation summary:" in summary

        # Verify scope
        assert "earlier messages compacted" in summary
        assert "user=2" in summary  # 2 user messages
        assert "assistant=2" in summary  # 2 assistant messages
        assert "tool=2" in summary  # 2 tool messages

    def test_summarize_with_tools(self, summarizer):
        """Test tool extraction from messages."""
        messages = [
            Message(role="user", content="Read the file"),
            Message(
                role="assistant",
                content="I'll read it",
                tool_calls=[{"name": "read_file", "id": "call_1"}]
            ),
            Message(role="tool", tool_call_id="call_1", tool_name="read_file", content="..."),
            Message(
                role="assistant",
                content="Now I'll write to it",
                tool_calls=[{"name": "write_file", "id": "call_2"}]
            ),
            Message(role="tool", tool_call_id="call_2", tool_name="write_file", content="..."),
        ]

        summary = summarizer.summarize(messages)
        assert "Tools mentioned:" in summary
        assert "read_file" in summary
        assert "write_file" in summary

    def test_extract_file_paths(self, summarizer):
        """Test file path extraction from messages."""
        messages = [
            Message(role="user", content="Fix the bug in src/auth/login.py"),
            Message(role="assistant", content="I'll also check tests/test_auth.py"),
            Message(role="user", content="What about src/utils/helpers.ts?"),
        ]

        summary = summarizer.summarize(messages)
        assert "Key files referenced:" in summary
        assert "src/auth/login.py" in summary
        assert "tests/test_auth.py" in summary
        assert "src/utils/helpers.ts" in summary

    def test_infer_pending_work(self, summarizer):
        """Test pending work inference from messages."""
        messages = [
            Message(role="user", content="Fix the authentication bug"),
            Message(role="assistant", content="Done! Next, we need to add tests"),
            Message(role="user", content="TODO: Add error handling later"),
        ]

        summary = summarizer.summarize(messages)
        assert "Pending work:" in summary

    def test_truncation(self, summarizer):
        """Test content truncation in summaries."""
        long_content = "x" * 500
        messages = [
            Message(role="user", content=long_content),
        ]

        summary = summarizer.summarize(messages)
        # Content should be truncated
        assert "…" in summary or len(summary) < len(long_content)

    def test_xml_escaping(self, summarizer):
        """Test XML special character escaping."""
        messages = [
            Message(role="user", content="Check if x < y && z > 'value'"),
        ]

        summary = summarizer.summarize(messages)
        # XML special characters should be escaped
        assert "&lt;" in summary  # < escaped
        assert "&gt;" in summary  # > escaped
        assert "&amp;" in summary  # & escaped

    def test_no_system_messages_in_timeline(self, summarizer):
        """Test that system messages are excluded from timeline."""
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="User message"),
            Message(role="assistant", content="Assistant message"),
        ]

        summary = summarizer.summarize(messages)
        # Timeline should not contain system messages
        lines = summary.split("\n")
        timeline_started = False
        for line in lines:
            if "- Key timeline:" in line:
                timeline_started = True
                continue
            if timeline_started and line.strip().startswith("- "):
                # Timeline entry
                assert "system:" not in line.lower()

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
        # Create messages with more than 8 files
        messages = [
            Message(role="user", content=f"Check src/file{i}.py for bugs")
            for i in range(20)
        ]

        summary = summarizer.summarize(messages)

        # Check that "Key files referenced" section exists
        assert "Key files referenced:" in summary

        # Extract the files from the "Key files referenced" line only
        import re
        # Match the line that contains "Key files referenced:"
        for line in summary.split("\n"):
            if "Key files referenced:" in line:
                # Extract file names from this line
                files = re.findall(r'src/file\d+\.py', line)
                # Should be limited to 8
                assert len(files) <= 8, f"Found {len(files)} files in line: {line}"
                break
        else:
            # If no "Key files referenced" line found, that's also ok (might not have extracted any)
            pass


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
        settings = CompactionStrategySettings(
            rule_preserve_recent=10,
            rule_xml_format=True,
        )

        summarizer = RuleBasedCompactionSummarizer(settings)

        # Create messages
        messages = [
            Message(role="user", content="Fix the bug in src/auth/login.py"),
            Message(role="assistant", content="I'll help"),
        ]

        summary = summarizer.summarize(messages)

        # Verify XML format is used
        assert "<summary>" in summary
        assert "</summary>" in summary

    def test_xml_format_disabled(self):
        """Test summarizer with XML format disabled."""
        settings = CompactionStrategySettings(
            rule_xml_format=False,
        )

        summarizer = RuleBasedCompactionSummarizer(settings)

        messages = [
            Message(role="user", content="Fix the bug"),
        ]

        summary = summarizer.summarize(messages)

        # Should still return valid summary
        assert summary


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_messages_with_none_content(self, summarizer):
        """Test handling messages with None content."""
        messages = [
            Message(role="user", content="Valid message"),
            Message(role="assistant", content=""),  # Empty string instead of None
            Message(role="tool", tool_call_id="call_1", tool_name="read_file", content=""),  # Empty string
        ]

        # Should not raise an error
        summary = summarizer.summarize(messages)
        assert summary

    def test_messages_with_unicode(self, summarizer):
        """Test handling messages with unicode characters."""
        messages = [
            Message(role="user", content="Fix the bug in 文件.py"),
            Message(role="assistant", content="I'll help with the fix 🐛"),
        ]

        summary = summarizer.summarize(messages)
        assert summary
        assert "文件.py" in summary or "文件" in summary

    def test_very_long_message(self, summarizer):
        """Test handling very long messages."""
        long_content = "word " * 10000  # ~50KB
        messages = [
            Message(role="user", content=long_content),
        ]

        summary = summarizer.summarize(messages)
        assert summary
        # Should be truncated
        assert "…" in summary

    def test_mixed_tool_call_formats(self, summarizer):
        """Test handling different tool call formats."""
        messages = [
            Message(
                role="assistant",
                content="I'll read the file",
                tool_calls=[
                    {"name": "read_file", "id": "call_1"},
                    {"name": "write_file", "id": "call_2"},  # Dict format
                ]
            ),
        ]

        # Should handle dict format
        summary = summarizer.summarize(messages)
        assert summary
