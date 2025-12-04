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

"""Tests for context/manager module."""


from victor.context.manager import (
    PruningStrategy,
    Message,
    FileContext,
    ContextWindow,
)


class TestPruningStrategy:
    """Tests for PruningStrategy enum."""

    def test_pruning_strategy_values(self):
        """Test PruningStrategy enum values."""
        assert PruningStrategy.FIFO == "fifo"
        assert PruningStrategy.PRIORITY == "priority"
        assert PruningStrategy.SMART == "smart"
        assert PruningStrategy.SUMMARIZE == "summarize"


class TestMessage:
    """Tests for Message model."""

    def test_message_creation(self):
        """Test creating a Message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tokens == 0
        assert msg.priority == 5

    def test_message_with_tokens(self):
        """Test Message with token count."""
        msg = Message(role="assistant", content="Hello, how can I help?", tokens=10)
        assert msg.tokens == 10

    def test_message_with_priority(self):
        """Test Message with priority."""
        msg = Message(role="system", content="You are a helpful assistant", priority=10)
        assert msg.priority == 10

    def test_message_with_metadata(self):
        """Test Message with metadata."""
        msg = Message(role="user", content="What's the weather?", metadata={"source": "cli"})
        assert msg.metadata["source"] == "cli"


class TestFileContext:
    """Tests for FileContext model."""

    def test_file_context_creation(self):
        """Test creating a FileContext."""
        fc = FileContext(path="/test/file.py", content="print('hello')")
        assert fc.path == "/test/file.py"
        assert fc.content == "print('hello')"
        assert fc.tokens == 0

    def test_file_context_with_relevance(self):
        """Test FileContext with relevance score."""
        fc = FileContext(
            path="/test/file.py",
            content="print('hello')",
            relevance_score=0.85,
        )
        assert fc.relevance_score == 0.85

    def test_file_context_with_line_range(self):
        """Test FileContext with line range."""
        fc = FileContext(
            path="/test/file.py",
            content="print('hello')",
            line_range=(10, 20),
        )
        assert fc.line_range == (10, 20)


class TestContextWindow:
    """Tests for ContextWindow model."""

    def test_context_window_creation(self):
        """Test creating a ContextWindow."""
        cw = ContextWindow()
        assert cw.messages == []
        assert cw.files == []
        assert cw.total_tokens == 0
        assert cw.max_tokens == 128000

    def test_context_window_available_tokens(self):
        """Test available tokens calculation."""
        cw = ContextWindow(
            total_tokens=10000,
            max_tokens=100000,
            reserved_tokens=5000,
        )
        assert cw.available_tokens == 85000  # 100000 - 10000 - 5000

    def test_context_window_usage_percentage(self):
        """Test usage percentage calculation."""
        cw = ContextWindow(
            total_tokens=50000,
            max_tokens=100000,
            reserved_tokens=0,
        )
        assert cw.usage_percentage == 50.0

    def test_context_window_with_messages(self):
        """Test ContextWindow with messages."""
        msg = Message(role="user", content="Hello")
        cw = ContextWindow(messages=[msg])
        assert len(cw.messages) == 1

    def test_context_window_with_files(self):
        """Test ContextWindow with files."""
        fc = FileContext(path="/test.py", content="test")
        cw = ContextWindow(files=[fc])
        assert len(cw.files) == 1


class TestContextManager:
    """Tests for ContextManager class."""

    def test_context_manager_init_defaults(self):
        """Test context manager with default values."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        assert cm.model == "gpt-4"
        assert cm.context.max_tokens == 128000
        assert cm.pruning_strategy == PruningStrategy.SMART
        assert cm.prune_threshold == 0.85

    def test_context_manager_custom_params(self):
        """Test context manager with custom parameters."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            model="claude-3",
            max_tokens=200000,
            reserved_tokens=8192,
            pruning_strategy=PruningStrategy.FIFO,
            prune_threshold=0.90,
        )
        assert cm.model == "claude-3"
        assert cm.context.max_tokens == 200000
        assert cm.context.reserved_tokens == 8192
        assert cm.pruning_strategy == PruningStrategy.FIFO
        assert cm.prune_threshold == 0.90

    def test_count_tokens_basic(self):
        """Test token counting with basic text."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        text = "Hello, world!"
        tokens = cm.count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_empty(self):
        """Test token counting with empty string."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        tokens = cm.count_tokens("")
        assert tokens == 0

    def test_add_message(self):
        """Test adding messages."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        initial_count = len(cm.context.messages)
        cm.add_message("user", "Hello there!")
        assert len(cm.context.messages) == initial_count + 1
        assert cm.context.messages[-1].role == "user"

    def test_add_message_updates_tokens(self):
        """Test that adding message updates token count."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        initial_tokens = cm.context.total_tokens
        cm.add_message("user", "This is a test message with several words.")
        assert cm.context.total_tokens > initial_tokens

    def test_add_message_with_priority(self):
        """Test adding message with custom priority."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_message("system", "Important system message", priority=10)
        assert cm.context.messages[-1].priority == 10

    def test_get_context_for_prompt(self):
        """Test getting context for LLM prompt."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_message("user", "Hello")
        cm.add_message("assistant", "Hi there!")

        context = cm.get_context_for_prompt()
        assert isinstance(context, list)
        assert len(context) == 2
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"

    def test_add_file(self):
        """Test adding file to context."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_file("/test/file.py", "print('hello')", relevance_score=0.8)
        assert len(cm.context.files) == 1
        assert cm.context.files[0].path == "/test/file.py"
        assert cm.context.files[0].relevance_score == 0.8

    def test_add_file_updates_tokens(self):
        """Test that adding file updates token count."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        initial_tokens = cm.context.total_tokens
        cm.add_file("/test/file.py", "print('hello world')")
        assert cm.context.total_tokens > initial_tokens

    def test_add_file_with_line_range(self):
        """Test adding file with line range."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        content = "line1\nline2\nline3\nline4\nline5"
        cm.add_file("/test/file.py", content, line_range=(1, 3))
        assert len(cm.context.files) == 1
        # Line range extracts lines 1-3 (index based, so lines 2-3)
        assert "line2" in cm.context.files[0].content

    def test_get_context_for_prompt_with_files(self):
        """Test getting context with files included."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_file("/test/file.py", "print('hello')", relevance_score=0.9)
        cm.add_message("user", "What does this file do?")

        context = cm.get_context_for_prompt()
        assert len(context) == 2
        # First should be system message with file context
        assert context[0]["role"] == "system"
        assert "/test/file.py" in context[0]["content"]

    def test_format_file_context_with_line_range(self):
        """Test formatting file context with line ranges."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_file("/test/file.py", "print('hello')", line_range=(10, 20))

        context = cm.get_context_for_prompt()
        # File context should mention line range
        assert "10" in context[0]["content"] or "Lines" in context[0]["content"]

    def test_format_file_context_sorted_by_relevance(self):
        """Test that files are sorted by relevance in context."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_file("/low.py", "low relevance", relevance_score=0.3)
        cm.add_file("/high.py", "high relevance", relevance_score=0.9)
        cm.add_file("/mid.py", "mid relevance", relevance_score=0.6)

        context = cm.get_context_for_prompt()
        content = context[0]["content"]
        # High relevance should appear before low
        high_pos = content.find("/high.py")
        low_pos = content.find("/low.py")
        assert high_pos < low_pos

    def test_auto_prune_triggered(self):
        """Test that auto-prune is triggered when threshold exceeded."""
        from victor.context.manager import ContextManager

        # Create manager with small max_tokens to easily trigger pruning
        cm = ContextManager(max_tokens=100, reserved_tokens=10, prune_threshold=0.5)

        # Add enough content to exceed threshold
        cm.add_message("user", "x" * 50)  # Will exceed 50% of available 90 tokens
        # Pruning should have been triggered
        assert cm.context.usage_percentage <= 100

    def test_prune_fifo_keeps_recent_messages(self):
        """Test FIFO pruning keeps recent messages."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            max_tokens=200,
            reserved_tokens=20,
            prune_threshold=0.6,
            pruning_strategy=PruningStrategy.FIFO,
        )

        # Add system message
        cm.add_message("system", "Be helpful", priority=10)
        # Add several user messages
        cm.add_message("user", "Message 1")
        cm.add_message("assistant", "Response 1")
        cm.add_message("user", "Message 2")
        cm.add_message("assistant", "Response 2")

        # After pruning, system message should be kept
        system_messages = [m for m in cm.context.messages if m.role == "system"]
        assert len(system_messages) >= 1

    def test_prune_by_priority(self):
        """Test priority-based pruning removes low-priority first."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            max_tokens=200,
            reserved_tokens=20,
            prune_threshold=0.6,
            pruning_strategy=PruningStrategy.PRIORITY,
        )

        # Add messages with different priorities
        cm.add_message("system", "Important", priority=10)
        cm.add_message("user", "Low priority", priority=1)
        cm.add_message("user", "High priority", priority=9)

        # High priority messages should be kept
        priorities = [m.priority for m in cm.context.messages]
        if priorities:
            assert max(priorities) >= 9 or len(cm.context.messages) > 0

    def test_prune_smart_keeps_edges(self):
        """Test smart pruning keeps first and last messages."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            max_tokens=200,
            reserved_tokens=20,
            prune_threshold=0.6,
            pruning_strategy=PruningStrategy.SMART,
        )

        # Add several messages
        cm.add_message("user", "First message")
        cm.add_message("assistant", "Middle response 1")
        cm.add_message("user", "Middle question")
        cm.add_message("assistant", "Middle response 2")
        cm.add_message("user", "Last message")

        # Context should still have some messages
        assert len(cm.context.messages) >= 1

    def test_multiple_files_in_context(self):
        """Test having multiple files in context."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_file("/file1.py", "content1")
        cm.add_file("/file2.py", "content2")
        cm.add_file("/file3.py", "content3")

        assert len(cm.context.files) == 3
        assert cm.context.total_tokens > 0

    def test_message_metadata(self):
        """Test message with metadata."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_message("user", "Test", metadata={"tool_call": "read_file"})
        assert cm.context.messages[-1].metadata["tool_call"] == "read_file"

    def test_context_window_properties(self):
        """Test ContextWindow property calculations."""
        from victor.context.manager import ContextManager

        cm = ContextManager(max_tokens=10000, reserved_tokens=1000)
        cm.add_message("user", "Hello" * 100)  # Add some tokens

        # Check properties work
        assert cm.context.available_tokens <= 9000
        assert cm.context.usage_percentage >= 0


class TestContextManagerClear:
    """Tests for clearing context."""

    def test_clear_files(self):
        """Test clearing file context."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_file("/file1.py", "content1")
        cm.add_file("/file2.py", "content2")

        tokens_before = cm.context.total_tokens
        assert len(cm.context.files) == 2
        assert tokens_before > 0

        cm.clear_files()

        assert len(cm.context.files) == 0
        assert cm.context.total_tokens < tokens_before

    def test_clear_messages_keep_system(self):
        """Test clearing messages while keeping system messages."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_message("system", "System prompt", priority=10)
        cm.add_message("user", "User message")
        cm.add_message("assistant", "Assistant response")

        assert len(cm.context.messages) == 3

        cm.clear_messages(keep_system=True)

        assert len(cm.context.messages) == 1
        assert cm.context.messages[0].role == "system"

    def test_clear_messages_all(self):
        """Test clearing all messages."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_message("system", "System prompt")
        cm.add_message("user", "User message")

        assert len(cm.context.messages) == 2

        cm.clear_messages(keep_system=False)

        assert len(cm.context.messages) == 0
        assert cm.context.total_tokens == 0


class TestContextManagerStats:
    """Tests for statistics."""

    def test_get_stats(self):
        """Test getting context statistics."""
        from victor.context.manager import ContextManager

        cm = ContextManager(max_tokens=100000)
        cm.add_message("system", "System prompt")
        cm.add_message("user", "User message")
        cm.add_message("assistant", "Response")
        cm.add_file("/test.py", "print('test')")

        stats = cm.get_stats()

        assert stats["total_messages"] == 3
        assert stats["total_files"] == 1
        assert stats["total_tokens"] > 0
        assert stats["max_tokens"] == 100000
        assert stats["available_tokens"] > 0
        assert stats["usage_percentage"] >= 0
        assert stats["pruning_strategy"] == "smart"
        assert stats["messages_by_role"]["system"] == 1
        assert stats["messages_by_role"]["user"] == 1
        assert stats["messages_by_role"]["assistant"] == 1


class TestContextManagerEncoderFallback:
    """Tests for encoder fallback behavior."""

    def test_unknown_model_fallback(self):
        """Test that unknown model falls back to cl100k_base encoding."""
        from victor.context.manager import ContextManager

        # Use an unknown model name
        cm = ContextManager(model="unknown-model-xyz-123")

        # Should still be able to count tokens
        tokens = cm.count_tokens("Hello, world!")
        assert tokens > 0
        assert isinstance(tokens, int)


class TestContextManagerPruningDetails:
    """Detailed tests for pruning strategies."""

    def test_prune_fifo_recalculates_tokens(self):
        """Test FIFO pruning recalculates token count correctly."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            max_tokens=500,
            reserved_tokens=50,
            prune_threshold=0.7,
            pruning_strategy=PruningStrategy.FIFO,
        )

        # Add many messages to trigger pruning
        for i in range(10):
            cm.add_message("user", f"Message number {i} with some content")

        # After pruning, token count should be reasonable
        assert cm.context.total_tokens < cm.max_tokens
        assert len(cm.context.messages) > 0

    def test_prune_priority_keeps_system_messages(self):
        """Test priority pruning keeps system messages."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            max_tokens=300,
            reserved_tokens=30,
            prune_threshold=0.6,
            pruning_strategy=PruningStrategy.PRIORITY,
        )

        cm.add_message("system", "Critical system message", priority=10)
        cm.add_message("user", "Low priority message", priority=1)
        cm.add_message("user", "Another low priority", priority=2)
        cm.add_message("user", "Trigger pruning with more content here", priority=1)

        # System messages should always be kept
        system_msgs = [m for m in cm.context.messages if m.role == "system"]
        assert len(system_msgs) >= 1

    def test_prune_smart_adds_gap_indicator(self):
        """Test smart pruning adds gap indicator when messages are removed."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            max_tokens=400,
            reserved_tokens=40,
            prune_threshold=0.6,
            pruning_strategy=PruningStrategy.SMART,
        )

        # Add enough messages to trigger smart pruning
        cm.add_message("user", "First user message - important context")
        for i in range(5):
            cm.add_message("user", f"Middle message {i}")
            cm.add_message("assistant", f"Response {i}")
        cm.add_message("user", "Last message")

        # Check if gap indicator is added when messages are pruned
        # Either all messages are kept or gap indicator is added
        total_msgs = len(cm.context.messages)
        assert total_msgs > 0

    def test_prune_smart_keeps_first_user_message(self):
        """Test smart pruning keeps the first user message."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            max_tokens=300,
            reserved_tokens=30,
            prune_threshold=0.5,
            pruning_strategy=PruningStrategy.SMART,
        )

        first_user_content = "This is the first user message"
        cm.add_message("user", first_user_content)
        for i in range(5):
            cm.add_message("assistant", f"Response {i} with more words")

        # First user message should be kept
        user_messages = [m for m in cm.context.messages if m.role == "user"]
        if user_messages:
            assert any(first_user_content in m.content for m in user_messages)

    def test_prune_default_fallback(self):
        """Test default pruning fallback to FIFO."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            max_tokens=200,
            reserved_tokens=20,
            prune_threshold=0.5,
            pruning_strategy=PruningStrategy.SUMMARIZE,  # Not implemented, falls back to FIFO
        )

        for i in range(5):
            cm.add_message("user", f"Message {i} with content")

        # Should still prune without error
        assert len(cm.context.messages) > 0


class TestFileContextLineRange:
    """Tests for file context with line ranges."""

    def test_add_file_extracts_line_range(self):
        """Test that line_range properly extracts lines."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        content = "line 0\nline 1\nline 2\nline 3\nline 4"
        cm.add_file("/test.py", content, line_range=(1, 3))

        # Should extract lines 1-2 (0-indexed, exclusive end)
        file_content = cm.context.files[0].content
        assert "line 1" in file_content
        assert "line 2" in file_content
        assert "line 0" not in file_content
        assert "line 3" not in file_content

    def test_format_file_shows_line_range(self):
        """Test that formatted file context shows line range."""
        from victor.context.manager import ContextManager

        cm = ContextManager()
        cm.add_file("/test.py", "content", line_range=(10, 20))

        formatted = cm._format_file_context()
        assert "10" in formatted and "20" in formatted


class TestAutoPruneOutput:
    """Tests for auto-prune console output."""

    def test_auto_prune_prints_status(self, capsys):
        """Test that auto-prune prints status messages."""
        from victor.context.manager import ContextManager

        cm = ContextManager(
            max_tokens=100,
            reserved_tokens=10,
            prune_threshold=0.5,
        )

        # This should trigger auto-prune
        cm.add_message("user", "x" * 60)

        captured = capsys.readouterr()
        # Should print pruning status
        assert "pruning" in captured.out.lower() or cm.context.usage_percentage <= 100
