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
