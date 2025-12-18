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

"""Tests for session management commands - achieving 70%+ coverage."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from victor.ui.commands.session import SessionCommands
from victor.ui.commands.base import CommandContext


class TestSessionCommands:
    """Tests for SessionCommands class."""

    @pytest.fixture
    def session_commands(self):
        """Create SessionCommands instance."""
        return SessionCommands()

    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = MagicMock()
        ctx.agent.messages = [{"role": "user", "content": "hello"}]
        ctx.agent.model = "test-model"
        ctx.agent.provider_name = "test-provider"
        ctx.agent._messages = list(ctx.agent.messages)
        ctx.agent._system_added = True
        ctx.console = MagicMock()
        return ctx

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock session manager."""
        manager = MagicMock()
        return manager

    def test_group_name(self, session_commands):
        """Test group_name property."""
        assert session_commands.group_name == "session"

    def test_group_description(self, session_commands):
        """Test group_description property."""
        assert session_commands.group_description == "Manage conversation sessions"

    def test_get_commands_returns_five_commands(self, session_commands):
        """Test get_commands returns 5 commands."""
        commands = session_commands.get_commands()
        assert len(commands) == 5

    def test_get_commands_names(self, session_commands):
        """Test command names."""
        commands = session_commands.get_commands()
        command_names = [cmd.name for cmd in commands]
        assert "save" in command_names
        assert "load" in command_names
        assert "sessions" in command_names
        assert "resume" in command_names
        assert "clear" in command_names

    def test_sessions_command_has_alias(self, session_commands):
        """Test sessions command has 'ls' alias."""
        commands = session_commands.get_commands()
        sessions_cmd = next(c for c in commands if c.name == "sessions")
        assert "ls" in sessions_cmd.aliases


class TestSaveCommand:
    """Tests for save command."""

    @pytest.fixture
    def session_commands(self):
        """Create SessionCommands instance."""
        return SessionCommands()

    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = MagicMock()
        ctx.agent.messages = [{"role": "user", "content": "hello"}]
        ctx.agent.model = "test-model"
        ctx.agent.provider_name = "test-provider"
        return ctx

    def test_save_no_agent(self, session_commands):
        """Test save with no active agent."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = None

        session_commands._cmd_save(ctx, [])

        ctx.print_error.assert_called_once_with("No active session to save")

    @patch("victor.agent.session.get_session_manager")
    def test_save_no_session_manager(self, mock_get_manager, session_commands, mock_context):
        """Test save when session manager not available."""
        mock_get_manager.return_value = None

        session_commands._cmd_save(mock_context, [])

        mock_context.print_error.assert_called_once_with("Session manager not available")

    @patch("victor.agent.session.get_session_manager")
    def test_save_success_without_name(self, mock_get_manager, session_commands, mock_context):
        """Test successful save without name."""
        mock_manager = MagicMock()
        mock_manager.save_session.return_value = "session-123"
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_save(mock_context, [])

        mock_manager.save_session.assert_called_once_with(
            messages=mock_context.agent.messages,
            model=mock_context.agent.model,
            provider_name=mock_context.agent.provider_name,
            name=None,
        )
        mock_context.print_success.assert_called_once_with("Session saved: session-123")

    @patch("victor.agent.session.get_session_manager")
    def test_save_success_with_name(self, mock_get_manager, session_commands, mock_context):
        """Test successful save with name."""
        mock_manager = MagicMock()
        mock_manager.save_session.return_value = "session-456"
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_save(mock_context, ["my-session"])

        mock_manager.save_session.assert_called_once_with(
            messages=mock_context.agent.messages,
            model=mock_context.agent.model,
            provider_name=mock_context.agent.provider_name,
            name="my-session",
        )
        mock_context.print_success.assert_called_once_with("Session saved: session-456")

    @patch("victor.agent.session.get_session_manager")
    def test_save_exception(self, mock_get_manager, session_commands, mock_context):
        """Test save handling exception."""
        mock_manager = MagicMock()
        mock_manager.save_session.side_effect = Exception("Save failed")
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_save(mock_context, [])

        mock_context.print_error.assert_called_once()
        assert "Failed to save session" in mock_context.print_error.call_args[0][0]


class TestLoadCommand:
    """Tests for load command."""

    @pytest.fixture
    def session_commands(self):
        """Create SessionCommands instance."""
        return SessionCommands()

    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = MagicMock()
        ctx.agent._messages = []
        return ctx

    def test_load_no_args(self, session_commands, mock_context):
        """Test load without session_id argument."""
        session_commands._cmd_load(mock_context, [])

        mock_context.print_error.assert_called_once_with("Usage: /load <session_id>")

    @patch("victor.agent.session.get_session_manager")
    def test_load_no_session_manager(self, mock_get_manager, session_commands, mock_context):
        """Test load when session manager not available."""
        mock_get_manager.return_value = None

        session_commands._cmd_load(mock_context, ["session-123"])

        mock_context.print_error.assert_called_once_with("Session manager not available")

    @patch("victor.agent.session.get_session_manager")
    def test_load_session_not_found(self, mock_get_manager, session_commands, mock_context):
        """Test load when session not found."""
        mock_manager = MagicMock()
        mock_manager.load_session.return_value = None
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_load(mock_context, ["nonexistent"])

        mock_context.print_error.assert_called_once_with("Session not found: nonexistent")

    @patch("victor.agent.session.get_session_manager")
    def test_load_success_with_agent(self, mock_get_manager, session_commands, mock_context):
        """Test successful load with active agent."""
        mock_session = MagicMock()
        mock_session.name = "My Session"
        mock_session.session_id = "session-123"
        mock_session.messages = [{"role": "user", "content": "test"}]
        mock_session.model = "gpt-4"

        mock_manager = MagicMock()
        mock_manager.load_session.return_value = mock_session
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_load(mock_context, ["session-123"])

        assert mock_context.agent._messages == list(mock_session.messages)
        mock_context.print_success.assert_called_once_with("Loaded session: My Session")
        mock_context.print.assert_any_call("  Messages: 1")
        mock_context.print.assert_any_call("  Model: gpt-4")

    @patch("victor.agent.session.get_session_manager")
    def test_load_success_without_name(self, mock_get_manager, session_commands, mock_context):
        """Test successful load when session has no name."""
        mock_session = MagicMock()
        mock_session.name = None
        mock_session.session_id = "session-789"
        mock_session.messages = []
        mock_session.model = "claude-3"

        mock_manager = MagicMock()
        mock_manager.load_session.return_value = mock_session
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_load(mock_context, ["session-789"])

        mock_context.print_success.assert_called_once_with("Loaded session: session-789")

    @patch("victor.agent.session.get_session_manager")
    def test_load_no_agent(self, mock_get_manager, session_commands):
        """Test load without active agent."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = None

        mock_session = MagicMock()
        mock_session.name = "Test"
        mock_session.messages = []
        mock_session.model = "test"

        mock_manager = MagicMock()
        mock_manager.load_session.return_value = mock_session
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_load(ctx, ["session-123"])

        ctx.print_error.assert_called_once_with("No agent to load session into")

    @patch("victor.agent.session.get_session_manager")
    def test_load_exception(self, mock_get_manager, session_commands, mock_context):
        """Test load handling exception."""
        mock_manager = MagicMock()
        mock_manager.load_session.side_effect = Exception("Load failed")
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_load(mock_context, ["session-123"])

        mock_context.print_error.assert_called_once()
        assert "Failed to load session" in mock_context.print_error.call_args[0][0]


class TestSessionsCommand:
    """Tests for sessions (list) command."""

    @pytest.fixture
    def session_commands(self):
        """Create SessionCommands instance."""
        return SessionCommands()

    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        ctx = MagicMock(spec=CommandContext)
        ctx.console = MagicMock()
        return ctx

    @patch("victor.agent.session.get_session_manager")
    def test_sessions_no_manager(self, mock_get_manager, session_commands, mock_context):
        """Test sessions when manager not available."""
        mock_get_manager.return_value = None

        session_commands._cmd_sessions(mock_context, [])

        mock_context.print_error.assert_called_once_with("Session manager not available")

    @patch("victor.agent.session.get_session_manager")
    def test_sessions_empty_list(self, mock_get_manager, session_commands, mock_context):
        """Test sessions with no saved sessions."""
        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = []
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_sessions(mock_context, [])

        mock_context.print.assert_called_once_with("[dim]No saved sessions[/dim]")

    @patch("victor.agent.session.get_session_manager")
    def test_sessions_list_success(self, mock_get_manager, session_commands, mock_context):
        """Test successful sessions listing."""
        mock_session1 = MagicMock()
        mock_session1.session_id = "12345678abcd"
        mock_session1.name = "Session 1"
        mock_session1.model = "gpt-4"
        mock_session1.messages = [{"role": "user", "content": "hi"}]
        mock_session1.created_at = datetime(2025, 1, 15, 10, 30)

        mock_session2 = MagicMock()
        mock_session2.session_id = "abcdef123456"
        mock_session2.name = None
        mock_session2.model = None
        mock_session2.messages = []
        mock_session2.created_at = None

        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = [mock_session1, mock_session2]
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_sessions(mock_context, [])

        mock_manager.list_sessions.assert_called_once_with(limit=20)
        # Verify table was printed to console
        mock_context.console.print.assert_called_once()

    @patch("victor.agent.session.get_session_manager")
    def test_sessions_exception(self, mock_get_manager, session_commands, mock_context):
        """Test sessions handling exception."""
        mock_manager = MagicMock()
        mock_manager.list_sessions.side_effect = Exception("List failed")
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_sessions(mock_context, [])

        mock_context.print_error.assert_called_once()
        assert "Failed to list sessions" in mock_context.print_error.call_args[0][0]


class TestResumeCommand:
    """Tests for resume command."""

    @pytest.fixture
    def session_commands(self):
        """Create SessionCommands instance."""
        return SessionCommands()

    @pytest.fixture
    def mock_context(self):
        """Create mock command context."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = MagicMock()
        ctx.agent._messages = []
        return ctx

    @patch("victor.agent.session.get_session_manager")
    def test_resume_no_manager(self, mock_get_manager, session_commands, mock_context):
        """Test resume when manager not available."""
        mock_get_manager.return_value = None

        session_commands._cmd_resume(mock_context, [])

        mock_context.print_error.assert_called_once_with("Session manager not available")

    @patch("victor.agent.session.get_session_manager")
    def test_resume_no_sessions(self, mock_get_manager, session_commands, mock_context):
        """Test resume with no sessions to resume."""
        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = []
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_resume(mock_context, [])

        mock_context.print.assert_called_once_with("[dim]No sessions to resume[/dim]")

    @patch("victor.agent.session.get_session_manager")
    def test_resume_success_with_name(self, mock_get_manager, session_commands, mock_context):
        """Test successful resume with named session."""
        mock_session = MagicMock()
        mock_session.name = "Last Session"
        mock_session.session_id = "session-abc"
        mock_session.messages = [{"role": "user", "content": "test"}]

        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = [mock_session]
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_resume(mock_context, [])

        mock_manager.list_sessions.assert_called_once_with(limit=1)
        assert mock_context.agent._messages == list(mock_session.messages)
        mock_context.print_success.assert_called_once_with("Resumed session: Last Session")

    @patch("victor.agent.session.get_session_manager")
    def test_resume_success_without_name(self, mock_get_manager, session_commands, mock_context):
        """Test successful resume with unnamed session."""
        mock_session = MagicMock()
        mock_session.name = None
        mock_session.session_id = "12345678abcdef"
        mock_session.messages = []

        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = [mock_session]
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_resume(mock_context, [])

        # Should use truncated session_id
        mock_context.print_success.assert_called_once_with("Resumed session: 12345678")

    @patch("victor.agent.session.get_session_manager")
    def test_resume_no_agent(self, mock_get_manager, session_commands):
        """Test resume without active agent."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = None

        mock_session = MagicMock()
        mock_session.name = "Test"
        mock_session.messages = []

        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = [mock_session]
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_resume(ctx, [])

        ctx.print_error.assert_called_once_with("No agent to resume session into")

    @patch("victor.agent.session.get_session_manager")
    def test_resume_exception(self, mock_get_manager, session_commands, mock_context):
        """Test resume handling exception."""
        mock_manager = MagicMock()
        mock_manager.list_sessions.side_effect = Exception("Resume failed")
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_resume(mock_context, [])

        mock_context.print_error.assert_called_once()
        assert "Failed to resume session" in mock_context.print_error.call_args[0][0]


class TestClearCommand:
    """Tests for clear command."""

    @pytest.fixture
    def session_commands(self):
        """Create SessionCommands instance."""
        return SessionCommands()

    def test_clear_no_agent(self, session_commands):
        """Test clear with no active agent."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = None

        session_commands._cmd_clear(ctx, [])

        ctx.print_error.assert_called_once_with("No active session to clear")

    def test_clear_success(self, session_commands):
        """Test successful clear."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = MagicMock()
        ctx.agent.messages = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        ctx.agent._messages = MagicMock()
        ctx.agent._system_added = True

        session_commands._cmd_clear(ctx, [])

        ctx.agent._messages.clear.assert_called_once()
        assert ctx.agent._system_added is False
        ctx.print_success.assert_called_once()
        assert "Cleared 3 messages" in ctx.print_success.call_args[0][0]

    def test_clear_empty_history(self, session_commands):
        """Test clear with empty history."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = MagicMock()
        ctx.agent.messages = []
        ctx.agent._messages = MagicMock()
        ctx.agent._system_added = False

        session_commands._cmd_clear(ctx, [])

        ctx.agent._messages.clear.assert_called_once()
        ctx.print_success.assert_called_once_with("Cleared 0 messages from history")


class TestCommandProperties:
    """Tests for command properties and metadata."""

    @pytest.fixture
    def session_commands(self):
        """Create SessionCommands instance."""
        return SessionCommands()

    def test_save_command_usage(self, session_commands):
        """Test save command usage string."""
        commands = session_commands.get_commands()
        save_cmd = next(c for c in commands if c.name == "save")
        assert save_cmd.usage == "/save [name]"

    def test_load_command_usage(self, session_commands):
        """Test load command usage string."""
        commands = session_commands.get_commands()
        load_cmd = next(c for c in commands if c.name == "load")
        assert load_cmd.usage == "/load <session_id>"

    def test_all_commands_have_group(self, session_commands):
        """Test all commands have correct group."""
        commands = session_commands.get_commands()
        for cmd in commands:
            assert cmd.group == "session"

    def test_all_commands_have_handlers(self, session_commands):
        """Test all commands have handlers."""
        commands = session_commands.get_commands()
        for cmd in commands:
            assert callable(cmd.handler)

    def test_all_commands_have_descriptions(self, session_commands):
        """Test all commands have descriptions."""
        commands = session_commands.get_commands()
        for cmd in commands:
            assert cmd.description
            assert len(cmd.description) > 0


class TestEdgeCases:
    """Edge case tests for session commands."""

    @pytest.fixture
    def session_commands(self):
        """Create SessionCommands instance."""
        return SessionCommands()

    @patch("victor.agent.session.get_session_manager")
    def test_save_with_multiple_args(self, mock_get_manager, session_commands):
        """Test save only uses first arg as name."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = MagicMock()
        ctx.agent.messages = []
        ctx.agent.model = "test"
        ctx.agent.provider_name = "test"

        mock_manager = MagicMock()
        mock_manager.save_session.return_value = "id"
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_save(ctx, ["name1", "name2", "name3"])

        mock_manager.save_session.assert_called_once()
        call_kwargs = mock_manager.save_session.call_args[1]
        assert call_kwargs["name"] == "name1"

    @patch("victor.agent.session.get_session_manager")
    def test_load_with_partial_id(self, mock_get_manager, session_commands):
        """Test load with partial session ID."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = MagicMock()
        ctx.agent._messages = []

        mock_session = MagicMock()
        mock_session.name = "Test"
        mock_session.messages = []
        mock_session.model = "test"

        mock_manager = MagicMock()
        mock_manager.load_session.return_value = mock_session
        mock_get_manager.return_value = mock_manager

        session_commands._cmd_load(ctx, ["abc"])

        mock_manager.load_session.assert_called_once_with("abc")

    @patch("victor.agent.session.get_session_manager")
    def test_sessions_with_args_ignored(self, mock_get_manager, session_commands):
        """Test sessions command ignores extra args."""
        ctx = MagicMock(spec=CommandContext)
        ctx.console = MagicMock()

        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = []
        mock_get_manager.return_value = mock_manager

        # Extra args should be ignored
        session_commands._cmd_sessions(ctx, ["extra", "args"])

        mock_manager.list_sessions.assert_called_once_with(limit=20)

    @patch("victor.agent.session.get_session_manager")
    def test_resume_with_args_ignored(self, mock_get_manager, session_commands):
        """Test resume command ignores extra args."""
        ctx = MagicMock(spec=CommandContext)
        ctx.agent = MagicMock()
        ctx.agent._messages = []

        mock_session = MagicMock()
        mock_session.name = "Test"
        mock_session.session_id = "12345678"
        mock_session.messages = []

        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = [mock_session]
        mock_get_manager.return_value = mock_manager

        # Extra args should be ignored
        session_commands._cmd_resume(ctx, ["extra", "args"])

        mock_manager.list_sessions.assert_called_once_with(limit=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
