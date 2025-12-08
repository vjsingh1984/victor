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

"""Tests for slash_commands module."""

import io
import pytest
from unittest.mock import MagicMock, AsyncMock

from rich.console import Console

from victor.ui.slash_commands import SlashCommand, SlashCommandHandler


class TestSlashCommand:
    """Tests for SlashCommand dataclass."""

    def test_basic_command(self):
        """Test creating a basic slash command."""
        handler = MagicMock()
        cmd = SlashCommand(
            name="test",
            description="A test command",
            handler=handler,
        )
        assert cmd.name == "test"
        assert cmd.description == "A test command"
        assert cmd.handler is handler
        assert cmd.aliases == []
        assert cmd.usage == "/test"

    def test_command_with_aliases(self):
        """Test command with aliases."""
        cmd = SlashCommand(
            name="help",
            description="Show help",
            handler=MagicMock(),
            aliases=["?", "h"],
        )
        assert cmd.aliases == ["?", "h"]

    def test_command_with_custom_usage(self):
        """Test command with custom usage string."""
        cmd = SlashCommand(
            name="model",
            description="Switch model",
            handler=MagicMock(),
            usage="/model [model_name]",
        )
        assert cmd.usage == "/model [model_name]"


class TestSlashCommandHandlerInit:
    """Tests for SlashCommandHandler initialization."""

    def test_init_with_defaults(self):
        """Test handler initialization with defaults."""
        console = Console(file=io.StringIO())
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.console is console
        assert handler.settings is settings
        assert handler.agent is None
        assert len(handler._commands) > 0  # Default commands registered

    def test_init_with_agent(self):
        """Test handler initialization with agent."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        agent = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings, agent=agent)

        assert handler.agent is agent

    def test_set_agent(self):
        """Test setting agent after initialization."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        agent = MagicMock()
        handler.set_agent(agent)

        assert handler.agent is agent


class TestSlashCommandHandlerDefaultCommands:
    """Tests for default command registration."""

    def test_help_command_registered(self):
        """Test help command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert "help" in handler._commands
        assert "?" in handler._commands  # alias

    def test_model_command_registered(self):
        """Test model command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert "model" in handler._commands
        assert "models" in handler._commands  # alias

    def test_clear_command_registered(self):
        """Test clear command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert "clear" in handler._commands
        assert "reset" in handler._commands  # alias

    def test_tools_command_registered(self):
        """Test tools command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert "tools" in handler._commands

    def test_exit_command_registered(self):
        """Test exit command is registered."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert "exit" in handler._commands
        assert "quit" in handler._commands  # alias


class TestSlashCommandHandlerRegister:
    """Tests for command registration."""

    def test_register_new_command(self):
        """Test registering a new command."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        custom_handler = MagicMock()
        cmd = SlashCommand(
            name="custom",
            description="Custom command",
            handler=custom_handler,
            aliases=["c"],
        )
        handler.register(cmd)

        assert "custom" in handler._commands
        assert "c" in handler._commands

    def test_register_overwrites_existing(self):
        """Test registering command with same name overwrites."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        handler1 = MagicMock()
        handler2 = MagicMock()

        cmd1 = SlashCommand(name="test", description="First", handler=handler1)
        cmd2 = SlashCommand(name="test", description="Second", handler=handler2)

        handler.register(cmd1)
        handler.register(cmd2)

        assert handler._commands["test"].handler is handler2


class TestSlashCommandHandlerIsCommand:
    """Tests for is_command method."""

    def test_is_command_true(self):
        """Test is_command returns True for slash commands."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.is_command("/help") is True
        assert handler.is_command("/model gpt-4") is True
        assert handler.is_command("  /clear") is True

    def test_is_command_false(self):
        """Test is_command returns False for non-commands."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        assert handler.is_command("hello") is False
        assert handler.is_command("help me") is False
        assert handler.is_command("") is False


class TestSlashCommandHandlerParseCommand:
    """Tests for parse_command method."""

    def test_parse_simple_command(self):
        """Test parsing simple command without args."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        name, args = handler.parse_command("/help")
        assert name == "help"
        assert args == []

    def test_parse_command_with_args(self):
        """Test parsing command with arguments."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        name, args = handler.parse_command("/model gpt-4")
        assert name == "model"
        assert args == ["gpt-4"]

    def test_parse_command_multiple_args(self):
        """Test parsing command with multiple arguments."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        name, args = handler.parse_command("/save my_session --force")
        assert name == "save"
        assert args == ["my_session", "--force"]

    def test_parse_command_with_whitespace(self):
        """Test parsing command with extra whitespace."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        name, args = handler.parse_command("  /help  ")
        assert name == "help"


class TestSlashCommandHandlerExecute:
    """Tests for execute method."""

    @pytest.mark.asyncio
    async def test_execute_known_command(self):
        """Test executing a known command."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        result = await handler.execute("/help")
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_unknown_command(self):
        """Test executing an unknown command still returns True (handled)."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        result = await handler.execute("/nonexistent")
        # Unknown commands are still "handled" (error message shown)
        assert result is True
        assert "unknown" in stdout.getvalue().lower() or "nonexistent" in stdout.getvalue().lower()

    @pytest.mark.asyncio
    async def test_execute_command_by_alias(self):
        """Test executing command using alias."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        result = await handler.execute("/?")  # Alias for help
        assert result is True

    @pytest.mark.asyncio
    async def test_execute_async_command(self):
        """Test executing an async command handler."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        async_handler = AsyncMock()
        cmd = SlashCommand(
            name="async_test",
            description="Async test command",
            handler=async_handler,
        )
        handler.register(cmd)

        await handler.execute("/async_test arg1 arg2")
        async_handler.assert_called_once_with(["arg1", "arg2"])


class TestSlashCommandHandlerHelp:
    """Tests for help command."""

    @pytest.mark.asyncio
    async def test_help_shows_commands(self):
        """Test help command shows available commands."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/help")
        output = stdout.getvalue()

        assert "help" in output.lower() or "commands" in output.lower()

    @pytest.mark.asyncio
    async def test_help_specific_command(self):
        """Test help for specific command."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/help model")
        output = stdout.getvalue()

        # Should show model command info
        assert "model" in output.lower()


class TestSlashCommandHandlerClear:
    """Tests for clear command."""

    @pytest.mark.asyncio
    async def test_clear_with_agent(self):
        """Test clear command with agent."""
        console = Console(file=io.StringIO())
        settings = MagicMock()
        agent = MagicMock()
        agent.reset_conversation = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings, agent=agent)

        await handler.execute("/clear")
        agent.reset_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_without_agent(self):
        """Test clear command without agent."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/clear")
        # Should complete without error


class TestSlashCommandHandlerTools:
    """Tests for tools command."""

    @pytest.mark.asyncio
    async def test_tools_without_agent(self):
        """Test tools command without agent shows message."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/tools")
        # Should complete without error

    @pytest.mark.asyncio
    async def test_tools_with_agent(self):
        """Test tools command with agent shows tool list."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        agent = MagicMock()
        agent.get_tools = MagicMock(
            return_value=[
                MagicMock(name="read_file", description="Read a file"),
                MagicMock(name="write_file", description="Write a file"),
            ]
        )

        handler = SlashCommandHandler(console=console, settings=settings, agent=agent)

        await handler.execute("/tools")
        output = stdout.getvalue()

        assert "read_file" in output or "write_file" in output or "tools" in output.lower()


class TestSlashCommandHandlerExit:
    """Tests for exit command."""

    @pytest.mark.asyncio
    async def test_exit_raises_exception(self):
        """Test exit command raises SystemExit."""
        console = Console(file=io.StringIO())
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        with pytest.raises(SystemExit):
            await handler.execute("/exit")


class TestSlashCommandHandlerStatus:
    """Tests for status command."""

    @pytest.mark.asyncio
    async def test_status_without_agent(self):
        """Test status command without agent."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()

        handler = SlashCommandHandler(console=console, settings=settings)

        await handler.execute("/status")
        # Should complete without error

    @pytest.mark.asyncio
    async def test_status_with_agent(self):
        """Test status command with agent shows info."""
        stdout = io.StringIO()
        console = Console(file=stdout, force_terminal=False)
        settings = MagicMock()
        agent = MagicMock()
        agent.provider = MagicMock()
        agent.provider.name = "anthropic"
        agent.settings = MagicMock()
        agent.settings.model = "claude-3"

        handler = SlashCommandHandler(console=console, settings=settings, agent=agent)

        await handler.execute("/status")
        output = stdout.getvalue()

        # Should show some status info
        assert len(output) > 0
