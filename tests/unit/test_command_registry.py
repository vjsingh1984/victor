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

"""Tests for the command registry system."""

import pytest
from unittest.mock import MagicMock
from typing import List

from rich.console import Console

from victor.ui.commands.base import (
    SlashCommand,
    CommandContext,
    CommandGroup,
    CommandRegistry,
    get_command_registry,
    set_command_registry,
    reset_command_registry,
)


class MockSettings:
    """Mock settings for testing."""

    pass


class TestSlashCommand:
    """Tests for SlashCommand dataclass."""

    def test_basic_command(self):
        """Test basic command creation."""
        cmd = SlashCommand(
            name="test",
            description="A test command",
            handler=lambda ctx, args: None,
        )

        assert cmd.name == "test"
        assert cmd.description == "A test command"
        assert cmd.usage == "/test"
        assert cmd.aliases == []
        assert cmd.group == "general"
        assert not cmd.hidden

    def test_command_with_aliases(self):
        """Test command with aliases."""
        cmd = SlashCommand(
            name="help",
            description="Show help",
            handler=lambda ctx, args: None,
            aliases=["?", "h"],
        )

        assert cmd.aliases == ["?", "h"]

    def test_command_with_custom_usage(self):
        """Test command with custom usage string."""
        cmd = SlashCommand(
            name="load",
            description="Load session",
            handler=lambda ctx, args: None,
            usage="/load <session_id>",
        )

        assert cmd.usage == "/load <session_id>"


class TestCommandContext:
    """Tests for CommandContext."""

    def test_context_creation(self):
        """Test context creation."""
        console = Console()
        settings = MockSettings()

        ctx = CommandContext(console, settings)

        assert ctx.console is console
        assert ctx.settings is settings
        assert ctx.agent is None

    def test_context_with_agent(self):
        """Test context with agent."""
        console = Console()
        settings = MockSettings()
        agent = MagicMock()

        ctx = CommandContext(console, settings, agent)

        assert ctx.agent is agent


class TestCommandGroup:
    """Tests for CommandGroup base class."""

    def test_custom_group(self):
        """Test creating a custom command group."""

        class TestGroup(CommandGroup):
            @property
            def group_name(self) -> str:
                return "test"

            @property
            def group_description(self) -> str:
                return "Test commands"

            def get_commands(self) -> List[SlashCommand]:
                return [
                    SlashCommand("foo", "Foo command", lambda ctx, args: None, group="test"),
                    SlashCommand("bar", "Bar command", lambda ctx, args: None, group="test"),
                ]

        group = TestGroup()

        assert group.group_name == "test"
        assert group.group_description == "Test commands"
        assert len(group.get_commands()) == 2


class TestCommandRegistry:
    """Tests for CommandRegistry."""

    def setup_method(self):
        """Setup for each test."""
        self.console = Console()
        self.settings = MockSettings()
        self.registry = CommandRegistry(self.console, self.settings)
        reset_command_registry()

    def test_register_command(self):
        """Test registering a command."""
        cmd = SlashCommand("test", "Test command", lambda ctx, args: None)

        self.registry.register(cmd)

        assert self.registry.get("test") is cmd

    def test_register_command_with_aliases(self):
        """Test registering a command with aliases."""
        cmd = SlashCommand(
            "help",
            "Show help",
            lambda ctx, args: None,
            aliases=["?", "h"],
        )

        self.registry.register(cmd)

        assert self.registry.get("help") is cmd
        assert self.registry.get("?") is cmd
        assert self.registry.get("h") is cmd

    def test_register_group(self):
        """Test registering a command group."""

        class TestGroup(CommandGroup):
            @property
            def group_name(self) -> str:
                return "test"

            def get_commands(self) -> List[SlashCommand]:
                return [
                    SlashCommand("foo", "Foo", lambda ctx, args: None, group="test"),
                ]

        group = TestGroup()
        self.registry.register_group(group)

        assert self.registry.get("foo") is not None

    def test_get_unknown_command(self):
        """Test getting unknown command returns None."""
        assert self.registry.get("unknown") is None

    def test_is_command(self):
        """Test is_command detection."""
        cmd = SlashCommand("test", "Test", lambda ctx, args: None)
        self.registry.register(cmd)

        assert self.registry.is_command("/test")
        assert self.registry.is_command("/test arg1 arg2")
        assert not self.registry.is_command("/unknown")
        assert not self.registry.is_command("not a command")
        assert not self.registry.is_command("/")

    @pytest.mark.asyncio
    async def test_execute_sync_command(self):
        """Test executing a sync command."""
        executed = []

        def handler(ctx, args):
            executed.append(args)

        cmd = SlashCommand("test", "Test", handler)
        self.registry.register(cmd)

        result = await self.registry.execute("/test arg1 arg2")

        assert result is True
        assert executed == [["arg1", "arg2"]]

    @pytest.mark.asyncio
    async def test_execute_async_command(self):
        """Test executing an async command."""
        executed = []

        async def handler(ctx, args):
            executed.append(args)

        cmd = SlashCommand("test", "Test", handler)
        self.registry.register(cmd)

        result = await self.registry.execute("/test async")

        assert result is True
        assert executed == [["async"]]

    @pytest.mark.asyncio
    async def test_execute_unknown_command(self):
        """Test executing unknown command."""
        result = await self.registry.execute("/unknown")

        assert result is True  # Still returns True (command was processed)

    @pytest.mark.asyncio
    async def test_execute_not_a_command(self):
        """Test executing non-command text."""
        result = await self.registry.execute("not a command")

        assert result is False

    def test_list_commands(self):
        """Test listing commands."""
        cmd1 = SlashCommand("foo", "Foo", lambda ctx, args: None, group="a")
        cmd2 = SlashCommand("bar", "Bar", lambda ctx, args: None, group="b")
        cmd3 = SlashCommand("hidden", "Hidden", lambda ctx, args: None, hidden=True)

        self.registry.register(cmd1)
        self.registry.register(cmd2)
        self.registry.register(cmd3)

        commands = self.registry.list_commands()

        assert len(commands) == 2
        assert all(not c.hidden for c in commands)

    def test_list_commands_by_group(self):
        """Test listing commands filtered by group."""
        cmd1 = SlashCommand("foo", "Foo", lambda ctx, args: None, group="a")
        cmd2 = SlashCommand("bar", "Bar", lambda ctx, args: None, group="b")

        self.registry.register(cmd1)
        self.registry.register(cmd2)

        commands = self.registry.list_commands(group="a")

        assert len(commands) == 1
        assert commands[0].name == "foo"

    def test_list_groups(self):
        """Test listing command groups."""
        cmd1 = SlashCommand("foo", "Foo", lambda ctx, args: None, group="session")
        cmd2 = SlashCommand("bar", "Bar", lambda ctx, args: None, group="model")

        self.registry.register(cmd1)
        self.registry.register(cmd2)

        groups = self.registry.list_groups()

        assert "session" in groups
        assert "model" in groups

    def test_set_agent(self):
        """Test setting agent reference."""
        agent = MagicMock()

        self.registry.set_agent(agent)

        assert self.registry.agent is agent


class TestGlobalRegistry:
    """Tests for global registry management."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_command_registry()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_command_registry()

    def test_get_registry_returns_none_initially(self):
        """Test that get_command_registry returns None when not set."""
        assert get_command_registry() is None

    def test_set_and_get_registry(self):
        """Test setting and getting global registry."""
        console = Console()
        settings = MockSettings()
        registry = CommandRegistry(console, settings)

        set_command_registry(registry)

        assert get_command_registry() is registry

    def test_reset_registry(self):
        """Test resetting global registry."""
        console = Console()
        settings = MockSettings()
        registry = CommandRegistry(console, settings)

        set_command_registry(registry)
        reset_command_registry()

        assert get_command_registry() is None
