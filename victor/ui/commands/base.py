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

"""Base classes for slash command system.

This module provides the foundational classes for the modular command system:
- SlashCommand: Represents a single command
- CommandGroup: Groups related commands together
- CommandRegistry: Central registry for all command groups
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class SlashCommand:
    """Represents a slash command with its handler.

    Attributes:
        name: Primary command name (e.g., "help")
        description: Short description for help text
        handler: Function to execute (sync or async)
        aliases: Alternative names for the command
        usage: Usage string (e.g., "/help [command]")
        group: Command group name for organization
        hidden: If True, don't show in help listing
    """

    name: str
    description: str
    handler: Callable
    aliases: List[str] = field(default_factory=list)
    usage: Optional[str] = None
    group: str = "general"
    hidden: bool = False

    def __post_init__(self):
        if self.usage is None:
            self.usage = f"/{self.name}"


class CommandContext:
    """Context passed to command handlers.

    Provides access to common resources needed by commands.
    """

    def __init__(
        self,
        console: Console,
        settings: "Settings",
        agent: Optional["AgentOrchestrator"] = None,
    ):
        self.console = console
        self.settings = settings
        self.agent = agent

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print to console."""
        self.console.print(*args, **kwargs)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[red]Error:[/red] {message}")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[green]{message}[/green]")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[yellow]Warning:[/yellow] {message}")


class CommandGroup(ABC):
    """Base class for command groups.

    Subclass this to create related commands. Each group registers
    its commands with the central registry.

    Example:
        class SessionCommands(CommandGroup):
            @property
            def group_name(self) -> str:
                return "session"

            def get_commands(self) -> List[SlashCommand]:
                return [
                    SlashCommand("save", "Save session", self._cmd_save),
                    SlashCommand("load", "Load session", self._cmd_load),
                ]

            def _cmd_save(self, ctx: CommandContext, args: List[str]) -> None:
                ctx.print("Session saved!")
    """

    @property
    @abstractmethod
    def group_name(self) -> str:
        """Name of this command group."""
        ...

    @property
    def group_description(self) -> str:
        """Description of this command group."""
        return ""

    @abstractmethod
    def get_commands(self) -> List[SlashCommand]:
        """Return list of commands in this group."""
        ...


class CommandRegistry:
    """Central registry for all slash commands.

    Manages registration, lookup, and execution of commands.
    Supports both sync and async command handlers.
    """

    def __init__(self, console: Console, settings: "Settings"):
        self.console = console
        self.settings = settings
        self.agent: Optional["AgentOrchestrator"] = None
        self._commands: Dict[str, SlashCommand] = {}
        self._groups: Dict[str, CommandGroup] = {}
        self._alias_map: Dict[str, str] = {}

    def set_agent(self, agent: "AgentOrchestrator") -> None:
        """Set the agent reference for commands that need it."""
        self.agent = agent

    def register_group(self, group: CommandGroup) -> None:
        """Register a command group.

        Args:
            group: Command group to register
        """
        self._groups[group.group_name] = group
        for cmd in group.get_commands():
            self.register(cmd)

    def register(self, command: SlashCommand) -> None:
        """Register a single command.

        Args:
            command: Command to register
        """
        self._commands[command.name] = command
        for alias in command.aliases:
            self._alias_map[alias] = command.name
        logger.debug(f"Registered command: /{command.name}")

    def get(self, name: str) -> Optional[SlashCommand]:
        """Get a command by name or alias.

        Args:
            name: Command name or alias

        Returns:
            Command if found, None otherwise
        """
        # Check direct command name
        if name in self._commands:
            return self._commands[name]
        # Check aliases
        if name in self._alias_map:
            return self._commands[self._alias_map[name]]
        return None

    def is_command(self, text: str) -> bool:
        """Check if text is a slash command.

        Args:
            text: Input text to check

        Returns:
            True if text starts with / and matches a command
        """
        if not text.startswith("/"):
            return False
        parts = text[1:].split(None, 1)
        if not parts:
            return False
        return self.get(parts[0]) is not None

    async def execute(self, text: str) -> bool:
        """Execute a slash command.

        Args:
            text: Full command text (e.g., "/help model")

        Returns:
            True if command was executed, False otherwise
        """
        if not text.startswith("/"):
            return False

        parts = text[1:].split(None, 1)
        if not parts:
            return False

        cmd_name = parts[0].lower()
        args = parts[1].split() if len(parts) > 1 else []

        command = self.get(cmd_name)
        if command is None:
            self.console.print(f"[red]Unknown command:[/red] /{cmd_name}")
            self.console.print("Type [cyan]/help[/cyan] for available commands.")
            return True

        ctx = CommandContext(self.console, self.settings, self.agent)

        try:
            result = command.handler(ctx, args)
            if asyncio.iscoroutine(result):
                await result
            return True
        except Exception as e:
            logger.exception(f"Error executing /{cmd_name}")
            self.console.print(f"[red]Error executing /{cmd_name}:[/red] {e}")
            return True

    def list_commands(self, group: Optional[str] = None) -> List[SlashCommand]:
        """List all registered commands.

        Args:
            group: Filter by group name (optional)

        Returns:
            List of commands
        """
        commands = [c for c in self._commands.values() if not c.hidden]
        if group:
            commands = [c for c in commands if c.group == group]
        return sorted(commands, key=lambda c: (c.group, c.name))

    def list_groups(self) -> List[str]:
        """List all command groups.

        Returns:
            List of group names
        """
        groups = set(c.group for c in self._commands.values())
        return sorted(groups)

    def print_help(self, command_name: Optional[str] = None) -> None:
        """Print help for commands.

        Args:
            command_name: Specific command to get help for (optional)
        """
        if command_name:
            cmd = self.get(command_name)
            if cmd:
                self.console.print(f"[bold cyan]/{cmd.name}[/bold cyan]")
                self.console.print(f"  {cmd.description}")
                self.console.print(f"  Usage: [dim]{cmd.usage}[/dim]")
                if cmd.aliases:
                    self.console.print(f"  Aliases: [dim]{', '.join(cmd.aliases)}[/dim]")
            else:
                self.console.print(f"[red]Unknown command:[/red] {command_name}")
            return

        # Print all commands grouped
        table = Table(title="Available Commands", show_header=True)
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        table.add_column("Group", style="dim")

        for cmd in self.list_commands():
            table.add_row(f"/{cmd.name}", cmd.description, cmd.group)

        self.console.print(table)
        self.console.print("\nType [cyan]/help <command>[/cyan] for detailed help.")


# =============================================================================
# Global Registry
# =============================================================================

_registry: Optional[CommandRegistry] = None


def get_command_registry() -> Optional[CommandRegistry]:
    """Get the global command registry.

    Returns:
        Global registry if initialized, None otherwise
    """
    return _registry


def set_command_registry(registry: CommandRegistry) -> None:
    """Set the global command registry.

    Args:
        registry: Registry to use as global
    """
    global _registry
    _registry = registry


def reset_command_registry() -> None:
    """Reset the global command registry."""
    global _registry
    _registry = None
