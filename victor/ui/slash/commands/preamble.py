"""Preamble slash command for system prompt management.

Commands:
- /preamble show        — Display current system prompt structure
- /preamble sections    — List all sections with enabled/disabled status
- /preamble toggle <name> — Enable/disable a specific section
- /preamble set <text>  — Inject custom preamble at top of prompt
- /preamble set-bottom <text> — Inject custom preamble at bottom
- /preamble list        — Show active preamble entries
- /preamble remove <n>  — Remove preamble at index
- /preamble clear       — Remove all preambles
- /preamble reset       — Reset to default prompt structure
- /preamble optimize    — Show prompt optimization status

Examples:
    /preamble show
    /preamble toggle tool_hints
    /preamble set "Always use verbose logging"
    /preamble optimize
"""

from __future__ import annotations

import logging

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax

from victor.framework.preamble import PreambleManager, PreamblePosition
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class PreambleCommand(BaseSlashCommand):
    """Manage system prompt sections and user preambles."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="preamble",
            description="Manage system prompt sections and user preambles",
            usage="/preamble [show|sections|toggle|set|set-bottom|list|remove|clear|reset|optimize]",
            aliases=["prompt", "sysprompt"],
            category="advanced",
            requires_agent=False,
        )

    def execute(self, ctx: CommandContext) -> None:
        args = ctx.args if ctx.args else []
        if not args:
            self._show_help(ctx)
            return

        subcommand = args[0].lower()
        sub_args = args[1:]

        # Get or create PreambleManager
        manager = self._get_preamble_manager(ctx)

        commands = {
            "show": self._cmd_show,
            "sections": self._cmd_sections,
            "toggle": self._cmd_toggle,
            "set": self._cmd_set,
            "set-bottom": self._cmd_set_bottom,
            "list": self._cmd_list,
            "remove": self._cmd_remove,
            "clear": self._cmd_clear,
            "reset": self._cmd_reset,
            "optimize": self._cmd_optimize,
            "help": self._show_help,
        }

        handler = commands.get(subcommand)
        if handler:
            handler(ctx, manager, sub_args)
        else:
            ctx.console.print(f"[red]Unknown preamble subcommand: {subcommand}[/]")
            self._show_help(ctx)

    def _show_help(self, ctx: CommandContext, *args) -> None:
        """Show help for preamble command."""
        help_text = Text()
        help_text.append("Preamble Commands:\n\n", style="bold")
        help_text.append("  /preamble show          ", style="cyan")
        help_text.append("Display current system prompt structure\n")
        help_text.append("  /preamble sections      ", style="cyan")
        help_text.append("List all sections with status\n")
        help_text.append("  /preamble toggle <name> ", style="cyan")
        help_text.append("Enable/disable a section\n")
        help_text.append("  /preamble set <text>    ", style="cyan")
        help_text.append("Add preamble at top\n")
        help_text.append("  /preamble set-bottom <text> ", style="cyan")
        help_text.append("Add preamble at bottom\n")
        help_text.append("  /preamble list          ", style="cyan")
        help_text.append("Show active preambles\n")
        help_text.append("  /preamble remove <n>    ", style="cyan")
        help_text.append("Remove preamble at index\n")
        help_text.append("  /preamble clear         ", style="cyan")
        help_text.append("Remove all preambles\n")
        help_text.append("  /preamble reset         ", style="cyan")
        help_text.append("Reset to defaults\n")
        help_text.append("  /preamble optimize      ", style="cyan")
        help_text.append("Show optimization status\n")
        ctx.console.print(Panel(help_text, title="Preamble Help"))

    def _cmd_show(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """Show current system prompt structure."""
        sections = manager.get_active_sections()
        if not sections:
            ctx.console.print("[yellow]No active sections found[/]")
            return

        table = Table(title="System Prompt Structure", box=None)
        table.add_column("Section", style="cyan", no_wrap=True)
        table.add_column("Priority", style="dim")
        table.add_column("Status", no_wrap=True)
        table.add_column("Preview", style="dim")

        for name, info in sorted(sections.items(), key=lambda x: x[1].priority):
            status = "[green]enabled[/]" if info.enabled else "[red]disabled[/]"
            evolved = " [blue]★[/]" if info.evolved else ""
            preview = info.content[:60].replace("\n", " ") if info.content else "[dim]empty[/]"
            table.add_row(
                f"{name}{evolved}",
                str(info.priority),
                status,
                preview,
            )

        ctx.console.print(table)

        # Show total character count
        total_chars = sum(len(s.full_content) for s in sections.values())
        ctx.console.print(f"\n[dim]Total: {len(sections)} sections, ~{total_chars} chars[/]")

    def _cmd_sections(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """List all sections with enable/disable status."""
        toggleable = manager.list_toggleable_sections()
        if not toggleable:
            ctx.console.print("[yellow]No toggleable sections found[/]")
            return

        table = Table(title="Toggleable Sections", box=None)
        table.add_column("Section", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Description")

        for name, desc in sorted(toggleable.items()):
            enabled = manager.is_section_enabled(name)
            status = "[green]enabled[/]" if enabled else "[red]disabled[/]"
            table.add_row(name, status, desc)

        ctx.console.print(table)
        ctx.console.print("\n[dim]Use /preamble toggle <name> to enable/disable[/]")

    def _cmd_toggle(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """Toggle a section on/off."""
        if not args:
            ctx.console.print("[red]Usage: /preamble toggle <section_name>[/]")
            return

        name = args[0]
        current = manager.is_section_enabled(name)
        result = manager.toggle_section(name, not current)

        if result:
            status = "enabled" if not current else "disabled"
            ctx.console.print(f"[green]Section '{name}' {status}[/]")
        else:
            ctx.console.print(f"[red]Could not toggle '{name}'. Not found or protected.[/]")

    def _cmd_set(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """Set a preamble at the top of the prompt."""
        if not args:
            ctx.console.print("[red]Usage: /preamble set <text>[/]")
            return

        text = " ".join(args)
        manager.set_preamble(text, position=PreamblePosition.TOP)
        ctx.console.print("[green]Preamble added at top of system prompt[/]")

    def _cmd_set_bottom(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """Set a preamble at the bottom of the prompt."""
        if not args:
            ctx.console.print("[red]Usage: /preamble set-bottom <text>[/]")
            return

        text = " ".join(args)
        manager.set_preamble(text, position=PreamblePosition.BOTTOM)
        ctx.console.print("[green]Preamble added at bottom of system prompt[/]")

    def _cmd_list(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """List active preambles."""
        preambles = manager.list_preambles()
        if not preambles:
            ctx.console.print("[yellow]No active preambles[/]")
            return

        table = Table(title="Active Preambles", box=None)
        table.add_column("#", style="dim")
        table.add_column("Position", style="cyan")
        table.add_column("Preview", style="dim")

        for i, entry in enumerate(preambles):
            position = entry.position.value
            if entry.target_section:
                position += f" ({entry.target_section})"
            preview = entry.text[:80].replace("\n", " ")
            table.add_row(str(i), position, preview)

        ctx.console.print(table)

    def _cmd_remove(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """Remove a preamble by index."""
        if not args or not args[0].isdigit():
            ctx.console.print("[red]Usage: /preamble remove <index>[/]")
            return

        index = int(args[0])
        result = manager.remove_preamble(index)
        if result:
            ctx.console.print(f"[green]Preamble at index {index} removed[/]")
        else:
            ctx.console.print(f"[red]Invalid index: {index}[/]")

    def _cmd_clear(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """Clear all preambles."""
        manager.clear_preambles()
        ctx.console.print("[green]All preambles cleared[/]")

    def _cmd_reset(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """Reset prompt to default state."""
        manager.reset()
        ctx.console.print("[green]Prompt reset to default state[/]")

    def _cmd_optimize(self, ctx: CommandContext, manager: PreambleManager, args: list) -> None:
        """Show prompt optimization status."""
        status = manager.get_optimization_status()

        table = Table(title="Prompt Optimization Status", box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Enabled", "[green]Yes[/]" if status.enabled else "[red]No[/]")
        table.add_row("Active Strategies", ", ".join(status.active_strategies) or "[dim]none[/]")
        table.add_row(
            "Evolved Sections",
            f"{status.evolved_sections} / {status.total_sections}",
        )
        table.add_row("Current Tier", status.current_tier)
        table.add_row(
            "Last Evolution",
            status.last_evolution or "[dim]never[/]",
        )

        ctx.console.print(table)

    def _get_preamble_manager(self, ctx: CommandContext) -> PreambleManager:
        """Get or create a PreambleManager from the context."""
        # Try to get from context first (shared across session)
        if hasattr(ctx, "_preamble_manager"):
            return ctx._preamble_manager

        # Create new with prompt builder from orchestrator
        try:
            prompt_builder = ctx.orchestrator.prompt_builder
            manager = PreambleManager(prompt_builder)
        except Exception:
            manager = PreambleManager()

        ctx._preamble_manager = manager
        return manager
