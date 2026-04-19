"""Slash commands for skill management in interactive chat.

Commands:
    /skills          — List available skills
    /skill <name>    — Activate a skill for the next message
    /skill off       — Disable auto-selection for this session
    /skill on        — Re-enable auto-selection
"""

from __future__ import annotations

import logging

from rich.table import Table

from victor.ui.slash.registry import register_command
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata

logger = logging.getLogger(__name__)


@register_command
class SkillsCommand(BaseSlashCommand):
    """List and manage skills in interactive chat."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="skills",
            description="List available skills and manage skill auto-selection",
            usage="/skills | /skill <name> | /skill off | /skill on",
            aliases=["skill"],
            category="skills",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        if not ctx.args:
            self._list_skills(ctx)
            return

        sub = ctx.args[0].lower()
        if sub == "list":
            self._list_skills(ctx)
        elif sub == "off":
            self._disable(ctx)
        elif sub == "on":
            self._enable(ctx)
        else:
            self._activate(ctx, sub)

    def _list_skills(self, ctx: CommandContext) -> None:
        """List all available skills."""
        matcher = getattr(ctx.agent, "_skill_matcher", None)
        if not matcher or not matcher._skills:
            ctx.console.print(
                "[yellow]No skills available. Skill auto-selection may be disabled.[/]"
            )
            return

        disabled = getattr(ctx.agent, "_skill_auto_disabled", False)
        status = "[red]OFF[/]" if disabled else "[green]ON[/]"

        table = Table(title=f"Skills ({len(matcher._skills)}) \u2014 auto-select: {status}")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Category", style="green")
        table.add_column("Phase", style="dim")
        table.add_column("Description")

        for name in sorted(matcher._skills):
            skill = matcher._skills[name]
            table.add_row(
                name,
                skill.category,
                getattr(skill, "phase", "action"),
                skill.description[:60],
            )

        ctx.console.print(table)
        ctx.console.print("[dim]Activate: /skill <name> | Toggle: /skill off|on[/]")

    def _activate(self, ctx: CommandContext, skill_name: str) -> None:
        """Manually activate a skill for the next message."""
        matcher = getattr(ctx.agent, "_skill_matcher", None)
        if not matcher:
            ctx.console.print("[red]Skill matcher not available[/]")
            return

        skill = matcher._skills.get(skill_name)
        if not skill:
            ctx.console.print(f"[red]Skill '{skill_name}' not found[/]")
            available = ", ".join(sorted(matcher._skills.keys()))
            ctx.console.print(f"[dim]Available: {available}[/]")
            return

        ctx.agent.inject_skill(skill)
        ctx.agent._manual_skill_active = True
        ctx.console.print(f"[green]\U0001f3af Activated:[/] [cyan]{skill_name}[/]")
        ctx.console.print(f"[dim]{skill.description}[/]")

    def _disable(self, ctx: CommandContext) -> None:
        """Disable auto-selection for this session."""
        ctx.agent._skill_auto_disabled = True
        ctx.console.print("[yellow]Skill auto-selection disabled for this session[/]")
        ctx.console.print("[dim]Re-enable with /skill on[/]")

    def _enable(self, ctx: CommandContext) -> None:
        """Re-enable auto-selection."""
        ctx.agent._skill_auto_disabled = False
        ctx.console.print("[green]Skill auto-selection re-enabled[/]")
