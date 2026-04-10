"""Skill management CLI commands.

Provides ``victor skill`` subcommands for listing, inspecting,
and searching skills across verticals and plugins.

Usage:
    victor skill list
    victor skill list --category coding
    victor skill info debug_test
    victor skill search "refactor"
"""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

skills_app = typer.Typer(name="skill", help="Manage and discover skills")
console = Console()


def _build_registry():
    """Build a SkillRegistry populated from verticals and entry points."""
    from victor.framework.skills import SkillRegistry

    registry = SkillRegistry()

    # Load from discovered verticals via VerticalLoader
    try:
        from victor.core.verticals.vertical_loader import VerticalLoader

        loader = VerticalLoader()
        loader.discover_verticals()
        for _name, vertical_cls in loader._discovered_verticals.items():
            if hasattr(vertical_cls, "get_skills"):
                try:
                    registry.from_vertical(vertical_cls)
                except Exception:
                    logger.debug("Failed to load skills from %s", _name)
    except Exception:
        logger.debug("Could not load verticals for skill discovery", exc_info=True)

    # Load from entry points
    try:
        registry.from_entry_points()
    except Exception:
        logger.debug("Could not load skills from entry points", exc_info=True)

    return registry


def _render_skills_table(skills, title: str = "Skills") -> None:
    """Render a list of skills as a rich table."""
    if not skills:
        console.print("[dim]No skills found.[/]")
        return

    table = Table(title=title, show_lines=False)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Description")
    table.add_column("Tools", style="dim")

    for skill in sorted(skills, key=lambda s: s.name):
        tool_count = len(skill.required_tools) + len(skill.optional_tools)
        table.add_row(
            skill.name,
            skill.category,
            skill.description,
            f"{tool_count} tools",
        )

    console.print(table)


@skills_app.command("list")
def list_skills(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (e.g., 'coding', 'devops', 'analysis')",
    ),
) -> None:
    """List all discovered skills."""
    registry = _build_registry()

    if category:
        skills = registry.search(category=category)
        title = f"Skills (category: {category})"
    else:
        skills = registry.list_all()
        title = "Skills"

    _render_skills_table(skills, title=title)


@skills_app.command("info")
def skill_info(
    name: str = typer.Argument(help="Skill name to inspect"),
) -> None:
    """Show detailed information about a skill."""
    registry = _build_registry()
    skill = registry.get_optional(name)

    if skill is None:
        console.print(f"[red]Skill '{name}' not found.[/]")
        return

    console.print(f"[bold cyan]{skill.name}[/] v{skill.version}")
    console.print(f"[dim]Category:[/] {skill.category}")
    console.print(f"[dim]Description:[/] {skill.description}")
    console.print()

    if skill.required_tools:
        console.print("[bold]Required tools:[/]")
        for tool in skill.required_tools:
            console.print(f"  - {tool}")

    if skill.optional_tools:
        console.print("[bold]Optional tools:[/]")
        for tool in skill.optional_tools:
            console.print(f"  - {tool}")

    if skill.tags:
        console.print(f"[dim]Tags:[/] {', '.join(sorted(skill.tags))}")

    console.print(f"[dim]Max tool calls:[/] {skill.max_tool_calls}")

    if skill.prompt_fragment:
        console.print()
        console.print("[bold]Prompt fragment:[/]")
        console.print(f"  {skill.prompt_fragment[:200]}")


@skills_app.command("search")
def search_skills(
    query: str = typer.Argument(help="Search query (matches name, description, tags)"),
) -> None:
    """Search skills by keyword."""
    registry = _build_registry()
    results = registry.search(query=query)
    _render_skills_table(results, title=f"Search: '{query}'")
