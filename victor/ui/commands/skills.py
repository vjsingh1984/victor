"""Skill management CLI commands.

Provides ``victor skill`` subcommands for listing, inspecting,
searching, creating, and removing skills.

Usage:
    victor skill list
    victor skill list --category coding
    victor skill info debug_test
    victor skill search "refactor"
    victor skill create my_skill --description "..." --tools "read,grep"
    victor skill remove my_skill
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

skills_app = typer.Typer(name="skill", help="Manage and discover skills")
console = Console()


def _get_skills_dir() -> str:
    """Return the user skills directory (~/.victor/skills/)."""
    from victor.framework.skills import USER_SKILLS_DIR

    return USER_SKILLS_DIR


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

    # Load from user YAML skills (~/.victor/skills/)
    try:
        registry.from_user_skills()
    except Exception:
        logger.debug("Could not load user skills", exc_info=True)

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


@skills_app.command("create")
def create_skill(
    name: str = typer.Argument(help="Skill name (e.g., analyze_logs)"),
    description: str = typer.Option(..., "--description", "-d", help="What this skill does"),
    prompt: str = typer.Option(
        ..., "--prompt", "-p", help="Prompt fragment injected when skill is active"
    ),
    tools: str = typer.Option(
        ..., "--tools", "-t", help="Comma-separated required tools (e.g., read,grep,shell)"
    ),
    category: str = typer.Option("custom", "--category", "-c", help="Skill category"),
    optional_tools: Optional[str] = typer.Option(
        None, "--optional-tools", help="Comma-separated optional tools"
    ),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    max_tool_calls: int = typer.Option(20, "--max-tool-calls", help="Max tool calls"),
) -> None:
    """Create a new user skill (saved as YAML in ~/.victor/skills/)."""
    skills_dir = _get_skills_dir()
    os.makedirs(skills_dir, exist_ok=True)

    skill_data = {
        "name": name,
        "description": description,
        "category": category,
        "prompt_fragment": prompt,
        "required_tools": [t.strip() for t in tools.split(",") if t.strip()],
    }

    if optional_tools:
        skill_data["optional_tools"] = [t.strip() for t in optional_tools.split(",") if t.strip()]

    if tags:
        skill_data["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

    skill_data["max_tool_calls"] = max_tool_calls

    filepath = os.path.join(skills_dir, f"{name}.yaml")
    with open(filepath, "w") as f:
        yaml.dump(skill_data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created skill:[/] {name}")
    console.print(f"[dim]Saved to:[/] {filepath}")
    console.print()
    console.print("The skill will appear in [cyan]victor skill list[/] immediately.")


@skills_app.command("remove")
def remove_skill(
    name: str = typer.Argument(help="Name of the user skill to remove"),
) -> None:
    """Remove a user-created skill (deletes the YAML file from ~/.victor/skills/)."""
    skills_dir = _get_skills_dir()

    for ext in (".yaml", ".yml"):
        filepath = os.path.join(skills_dir, f"{name}{ext}")
        if os.path.exists(filepath):
            os.remove(filepath)
            console.print(f"[green]Removed skill:[/] {name}")
            console.print(f"[dim]Deleted:[/] {filepath}")
            return

    console.print(f"[yellow]User skill '{name}' not found in {skills_dir}[/]")
