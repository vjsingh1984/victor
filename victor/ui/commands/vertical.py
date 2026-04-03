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

"""Vertical management commands."""

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from victor.core.verticals.registry_manager import (
    InstalledVertical,
    VerticalRegistryManager,
    VerticalRuntimeInvalidationReason,
)
from victor.ui.commands.scaffold import new_vertical

# Initialize components
vertical_app = typer.Typer(
    name="vertical",
    help="Manage verticals (DEPRECATED: use 'victor plugin' instead)",
    deprecated=True,
)
console = Console()

_DEPRECATION_MSG = (
    "[yellow]Warning:[/] 'victor vertical' is deprecated and will be removed in v0.8.0. "
    "Use 'victor plugin' instead — it shows both verticals and plugins in a unified view."
)


def _deprecation_notice() -> None:
    """Print deprecation notice."""
    console.print(_DEPRECATION_MSG)
    console.print()


def _handle_error(message: str, detail: Optional[str] = None) -> None:
    """Handle and display errors."""
    if detail:
        console.print(f"[red]Error:[/] {message}")
        console.print(f"[dim]{detail}[/]")
        console.print()
    else:
        console.print(f"[red]Error:[/] {message}")
        raise typer.Exit(1)


@vertical_app.command("list")
def list_verticals(
    source: str = typer.Option(
        "all",
        "--source",
        "-s",
        help="Filter by source: 'all', 'installed', 'builtin', 'available'",
    ),
    category: str = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (e.g., 'security', 'data', 'devops')",
    ),
    tags: str = typer.Option(
        None,
        "--tags",
        "-t",
        help="Filter by tags (comma-separated)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information including tools and workflows",
    ),
) -> None:
    """List vertical packages with optional filtering.

    .. deprecated:: 0.7.0
        Use ``victor plugin list --type vertical`` instead.

    Shows built-in, installed, and available verticals with filtering options.

    Examples:
        # List all verticals
        victor vertical list

        # List only installed
        victor vertical list --source installed

        # List by category
        victor vertical list --category security

        # List by tags
        victor vertical list --tags "security,scanning"
    """
    _deprecation_notice()
    manager = VerticalRegistryManager()

    with console.status("[bold blue]Loading vertical list..."):
        try:
            verticals = manager.list_verticals(source=source)
        except Exception as e:
            _handle_error("Failed to list verticals", str(e))
            return

    # Filter by category if provided
    if category:
        verticals = [v for v in verticals if v.metadata and v.metadata.category == category]

    # Filter by tags if provided
    if tags:
        tag_list = [t.strip().lower() for t in tags.split(",")]
        verticals = [
            v
            for v in verticals
            if v.metadata
            and any(tg in [mt.lower() for mt in v.metadata.tags] for tg in tag_list)
        ]

    if not verticals:
        console.print(f"[yellow]No verticals found matching source='{source}'[/]")
        return

    table = Table(title="Vertical Ecosystem")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="magenta")
    table.add_column("Source", style="green")
    table.add_column("Description")

    if verbose:
        table.add_column("Capabilities")
        table.add_column("Tools")

    for v in verticals:
        row = [
            v.name,
            v.version,
            "Builtin" if v.is_builtin else "External",
            v.metadata.description if v.metadata else "[dim]No description available[/]",
        ]

        if verbose:
            cs = v.metadata.class_spec if v.metadata else None
            caps = ", ".join(cs.provides_capabilities) if cs and cs.provides_capabilities else "-"
            tool_list = cs.provides_tools if cs and cs.provides_tools else []
            tools = ", ".join(tool_list[:5]) if tool_list else "-"
            if len(tool_list) > 5:
                tools += f" (+{len(tool_list) - 5} more)"
            row.extend([caps, tools])

        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]Found {len(verticals)} vertical(s)[/]")


@vertical_app.command("info")
def vertical_info(name: str) -> None:
    """Show detailed information for a vertical."""
    manager = VerticalRegistryManager()

    try:
        vertical = manager.get_info(name)
        if not vertical:
            _handle_error(f"Vertical '{name}' not found")
            return

        console.print(f"\n[bold cyan]Vertical: {vertical.name}[/]")
        console.print(f"[bold]Version:[/] {vertical.version}")
        console.print(f"[bold]Source:[/] {'Built-in' if vertical.is_builtin else 'External'}")
        console.print(f"[bold]Location:[/] {vertical.location or 'Embedded'}")

        if vertical.metadata:
            m = vertical.metadata
            console.print(f"\n[bold]Description:[/] {m.description}")
            if m.authors:
                author_names = [a.name if hasattr(a, "name") else str(a) for a in m.authors]
                console.print(f"[bold]Authors:[/] {', '.join(author_names)}")
            if m.category:
                console.print(f"[bold]Category:[/] {m.category}")
            if m.tags:
                console.print(f"[bold]Tags:[/] {', '.join(m.tags)}")

            if m.class_spec:
                if m.class_spec.provides_capabilities:
                    console.print("\n[bold]Capabilities:[/]")
                    for cap in m.class_spec.provides_capabilities:
                        console.print(f"  • {cap}")

                if m.class_spec.provides_tools:
                    console.print("\n[bold]Provides Tools:[/]")
                    for tool in m.class_spec.provides_tools:
                        console.print(f"  • {tool}")

                if m.class_spec.provides_workflows:
                    console.print("\n[bold]Provides Workflows:[/]")
                    for wf in m.class_spec.provides_workflows:
                        console.print(f"  • {wf}")
        else:
            console.print("\n[yellow]No detailed metadata available[/]")

    except Exception as e:
        _handle_error(f"Failed to load information for '{name}'", str(e))


@vertical_app.command("install")
def install_vertical(
    package: str = typer.Argument(..., help="Package name, path, or URL"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-installation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be installed"),
) -> None:
    """Install a new vertical plugin."""
    manager = VerticalRegistryManager()

    if dry_run:
        console.print(f"[yellow]Dry run:[/] Would install package '{package}'")
        return

    with console.status(f"[bold blue]Installing {package}..."):
        try:
            result = manager.install(package, force=force)
            if result:
                console.print(f"[green]Successfully installed {package}[/]")
            else:
                _handle_error(f"Failed to install {package}")
        except Exception as e:
            _handle_error("Installation failed", str(detail=str(e)))


@vertical_app.command("uninstall")
def uninstall_vertical(name: str) -> None:
    """Uninstall a vertical plugin."""
    manager = VerticalRegistryManager()

    with console.status(f"[bold blue]Uninstalling {name}..."):
        try:
            if manager.uninstall(name):
                console.print(f"[green]Successfully uninstalled {name}[/]")
            else:
                _handle_error(f"Failed to uninstall {name}")
        except Exception as e:
            _handle_error("Uninstallation failed", str(e))


@vertical_app.command("new")
def scaffold_vertical(
    name: str = typer.Argument(..., help="Name of the new vertical"),
    description: str = typer.Option("A new Victor vertical", "--description", "-d"),
    service_provider: bool = typer.Option(
        False, "--service", help="Include service provider boilerplate"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite if directory exists"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show files that would be created"),
) -> None:
    """Scaffold a new vertical package."""
    from victor.ui.commands.scaffold import new_vertical

    # Call the scaffold command
    new_vertical(
        name=name,
        description=description,
        service_provider=service_provider,
        force=force,
        dry_run=dry_run,
    )


__all__ = ["vertical_app"]
