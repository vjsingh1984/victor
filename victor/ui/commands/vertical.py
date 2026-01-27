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

"""Vertical registry commands for managing third-party vertical packages.

This module provides CLI commands for:
- Installing verticals from PyPI, git, or local paths
- Listing available and installed verticals
- Searching the vertical registry
- Displaying detailed vertical information
- Uninstalling verticals
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from victor.core.verticals.registry_manager import (
    InstalledVertical,
    PackageSpec,
    VerticalRegistryManager,
)

vertical_app = typer.Typer(
    name="vertical",
    help="Manage vertical packages - install, list, search, and uninstall.",
)
console = Console()


@vertical_app.command("install")
def install_vertical(
    package: str = typer.Argument(
        ...,
        help="Package specification (e.g., 'victor-security', 'git+https://...', '/local/path')",
    ),
    no_validate: bool = typer.Option(
        False,
        "--no-validate",
        help="Skip validation checks before installation",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be installed without actually installing",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed installation output",
    ),
) -> None:
    """Install a vertical package from PyPI, git, or local path.

    Examples:
        # Install from PyPI
        victor vertical install victor-security

        # Install with version constraint
        victor vertical install "victor-security>=0.5.0"

        # Install from git
        victor vertical install "git+https://github.com/user/victor-security.git"

        # Install from local path
        victor vertical install ./path/to/package

        # Install with extras
        victor vertical install "victor-security[full]"

        # Dry run
        victor vertical install victor-security --dry-run
    """
    # Parse package spec
    try:
        spec = PackageSpec.parse(package)
    except ValueError as e:
        console.print(f"[red]Error: Invalid package specification: {e}[/]")
        raise typer.Exit(1)

    # Display header
    console.print()
    console.print(
        Panel(
            f"[bold blue]Installing vertical package[/]\n"
            f"[dim]Name:[/] {spec.name}\n"
            f"[dim]Source:[/] {spec.source.value}\n"
            f"[dim]Spec:[/] {package}",
            title="Victor Vertical Registry",
            border_style="blue",
        )
    )
    console.print()

    # Create registry manager
    manager = VerticalRegistryManager(dry_run=dry_run)

    # Install
    validate = not no_validate
    success, message = manager.install(spec, validate=validate, verbose=verbose)

    if success:
        console.print(f"[green]Success:[/] {message}")
        console.print()
        console.print("[bold]Next steps:[/]")
        console.print(f"  1. Restart Victor to load the new vertical")
        console.print(f"  2. Use 'victor vertical info {spec.name}' to see details")
        console.print()
    else:
        console.print(f"[red]Error:[/] {message}")
        raise typer.Exit(1)


@vertical_app.command("uninstall")
def uninstall_vertical(
    name: str = typer.Argument(
        ...,
        help="Name of the vertical to uninstall",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be uninstalled without actually uninstalling",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed uninstallation output",
    ),
) -> None:
    """Uninstall a vertical package.

    Examples:
        victor vertical uninstall victor-security
        victor vertical uninstall victor-security --dry-run
    """
    # Display header
    console.print()
    console.print(
        Panel(
            f"[bold blue]Uninstalling vertical package[/]\n" f"[dim]Name:[/] {name}",
            title="Victor Vertical Registry",
            border_style="blue",
        )
    )
    console.print()

    # Create registry manager
    manager = VerticalRegistryManager(dry_run=dry_run)

    # Uninstall
    success, message = manager.uninstall(name, verbose=verbose)

    if success:
        console.print(f"[green]Success:[/] {message}")
        console.print()
        console.print("[bold]Next steps:[/]")
        console.print(f"  1. Restart Victor to complete removal")
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

        # List with verbose output
        victor vertical list --verbose
    """
    # Validate source
    valid_sources = ["all", "installed", "builtin", "available"]
    if source not in valid_sources:
        console.print(f"[red]Error: Invalid source '{source}'[/]")
        console.print(f"[dim]Valid sources: {', '.join(valid_sources)}[/]")
        raise typer.Exit(1)

    # Create registry manager
    manager = VerticalRegistryManager()

    # Get verticals
    console.print()
    console.print(f"[dim]Loading verticals (source: {source})...[/]")
    verticals = manager.list_verticals(source=source)

    # Apply filters
    if category:
        verticals = [v for v in verticals if v.metadata and v.metadata.category == category]
        if verticals:
            console.print(f"[dim]Filtering by category: {category}[/]")

    if tags:
        tag_list = [t.strip().lower() for t in tags.split(",")]
        verticals = [
            v for v in verticals if v.metadata and any(tag in v.metadata.tags for tag in tag_list)
        ]
        if verticals:
            console.print(f"[dim]Filtering by tags: {tags}[/]")

    if not verticals:
        console.print(f"[yellow]No verticals found matching the criteria[/]")
        return

    # Display table
    table = Table(title=f"Verticals ({source})", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Version", style="green")
    table.add_column("Type", style="yellow")

    if verbose:
        table.add_column("Category", style="magenta")
        table.add_column("Tools", style="blue")
        table.add_column("Workflows", style="blue")
        table.add_column("Description", style="white")
    else:
        table.add_column("Description", style="white")

    for vertical in sorted(verticals, key=lambda v: v.name):
        # Get description from metadata or use default
        if vertical.metadata:
            description = vertical.metadata.description
        else:
            description = "No description available"

        # Truncate if too long
        if len(description) > 60:
            description = description[:57] + "..."

        # Type indicator
        vtype = "Built-in" if vertical.is_builtin else "External"

        if verbose:
            # Get additional metadata
            category = vertical.metadata.category if vertical.metadata else "-"  # type: ignore[assignment]
            tools = (
                ", ".join(vertical.metadata.class_spec.provides_tools[:3])
                if vertical.metadata and vertical.metadata.class_spec.provides_tools
                else "-"
            )
            if vertical.metadata and len(vertical.metadata.class_spec.provides_tools) > 3:
                tools += "..."

            workflows = (
                ", ".join(vertical.metadata.class_spec.provides_workflows[:3])
                if vertical.metadata and vertical.metadata.class_spec.provides_workflows
                else "-"
            )
            if vertical.metadata and len(vertical.metadata.class_spec.provides_workflows) > 3:
                workflows += "..."

            table.add_row(
                vertical.name,
                vertical.version,
                vtype,
                category,
                tools,
                workflows,
                description,
            )
        else:
            table.add_row(
                vertical.name,
                vertical.version,
                vtype,
                description,
            )

    console.print(table)
    console.print()
    console.print(f"[dim]Total: {len(verticals)} vertical(s)[/]")

    # Show hint for filtering
    if not category and not tags:
        console.print(
            "[dim]Tip: Use --category or --tags to filter results, --verbose for more details[/]"
        )
    console.print()


@vertical_app.command("search")
def search_verticals(
    query: str = typer.Argument(
        ...,
        help="Search query (matches name, description, tags)",
    ),
) -> None:
    """Search for vertical packages.

    Searches through all verticals (built-in, installed, and available)
    matching the query.

    Examples:
        victor vertical search security
        victor vertical search "data analysis"
        victor vertical search monitoring
    """
    # Create registry manager
    manager = VerticalRegistryManager()

    # Search
    console.print()
    console.print(f"[dim]Searching for '{query}'...[/]")
    results = manager.search(query)

    if not results:
        console.print(f"[yellow]No verticals found matching '{query}'[/]")
        return

    # Display table
    table = Table(title=f"Search Results: '{query}'", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Version", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Description", style="white")

    for vertical in results[:10]:  # Limit to 10 results
        # Get description from metadata or use default
        if vertical.metadata:
            description = vertical.metadata.description
        else:
            description = "No description available"

        # Highlight query match in description
        query_lower = query.lower()
        if query_lower in description.lower():
            # Simple highlighting (first match)
            idx = description.lower().index(query_lower)
            description = (
                description[:idx]
                + "[bold red]"
                + description[idx : idx + len(query)]
                + "[/]"
                + description[idx + len(query) :]
            )

        # Truncate if too long
        if len(description) > 60:
            description = description[:57] + "..."

        # Type indicator
        vtype = "Built-in" if vertical.is_builtin else "External"

        table.add_row(
            vertical.name,
            vertical.version,
            vtype,
            description,
        )

    console.print(table)
    console.print()
    console.print(f"[dim]Found {len(results)} result(s)[/]")
    console.print()


@vertical_app.command("info")
def show_vertical_info(
    name: str = typer.Argument(
        ...,
        help="Name of the vertical to show information for",
    ),
) -> None:
    """Show detailed information about a vertical.

    Displays metadata, dependencies, compatibility info, etc.

    Examples:
        victor vertical info security
        victor vertical info victor-security
    """
    # Create registry manager
    manager = VerticalRegistryManager()

    # Get info
    console.print()
    console.print(f"[dim]Loading information for '{name}'...[/]")
    vertical = manager.get_info(name)

    if not vertical:
        console.print(f"[yellow]Vertical '{name}' not found[/]")
        console.print()
        console.print("[dim]Use 'victor vertical list' to see available verticals[/]")
        raise typer.Exit(1)

    # Display info
    console.print()
    console.print(
        Panel(
            f"[bold cyan]{vertical.name}[/]\n"
            f"[dim]Version:[/] {vertical.version}\n"
            f"[dim]Type:[/] {'Built-in' if vertical.is_builtin else 'External'}\n"
            f"[dim]Location:[/] {vertical.location}",
            title="Vertical Information",
            border_style="cyan",
        )
    )
    console.print()

    # Display metadata if available
    if vertical.metadata:
        metadata = vertical.metadata

        # Description
        console.print("[bold]Description:[/]")
        console.print(f"  {metadata.description}")
        console.print()

        # Authors
        if metadata.authors:
            console.print("[bold]Authors:[/]")
            for author in metadata.authors:
                if author.email:
                    console.print(f"  - {author.name} <{author.email}>")
                else:
                    console.print(f"  - {author.name}")
            console.print()

        # Links
        console.print("[bold]Links:[/]")
        if metadata.homepage:
            console.print(f"  Homepage:     {metadata.homepage}")
        if metadata.repository:
            console.print(f"  Repository:   {metadata.repository}")
        if metadata.documentation:
            console.print(f"  Documentation: {metadata.documentation}")
        if metadata.issues:
            console.print(f"  Issues:       {metadata.issues}")
        console.print()

        # Requirements
        console.print("[bold]Requirements:[/]")
        console.print(f"  Victor:       {metadata.requires_victor}")
        console.print(f"  License:      {metadata.license}")
        console.print(f"  Python:       {metadata.compatibility.python_version}")
        console.print()

        # Class specification
        console.print("[bold]Entry Point:[/]")
        console.print(f"  Module:       {metadata.class_spec.module}")
        console.print(f"  Class:        {metadata.class_spec.class_name}")
        console.print()

        # Capabilities
        if metadata.class_spec.provides_tools:
            console.print("[bold]Provides Tools:[/]")
            for tool in metadata.class_spec.provides_tools:
                console.print(f"  - {tool}")
            console.print()

        if metadata.class_spec.provides_workflows:
            console.print("[bold]Provides Workflows:[/]")
            for workflow in metadata.class_spec.provides_workflows:
                console.print(f"  - {workflow}")
            console.print()

        if metadata.class_spec.provides_capabilities:
            console.print("[bold]Provides Capabilities:[/]")
            for capability in metadata.class_spec.provides_capabilities:
                console.print(f"  - {capability}")
            console.print()

        # Dependencies
        if metadata.dependencies.python:
            console.print("[bold]Python Dependencies:[/]")
            for dep in metadata.dependencies.python:
                console.print(f"  - {dep}")
            console.print()

        if metadata.dependencies.verticals:
            console.print("[bold]Vertical Dependencies:[/]")
            for dep in metadata.dependencies.verticals:
                console.print(f"  - {dep}")
            console.print()

        # Tags
        if metadata.tags:
            tags_str = ", ".join(metadata.tags)
            console.print(f"[bold]Tags:[/] {tags_str}")
            console.print()

    else:
        console.print("[yellow]No detailed metadata available[/]")
        console.print()


@vertical_app.command("create")
def create_vertical(
    name: str = typer.Argument(
        ...,
        help="Name of the new vertical (e.g., 'security', 'analytics')",
    ),
    description: str = typer.Option(
        None,
        "--description",
        "-d",
        help="Description of the vertical's purpose",
    ),
    service_provider: bool = typer.Option(
        False,
        "--service-provider",
        "-s",
        help="Include service_provider.py for DI container registration",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing files if vertical already exists",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be created without actually creating files",
    ),
) -> None:
    """Generate a new vertical structure from templates.

    This is a convenience alias for 'victor vertical create' from the scaffold command.
    For more details, see 'victor vertical create --help'.

    Examples:
        victor vertical create security --description "Security analysis assistant"
        victor vertical create analytics -d "Data analytics" --service-provider
        victor vertical create ml --dry-run
    """
    # Import the scaffold command
    from victor.ui.commands.scaffold import new_vertical

    # Call the scaffold command
    new_vertical(
        name=name,
        description=description,
        service_provider=service_provider,
        force=force,
        dry_run=dry_run,
    )


@vertical_app.command("list-templates")
def list_templates(
    category: str = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category",
    ),
    tag: str = typer.Option(
        None,
        "--tag",
        "-t",
        help="Filter by tag",
    ),
) -> None:
    """List available vertical templates.

    Examples:
        victor vertical list-templates
        victor vertical list-templates --category development
        victor vertical list-templates --tag security
    """
    from victor.framework.vertical_template_registry import get_template_registry

    registry = get_template_registry()

    # Load built-in templates if not already loaded
    if not registry.list_names():
        from pathlib import Path

        templates_dir = Path(__file__).parent.parent.parent / "config" / "templates"
        if templates_dir.exists():
            count = registry.load_directory(templates_dir, pattern="*_vertical_template.yaml")
            console.print(f"[dim]Loaded {count} built-in templates[/]\n")

    templates = registry.list_all()

    # Apply filters
    if category:
        templates = [t for t in templates if t.metadata.category == category]
    if tag:
        templates = [t for t in templates if tag in t.metadata.tags]

    if not templates:
        console.print("[yellow]No templates found[/]")
        return

    # Display table
    table = Table(title="Vertical Templates", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Version", style="green")
    table.add_column("Category", style="yellow")
    table.add_column("Description", style="white")
    table.add_column("Tools", style="blue")
    table.add_column("Stages", style="magenta")

    for template in sorted(templates, key=lambda t: t.metadata.name):
        description = template.metadata.description
        if len(description) > 50:
            description = description[:47] + "..."

        table.add_row(
            template.metadata.name,
            template.metadata.version,
            template.metadata.category,
            description,
            str(len(template.tools)),
            str(len(template.stages)),
        )

    console.print(table)
    console.print()
    console.print(f"[dim]Total: {len(templates)} template(s)[/]")


@vertical_app.command("validate-template")
def validate_template(
    path: str = typer.Argument(
        ...,
        help="Path to template YAML file",
    ),
) -> None:
    """Validate a vertical template YAML file.

    Examples:
        victor vertical validate-template templates/my_vertical.yaml
        victor vertical validate-template victor/config/templates/base_vertical_template.yaml
    """
    from victor.framework.vertical_template_registry import get_template_registry

    registry = get_template_registry()

    console.print()
    console.print(f"[dim]Validating template: {path}[/]")

    # Load template
    template = registry.load_from_yaml(path, resolve_inheritance=False)

    if template is None:
        console.print("[red]Error: Failed to load template[/]")
        raise typer.Exit(1)

    # Validate template
    errors = template.validate()

    if errors:
        console.print()
        console.print("[red]Validation errors:[/]")
        for error in errors:
            console.print(f"  [red]✗[/] {error}")
        console.print()
        raise typer.Exit(1)
    else:
        console.print("[green]Template is valid![/]")
        console.print()

        # Show template info
        console.print("[bold]Template Information:[/]")
        console.print(f"  Name: {template.metadata.name}")
        console.print(f"  Version: {template.metadata.version}")
        console.print(f"  Category: {template.metadata.category}")
        console.print(f"  Tools: {len(template.tools)}")
        console.print(f"  Stages: {len(template.stages)}")
        console.print(f"  Workflows: {len(template.workflows)}")
        console.print(f"  Teams: {len(template.teams)}")
        console.print(f"  Capabilities: {len(template.capabilities)}")
        console.print()


@vertical_app.command("export-template")
def export_template(
    vertical_name: str = typer.Argument(
        ...,
        help="Name of the vertical to export (e.g., 'coding', 'research')",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: {vertical_name}_template.yaml)",
    ),
) -> None:
    """Export an existing vertical to a template YAML file.

    This creates a template from an existing vertical implementation,
    which can then be used as a base for creating new verticals.

    Examples:
        victor vertical export-template coding
        victor vertical export-template research --output templates/research_template.yaml
    """
    from pathlib import Path

    from victor.core.verticals.base import VerticalRegistry
    from victor.framework.vertical_template_registry import get_template_registry

    # Get vertical class
    vertical_class = VerticalRegistry.get(vertical_name)

    if vertical_class is None:
        console.print(f"[red]Error: Vertical '{vertical_name}' not found[/]")
        console.print()
        console.print(
            "[dim]Use 'victor vertical list --source builtin' to see available verticals[/]"
        )
        raise typer.Exit(1)

    console.print()
    console.print(f"[dim]Exporting vertical '{vertical_name}' to template...[/]")

    # Extract template from vertical
    from victor.framework.vertical_extractor import VerticalExtractor

    extractor = VerticalExtractor()
    template = extractor.extract_from_class(vertical_class)

    if template is None:
        console.print("[red]Error: Failed to extract template from vertical[/]")
        raise typer.Exit(1)

    # Determine output path
    if output is None:
        output = f"{vertical_name}_template.yaml"  # type: ignore[unreachable]

    output_path = Path(output)

    # Save template
    registry = get_template_registry()
    success = registry.save_to_yaml(template, output_path, validate=True)

    if success:
        console.print(f"[green]Template exported to: {output_path}[/]")
        console.print()
        console.print("[bold]Next steps:[/]")
        console.print(f"  1. Review and customize the template: {output_path}")
        console.print(f"  2. Validate: victor vertical validate-template {output_path}")
        console.print(
            f"  3. Use to create new vertical: victor vertical create my_vertical --template {output_path}"
        )
        console.print()
    else:
        console.print("[red]Error: Failed to save template[/]")
        raise typer.Exit(1)


__all__ = ["vertical_app"]
