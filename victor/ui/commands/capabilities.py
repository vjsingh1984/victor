# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Capability discovery CLI command for Victor.

This module provides the `victor capabilities` command for discovering
all available tools, verticals, personas, teams, and other features.

Follows SRP: Single responsibility of capability aggregation and display.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

capabilities_app = typer.Typer(
    name="capabilities",
    help="Discover Victor's capabilities: tools, verticals, personas, teams, and more.",
)

console = Console()


@dataclass
class CapabilityManifest:
    """Aggregated manifest of all Victor capabilities."""

    tools: List[str] = field(default_factory=list)
    tool_categories: List[str] = field(default_factory=list)
    verticals: List[str] = field(default_factory=list)
    personas: List[str] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)
    chains: List[str] = field(default_factory=list)
    handlers: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    providers: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "tools": self.tools,
            "tool_categories": self.tool_categories,
            "verticals": self.verticals,
            "personas": self.personas,
            "teams": self.teams,
            "chains": self.chains,
            "handlers": self.handlers,
            "task_types": self.task_types,
            "providers": self.providers,
            "events": self.events,
            "summary": {
                "total_tools": len(self.tools),
                "total_verticals": len(self.verticals),
                "total_personas": len(self.personas),
                "total_teams": len(self.teams),
                "total_chains": len(self.chains),
                "total_providers": len(self.providers),
            },
        }


class CapabilityDiscovery:
    """Aggregates discovery across all Victor registries.

    This class provides a unified interface for discovering all capabilities
    in Victor, aggregating from multiple specialized registries.

    Design Principles:
    - SRP: Single responsibility of capability aggregation
    - OCP: New registries can be added without modifying existing code
    - DIP: Depends on registry abstractions, not concrete implementations
    """

    def discover_all(self) -> CapabilityManifest:
        """Discover all capabilities from all registries."""
        manifest = CapabilityManifest()

        # Discover tools
        manifest.tools = self._discover_tools()
        manifest.tool_categories = self._discover_tool_categories()

        # Discover verticals
        manifest.verticals = self._discover_verticals()

        # Discover personas
        manifest.personas = self._discover_personas()

        # Discover teams
        manifest.teams = self._discover_teams()

        # Discover chains/workflows
        manifest.chains = self._discover_chains()

        # Discover handlers
        manifest.handlers = self._discover_handlers()

        # Discover task types
        manifest.task_types = self._discover_task_types()

        # Discover providers
        manifest.providers = self._discover_providers()

        # Discover event types
        manifest.events = self._discover_events()

        return manifest

    def discover_by_vertical(self, vertical: str) -> Dict[str, Any]:
        """Discover capabilities filtered by vertical."""
        result: Dict[str, Any] = {"vertical": vertical}

        try:
            from victor.framework.persona_registry import get_persona_registry

            registry = get_persona_registry()
            result["personas"] = registry.list_personas(vertical=vertical)
        except Exception:
            result["personas"] = []

        try:
            from victor.framework.chain_registry import get_chain_registry

            chain_registry = get_chain_registry()
            result["chains"] = chain_registry.list_chains(vertical=vertical)
        except Exception:
            result["chains"] = []

        try:
            from victor.framework.team_registry import get_team_registry

            team_registry = get_team_registry()
            teams_data = team_registry.find_by_vertical(vertical)
            result["teams"] = list(teams_data.keys()) if teams_data else []
        except Exception:
            result["teams"] = []

        try:
            from victor.framework.handler_registry import get_handler_registry

            handler_registry = get_handler_registry()
            result["handlers"] = handler_registry.list_by_vertical(vertical)
        except Exception:
            result["handlers"] = []

        try:
            from victor.framework.task_types import get_task_type_registry

            task_type_registry = get_task_type_registry()
            result["task_types"] = task_type_registry.list_types(vertical=vertical)
        except Exception:
            result["task_types"] = []

        return result

    def _discover_tools(self) -> List[str]:
        """Discover all registered tools."""
        try:
            from victor.tools.registry import ToolRegistry

            registry = ToolRegistry()
            tools = registry.list_tools(only_enabled=False)
            return [t.name if hasattr(t, "name") else str(t) for t in tools]
        except Exception:
            # Fallback to shared registry
            try:
                from victor.tools.shared_registry import SharedToolRegistry

                registry = SharedToolRegistry.get_instance()
                return list(registry._tools.keys())
            except Exception:
                return []

    def _discover_tool_categories(self) -> List[str]:
        """Discover all tool categories."""
        try:
            from victor.framework.tools import get_category_registry

            registry = get_category_registry()
            return list(registry.get_all_categories())
        except Exception:
            return []

    def _discover_verticals(self) -> List[str]:
        """Discover all registered verticals."""
        try:
            from victor.core.verticals.base import VerticalRegistry

            return VerticalRegistry.list_names()
        except Exception:
            return []

    def _discover_personas(self) -> List[str]:
        """Discover all registered personas."""
        try:
            from victor.framework.persona_registry import get_persona_registry

            registry = get_persona_registry()
            return registry.list_personas()
        except Exception:
            return []

    def _discover_teams(self) -> List[str]:
        """Discover all registered team specs."""
        try:
            from victor.framework.team_registry import get_team_registry

            registry = get_team_registry()
            return registry.list_teams()
        except Exception:
            return []

    def _discover_chains(self) -> List[str]:
        """Discover all registered chains/workflows."""
        try:
            from victor.framework.chain_registry import get_chain_registry

            registry = get_chain_registry()
            return registry.list_chains()
        except Exception:
            return []

    def _discover_handlers(self) -> List[str]:
        """Discover all registered compute handlers."""
        try:
            from victor.framework.handler_registry import get_handler_registry

            registry = get_handler_registry()
            return registry.list_handlers()
        except Exception:
            return []

    def _discover_task_types(self) -> List[str]:
        """Discover all registered task types."""
        try:
            from victor.framework.task_types import get_task_type_registry

            registry = get_task_type_registry()
            return registry.list_types()
        except Exception:
            return []

    def _discover_providers(self) -> List[str]:
        """Discover all registered LLM providers."""
        try:
            from victor.providers.registry import ProviderRegistry

            return ProviderRegistry.list_providers()
        except Exception:
            return []

    def _discover_events(self) -> List[str]:
        """Discover all supported event types."""
        try:
            from victor.framework.event_registry import get_event_registry

            registry = get_event_registry()
            event_types = registry.list_supported_types()
            return [et.value if hasattr(et, "value") else str(et) for et in event_types]
        except Exception:
            return []


# Singleton instance
_discovery: Optional[CapabilityDiscovery] = None


def get_capability_discovery() -> CapabilityDiscovery:
    """Get or create the capability discovery singleton."""
    global _discovery
    if _discovery is None:
        _discovery = CapabilityDiscovery()
    return _discovery


@capabilities_app.callback(invoke_without_command=True)
def capabilities_main(
    ctx: typer.Context,
    vertical: Optional[str] = typer.Option(
        None, "--vertical", "-v", help="Filter by vertical (e.g., coding, devops)"
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON for programmatic use"
    ),
) -> None:
    """Discover all Victor capabilities.

    Examples:
        victor capabilities              # Show all capabilities
        victor capabilities --vertical coding  # Filter by vertical
        victor capabilities --json       # JSON output for tooling
    """
    if ctx.invoked_subcommand is not None:
        return

    discovery = get_capability_discovery()

    if vertical:
        manifest = discovery.discover_by_vertical(vertical)
        if json_output:
            import json

            console.print(json.dumps(manifest, indent=2))
        else:
            _display_vertical_capabilities(vertical, manifest)
    else:
        all_manifest = discovery.discover_all()
        if json_output:
            import json

            console.print(json.dumps(all_manifest.to_dict(), indent=2))
        else:
            _display_full_capabilities(all_manifest)


def _display_full_capabilities(manifest: CapabilityManifest) -> None:
    """Display full capability manifest in rich format."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Victor Capabilities[/bold cyan]\n"
            "[dim]Discover available tools, verticals, personas, and more[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Summary table
    summary_table = Table(title="Capability Summary", show_header=True)
    summary_table.add_column("Category", style="cyan")
    summary_table.add_column("Count", justify="right", style="green")
    summary_table.add_column("Examples", style="dim")

    summary_table.add_row(
        "Tools",
        str(len(manifest.tools)),
        ", ".join(manifest.tools[:3]) + ("..." if len(manifest.tools) > 3 else ""),
    )
    summary_table.add_row(
        "Tool Categories",
        str(len(manifest.tool_categories)),
        ", ".join(manifest.tool_categories[:3])
        + ("..." if len(manifest.tool_categories) > 3 else ""),
    )
    summary_table.add_row(
        "Verticals",
        str(len(manifest.verticals)),
        ", ".join(manifest.verticals[:5]),
    )
    summary_table.add_row(
        "Providers",
        str(len(manifest.providers)),
        ", ".join(manifest.providers[:5]) + ("..." if len(manifest.providers) > 5 else ""),
    )
    summary_table.add_row(
        "Personas",
        str(len(manifest.personas)),
        ", ".join(manifest.personas[:3]) + ("..." if len(manifest.personas) > 3 else ""),
    )
    summary_table.add_row(
        "Teams",
        str(len(manifest.teams)),
        ", ".join(manifest.teams[:3]) + ("..." if len(manifest.teams) > 3 else ""),
    )
    summary_table.add_row(
        "Chains/Workflows",
        str(len(manifest.chains)),
        ", ".join(manifest.chains[:3]) + ("..." if len(manifest.chains) > 3 else ""),
    )
    summary_table.add_row(
        "Task Types",
        str(len(manifest.task_types)),
        ", ".join(manifest.task_types[:3]) + ("..." if len(manifest.task_types) > 3 else ""),
    )
    summary_table.add_row(
        "Handlers",
        str(len(manifest.handlers)),
        ", ".join(manifest.handlers[:3]) + ("..." if len(manifest.handlers) > 3 else ""),
    )
    summary_table.add_row(
        "Event Types",
        str(len(manifest.events)),
        ", ".join(manifest.events[:3]) + ("..." if len(manifest.events) > 3 else ""),
    )

    console.print(summary_table)
    console.print()

    # Helpful tips
    console.print("[dim]Use subcommands for detailed lists:[/dim]")
    console.print("  [cyan]victor capabilities tools[/cyan]      - List all tools")
    console.print("  [cyan]victor capabilities verticals[/cyan]  - List all verticals")
    console.print("  [cyan]victor capabilities providers[/cyan]  - List all providers")
    console.print("  [cyan]victor capabilities --vertical coding[/cyan] - Filter by vertical")
    console.print()


def _display_vertical_capabilities(vertical: str, manifest: Dict[str, Any]) -> None:
    """Display capabilities for a specific vertical."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Capabilities for {vertical} vertical[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    tree = Tree(f"[bold]{vertical}[/bold]")

    for category, items in manifest.items():
        if category == "vertical" or not items:
            continue
        branch = tree.add(f"[cyan]{category}[/cyan] ({len(items)})")
        for item in items[:10]:  # Show first 10
            branch.add(f"[dim]{item}[/dim]")
        if len(items) > 10:
            branch.add(f"[dim]... and {len(items) - 10} more[/dim]")

    console.print(tree)
    console.print()


@capabilities_app.command("tools")
def list_tools(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON output"),
) -> None:
    """List all available tools."""
    discovery = get_capability_discovery()
    manifest = discovery.discover_all()

    if json_output:
        import json

        data = {
            "tools": manifest.tools,
            "categories": manifest.tool_categories,
            "total": len(manifest.tools),
        }
        console.print(json.dumps(data, indent=2))
        return

    console.print()
    console.print(Panel.fit("[bold cyan]Available Tools[/bold cyan]", border_style="cyan"))
    console.print()

    # Show categories
    if manifest.tool_categories:
        console.print("[bold]Tool Categories:[/bold]")
        for cat in sorted(manifest.tool_categories):
            console.print(f"  [cyan]{cat}[/cyan]")
        console.print()

    # Show tools
    console.print(f"[bold]Tools ({len(manifest.tools)} total):[/bold]")
    for tool in sorted(manifest.tools):
        console.print(f"  [dim]{tool}[/dim]")
    console.print()


@capabilities_app.command("verticals")
def list_verticals(
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON output"),
) -> None:
    """List all available verticals."""
    discovery = get_capability_discovery()
    manifest = discovery.discover_all()

    if json_output:
        import json

        console.print(json.dumps({"verticals": manifest.verticals}, indent=2))
        return

    console.print()
    console.print(Panel.fit("[bold cyan]Available Verticals[/bold cyan]", border_style="cyan"))
    console.print()

    for v in sorted(manifest.verticals):
        console.print(f"  [cyan]{v}[/cyan]")
    console.print()
    console.print(f"[dim]Total: {len(manifest.verticals)} verticals[/dim]")
    console.print()


@capabilities_app.command("providers")
def list_providers(
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON output"),
) -> None:
    """List all available LLM providers."""
    discovery = get_capability_discovery()
    manifest = discovery.discover_all()

    if json_output:
        import json

        console.print(json.dumps({"providers": manifest.providers}, indent=2))
        return

    console.print()
    console.print(Panel.fit("[bold cyan]Available Providers[/bold cyan]", border_style="cyan"))
    console.print()

    for p in sorted(manifest.providers):
        console.print(f"  [cyan]{p}[/cyan]")
    console.print()
    console.print(f"[dim]Total: {len(manifest.providers)} providers[/dim]")
    console.print()


@capabilities_app.command("teams")
def list_teams(
    vertical: Optional[str] = typer.Option(None, "--vertical", "-v", help="Filter by vertical"),
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON output"),
) -> None:
    """List all available team configurations."""
    discovery = get_capability_discovery()

    if vertical:
        data = discovery.discover_by_vertical(vertical)
        teams = data.get("teams", [])
    else:
        manifest = discovery.discover_all()
        teams = manifest.teams

    if json_output:
        import json

        console.print(json.dumps({"teams": teams, "vertical": vertical}, indent=2))
        return

    console.print()
    title = "[bold cyan]Available Teams"
    if vertical:
        title += f" ({vertical})"
    title += "[/bold cyan]"
    console.print(Panel.fit(title, border_style="cyan"))
    console.print()

    for t in sorted(teams):
        console.print(f"  [cyan]{t}[/cyan]")
    console.print()
    console.print(f"[dim]Total: {len(teams)} teams[/dim]")
    console.print()


@capabilities_app.command("personas")
def list_personas(
    vertical: Optional[str] = typer.Option(None, "--vertical", "-v", help="Filter by vertical"),
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON output"),
) -> None:
    """List all available personas."""
    discovery = get_capability_discovery()

    if vertical:
        data = discovery.discover_by_vertical(vertical)
        personas = data.get("personas", [])
    else:
        manifest = discovery.discover_all()
        personas = manifest.personas

    if json_output:
        import json

        console.print(json.dumps({"personas": personas, "vertical": vertical}, indent=2))
        return

    console.print()
    title = "[bold cyan]Available Personas"
    if vertical:
        title += f" ({vertical})"
    title += "[/bold cyan]"
    console.print(Panel.fit(title, border_style="cyan"))
    console.print()

    for p in sorted(personas):
        console.print(f"  [cyan]{p}[/cyan]")
    console.print()
    console.print(f"[dim]Total: {len(personas)} personas[/dim]")
    console.print()
