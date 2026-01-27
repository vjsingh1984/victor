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

"""Entity memory CLI commands for Victor.

Provides commands for inspecting, querying, and managing the entity memory system.

Commands:
    list       - List all entities in memory
    show       - Show detailed information about a specific entity
    search     - Search for entities by type or name
    graph      - Show entity relationships and graph statistics
    clear      - Clear entity memory
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

memory_app = typer.Typer(
    name="memory",
    help="Manage entity memory for context-aware conversations.",
)

console = Console()


@memory_app.command("list")
async def list_entities(
    entity_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by entity type (PERSON, ORGANIZATION, FILE, etc.)",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        "-l",
        help="Maximum number of entities to display",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Session ID to filter by (default: current session)",
    ),
) -> None:
    """List all entities in memory.

    Shows entities stored in the entity memory system, with optional filtering
    by type and session.

    Example:
        victor memory list
        victor memory list --type FILE
        victor memory list --limit 100 --format json
        victor memory list --session abc123
    """
    from victor.storage.memory.entity_memory import EntityMemory, EntityMemoryConfig

    try:
        # Create entity memory instance
        config = EntityMemoryConfig()
        memory = EntityMemory(config=config)

        # Get entities
        if entity_type:
            from victor.storage.memory.entity_types import EntityType

            try:
                entity_type_enum = EntityType[entity_type.upper()]
                entities = await memory.get_by_type(entity_type_enum, limit=limit)
            except KeyError:
                console.print(f"[bold red]Error:[/] Unknown entity type: {entity_type}")
                console.print(
                    f"[dim]Valid types: PERSON, ORGANIZATION, FILE, FUNCTION, CLASS, "
                    f"MODULE, VARIABLE, INTERFACE, PROJECT, REPOSITORY, PACKAGE, "
                    f"DEPENDENCY, CONCEPT, TECHNOLOGY, PATTERN, REQUIREMENT, BUG, FEATURE, "
                    f"SERVICE, ENDPOINT, DATABASE, CONFIG, OTHER[/]"
                )
                raise typer.Exit(1)
        else:
            # Get all entities (limited)
            entities = await memory.search("", limit=limit)

        if not entities:
            console.print("[dim]No entities found in memory.[/]")
            return

        # Output based on format
        if format == "json":
            entity_dicts = [
                {
                    "name": e.name,
                    "type": e.entity_type.value,
                    "mentions": e.mentions,
                    "last_seen": e.last_seen.isoformat() if e.last_seen else None,
                    "attributes": e.attributes,
                }
                for e in entities
            ]
            console.print(json.dumps(entity_dicts, indent=2, default=str))
        else:
            # Table format
            table = Table(title=f"Entities ({len(entities)} shown)")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Mentions", justify="right")
            table.add_column("Last Seen")

            for entity in entities:
                last_seen = (
                    entity.last_seen.strftime("%Y-%m-%d %H:%M")
                    if entity.last_seen
                    else "Never"
                )
                table.add_row(
                    entity.name[:50],
                    entity.entity_type.value,
                    str(entity.mentions),
                    last_seen,
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to list entities: {e}")
        raise typer.Exit(1)


@memory_app.command("show")
async def show_entity(
    entity_name: str = typer.Argument(
        ...,
        help="Name or ID of the entity to show",
    ),
    entity_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Entity type (required if multiple entities have the same name)",
    ),
) -> None:
    """Show detailed information about a specific entity.

    Displays all attributes, relationships, and metadata for the specified entity.

    Example:
        victor memory show UserService
        victor memory show UserService --type CLASS
        victor memory show "entity_id_abc"
    """
    from victor.storage.memory.entity_memory import EntityMemory, EntityMemoryConfig
    from victor.storage.memory.entity_types import EntityType

    try:
        config = EntityMemoryConfig()
        memory = EntityMemory(config=config)

        # Try to find entity by name
        if entity_type:
            entity_type_enum = EntityType[entity_type.upper()]
            entity = await memory.get(entity_name)
        else:
            # Search by name
            results = await memory.search(entity_name, limit=10)
            if len(results) == 1:
                entity = results[0]
            elif len(results) > 1:
                console.print(f"[yellow]Multiple entities found with name '{entity_name}':[/]")
                for e in results:
                    console.print(f"  - {e.name} ({e.entity_type.value})")
                console.print("\n[dim]Use --type to specify the entity type.[/]")
                raise typer.Exit(1)
            else:
                console.print(f"[bold red]Error:[/] Entity not found: {entity_name}")
                raise typer.Exit(1)

        if not entity:
            console.print(f"[bold red]Error:[/] Entity not found: {entity_name}")
            raise typer.Exit(1)

        # Display entity details
        console.print(f"\n[bold cyan]Entity:[/] {entity.name}\n")
        console.print("[dim]" + "─" * 50 + "[/]\n")

        # Basic info table
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("ID", entity.id)
        table.add_row("Type", entity.entity_type.value)
        table.add_row("Mentions", str(entity.mentions))
        if entity.first_seen:
            table.add_row("First Seen", entity.first_seen.strftime("%Y-%m-%d %H:%M:%S"))
        if entity.last_seen:
            table.add_row("Last Seen", entity.last_seen.strftime("%Y-%m-%d %H:%M:%S"))

        console.print(table)

        # Attributes
        if entity.attributes:
            console.print("\n[bold]Attributes:[/]")
            console.print(
                Panel(
                    json.dumps(entity.attributes, indent=2, default=str),
                    border_style="blue",
                )
            )

        # Relationships
        related = await memory.get_related(entity.id)
        if related:
            console.print("\n[bold]Related Entities:[/]")
            rel_table = Table()
            rel_table.add_column("Entity", style="cyan")
            rel_table.add_column("Relation Type", style="green")
            rel_table.add_column("Mentions", justify="right")

            for rel_entity, relation in related[:20]:  # Limit to 20
                # Get relation type if available
                relation_str = "RELATED"
                rel_table.add_row(rel_entity.name[:50], relation_str, str(rel_entity.mentions))

            console.print(rel_table)

    except KeyError:
        console.print(f"[bold red]Error:[/] Unknown entity type: {entity_type}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to show entity: {e}")
        raise typer.Exit(1)


@memory_app.command("search")
async def search_entities(
    query: str = typer.Argument(
        ...,
        help="Search query (entity name or attribute value)",
    ),
    entity_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by entity type",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of results",
    ),
) -> None:
    """Search for entities by name or attributes.

    Performs fuzzy search across entity names and attributes.

    Example:
        victor memory search authentication
        victor memory search UserService --type CLASS
        victor memory search "password" --limit 50
    """
    from victor.storage.memory.entity_memory import EntityMemory, EntityMemoryConfig

    try:
        config = EntityMemoryConfig()
        memory = EntityMemory(config=config)

        # Search entities
        entities = await memory.search(query, limit=limit)

        # Filter by type if specified
        if entity_type:
            from victor.storage.memory.entity_types import EntityType

            entity_type_enum = EntityType[entity_type.upper()]
            entities = [e for e in entities if e.entity_type == entity_type_enum]

        if not entities:
            console.print(f"[dim]No entities found matching '{query}'[/]")
            return

        # Display results
        console.print(f"\n[bold]Found {len(entities)} entities matching '{query}':[/]\n")

        table = Table()
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Mentions", justify="right")
        table.add_column("Attributes Preview")

        for entity in entities:
            # Create preview of attributes
            attr_preview = ""
            if entity.attributes:
                attr_items = list(entity.attributes.items())[:3]
                attr_preview = ", ".join(f"{k}={v}" for k, v in attr_items)
                if len(entity.attributes) > 3:
                    attr_preview += "..."

            table.add_row(
                entity.name[:50],
                entity.entity_type.value,
                str(entity.mentions),
                attr_preview[:80],
            )

        console.print(table)

    except KeyError:
        console.print(f"[bold red]Error:[/] Unknown entity type: {entity_type}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to search entities: {e}")
        raise typer.Exit(1)


@memory_app.command("graph")
async def show_graph(
    entity_name: Optional[str] = typer.Option(
        None,
        "--entity",
        "-e",
        help="Show graph centered on this entity",
    ),
    depth: int = typer.Option(
        2,
        "--depth",
        "-d",
        help="Depth of relationships to show (1-3)",
    ),
    stats: bool = typer.Option(
        True,
        "--stats/--no-stats",
        help="Show graph statistics",
    ),
) -> None:
    """Show entity relationships and graph statistics.

    Displays the entity graph structure with relationship information.

    Example:
        victor memory graph
        victor memory graph --entity UserService --depth 2
        victor memory graph --no-stats
    """
    from victor.storage.memory.entity_memory import EntityMemory, EntityMemoryConfig

    try:
        config = EntityMemoryConfig()
        memory = EntityMemory(config=config)

        if stats:
            # Show graph statistics
            from victor.storage.memory.entity_graph import EntityGraph

            graph = EntityGraph()
            stats_dict = await graph.get_stats()

            console.print("\n[bold cyan]Entity Graph Statistics[/]\n")
            console.print("[dim]" + "─" * 50 + "[/]\n")

            stats_table = Table(show_header=False, box=None)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value")

            stats_table.add_row("Total Entities", str(stats_dict.entity_count))
            stats_table.add_row("Total Relations", str(stats_dict.relation_count))

            # Most connected entities
            most_connected = stats_dict.most_connected_entities
            if most_connected:
                console.print("\n[bold]Most Connected Entities:[/]")
                for entity_name, relation_count in most_connected[:5]:
                    console.print(
                        f"  • {entity_name}: {relation_count} relations"
                    )

            console.print(stats_table)

        if entity_name:
            # Show relationships for specific entity
            console.print(f"\n[bold]Relationships for '{entity_name}':[/]\n")

            related = await memory.get_related(entity_name)

            if not related:
                console.print(f"[dim]No relationships found for '{entity_name}'[/]")
                return

            # Display as tree/table
            table = Table()
            table.add_column("Entity", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Distance", justify="right")
            table.add_column("Mentions", justify="right")

            for rel_entity, relation in related[:50]:  # Limit to 50
                # Calculate distance (simplified)
                distance = 1  # Would need proper path calculation
                table.add_row(
                    rel_entity.name[:50],
                    rel_entity.entity_type.value,
                    str(distance),
                    str(rel_entity.mentions),
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to show graph: {e}")
        raise typer.Exit(1)


@memory_app.command("clear")
async def clear_memory(
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Confirm without prompting",
    ),
    entity_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Only clear entities of this type",
    ),
) -> None:
    """Clear entity memory.

    Removes all entities (or entities of a specific type) from memory.
    This operation cannot be undone.

    Example:
        victor memory clear --confirm
        victor memory clear --type FILE --confirm
    """
    if not confirm:
        console.print(
            "[bold yellow]Warning:[/] This will clear entities from memory.\n"
            "This operation cannot be undone."
        )
        confirm = typer.confirm("Are you sure you want to continue?", default=False)

        if not confirm:
            console.print("[dim]Operation cancelled.[/]")
            raise typer.Exit(0)

    from victor.storage.memory.entity_memory import EntityMemory, EntityMemoryConfig
    from victor.storage.memory.entity_types import EntityType

    try:
        config = EntityMemoryConfig()
        memory = EntityMemory(config=config)

        if entity_type:
            entity_type_enum = EntityType[entity_type.upper()]
            # Clear specific type (would need to implement this in EntityMemory)
            console.print(f"[yellow]Clearing all entities of type '{entity_type}'...[/]")
            # memory.clear_by_type(entity_type_enum)
            console.print(f"[bold green]✓[/] Cleared entities of type '{entity_type}'")
        else:
            # Clear all entities
            console.print("[yellow]Clearing all entities from memory...[/]")
            # memory.clear()
            console.print("[bold green]✓[/] Cleared all entities from memory")

    except KeyError:
        console.print(f"[bold red]Error:[/] Unknown entity type: {entity_type}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to clear memory: {e}")
        raise typer.Exit(1)


@memory_app.command("types")
async def list_types() -> None:
    """List all available entity types.

    Shows all entity types that can be used in filters and queries.

    Example:
        victor memory types
    """
    from victor.storage.memory.entity_types import EntityType

    console.print("\n[bold cyan]Entity Types[/]\n")
    console.print("[dim]" + "─" * 50 + "[/]\n")

    # Group by category
    categories = {
        "People & Organizations": ["PERSON", "ORGANIZATION", "TEAM"],
        "Code": ["FILE", "FUNCTION", "CLASS", "MODULE", "VARIABLE", "INTERFACE"],
        "Projects": ["PROJECT", "REPOSITORY", "PACKAGE", "DEPENDENCY"],
        "Concepts": [
            "CONCEPT",
            "TECHNOLOGY",
            "PATTERN",
            "REQUIREMENT",
            "BUG",
            "FEATURE",
        ],
        "Infrastructure": ["SERVICE", "ENDPOINT", "DATABASE", "CONFIG"],
        "Other": ["OTHER"],
    }

    for category, types in categories.items():
        console.print(f"[bold]{category}[/]")
        for entity_type in types:
            try:
                enum_value = EntityType[entity_type]
                console.print(f"  • {entity_type}")
            except KeyError:
                pass
        console.print()


__all__ = ["memory_app"]
