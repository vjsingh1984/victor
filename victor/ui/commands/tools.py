import typer
import asyncio
import importlib
import inspect
import os
from typing import List, Tuple
from rich.console import Console
from rich.table import Table

from victor.config.settings import load_settings

# ToolDefinition is imported for type hinting, though not directly used in the logic
from victor.providers.base import ToolDefinition  # noqa: F401


tools_app = typer.Typer(name="tools", help="Manage Victor's integrated tools.")
console = Console()


def _discover_tools_lightweight() -> List[Tuple[str, str, str]]:
    """Discover tools without full orchestrator initialization.

    Dynamically discovers tools from the victor/tools directory by scanning
    for @tool decorated functions and BaseTool subclasses.

    Returns:
        List of tuples: (name, description, cost_tier)
    """
    from victor.tools.base import BaseTool as BaseToolClass

    tools_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tools")
    excluded_files = {
        "__init__.py",
        "base.py",
        "decorators.py",
        "semantic_selector.py",
        "enums.py",
        "registry.py",
        "metadata.py",
        "metadata_registry.py",
        "tool_names.py",
        "output_utils.py",
        "shared_ast_utils.py",
        "dependency_graph.py",
        "plugin_registry.py",
    }

    discovered_tools = []

    for filename in os.listdir(tools_dir):
        if filename.endswith(".py") and filename not in excluded_files:
            module_name = f"victor.tools.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for _name, obj in inspect.getmembers(module):
                    # Check @tool decorated functions
                    if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                        tool_instance = getattr(obj, "Tool", None)
                        if tool_instance:
                            name = tool_instance.name
                            description = tool_instance.description or "No description"
                            cost_tier = getattr(tool_instance, "cost_tier", None)
                            cost_str = cost_tier.value if cost_tier else "unknown"
                            discovered_tools.append((name, description, cost_str))
                    # Check BaseTool class instances
                    elif (
                        inspect.isclass(obj)
                        and issubclass(obj, BaseToolClass)
                        and obj is not BaseToolClass
                        and hasattr(obj, "name")
                    ):
                        try:
                            tool_instance = obj()
                            name = tool_instance.name
                            description = tool_instance.description or "No description"
                            cost_tier = getattr(tool_instance, "cost_tier", None)
                            cost_str = cost_tier.value if cost_tier else "unknown"
                            discovered_tools.append((name, description, cost_str))
                        except Exception:
                            # Skip tools that can't be instantiated
                            pass
            except Exception as e:
                # Log but continue with other modules
                pass

    return discovered_tools


@tools_app.command("list")
def list_tools(  # Keep the command name as 'list' for the CLI
    profile: str = typer.Option(
        "default",
        "--profile",
        "-p",
        help="Profile to use from profiles.yaml to initialize agent for tool listing.",
    ),
    lightweight: bool = typer.Option(
        False,
        "--lightweight",
        "-l",
        help="Fast discovery mode without full agent initialization (shows all tools, no enabled/disabled status).",
    ),
) -> None:
    """List all available tools with their descriptions.

    By default, initializes the full agent to show enabled/disabled status.
    Use --lightweight for faster listing without agent initialization.

    Examples:
        victor tools list
        victor tools list --lightweight
        victor tools list -l
    """
    if lightweight:
        _list_tools_lightweight()
    else:
        asyncio.run(_list_tools_async(profile))


def _list_tools_lightweight() -> None:
    """List tools using lightweight discovery (no agent initialization)."""
    console.print("[dim]Discovering tools (lightweight mode)...[/]")

    try:
        discovered_tools = _discover_tools_lightweight()

        if not discovered_tools:
            console.print("[yellow]No tools found.[/]")
            return

        # Remove duplicates and sort
        unique_tools = {}
        for name, description, cost_tier in discovered_tools:
            if name not in unique_tools:
                unique_tools[name] = (description, cost_tier)

        table = Table(title="Available Tools (Lightweight Discovery)", show_header=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Cost Tier", style="yellow")

        # Sort by name
        for name in sorted(unique_tools.keys()):
            description, cost_tier = unique_tools[name]
            # Take only the first line of description
            short_desc = description.split("\n")[0][:80]
            if len(description.split("\n")[0]) > 80:
                short_desc += "..."
            table.add_row(name, short_desc, cost_tier)

        console.print(table)
        console.print(f"\n[dim]Found {len(unique_tools)} tools[/]")
        console.print("[dim]Note: Use without --lightweight for enabled/disabled status[/]")

    except Exception as e:
        console.print(f"[red]Error discovering tools:[/] {e}")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)


async def _list_tools_async(profile: str) -> None:
    """List tools with full agent initialization (shows enabled/disabled status)."""
    from victor.agent.orchestrator import AgentOrchestrator

    console.print(f"[dim]Initializing agent with profile '{profile}' to list tools...[/]")
    settings = load_settings()
    agent = None
    try:
        # AgentOrchestrator.from_settings is an async static method
        agent = await AgentOrchestrator.from_settings(settings, profile)

        # Retrieve tools from the agent's ToolRegistry
        # list_tools(only_enabled=False) gets all registered tools, regardless of current enable/disable status
        all_tools = agent.tools.list_tools(only_enabled=False)

        if not all_tools:
            console.print("[yellow]No tools found.[/]")
            return

        table = Table(title="Available Tools", show_header=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Enabled", style="yellow")

        # Get the current enabled/disabled status for each tool
        tool_states = agent.tools.get_tool_states()

        # Sort tools by name for consistent output
        for tool in sorted(all_tools, key=lambda t: t.name):
            # Take only the first line of the description for brevity in the table
            description = (
                tool.description.split("\\n")[0]
                if tool.description
                else "No description available."
            )
            # Check the actual enabled status from the tool_states map
            enabled_status = (
                "[green]✓ Yes[/]" if tool_states.get(tool.name, False) else "[dim]✗ No[/]"
            )
            table.add_row(tool.name, description, enabled_status)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing tools:[/] {e}")
        # Log full traceback for debugging purposes during development
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)
    finally:
        # Ensure agent resources are properly cleaned up
        if agent:
            await agent.graceful_shutdown()
