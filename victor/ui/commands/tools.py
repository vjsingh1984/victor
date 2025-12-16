import typer
import asyncio
from rich.console import Console
from rich.table import Table

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings

# ToolDefinition is imported for type hinting, though not directly used in the logic
from victor.providers.base import ToolDefinition  # noqa: F401


tools_app = typer.Typer(name="tools", help="Manage Victor's integrated tools.")
console = Console()


@tools_app.command("list")
def list_tools(  # Keep the command name as 'list' for the CLI
    profile: str = typer.Option(
        "default",
        "--profile",
        "-p",
        help="Profile to use from profiles.yaml to initialize agent for tool listing.",
    ),
) -> None:
    """List all available tools with their descriptions."""
    asyncio.run(_list_tools_async(profile))


async def _list_tools_async(profile: str) -> None:
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
