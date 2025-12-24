import typer
import asyncio
import logging
import sys
import os
import inspect
import importlib

from rich.console import Console

from victor.mcp.server import MCPServer
from victor.tools.base import ToolRegistry

mcp_app = typer.Typer(name="mcp", help="Run Victor as an MCP server.")
console = Console()


@mcp_app.callback(invoke_without_command=True)
def mcp(
    ctx: typer.Context,
    stdio: bool = typer.Option(
        True,
        "--stdio/--no-stdio",
        help="Run in stdio mode (for MCP clients)",
    ),
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Set logging level",
    ),
):
    """Run Victor as an MCP server."""
    if ctx.invoked_subcommand is None:
        _mcp(stdio, log_level)


def _mcp(stdio: bool, log_level: str):
    # Configure logging to stderr (stdout is for MCP protocol)
    log_level = log_level.upper()
    if log_level == "WARN":
        log_level = "WARNING"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True,
    )

    if stdio:
        asyncio.run(_run_mcp_server())
    else:
        console.print("[red]Only stdio mode is currently supported[/]")
        raise typer.Exit(1)


async def _run_mcp_server() -> None:
    """Run MCP server with all registered tools."""
    # Create tool registry
    registry = ToolRegistry()

    # Dynamic tool discovery (same pattern as orchestrator)
    tools_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tools")
    excluded_files = {"__init__.py", "base.py", "decorators.py", "semantic_selector.py"}
    registered_tools_count = 0

    for filename in os.listdir(tools_dir):
        if filename.endswith(".py") and filename not in excluded_files:
            module_name = f"victor.tools.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for _name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                        registry.register(obj)
                        registered_tools_count += 1
            except Exception as e:
                print(f"Warning: Failed to load tools from {module_name}: {e}", file=sys.stderr)

    server = MCPServer(
        name="Victor MCP Server",
        version="1.0.0",
        tool_registry=registry,
    )

    print(f"Victor MCP Server starting with {registered_tools_count} tools", file=sys.stderr)
    await server.start_stdio_server()
