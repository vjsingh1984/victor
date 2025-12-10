import typer
import asyncio
import logging
from rich.console import Console
from rich.panel import Panel

serve_app = typer.Typer(name="serve", help="Start the Victor API server for IDE integrations.")
console = Console()

@serve_app.callback(invoke_without_command=True)
def serve(
    ctx: typer.Context,
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind the server to",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        "-p",
        help="Port to listen on",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARN, ERROR)",
    ),
    profile: str = typer.Option(
        "default",
        "--profile",
        help="Profile to use for the server",
    ),
):
    """Start the Victor API server for IDE integrations."""
    if ctx.invoked_subcommand is None:
        _serve(host, port, log_level, profile)

def _serve(host: str, port: int, log_level: str, profile: str):
    # Configure logging
    log_level = log_level.upper()
    if log_level == "WARN":
        log_level = "WARNING"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    console.print(
        Panel(
            f"[bold blue]Victor API Server[/]\n\n"
            f"[bold]Host:[/] [cyan]{host}[/]\n"
            f"[bold]Port:[/] [cyan]{port}[/]\n"
            f"[bold]Profile:[/] [cyan]{profile}[/]\n\n"
            f"[dim]Press Ctrl+C to stop[/]",
            title="Victor Server",
            border_style="blue",
        )
    )

    asyncio.run(_run_server(host, port, profile))


async def _run_server(host: str, port: int, profile: str) -> None:
    """Run the API server."""
    try:
        from pathlib import Path

        from victor.api.server import VictorAPIServer

        server = VictorAPIServer(
            host=host,
            port=port,
            workspace_root=str(Path.cwd()),
        )

        # Run server asynchronously to avoid nested event loop errors
        runner = await server.start_async()
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    except ImportError as e:
        console.print(f"[red]Error:[/] Missing dependency for server: {e}")
        console.print("\nInstall with: [bold]pip install aiohttp[/]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
