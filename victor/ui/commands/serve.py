import typer
import asyncio
import logging
from enum import Enum
from typing import Optional
from rich.console import Console
from rich.panel import Panel

serve_app = typer.Typer(name="serve", help="Start the Victor API server for IDE integrations.")
console = Console()


class ServerBackend(str, Enum):
    """Server backend options."""

    AIOHTTP = "aiohttp"
    FASTAPI = "fastapi"


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
    backend: ServerBackend = typer.Option(
        ServerBackend.FASTAPI,
        "--backend",
        "-b",
        help="Server backend: 'fastapi' (default, with OpenAPI docs) or 'aiohttp' (legacy)",
    ),
):
    """Start the Victor API server for IDE integrations.

    The server provides REST API endpoints for IDE integrations (VS Code, JetBrains, etc.)
    and external tool access.

    Examples:
        victor serve                          # Start with FastAPI backend (default)
        victor serve -p 8000                  # FastAPI on custom port
        victor serve --backend aiohttp        # Use legacy aiohttp backend
    """
    if ctx.invoked_subcommand is None:
        _serve(host, port, log_level, profile, backend)


def _serve(host: str, port: int, log_level: str, profile: str, backend: ServerBackend):
    # Configure logging
    log_level = log_level.upper()
    if log_level == "WARN":
        log_level = "WARNING"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Backend-specific information
    backend_info = ""
    if backend == ServerBackend.FASTAPI:
        backend_info = f"\n[bold]Docs:[/] [cyan]http://{host}:{port}/docs[/]"

    console.print(
        Panel(
            f"[bold blue]Victor API Server[/]\n\n"
            f"[bold]Host:[/] [cyan]{host}[/]\n"
            f"[bold]Port:[/] [cyan]{port}[/]\n"
            f"[bold]Backend:[/] [cyan]{backend.value}[/]\n"
            f"[bold]Profile:[/] [cyan]{profile}[/]"
            f"{backend_info}\n\n"
            f"[dim]Press Ctrl+C to stop[/]",
            title="Victor Server",
            border_style="blue",
        )
    )

    if backend == ServerBackend.FASTAPI:
        asyncio.run(_run_fastapi_server(host, port, profile))
    else:
        asyncio.run(_run_aiohttp_server(host, port, profile))


async def _run_aiohttp_server(host: str, port: int, profile: str) -> None:
    """Run the aiohttp API server (legacy backend)."""
    try:
        from pathlib import Path

        from victor.integrations.api.server import VictorAPIServer

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


async def _run_fastapi_server(host: str, port: int, profile: str) -> None:
    """Run the FastAPI server (modern backend with OpenAPI docs)."""
    try:
        from pathlib import Path

        import uvicorn

        from victor.integrations.api.fastapi_server import VictorFastAPIServer

        server = VictorFastAPIServer(
            host=host,
            port=port,
            workspace_root=str(Path.cwd()),
        )

        config = uvicorn.Config(
            server.app,
            host=host,
            port=port,
            log_level="info",
        )
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()

    except ImportError as e:
        console.print(f"[red]Error:[/] Missing dependency for FastAPI server: {e}")
        console.print("\nInstall with: [bold]pip install fastapi uvicorn[/]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
