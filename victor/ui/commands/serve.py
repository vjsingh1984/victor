import typer
import asyncio
import logging
from enum import Enum
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from victor.ui.commands.utils import setup_logging

serve_app = typer.Typer(
    name="serve",
    help="Start Victor servers (API, HITL, etc.).",
)
console = Console()
logger = logging.getLogger(__name__)


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
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Set logging level (defaults to INFO for serve or VICTOR_LOG_LEVEL env var)",
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
    enable_hitl: bool = typer.Option(
        False,
        "--enable-hitl",
        help="Enable HITL (Human-in-the-Loop) endpoints for workflow approvals",
    ),
    hitl_auth_token: Optional[str] = typer.Option(
        None,
        "--hitl-auth-token",
        help="Bearer token for HITL endpoints (or set HITL_AUTH_TOKEN env var)",
        envvar="HITL_AUTH_TOKEN",
    ),
):
    """Start the Victor API server for IDE integrations.

    The server provides REST API endpoints for IDE integrations (VS Code, JetBrains, etc.)
    and external tool access.

    Examples:
        victor serve                          # Start with FastAPI backend (default)
        victor serve -p 8000                  # FastAPI on custom port
        victor serve --backend aiohttp        # Use legacy aiohttp backend
        victor serve --enable-hitl            # Enable HITL endpoints for approvals
    """
    if ctx.invoked_subcommand is None:
        _serve(host, port, log_level, profile, backend, enable_hitl, hitl_auth_token)


def _serve(
    host: str,
    port: int,
    log_level: Optional[str],
    profile: str,
    backend: ServerBackend,
    enable_hitl: bool = False,
    hitl_auth_token: Optional[str] = None,
):
    # Normalize log level if provided
    if log_level is not None:
        log_level = log_level.upper()
        if log_level == "WARN":
            log_level = "WARNING"

    # Use centralized logging config (serve has INFO default in logging_config.yaml)
    setup_logging(command="serve", cli_log_level=log_level)

    # Backend-specific information
    backend_info = ""
    if backend == ServerBackend.FASTAPI:
        backend_info = f"\n[bold]Docs:[/] [cyan]http://{host}:{port}/docs[/]"

    # HITL-specific information
    hitl_info = ""
    if enable_hitl and backend == ServerBackend.FASTAPI:
        hitl_info = f"\n[bold]HITL UI:[/] [cyan]http://{host}:{port}/hitl/ui[/]"

    console.print(
        Panel(
            f"[bold blue]Victor API Server[/]\n\n"
            f"[bold]Host:[/] [cyan]{host}[/]\n"
            f"[bold]Port:[/] [cyan]{port}[/]\n"
            f"[bold]Backend:[/] [cyan]{backend.value}[/]\n"
            f"[bold]Profile:[/] [cyan]{profile}[/]"
            f"{backend_info}"
            f"{hitl_info}\n\n"
            f"[dim]Press Ctrl+C to stop[/]",
            title="Victor Server",
            border_style="blue",
        )
    )

    if backend == ServerBackend.FASTAPI:
        asyncio.run(_run_fastapi_server(host, port, profile, enable_hitl, hitl_auth_token))
    else:
        if enable_hitl:
            console.print("[yellow]Warning:[/] HITL endpoints only available with FastAPI backend")
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


async def _run_fastapi_server(
    host: str,
    port: int,
    profile: str,
    enable_hitl: bool = False,
    hitl_auth_token: Optional[str] = None,
) -> None:
    """Run the FastAPI server (modern backend with OpenAPI docs)."""
    try:
        from pathlib import Path

        import uvicorn

        from victor.integrations.api.fastapi_server import VictorFastAPIServer

        server = VictorFastAPIServer(
            host=host,
            port=port,
            workspace_root=str(Path.cwd()),
            enable_hitl=enable_hitl,
            hitl_auth_token=hitl_auth_token,
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


# ============================================================================
# HITL Server Subcommand
# ============================================================================


@serve_app.command(name="hitl")
def serve_hitl(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Host to bind the server to (0.0.0.0 for all interfaces)",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Port to listen on",
    ),
    auth_token: Optional[str] = typer.Option(
        None,
        "--auth-token",
        "-t",
        help="Bearer token for authentication (or set HITL_AUTH_TOKEN env var)",
        envvar="HITL_AUTH_TOKEN",
    ),
    require_auth: bool = typer.Option(
        False,
        "--require-auth",
        "-a",
        help="Require authentication for API access",
    ),
    persistent: bool = typer.Option(
        False,
        "--persistent",
        help="Use SQLite for persistent storage (survives restarts)",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Path to SQLite database (default: ~/.victor/hitl.db)",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Set logging level (defaults to INFO)",
    ),
):
    """Start the HITL (Human-in-the-Loop) approval server.

    This server provides a web UI and REST API for remote workflow approvals.
    Use this when deploying workflows to Docker, Kubernetes, Lambda, etc.
    where CLI-based approval isn't possible.

    Examples:
        victor serve hitl                        # Start on 0.0.0.0:8080 (in-memory)
        victor serve hitl --persistent           # Use SQLite storage
        victor serve hitl -p 9000                # Custom port
        victor serve hitl --require-auth -t secret  # With auth

    Access:
        Web UI: http://localhost:8080/hitl/ui
        API:    http://localhost:8080/hitl/requests
        Docs:   http://localhost:8080/docs
    """
    # Normalize log level if provided
    if log_level is not None:
        log_level = log_level.upper()
        if log_level == "WARN":
            log_level = "WARNING"

    setup_logging(command="serve", cli_log_level=log_level)

    auth_info = ""
    if require_auth:
        if auth_token:
            auth_info = "\n[bold]Auth:[/] [green]Enabled[/] (token configured)"
        else:
            auth_info = "\n[bold]Auth:[/] [yellow]Enabled but no token set![/]"
    else:
        auth_info = "\n[bold]Auth:[/] [dim]Disabled[/]"

    storage_info = ""
    if persistent:
        from pathlib import Path

        effective_db = db_path or str(Path.home() / ".victor" / "hitl.db")
        storage_info = f"\n[bold]Storage:[/] [green]SQLite[/] ({effective_db})"
    else:
        storage_info = "\n[bold]Storage:[/] [dim]In-memory[/]"

    console.print(
        Panel(
            f"[bold blue]Victor HITL Server[/]\n\n"
            f"[bold]Host:[/] [cyan]{host}[/]\n"
            f"[bold]Port:[/] [cyan]{port}[/]"
            f"{auth_info}"
            f"{storage_info}\n\n"
            f"[bold]Web UI:[/] [cyan]http://localhost:{port}/hitl/ui[/]\n"
            f"[bold]API:[/] [cyan]http://localhost:{port}/hitl/requests[/]\n"
            f"[bold]Docs:[/] [cyan]http://localhost:{port}/docs[/]\n\n"
            f"[dim]Press Ctrl+C to stop[/]",
            title="HITL Server",
            border_style="blue",
        )
    )

    asyncio.run(_run_hitl_server(host, port, require_auth, auth_token, persistent, db_path))


async def _run_hitl_server(
    host: str,
    port: int,
    require_auth: bool,
    auth_token: Optional[str],
    persistent: bool = False,
    db_path: Optional[str] = None,
) -> None:
    """Run the HITL API server."""
    try:
        import uvicorn

        from victor.workflows.hitl_api import (
            HITLStore,
            SQLiteHITLStore,
            create_hitl_app,
        )

        # Create appropriate store
        if persistent:
            store = SQLiteHITLStore(db_path=db_path)
            logger.info(f"Using SQLite HITL store: {store.db_path}")
        else:
            store = HITLStore()
            logger.info("Using in-memory HITL store")

        app = create_hitl_app(
            store=store,  # type: ignore[arg-type]
            require_auth=require_auth,
            auth_token=auth_token,
        )

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    except ImportError as e:
        console.print(f"[red]Error:[/] Missing dependency for HITL server: {e}")
        console.print("\nInstall with: [bold]pip install fastapi uvicorn[/]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]HITL server stopped[/]")
    except Exception as e:
        logger.exception("HITL server error")
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)
