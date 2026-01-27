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
    help="Start Victor servers (API, HITL, Workflow Editor, etc.).",
)
console = Console()
logger = logging.getLogger(__name__)


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
    enable_hitl: bool = typer.Option(
        True,
        "--enable-hitl/--no-hitl",
        help="Enable HITL (Human-in-the-Loop) endpoints for workflow approvals (default: enabled)",
    ),
    hitl_auth_token: Optional[str] = typer.Option(
        None,
        "--hitl-auth-token",
        help="Bearer token for HITL endpoints (or set HITL_AUTH_TOKEN env var)",
        envvar="HITL_AUTH_TOKEN",
    ),
    minimal: bool = typer.Option(
        False,
        "--minimal",
        help="Run minimal server without workflow editor and extra UIs (API only)",
    ),
):
    """Start the Victor API server for IDE integrations.

    The server provides REST API endpoints for IDE integrations (VS Code, JetBrains, etc.)
    and external tool access. By default, includes workflow editor and HITL UI.

    Examples:
        victor serve                          # Start full server with all UIs (default)
        victor serve --minimal                # Start minimal server (API only, no UIs)
        victor serve -p 9000                  # Custom port
        victor serve --no-hitl                # Disable HITL endpoints
        victor serve --host 0.0.0.0           # Listen on all interfaces

    Available UIs (when not using --minimal):
        - Landing Page:     http://localhost:8765/
        - API Docs:         http://localhost:8765/docs
        - Workflow Editor:  http://localhost:8765/ui/workflow-editor
        - HITL Approvals:   http://localhost:8765/ui/hitl (if HITL enabled)
    """
    if ctx.invoked_subcommand is None:
        _serve(host, port, log_level, profile, enable_hitl, hitl_auth_token, minimal)


def _serve(
    host: str,
    port: int,
    log_level: Optional[str],
    profile: str,
    enable_hitl: bool = True,
    hitl_auth_token: Optional[str] = None,
    minimal: bool = False,
):
    # Normalize log level if provided
    if log_level is not None:
        log_level = log_level.upper()
        if log_level == "WARN":
            log_level = "WARNING"

    # Use centralized logging config (serve has INFO default in logging_config.yaml)
    setup_logging(command="serve", cli_log_level=log_level)

    # Determine which server to use
    if minimal:
        # Use minimal server (VictorFastAPIServer directly)
        asyncio.run(_run_minimal_server(host, port, profile, enable_hitl, hitl_auth_token))
    else:
        # Use unified orchestrator (full server with workflow editor and all UIs)
        asyncio.run(_run_unified_server(host, port, profile, enable_hitl, hitl_auth_token))


async def _run_minimal_server(
    host: str,
    port: int,
    profile: str,
    enable_hitl: bool = True,
    hitl_auth_token: Optional[str] = None,
) -> None:
    """Run the minimal FastAPI server (API only, no workflow editor or extra UIs)."""
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

        # Display minimal server banner
        hitl_info = ""
        if enable_hitl:
            hitl_info = f"\n[bold]HITL:[/] [green]Enabled[/] (http://{host}:{port}/api/v1/hitl)"

        console.print(
            Panel(
                f"[bold blue]Victor API Server (Minimal)[/]\n\n"
                f"[bold]Host:[/] [cyan]{host}[/]\n"
                f"[bold]Port:[/] [cyan]{port}[/]\n"
                f"[bold]Profile:[/] [cyan]{profile}[/]"
                f"\n[bold]API Docs:[/] [cyan]http://{host}:{port}/docs[/]"
                f"{hitl_info}\n\n"
                f"[dim]Press Ctrl+C to stop[/]",
                title="Victor Server",
                border_style="blue",
            )
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


async def _run_unified_server(
    host: str,
    port: int,
    profile: str,
    enable_hitl: bool = True,
    hitl_auth_token: Optional[str] = None,
) -> None:
    """Run the unified server with workflow editor and all UIs (default)."""
    try:
        from pathlib import Path

        import uvicorn

        from victor.integrations.api.unified_orchestrator import create_unified_server

        # Create unified server with all features
        app = create_unified_server(
            host=host,
            port=port,
            workspace_root=str(Path.cwd()),
            enable_hitl=enable_hitl,
            hitl_persistent=False,  # Use in-memory HITL store for serve command
            hitl_auth_token=hitl_auth_token,
            enable_cors=True,
        )

        # Display full server banner
        hitl_info = ""
        if enable_hitl:
            hitl_info = f"\n[bold]HITL UI:[/] [cyan]http://{host}:{port}/ui/hitl[/]"

        console.print(
            Panel(
                f"[bold blue]Victor API Server (Full)[/]\n\n"
                f"[bold]Host:[/] [cyan]{host}[/]\n"
                f"[bold]Port:[/] [cyan]{port}[/]\n"
                f"[bold]Profile:[/] [cyan]{profile}[/]\n\n"
                f"[bold]Available UIs:[/]"
                f"\n  [bold]Landing:[/]     [cyan]http://{host}:{port}/[/]"
                f"\n  [bold]API Docs:[/]     [cyan]http://{host}:{port}/docs[/]"
                f"\n  [bold]Workflows:[/]    [cyan]http://{host}:{port}/ui/workflow-editor[/]"
                f"{hitl_info}\n\n"
                f"[dim]Press Ctrl+C to stop[/]",
                title="Victor Server",
                border_style="blue",
            )
        )

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
        )
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()

    except ImportError as e:
        console.print(f"[red]Error:[/] Missing dependency for unified server: {e}")
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
            store = HITLStore()  # type: ignore[assignment]
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
