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

"""Command-line interface for Victor - Enterprise-Ready AI Coding Assistant."""

import asyncio
import logging
import os
import sys
from typing import Any, Optional

try:
    import readline  # noqa: F401 - readline modifies input() behavior

    _history_enabled = True
    readline.set_history_length(1000)
except Exception:
    _history_enabled = False

import typer


# These imports are after the readline try/except block intentionally
from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.prompt import Confirm, Prompt  # noqa: E402

from victor import __version__  # noqa: E402
from victor.agent.orchestrator import AgentOrchestrator  # noqa: E402
from victor.agent.safety import (  # noqa: E402
    ConfirmationRequest,
    RiskLevel,
    set_confirmation_callback,
)
from victor.config.settings import load_settings  # noqa: E402
from victor.ui.commands import SlashCommandHandler  # noqa: E402

# Configure default logging (can be overridden by CLI argument)
logger = logging.getLogger(__name__)


async def _check_codebase_index(cwd: str, console_obj: Console, silent: bool = False) -> None:
    """Check codebase index status at startup and reindex if needed.

    This ensures the semantic search index is up-to-date based on file mtimes.

    Args:
        cwd: Current working directory (codebase root)
        console_obj: Rich console for output
        silent: If True, only show output for actual changes
    """
    try:
        from victor.codebase.indexer import CodebaseIndex

        # Create index instance (without embeddings for quick check)
        index = CodebaseIndex(
            root_path=cwd,
            use_embeddings=False,  # Skip embeddings for startup check
            enable_watcher=False,  # Don't start watcher yet
        )

        # Check staleness based on mtimes
        is_stale, modified, deleted = index.check_staleness_by_mtime()

        if not is_stale:
            if not silent:
                logger.debug("Codebase index is up to date")
            return

        total_changes = len(modified) + len(deleted)

        if not silent:
            console_obj.print(
                f"[dim]Index stale: {len(modified)} modified, {len(deleted)} deleted files[/]"
            )

        # Perform quick reindex (without embeddings)
        if total_changes <= 10:
            await index.incremental_reindex()
            if not silent:
                console_obj.print(f"[green]Incrementally reindexed {total_changes} files[/]")
        else:
            await index.reindex()
            if not silent:
                console_obj.print(f"[green]Full reindex completed ({len(index.files)} files)[/]")

    except ImportError:
        logger.debug("Codebase indexer not available")
    except Exception as e:
        logger.debug(f"Codebase index check failed: {e}")


async def _preload_semantic_index(
    cwd: str,
    settings: Any,
    console_obj: Console,
    force: bool = False,
) -> bool:
    """Preload semantic codebase index with embeddings upfront.

    This builds the vector embeddings for semantic_code_search at startup,
    avoiding the 20-30 second delay on first search.

    Args:
        cwd: Current working directory (codebase root)
        settings: Application settings
        console_obj: Rich console for output
        force: Force rebuild even if index exists

    Returns:
        True if index was successfully built/loaded
    """
    try:
        from pathlib import Path
        from victor.tools.code_search_tool import _get_or_build_index, _INDEX_CACHE
        import time

        root_path = Path(cwd).resolve()

        # Check if already cached in memory
        cache_entry = _INDEX_CACHE.get(str(root_path))
        if cache_entry and not force:
            console_obj.print("[dim]✓ Semantic index already loaded[/]")
            return True

        console_obj.print("[dim]⏳ Building semantic code index (one-time)...[/]")
        start_time = time.time()

        # Build the index with embeddings
        index, rebuilt = await _get_or_build_index(root_path, settings, force_reindex=force)

        elapsed = time.time() - start_time

        if rebuilt:
            console_obj.print(f"[green]✓ Semantic index built in {elapsed:.1f}s[/]")
        else:
            console_obj.print(f"[green]✓ Semantic index loaded in {elapsed:.1f}s[/]")

        return True

    except ImportError as e:
        logger.warning(f"Semantic indexing dependencies not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to preload semantic index: {e}")
        console_obj.print(f"[yellow]⚠ Semantic index preload failed: {e}[/]")
        return False


app = typer.Typer(
    name="victor",
    help="Victor - Enterprise-Ready AI Coding Assistant.",
    add_completion=False,
)

console = Console()


async def _cli_confirmation_callback(request: ConfirmationRequest) -> bool:
    """Prompt user for confirmation of dangerous operations.

    Args:
        request: Confirmation request with risk details

    Returns:
        True if user confirms, False to cancel
    """
    # Format risk level with color
    risk_colors = {
        RiskLevel.SAFE: "green",
        RiskLevel.LOW: "blue",
        RiskLevel.MEDIUM: "yellow",
        RiskLevel.HIGH: "red",
        RiskLevel.CRITICAL: "bold red",
    }
    color = risk_colors.get(request.risk_level, "white")

    # Show warning panel
    console.print()
    console.print(
        Panel(
            f"[{color}]{request.risk_level.value.upper()} RISK[/{color}]\n\n"
            f"Tool: [bold]{request.tool_name}[/bold]\n"
            f"Action: {request.description}\n\n"
            + (
                "Details:\n" + "\n".join(f"  • {d}" for d in request.details)
                if request.details
                else ""
            ),
            title="⚠️  Confirmation Required",
            border_style=color,
        )
    )

    # Prompt for confirmation
    try:
        return Confirm.ask("Proceed with this operation?", default=False)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Operation cancelled[/dim]")
        return False


def _setup_safety_confirmation() -> None:
    """Set up the CLI confirmation callback for dangerous operations."""
    set_confirmation_callback(_cli_confirmation_callback)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"Victor v{__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Victor - Enterprise-Ready AI Coding Assistant.

    \b
    Usage:
        victor              # Start interactive CLI mode (default)
        victor chat "msg"   # One-shot message
        victor --help       # Show this help

    \b
    Examples:
        victor                          # Interactive CLI mode
        victor chat "hello"             # One-shot query
        victor chat --log-level DEBUG   # With debug logging
    """
    # Only run if no subcommand is being invoked
    if ctx.invoked_subcommand is None:
        _run_default_interactive()


def _run_default_interactive() -> None:
    """Run the default interactive CLI mode with default options."""
    # Configure console logging
    log_level = os.getenv("VICTOR_LOG_LEVEL", "WARNING").upper()
    if log_level == "WARN":
        log_level = "WARNING"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    settings = load_settings()
    _setup_safety_confirmation()
    asyncio.run(run_interactive(settings, "default", True, False))


@app.command()
def chat(
    message: Optional[str] = typer.Argument(
        None,
        help="Message to send to the agent (starts interactive mode if not provided)",
    ),
    profile: str = typer.Option(
        "default",
        "--profile",
        "-p",
        help="Profile to use from profiles.yaml",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream responses",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARN, ERROR). Defaults to WARNING or VICTOR_LOG_LEVEL env var.",
        case_sensitive=False,
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable extended thinking/reasoning mode (Claude models). Shows model's reasoning process.",
    ),
    # Automation-friendly options
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output response as JSON object (for automation/scripting).",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Output plain text without Rich formatting.",
    ),
    code_only: bool = typer.Option(
        False,
        "--code-only",
        help="Extract and output only code blocks from response.",
    ),
    stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read input from stdin (supports multi-line).",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress status messages (only output response).",
    ),
    input_file: Optional[str] = typer.Option(
        None,
        "--input-file",
        "-f",
        help="Read input from file instead of argument.",
    ),
    preindex: bool = typer.Option(
        False,
        "--preindex",
        help="Preload semantic code index at startup (avoids 20-30s delay on first search).",
    ),
) -> None:
    """Start interactive chat or send a one-shot message.

    For automation/scripting, use --json, --plain, or --code-only to get
    machine-parseable output.

    Examples:
        # Interactive CLI mode (default)
        victor chat

        # CLI mode with debug logging
        victor chat --log-level DEBUG

        # One-shot command
        victor chat "Write a Python function to calculate Fibonacci numbers"

        # Use specific profile
        victor chat --profile claude "Explain how async/await works"

        # Enable thinking mode (shows reasoning process)
        victor chat --thinking "Explain the best way to implement a caching layer"

        # Automation: JSON output for scripting
        victor chat --json "What is 2+2?"

        # Automation: Extract only code blocks
        victor chat --code-only "Write a function to check if a number is prime"

        # Automation: Read multi-line prompt from stdin
        echo "Write a Python function" | victor chat --stdin --code-only

        # Automation: Read prompt from file
        victor chat --input-file prompt.txt --json

        # Pipeline: quiet mode (no status messages)
        victor chat -q --plain "Summarize this" | head -10
    """
    # Import output formatter
    from victor.ui.output_formatter import InputReader, create_formatter

    # Determine if this is automation mode (non-interactive output)
    automation_mode = json_output or plain or code_only

    # Configure logging based on CLI argument or environment variable
    if log_level is None:
        # Use ERROR level for automation, WARNING for interactive to reduce clutter
        log_level = os.getenv("VICTOR_LOG_LEVEL", "ERROR" if automation_mode else "WARNING")

    log_level = log_level.upper()
    valid_levels = ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]

    if log_level not in valid_levels:
        console.print(
            f"[bold red]Error:[/] Invalid log level '{log_level}'. Valid options: {', '.join(valid_levels)}"
        )
        raise typer.Exit(1)

    # Map WARN to WARNING for Python logging compatibility
    if log_level == "WARN":
        log_level = "WARNING"

    # Configure logging - log to stderr for automation compatibility
    log_stream = sys.stderr if automation_mode else None
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing configuration
        stream=log_stream,
    )
    # Silence noisy third-party loggers
    from victor.agent.debug_logger import configure_logging_levels

    configure_logging_levels(log_level)

    # Create output formatter based on flags
    formatter = create_formatter(
        json_mode=json_output,
        plain=plain,
        code_only=code_only,
        quiet=quiet,
        stream=stream and not json_output,  # Don't stream for JSON mode
    )

    # Read input from various sources
    actual_message = InputReader.read_message(
        argument=message,
        from_stdin=stdin,
        input_file=input_file,
    )

    # Load settings
    settings = load_settings()

    # Set up safety confirmation callbacks for dangerous operations
    _setup_safety_confirmation()

    if actual_message:
        # One-shot mode
        asyncio.run(
            run_oneshot(
                actual_message,
                settings,
                profile,
                stream and not json_output,  # Don't stream for JSON
                thinking,
                formatter=formatter,
                preindex=preindex,
            )
        )
    elif stdin or input_file:
        # --stdin or --input-file was specified but no input received
        formatter.error("No input received from stdin or file")
        raise typer.Exit(1)
    else:
        # Interactive CLI mode
        asyncio.run(
            run_interactive(settings, profile, stream, thinking, preindex=preindex)
        )


async def run_oneshot(
    message: str,
    settings: Any,
    profile: str,
    stream: bool,
    thinking: bool = False,
    formatter: Optional[Any] = None,
    preindex: bool = False,
) -> None:
    """Run a single message and exit.

    Args:
        message: Message to send
        settings: Application settings
        profile: Profile name
        stream: Whether to stream response
        thinking: Whether to enable thinking mode
        formatter: Optional OutputFormatter for automation-friendly output
        preindex: Whether to preload semantic code index upfront
    """
    from victor.ui.output_formatter import OutputFormatter, OutputMode, create_formatter

    # Use provided formatter or create default
    if formatter is None:
        formatter = create_formatter()

    # Set one-shot mode in settings (orchestrator reads this to auto-continue on ASKING_INPUT)
    settings.one_shot_mode = True

    try:
        # Create agent with thinking mode if requested
        agent = await AgentOrchestrator.from_settings(settings, profile, thinking=thinking)

        if thinking:
            logger.info("Extended thinking mode enabled (for supported models like Claude)")

        # Check codebase index at startup (silent - only log if changes)
        await _check_codebase_index(os.getcwd(), console, silent=True)

        # Preload semantic code index if requested (avoids delay on first search)
        if preindex:
            await _preload_semantic_index(os.getcwd(), settings, console)

        # Start background embedding preload to avoid blocking first query
        agent.start_embedding_preload()

        content_buffer = ""
        usage_stats = None
        model_name = None

        if stream and agent.provider.supports_streaming():
            # Start live markdown rendering (RICH mode only)
            formatter.start_streaming()
            async for chunk in agent.stream_chat(message):
                # Handle status messages separately (tool status, thinking indicator)
                if chunk.metadata and "status" in chunk.metadata:
                    # Temporarily pause live rendering to show status
                    formatter.end_streaming()
                    formatter.status(chunk.metadata["status"])
                    formatter.start_streaming()
                elif chunk.metadata and "file_preview" in chunk.metadata:
                    # Show file content preview (like Claude Code)
                    formatter.end_streaming()
                    path = chunk.metadata.get("path", "")
                    preview = chunk.metadata["file_preview"]
                    # Display as a code block with syntax highlighting
                    from rich.syntax import Syntax
                    from rich.panel import Panel
                    ext = path.split(".")[-1] if "." in path else "txt"
                    syntax = Syntax(preview, ext, theme="monokai", line_numbers=False)
                    console.print(Panel(syntax, title=f"[dim]{path}[/]", border_style="dim"))
                    formatter.start_streaming()
                elif chunk.metadata and "edit_preview" in chunk.metadata:
                    # Show edit diff preview
                    formatter.end_streaming()
                    path = chunk.metadata.get("path", "")
                    preview = chunk.metadata["edit_preview"]
                    console.print(f"[dim]{path}:[/]")
                    for line in preview.split("\n"):
                        if line.startswith("-"):
                            console.print(f"[red]{line}[/]")
                        elif line.startswith("+"):
                            console.print(f"[green]{line}[/]")
                        else:
                            console.print(f"[dim]{line}[/]")
                    formatter.start_streaming()
                elif chunk.content:
                    formatter.stream_chunk(chunk.content)
                    content_buffer += chunk.content
            # Finalize response (also ends streaming)
            formatter.response(content=content_buffer)
        else:
            response = await agent.chat(message)
            content_buffer = response.content
            usage_stats = response.usage
            model_name = response.model
            formatter.response(
                content=content_buffer,
                usage=usage_stats,
                model=model_name,
            )

        await agent.provider.close()

    except Exception as e:
        formatter.error(str(e))
        raise typer.Exit(1)


async def run_interactive(
    settings: Any,
    profile: str,
    stream: bool,
    thinking: bool = False,
    preindex: bool = False,
) -> None:
    """Run interactive CLI mode.

    Args:
        settings: Application settings
        profile: Profile name
        stream: Whether to stream responses
        thinking: Whether to enable thinking mode
        preindex: Whether to preload semantic code index upfront
    """
    agent = None
    try:
        # Load profile info
        try:
            profiles = settings.load_profiles()
        except Exception as e:
            console.print(f"[bold red]Error loading profiles:[/] {e}")
            raise typer.Exit(1)

        profile_config = profiles.get(profile)

        if not profile_config:
            console.print(f"[bold red]Error:[/] Profile '{profile}' not found")
            raise typer.Exit(1)

        # Create agent with thinking mode if requested
        agent = await AgentOrchestrator.from_settings(settings, profile, thinking=thinking)

        if thinking:
            logger.info("Extended thinking mode enabled (for supported models like Claude)")

        # Check codebase index at startup (show output in interactive mode)
        await _check_codebase_index(os.getcwd(), console, silent=False)

        # Preload semantic code index if requested (avoids delay on first search)
        if preindex:
            await _preload_semantic_index(os.getcwd(), settings, console)

        # Start background embedding preload to avoid blocking first query
        agent.start_embedding_preload()

        # Initialize slash command handler
        cmd_handler = SlashCommandHandler(console, settings, agent)

        # Run CLI REPL
        await _run_cli_repl(agent, settings, cmd_handler, profile_config, stream)

    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)
    finally:
        # Cleanup - always close provider if agent was created
        if agent and agent.provider:
            await agent.provider.close()


async def _run_cli_repl(
    agent: AgentOrchestrator,
    settings: Any,
    cmd_handler: SlashCommandHandler,
    profile_config: Any,
    stream: bool,
) -> None:
    """Run the CLI-based REPL (fallback for unsupported terminals).

    Args:
        agent: The agent orchestrator
        settings: Application settings
        cmd_handler: Slash command handler
        profile_config: Profile configuration
        stream: Whether to stream responses
    """
    console.print(
        Panel(
            f"[bold blue]Victor[/] - Enterprise-Ready AI Coding Assistant\n\n"
            f"[bold]Provider:[/] [cyan]{profile_config.provider}[/]  "
            f"[bold]Model:[/] [cyan]{profile_config.model}[/]\n\n"
            f"Type [bold]/help[/] for commands, [bold]/exit[/] or [bold]Ctrl+D[/] to quit.",
            title="Victor CLI",
            border_style="blue",
        )
    )

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[green]You[/]")

            if not user_input.strip():
                continue

            # Handle exit commands
            if user_input.strip().lower() in ("exit", "quit", "/exit", "/quit"):
                console.print("[dim]Goodbye![/]")
                break

            # Handle slash commands
            if cmd_handler.is_command(user_input):
                await cmd_handler.execute(user_input)
                continue

            # Handle clear command
            if user_input.strip().lower() == "clear":
                agent.reset_conversation()
                console.print("[green]Conversation cleared[/]")
                continue

            # Send message to agent
            console.print("[blue]Assistant:[/]")

            if stream:
                from victor.agent.response_sanitizer import sanitize_response

                # Stream the response with live markdown rendering
                content_buffer = ""
                live_display = None
                try:
                    live_display = Live(Markdown(""), console=console, refresh_per_second=10)
                    live_display.start()

                    async for chunk in agent.stream_chat(user_input):
                        # Handle status messages separately (tool status, thinking indicator)
                        if chunk.metadata and "status" in chunk.metadata:
                            live_display.stop()
                            console.print(f"[dim]{chunk.metadata['status']}[/]")
                            live_display = Live(
                                Markdown(content_buffer), console=console, refresh_per_second=10
                            )
                            live_display.start()
                        elif chunk.metadata and "file_preview" in chunk.metadata:
                            # Show file content preview (like Claude Code)
                            live_display.stop()
                            path = chunk.metadata.get("path", "")
                            preview = chunk.metadata["file_preview"]
                            from rich.syntax import Syntax
                            from rich.panel import Panel
                            ext = path.split(".")[-1] if "." in path else "txt"
                            syntax = Syntax(preview, ext, theme="monokai", line_numbers=False)
                            console.print(Panel(syntax, title=f"[dim]{path}[/]", border_style="dim"))
                            live_display = Live(
                                Markdown(content_buffer), console=console, refresh_per_second=10
                            )
                            live_display.start()
                        elif chunk.metadata and "edit_preview" in chunk.metadata:
                            # Show edit diff preview
                            live_display.stop()
                            path = chunk.metadata.get("path", "")
                            preview = chunk.metadata["edit_preview"]
                            console.print(f"[dim]{path}:[/]")
                            for line in preview.split("\n"):
                                if line.startswith("-"):
                                    console.print(f"[red]{line}[/]")
                                elif line.startswith("+"):
                                    console.print(f"[green]{line}[/]")
                                else:
                                    console.print(f"[dim]{line}[/]")
                            live_display = Live(
                                Markdown(content_buffer), console=console, refresh_per_second=10
                            )
                            live_display.start()
                        elif chunk.content:
                            # Accumulate raw content - don't sanitize per-chunk
                            # as that strips whitespace-only chunks (spaces between words)
                            content_buffer += chunk.content
                            # Update live display with accumulated markdown
                            live_display.update(Markdown(content_buffer))
                finally:
                    if live_display:
                        live_display.stop()

                # Sanitize the full response once streaming is complete
                # This removes thinking tokens and artifacts without losing spaces
                content_buffer = sanitize_response(content_buffer)
            else:
                # Non-streaming response
                response = await agent.chat(user_input)
                console.print(Markdown(response.content))

        except KeyboardInterrupt:
            console.print("\n[dim]Use /exit or Ctrl+D to quit[/]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            logger.exception("Error in CLI REPL")


@app.command()
def index(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force rebuild index even if up-to-date",
    ),
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="Path to codebase root directory",
    ),
) -> None:
    """Build semantic code search index for the codebase.

    This command builds vector embeddings for semantic_code_search,
    eliminating the 20-30 second delay on first search.

    The index is stored in {codebase}/.victor/embeddings/ and is
    automatically updated on subsequent runs if files changed.

    Examples:
        # Build index for current directory
        victor index

        # Force rebuild
        victor index --force

        # Build for specific path
        victor index --path /path/to/project
    """
    import time

    settings = load_settings()
    cwd = os.path.abspath(path)

    if not os.path.isdir(cwd):
        console.print(f"[red]Error:[/] Path '{path}' is not a directory")
        raise typer.Exit(1)

    console.print(f"[dim]Indexing codebase at: {cwd}[/]")

    async def _build_index() -> bool:
        return await _preload_semantic_index(cwd, settings, console, force=force)

    start_time = time.time()
    success = asyncio.run(_build_index())
    elapsed = time.time() - start_time

    if success:
        console.print(f"\n[green]✓ Indexing complete in {elapsed:.1f}s[/]")
        console.print(
            "[dim]Run [cyan]victor chat --preindex[/cyan] to use pre-loaded index, "
            "or index will load from disk on first semantic search.[/]"
        )
    else:
        console.print(f"\n[red]✗ Indexing failed[/]")
        raise typer.Exit(1)


@app.command(name="config-validate")
def config_validate(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed validation results",
    ),
    check_connectivity: bool = typer.Option(
        False,
        "--check-connectivity",
        "-c",
        help="Test provider connectivity (requires network/services)",
    ),
) -> None:
    """Validate configuration files and profiles.

    This command checks:
    - Configuration file exists and is valid YAML
    - All profiles have required fields
    - Provider configurations are valid
    - API keys are set for cloud providers
    - (Optional) Provider connectivity

    Examples:
        # Basic validation
        victor config-validate

        # Verbose output
        victor config-validate --verbose

        # Include connectivity checks
        victor config-validate --check-connectivity
    """
    import yaml

    errors: list[str] = []
    warnings: list[str] = []
    checks_passed = 0
    checks_total = 0

    settings = load_settings()
    config_dir = settings.get_config_dir()
    profiles_file = config_dir / "profiles.yaml"

    # Check 1: Config directory exists
    checks_total += 1
    if config_dir.exists():
        checks_passed += 1
        if verbose:
            console.print(f"[green]✓[/] Config directory exists: {config_dir}")
    else:
        errors.append(f"Config directory not found: {config_dir}")
        console.print(f"[red]✗[/] Config directory not found: {config_dir}")
        console.print("\nRun [bold]victor init[/] to create configuration")
        raise typer.Exit(1)

    # Check 2: Profiles file exists
    checks_total += 1
    if profiles_file.exists():
        checks_passed += 1
        if verbose:
            console.print(f"[green]✓[/] Profiles file exists: {profiles_file}")
    else:
        errors.append(f"Profiles file not found: {profiles_file}")
        console.print(f"[red]✗[/] Profiles file not found: {profiles_file}")
        console.print("\nRun [bold]victor init[/] to create configuration")
        raise typer.Exit(1)

    # Check 3: Valid YAML syntax
    checks_total += 1
    try:
        with open(profiles_file, "r") as f:
            raw_config = yaml.safe_load(f)
        checks_passed += 1
        if verbose:
            console.print("[green]✓[/] YAML syntax is valid")
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML syntax: {e}")
        console.print(f"[red]✗[/] Invalid YAML syntax: {e}")
        raise typer.Exit(1)

    # Check 4: Profiles section exists
    checks_total += 1
    if raw_config and "profiles" in raw_config:
        checks_passed += 1
        if verbose:
            console.print("[green]✓[/] 'profiles' section found")
    else:
        errors.append("Missing 'profiles' section in configuration")
        console.print("[red]✗[/] Missing 'profiles' section in configuration")
        raise typer.Exit(1)

    # Load and validate profiles
    try:
        profiles = settings.load_profiles()
    except Exception as e:
        errors.append(f"Failed to load profiles: {e}")
        console.print(f"[red]✗[/] Failed to load profiles: {e}")
        raise typer.Exit(1)

    # Check 5: At least one profile exists
    checks_total += 1
    if profiles:
        checks_passed += 1
        if verbose:
            console.print(f"[green]✓[/] Found {len(profiles)} profile(s)")
    else:
        warnings.append("No profiles defined")
        console.print("[yellow]⚠[/] No profiles defined")

    # Validate each profile
    from victor.providers.registry import ProviderRegistry

    available_providers = ProviderRegistry.list_providers()

    if verbose:
        console.print("\n[bold]Profile Validation:[/]")

    for name, profile in profiles.items():
        # Check provider is valid
        checks_total += 1
        if profile.provider in available_providers:
            checks_passed += 1
            if verbose:
                console.print(f"  [green]✓[/] [{name}] Provider '{profile.provider}' is valid")
        else:
            errors.append(f"[{name}] Unknown provider: {profile.provider}")
            console.print(f"  [red]✗[/] [{name}] Unknown provider: {profile.provider}")

        # Check model is specified
        checks_total += 1
        if profile.model:
            checks_passed += 1
            if verbose:
                console.print(f"  [green]✓[/] [{name}] Model specified: {profile.model}")
        else:
            errors.append(f"[{name}] No model specified")
            console.print(f"  [red]✗[/] [{name}] No model specified")

        # Check temperature range
        checks_total += 1
        if 0.0 <= profile.temperature <= 2.0:
            checks_passed += 1
            if verbose:
                console.print(f"  [green]✓[/] [{name}] Temperature {profile.temperature} is valid")
        else:
            errors.append(f"[{name}] Temperature {profile.temperature} out of range [0.0, 2.0]")
            console.print(f"  [red]✗[/] [{name}] Temperature out of range")

        # Check max_tokens
        checks_total += 1
        if profile.max_tokens > 0:
            checks_passed += 1
            if verbose:
                console.print(f"  [green]✓[/] [{name}] Max tokens {profile.max_tokens} is valid")
        else:
            errors.append(f"[{name}] Invalid max_tokens: {profile.max_tokens}")
            console.print(f"  [red]✗[/] [{name}] Invalid max_tokens")

        # Check API keys for cloud providers
        if profile.provider in ["anthropic", "openai", "google", "xai", "grok"]:
            checks_total += 1
            provider_settings = settings.get_provider_settings(profile.provider)
            api_key = provider_settings.get("api_key")
            if api_key:
                checks_passed += 1
                if verbose:
                    console.print(
                        f"  [green]✓[/] [{name}] API key configured for {profile.provider}"
                    )
            else:
                warnings.append(f"[{name}] No API key for {profile.provider}")
                console.print(
                    f"  [yellow]⚠[/] [{name}] No API key configured for {profile.provider}"
                )

    # Optional connectivity checks
    if check_connectivity:
        console.print("\n[bold]Connectivity Checks:[/]")
        asyncio.run(_check_connectivity(settings, profiles, verbose))

    # Summary
    console.print("\n" + "─" * 50)
    if errors:
        console.print(f"[red]Validation failed with {len(errors)} error(s)[/]")
        for err in errors:
            console.print(f"  [red]•[/] {err}")
        raise typer.Exit(1)
    elif warnings:
        console.print(f"[yellow]Validation passed with {len(warnings)} warning(s)[/]")
        for warn in warnings:
            console.print(f"  [yellow]•[/] {warn}")
        console.print(f"\n[green]✓[/] {checks_passed}/{checks_total} checks passed")
    else:
        console.print(
            f"[green]✓ Validation passed![/] {checks_passed}/{checks_total} checks passed"
        )


async def _check_connectivity(settings: Any, profiles: dict, verbose: bool) -> None:
    """Check provider connectivity for profiles."""
    checked_providers: set[str] = set()

    for _name, profile in profiles.items():
        provider = profile.provider
        if provider in checked_providers:
            continue
        checked_providers.add(provider)

        if provider == "ollama":
            from victor.providers.ollama_provider import OllamaProvider

            provider_settings = settings.get_provider_settings(provider)
            try:
                ollama = OllamaProvider(**provider_settings)
                models = await ollama.list_models()
                if models:
                    console.print(f"  [green]✓[/] Ollama: Connected ({len(models)} models)")
                else:
                    console.print("  [yellow]⚠[/] Ollama: Connected but no models installed")
                await ollama.close()
            except Exception as e:
                console.print(f"  [red]✗[/] Ollama: Cannot connect - {e}")
        else:
            if verbose:
                console.print(f"  [dim]→[/] {provider}: Connectivity check not implemented")


@app.command()
def init(
    update: bool = typer.Option(False, "--update", "-u", help="Update existing init.md preserving user edits"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing init.md completely"),
    learn: bool = typer.Option(False, "--learn", "-L", help="Enhance with conversation history insights"),
    index: bool = typer.Option(False, "--index", "-i", help="Use SQLite symbol store for multi-language analysis"),
    deep: bool = typer.Option(False, "--deep", "-d", help="Use LLM for deep analysis (any language)"),
    symlinks: bool = typer.Option(False, "--symlinks", "-l", help="Create CLAUDE.md and other tool aliases"),
    config_only: bool = typer.Option(False, "--config", "-c", help="Only setup global config, skip project analysis"),
) -> None:
    """Initialize project context and configuration.

    This command does two things:
    1. Creates global config files (~/.victor/profiles.yaml) if missing
    2. Analyzes your codebase and creates .victor/init.md (like CLAUDE.md)

    Examples:
        victor init              # Full initialization (config + project analysis)
        victor init --update     # Update analysis, preserve user edits
        victor init --force      # Regenerate init.md completely
        victor init --learn      # Include conversation history insights
        victor init --index      # Multi-language symbol indexing
        victor init --deep       # Use LLM for comprehensive analysis
        victor init --symlinks   # Also create CLAUDE.md symlink
        victor init --config     # Only setup global config
    """
    from victor.config.settings import get_project_paths, VICTOR_CONTEXT_FILE, VICTOR_DIR_NAME

    paths = get_project_paths()

    # Step 1: Global config setup
    config_dir = paths.global_victor_dir
    config_dir.mkdir(parents=True, exist_ok=True)

    profiles_file = config_dir / "profiles.yaml"
    if not profiles_file.exists():
        console.print(f"[dim]Creating default configuration at {profiles_file}[/]")

        # Create a basic default profile
        default_config = """profiles:
  default:
    provider: ollama
    model: qwen2.5-coder:7b
    temperature: 0.7
    max_tokens: 4096

providers:
  ollama:
    base_url: http://localhost:11434
"""
        profiles_file.write_text(default_config)
        console.print(f"[green]✓[/] Global config created at {config_dir}")
    else:
        console.print(f"[dim]Global config exists at {config_dir}[/]")

    if config_only:
        console.print(f"\nEdit {profiles_file} to customize profiles")
        return

    # Step 2: Project codebase analysis
    console.print("")
    target_path = paths.project_context_file
    paths.project_victor_dir.mkdir(parents=True, exist_ok=True)

    existing_content = None
    if target_path.exists():
        existing_content = target_path.read_text(encoding="utf-8")

        if not force and not update:
            console.print(f"[yellow]{VICTOR_CONTEXT_FILE} already exists at {target_path}[/]")
            console.print("")
            console.print("[bold]Options:[/]")
            console.print("  [cyan]victor init --update[/]   Merge new analysis (preserves your edits)")
            console.print("  [cyan]victor init --force[/]    Overwrite completely")
            console.print("  [cyan]victor init --learn[/]    Enhance with conversation history insights")
            console.print("  [cyan]victor init --index[/]    Multi-language symbol indexing")
            console.print("  [cyan]victor init --deep[/]     LLM-powered deep analysis")
            return

    if deep:
        console.print("[yellow]--deep requires an interactive session. Use 'victor' then '/init --deep'[/]")
        console.print("[dim]Running quick analysis instead...[/]")

    # Determine analysis mode
    if learn:
        console.print("[dim]Analyzing codebase + learning from conversation history...[/]")
    elif index:
        console.print("[dim]Indexing codebase (multi-language symbol analysis)...[/]")
    else:
        console.print("[dim]Analyzing codebase (quick mode)...[/]")

    try:
        import asyncio

        if learn:
            # Use SymbolStore + conversation history insights
            from victor.context.codebase_analyzer import generate_enhanced_init_md
            new_content = asyncio.run(generate_enhanced_init_md())
        elif index:
            # Use SymbolStore for accurate multi-language symbol extraction
            from victor.context.codebase_analyzer import generate_victor_md_from_index
            new_content = asyncio.run(generate_victor_md_from_index())
        else:
            # Quick regex-based analysis
            from victor.context.codebase_analyzer import generate_smart_victor_md
            new_content = generate_smart_victor_md()

        # Handle update mode
        if update and existing_content:
            # Import merge function from slash commands
            from victor.ui.slash_commands import SlashCommandHandler
            handler = SlashCommandHandler(console, None)
            content = handler._merge_init_content(existing_content, new_content)
            console.print("[dim]  Merged with existing content[/]")
        else:
            content = new_content

        # Write the file
        target_path.write_text(content, encoding="utf-8")
        console.print(f"[green]✓[/] Created {target_path}")

        # Show what was detected
        component_count = content.count("| `")
        pattern_count = content.count(". **") + content.count("Pattern:")
        console.print(f"[dim]  - Detected {component_count} key components[/]")
        console.print(f"[dim]  - Found {pattern_count} architecture patterns[/]")

        # Create symlinks if requested
        if symlinks:
            from victor.context.codebase_analyzer import (
                CONTEXT_FILE_ALIASES,
                create_context_symlinks,
            )

            console.print("\n[dim]Creating symlinks for other AI tools...[/]")
            results = create_context_symlinks()

            for alias, status in results.items():
                tool_name = CONTEXT_FILE_ALIASES.get(alias, "Unknown")
                if status == "created":
                    console.print(f"  [green]✓[/] {alias} -> {VICTOR_DIR_NAME}/{VICTOR_CONTEXT_FILE} ({tool_name})")
                elif status == "exists":
                    console.print(f"  [dim]○[/] {alias} (already linked)")
                elif status == "exists_file":
                    console.print(f"  [yellow]![/] {alias} (file exists, not a symlink)")

        console.print(f"\n[dim]Review and customize {target_path} as needed.[/]")

    except Exception as e:
        console.print(f"[red]Failed to create {VICTOR_DIR_NAME}/{VICTOR_CONTEXT_FILE}:[/] {e}")
        import traceback
        traceback.print_exc()


@app.command()
def keys(
    setup: bool = typer.Option(False, "--setup", "-s", help="Create API keys template file"),
    list_keys: bool = typer.Option(False, "--list", "-l", help="List configured providers"),
    provider: Optional[str] = typer.Option(None, "--set", help="Set API key for a provider"),
    keyring: bool = typer.Option(False, "--keyring", "-k", help="Store key in system keyring (secure)"),
    migrate: bool = typer.Option(False, "--migrate", help="Migrate keys from file to keyring"),
    delete_keyring: Optional[str] = typer.Option(None, "--delete-keyring", help="Delete key from keyring"),
) -> None:
    """Manage API keys for cloud providers.

    API keys can be stored in:
    - System keyring (recommended - encrypted storage)
    - File (~/.victor/api_keys.yaml)
    - Environment variables

    Keyring Support by Platform:
    - macOS: Keychain (built-in)
    - Windows: Credential Manager (built-in)
    - Linux: Secret Service (GNOME Keyring, KWallet)
    - iOS/Android: Not directly supported (use env vars)

    Examples:
        victor keys --setup                  # Create template file
        victor keys --list                   # Show configured providers
        victor keys --set anthropic          # Set key in file
        victor keys --set anthropic --keyring  # Set key in keyring (secure)
        victor keys --migrate                # Migrate all keys to keyring
        victor keys --delete-keyring openai  # Remove key from keyring
    """
    from victor.config.api_keys import (
        APIKeyManager,
        get_configured_providers,
        create_api_keys_template,
        DEFAULT_KEYS_FILE,
        PROVIDER_ENV_VARS,
        is_keyring_available,
        _get_key_from_keyring,
        _set_key_in_keyring,
        _delete_key_from_keyring,
    )
    from rich.table import Table
    import platform

    manager = APIKeyManager()

    # Show keyring platform info
    def get_keyring_info() -> tuple[str, str]:
        """Get keyring backend and platform info."""
        system = platform.system()
        if system == "Darwin":
            return "macOS Keychain", "[green]Supported[/]"
        elif system == "Windows":
            return "Windows Credential Manager", "[green]Supported[/]"
        elif system == "Linux":
            return "Secret Service (GNOME Keyring/KWallet)", "[green]Supported[/]"
        else:
            return "Unknown", "[yellow]May not be supported[/]"

    if delete_keyring:
        # Delete key from keyring
        provider_name = delete_keyring.lower()
        if not is_keyring_available():
            console.print("[red]Keyring not available.[/] Install with: pip install keyring")
            raise typer.Exit(1)

        if _delete_key_from_keyring(provider_name):
            console.print(f"[green]✓[/] Deleted [cyan]{provider_name}[/] from keyring")
        else:
            console.print(f"[yellow]Key for {provider_name} not found in keyring[/]")
        return

    if migrate:
        # Migrate keys from file to keyring
        if not is_keyring_available():
            console.print("[red]Keyring not available.[/] Install with: pip install keyring")
            console.print("\n[dim]Platform requirements:[/]")
            console.print("  macOS: Built-in Keychain support")
            console.print("  Windows: Built-in Credential Manager")
            console.print("  Linux: Install gnome-keyring or kwallet")
            raise typer.Exit(1)

        backend_name, status = get_keyring_info()
        console.print(f"[cyan]Keyring backend:[/] {backend_name} {status}")
        console.print()

        if not DEFAULT_KEYS_FILE.exists():
            console.print(f"[yellow]No keys file found at {DEFAULT_KEYS_FILE}[/]")
            raise typer.Exit(1)

        # Load keys from file
        import yaml
        with open(DEFAULT_KEYS_FILE) as f:
            data = yaml.safe_load(f) or {}

        api_keys = data.get("api_keys", data)
        migrated = 0
        failed = 0

        for prov, key in api_keys.items():
            if key and not key.startswith("your_"):
                if _set_key_in_keyring(prov, key):
                    console.print(f"  [green]✓[/] Migrated [cyan]{prov}[/]")
                    migrated += 1
                else:
                    console.print(f"  [red]✗[/] Failed: [cyan]{prov}[/]")
                    failed += 1

        console.print()
        console.print(f"[green]Migrated {migrated} keys to keyring[/]")
        if failed:
            console.print(f"[red]Failed to migrate {failed} keys[/]")

        if migrated > 0:
            console.print()
            console.print("[yellow]Security recommendation:[/]")
            console.print(f"  Delete the file: rm {DEFAULT_KEYS_FILE}")
            console.print("  Keys are now stored securely in system keyring")
        return

    if setup:
        # Create template file
        if DEFAULT_KEYS_FILE.exists():
            if not Confirm.ask(f"[yellow]{DEFAULT_KEYS_FILE} already exists. Overwrite?[/]"):
                console.print("[dim]Cancelled[/]")
                return

        DEFAULT_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        template = create_api_keys_template()
        DEFAULT_KEYS_FILE.write_text(template)

        # Set secure permissions
        import os

        os.chmod(DEFAULT_KEYS_FILE, 0o600)

        console.print(f"[green]✓[/] Created API keys template at [cyan]{DEFAULT_KEYS_FILE}[/]")
        console.print("\n[yellow]Next steps:[/]")
        console.print(f"  1. Edit [cyan]{DEFAULT_KEYS_FILE}[/]")
        console.print("  2. Replace placeholder values with your actual API keys")
        console.print("  3. File permissions already set to 0600 (owner-only)")
        console.print()
        console.print("[dim]Or use keyring for more secure storage:[/]")
        console.print("  victor keys --set anthropic --keyring")
        return

    if provider:
        # Set API key for a provider
        provider = provider.lower()
        if provider not in PROVIDER_ENV_VARS:
            console.print(f"[red]Unknown provider:[/] {provider}")
            console.print(f"Valid providers: {', '.join(sorted(PROVIDER_ENV_VARS.keys()))}")
            raise typer.Exit(1)

        # Check keyring availability if requested
        if keyring and not is_keyring_available():
            console.print("[red]Keyring not available.[/] Install with: pip install keyring")
            console.print("[dim]Falling back to file storage...[/]")
            keyring = False

        storage_type = "keyring" if keyring else "file"
        console.print(f"[cyan]Setting API key for {provider}[/] (storage: {storage_type})")
        console.print("[dim]Paste your API key (input hidden):[/]")

        import getpass

        key = getpass.getpass("")

        if not key.strip():
            console.print("[red]No key provided. Cancelled.[/]")
            raise typer.Exit(1)

        if manager.set_key(provider, key.strip(), use_keyring=keyring):
            location = "system keyring" if keyring else str(DEFAULT_KEYS_FILE)
            console.print(f"[green]✓[/] API key for [cyan]{provider}[/] saved to {location}")
        else:
            console.print("[red]Failed to save API key[/]")
            raise typer.Exit(1)
        return

    # Default: list configured providers with source info
    configured = get_configured_providers()

    table = Table(title="API Keys Status", show_header=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Status")
    table.add_column("Source")
    table.add_column("Env Var")

    provider_models = {
        "anthropic": "claude-3-opus, claude-sonnet-4",
        "openai": "gpt-4o, gpt-4-turbo",
        "google": "gemini-2.0-flash-exp, gemini-1.5-pro",
        "xai": "grok-beta, grok-2",
        "moonshot": "kimi-k2-thinking, kimi-k2-instruct",
        "deepseek": "deepseek-chat, deepseek-reasoner",
    }

    import os
    for prov, env_var in sorted(PROVIDER_ENV_VARS.items()):
        if prov in ("kimi",):  # Skip aliases
            continue

        # Determine source
        source = "[dim]--[/]"
        if os.environ.get(env_var):
            source = "[green]env[/]"
        elif is_keyring_available() and _get_key_from_keyring(prov):
            source = "[blue]keyring[/]"
        elif prov in configured:
            source = "[yellow]file[/]"

        status = "[green]✓ Configured[/]" if prov in configured else "[dim]Not set[/]"
        table.add_row(prov, status, source, env_var)

    console.print(table)

    # Show keyring status
    console.print()
    backend_name, status = get_keyring_info()
    keyring_status = "[green]✓ Available[/]" if is_keyring_available() else "[yellow]Not installed[/]"
    console.print(f"[dim]Keyring:[/] {backend_name} {keyring_status}")
    console.print(f"[dim]Keys file:[/] {DEFAULT_KEYS_FILE}")

    if not configured:
        console.print()
        console.print("[yellow]No API keys configured.[/]")
        console.print("  [cyan]victor keys --setup[/]          Create template file")
        console.print("  [cyan]victor keys --set anthropic --keyring[/]  Store in keyring (secure)")


@app.command()
def security(
    status: bool = typer.Option(True, "--status", "-s", help="Show security status"),
    trust_plugin: Optional[str] = typer.Option(None, "--trust-plugin", help="Trust a plugin by path"),
    untrust_plugin: Optional[str] = typer.Option(None, "--untrust-plugin", help="Remove plugin from trust store"),
    list_plugins: bool = typer.Option(False, "--list-plugins", "-l", help="List trusted plugins"),
    verify_cache: bool = typer.Option(False, "--verify-cache", help="Verify embedding cache integrity"),
    verify_all: bool = typer.Option(False, "--verify-all", "-a", help="Run comprehensive security verification"),
) -> None:
    """Security status and plugin trust management.

    View security posture and manage trusted plugins.

    Examples:
        victor security                    # Show security status
        victor security --verify-all       # Comprehensive security check
        victor security --verify-cache     # Verify cache integrity
        victor security --trust-plugin ./my_plugin
        victor security --list-plugins
    """
    from victor.config.secure_paths import (
        get_security_status,
        trust_plugin as do_trust_plugin,
        untrust_plugin as do_untrust_plugin,
        list_trusted_plugins,
        verify_cache_integrity,
        get_victor_dir,
        create_cache_manifest,
        get_sandbox_summary,
        validate_victor_dir_name,
        get_secure_home,
        get_secure_xdg_config_home,
        get_secure_xdg_data_home,
    )
    from victor.config.api_keys import is_keyring_available, get_configured_providers
    from rich.table import Table
    from rich.panel import Panel

    if trust_plugin:
        # Trust a plugin
        plugin_path = Path(trust_plugin).expanduser().resolve()
        if not plugin_path.exists():
            console.print(f"[red]Plugin not found:[/] {plugin_path}")
            raise typer.Exit(1)

        if do_trust_plugin(plugin_path):
            console.print(f"[green]✓[/] Trusted plugin: [cyan]{plugin_path.name}[/]")
        else:
            console.print("[red]Failed to trust plugin[/]")
            raise typer.Exit(1)
        return

    if untrust_plugin:
        # Untrust a plugin
        if do_untrust_plugin(untrust_plugin):
            console.print(f"[green]✓[/] Removed [cyan]{untrust_plugin}[/] from trust store")
        else:
            console.print(f"[yellow]Plugin {untrust_plugin} not found in trust store[/]")
        return

    if list_plugins:
        # List trusted plugins
        plugins = list_trusted_plugins()
        if not plugins:
            console.print("[dim]No trusted plugins[/]")
            return

        table = Table(title="Trusted Plugins", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Hash (truncated)")
        table.add_column("Path")

        for plugin in plugins:
            table.add_row(
                plugin["name"],
                plugin.get("hash", "")[:16] + "...",
                plugin.get("path", "")
            )

        console.print(table)
        return

    if verify_cache:
        # Verify cache integrity
        embeddings_dir = get_victor_dir() / "embeddings"
        if not embeddings_dir.exists():
            console.print("[dim]No embeddings cache found[/]")
            return

        console.print("[cyan]Verifying cache integrity...[/]")
        is_valid, tampered = verify_cache_integrity(embeddings_dir)

        if is_valid:
            console.print("[green]✓ Cache integrity verified[/]")
        else:
            console.print("[red]✗ Cache integrity check failed![/]")
            for file in tampered[:5]:
                console.print(f"  [red]Tampered:[/] {file}")
            if len(tampered) > 5:
                console.print(f"  ... and {len(tampered) - 5} more")

        # Offer to recreate manifest
        if not is_valid:
            if Confirm.ask("[yellow]Recreate manifest with current files?[/]"):
                if create_cache_manifest(embeddings_dir):
                    console.print("[green]✓ Manifest recreated[/]")
                else:
                    console.print("[red]Failed to recreate manifest[/]")
        return

    if verify_all:
        # Comprehensive security verification
        import os
        console.print(Panel.fit("[bold cyan]Comprehensive Security Verification[/]"))
        console.print()

        all_passed = True
        issues = []

        # 1. HOME manipulation check
        console.print("[cyan]1. Checking HOME environment validation...[/]")
        secure_home = get_secure_home()
        env_home = os.environ.get("HOME", "")
        if str(secure_home) == env_home:
            console.print("   [green]✓ HOME environment matches passwd database[/]")
        else:
            console.print(f"   [yellow]⚠ HOME differs: env={env_home}, passwd={secure_home}[/]")
            issues.append("HOME environment may be manipulated")

        # 2. VICTOR_DIR_NAME validation
        console.print("[cyan]2. Checking VICTOR_DIR_NAME validation...[/]")
        dir_name = os.environ.get("VICTOR_DIR_NAME", ".victor")
        validated_name, is_safe = validate_victor_dir_name(dir_name)
        if is_safe:
            console.print(f"   [green]✓ VICTOR_DIR_NAME '{validated_name}' is valid[/]")
        else:
            console.print(f"   [red]✗ VICTOR_DIR_NAME blocked: {dir_name}[/]")
            all_passed = False
            issues.append(f"VICTOR_DIR_NAME contains path traversal: {dir_name}")

        # 3. XDG path validation
        console.print("[cyan]3. Checking XDG path validation...[/]")
        xdg_config = get_secure_xdg_config_home()
        xdg_data = get_secure_xdg_data_home()
        console.print(f"   [green]✓ XDG_CONFIG_HOME: {xdg_config}[/]")
        console.print(f"   [green]✓ XDG_DATA_HOME: {xdg_data}[/]")

        # 4. Keyring availability
        console.print("[cyan]4. Checking keyring availability...[/]")
        if is_keyring_available():
            console.print("   [green]✓ System keyring is available[/]")
        else:
            console.print("   [yellow]⚠ System keyring not available (install keyring package)[/]")
            issues.append("Keyring not installed - API keys stored in plaintext file")

        # 5. Cache integrity
        console.print("[cyan]5. Checking cache integrity...[/]")
        embeddings_dir = get_victor_dir() / "embeddings"
        if embeddings_dir.exists():
            is_valid, tampered = verify_cache_integrity(embeddings_dir)
            if is_valid:
                console.print("   [green]✓ Embeddings cache integrity verified[/]")
            else:
                console.print(f"   [red]✗ Cache integrity failed: {len(tampered)} tampered files[/]")
                all_passed = False
                issues.append(f"Cache tampering detected: {len(tampered)} files modified")
        else:
            console.print("   [dim]• No embeddings cache found[/]")

        # 6. Plugin sandbox status
        console.print("[cyan]6. Checking plugin sandbox configuration...[/]")
        sandbox = get_sandbox_summary()
        policy = sandbox["policy"]
        console.print(f"   • Trust required: {'[green]Yes[/]' if policy['require_trust'] else '[yellow]No[/]'}")
        console.print(f"   • Network allowed: {'[yellow]Yes[/]' if policy['allow_network'] else '[green]Restricted[/]'}")
        console.print(f"   • Subprocess allowed: {'[yellow]Yes[/]' if policy['allow_subprocess'] else '[green]Restricted[/]'}")
        console.print(f"   • Trusted plugins: {sandbox['trusted_plugins']['count']}")

        # 7. API key source check
        console.print("[cyan]7. Checking API key sources...[/]")
        configured = get_configured_providers()
        for provider in configured[:5]:
            # Check source
            env_var = os.environ.get(f"{provider.upper()}_API_KEY")
            if env_var:
                console.print(f"   • {provider}: [green]environment (secure)[/]")
            else:
                console.print(f"   • {provider}: [yellow]file-based[/]")
        if len(configured) > 5:
            console.print(f"   ... and {len(configured) - 5} more")

        # Summary
        console.print()
        if all_passed and not issues:
            console.print("[bold green]✓ All security checks passed![/]")
        else:
            console.print("[bold yellow]Security Issues Found:[/]")
            for issue in issues:
                console.print(f"  [red]•[/] {issue}")

        return

    # Default: show security status
    sec_status = get_security_status()

    # Platform info
    console.print(Panel.fit(
        f"[cyan]Platform:[/] {sec_status['platform']['system']} {sec_status['platform']['release']}",
        title="Security Status"
    ))

    # Security checks table
    table = Table(show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    # Home security
    table.add_row(
        "Home Validation",
        "[green]✓ Secure[/]",
        sec_status["home_security"]["secure_home"]
    )

    # Keyring
    if sec_status["keyring"]["available"]:
        table.add_row(
            "System Keyring",
            "[green]✓ Available[/]",
            sec_status["keyring"]["backend"]
        )
    else:
        table.add_row(
            "System Keyring",
            "[yellow]Not installed[/]",
            "pip install keyring"
        )

    # API keys
    configured = get_configured_providers()
    table.add_row(
        "API Keys",
        f"[green]{len(configured)} configured[/]" if configured else "[dim]None[/]",
        ", ".join(configured[:3]) + ("..." if len(configured) > 3 else "")
    )

    # Plugin trust
    trusted_plugins = sec_status["plugins"]["trusted_count"]
    table.add_row(
        "Trusted Plugins",
        f"[green]{trusted_plugins}[/]" if trusted_plugins else "[dim]None[/]",
        f"{len(sec_status['plugins']['plugin_dirs'])} plugin dirs"
    )

    # Cache integrity
    if sec_status["cache_integrity"]["embeddings_verified"]:
        table.add_row(
            "Cache Integrity",
            "[green]✓ Verified[/]",
            "Embeddings cache valid"
        )
    else:
        table.add_row(
            "Cache Integrity",
            "[dim]Not verified[/]",
            "Run --verify-cache"
        )

    console.print(table)

    # Recommendations
    recommendations = []
    if not sec_status["keyring"]["available"]:
        recommendations.append("Install keyring: [cyan]pip install keyring[/]")
    if not configured:
        recommendations.append("Configure API keys: [cyan]victor keys --setup[/]")
    if not sec_status["cache_integrity"]["embeddings_verified"]:
        recommendations.append("Verify cache: [cyan]victor security --verify-cache[/]")

    if recommendations:
        console.print()
        console.print("[yellow]Recommendations:[/]")
        for rec in recommendations:
            console.print(f"  • {rec}")


@app.command()
def providers() -> None:
    """List all available providers."""
    from victor.providers.registry import ProviderRegistry
    from rich.table import Table

    available_providers = ProviderRegistry.list_providers()

    table = Table(title="Available Providers", show_header=True)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Features")

    provider_info = {
        "ollama": ("✅ Ready", "Local models, Free, Tool calling"),
        "anthropic": ("✅ Ready", "Claude, Tool calling, Streaming"),
        "openai": ("✅ Ready", "GPT-4/3.5, Function calling, Vision"),
        "google": ("✅ Ready", "Gemini, 1M context, Multimodal"),
        "xai": ("✅ Ready", "Grok, Real-time info, Vision"),
        "grok": ("✅ Ready", "Alias for xai"),
        "lmstudio": ("✅ Ready", "Local models via LMStudio"),
        "vllm": ("✅ Ready", "High-throughput local inference"),
        "moonshot": ("✅ Ready", "Kimi K2, 256K context, Reasoning"),
        "kimi": ("✅ Ready", "Alias for moonshot"),
        "deepseek": ("✅ Ready", "DeepSeek-V3, 128K, Cheap"),
        "groqcloud": ("✅ Ready", "Ultra-fast LPU, Free tier, Tool calling"),
    }

    for provider in sorted(available_providers):
        status, features = provider_info.get(provider, ("❓ Unknown", ""))
        table.add_row(provider, status, features)

    console.print(table)
    console.print("\n[dim]Use 'victor profiles' to see configured profiles[/]")


@app.command()
def profiles_cmd() -> None:
    """List configured profiles."""
    from rich.table import Table

    settings = load_settings()
    profiles = settings.load_profiles()

    if not profiles:
        console.print("[yellow]No profiles configured[/]")
        console.print("Run [bold]victor init[/] to create default configuration")
        return

    table = Table(title="Configured Profiles", show_header=True)
    table.add_column("Profile", style="cyan", no_wrap=True)
    table.add_column("Provider", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Temperature")
    table.add_column("Max Tokens")

    for name, profile in profiles.items():
        table.add_row(
            name,
            profile.provider,
            profile.model,
            f"{profile.temperature}",
            f"{profile.max_tokens}",
        )

    console.print(table)
    console.print(f"\n[dim]Config file: {settings.get_config_dir() / 'profiles.yaml'}[/]")


@app.command()
def models(
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help="Provider to list models from",
    ),
) -> None:
    """List available models for a provider."""
    asyncio.run(list_models_async(provider))


async def list_models_async(provider: str) -> None:
    """Async function to list models."""
    from rich.table import Table

    settings = load_settings()

    try:
        # Special handling for Ollama
        if provider == "ollama":
            from victor.providers.ollama_provider import OllamaProvider

            provider_settings = settings.get_provider_settings(provider)
            ollama = OllamaProvider(**provider_settings)

            try:
                models_list = await ollama.list_models()

                if not models_list:
                    console.print(f"[yellow]No models found for {provider}[/]")
                    console.print("\nPull a model with: [bold]ollama pull qwen2.5-coder:7b[/]")
                    return

                table = Table(title=f"Available Models ({provider})", show_header=True)
                table.add_column("Model", style="cyan", no_wrap=True)
                table.add_column("Size", style="yellow")
                table.add_column("Modified", style="dim")

                for model in models_list:
                    name = model.get("name", "unknown")
                    size = model.get("size", 0)
                    size_gb = size / (1024**3) if size else 0

                    modified = model.get("modified_at", "")
                    if modified:
                        # Format timestamp
                        from datetime import datetime

                        try:
                            dt = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                            modified = dt.strftime("%Y-%m-%d")
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Failed to parse timestamp '{modified}': {e}")
                            pass

                    table.add_row(
                        name,
                        f"{size_gb:.1f} GB" if size_gb > 0 else "unknown",
                        modified,
                    )

                console.print(table)
                console.print("\n[dim]Use a model with: [bold]victor --profile <profile>[/dim]")

                await ollama.close()

            except Exception as e:
                console.print(f"[red]Error listing models:[/] {e}")
                console.print("\nMake sure Ollama is running: [bold]ollama serve[/]")

        else:
            console.print(f"[yellow]Model listing not yet implemented for {provider}[/]")
            console.print("Currently only Ollama supports model listing via CLI")

    except Exception as e:
        console.print(f"[red]Error:[/] {e}")


@app.command()
def serve(
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
) -> None:
    """Start the Victor API server for IDE integrations.

    The server provides REST API endpoints that VS Code and other IDEs
    can connect to for AI-powered coding assistance.

    Examples:
        # Start server with defaults (localhost:8765)
        victor serve

        # Custom port
        victor serve --port 9000

        # Debug logging
        victor serve --log-level DEBUG

        # Allow external connections (use with caution)
        victor serve --host 0.0.0.0
    """
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

        # Run server
        server.run()

    except ImportError as e:
        console.print(f"[red]Error:[/] Missing dependency for server: {e}")
        console.print("\nInstall with: [bold]pip install aiohttp[/]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped[/]")
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def mcp(
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
) -> None:
    """Run Victor as an MCP server.

    This exposes Victor's 54+ tools through the Model Context Protocol,
    allowing MCP clients (like Claude Desktop) to use them.

    Examples:
        # Run MCP server (stdio mode)
        victor mcp

        # With debug logging
        victor mcp --log-level DEBUG

    To configure in Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json):

        {
          "mcpServers": {
            "victor": {
              "command": "victor",
              "args": ["mcp"]
            }
          }
        }
    """
    import sys

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
    import importlib
    import inspect
    import os
    import sys

    from victor.mcp.server import MCPServer
    from victor.tools.base import ToolRegistry

    # Create tool registry
    registry = ToolRegistry()

    # Dynamic tool discovery (same pattern as orchestrator)
    tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
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


@app.command()
def test_provider(
    provider: str = typer.Argument(..., help="Provider name to test"),
) -> None:
    """Test if a provider is working correctly."""
    console.print(f"Testing provider: [cyan]{provider}[/]")

    asyncio.run(test_provider_async(provider))


async def test_provider_async(provider: str) -> None:
    """Async function to test provider."""
    from victor.providers.registry import ProviderRegistry

    settings = load_settings()

    try:
        # Check if provider is registered
        if not ProviderRegistry.is_registered(provider):
            console.print(f"[red]✗[/] Provider '{provider}' not found")
            console.print(f"\nAvailable providers: {', '.join(ProviderRegistry.list_providers())}")
            return

        console.print("[green]✓[/] Provider registered")

        # Get provider settings
        provider_settings = settings.get_provider_settings(provider)

        # Check API key for cloud providers
        if provider in ["anthropic", "openai", "google", "xai", "grok"]:
            api_key = provider_settings.get("api_key")
            if not api_key:
                console.print(f"[red]✗[/] No API key configured for {provider}")
                console.print(f"\nSet environment variable: [bold]{provider.upper()}_API_KEY[/]")
                return
            console.print("[green]✓[/] API key configured")

        # For Ollama, test connection
        if provider == "ollama":
            from victor.providers.ollama_provider import OllamaProvider

            ollama = OllamaProvider(**provider_settings)
            try:
                models = await ollama.list_models()
                if models:
                    console.print(f"[green]✓[/] Ollama is running with {len(models)} models")
                else:
                    console.print("[yellow]⚠[/] Ollama is running but no models installed")
                    console.print("\nPull a model: [bold]ollama pull qwen2.5-coder:7b[/]")

                await ollama.close()

            except Exception as e:
                console.print(f"[red]✗[/] Cannot connect to Ollama: {e}")
                console.print("\nMake sure Ollama is running: [bold]ollama serve[/]")
                return

        console.print(f"\n[green]✓[/] Provider {provider} is ready to use!")

    except Exception as e:
        console.print(f"[red]✗[/] Error: {e}")


if __name__ == "__main__":
    app()
