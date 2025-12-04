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


def _check_tui_compatibility() -> tuple[bool, str]:
    """Check if the terminal supports TUI features.

    Returns:
        Tuple of (is_compatible, reason_if_not)
    """
    # Check if running in a TTY
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False, "Not running in an interactive terminal (no TTY)"

    # Check terminal size
    try:
        import shutil

        cols, rows = shutil.get_terminal_size()
        if cols < 40 or rows < 10:
            return False, f"Terminal too small ({cols}x{rows}). Need at least 40x10"
    except Exception:
        pass  # Best effort check

    # Check for TERM environment variable
    term = os.environ.get("TERM", "")
    if term in ("dumb", ""):
        return False, f"Unsupported terminal type: '{term}'"

    # Check for known problematic environments
    if os.environ.get("EMACS"):
        return False, "Running in Emacs shell (use M-x shell instead)"

    # Check if Textual can import successfully
    try:
        from textual.app import App  # noqa: F401

        return True, ""
    except ImportError as e:
        return False, f"Textual library not available: {e}"
    except Exception as e:
        return False, f"TUI initialization error: {e}"


# These imports are after the readline try/except block intentionally
from rich.console import Console  # noqa: E402
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


app = typer.Typer(
    name="victor",
    help="Victor - Enterprise-Ready AI Coding Assistant. Code to Victory with Any AI.",
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
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        "--cli",
        help="Disable TUI and use CLI mode with console logging.",
    ),
) -> None:
    """Victor - Enterprise-Ready AI Coding Assistant. Code to Victory with Any AI.

    \b
    Usage:
        victor              # Start interactive TUI mode (default)
        victor --no-tui     # Start interactive CLI mode (for debugging)
        victor chat "msg"   # One-shot message
        victor --help       # Show this help

    \b
    Examples:
        victor                          # Interactive TUI mode
        victor --no-tui                 # Interactive CLI mode with logs
        victor chat "hello"             # One-shot query
        victor chat --no-tui "hello"    # One-shot with debug output
    """
    # Only run if no subcommand is being invoked
    if ctx.invoked_subcommand is None:
        # Default: run interactive mode (TUI unless --no-tui)
        _run_default_interactive(force_cli=no_tui)


def _configure_tui_logging() -> None:
    """Configure logging for TUI mode.

    In TUI mode:
    - Default log level is WARNING (unless overridden by VICTOR_LOG_LEVEL)
    - Logs go to a file (~/.victor/logs/victor.log) instead of stdout
    """
    from pathlib import Path
    from victor.config.settings import get_project_paths

    # TUI mode defaults to WARNING level to avoid cluttering the UI
    log_level = os.getenv("VICTOR_LOG_LEVEL", "WARNING").upper()
    if log_level == "WARN":
        log_level = "WARNING"

    # Create logs directory
    log_dir = get_project_paths().global_logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "victor.log"

    # Configure logging to file only for TUI mode
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
        force=True,
    )

    # Suppress noisy third-party loggers
    from victor.agent.debug_logger import configure_logging_levels

    configure_logging_levels(log_level)


def _run_default_interactive(force_cli: bool = False) -> None:
    """Run the default interactive mode with default options.

    Args:
        force_cli: If True, use CLI mode instead of TUI.
    """
    if force_cli:
        # CLI mode - configure console logging
        log_level = os.getenv("VICTOR_LOG_LEVEL", "INFO").upper()
        if log_level == "WARN":
            log_level = "WARNING"
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
    else:
        # TUI mode - configure file-based logging
        _configure_tui_logging()

    settings = load_settings()
    _setup_safety_confirmation()
    asyncio.run(run_interactive(settings, "default", True, False, force_cli=force_cli))


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
        help="Set logging level (DEBUG, INFO, WARN, ERROR). Defaults to INFO or VICTOR_LOG_LEVEL env var.",
        case_sensitive=False,
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable extended thinking/reasoning mode (Claude models). Shows model's reasoning process.",
    ),
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        "--cli",
        help="Disable TUI and use CLI mode. Useful for debugging with console logs.",
    ),
) -> None:
    """Start interactive chat or send a one-shot message.

    By default, Victor uses a modern TUI (Text User Interface) for interactive mode.
    Use --no-tui or --cli to switch to classic CLI mode with console log output.

    Examples:
        # Interactive TUI mode (default)
        victor chat

        # CLI mode with debug logging (for debugging)
        victor chat --no-tui --log-level DEBUG

        # One-shot command
        victor chat "Write a Python function to calculate Fibonacci numbers"

        # Use specific profile
        victor chat --profile claude "Explain how async/await works"

        # Enable thinking mode (shows reasoning process)
        victor chat --thinking "Explain the best way to implement a caching layer"
    """
    # Configure logging based on CLI argument or environment variable
    if log_level is None:
        log_level = os.getenv("VICTOR_LOG_LEVEL", "INFO")

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

    # Configure logging based on mode
    # TUI mode: log to file at WARNING level by default (unless overridden)
    # CLI mode: log to console at specified level
    if not no_tui and not message:
        # TUI mode - use file-based logging to avoid cluttering the UI
        _configure_tui_logging()
    else:
        # CLI/one-shot mode - log to console
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # Override any existing configuration
        )
        # Silence noisy third-party loggers
        from victor.agent.debug_logger import configure_logging_levels

        configure_logging_levels(log_level)

    # Load settings
    settings = load_settings()

    # Set up safety confirmation callbacks for dangerous operations
    _setup_safety_confirmation()

    if message:
        # One-shot mode
        asyncio.run(run_oneshot(message, settings, profile, stream, thinking))
    else:
        # Interactive mode (TUI by default, CLI with --no-tui)
        asyncio.run(run_interactive(settings, profile, stream, thinking, force_cli=no_tui))


async def run_oneshot(
    message: str,
    settings: Any,
    profile: str,
    stream: bool,
    thinking: bool = False,
) -> None:
    """Run a single message and exit.

    Args:
        message: Message to send
        settings: Application settings
        profile: Profile name
        stream: Whether to stream response
        thinking: Whether to enable thinking mode
    """
    try:
        # Create agent with thinking mode if requested
        agent = await AgentOrchestrator.from_settings(settings, profile, thinking=thinking)

        if thinking:
            logger.info("Extended thinking mode enabled (for supported models like Claude)")

        # Check codebase index at startup (silent - only log if changes)
        await _check_codebase_index(os.getcwd(), console, silent=True)

        # Start background embedding preload to avoid blocking first query
        agent.start_embedding_preload()

        if stream and agent.provider.supports_streaming():
            async for chunk in agent.stream_chat(message):
                if chunk.content:
                    console.print(chunk.content, end="")
            console.print()  # New line at end
        else:
            response = await agent.chat(message)
            console.print(Markdown(response.content))

        await agent.provider.close()

    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(1)


async def run_interactive(
    settings: Any,
    profile: str,
    stream: bool,
    thinking: bool = False,
    force_cli: bool = False,
) -> None:
    """Run interactive mode (TUI or CLI fallback).

    Args:
        settings: Application settings
        profile: Profile name
        stream: Whether to stream responses (used in TUI)
        thinking: Whether to enable thinking mode
        force_cli: Force CLI mode even if TUI is available
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

        # Start background embedding preload to avoid blocking first query
        agent.start_embedding_preload()

        # Initialize slash command handler
        cmd_handler = SlashCommandHandler(console, settings, agent)

        # Check if TUI is supported
        tui_compatible, tui_reason = _check_tui_compatibility()

        if force_cli or not tui_compatible:
            if not force_cli and tui_reason:
                console.print(
                    Panel(
                        f"[yellow]TUI mode unavailable:[/] {tui_reason}\n\n"
                        "[dim]Falling back to CLI mode. For full TUI experience:[/]\n"
                        "  - Use a modern terminal (iTerm2, Windows Terminal, etc.)\n"
                        "  - Ensure terminal size is at least 40x10\n"
                        "  - Run in an interactive shell (not piped)\n\n"
                        "[dim]You can also use:[/]\n"
                        '  [cyan]victor chat "your message"[/] for one-shot queries',
                        title="Victor - CLI Mode",
                        border_style="yellow",
                    )
                )
            # Run CLI fallback mode
            await _run_cli_repl(agent, settings, cmd_handler, profile_config, stream)
        else:
            # Run full TUI mode
            from victor.ui.tui import VictorApp

            app = VictorApp(
                agent=agent,
                settings=settings,
                cmd_handler=cmd_handler,
                provider=profile_config.provider,
                model=profile_config.model,
            )
            await app.run_async()

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
                # Stream the response
                content_buffer = ""
                async for chunk in agent.stream_chat(user_input):
                    if chunk.content:
                        console.print(chunk.content, end="")
                        content_buffer += chunk.content
                console.print()  # Newline after streaming
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
def init() -> None:
    """Initialize configuration files."""
    from pathlib import Path

    config_dir = Path.home() / ".victor"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Copy example profiles if they don't exist
    profiles_file = config_dir / "profiles.yaml"
    if not profiles_file.exists():
        console.print(f"Creating default configuration at {profiles_file}")

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
        console.print("[green]✓[/] Configuration created successfully!")
    else:
        console.print("[yellow]Configuration already exists[/]")

    console.print(f"\nConfiguration directory: {config_dir}")
    console.print(f"Edit {profiles_file} to customize profiles")


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
