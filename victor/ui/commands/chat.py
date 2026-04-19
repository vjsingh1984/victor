import typer
import os
import sys
import time
import uuid
from typing import Optional, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings, ProfileConfig
from victor.core.async_utils import run_sync
from victor.framework.shim import FrameworkShim
from victor.core.verticals import get_vertical, list_verticals
from victor.ui.output_formatter import InputReader, create_formatter
from victor.ui.commands.utils import (
    preload_semantic_index,
    check_codebase_index,
    get_rl_profile_suggestion,
    setup_safety_confirmation,
    setup_logging,
    graceful_shutdown,
)
from victor.workflows import (
    load_workflow_from_file,
    YAMLWorkflowError,
    StateGraphExecutor,
    ExecutorConfig,
)
from victor.workflows.visualization import (
    WorkflowVisualizer,
    OutputFormat as VizFormat,
    RenderBackend,
    get_available_backends,
)

# Contextual error formatting
try:
    from victor.framework.contextual_errors import format_exception_for_user
except ImportError:
    # Fallback if framework module is not available
    def format_exception_for_user(e):
        return str(e)


chat_app = typer.Typer(name="chat", help="Start interactive chat or send a one-shot message.")
console = Console()


def _display_skill_preview(con: Console, agent: Any, message: str) -> None:
    """Show skill auto-selection feedback before the LLM response.

    Does a preview match (deterministic, same as coordinator will use)
    to display which skill(s) will be activated.
    """
    matcher = getattr(agent, "_skill_matcher", None)
    if matcher is None or not getattr(matcher, "_initialized", False):
        return
    if getattr(agent, "_skill_auto_disabled", False):
        return
    try:
        matches = matcher.match_multiple_sync(message)
        if not matches:
            return
        if len(matches) == 1:
            skill, score = matches[0]
            con.print(f"[dim]\U0001f3af Skill: {skill.name} ({score:.2f})[/]")
        else:
            names = " \u2192 ".join(s.name for s, _ in matches)
            con.print(f"[dim]\U0001f3af Skills: {names}[/]")
    except Exception:
        pass


@chat_app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    message: Optional[str] = typer.Argument(
        None,
        help="Message to send (put BEFORE options, or use -m instead). Interactive mode if omitted.",
    ),
    message_opt: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="Message to send (alternative to positional arg, works anywhere in command).",
    ),
    # Core options (always visible - beginner friendly)
    profile: str = typer.Option(
        "default",
        "--profile",
        "-p",
        help="Profile to use from profiles.yaml",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Override provider (e.g., ollama, anthropic, openai). Uses provider's default model if --model not specified.",
        case_sensitive=False,
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override model identifier (optional if --provider specified).",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream responses in real-time",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable extended thinking/reasoning mode (Claude models). Shows model's reasoning process.",
    ),
    # Output options
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output response as JSON object (for automation/scripting).",
        rich_help_panel="Output Format",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Output plain text without Rich formatting.",
        rich_help_panel="Output Format",
    ),
    code_only: bool = typer.Option(
        False,
        "--code-only",
        help="Extract and output only code blocks from response.",
        rich_help_panel="Output Format",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress status messages (only output response).",
        rich_help_panel="Output Format",
    ),
    # Input options
    stdin: bool = typer.Option(
        False,
        "--stdin",
        help="Read input from stdin (supports multi-line).",
        rich_help_panel="Input",
    ),
    input_file: Optional[str] = typer.Option(
        None,
        "--input-file",
        "-f",
        help="Read input from file instead of argument.",
        rich_help_panel="Input",
    ),
    # Logging options
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARN, ERROR). Defaults to WARNING or VICTOR_LOG_LEVEL env var.",
        case_sensitive=False,
        rich_help_panel="Logging",
    ),
    # Advanced options (grouped for progressive disclosure)
    debug_modules: str = typer.Option(
        None,
        "--debug-modules",
        help="Comma-separated modules to set to DEBUG level (e.g., code_search,agent_adapter).",
        rich_help_panel="Advanced Logging",
    ),
    enable_observability: bool = typer.Option(
        True,
        "--observability/--no-observability",
        help="Enable observability integration for event tracking.",
        rich_help_panel="Advanced Logging",
    ),
    log_events: bool = typer.Option(
        False,
        "--log-events",
        help="Enable JSONL event logging to ~/.victor/logs/victor.log for dashboard visualization.",
        rich_help_panel="Advanced Logging",
    ),
    show_reasoning: bool = typer.Option(
        False,
        "--show-reasoning",
        help="Show LLM reasoning/thinking content in output.",
        rich_help_panel="Advanced Output",
    ),
    renderer: str = typer.Option(
        "auto",
        "--renderer",
        help="Renderer to use for streaming output: auto, rich, rich-text, or text.",
        case_sensitive=False,
        rich_help_panel="Advanced Output",
    ),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        help="Initial agent mode: build, plan, or explore.",
        case_sensitive=False,
        rich_help_panel="Advanced Agent Behavior",
    ),
    tool_budget: Optional[int] = typer.Option(
        None,
        "--tool-budget",
        help="Override tool call budget for this session.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    max_iterations: Optional[int] = typer.Option(
        None,
        "--max-iterations",
        help="Override maximum total iterations for this session.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    preindex: bool = typer.Option(
        False,
        "--preindex",
        help="Preload semantic code index at startup (avoids 20-30s delay on first search).",
        rich_help_panel="Advanced Agent Behavior",
    ),
    vertical: Optional[str] = typer.Option(
        None,
        "--vertical",
        "-V",
        help=f"Vertical template to use ({', '.join(list_verticals()) or 'coding, research, devops'}).",
        rich_help_panel="Advanced Agent Behavior",
    ),
    auto_skill: Optional[bool] = typer.Option(
        None,
        "--auto-skill/--no-auto-skill",
        help="Enable/disable automatic skill selection based on message content.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    enable_planning: Optional[bool] = typer.Option(
        None,
        "--planning/--no-planning",
        help="Enable structured planning for complex multi-step tasks.",
        rich_help_panel="Advanced Agent Behavior",
    ),
    planning_model: Optional[str] = typer.Option(
        None,
        "--planning-model",
        help="Override model for planning tasks (e.g., 'deepseek-chat').",
        rich_help_panel="Advanced Agent Behavior",
    ),
    # Workflow options (grouped separately)
    workflow: Optional[str] = typer.Option(
        None,
        "--workflow",
        "-w",
        help="Path to YAML workflow file to execute. Runs workflow instead of chat mode.",
        rich_help_panel="Workflow",
    ),
    validate_workflow: bool = typer.Option(
        False,
        "--validate",
        help="Validate YAML workflow file without executing. Use with --workflow.",
        rich_help_panel="Workflow",
    ),
    render_format: Optional[str] = typer.Option(
        None,
        "--render",
        "-r",
        help="Render workflow DAG (ascii, mermaid, d2, dot, plantuml, svg, png).",
        rich_help_panel="Workflow",
    ),
    render_output: Optional[str] = typer.Option(
        None,
        "--render-output",
        "-o",
        help="Output file for rendered diagram. Required for svg/png formats.",
        rich_help_panel="Workflow",
    ),
    # Session options (grouped separately)
    list_sessions: bool = typer.Option(
        False,
        "--sessions",
        help="List saved sessions and exit (top 20).",
        rich_help_panel="Session",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--sessionid",
        help="Resume specific session by ID.",
        rich_help_panel="Session",
    ),
    # Auth & Compatibility (expert options)
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help="Override base URL for local providers (ollama, lmstudio, vllm).",
        rich_help_panel="Expert Auth & Compatibility",
    ),
    auth_mode: Optional[str] = typer.Option(
        None,
        "--auth-mode",
        help="Authentication mode: 'api_key' (default) or 'oauth' (for OpenAI Codex, Qwen).",
        case_sensitive=False,
        rich_help_panel="Expert Auth & Compatibility",
    ),
    coding_plan: bool = typer.Option(
        False,
        "--coding-plan",
        help="Use coding plan endpoint (Z.AI). Routes to api.z.ai/api/coding/paas/v4/.",
        rich_help_panel="Expert Auth & Compatibility",
    ),
    legacy_mode: bool = typer.Option(
        False,
        "--legacy",
        help="Use legacy orchestrator creation path (bypasses FrameworkShim). For troubleshooting only.",
        rich_help_panel="Expert Auth & Compatibility",
    ),
    tui: bool = typer.Option(
        False,
        "--tui/--no-tui",
        help="Use modern TUI interface. Use --no-tui for simple CLI mode (default).",
        rich_help_panel="Expert Interface",
    ),
):
    """Start interactive chat or send a one-shot message."""
    if ctx.invoked_subcommand is None:
        renderer = renderer.lower()
        if renderer not in {"auto", "rich", "rich-text", "text"}:
            console.print(
                "[bold red]Error:[/] Invalid renderer. Choose from auto, rich, rich-text, text."
            )
            raise typer.Exit(1)

        if mode:
            mode = mode.lower()
            if mode not in {"build", "plan", "explore"}:
                console.print("[bold red]Error:[/] Invalid mode. Choose from build, plan, explore.")
                raise typer.Exit(1)

        # Handle workflow mode (--workflow and --validate/--render)
        if workflow or validate_workflow or render_format:
            if (validate_workflow or render_format) and not workflow:
                console.print(
                    "[bold red]Error:[/] --validate/--render require --workflow to specify a file."
                )
                raise typer.Exit(1)

            run_sync(
                run_workflow_mode(
                    workflow_path=workflow,  # type: ignore
                    validate_only=validate_workflow,
                    render_format=render_format,
                    render_output=render_output,
                    profile=profile,
                    vertical=vertical,
                    log_level=log_level,
                )
            )
            return

        automation_mode = json_output or plain or code_only

        # Smart TUI detection: disable if non-interactive terminal or automation mode
        use_tui = tui and not automation_mode and sys.stdin.isatty() and sys.stdout.isatty()

        # Use ERROR level for automation modes (cleaner output)
        if log_level is None and automation_mode:
            log_level = "ERROR"

        # Validate log level if explicitly provided
        if log_level is not None:
            log_level = log_level.upper()
            valid_levels = ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]

            if log_level not in valid_levels:
                console.print(
                    f"[bold red]Error:[/ ] Invalid log level '{log_level}'. Valid options: {', '.join(valid_levels)}"
                )
                raise typer.Exit(1)

            if log_level == "WARN":
                log_level = "WARNING"

            # For troubleshooting, prefer plain text output when verbose logging is enabled
            if log_level in {"DEBUG", "INFO"}:
                renderer = "text"

        # Use centralized logging config with session ID if resuming
        setup_logging(
            command="chat",
            cli_log_level=log_level,
            cli_debug_modules=debug_modules,
            stream=sys.stderr,
            session_id=session_id,  # Use --sessionid flag value for logging
        )

        # Handle --sessions flag (list sessions and exit)
        if list_sessions:
            from victor.agent.sqlite_session_persistence import (
                get_sqlite_session_persistence,
            )

            persistence = get_sqlite_session_persistence()
            sessions = persistence.list_sessions(limit=20)

            if not sessions:
                console.print("[dim]No sessions found[/]")
                raise typer.Exit(0)

            table = Table(title="Saved Sessions (top 20)")
            table.add_column("Session ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Provider", style="blue")
            table.add_column("Messages", justify="right")
            table.add_column("Created", style="dim")

            for session in sessions:
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(session["created_at"])
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_str = session["created_at"][:16]

                title = (
                    session["title"][:40] + "..."
                    if len(session["title"]) > 40
                    else session["title"]
                )
                table.add_row(
                    session["session_id"],
                    title,
                    session["model"],
                    session["provider"],
                    str(session["message_count"]),
                    date_str,
                )

            console.print(table)
            console.print(f"\n[dim]Total: {len(sessions)} session(s)[/]")
            console.print("[dim]Use 'victor chat --sessionid <id>' to resume a session[/]")
            raise typer.Exit(0)

        from victor.agent.debug_logger import configure_logging_levels

        # Apply debug logger levels if explicitly specified
        if log_level:
            configure_logging_levels(log_level)

        formatter = create_formatter(
            json_mode=json_output,
            plain=plain,
            code_only=code_only,
            quiet=quiet,
            stream=stream and not json_output,
        )

        # --message / -m option takes precedence over positional argument
        effective_message = message_opt or message

        actual_message = InputReader.read_message(
            argument=effective_message,
            from_stdin=stdin,
            input_file=input_file,
        )

        settings = load_settings()

        # Early configuration validation (P0: catch errors at startup, not runtime)
        try:
            from victor.config.validation import (
                validate_configuration,
                format_validation_result,
            )

            validation_result = validate_configuration(settings)
            if not validation_result.is_valid:
                # Configuration has errors - display and exit
                console.print("[bold red]Configuration Validation Failed:[/]")
                console.print(format_validation_result(validation_result))
                console.print("")
                console.print("[yellow]Run 'victor config validate' for detailed diagnostics[/]")
                raise typer.Exit(1)
        except Exception as e:
            # If validation itself fails, log it but don't block startup
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Configuration validation skipped due to error: {e}")

        # Apply CLI flags to settings
        if log_events:
            settings.observability.enable_observability_logging = True

        setup_safety_confirmation()

        # Apply provider/model/endpoint overrides by creating a synthetic profile
        if provider or model or endpoint:
            # Early validation: endpoint is only supported for local providers
            if endpoint and provider:
                provider_lower = provider.lower()
                local_providers = {"ollama", "lmstudio", "vllm", "mlx", "llama.cpp"}
                if provider_lower not in local_providers:
                    console.print(
                        f"[bold red]Error:[/] --endpoint is only supported for local providers "
                        f"(ollama, lmstudio, vllm, mlx, llama.cpp). Got: {provider}"
                    )
                    console.print(
                        "[dim]Hint: Cloud providers (anthropic, openai, google, etc.) use fixed endpoints.[/]"
                    )
                    raise typer.Exit(1)

            # If only provider is specified, resolve default model
            if provider and not model:
                provider = provider.lower()
                # Default models for each provider
                default_models = {
                    "anthropic": "claude-3-5-sonnet-20241022",
                    "openai": "gpt-4o",
                    "google": "gemini-2.0-flash-exp",
                    "ollama": "qwen2.5-coder:7b",
                    "lmstudio": "local-model",
                    "vllm": "local-model",
                    "deepseek": "deepseek-chat",
                    "xai": "grok-beta",
                    "zai": "glm-4.7",
                    "cohere": "command-r-plus",
                    "azure": "gpt-4o",
                }

                model = default_models.get(provider)
                if model is None:
                    console.print(
                        f"[bold red]Error:[/] No default model known for provider '{provider}'. "
                        f"Please specify --model."
                    )
                    raise typer.Exit(1)
                console.print(f"[dim]Using default model for {provider}: {model}[/]")
            elif provider:
                provider = provider.lower()

            extra_fields = {}
            if endpoint:
                extra_fields["base_url"] = endpoint
                if provider in {"ollama", "lmstudio", "vllm"}:
                    if provider == "ollama":
                        settings.provider.ollama_base_url = endpoint
                    elif provider == "lmstudio":
                        settings.provider.lmstudio_base_urls = [endpoint]
                    elif provider == "vllm":
                        settings.provider.vllm_base_url = endpoint

            if auth_mode:
                extra_fields["auth_mode"] = auth_mode.lower()
            if coding_plan:
                extra_fields["coding_plan"] = True

            override_profile = ProfileConfig(
                provider=provider,
                model=model,
                temperature=settings.provider.default_temperature,
                max_tokens=settings.provider.default_max_tokens,
                **extra_fields,
            )

            # Replace profile loader to use the synthetic profile
            settings.load_profiles = lambda: {profile: override_profile}  # type: ignore[attr-defined]
            settings.provider.default_provider = provider
            settings.provider.default_model = model

        # Apply auto-skill CLI override to settings
        if auto_skill is not None:
            settings.skill_auto_select_enabled = auto_skill

        if actual_message:
            run_sync(
                run_oneshot(
                    actual_message,
                    settings,
                    profile,
                    stream and not json_output,
                    thinking,
                    formatter=formatter,
                    preindex=preindex,
                    renderer_choice=renderer,
                    mode=mode,
                    tool_budget=tool_budget,
                    max_iterations=max_iterations,
                    vertical=vertical,
                    enable_observability=enable_observability,
                    enable_planning=enable_planning,
                    planning_model=planning_model,
                    legacy_mode=legacy_mode,
                    show_reasoning=show_reasoning,
                )
            )
        elif stdin or input_file:
            formatter.error("No input received from stdin or file")
            raise typer.Exit(1)
        else:
            run_sync(
                run_interactive(
                    settings,
                    profile,
                    stream,
                    thinking,
                    preindex=preindex,
                    renderer_choice=renderer,
                    mode=mode,
                    tool_budget=tool_budget,
                    max_iterations=max_iterations,
                    vertical=vertical,
                    enable_observability=enable_observability,
                    enable_planning=enable_planning,
                    planning_model=planning_model,
                    legacy_mode=legacy_mode,
                    use_tui=use_tui,
                    resume_session_id=session_id,
                    show_reasoning=show_reasoning,
                )
            )
            return


def _run_default_interactive() -> None:
    """Run the default interactive CLI mode with default options."""
    # Use centralized logging config (respects ~/.victor/config.yaml and env vars)
    setup_logging(command="chat")

    settings = load_settings()
    setup_safety_confirmation()
    run_sync(run_interactive(settings, "default", True, False, use_tui=False))


async def run_oneshot(
    message: str,
    settings: Any,
    profile: str,
    stream: bool,
    thinking: bool = False,
    formatter: Optional[Any] = None,
    preindex: bool = False,
    renderer_choice: str = "auto",
    mode: Optional[str] = None,
    tool_budget: Optional[int] = None,
    max_iterations: Optional[int] = None,
    vertical: Optional[str] = None,
    enable_observability: bool = True,
    enable_planning: Optional[bool] = None,
    planning_model: Optional[str] = None,
    legacy_mode: bool = False,
    show_reasoning: bool = False,
) -> None:
    """Run a single message and exit.

    By default, uses FrameworkShim to bridge CLI with framework features
    like observability and vertical configuration. Use legacy_mode=True
    to bypass the framework and use direct orchestrator creation.
    """
    from victor.ui.output_formatter import create_formatter

    if formatter is None:
        formatter = create_formatter()

    settings.automation.one_shot_mode = True
    start_time = time.time()
    session_id = str(uuid.uuid4())
    success = False
    tool_calls_made = 0

    try:
        from victor.framework.task import TaskComplexityService as ComplexityClassifier

        classifier = ComplexityClassifier(use_semantic=False)
        classification = classifier.classify(message)
        _task_type = classification.complexity.value  # Currently unused, logged for debugging
    except Exception:
        pass

    # If planning is enabled, disable thinking mode to avoid context bloat
    # Planning mode generates structured plans which don't need extended reasoning
    if enable_planning and thinking:
        console.print(
            "[yellow]Planning mode enabled: disabling thinking mode to avoid context overflow.[/]"
        )
        thinking = False

    # Auto-enable show_reasoning for thinking models (GLM-5.x, DeepSeek-R1, Qwen3)
    if not show_reasoning:
        try:
            from victor.agent.tool_calling.capabilities import ModelCapabilityLoader

            caps = ModelCapabilityLoader().get_capabilities(
                settings.provider.default_provider,
                settings.provider.default_model,
            )
            if caps and caps.thinking_mode:
                show_reasoning = True
        except Exception:
            pass

    agent = None
    shim: Optional[FrameworkShim] = None
    try:
        # Unified initialization via AgentFactory (replaces legacy/framework split)
        from victor.framework.agent_factory import AgentFactory, InitializationError

        vertical_class = get_vertical(vertical) if vertical else None

        # Show initialization progress for first-time setup (can take 20-30s)
        with console.status("[bold green]Initializing Victor...[/]", spinner="dots") as status:
            status.update("Loading configuration...")
            factory = AgentFactory(
                settings=settings,
                profile=profile,
                vertical=vertical_class,
                thinking=thinking,
                session_id=session_id,
                enable_observability=enable_observability,
                tool_budget=tool_budget,
                max_iterations=max_iterations if "max_iterations" in dir() else None,
            )

            status.update("Creating agent...")
            agent = await factory.create()

            # Set planning model override if provided (for planning coordinator)
            if planning_model:
                agent._planning_model_override = planning_model

            # Note: Observability (shim) is handled by AgentFactory internally
            # The factory creates the agent with framework features already wired

            if tool_budget is not None:
                agent.unified_tracker.set_tool_budget(tool_budget, user_override=True)
            if max_iterations is not None:
                agent.unified_tracker.set_max_iterations(max_iterations, user_override=True)

            if mode:
                from victor.agent.mode_controller import AgentMode, get_mode_controller

                controller = get_mode_controller()
                try:
                    controller.switch_mode(AgentMode(mode))
                except Exception:
                    # Fallback: ignore invalid mode silently (validated earlier)
                    pass

            if preindex:
                await preload_semantic_index(os.getcwd(), settings, console)

            agent.start_embedding_preload()

            # Planning mode requires non-streaming (plan generation → step execution → summary)
            use_streaming = stream and agent.provider.supports_streaming() and not enable_planning

            # Display skill auto-selection feedback before response
            _display_skill_preview(console, agent, message)

            if use_streaming:
                from victor.ui.rendering import (
                    FormatterRenderer,
                    LiveDisplayRenderer,
                    stream_response,
                )

                use_live = renderer_choice in {"rich", "auto"}
                if renderer_choice in {"rich-text", "text"}:
                    use_live = False
                renderer = (
                    LiveDisplayRenderer(console)
                    if use_live
                    else FormatterRenderer(formatter, console)
                )
                await stream_response(
                    agent, message, renderer, suppress_thinking=not show_reasoning
                )
            else:
                # Use streaming pipeline with BufferedRenderer to capture
                # tool calls and reasoning that agent.chat() would swallow
                from victor.ui.rendering import (
                    BufferedRenderer,
                    stream_response,
                )

                buffered = BufferedRenderer(
                    show_reasoning=show_reasoning,
                    plain=formatter._plain if hasattr(formatter, "_plain") else False,
                )
                await stream_response(
                    agent, message, buffered, suppress_thinking=not show_reasoning
                )
                buffered.flush(console)

            success = True
            if hasattr(agent, "get_session_metrics"):
                metrics = agent.get_session_metrics()
                tool_calls_made = metrics.get("tool_calls", 0) if metrics else 0

    except InitializationError as e:
        formatter.error(f"{e.stage}: {e.message}")
        for s in e.suggestions:
            formatter.info(f"  → {s}")
        if e.run_command:
            formatter.info(f"Run: {e.run_command}")
        raise typer.Exit(1)
    except Exception as e:
        formatter.error(str(e))
        raise typer.Exit(1)
    finally:
        # Emit session end event
        duration = time.time() - start_time
        if shim:
            shim.emit_session_end(
                tool_calls=tool_calls_made,
                duration_seconds=duration,
                success=success,
            )
        if agent:
            await graceful_shutdown(agent)


async def run_interactive(
    settings: Any,
    profile: str,
    stream: bool,
    thinking: bool = False,
    preindex: bool = False,
    renderer_choice: str = "auto",
    mode: Optional[str] = None,
    tool_budget: Optional[int] = None,
    max_iterations: Optional[int] = None,
    vertical: Optional[str] = None,
    enable_observability: bool = True,
    enable_planning: Optional[bool] = None,
    planning_model: Optional[str] = None,
    legacy_mode: bool = False,
    use_tui: bool = True,
    resume_session_id: Optional[str] = None,
    show_reasoning: bool = False,
) -> None:
    """Run interactive CLI mode.

    By default, uses FrameworkShim to bridge CLI with framework features
    like observability and vertical configuration. Use legacy_mode=True
    to bypass the framework and use direct orchestrator creation.

    Args:
        use_tui: If True, use the modern TUI interface. If False, use simple CLI.
        show_reasoning: If True, show LLM reasoning/thinking content in output.
    """
    # Auto-enable show_reasoning for thinking models (GLM-5.x, DeepSeek-R1, Qwen3)
    if not show_reasoning:
        try:
            from victor.agent.tool_calling.capabilities import ModelCapabilityLoader

            caps = ModelCapabilityLoader().get_capabilities(
                settings.provider.default_provider,
                settings.provider.default_model,
            )
            if caps and caps.thinking_mode:
                show_reasoning = True
        except Exception:
            pass

    agent = None
    shim: Optional[FrameworkShim] = None
    start_time = time.time()
    session_id = str(uuid.uuid4())
    success = False
    tool_calls_made = 0

    try:
        profiles = settings.load_profiles()
        profile_config = profiles.get(profile)

        if not profile_config:
            console.print(f"[bold red]Error:[/ ] Profile '{profile}' not found")
            raise typer.Exit(1)

        # Unified initialization via AgentFactory
        from victor.framework.agent_factory import AgentFactory, InitializationError

        vertical_class = get_vertical(vertical) if vertical else None
        factory = AgentFactory(
            settings=settings,
            profile=profile,
            vertical=vertical_class,
            thinking=thinking,
            session_id=session_id,
            enable_observability=enable_observability,
            tool_budget=tool_budget,
        )
        try:
            agent = await factory.create()

            # Set planning model override if provided (for planning coordinator)
            if planning_model:
                agent._planning_model_override = planning_model

            # Resume session if requested
            if resume_session_id:
                from victor.agent.sqlite_session_persistence import (
                    get_sqlite_session_persistence,
                )
                from victor.agent.message_history import MessageHistory
                from victor.agent.conversation_state import ConversationStateMachine

                persistence = get_sqlite_session_persistence()
                session_data = persistence.load_session(resume_session_id)

                if not session_data:
                    console.print(f"[bold red]Error:[/ ] Session not found: {resume_session_id}")
                    raise typer.Exit(1)

                # Restore conversation
                metadata = session_data.get("metadata", {})
                conversation_dict = session_data.get("conversation", {})

                agent.conversation = MessageHistory.from_dict(conversation_dict)
                agent.active_session_id = resume_session_id

                # Restore conversation state if available
                conversation_state_dict = session_data.get("conversation_state")
                if conversation_state_dict:
                    try:
                        agent.conversation_state = ConversationStateMachine.from_dict(
                            conversation_state_dict
                        )
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning:[/] Failed to restore conversation state: {e}"
                        )

                console.print(
                    f"[green]✓[/] Resumed session: {metadata.get('title', 'Untitled')} "
                    f"({metadata.get('message_count', 0)} messages)\n"
                )

            # Note: Observability (shim) is handled by AgentFactory internally
            # The factory creates the agent with framework features already wired
        except InitializationError as e:
            console.print(f"[red]Error ({e.stage}):[/] {e.message}")
            for s in e.suggestions:
                console.print(f"  → {s}")
            if e.run_command:
                console.print(f"\nRun: [bold]{e.run_command}[/]")
            raise typer.Exit(1)

        if tool_budget is not None:
            agent.unified_tracker.set_tool_budget(tool_budget, user_override=True)
        if max_iterations is not None:
            agent.unified_tracker.set_max_iterations(max_iterations, user_override=True)

        if mode:
            from victor.agent.mode_controller import AgentMode, get_mode_controller

            controller = get_mode_controller()
            try:
                controller.switch_mode(AgentMode(mode))
            except Exception:
                pass

        if use_tui:
            # Use modern TUI interface
            try:
                from victor.ui.tui import VictorTUI

                tui_app = VictorTUI(
                    agent=agent,
                    provider=profile_config.provider,
                    model=profile_config.model,
                    stream=stream,
                    settings=settings,  # Pass settings for slash commands
                )
                await tui_app.run_async()
            except ImportError:
                # Fall back to CLI if Textual not available
                console.print(
                    "[yellow]Warning:[/] TUI not available (textual not installed). "
                    "Using CLI mode."
                )
                use_tui = False

        if not use_tui:
            from victor.ui.commands import SlashCommandHandler

            cmd_handler = SlashCommandHandler(console, settings, agent)

            rl_suggestion = get_rl_profile_suggestion(profile_config.provider, profiles)
            await _run_cli_repl(
                agent,
                settings,
                cmd_handler,
                profile_config,
                stream,
                rl_suggestion,
                renderer_choice=renderer_choice,
                vertical_name=vertical,
                enable_planning=enable_planning,
                show_reasoning=show_reasoning,
            )

        success = True
        if hasattr(agent, "get_session_metrics"):
            metrics = agent.get_session_metrics()
            tool_calls_made = metrics.get("tool_calls", 0) if metrics else 0

    except Exception as e:
        console.print(f"[bold red]Error:[/ ] {str(e)}")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)
    finally:
        # Emit session end event
        duration = time.time() - start_time
        if shim:
            shim.emit_session_end(
                tool_calls=tool_calls_made,
                duration_seconds=duration,
                success=success,
            )
        if agent:
            await graceful_shutdown(agent)


def _create_cli_prompt_session():
    """Create a prompt_toolkit PromptSession with persistent history.

    Loads previous user messages from the conversation database and persists
    new input to ~/.victor/chat_history for cross-session Up/Down navigation.
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory, InMemoryHistory

    # Use a persistent history file in ~/.victor/
    try:
        from victor.config.settings import get_project_paths

        history_file = get_project_paths().global_victor_dir / "chat_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(history_file))

        # Seed from conversation DB if history file is empty/new
        if not history_file.exists() or history_file.stat().st_size == 0:
            try:
                import sqlite3

                db_path = get_project_paths().conversation_db
                if db_path.exists():
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.execute("""
                            SELECT DISTINCT content
                            FROM messages
                            WHERE role = 'user'
                              AND content IS NOT NULL
                              AND content != ''
                              AND LENGTH(content) < 4000
                              AND content NOT LIKE '<TOOL_OUTPUT%'
                              AND content NOT LIKE '<%'
                              AND content NOT LIKE '{%'
                            ORDER BY timestamp ASC
                            LIMIT 100
                            """)
                        for (msg,) in cursor.fetchall():
                            history.store_string(msg)
            except Exception:
                pass  # DB seeding is best-effort

    except Exception:
        history = InMemoryHistory()

    return PromptSession(history=history)


async def _run_cli_repl(
    agent: AgentOrchestrator,
    settings: Any,
    cmd_handler: Any,
    profile_config: Any,
    stream: bool,
    rl_suggestion: Optional[tuple[str, str, float]] = None,
    renderer_choice: str = "auto",
    vertical_name: Optional[str] = None,
    enable_planning: Optional[bool] = None,
    show_reasoning: bool = False,
) -> None:
    """Run the CLI-based REPL (fallback for unsupported terminals)."""
    vertical_display = f"  [bold]Vertical:[/ ] [magenta]{vertical_name}[/]" if vertical_name else ""
    panel_content = (
        f"[bold blue]Victor[/] - Open-source agentic AI framework\n\n"
        f"[bold]Provider:[/ ] [cyan]{profile_config.provider}[/]  "
        f"[bold]Model:[/ ] [cyan]{profile_config.model}[/]{vertical_display}\n\n"
        f"Type [bold]/help[/] for commands, [bold]/exit[/] or [bold]Ctrl+D[/] to quit.\n"
        f"Use [bold]Up/Down[/] arrows to browse conversation history."
    )

    if rl_suggestion:
        profile_name, provider_name, q_value = rl_suggestion
        panel_content += (
            f"\n\n[dim]RL Tip: [cyan]{provider_name}[/] has higher Q-value ({q_value:.2f}) - "
            f"try [cyan]-p {profile_name}[/][/]"
        )

    console.print(
        Panel(
            panel_content,
            title="Victor CLI",
            border_style="blue",
        )
    )

    # Set up prompt_toolkit with persistent history for Up/Down arrow navigation
    prompt_session = _create_cli_prompt_session()

    while True:
        try:
            # Use prompt_async() — prompt_toolkit's native async input.
            # The sync .prompt() internally calls asyncio.run() which crashes
            # when we're already inside an event loop (run_sync → asyncio.run).
            user_input = await prompt_session.prompt_async("You> ")

            if not user_input.strip():
                continue

            if user_input.strip().lower() in ("exit", "quit", "/exit", "/quit"):
                console.print("[dim]Goodbye![/]")
                break

            if cmd_handler.is_command(user_input):
                await cmd_handler.execute(user_input)
                continue

            if user_input.strip().lower() == "clear":
                agent.reset_conversation()
                console.print("[green]Conversation cleared[/]")
                continue

            console.print("[blue]Assistant:[/]")

            # Planning mode requires non-streaming execution
            use_streaming = stream and not enable_planning

            if use_streaming:
                from victor.agent.response_sanitizer import sanitize_response
                from victor.ui.rendering import (
                    LiveDisplayRenderer,
                    FormatterRenderer,
                    stream_response,
                )

                use_live = renderer_choice in {"rich", "auto"}
                renderer = (
                    LiveDisplayRenderer(console)
                    if use_live
                    else FormatterRenderer(create_formatter(), console)
                )
                content_buffer = await stream_response(
                    agent,
                    user_input,
                    renderer,
                    suppress_thinking=not show_reasoning,
                )
                content_buffer = sanitize_response(content_buffer)
            else:
                response = await agent.chat(user_input, use_planning=enable_planning)
                console.print(Markdown(response.content))

        except KeyboardInterrupt:
            console.print("\n[dim]Use /exit or Ctrl+D to quit[/]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/ ] {e}")
            import traceback

            console.print(traceback.format_exc())


async def run_workflow_mode(
    workflow_path: str,
    validate_only: bool = False,
    render_format: Optional[str] = None,
    render_output: Optional[str] = None,
    profile: str = "default",
    vertical: Optional[str] = None,
    log_level: Optional[str] = None,
) -> None:
    """Run, validate, or render a YAML workflow file.

    Args:
        workflow_path: Path to the YAML workflow file
        validate_only: If True, only validate without executing
        render_format: Output format for rendering (ascii, mermaid, d2, dot, svg, png)
        render_output: Output file for rendered diagram
        profile: Profile to use for agent nodes
        vertical: Optional vertical for context
        log_level: Logging level
    """
    import json
    from pathlib import Path
    from rich.table import Table
    from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

    # Setup logging
    if log_level:
        setup_logging(command="workflow", cli_log_level=log_level)
    else:
        setup_logging(command="workflow")

    workflow_file = Path(workflow_path)
    if not workflow_file.exists():
        console.print(f"[bold red]Error:[/] Workflow file not found: {workflow_path}")
        raise typer.Exit(1)

    if workflow_file.suffix not in {".yaml", ".yml"}:
        console.print(f"[bold red]Error:[/] File must be .yaml or .yml: {workflow_path}")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Workflow:[/] {workflow_file.name}")
    console.print("[dim]" + "─" * 50 + "[/]")

    try:
        # Load and parse workflow(s)
        loaded = load_workflow_from_file(str(workflow_file))

        # Handle dict of workflows or single workflow
        from victor.workflows.definition import WorkflowDefinition

        workflows: dict[str, WorkflowDefinition] = {}
        if isinstance(loaded, dict):
            workflows = loaded
        else:
            workflows = {loaded.name: loaded}

        if not workflows:
            console.print("[bold red]Error:[/] No workflows found in file")
            raise typer.Exit(1)

        compiler = YAMLToStateGraphCompiler()
        all_validated = True

        for wf_name, workflow in workflows.items():
            console.print(f"\n[bold cyan]Workflow:[/] {wf_name}")

            # Display workflow info
            table = Table(show_header=False, box=None)
            table.add_column("Property", style="cyan")
            table.add_column("Value")

            table.add_row("Description", workflow.description or "(none)")
            table.add_row("Nodes", str(len(workflow.nodes)))
            table.add_row("Start Node", workflow.start_node)

            # Count node types
            node_types: dict[str, int] = {}
            for node in workflow.nodes.values():
                node_type = type(node).__name__
                node_types[node_type] = node_types.get(node_type, 0) + 1

            types_str = ", ".join(f"{t}: {c}" for t, c in sorted(node_types.items()))
            table.add_row("Node Types", types_str)

            console.print(table)

            try:
                # Validate using compiler
                _compiled = compiler.compile(workflow)
                console.print("[bold green]✓[/] Validation passed")
            except Exception as e:
                console.print(f"[bold red]✗[/] Validation failed: {e}")
                all_validated = False

        if not all_validated:
            raise typer.Exit(1)

        if validate_only and not render_format:
            console.print("\n[bold green]Validation complete.[/]")
            return

        # Handle rendering
        if render_format:
            # For rendering, use first workflow if multiple exist
            workflow = next(iter(workflows.values()))
            viz = WorkflowVisualizer(workflow)

            # Map format string to enum
            format_map = {
                "ascii": VizFormat.ASCII,
                "mermaid": VizFormat.MERMAID,
                "plantuml": VizFormat.PLANTUML,
                "dot": VizFormat.DOT,
                "d2": VizFormat.D2,
                "svg": VizFormat.SVG,
                "png": VizFormat.PNG,
            }

            fmt = format_map.get(render_format.lower())
            if not fmt:
                console.print(
                    f"[bold red]Error:[/] Unknown format '{render_format}'. "
                    f"Valid: {', '.join(format_map.keys())}"
                )
                raise typer.Exit(1)

            # SVG/PNG require output path
            if fmt in {VizFormat.SVG, VizFormat.PNG} and not render_output:
                # Generate default output path
                render_output = str(workflow_file.with_suffix(f".{render_format.lower()}"))
                console.print(f"[dim]Output: {render_output}[/]")

            console.print(f"\n[bold]Rendering as {render_format.upper()}...[/]")

            try:
                result = viz.render(fmt, render_output)

                if render_output:
                    console.print(f"[bold green]✓[/] Saved to {render_output}")
                else:
                    console.print("")
                    # Use markup=False to avoid interpreting [] as Rich markup
                    console.print(result, markup=False)

            except Exception as e:
                console.print(f"[bold red]✗[/] Rendering failed: {e}")

                # Show available backends
                backends = get_available_backends()
                console.print("\n[dim]Available backends:[/]")
                for name, avail in backends.items():
                    status = "[green]✓[/]" if avail else "[red]✗[/]"
                    console.print(f"  {status} {name}")

                raise typer.Exit(1)

            return

        # For execution, use first workflow if multiple exist
        workflow = next(iter(workflows.values()))

        # Execute workflow
        console.print(f"\n[bold]Executing workflow '{workflow.name}'...[/]\n")

        settings = load_settings()

        # Create orchestrator if agent nodes exist
        orchestrator = None
        from victor.workflows.definition import AgentNode

        has_agent_nodes = any(isinstance(node, AgentNode) for node in workflow.nodes.values())

        if has_agent_nodes:
            shim = FrameworkShim(
                settings,
                profile_name=profile,
                vertical=get_vertical(vertical) if vertical else None,
            )
            orchestrator = await shim.create_orchestrator()

        # Execute with StateGraphExecutor
        executor = StateGraphExecutor(
            orchestrator=orchestrator,
            config=ExecutorConfig(
                enable_checkpointing=False,  # Simple execution mode
                max_iterations=50,
            ),
        )

        result = await executor.execute(workflow, {})

        # Display result
        console.print("[dim]" + "─" * 50 + "[/]")

        if result.success:
            console.print("[bold green]✓[/] Workflow completed successfully")
            console.print(f"  [dim]Duration: {result.duration_seconds:.2f}s[/]")
            console.print(f"  [dim]Nodes executed: {', '.join(result.nodes_executed)}[/]")
            console.print(f"  [dim]Iterations: {result.iterations}[/]")

            if result.state:
                console.print("\n[bold]Final State:[/]")
                # Convert Pydantic model to dict if needed
                state_dict = (
                    result.state.to_dict() if hasattr(result.state, "to_dict") else result.state
                )
                # Filter internal keys
                display_state = {
                    k: v
                    for k, v in state_dict.items()
                    if not k.startswith("_") and k != "node_results"
                }
                if display_state:
                    console.print(json.dumps(display_state, indent=2, default=str))
        else:
            console.print("[bold red]✗[/] Workflow failed")
            if result.error:
                console.print(f"  [red]{result.error}[/]")

        # Cleanup orchestrator
        if orchestrator:
            await graceful_shutdown(orchestrator)

    except YAMLWorkflowError as e:
        console.print("[bold red]✗[/] Workflow validation failed")
        console.print(f"  [red]{e}[/]")
        console.print("\n[yellow]💡 Run 'victor doctor' for diagnostics[/]")
        raise typer.Exit(1)

    except Exception as e:
        # Use contextual error formatting for better UX
        error_message = format_exception_for_user(e)
        console.print(f"[bold red]Error:[/]\n{error_message}")
        console.print("\n[yellow]💡 Run 'victor doctor' for diagnostics[/]")
        import traceback

        # Still show traceback in debug mode
        if os.getenv("VICTOR_DEBUG"):
            console.print(traceback.format_exc())
        raise typer.Exit(1)
