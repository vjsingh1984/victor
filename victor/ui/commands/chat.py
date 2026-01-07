import typer
import asyncio
import os
import sys
import time
import uuid
from typing import Optional, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings, ProfileConfig
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

chat_app = typer.Typer(name="chat", help="Start interactive chat or send a one-shot message.")
console = Console()


@chat_app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
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
    renderer: str = typer.Option(
        "auto",
        "--renderer",
        help="Renderer to use for streaming output: auto, rich, rich-text, or text.",
        case_sensitive=False,
    ),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        help="Initial agent mode: build, plan, or explore.",
        case_sensitive=False,
    ),
    tool_budget: Optional[int] = typer.Option(
        None,
        "--tool-budget",
        help="Override tool call budget for this session.",
    ),
    max_iterations: Optional[int] = typer.Option(
        None,
        "--max-iterations",
        help="Override maximum total iterations for this session.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Override provider (e.g., ollama, lmstudio, vllm, openai).",
        case_sensitive=False,
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override model identifier.",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help="Override base URL for local providers (ollama, lmstudio, vllm).",
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
    vertical: Optional[str] = typer.Option(
        None,
        "--vertical",
        "-V",
        help=f"Vertical template to use ({', '.join(list_verticals()) or 'coding, research, devops'}). "
        "By default, no vertical is applied (uses standard CodingAssistant behavior with framework features).",
    ),
    workflow: Optional[str] = typer.Option(
        None,
        "--workflow",
        "-w",
        help="Path to YAML workflow file to execute. Runs workflow instead of chat mode.",
    ),
    validate_workflow: bool = typer.Option(
        False,
        "--validate",
        help="Validate YAML workflow file without executing. Use with --workflow.",
    ),
    render_format: Optional[str] = typer.Option(
        None,
        "--render",
        "-r",
        help="Render workflow DAG (ascii, mermaid, d2, dot, plantuml, svg, png). Use with --workflow.",
    ),
    render_output: Optional[str] = typer.Option(
        None,
        "--render-output",
        "-o",
        help="Output file for rendered diagram. Required for svg/png formats.",
    ),
    legacy_mode: bool = typer.Option(
        False,
        "--legacy",
        help="Use legacy orchestrator creation path (bypasses FrameworkShim). "
        "For backward compatibility and troubleshooting.",
    ),
    enable_observability: bool = typer.Option(
        True,
        "--observability/--no-observability",
        help="Enable observability integration for event tracking.",
    ),
    log_events: bool = typer.Option(
        False,
        "--log-events",
        help="Enable JSONL event logging to ~/.victor/logs/victor.log for dashboard visualization.",
    ),
    tui: bool = typer.Option(
        False,
        "--tui/--no-tui",
        help="Use modern TUI interface. Use --no-tui for simple CLI mode (default).",
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

            asyncio.run(
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

        # Use centralized logging config
        setup_logging(command="chat", cli_log_level=log_level, stream=sys.stderr)

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

        actual_message = InputReader.read_message(
            argument=message,
            from_stdin=stdin,
            input_file=input_file,
        )

        settings = load_settings()

        # Apply CLI flags to settings
        if log_events:
            settings.enable_observability_logging = True

        setup_safety_confirmation()

        # Apply provider/model/endpoint overrides by creating a synthetic profile
        if provider or model or endpoint:
            if not provider or not model:
                console.print(
                    "[bold red]Error:[/] --provider and --model must be provided together when overriding profiles."
                )
                raise typer.Exit(1)
            provider = provider.lower()

            extra_fields = {}
            if endpoint:
                extra_fields["base_url"] = endpoint
                if provider in {"ollama", "lmstudio", "vllm"}:
                    if provider == "ollama":
                        settings.ollama_base_url = endpoint
                    elif provider == "lmstudio":
                        settings.lmstudio_base_urls = [endpoint]
                    elif provider == "vllm":
                        settings.vllm_base_url = endpoint
                else:
                    console.print(
                        "[bold yellow]Warning:[/] --endpoint is ignored for this provider."
                    )

            override_profile = ProfileConfig(
                provider=provider,
                model=model,
                temperature=settings.default_temperature,
                max_tokens=settings.default_max_tokens,
                **extra_fields,
            )

            # Replace profile loader to use the synthetic profile
            settings.load_profiles = lambda: {profile: override_profile}  # type: ignore[attr-defined]
            settings.default_provider = provider
            settings.default_model = model

        if actual_message:
            asyncio.run(
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
                    legacy_mode=legacy_mode,
                )
            )
        elif stdin or input_file:
            formatter.error("No input received from stdin or file")
            raise typer.Exit(1)
        else:
            asyncio.run(
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
                    legacy_mode=legacy_mode,
                    # Smart TUI detection: disable if non-interactive terminal or automation mode
                    use_tui=tui
                    and not automation_mode
                    and sys.stdin.isatty()
                    and sys.stdout.isatty(),
                )
            )


def _run_default_interactive() -> None:
    """Run the default interactive CLI mode with default options."""
    # Use centralized logging config (respects ~/.victor/config.yaml and env vars)
    setup_logging(command="chat")

    settings = load_settings()
    setup_safety_confirmation()
    asyncio.run(run_interactive(settings, "default", True, False, use_tui=False))


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
    legacy_mode: bool = False,
) -> None:
    """Run a single message and exit.

    By default, uses FrameworkShim to bridge CLI with framework features
    like observability and vertical configuration. Use legacy_mode=True
    to bypass the framework and use direct orchestrator creation.
    """
    from victor.ui.output_formatter import create_formatter

    if formatter is None:
        formatter = create_formatter()

    settings.one_shot_mode = True
    start_time = time.time()
    session_id = str(uuid.uuid4())
    success = False
    token_count = 0
    tool_calls_made = 0
    task_type: Optional[str] = None

    try:
        from victor.agent.complexity_classifier import ComplexityClassifier

        classifier = ComplexityClassifier(use_semantic=False)
        classification = classifier.classify(message)
        task_type = classification.complexity.value
    except Exception:
        pass

    agent = None
    shim: Optional[FrameworkShim] = None
    try:
        if tool_budget is not None:
            settings.tool_call_budget = tool_budget

        if legacy_mode:
            # Legacy path: direct orchestrator creation (no framework features)
            agent = await AgentOrchestrator.from_settings(settings, profile, thinking=thinking)
        else:
            # Framework path: use FrameworkShim with observability and verticals
            vertical_class = get_vertical(vertical) if vertical else None

            shim = FrameworkShim(
                settings,
                profile_name=profile,
                thinking=thinking,
                vertical=vertical_class,
                enable_observability=enable_observability,
                session_id=session_id,
            )
            agent = await shim.create_orchestrator()

            # Emit session start event if observability is enabled
            shim.emit_session_start(
                {
                    "mode": "oneshot",
                    "task_type": task_type,
                    "vertical": vertical,
                    "thinking": thinking,
                }
            )

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

        content_buffer = ""
        usage_stats = None
        model_name = None

        if stream and agent.provider.supports_streaming():
            from victor.ui.rendering import FormatterRenderer, LiveDisplayRenderer, stream_response

            use_live = renderer_choice in {"rich", "auto"}
            if renderer_choice in {"rich-text", "text"}:
                use_live = False
            renderer = (
                LiveDisplayRenderer(console) if use_live else FormatterRenderer(formatter, console)
            )
            content_buffer = await stream_response(agent, message, renderer)
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

        success = True
        if usage_stats:
            token_count = getattr(usage_stats, "total_tokens", 0) or 0
        if hasattr(agent, "get_session_metrics"):
            metrics = agent.get_session_metrics()
            tool_calls_made = metrics.get("tool_calls", 0) if metrics else 0
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
    legacy_mode: bool = False,
    use_tui: bool = True,
) -> None:
    """Run interactive CLI mode.

    By default, uses FrameworkShim to bridge CLI with framework features
    like observability and vertical configuration. Use legacy_mode=True
    to bypass the framework and use direct orchestrator creation.

    Args:
        use_tui: If True, use the modern TUI interface. If False, use simple CLI.
    """
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

        if tool_budget is not None:
            settings.tool_call_budget = tool_budget

        if legacy_mode:
            # Legacy path: direct orchestrator creation (no framework features)
            agent = await AgentOrchestrator.from_settings(settings, profile, thinking=thinking)
        else:
            # Framework path: use FrameworkShim with observability and verticals
            vertical_class = get_vertical(vertical) if vertical else None

            shim = FrameworkShim(
                settings,
                profile_name=profile,
                thinking=thinking,
                vertical=vertical_class,
                enable_observability=enable_observability,
                session_id=session_id,
            )
            agent = await shim.create_orchestrator()

            # Emit session start event if observability is enabled
            shim.emit_session_start(
                {
                    "mode": "interactive",
                    "vertical": vertical,
                    "thinking": thinking,
                    "provider": profile_config.provider,
                    "model": profile_config.model,
                }
            )

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


async def _run_cli_repl(
    agent: AgentOrchestrator,
    settings: Any,
    cmd_handler: Any,
    profile_config: Any,
    stream: bool,
    rl_suggestion: Optional[tuple[str, str, float]] = None,
    renderer_choice: str = "auto",
    vertical_name: Optional[str] = None,
) -> None:
    """Run the CLI-based REPL (fallback for unsupported terminals)."""
    vertical_display = f"  [bold]Vertical:[/ ] [magenta]{vertical_name}[/]" if vertical_name else ""
    panel_content = (
        f"[bold blue]Victor[/] - Open-source AI coding assistant\n\n"
        f"[bold]Provider:[/ ] [cyan]{profile_config.provider}[/]  "
        f"[bold]Model:[/ ] [cyan]{profile_config.model}[/]{vertical_display}\n\n"
        f"Type [bold]/help[/] for commands, [bold]/exit[/] or [bold]Ctrl+D[/] to quit."
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

    while True:
        try:
            user_input = Prompt.ask("[green]You[/]")

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

            if stream:
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
                content_buffer = await stream_response(agent, user_input, renderer)
                content_buffer = sanitize_response(content_buffer)
            else:
                response = await agent.chat(user_input)
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
                # Filter internal keys
                display_state = {
                    k: v
                    for k, v in result.state.items()
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
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(1)
