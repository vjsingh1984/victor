import typer
import asyncio
import os
import sys
from typing import Optional, Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings
from victor.ui.output_formatter import InputReader, create_formatter
from victor.ui.commands.utils import (
    preload_semantic_index,
    check_codebase_index,
    get_rl_profile_suggestion,
    setup_safety_confirmation,
    configure_logging,
    graceful_shutdown
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
):
    """Start interactive chat or send a one-shot message."""
    if ctx.invoked_subcommand is None:
        renderer = renderer.lower()
        if renderer not in {"auto", "rich", "rich-text", "text"}:
            console.print("[bold red]Error:[/] Invalid renderer. Choose from auto, rich, rich-text, text.")
            raise typer.Exit(1)

        if mode:
            mode = mode.lower()
            if mode not in {"build", "plan", "explore"}:
                console.print("[bold red]Error:[/] Invalid mode. Choose from build, plan, explore.")
                raise typer.Exit(1)

        automation_mode = json_output or plain or code_only

        if log_level is None:
            log_level = os.getenv("VICTOR_LOG_LEVEL", "ERROR" if automation_mode else "WARNING")

        log_level = log_level.upper()
        valid_levels = ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"]

        if log_level not in valid_levels:
            console.print(
                f"[bold red]Error:[/ ] Invalid log level '{log_level}'. Valid options: {', '.join(valid_levels)}"
            )
            raise typer.Exit(1)

        if log_level == "WARN":
            log_level = "WARNING"

        configure_logging(log_level, stream=sys.stderr)
        
        from victor.agent.debug_logger import configure_logging_levels
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
        setup_safety_confirmation()

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
                )
            )

def _run_default_interactive() -> None:
    """Run the default interactive CLI mode with default options."""
    log_level = os.getenv("VICTOR_LOG_LEVEL", "WARNING").upper()
    if log_level == "WARN":
        log_level = "WARNING"
    configure_logging(log_level)

    settings = load_settings()
    setup_safety_confirmation()
    asyncio.run(run_interactive(settings, "default", True, False))

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
) -> None:
    """Run a single message and exit."""
    import time
    import uuid
    from victor.ui.output_formatter import create_formatter
    
    if formatter is None:
        formatter = create_formatter()

    settings.one_shot_mode = True
    start_time = time.time()
    session_id = str(uuid.uuid4())[:8]
    success = False
    token_count = 0
    tool_calls_made = 0
    task_type: Optional[str] = None

    try:
        from victor.agent.complexity_classifier import ComplexityClassifier
        classifier = ComplexityClassifier(use_semantic=False)
        classification = classifier.classify(message)
        task_type = classification.complexity.value
    except Exception as e:
        pass

    agent = None
    try:
        if tool_budget is not None:
            settings.tool_call_budget = tool_budget

        agent = await AgentOrchestrator.from_settings(settings, profile, thinking=thinking)
        
        if tool_budget is not None:
            agent.unified_tracker.set_tool_budget(tool_budget)
        if max_iterations is not None:
            agent.unified_tracker.max_total_iterations = max_iterations

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
            from victor.ui.stream_renderer import FormatterRenderer, LiveDisplayRenderer, stream_response
            use_live = renderer_choice in {"rich", "auto"}
            if renderer_choice in {"rich-text", "text"}:
                use_live = False
            renderer = LiveDisplayRenderer(console) if use_live else FormatterRenderer(formatter, console)
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
) -> None:
    """Run interactive CLI mode."""
    agent = None
    try:
        profiles = settings.load_profiles()
        profile_config = profiles.get(profile)

        if not profile_config:
            console.print(f"[bold red]Error:[/ ] Profile '{profile}' not found")
            raise typer.Exit(1)

        if tool_budget is not None:
            settings.tool_call_budget = tool_budget

        agent = await AgentOrchestrator.from_settings(settings, profile, thinking=thinking)
        
        if tool_budget is not None:
            agent.unified_tracker.set_tool_budget(tool_budget)
        if max_iterations is not None:
            agent.unified_tracker.max_total_iterations = max_iterations

        if mode:
            from victor.agent.mode_controller import AgentMode, get_mode_controller
            controller = get_mode_controller()
            try:
                controller.switch_mode(AgentMode(mode))
            except Exception:
                pass
        
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
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/ ] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)
    finally:
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
) -> None:
    """Run the CLI-based REPL (fallback for unsupported terminals)."""
    panel_content = (
        f"[bold blue]Victor[/] - Enterprise-Ready AI Coding Assistant\n\n"
        f"[bold]Provider:[/ ] [cyan]{profile_config.provider}[/]  "
        f"[bold]Model:[/ ] [cyan]{profile_config.model}[/]\n\n"
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
                from victor.ui.stream_renderer import LiveDisplayRenderer, FormatterRenderer, stream_response

                use_live = renderer_choice in {"rich", "auto"}
                renderer = LiveDisplayRenderer(console) if use_live else FormatterRenderer(create_formatter(), console)
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
