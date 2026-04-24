import typer
import os
import sys
import time
import uuid
from types import SimpleNamespace
from typing import Optional, Any

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import get_project_paths, load_settings, ProfileConfig
from victor.core.async_utils import run_sync
from victor.framework.shim import FrameworkShim
from victor.core.verticals import get_vertical, list_verticals
from victor.core.errors import (
    ConfigurationError,
    ProviderError,
    ProviderConnectionError,
    ProviderAuthError,
    ProviderNotFoundError,
    VictorError,
)
from victor.ui.output_formatter import InputReader, create_formatter
from victor.ui.rendering.utils import render_status_message
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



chat_app = typer.Typer(
    name="chat",
    help="""Start interactive chat or send a one-shot message.

    **Basic Usage:**
        victor chat                    # Start interactive chat
        victor chat "Hello, Victor!"    # Send one-shot message

    **Advanced Options:**
        Use --help-full to see all 37 options organized by category.
        Workflow options: Use 'victor workflow' command instead.
        Session options: Use 'victor sessions' command instead.
    """,
)
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


def _print_tool_output_mode_banner(con: Console, tool_settings: Any) -> None:
    """Render a compact banner describing tool-output preview behavior."""
    con.print(_describe_tool_output_mode(tool_settings))


def _describe_tool_output_mode(tool_settings: Any) -> str:
    """Return the user-facing tool-output mode summary."""
    summary, preview_state = _tool_output_mode_parts(tool_settings)
    icon = "[green]✓[/]" if summary == "safe read-heavy pruning" else "[yellow]![/]"
    return f"{icon} [bold]Tool output[/] [dim]{summary} • {preview_state}[/]"


def _summarize_tool_output_mode(tool_settings: Any) -> str:
    """Return a plain-text startup message describing tool-output behavior."""
    summary, preview_state = _tool_output_mode_parts(tool_settings)
    return f"Tool output: {summary}, {preview_state}"


def _tool_output_mode_parts(tool_settings: Any) -> tuple[str, str]:
    """Return summary parts for tool-output startup messaging."""
    preview_enabled = bool(getattr(tool_settings, "tool_output_preview_enabled", True))
    safe_only = bool(getattr(tool_settings, "tool_output_pruning_safe_only", True))

    if safe_only:
        summary = "safe read-heavy pruning"
    else:
        summary = "broader pruning"

    preview_state = "preview on" if preview_enabled else "preview off"
    return summary, preview_state


def _should_render_cli_chrome(formatter: Any) -> bool:
    """Return True when stdout can safely include Rich-only CLI chrome."""
    config = getattr(formatter, "config", None)
    if config is None:
        return True

    quiet = getattr(config, "quiet", False)
    if isinstance(quiet, bool) and quiet:
        return False

    mode = getattr(config, "mode", None)
    if mode is None:
        return True

    mode_value = getattr(mode, "value", mode)
    return str(mode_value).lower() == "rich"


def _should_render_interactive_tool_banner(use_tui: bool) -> bool:
    """Return True when interactive startup should print CLI tool-output chrome."""
    return not use_tui


def _print_interactive_startup_messages(con: Console, messages: list[str]) -> None:
    """Print queued startup notices for interactive CLI surfaces."""
    for message in messages:
        render_status_message(con, message)


def _summarize_smart_routing(settings: Any, enable_smart_routing: bool, routing_profile: str) -> list[str]:
    """Return startup messages describing smart-routing state."""
    if not enable_smart_routing:
        return []
    if getattr(settings, "smart_routing_enabled", False):
        return [f"Smart routing enabled (profile={routing_profile})"]
    return ["Smart routing is currently disabled via feature flag."]


def _summarize_compaction_overrides(
    *,
    compaction_threshold: Optional[float] = None,
    adaptive_threshold: Optional[bool] = None,
    compaction_min_threshold: Optional[float] = None,
    compaction_max_threshold: Optional[float] = None,
) -> list[str]:
    """Return startup messages describing CLI compaction overrides."""
    messages: list[str] = []
    if compaction_threshold is not None:
        messages.append(f"Compaction threshold set to {compaction_threshold:.0%}")
    if adaptive_threshold is True:
        min_thresh = compaction_min_threshold or 0.35
        max_thresh = compaction_max_threshold or 0.70
        messages.append(f"Adaptive threshold enabled ({min_thresh:.0%}-{max_thresh:.0%})")
    elif adaptive_threshold is False:
        messages.append("Adaptive threshold disabled")
    return messages


class _NoopStatus:
    """No-op replacement for Rich status in automation output modes."""

    def __enter__(self) -> "_NoopStatus":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def stop(self) -> None:
        pass


def _build_cli_panel(
    profile_config: Any,
    vertical_name: Optional[str] = None,
    rl_suggestion: Optional[tuple[str, str, float]] = None,
    settings: Any = None,
) -> Panel:
    """Build the interactive CLI header panel."""
    brand = Text()
    brand.append("Victor", style="bold white")
    brand.append("  ", style="dim")
    brand.append("Open-source agentic AI framework", style="dim")

    meta = Text()
    meta.append("Provider ", style="dim")
    # Handle None profile_config (default settings)
    provider_name = profile_config.provider if profile_config else settings.provider.default_provider
    model_name = profile_config.model if profile_config else settings.provider.default_model
    meta.append(str(provider_name), style="bold cyan")
    meta.append("  •  ", style="dim")
    meta.append("Model ", style="dim")
    meta.append(str(model_name), style="bold green")
    if vertical_name:
        meta.append("  •  ", style="dim")
        meta.append("Vertical ", style="dim")
        meta.append(vertical_name, style="bold magenta")

    controls = Text()
    controls.append("/help", style="bold")
    controls.append(" commands  •  ", style="dim")
    controls.append("/exit", style="bold")
    controls.append(" or Ctrl+D  •  history with Up/Down", style="dim")

    lines: list[Any] = [brand, meta, controls]

    if rl_suggestion:
        profile_name, provider_name, q_value = rl_suggestion
        hint = Text()
        hint.append("Routing hint ", style="dim")
        hint.append(provider_name, style="bold cyan")
        hint.append(f"  Q={q_value:.2f}", style="yellow")
        hint.append("  •  victor chat -p ", style="dim")
        hint.append(profile_name, style="bold cyan")
        lines.append(hint)

    return Panel(
        Group(*lines),
        title="Victor CLI",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _configure_agent_compaction(
    agent: Any,
    *,
    compaction_threshold: Optional[float] = None,
    adaptive_threshold: Optional[bool] = None,
    compaction_min_threshold: Optional[float] = None,
    compaction_max_threshold: Optional[float] = None,
    con: Console = console,
    show_status: bool = True,
) -> None:
    """Apply CLI compaction overrides to an agent/orchestrator if supported."""
    if not any(
        (
            compaction_threshold is not None,
            adaptive_threshold is not None,
            compaction_min_threshold is not None,
            compaction_max_threshold is not None,
        )
    ):
        return

    orchestrator_getter = getattr(agent, "get_orchestrator", None)
    orchestrator = orchestrator_getter() if callable(orchestrator_getter) else agent
    compactor = getattr(orchestrator, "context_compactor", None)
    if compactor is None:
        return

    if compaction_threshold is not None:
        if not (0.1 <= compaction_threshold <= 0.95):
            if show_status:
                con.print(f"\n[red]✗[/] Invalid compaction threshold: {compaction_threshold}")
                con.print("[yellow]Threshold must be between 0.1 (10%) and 0.95 (95%)[/]")
            raise typer.Exit(code=1)
        compactor.config.proactive_threshold = compaction_threshold
        if show_status:
            con.print(f"[dim]Compaction threshold set to {compaction_threshold:.0%}[/]")

    if adaptive_threshold is not None:
        if adaptive_threshold:
            from victor.agent.adaptive_compaction import AdaptiveCompactionThreshold

            min_thresh = compaction_min_threshold or 0.35
            max_thresh = compaction_max_threshold or 0.70
            if min_thresh >= max_thresh:
                if show_status:
                    con.print(
                        f"\n[red]✗[/] Min threshold ({min_thresh}) must be less than max "
                        f"({max_thresh})"
                    )
                raise typer.Exit(code=1)

            adaptive = AdaptiveCompactionThreshold(
                min_threshold=min_thresh,
                max_threshold=max_thresh,
            )
            compactor.set_adaptive_threshold(adaptive)
            if show_status:
                con.print(
                    f"[dim]Adaptive threshold enabled ({min_thresh:.0%}-{max_thresh:.0%})[/]"
                )
        else:
            compactor.disable_adaptive_threshold()
            if show_status:
                con.print("[dim]Adaptive threshold disabled[/]")


@chat_app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    help_full: bool = typer.Option(
        False,
        "--help-full",
        help="Show full help with all 37 options (advanced usage).",
        is_eager=True,
    ),
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
    # Compaction options
    compaction_threshold: Optional[float] = typer.Option(
        None,
        "--compaction-threshold",
        help="Compaction threshold (0.1-0.95). Default: 0.50. Lower = earlier compaction.",
        rich_help_panel="Compaction",
    ),
    adaptive_threshold: Optional[bool] = typer.Option(
        None,
        "--adaptive-threshold/--no-adaptive-threshold",
        help="Enable adaptive threshold based on conversation patterns (35-70% range).",
        rich_help_panel="Compaction",
    ),
    compaction_min_threshold: Optional[float] = typer.Option(
        None,
        "--compaction-min-threshold",
        help="Minimum adaptive threshold (0.1-0.8). Default: 0.35. Used with --adaptive-threshold.",
        rich_help_panel="Compaction",
    ),
    compaction_max_threshold: Optional[float] = typer.Option(
        None,
        "--compaction-max-threshold",
        help="Maximum adaptive threshold (0.2-0.95). Default: 0.70. Used with --adaptive-threshold.",
        rich_help_panel="Compaction",
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
    # Smart Routing options (Phase 11 - Intelligent Provider Selection)
    enable_smart_routing: bool = typer.Option(
        False,
        "--enable-smart-routing",
        help="Enable automatic provider routing based on health, resources, cost, latency, and performance.",
        rich_help_panel="Smart Routing",
    ),
    routing_profile: str = typer.Option(
        "balanced",
        "--routing-profile",
        help="Routing profile to use (balanced, cost-optimized, performance, local-first).",
        rich_help_panel="Smart Routing",
    ),
    fallback_chain: Optional[str] = typer.Option(
        None,
        "--fallback-chain",
        help="Custom fallback chain (comma-separated provider names, e.g., 'ollama,anthropic,openai').",
        rich_help_panel="Smart Routing",
    ),
    # Tool Output Preview options (safe-default pruning)
    tool_preview: bool = typer.Option(
        True,
        "--tool-preview/--no-tool-preview",
        help="Show tool output preview (default: yes). Disable to show only status line.",
        rich_help_panel="Tool Output",
    ),
    enable_pruning: bool = typer.Option(
        False,
        "--enable-pruning",
        help="Broaden tool output pruning beyond the safe default read-heavy scope.",
        rich_help_panel="Tool Output",
    ),
):
    """Start interactive chat or send a one-shot message.

    \f
    **Quick Start:**
        victor chat                    # Interactive with defaults
        victor chat "Your message"     # One-shot message

    **For advanced options and examples:**
        victor chat --help-full
    """
    # Handle --help-full flag for comprehensive help
    if help_full:
        from rich.markdown import Markdown

        full_help = """
# Victor Chat - Full Help

**Victor** has 37 options organized into 11 categories for progressive disclosure.

## Quick Reference
- Basic chat: `victor chat`
- One-shot: `victor chat "your message"`
- Switch provider: `victor chat -p anthropic`
- Use profile: `victor chat --profile coding`

## Core Options (Beginner-Friendly)
These are the most commonly used options:

| Option | Description | Example |
|--------|-------------|---------|
| `--profile`, `-p` | Profile from profiles.yaml | `-p coding` |
| `--provider` | Override provider | `--provider ollama` |
| `--model` | Override model | `--model llama3.2` |
| `--stream` | Enable streaming (default: ON) | `--no-stream` |
| `--thinking` | Enable extended reasoning | `--thinking` |

## Category Reference

### Output Format (4 options)
`--json`, `--plain`, `--code-only`, `--quiet`, `-q`

### Input (2 options)
`--stdin`, `--input-file`, `-f`

### Logging (2 options)
`--log-level`, `-l`, `--debug-modules`

### Advanced Agent Behavior (8 options)
`--mode`, `--tool-budget`, `--max-iterations`, `--preindex`,
`--vertical`, `-V`, `--auto-skill`, `--planning`, `--planning-model`

### Workflow (4 options)
Use `victor workflow` command instead for full workflow features.
`--workflow`, `-w`, `--validate`, `--render`, `-r`, `--render-output`, `-o`

### Session (2 options)
Use `victor sessions` command instead for session management.
`--sessions`, `--sessionid`

### Expert Options (9 options)
Advanced debugging and compatibility options:
- **Logging**: `--observability`, `--log-events`
- **Output**: `--show-reasoning`, `--renderer`
- **Auth**: `--endpoint`, `--auth-mode`, `--coding-plan`
- **Interface**: `--tui`, `--legacy`

## Examples

**Interactive Chat:**
```bash
victor chat                              # Interactive with defaults
victor chat -p anthropic --thinking      # Use Claude with reasoning
victor chat -p ollama --model llama3.2  # Use local model
```

**One-Shot Messages:**
```bash
victor chat "Explain recursion"         # Basic message
victor chat "Analyze this file" -f code.py --preindex
victor chat "Plan a REST API" --mode plan
```

**Automation:**
```bash
victor chat "Generate tests" --json --quiet > tests.json
victor chat "Refactor" --code-only > refactored.py
```

**Workflows:**
```bash
victor workflow validate ./workflow.yaml
victor workflow render ./workflow.yaml --format svg -o diagram.svg
```

**Sessions:**
```bash
victor chat --sessions                    # List recent sessions
victor chat --sessionid abc123            # Resume session
```

## See Also
- `victor workflow --help` - Workflow commands
- `victor sessions --help` - Session management
- `victor profile --help` - Profile management
- `victor init --wizard` - First-time setup
"""
        console.print(Markdown(full_help))
        raise typer.Exit(0)

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

        # Periodic cleanup: check if history file needs rotation (once per 7 days)
        try:
            history_file = get_project_paths().project_victor_dir / "chat_history"
            max_entries = settings.ui.cli_history_max_entries  # Use configured max entries
            _maybe_rotate_history_file(history_file, max_entries=max_entries)
        except Exception:
            pass  # Periodic cleanup is best-effort

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
                    compaction_threshold=compaction_threshold,
                    adaptive_threshold=adaptive_threshold,
                    compaction_min_threshold=compaction_min_threshold,
                    compaction_max_threshold=compaction_max_threshold,
                    # Smart routing
                    enable_smart_routing=enable_smart_routing,
                    routing_profile=routing_profile,
                    fallback_chain=fallback_chain,
                    # Tool output preview
                    tool_preview=tool_preview,
                    enable_pruning=enable_pruning,
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
                    compaction_threshold=compaction_threshold,
                    adaptive_threshold=adaptive_threshold,
                    compaction_min_threshold=compaction_min_threshold,
                    compaction_max_threshold=compaction_max_threshold,
                    # Smart routing
                    enable_smart_routing=enable_smart_routing,
                    routing_profile=routing_profile,
                    fallback_chain=fallback_chain,
                    # Tool output preview
                    tool_preview=tool_preview,
                    enable_pruning=enable_pruning,
                )
            )
            return


def _configure_smart_routing(
    settings: Any,
    console: Console,
    enable_smart_routing: bool,
    routing_profile: str,
    fallback_chain: Optional[str],
    show_status: bool = True,
) -> None:
    """Apply smart-routing settings when the feature flag permits."""
    if not enable_smart_routing:
        return
    from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

    feature_manager = get_feature_flag_manager()
    if feature_manager.is_enabled(FeatureFlag.USE_SMART_ROUTING):
        settings.smart_routing_enabled = True
        settings.smart_routing_profile = routing_profile
        if fallback_chain:
            settings.smart_routing_fallback_chain = [
                p.strip() for p in fallback_chain.split(",")
            ]
        if show_status:
            console.print(f"[green]✓[/] Smart routing enabled (profile={routing_profile})")
    else:
        if show_status:
            console.print("[yellow]Smart routing is currently disabled via feature flag.[/]")
            console.print(
                "Enable with: [cyan]export VICTOR_USE_SMART_ROUTING=true[/] or "
                "[cyan]--enable-smart-routing[/] flag in beta period"
            )


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
    compaction_threshold: Optional[float] = None,
    adaptive_threshold: Optional[bool] = None,
    compaction_min_threshold: Optional[float] = None,
    compaction_max_threshold: Optional[float] = None,
    # Smart routing parameters
    enable_smart_routing: bool = False,
    routing_profile: str = "balanced",
    fallback_chain: Optional[str] = None,
    # Tool output preview parameters
    tool_preview: bool = True,
    enable_pruning: bool = False,
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
    show_cli_chrome = _should_render_cli_chrome(formatter)

    _configure_smart_routing(
        settings,
        console,
        enable_smart_routing,
        routing_profile,
        fallback_chain,
        show_status=show_cli_chrome,
    )

    # Configure tool output preview (safe-default pruning)
    from victor.config.tool_settings import ToolSettings

    # Apply tool output preview settings from flags
    if not tool_preview:
        settings.tool_settings.tool_output_preview_enabled = False
    if enable_pruning:
        settings.tool_settings.tool_output_pruning_enabled = True
        settings.tool_settings.tool_output_pruning_safe_only = False

    tool_settings = settings.tool_settings if hasattr(settings, "tool_settings") else ToolSettings()
    if show_cli_chrome:
        _print_tool_output_mode_banner(console, tool_settings)

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
        if show_cli_chrome:
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
    provider_for_suggestions = getattr(settings.provider, "default_provider", "unknown")
    try:
        # Unified initialization via AgentFactory (replaces legacy/framework split)
        from victor.framework.agent_factory import AgentFactory, InitializationError

        # Pass original vertical string to AgentFactory (preserves name for error reporting)
        # AgentFactory will resolve it to a class internally
        vertical_param = vertical if vertical else None

        # Show initialization progress for first-time setup (can take 20-30s)
        status_context = (
            console.status("[bold green]Initializing Victor...[/]", spinner="dots")
            if show_cli_chrome
            else _NoopStatus()
        )
        with status_context as status:
            # Step 1: Load and validate configuration
            status.update("Loading configuration...")

            # Configuration validation (P1-6: Display configuration validation status)
            try:
                from victor.config.validation import validate_configuration, format_validation_result

                status.update("Validating configuration...")
                validation_result = validate_configuration(settings)

                if not validation_result.is_valid():
                    # Configuration has errors - display and exit
                    status.stop()  # Stop the spinner to show errors clearly
                    formatted_validation = format_validation_result(validation_result)
                    if show_cli_chrome:
                        console.print("[bold red]Configuration Validation Failed:[/]")
                        console.print(formatted_validation)
                        console.print("")
                        console.print(
                            "[yellow]Run 'victor config validate' for detailed diagnostics[/]"
                        )
                    else:
                        formatter.error("Configuration validation failed", formatted_validation)
                    raise typer.Exit(1)
                elif validation_result.has_warnings():
                    # Configuration has warnings but is valid - show summary
                    warning_count = len(validation_result.warnings)
                    if show_cli_chrome and warning_count > 0:
                        console.print(f"[dim]✓ Configuration valid with {warning_count} warning(s)[/]")
                else:
                    # Configuration is completely valid
                    if show_cli_chrome:
                        console.print("[dim]✓ Configuration valid[/]")
            except typer.Exit:
                raise
            except Exception as e:
                # If validation itself fails, log it but don't block startup
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Configuration validation skipped due to error: {e}")
                if show_cli_chrome:
                    console.print("[dim yellow]⚠ Configuration validation skipped[/]")

            # Validate default model existence (Phase 1 UX improvement)
            from victor.config.settings import validate_default_model

            model_valid, model_warning = validate_default_model(settings)
            if not model_valid and model_warning:
                if not show_cli_chrome:
                    formatter.error("Configuration warning", model_warning)
                    raise typer.Exit(code=1)

                # Model validation failed - show warning and offer to continue
                console.print("\n[yellow]⚠ Configuration Warning:[/]")
                console.print(model_warning)
                console.print()

                # Ask user if they want to continue
                from rich.prompt import Confirm

                if not Confirm.ask("Continue anyway?", default=False, show_default=True):
                    console.print("\n[yellow]Setup cancelled.[/]")
                    console.print("Fix the issue above and run 'victor chat' again.\n")
                    raise typer.Exit(code=1)

            # Step 2: Initialize factory
            status.update("Initializing agent factory...")
            try:
                factory = AgentFactory(
                    settings=settings,
                    profile=profile,
                    vertical=vertical_param,
                    thinking=thinking,
                    session_id=session_id,
                    enable_observability=enable_observability,
                    tool_budget=tool_budget,
                    max_iterations=max_iterations if "max_iterations" in dir() else None,
                )
            except ConfigurationError as e:
                console.print(f"\n[red]✗[/] Configuration error: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Run 'victor doctor' to diagnose configuration issues")
                console.print("  • Check your profile configuration: victor profile list")
                console.print("  • Validate config: victor config validate\n")
                raise typer.Exit(code=1)
            except (ProviderConnectionError, ProviderNotFoundError) as e:
                console.print(f"\n[red]✗[/] Provider error: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Check if provider is available: victor doctor --providers")
                console.print("  • Verify provider configuration in profiles.yaml")
                console.print("  • Try a different profile: victor profile list\n")
                raise typer.Exit(code=1)
            except ProviderAuthError as e:
                console.print(f"\n[red]✗[/] Authentication error: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Verify API key is set: victor doctor --credentials")
                console.print("  • Check API key has required permissions")
                console.print("  • Try re-exporting your API key\n")
                raise typer.Exit(code=1)
            except Exception as e:
                # Unexpected error - show full traceback in debug mode
                console.print(f"\n[red]✗[/] Failed to initialize agent factory: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Run 'victor doctor' to diagnose configuration issues")
                console.print("  • Check your profile configuration: victor profile list")
                console.print("  • Try default profile: victor chat --profile default\n")
                if os.getenv("VICTOR_DEBUG"):
                    import traceback
                    console.print(traceback.format_exc())
                raise typer.Exit(code=1)

            # Step 3: Create agent
            status.update("Creating agent...")
            try:
                agent = await factory.create()
            except ConfigurationError as e:
                console.print(f"\n[red]✗[/] Configuration error during agent creation: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Check vertical configuration: victor vertical list")
                console.print("  • Validate config: victor config validate")
                console.print("  • Try default vertical: victor chat --vertical default\n")
                raise typer.Exit(code=1)
            except ProviderError as e:
                console.print(f"\n[red]✗[/] Provider error during agent creation: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Check provider status: victor doctor --providers")
                console.print("  • Verify provider is running (for local models)")
                console.print("  • Try a different provider: victor chat --provider ollama\n")
                raise typer.Exit(code=1)
            except Exception as e:
                # Unexpected error - show full traceback in debug mode
                console.print(f"\n[red]✗[/] Failed to create agent: {e}")
                console.print("\n[yellow]Suggestions:[/]")
                console.print("  • Check if provider is configured: victor doctor --providers")
                console.print("  • Verify API keys or local model availability")
                console.print("  • Try a different profile: victor profile list\n")
                if os.getenv("VICTOR_DEBUG"):
                    import traceback
                    console.print(traceback.format_exc())
                raise typer.Exit(code=1)

            # Set planning model override if provided (for planning coordinator)
            if planning_model:
                agent._planning_model_override = planning_model

            # Note: Observability (shim) is handled by AgentFactory internally
            # The factory creates the agent with framework features already wired

            # Step 4: Configure agent settings
            status.update("Configuring agent settings...")
            if tool_budget is not None:
                agent.unified_tracker.set_tool_budget(tool_budget, user_override=True)
            if max_iterations is not None:
                agent.unified_tracker.set_max_iterations(max_iterations, user_override=True)

            _configure_agent_compaction(
                agent,
                compaction_threshold=compaction_threshold,
                adaptive_threshold=adaptive_threshold,
                compaction_min_threshold=compaction_min_threshold,
                compaction_max_threshold=compaction_max_threshold,
                con=console,
                show_status=show_cli_chrome,
            )

            if mode:
                from victor.agent.mode_controller import AgentMode, get_mode_controller

                controller = get_mode_controller()
                try:
                    controller.switch_mode(AgentMode(mode))
                except Exception as e:
                    if show_cli_chrome:
                        console.print(f"\n[yellow]Warning:[/] Failed to set mode '{mode}': {e}")
                        console.print("  Continuing with default mode.\n")

            # Step 5: Preload semantic index if requested
            if preindex:
                status.update("Preloading semantic index...")
                try:
                    await preload_semantic_index(os.getcwd(), settings, console)
                except Exception as e:
                    if show_cli_chrome:
                        console.print(f"\n[yellow]Warning:[/] Failed to preload semantic index: {e}")
                        console.print(
                            "  Continuing without preindex (first search will be slower).\n"
                        )

            # Step 6: Start embedding preload
            status.update("Starting embedding preload...")
            try:
                agent.start_embedding_preload()
            except Exception as e:
                if show_cli_chrome:
                    console.print(f"\n[yellow]Warning:[/] Failed to start embedding preload: {e}")
                    console.print("  Continuing without embeddings (some features may be slower).\n")

            # Planning mode requires non-streaming (plan generation → step execution → summary)
            use_streaming = stream and agent.provider.supports_streaming() and not enable_planning

            # Display skill auto-selection feedback before response
            if show_cli_chrome:
                _display_skill_preview(console, agent, message)

            if use_streaming:
                from victor.ui.rendering import (
                    FormatterRenderer,
                    LiveDisplayRenderer,
                    stream_response,
                )

                use_live = show_cli_chrome and renderer_choice in {"rich", "auto"}
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
                if show_cli_chrome:
                    buffered.flush(console)
                else:
                    formatter.response(content=buffered.finalize())

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
    except typer.Exit:
        raise
    except Exception as e:
        # Use contextual error formatting for better UX
        error_message = format_exception_for_user(e)
        formatter.error(error_message)

        # Suggest provider switching for provider-specific errors
        error_str = str(e).lower()
        is_provider_error = any(
            term in error_str
            for term in [
                "api key",
                "unauthorized",
                "rate limit",
                "timeout",
                "connection",
                "network",
            ]
        )

        if is_provider_error:
            if provider_for_suggestions != "ollama":
                formatter.info(
                    "💡 Try switching to a different provider:\n"
                    "  victor chat -p ollama  # Local models (free)\n"
                    "  victor chat -p openai  # GPT models"
                )
            else:
                formatter.info(
                    "💡 Try switching to a cloud provider:\n"
                    "  victor chat -p anthropic  # Claude models\n"
                    "  victor chat -p openai  # GPT models"
                )

        formatter.info("Run 'victor doctor' for diagnostics")

        # Show traceback in debug mode only
        if os.getenv("VICTOR_DEBUG"):
            import traceback

            formatter.error(traceback.format_exc())

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
    compaction_threshold: Optional[float] = None,
    adaptive_threshold: Optional[bool] = None,
    compaction_min_threshold: Optional[float] = None,
    compaction_max_threshold: Optional[float] = None,
    # Smart routing parameters
    enable_smart_routing: bool = False,
    routing_profile: str = "balanced",
    fallback_chain: Optional[str] = None,
    # Tool output preview parameters
    tool_preview: bool = True,
    enable_pruning: bool = False,
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
    show_startup_cli_chrome = _should_render_interactive_tool_banner(use_tui)
    smart_routing_status_shown = False
    compaction_status_shown = False
    tui_startup_messages: list[str] = []
    tui_status_messages: list[str] = []

    _configure_smart_routing(
        settings,
        console,
        enable_smart_routing,
        routing_profile,
        fallback_chain,
        show_status=show_startup_cli_chrome,
    )
    smart_routing_status_shown = show_startup_cli_chrome and enable_smart_routing
    if use_tui:
        tui_status_messages.extend(
            _summarize_smart_routing(settings, enable_smart_routing, routing_profile)
        )

    # Configure tool output preview (safe-default pruning)
    from victor.config.tool_settings import ToolSettings

    # Apply tool output preview settings from flags
    if not tool_preview:
        settings.tool_settings.tool_output_preview_enabled = False
    if enable_pruning:
        settings.tool_settings.tool_output_pruning_enabled = True
        settings.tool_settings.tool_output_pruning_safe_only = False

    tool_settings = settings.tool_settings if hasattr(settings, "tool_settings") else ToolSettings()
    tool_banner_shown = False
    if show_startup_cli_chrome:
        _print_tool_output_mode_banner(console, tool_settings)
        tool_banner_shown = True
    elif use_tui:
        tui_status_messages.append(_summarize_tool_output_mode(tool_settings))

    try:
        profiles = settings.load_profiles()
        profile_config = profiles.get(profile)

        if not profile_config:
            # Profile not found - show available profiles
            if use_tui:
                tui_startup_messages.append(
                    f"Profile '{profile}' not found. Using default settings "
                    f"({settings.provider.default_provider}/{settings.provider.default_model})."
                )
                if profiles:
                    available = ", ".join(sorted(profiles.keys()))
                    tui_startup_messages.append(f"Available profiles: {available}")
                else:
                    tui_startup_messages.append(
                        "No profiles configured. Create profiles in ~/.victor/profiles.yaml."
                    )
            else:
                console.print(f"[bold red]Error:[/ ] Profile '{profile}' not found\n")

                if profiles:
                    console.print("[bold]Available profiles:[/]")
                    for profile_name in sorted(profiles.keys()):
                        profile_info = profiles[profile_name]
                        # ProfileConfig is a Pydantic model with direct attributes
                        provider = getattr(profile_info, 'provider', settings.provider.default_provider)
                        model = getattr(profile_info, 'model', settings.provider.default_model)
                        console.print(
                            f"  • [cyan]{profile_name}[/] (provider: {provider}, model: {model})"
                        )

                    console.print(
                        f"\n[dim]Using default settings "
                        f"({settings.provider.default_provider}/{settings.provider.default_model})[/]"
                    )
                else:
                    console.print("[yellow]No profiles configured. Using default settings.[/]")
                    console.print("[dim]Tip:[/] Create profiles in ~/.victor/profiles.yaml[/]\n")

            # Continue with default settings
            profile_config = None
        profile_display = profile_config or SimpleNamespace(
            provider=settings.provider.default_provider,
            model=settings.provider.default_model,
        )

        # Unified initialization via AgentFactory
        from victor.framework.agent_factory import AgentFactory, InitializationError

        # Pass original vertical string to AgentFactory (preserves name for error reporting)
        vertical_param = vertical if vertical else None

        # Use "default" profile if profile_config is None (default settings)
        profile_to_use = profile if profile_config else "default"

        factory = AgentFactory(
            settings=settings,
            profile=profile_to_use,
            vertical=vertical_param,
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
                        if use_tui:
                            tui_startup_messages.append(
                                f"Warning: failed to restore conversation state: {e}"
                            )
                        else:
                            console.print(
                                f"[yellow]Warning:[/] Failed to restore conversation state: {e}"
                            )

                resumed_message = (
                    f"Resumed session: {metadata.get('title', 'Untitled')} "
                    f"({metadata.get('message_count', 0)} messages)"
                )
                if use_tui:
                    tui_startup_messages.append(resumed_message)
                else:
                    console.print(f"[green]✓[/] {resumed_message}\n")

            # Note: Observability (shim) is handled by AgentFactory internally
            # The factory creates the agent with framework features already wired

            # Initialize file watchers for automatic cache invalidation
            import os

            from victor.core.indexing.watcher_initializer import initialize_from_context

            # Build execution context for file watcher initialization
            exec_ctx = {
                "cwd": os.getcwd(),
                "settings": settings,
            }

            try:
                await initialize_from_context(exec_ctx)
            except Exception as e:
                # Non-fatal: log warning but continue
                if use_tui:
                    tui_startup_messages.append(
                        f"Warning: failed to initialize file watchers: {e}. "
                        "Continuing without automatic file watching."
                    )
                else:
                    console.print(
                        f"[yellow]Warning:[/] Failed to initialize file watchers: {e}"
                    )
                    console.print("  Continuing without automatic file watching.\n")
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
        _configure_agent_compaction(
            agent,
            compaction_threshold=compaction_threshold,
            adaptive_threshold=adaptive_threshold,
            compaction_min_threshold=compaction_min_threshold,
            compaction_max_threshold=compaction_max_threshold,
            con=console,
            show_status=show_startup_cli_chrome,
        )
        compaction_status_shown = show_startup_cli_chrome and any(
            (
                compaction_threshold is not None,
                adaptive_threshold is not None,
                compaction_min_threshold is not None,
                compaction_max_threshold is not None,
            )
        )
        if use_tui:
            tui_status_messages.extend(
                _summarize_compaction_overrides(
                    compaction_threshold=compaction_threshold,
                    adaptive_threshold=adaptive_threshold,
                    compaction_min_threshold=compaction_min_threshold,
                    compaction_max_threshold=compaction_max_threshold,
                )
            )

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
                    provider=profile_display.provider,
                    model=profile_display.model,
                    stream=stream,
                    settings=settings,  # Pass settings for slash commands
                    startup_messages=[*tui_status_messages, *tui_startup_messages],
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
            if enable_smart_routing and not smart_routing_status_shown:
                _configure_smart_routing(
                    settings,
                    console,
                    enable_smart_routing,
                    routing_profile,
                    fallback_chain,
                    show_status=True,
                )
                smart_routing_status_shown = True
            if not tool_banner_shown:
                _print_tool_output_mode_banner(console, tool_settings)
                tool_banner_shown = True
            if not compaction_status_shown and any(
                (
                    compaction_threshold is not None,
                    adaptive_threshold is not None,
                    compaction_min_threshold is not None,
                    compaction_max_threshold is not None,
                )
            ):
                _configure_agent_compaction(
                    agent,
                    compaction_threshold=compaction_threshold,
                    adaptive_threshold=adaptive_threshold,
                    compaction_min_threshold=compaction_min_threshold,
                    compaction_max_threshold=compaction_max_threshold,
                    con=console,
                    show_status=True,
                )
                compaction_status_shown = True
            from victor.ui.commands import SlashCommandHandler

            cmd_handler = SlashCommandHandler(console, settings, agent)

            rl_suggestion = get_rl_profile_suggestion(profile_display.provider, profiles)
            await _run_cli_repl(
                agent,
                settings,
                cmd_handler,
                profile_display,
                stream,
                rl_suggestion,
                renderer_choice=renderer_choice,
                vertical_name=vertical,
                enable_planning=enable_planning,
                show_reasoning=show_reasoning,
                profile_name=profile,
                startup_messages=tui_startup_messages,
            )

        success = True
        if hasattr(agent, "get_session_metrics"):
            metrics = agent.get_session_metrics()
            tool_calls_made = metrics.get("tool_calls", 0) if metrics else 0

    except Exception as e:
        # Use contextual error formatting for better UX
        error_message = format_exception_for_user(e)
        console.print(f"[bold red]Error:[/]\n{error_message}")

        # Show traceback in debug mode only
        import os
        if os.getenv("VICTOR_DEBUG"):
            import traceback

            console.print(traceback.format_exc())

        # Suggest provider switching for provider-specific errors
        error_str = str(e).lower()
        is_provider_error = any(
            term in error_str
            for term in [
                "api key",
                "unauthorized",
                "rate limit",
                "timeout",
                "connection",
                "network",
            ]
        )

        if is_provider_error:
            console.print(
                "\n[yellow]💡 Try switching to a different provider:[/]"
                "\n  [dim]victor chat -p ollama[/]  # Local models (free)"
                "\n  [dim]victor chat -p anthropic[/]  # Claude models"
            )

        console.print("\n[yellow]💡 Run 'victor doctor' for diagnostics[/]")
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

        # Cleanup file watchers
        try:
            from victor.core.indexing.watcher_initializer import cleanup_session

            await cleanup_session()
        except Exception as e:
            # Non-fatal: log warning but don't fail shutdown
            console.print(f"[dim]Note: Failed to cleanup file watchers: {e}[/]")


def _prune_history_file(history_file, max_entries: int = 250) -> None:
    """Prune history file to keep only the most recent entries.

    This prevents the history file from growing unbounded, which would
    slow down prompt_toolkit's history search on every keystroke.

    Args:
        history_file: Path to history file
        max_entries: Maximum number of entries to keep (default 200)
    """
    if not history_file.exists():
        return

    try:
        with open(history_file, 'r') as f:
            lines = f.readlines()

        # If file is small enough, no pruning needed
        if len(lines) <= max_entries:
            return

        # Keep only the last max_entries lines
        pruned_lines = lines[-max_entries:]

        # Write back to file
        with open(history_file, 'w') as f:
            f.writelines(pruned_lines)

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to prune history file {history_file}: {e}")


def _maybe_rotate_history_file(history_file, max_entries: int = 250) -> None:
    """Periodically rotate history file if it grows too large.

    This is called periodically to prevent the history file from
    growing unbounded over time.

    Args:
        history_file: Path to history file
        max_entries: Maximum entries before rotation
    """
    if not history_file.exists():
        return

    try:
        import time

        # Only check once per session
        rotation_marker = history_file.with_suffix('.last_rotation')

        if rotation_marker.exists():
            last_rotation = rotation_marker.stat().st_mtime
            # Only rotate if more than 7 days since last rotation
            if time.time() - last_rotation < 7 * 24 * 3600:
                return

        # Count lines
        with open(history_file, 'r') as f:
            line_count = sum(1 for _ in f)

        # If too large, prune
        if line_count > max_entries * 1.5:  # 50% buffer
            _prune_history_file(history_file, max_entries)

            # Update rotation marker
            rotation_marker.touch()

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Failed to rotate history file: {e}")


def _check_history_health(history_file, max_entries: int = 250) -> None:
    """Check history file health and warn if too large.

    Args:
        history_file: Path to history file
        max_entries: Configured maximum entries (for warning threshold)
    """
    if not history_file.exists():
        return

    try:
        line_count = sum(1 for _ in open(history_file))
        size_kb = history_file.stat().st_size / 1024

        # Warning threshold: 2x the configured max_entries or 500, whichever is higher
        warning_threshold = max(max_entries * 2, 500)
        size_warning_threshold = 50  # KB

        if line_count > warning_threshold or size_kb > size_warning_threshold:
            console.print(
                f"[yellow]Warning:[/] CLI history file is large "
                f"({line_count:,} entries, {size_kb:.1f} KB). "
                f"This may slow down typing. "
                f"Run 'victor chat cleanup-history' to optimize."
            )
    except Exception:
        pass


@chat_app.command("cleanup-history")
def cleanup_history_command(
    max_entries: int = typer.Option(
        250,
        "--max-entries",
        "-m",
        help="Maximum entries to keep (default: 250, range: 10-1000).",
        min=10,
        max=1000,
    ),
):
    """Clean up CLI chat history to improve typing performance.

    Reduces the history file to the most recent entries, which improves
    prompt_toolkit's search performance and makes typing feel snappier.

    **Examples:**
        victor chat cleanup-history          # Keep default 250 entries
        victor chat cleanup-history -m 500   # Keep 500 entries
    """
    from victor.config.settings import get_project_paths

    history_file = get_project_paths().project_victor_dir / "chat_history"

    if not history_file.exists():
        console.print("[green]✓[/] No history file to clean up")
        return

    try:
        # Count original lines
        with open(history_file, "r") as f:
            original_lines = sum(1 for _ in f)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to read history file: {e}")
        raise typer.Exit(1)

    # Prune the history file
    _prune_history_file(history_file, max_entries=max_entries)

    # Count new lines
    try:
        with open(history_file, "r") as f:
            new_lines = sum(1 for _ in f)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to read history file after pruning: {e}")
        raise typer.Exit(1)

    removed = original_lines - new_lines
    size_kb = history_file.stat().st_size / 1024

    console.print(f"[green]✓[/] Cleaned up history file:")
    console.print(f"  Entries: {original_lines:,} → {new_lines:,} (removed {removed:,})")
    console.print(f"  File size: {size_kb:.1f} KB")
    console.print()
    console.print("[dim]Typing should feel snappier now![/]")


def _create_cli_prompt_session(settings=None):
    """Create a prompt_toolkit PromptSession with persistent history.

    Loads previous user messages from the conversation database and persists
    new input to .victor/chat_history for project-specific history.

    Args:
        settings: Optional settings object (to avoid redundant load_settings call)
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    # Get max_entries from settings or use default
    if settings and hasattr(settings, 'ui'):
        max_entries = settings.ui.cli_history_max_entries
    else:
        max_entries = 250  # Default limit for CLI history

    # Use a persistent history file in project-specific .victor/
    try:
        from victor.config.settings import get_project_paths

        history_file = get_project_paths().project_victor_dir / "chat_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        # Prune history file if too large
        _prune_history_file(history_file, max_entries=max_entries)

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

                        # Prune again after seeding to ensure we stay at max_entries
                        _prune_history_file(history_file, max_entries=max_entries)
            except Exception:
                pass  # DB seeding is best-effort

    except Exception:
        history = InMemoryHistory()

    # Create key bindings for the session
    key_bindings = KeyBindings()

    return PromptSession(
        history=history,
        key_bindings=key_bindings,
        complete_while_typing=False,  # Disable per-keystroke completion checks for better performance
        enable_history_search=False,  # Disable history search on typing for better performance
    )


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
    profile_name: str = "default",
    startup_messages: Optional[list[str]] = None,
) -> None:
    """Run the CLI-based REPL (fallback for unsupported terminals)."""
    console.print(_build_cli_panel(profile_config, vertical_name, rl_suggestion, settings))
    if startup_messages:
        _print_interactive_startup_messages(console, startup_messages)

    # Set up prompt_toolkit with persistent history for Up/Down arrow navigation
    prompt_session = _create_cli_prompt_session(settings=settings)

    # Add Ctrl+O hotkey to expand last tool output.
    # _active_renderer is updated each turn so the binding always refers to
    # the most recent renderer instance.
    from prompt_toolkit.keys import Keys

    _active_renderer = None

    @prompt_session.key_bindings.add(Keys.ControlO)
    def _expand_output(event):
        """Expand last tool output to show full content."""
        if _active_renderer is not None and hasattr(_active_renderer, "expand_last_output"):
            _active_renderer.expand_last_output()

    while True:
        renderer = None
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

            console.print("[bold cyan]Assistant[/]")

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
                if use_live:
                    renderer = LiveDisplayRenderer(console)
                else:
                    formatter = create_formatter()
                    renderer = FormatterRenderer(formatter, console)

                # Release previous renderer before swapping to prevent Live display leaks.
                if _active_renderer is not None and hasattr(_active_renderer, "cleanup"):
                    _active_renderer.cleanup()
                # Update binding target so Ctrl+O always refers to the current renderer
                _active_renderer = renderer

                content_buffer = await stream_response(
                    agent,
                    user_input,
                    renderer,
                    suppress_thinking=not show_reasoning,
                )
                content_buffer = sanitize_response(content_buffer)

                # Renderers own streamed output. Re-printing the returned buffer here
                # duplicates the assistant response for FormatterRenderer.
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
            # Use contextual error formatting for better UX
            error_message = format_exception_for_user(e)
            console.print(f"[bold red]Error:[/]\n{error_message}")

            # Show traceback in debug mode only
            import os  # Import os for getenv
            if os.getenv("VICTOR_DEBUG"):
                import traceback

                console.print(traceback.format_exc())

            # Save conversation state on error for recovery
            try:
                if hasattr(agent, "get_conversation_history"):
                    from victor.agent.sqlite_session_persistence import (
                        get_sqlite_session_persistence,
                    )

                    persistence = get_sqlite_session_persistence()
                    conversation = agent.get_conversation_history()

                    # Get current provider/model for context
                    provider = getattr(profile_config, "provider", "unknown")
                    model = getattr(profile_config, "model", "unknown")

                    session_id = persistence.save_session(
                        conversation=conversation,
                        model=model,
                        provider=provider,
                        profile=profile_name,
                        tags=["error-recovery", "auto-saved"],
                    )
                    console.print(
                        f"\n[dim]💾 Conversation auto-saved. Resume with: victor chat --resume {session_id}[/]"
                    )
            except Exception:
                # Silently fail if save doesn't work — error message is more important
                pass

            # Suggest provider switching for provider-specific errors
            error_str = str(e).lower()
            is_provider_error = any(
                term in error_str
                for term in [
                    "api key",
                    "unauthorized",
                    "rate limit",
                    "timeout",
                    "connection",
                    "network",
                ]
            )

            if is_provider_error and provider != "ollama":
                console.print(
                    "\n[yellow]💡 Try switching to a different provider:[/]"
                    "\n  [dim]victor chat -p ollama[/]  # Local models (free)[/]"
                    "\n  [dim]victor chat -p openai[/]  # GPT models[/]"
                )
            elif is_provider_error:
                console.print(
                    "\n[yellow]💡 Try switching to a cloud provider:[/]"
                    "\n  [dim]victor chat -p anthropic[/]  # Claude models[/]"
                    "\n  [dim]victor chat -p openai[/]  # GPT models[/]"
                )

            console.print("\n[yellow]💡 Run 'victor doctor' for diagnostics[/]")

        finally:
            # Ensure renderer cleanup happens in ALL exit paths
            if renderer is not None and hasattr(renderer, "cleanup"):
                renderer.cleanup()
                if _active_renderer == renderer:
                    _active_renderer = None


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
    except ConfigurationError as e:
        console.print(f"[bold red]✗[/] Workflow configuration error: {e}")
        console.print("\n[yellow]Suggestions:[/]")
        console.print("  • Check workflow YAML syntax")
        console.print("  • Validate workflow: victor workflow validate <file>")
        console.print("  • Run 'victor doctor' for diagnostics\n")
        raise typer.Exit(1)
    except VictorError as e:
        # Known Victor errors - use contextual formatting
        error_message = format_exception_for_user(e)
        console.print(f"[bold red]Error:[/]\n{error_message}")
        console.print("\n[yellow]💡 Run 'victor doctor' for diagnostics[/]")
        if os.getenv("VICTOR_DEBUG"):
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        # Unexpected error - always show in debug mode
        console.print(f"[bold red]Unexpected error:[/]\n{e}")
        console.print("\n[yellow]Suggestions:[/]")
        console.print("  • This may be a bug - please report it")
        console.print("  • Run 'victor doctor' for diagnostics")
        console.print("  • Enable debug mode: VICTOR_DEBUG=1 victor chat\n")
        import traceback
        if os.getenv("VICTOR_DEBUG"):
            console.print(traceback.format_exc())
        raise typer.Exit(1)
