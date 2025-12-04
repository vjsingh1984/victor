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

"""Interactive slash commands for Victor CLI.

This module provides slash command handling for the interactive REPL,
similar to Claude Code's /help, /model, etc.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

from victor.agent.session import get_session_manager

logger = logging.getLogger(__name__)


class SlashCommand:
    """Represents a slash command with its handler."""

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable,
        aliases: Optional[List[str]] = None,
        usage: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.aliases = aliases or []
        self.usage = usage or f"/{name}"


class SlashCommandHandler:
    """Handles slash commands in the interactive REPL."""

    def __init__(
        self,
        console: Console,
        settings: "Settings",
        agent: Optional["AgentOrchestrator"] = None,
    ):
        self.console = console
        self.settings = settings
        self.agent = agent
        self._commands: Dict[str, SlashCommand] = {}
        self._register_default_commands()

    def set_agent(self, agent: "AgentOrchestrator") -> None:
        """Set the agent reference (for commands that need it)."""
        self.agent = agent

    def _register_default_commands(self) -> None:
        """Register built-in slash commands."""
        self.register(
            SlashCommand(
                name="help",
                description="Show available commands",
                handler=self._cmd_help,
                aliases=["?", "commands"],
                usage="/help [command]",
            )
        )

        self.register(
            SlashCommand(
                name="init",
                description="Initialize .victor.md with smart codebase analysis",
                handler=self._cmd_init,
                usage="/init [--force] [--simple] [--smart] [--symlinks]",
            )
        )

        self.register(
            SlashCommand(
                name="model",
                description="List available models or switch model",
                handler=self._cmd_model,
                aliases=["models"],
                usage="/model [model_name]",
            )
        )

        self.register(
            SlashCommand(
                name="profile",
                description="Show or switch profile",
                handler=self._cmd_profile,
                aliases=["profiles"],
                usage="/profile [profile_name]",
            )
        )

        self.register(
            SlashCommand(
                name="clear",
                description="Clear conversation history",
                handler=self._cmd_clear,
                aliases=["reset"],
            )
        )

        self.register(
            SlashCommand(
                name="context",
                description="Show loaded project context (.victor.md)",
                handler=self._cmd_context,
                aliases=["ctx", "memory"],
            )
        )

        self.register(
            SlashCommand(
                name="lmstudio",
                description="Probe LMStudio endpoints and suggest a VRAM-friendly model",
                handler=self._cmd_lmstudio,
                aliases=["lm"],
                usage="/lmstudio",
            )
        )

        self.register(
            SlashCommand(
                name="tools",
                description="List available tools",
                handler=self._cmd_tools,
                usage="/tools [search_pattern]",
            )
        )

        self.register(
            SlashCommand(
                name="status",
                description="Show current session status",
                handler=self._cmd_status,
                aliases=["info"],
            )
        )

        self.register(
            SlashCommand(
                name="config",
                description="Show current configuration",
                handler=self._cmd_config,
                aliases=["settings"],
            )
        )

        # Session management commands
        self.register(
            SlashCommand(
                name="save",
                description="Save current conversation to a session file",
                handler=self._cmd_save,
                usage="/save [name]",
            )
        )

        self.register(
            SlashCommand(
                name="load",
                description="Load a saved session",
                handler=self._cmd_load,
                usage="/load <session_id>",
            )
        )

        self.register(
            SlashCommand(
                name="sessions",
                description="List saved sessions",
                handler=self._cmd_sessions,
                aliases=["history"],
                usage="/sessions [limit]",
            )
        )

        # UX parity commands (Claude Code, Cursor, etc.)
        self.register(
            SlashCommand(
                name="compact",
                description="Compress conversation history (use --smart for AI summarization)",
                handler=self._cmd_compact,
                aliases=["summarize"],
                usage="/compact [--smart]",
            )
        )

        self.register(
            SlashCommand(
                name="mcp",
                description="Show MCP server status and connections",
                handler=self._cmd_mcp,
                aliases=["servers"],
                usage="/mcp",
            )
        )

        self.register(
            SlashCommand(
                name="review",
                description="Request a code review for recent changes",
                handler=self._cmd_review,
                usage="/review [file or directory]",
            )
        )

        self.register(
            SlashCommand(
                name="bug",
                description="Report an issue or bug",
                handler=self._cmd_bug,
                aliases=["issue", "feedback"],
                usage="/bug",
            )
        )

        self.register(
            SlashCommand(
                name="exit",
                description="Exit Victor",
                handler=self._cmd_exit,
                aliases=["quit", "bye"],
                usage="/exit",
            )
        )

        # Undo/Redo commands (OpenCode parity)
        self.register(
            SlashCommand(
                name="undo",
                description="Undo the last file change(s)",
                handler=self._cmd_undo,
                usage="/undo",
            )
        )

        self.register(
            SlashCommand(
                name="redo",
                description="Redo the last undone change(s)",
                handler=self._cmd_redo,
                usage="/redo",
            )
        )

        self.register(
            SlashCommand(
                name="history",
                description="Show file change history",
                handler=self._cmd_history,
                aliases=["timeline"],
                usage="/history [limit]",
            )
        )

        self.register(
            SlashCommand(
                name="theme",
                description="Toggle between dark and light theme",
                handler=self._cmd_theme,
                aliases=["dark", "light"],
                usage="/theme [dark|light]",
            )
        )

        self.register(
            SlashCommand(
                name="changes",
                description="View, diff, or revert file changes",
                handler=self._cmd_changes,
                aliases=["diff", "undo", "rollback"],
                usage="/changes [show|revert|stash] [file]",
            )
        )

        self.register(
            SlashCommand(
                name="cost",
                description="Show estimated token usage and cost for this session",
                handler=self._cmd_cost,
                aliases=["usage", "tokens", "stats"],
                usage="/cost",
            )
        )

        # Codex CLI parity commands
        self.register(
            SlashCommand(
                name="approvals",
                description="Configure what actions require user approval",
                handler=self._cmd_approvals,
                aliases=["safety"],
                usage="/approvals [suggest|auto|full-auto]",
            )
        )

        self.register(
            SlashCommand(
                name="resume",
                description="Resume the most recent session",
                handler=self._cmd_resume,
                usage="/resume",
            )
        )

        # Cursor parity commands
        self.register(
            SlashCommand(
                name="plan",
                description="Enter planning mode - research before coding",
                handler=self._cmd_plan,
                usage="/plan [task description]",
            )
        )

        self.register(
            SlashCommand(
                name="search",
                description="Toggle web search capability",
                handler=self._cmd_search,
                aliases=["web"],
                usage="/search [on|off]",
            )
        )

        # Gemini CLI parity commands
        self.register(
            SlashCommand(
                name="copy",
                description="Copy last assistant response to clipboard",
                handler=self._cmd_copy,
                usage="/copy",
            )
        )

        self.register(
            SlashCommand(
                name="directory",
                description="Show or change working directory",
                handler=self._cmd_directory,
                aliases=["dir", "cd", "pwd"],
                usage="/directory [path]",
            )
        )

        # Snapshot and commit commands (Aider/Cline parity)
        self.register(
            SlashCommand(
                name="snapshots",
                description="Manage workspace snapshots for safe rollback",
                handler=self._cmd_snapshots,
                aliases=["snap"],
                usage="/snapshots [list|create|restore|diff|clear] [id]",
            )
        )

        self.register(
            SlashCommand(
                name="commit",
                description="Commit current changes with AI-generated message",
                handler=self._cmd_commit,
                aliases=["ci"],
                usage="/commit [message]",
            )
        )

        # Agent mode commands (OpenCode parity)
        self.register(
            SlashCommand(
                name="mode",
                description="Switch agent mode (build/plan/explore)",
                handler=self._cmd_mode,
                aliases=["m"],
                usage="/mode [build|plan|explore]",
            )
        )

        self.register(
            SlashCommand(
                name="build",
                description="Switch to build mode for implementation",
                handler=lambda args: self._cmd_mode(["build"]),
                usage="/build",
            )
        )

        self.register(
            SlashCommand(
                name="explore",
                description="Switch to explore mode for code navigation",
                handler=lambda args: self._cmd_mode(["explore"]),
                usage="/explore",
            )
        )

        # Codebase indexing commands
        self.register(
            SlashCommand(
                name="reindex",
                description="Reindex codebase for semantic search",
                handler=self._cmd_reindex,
                aliases=["index"],
                usage="/reindex [--force] [--stats]",
            )
        )

        # Performance metrics command
        self.register(
            SlashCommand(
                name="metrics",
                description="Show streaming performance metrics and provider stats",
                handler=self._cmd_metrics,
                aliases=["perf", "performance"],
                usage="/metrics [summary|history|export] [--json|--csv]",
            )
        )

    def register(self, command: SlashCommand) -> None:
        """Register a slash command."""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._commands[alias] = command

    def is_command(self, text: str) -> bool:
        """Check if text is a slash command."""
        return text.strip().startswith("/")

    def parse_command(self, text: str) -> Tuple[str, List[str]]:
        """Parse command name and arguments from text.

        Args:
            text: Raw command text (e.g., "/model qwen2.5:7b")

        Returns:
            Tuple of (command_name, [args])
        """
        parts = text.strip().split()
        if not parts:
            return "", []

        cmd_name = parts[0].lstrip("/").lower()
        args = parts[1:] if len(parts) > 1 else []
        return cmd_name, args

    async def execute(self, text: str) -> bool:
        """Execute a slash command.

        Args:
            text: Raw command text

        Returns:
            True if command was handled, False otherwise
        """
        cmd_name, args = self.parse_command(text)

        if not cmd_name:
            return False

        command = self._commands.get(cmd_name)
        if not command:
            self.console.print(f"[red]Unknown command:[/] /{cmd_name}")
            self.console.print("Type [bold]/help[/] for available commands")
            return True

        try:
            result = command.handler(args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            self.console.print(f"[red]Command error:[/] {e}")
            logger.exception(f"Error executing command /{cmd_name}")

        return True

    # Command handlers

    def _cmd_help(self, args: List[str]) -> None:
        """Show help for commands."""
        if args:
            # Help for specific command
            cmd_name = args[0].lstrip("/")
            command = self._commands.get(cmd_name)
            if command:
                self.console.print(
                    Panel(
                        f"[bold]/{command.name}[/]\n\n"
                        f"{command.description}\n\n"
                        f"[dim]Usage:[/] {command.usage}\n"
                        f"[dim]Aliases:[/] {', '.join('/' + a for a in command.aliases) if command.aliases else 'none'}",
                        title=f"Help: /{command.name}",
                        border_style="blue",
                    )
                )
            else:
                self.console.print(f"[yellow]Unknown command:[/] /{cmd_name}")
            return

        # Show all commands
        table = Table(title="Available Commands", show_header=True)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Aliases", style="dim")

        # Get unique commands (skip aliases)
        seen = set()
        for name, cmd in sorted(self._commands.items()):
            if cmd.name in seen:
                continue
            seen.add(cmd.name)
            aliases = ", ".join(f"/{a}" for a in cmd.aliases) if cmd.aliases else ""
            table.add_row(f"/{cmd.name}", cmd.description, aliases)

        self.console.print(table)
        self.console.print("\n[dim]Type /help <command> for more details[/]")

    async def _cmd_init(self, args: List[str]) -> None:
        """Initialize .victor.md file with smart codebase analysis."""
        force = "--force" in args or "-f" in args
        simple = "--simple" in args or "-s" in args
        smart = "--smart" in args or "-a" in args  # -a for AI-powered
        symlinks = "--symlinks" in args or "-l" in args
        target_path = Path.cwd() / ".victor.md"

        if target_path.exists() and not force:
            self.console.print(f"[yellow].victor.md already exists at {target_path}[/]")
            self.console.print("Use [bold]/init --force[/] to overwrite")
            self.console.print("Use [bold]/init --simple[/] for basic template")
            self.console.print("Use [bold]/init --smart[/] for LLM-powered analysis (any language)")
            self.console.print("Use [bold]/init --symlinks[/] to create aliases for other tools")

            # Show current content preview
            content = target_path.read_text()[:500]
            self.console.print(
                Panel(
                    Markdown(
                        content + "\n\n..." if len(target_path.read_text()) > 500 else content
                    ),
                    title="Current .victor.md",
                    border_style="dim",
                )
            )
            return

        if smart:
            self.console.print("[dim]Analyzing codebase with LLM (works with any language)...[/]")
        else:
            self.console.print("[dim]Analyzing codebase...[/]")

        try:
            if smart:
                # Use LLM-powered generator (works with any project type)
                if not self.agent or not self.agent.provider:
                    self.console.print("[yellow]No active provider configured.[/]")
                    self.console.print(
                        "Use [bold]/init[/] without --smart, or configure a provider first."
                    )
                    return

                from victor.context.codebase_analyzer import generate_victor_md_with_llm

                provider_name = getattr(self.agent.provider, "name", "unknown")
                model_name = getattr(self.agent, "model", "unknown")
                self.console.print(f"[dim]Using provider: {provider_name}, model: {model_name}[/]")

                content = await generate_victor_md_with_llm(self.agent.provider, model=model_name)
            elif simple:
                # Use simple generator
                from victor.context.project_context import generate_victor_md

                content = generate_victor_md()
            else:
                # Use smart analyzer (Python-specific, no LLM)
                from victor.context.codebase_analyzer import generate_smart_victor_md

                content = generate_smart_victor_md()

            # Write the file
            target_path.write_text(content, encoding="utf-8")
            self.console.print(f"[green]✓[/] Created {target_path}")

            # Show what was detected
            lines = content.split("\n")
            component_count = content.count("| `")
            pattern_count = content.count(". **") + content.count("Pattern:")

            self.console.print(f"[dim]  - Detected {component_count} key components[/]")
            self.console.print(f"[dim]  - Found {pattern_count} architecture patterns[/]")

            # Create symlinks for other AI tools if requested
            if symlinks:
                from victor.context.codebase_analyzer import (
                    CONTEXT_FILE_ALIASES,
                    create_context_symlinks,
                )

                self.console.print("\n[dim]Creating symlinks for other AI tools...[/]")
                results = create_context_symlinks()

                for alias, status in results.items():
                    tool_name = CONTEXT_FILE_ALIASES.get(alias, "Unknown")
                    if status == "created":
                        self.console.print(f"  [green]✓[/] {alias} -> .victor.md ({tool_name})")
                    elif status == "exists":
                        self.console.print(f"  [dim]○[/] {alias} (already linked)")
                    elif status == "exists_file":
                        self.console.print(f"  [yellow]![/] {alias} (file exists, not a symlink)")
                    elif status == "exists_different":
                        self.console.print(f"  [yellow]![/] {alias} (symlink to different target)")
                    else:
                        self.console.print(f"  [red]✗[/] {alias}: {status}")

            self.console.print("\n[dim]Review and customize as needed.[/]")

            # Reload context if agent is available
            if self.agent and hasattr(self.agent, "project_context"):
                self.agent.project_context.load()
                self.console.print("[green]✓[/] Context reloaded")

        except Exception as e:
            self.console.print(f"[red]Failed to create .victor.md:[/] {e}")
            logger.exception("Error in /init command")

    async def _cmd_model(self, args: List[str]) -> None:
        """List or switch models."""
        if args:
            # Switch to specified model
            model_name = args[0]
            if self.agent:
                self.agent.model = model_name
                self.console.print(f"[green]✓[/] Switched to model: [cyan]{model_name}[/]")
            else:
                self.console.print("[yellow]No active session to switch model[/]")
            return

        # List available models
        self.console.print("[dim]Fetching available models...[/]")

        try:
            from victor.providers.ollama_provider import OllamaProvider

            provider_settings = self.settings.get_provider_settings("ollama")
            ollama = OllamaProvider(**provider_settings)

            models_list = await ollama.list_models()

            if not models_list:
                self.console.print("[yellow]No models found[/]")
                self.console.print("Pull a model: [bold]ollama pull qwen2.5-coder:7b[/]")
                await ollama.close()
                return

            table = Table(title="Available Models", show_header=True)
            table.add_column("Model", style="cyan")
            table.add_column("Size", style="yellow")
            table.add_column("Status", style="green")

            current_model = self.agent.model if self.agent else None

            for model in models_list:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                size_gb = size / (1024**3) if size else 0

                status = "← current" if name == current_model else ""
                table.add_row(
                    name,
                    f"{size_gb:.1f} GB" if size_gb > 0 else "?",
                    status,
                )

            self.console.print(table)
            self.console.print("\n[dim]Switch model: /model <model_name>[/]")

            await ollama.close()

        except Exception as e:
            self.console.print(f"[red]Error listing models:[/] {e}")
            self.console.print("Make sure Ollama is running")

    def _cmd_profile(self, args: List[str]) -> None:
        """Show or switch profile."""
        profiles = self.settings.load_profiles()

        if args:
            # Switch to specified profile
            profile_name = args[0]
            if profile_name in profiles:
                self.console.print("[yellow]Profile switching requires restart[/]")
                self.console.print(f"Run: [bold]victor --profile {profile_name}[/]")
            else:
                self.console.print(f"[red]Profile not found:[/] {profile_name}")
                self.console.print(f"Available: {', '.join(profiles.keys())}")
            return

        # Show profiles
        table = Table(title="Configured Profiles", show_header=True)
        table.add_column("Profile", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Model", style="yellow")

        for name, config in profiles.items():
            table.add_row(name, config.provider, config.model)

        self.console.print(table)

    def _cmd_clear(self, args: List[str]) -> None:
        """Clear conversation history."""
        if self.agent:
            self.agent.reset_conversation()
            self.console.print("[green]✓[/] Conversation cleared")
        else:
            self.console.print("[yellow]No active session[/]")

    def _cmd_context(self, args: List[str]) -> None:
        """Show loaded project context."""
        if not self.agent or not hasattr(self.agent, "project_context"):
            self.console.print("[yellow]No project context loaded[/]")
            self.console.print("Run [bold]/init[/] to create .victor.md")
            return

        ctx = self.agent.project_context
        if not ctx.content:
            self.console.print("[yellow]No .victor.md found in project[/]")
            self.console.print("Run [bold]/init[/] to create one")
            return

        self.console.print(
            Panel(
                Markdown(ctx.content),
                title=f"Project Context: {ctx.context_file}",
                border_style="blue",
            )
        )

    async def _cmd_lmstudio(self, args: List[str]) -> None:
        """Probe LMStudio endpoints and recommend a model within VRAM budget."""
        try:
            import httpx
        except Exception:
            self.console.print("[red]httpx is required for LMStudio probing[/]")
            return

        urls = getattr(self.settings, "lmstudio_base_urls", [])
        if not urls:
            self.console.print("[yellow]No LMStudio endpoints configured[/]")
            return

        vram = None
        try:
            vram = self.settings._detect_vram_gb()
        except Exception:
            vram = None

        if vram:
            self.console.print(f"[dim]Detected VRAM (best-effort): ~{vram:.1f} GB[/]")
        else:
            self.console.print("[dim]VRAM detection unavailable; showing all models[/]")

        self.console.print("[dim]Probing LMStudio endpoints...[/]")
        table = Table(title="LMStudio Endpoints", show_header=True)
        table.add_column("Endpoint", style="cyan")
        table.add_column("Reachable", style="green")
        table.add_column("Models (count)", style="yellow")
        table.add_column("Sample", style="white")

        recommended = None
        for url in urls:
            models = []
            try:
                resp = httpx.get(f"{url.rstrip('/')}/v1/models", timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json() or {}
                    models = [
                        str(m.get("id") or m.get("model") or "")
                        for m in data.get("data", [])
                        if isinstance(m, dict) and (m.get("id") or m.get("model"))
                    ]
            except Exception:
                pass

            reachable = bool(models)
            sample = ", ".join(models[:3]) if models else ""
            table.add_row(url, "yes" if reachable else "no", str(len(models)), sample)

            if recommended is None and reachable:
                recommended = self.settings._choose_default_lmstudio_model(
                    [url],
                    max_vram_gb=getattr(self.settings, "lmstudio_max_vram_gb", None),
                )

        self.console.print(table)

        if recommended:
            if vram:
                self.console.print(f"[green]Recommended (fits VRAM):[/] {recommended}")
            else:
                self.console.print(f"[green]Recommended:[/] {recommended}")
        else:
            self.console.print("[yellow]No reachable LMStudio endpoint found[/]")

    def _cmd_tools(self, args: List[str]) -> None:
        """List available tools."""
        if not self.agent:
            self.console.print("[yellow]No active session[/]")
            return

        search = args[0].lower() if args else None

        table = Table(title="Available Tools", show_header=True)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Status", style="green")

        tools = self.agent.tools.list_tools()
        for tool in sorted(tools, key=lambda t: t.name):
            tool_name = tool.name

            # Filter by search pattern
            if search and search not in tool_name.lower():
                continue

            desc = (tool.description or "")[:60]
            if len(tool.description or "") > 60:
                desc += "..."

            status = "enabled"
            table.add_row(tool_name, desc, status)

        self.console.print(table)
        self.console.print(f"\n[dim]Total: {len(self.agent.tools.list_tools())} tools[/]")

    def _cmd_status(self, args: List[str]) -> None:
        """Show session status."""
        if not self.agent:
            self.console.print("[yellow]No active session[/]")
            return

        self.console.print(
            Panel(
                f"[bold]Provider:[/] {self.agent.provider.__class__.__name__}\n"
                f"[bold]Model:[/] {self.agent.model}\n"
                f"[bold]Temperature:[/] {self.agent.temperature}\n"
                f"[bold]Max Tokens:[/] {self.agent.max_tokens}\n"
                f"[bold]Tool Budget:[/] {self.agent.tool_calls_used}/{self.agent.tool_budget}\n"
                f"[bold]Messages:[/] {len(self.agent.messages)}\n"
                f"[bold]Thinking Mode:[/] {'enabled' if self.agent.thinking else 'disabled'}\n"
                f"[bold]Project Context:[/] {'loaded' if self.agent.project_context.content else 'none'}",
                title="Session Status",
                border_style="blue",
            )
        )

    def _cmd_config(self, args: List[str]) -> None:
        """Show current configuration."""
        self.console.print(
            Panel(
                f"[bold]Provider:[/] {self.settings.default_provider}\n"
                f"[bold]Model:[/] {self.settings.default_model}\n"
                f"[bold]Ollama URL:[/] {self.settings.ollama_base_url}\n"
                f"[bold]Air-gapped:[/] {self.settings.airgapped_mode}\n"
                f"[bold]Semantic Selection:[/] {self.settings.use_semantic_tool_selection}\n"
                f"[bold]Embedding Model:[/] {self.settings.unified_embedding_model}\n"
                f"[bold]Tool Budget:[/] {self.settings.tool_call_budget}\n"
                f"[bold]Config Dir:[/] {self.settings.get_config_dir()}",
                title="Configuration",
                border_style="blue",
            )
        )

    def _cmd_save(self, args: List[str]) -> None:
        """Save current conversation to a session file."""
        if not self.agent:
            self.console.print("[yellow]No active session to save[/]")
            return

        # Get optional title from args
        title = " ".join(args) if args else None

        try:
            session_manager = get_session_manager()
            session_id = session_manager.save_session(
                conversation=self.agent.conversation,
                model=self.agent.model,
                provider=self.agent.provider_name,
                profile=getattr(self.settings, "current_profile", "default"),
                title=title,
                conversation_state=getattr(self.agent, "conversation_state", None),
            )
            self.console.print(
                Panel(
                    f"Session saved successfully!\n\n"
                    f"[bold]Session ID:[/] {session_id}\n"
                    f"[bold]Location:[/] {session_manager.session_dir / f'{session_id}.json'}\n\n"
                    f"[dim]Use '/load {session_id}' to restore this session[/]",
                    title="Session Saved",
                    border_style="green",
                )
            )
        except Exception as e:
            self.console.print(f"[red]Failed to save session:[/] {e}")
            logger.exception("Error saving session")

    def _cmd_load(self, args: List[str]) -> None:
        """Load a saved session."""
        if not args:
            self.console.print("[yellow]Usage:[/] /load <session_id>")
            self.console.print("[dim]Use '/sessions' to list available sessions[/]")
            return

        session_id = args[0]

        try:
            session_manager = get_session_manager()
            session = session_manager.load_session(session_id)

            if session is None:
                self.console.print(f"[red]Session not found:[/] {session_id}")
                return

            if not self.agent:
                self.console.print("[yellow]No active agent to load session into[/]")
                return

            # Restore conversation
            from victor.agent.message_history import MessageHistory
            from victor.agent.conversation_state import ConversationStateMachine

            self.agent.conversation = MessageHistory.from_dict(session.conversation)

            # Restore conversation state machine if available
            if session.conversation_state:
                self.agent.conversation_state = ConversationStateMachine.from_dict(
                    session.conversation_state
                )
                logger.info(
                    f"Restored conversation state: stage={self.agent.conversation_state.get_stage().name}"
                )

            self.console.print(
                Panel(
                    f"Session loaded successfully!\n\n"
                    f"[bold]Title:[/] {session.metadata.title}\n"
                    f"[bold]Model:[/] {session.metadata.model}\n"
                    f"[bold]Provider:[/] {session.metadata.provider}\n"
                    f"[bold]Messages:[/] {session.metadata.message_count}\n"
                    f"[bold]Created:[/] {session.metadata.created_at}",
                    title="Session Loaded",
                    border_style="green",
                )
            )
        except Exception as e:
            self.console.print(f"[red]Failed to load session:[/] {e}")
            logger.exception("Error loading session")

    def _cmd_sessions(self, args: List[str]) -> None:
        """List saved sessions."""
        # Parse limit from args
        limit = 10
        if args:
            try:
                limit = int(args[0])
            except ValueError:
                pass

        try:
            session_manager = get_session_manager()
            sessions = session_manager.list_sessions(limit=limit)

            if not sessions:
                self.console.print("[dim]No saved sessions found[/]")
                self.console.print(f"[dim]Sessions are stored in: {session_manager.session_dir}[/]")
                return

            # Create table
            table = Table(title=f"Saved Sessions (last {len(sessions)})")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Messages", justify="right")
            table.add_column("Updated", style="dim")

            for session in sessions:
                # Format date nicely
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(session.updated_at)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_str = session.updated_at[:16]

                table.add_row(
                    session.session_id,
                    session.title[:40] + "..." if len(session.title) > 40 else session.title,
                    session.model,
                    str(session.message_count),
                    date_str,
                )

            self.console.print(table)
            self.console.print("\n[dim]Use '/load <session_id>' to restore a session[/]")
        except Exception as e:
            self.console.print(f"[red]Failed to list sessions:[/] {e}")
            logger.exception("Error listing sessions")

    async def _cmd_compact(self, args: List[str]) -> None:
        """Compress conversation history to reduce context size.

        Args:
            args: Optional flags:
                --smart: Use AI to generate intelligent summary
                --keep N: Keep last N messages (default: auto)
        """
        if not self.agent:
            self.console.print("[yellow]No active session to compact[/]")
            return

        original_count = self.agent.conversation.message_count()

        if original_count < 5:
            self.console.print("[dim]Conversation is already small enough, nothing to compact[/]")
            return

        # Parse arguments
        use_smart = "--smart" in args or "-s" in args
        keep_recent = 6  # default

        # Parse --keep N
        for i, arg in enumerate(args):
            if arg in ("--keep", "-k") and i + 1 < len(args):
                try:
                    keep_recent = int(args[i + 1])
                except ValueError:
                    pass

        keep_recent = min(keep_recent, len(self.agent.conversation.messages) // 2)
        messages = self.agent.conversation.messages

        if use_smart:
            # AI-powered summarization
            self.console.print("[dim]Generating AI summary...[/]")

            conversation_text = "\n".join(
                [
                    (
                        f"{msg.role}: {msg.content[:200]}..."
                        if len(msg.content) > 200
                        else f"{msg.role}: {msg.content}"
                    )
                    for msg in messages[:20]
                ]
            )

            summary_prompt = f"""Summarize this conversation concisely. Focus on:
1. Main topics discussed
2. Key decisions made
3. Important context to remember

Conversation:
{conversation_text}

Provide a 2-3 sentence summary:"""

            try:
                response = await self.agent.chat(summary_prompt)
                summary = response.content

                self.agent.conversation.clear()
                self.agent.conversation.add_message(
                    "system", f"[AI Summary - {original_count} messages compacted]\n{summary}"
                )

                for msg in messages[-keep_recent:]:
                    self.agent.conversation.add_message(msg.role, msg.content)

                new_count = self.agent.conversation.message_count()
                self.console.print(
                    Panel(
                        f"[bold]AI Summary:[/]\n{summary}\n\n"
                        f"[dim]Reduced from {original_count} to {new_count} messages[/]",
                        title="Smart Compaction Complete",
                        border_style="green",
                    )
                )
                return

            except Exception as e:
                self.console.print(f"[yellow]AI summary failed, using basic compaction: {e}[/]")

        # Basic compaction (no AI)
        old_messages = messages[:-keep_recent] if keep_recent > 0 else messages
        summary_parts = []
        for msg in old_messages[:10]:
            role = msg.role
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"[{role}]: {content}")

        self.agent.conversation.clear()
        self.agent.conversation.add_message(
            "system",
            f"[Compacted - original had {original_count} messages]\n"
            f"Earlier discussion:\n" + "\n".join(summary_parts[:5]),
        )

        for msg in messages[-keep_recent:]:
            self.agent.conversation.add_message(msg.role, msg.content)

        new_count = self.agent.conversation.message_count()
        self.console.print(
            Panel(
                f"Conversation compacted!\n\n"
                f"[bold]Original messages:[/] {original_count}\n"
                f"[bold]After compaction:[/] {new_count}\n"
                f"[bold]Reduced by:[/] {original_count - new_count} messages\n\n"
                f"[dim]Tip: Use '/compact --smart' for AI-powered summarization[/]",
                title="Compacted",
                border_style="green",
            )
        )

    def _cmd_mcp(self, args: List[str]) -> None:
        """Show MCP server status and connections."""
        if not self.agent:
            self.console.print("[yellow]No active agent[/]")
            return

        # Check if MCP client is available
        mcp_client = getattr(self.agent, "mcp_client", None)
        if not mcp_client:
            self.console.print("[dim]No MCP servers configured[/]")
            self.console.print("\n[dim]Configure MCP servers in your settings or .victor.md[/]")
            return

        table = Table(title="MCP Servers")
        table.add_column("Server", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Tools", justify="right")

        # Get server info if available
        if hasattr(mcp_client, "server_info") and mcp_client.server_info:
            info = mcp_client.server_info
            status = "Connected" if mcp_client.initialized else "Disconnected"
            tools_count = len(mcp_client.tools) if hasattr(mcp_client, "tools") else 0
            table.add_row(info.name, status, str(tools_count))
        else:
            table.add_row("Default", "Not connected", "0")

        self.console.print(table)

    async def _cmd_review(self, args: List[str]) -> None:
        """Request a code review for recent changes."""
        if not self.agent:
            self.console.print("[yellow]No active agent for review[/]")
            return

        target = args[0] if args else "."

        # Check for git changes
        import subprocess

        try:
            result = subprocess.run(
                ["git", "diff", "--stat", target],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if not result.stdout.strip():
                self.console.print("[dim]No changes to review[/]")
                return

            # Show what will be reviewed
            self.console.print(
                Panel(
                    f"[bold]Changes to review:[/]\n{result.stdout}",
                    title="Code Review",
                    border_style="blue",
                )
            )

            # Queue review request to the agent
            review_prompt = f"Please review the following code changes and provide feedback on code quality, potential bugs, and improvements:\n\n```diff\n{result.stdout}\n```"
            self.console.print("\n[dim]Sending review request to agent...[/]")

            # Send to agent
            response = await self.agent.chat(review_prompt)
            self.console.print(Markdown(response.content))

        except FileNotFoundError:
            self.console.print("[yellow]Git not found. Cannot review without git.[/]")
        except subprocess.TimeoutExpired:
            self.console.print("[red]Git command timed out[/]")
        except Exception as e:
            self.console.print(f"[red]Error during review:[/] {e}")

    def _cmd_bug(self, args: List[str]) -> None:
        """Report an issue or bug."""
        self.console.print(
            Panel(
                "[bold]Report Issues[/]\n\n"
                "To report a bug or request a feature:\n\n"
                "1. [cyan]GitHub Issues[/]: https://github.com/vijaykumarsingh/victor/issues\n"
                "2. [cyan]Include[/]: Steps to reproduce, expected vs actual behavior\n"
                "3. [cyan]Attach[/]: Error messages, logs, and configuration\n\n"
                "[dim]Use '/status' to get session info for your report[/]",
                title="Report a Bug",
                border_style="yellow",
            )
        )

    def _cmd_exit(self, args: List[str]) -> None:
        """Exit Victor."""
        self.console.print("[dim]Goodbye! Session not saved.[/]")
        self.console.print("[dim]Use '/save' before exiting to preserve your session.[/]")
        raise SystemExit(0)

    def _cmd_undo(self, args: List[str]) -> None:
        """Undo the last file change(s)."""
        from victor.agent.change_tracker import get_change_tracker

        tracker = get_change_tracker()

        if not tracker.can_undo():
            self.console.print("[yellow]Nothing to undo[/]")
            return

        success, message, files = tracker.undo()

        if success:
            self.console.print(f"[green]✓[/] {message}")
            if files:
                for f in files[:5]:  # Show first 5 files
                    self.console.print(f"  [dim]↩ {f}[/]")
                if len(files) > 5:
                    self.console.print(f"  [dim]... and {len(files) - 5} more[/]")
        else:
            self.console.print(f"[red]✗[/] {message}")

    def _cmd_redo(self, args: List[str]) -> None:
        """Redo the last undone change(s)."""
        from victor.agent.change_tracker import get_change_tracker

        tracker = get_change_tracker()

        if not tracker.can_redo():
            self.console.print("[yellow]Nothing to redo[/]")
            return

        success, message, files = tracker.redo()

        if success:
            self.console.print(f"[green]✓[/] {message}")
            if files:
                for f in files[:5]:
                    self.console.print(f"  [dim]↪ {f}[/]")
                if len(files) > 5:
                    self.console.print(f"  [dim]... and {len(files) - 5} more[/]")
        else:
            self.console.print(f"[red]✗[/] {message}")

    def _cmd_history(self, args: List[str]) -> None:
        """Show file change history (timeline)."""
        from victor.agent.change_tracker import get_change_tracker

        tracker = get_change_tracker()
        limit = int(args[0]) if args and args[0].isdigit() else 10

        history = tracker.get_history(limit=limit)

        if not history:
            self.console.print("[dim]No change history[/]")
            return

        table = Table(title="Change History", show_header=True, header_style="bold cyan")
        table.add_column("Time", style="dim")
        table.add_column("Tool", style="cyan")
        table.add_column("Files", justify="right")
        table.add_column("Status")

        for entry in history:
            # Parse timestamp
            time_str = (
                entry["timestamp"].split("T")[1][:8]
                if "T" in entry["timestamp"]
                else entry["timestamp"]
            )
            status = "[yellow]undone[/]" if entry.get("undone") else "[green]applied[/]"
            files = ", ".join(Path(f).name for f in entry.get("files", [])[:3])
            if entry.get("file_count", 0) > 3:
                files += f" +{entry['file_count'] - 3}"

            table.add_row(
                time_str,
                entry.get("tool_name", "unknown"),
                str(entry.get("file_count", 0)),
                status,
            )

        self.console.print(table)
        self.console.print("\n[dim]Use /undo to revert the last change, /redo to reapply[/]")

    def _cmd_mode(self, args: List[str]) -> None:
        """Switch agent mode (build/plan/explore).

        Args:
            args: Optional mode name to switch to
        """
        from victor.agent.mode_controller import AgentMode, get_mode_controller

        manager = get_mode_controller()

        if not args:
            # Show current mode and available modes
            modes = manager.get_mode_list()
            status = manager.get_status()

            self.console.print(
                Panel(
                    f"[bold]Current Mode:[/] {status['name']} ({status['mode']})\n\n"
                    f"[dim]{status['description']}[/]\n\n"
                    "[bold]Available Modes:[/]",
                    title="Agent Mode",
                    border_style="blue",
                )
            )

            table = Table(show_header=True, header_style="bold")
            table.add_column("Mode", style="cyan")
            table.add_column("Description")
            table.add_column("Status")

            for mode_info in modes:
                status_str = "[green]active[/]" if mode_info["current"] else "[dim]available[/]"
                table.add_row(mode_info["mode"], mode_info["description"], status_str)

            self.console.print(table)
            self.console.print("\n[dim]Usage: /mode <mode_name> or /build, /plan, /explore[/]")
            return

        # Switch to specified mode
        mode_name = args[0].lower()

        try:
            new_mode = AgentMode(mode_name)
        except ValueError:
            self.console.print(f"[red]Unknown mode:[/] {mode_name}")
            self.console.print("[dim]Available modes: build, plan, explore[/]")
            return

        if manager.current_mode == new_mode:
            self.console.print(f"[yellow]Already in {mode_name} mode[/]")
            return

        old_mode = manager.current_mode
        manager.switch_mode(new_mode)

        # Show mode switch confirmation
        config = manager.config
        self.console.print(
            Panel(
                f"[bold green]Switched from {old_mode.value} to {new_mode.value} mode[/]\n\n"
                f"[dim]{config.description}[/]\n\n"
                f"[bold]Mode characteristics:[/]\n"
                f"  - Write confirmation: {'required' if config.require_write_confirmation else 'not required'}\n"
                f"  - Tool restrictions: {'read-only tools' if not config.allow_all_tools else 'all tools available'}",
                title=f"Mode: {config.name}",
                border_style="green",
            )
        )

    def _cmd_theme(self, args: List[str]) -> None:
        """Toggle between dark and light theme.

        Note: In CLI mode, this shows theme info.
        In TUI mode, use Ctrl+T for immediate toggle.
        """
        if not args:
            self.console.print(
                Panel(
                    "[bold]Theme Settings[/]\n\n"
                    "In TUI mode, press [bold]Ctrl+T[/] to toggle theme.\n\n"
                    "[bold]Commands:[/]\n"
                    "  /theme dark   - Switch to dark theme\n"
                    "  /theme light  - Switch to light theme\n\n"
                    "[dim]Theme changes are applied immediately in TUI mode.[/]",
                    title="Theme",
                    border_style="blue",
                )
            )
            return

        mode = args[0].lower()
        if mode in ("dark", "light"):
            self.console.print(f"[green]Theme preference set to:[/] {mode}")
            self.console.print("[dim]In TUI mode, use Ctrl+T to toggle.[/]")
        else:
            self.console.print(f"[red]Unknown theme:[/] {mode}")
            self.console.print("[dim]Use 'dark' or 'light'[/]")

    def _cmd_changes(self, args: List[str]) -> None:
        """Unified command for viewing, diffing, or reverting file changes.

        Subcommands:
            show [file]   - Show diff of changes (default)
            revert <file> - Revert specific file to last commit
            stash         - Stash all current changes
            list          - List modified files
        """
        import subprocess

        subcommand = args[0].lower() if args else "show"
        target = args[1] if len(args) > 1 else None

        # Handle aliases being used as command names
        if subcommand in ("diff", "undo", "rollback"):
            subcommand = "show" if subcommand == "diff" else "help"

        try:
            if subcommand == "show":
                # Show diff
                cmd = ["git", "diff", "--stat"]
                if target:
                    cmd.append(target)

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if not result.stdout.strip():
                    self.console.print("[dim]No uncommitted changes[/]")
                    return

                # Get detailed diff if specific file requested
                if target:
                    detail = subprocess.run(
                        ["git", "diff", target],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    diff_content = detail.stdout
                else:
                    diff_content = result.stdout

                self.console.print(
                    Panel(
                        f"```diff\n{diff_content[:2000]}{'...(truncated)' if len(diff_content) > 2000 else ''}\n```",
                        title="File Changes",
                        border_style="cyan",
                    )
                )

            elif subcommand == "list":
                # List modified files
                result = subprocess.run(
                    ["git", "status", "--short"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.stdout.strip():
                    self.console.print(
                        Panel(result.stdout, title="Modified Files", border_style="cyan")
                    )
                else:
                    self.console.print("[dim]No modified files[/]")

            elif subcommand == "revert":
                if not target:
                    self.console.print("[yellow]Usage:[/] /changes revert <file>")
                    self.console.print("[dim]This will discard uncommitted changes to the file[/]")
                    return

                # Confirm before reverting
                self.console.print(f"[yellow]Reverting:[/] {target}")
                self.console.print(
                    "[dim]This will discard all uncommitted changes to this file.[/]"
                )

                result = subprocess.run(
                    ["git", "checkout", "--", target],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    self.console.print(f"[green]✓[/] Reverted {target}")
                else:
                    self.console.print(f"[red]Failed:[/] {result.stderr}")

            elif subcommand == "stash":
                # Stash all changes
                result = subprocess.run(
                    ["git", "stash", "push", "-m", "Victor auto-stash"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if "No local changes" in result.stdout or "No local changes" in result.stderr:
                    self.console.print("[dim]No changes to stash[/]")
                elif result.returncode == 0:
                    self.console.print("[green]✓[/] Changes stashed successfully")
                    self.console.print("[dim]Use 'git stash pop' to restore[/]")
                else:
                    self.console.print(f"[red]Failed:[/] {result.stderr}")

            else:
                # Help
                self.console.print(
                    Panel(
                        "[bold]File Changes Commands[/]\n\n"
                        "[cyan]/changes[/] or [cyan]/changes show[/]\n"
                        "  Show diff of all uncommitted changes\n\n"
                        "[cyan]/changes show <file>[/]\n"
                        "  Show diff for a specific file\n\n"
                        "[cyan]/changes list[/]\n"
                        "  List all modified files\n\n"
                        "[cyan]/changes revert <file>[/]\n"
                        "  Revert a specific file to last commit\n\n"
                        "[cyan]/changes stash[/]\n"
                        "  Stash all current changes\n\n"
                        "[yellow]Tip:[/] Always commit or stash before major changes.",
                        title="Changes Help",
                        border_style="blue",
                    )
                )

        except FileNotFoundError:
            self.console.print("[yellow]Git not found[/]")
        except subprocess.TimeoutExpired:
            self.console.print("[red]Git command timed out[/]")
        except Exception as e:
            self.console.print(f"[red]Error:[/] {e}")

    def _cmd_cost(self, args: List[str]) -> None:
        """Show estimated token usage and cost for this session."""
        if not self.agent:
            self.console.print("[yellow]No active session[/]")
            return

        # Estimate tokens from conversation
        messages = self.agent.conversation.messages
        total_chars = sum(len(msg.content or "") for msg in messages)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token

        # Provider-specific pricing (rough estimates)
        pricing = {
            "anthropic": {"input": 3.0, "output": 15.0},  # per 1M tokens (Claude 3.5 Sonnet)
            "openai": {"input": 2.5, "output": 10.0},  # per 1M tokens (GPT-4o)
            "google": {"input": 1.25, "output": 5.0},  # per 1M tokens (Gemini 1.5)
            "ollama": {"input": 0.0, "output": 0.0},  # Local, free
        }

        provider = self.agent.provider_name.lower()
        price_info = pricing.get(provider, {"input": 0.0, "output": 0.0})

        # Rough split: 30% input, 70% output
        input_tokens = int(estimated_tokens * 0.3)
        output_tokens = int(estimated_tokens * 0.7)

        input_cost = (input_tokens / 1_000_000) * price_info["input"]
        output_cost = (output_tokens / 1_000_000) * price_info["output"]
        total_cost = input_cost + output_cost

        cost_display = f"${total_cost:.4f}" if total_cost > 0 else "Free (local)"

        self.console.print(
            Panel(
                f"[bold]Session Token Usage[/]\n\n"
                f"[bold]Provider:[/] {self.agent.provider_name}\n"
                f"[bold]Model:[/] {self.agent.model}\n"
                f"[bold]Messages:[/] {len(messages)}\n\n"
                f"[bold]Estimated Tokens:[/]\n"
                f"  Input:  ~{input_tokens:,}\n"
                f"  Output: ~{output_tokens:,}\n"
                f"  Total:  ~{estimated_tokens:,}\n\n"
                f"[bold]Estimated Cost:[/] {cost_display}\n\n"
                f"[dim]Note: Token counts are estimates. Actual usage may vary.[/]",
                title="Cost Estimate",
                border_style="blue",
            )
        )

    def _cmd_metrics(self, args: List[str]) -> None:
        """Show streaming performance metrics and provider stats."""
        if not self.agent:
            self.console.print("[yellow]No active session[/]")
            return

        # Check for export flags
        export_json = "--json" in args
        export_csv = "--csv" in args
        subcommand = args[0].lower() if args and not args[0].startswith("--") else "summary"

        collector = self.agent.streaming_metrics_collector
        if not collector:
            self.console.print("[yellow]Streaming metrics not enabled[/]")
            self.console.print("[dim]Enable with streaming_metrics_enabled=true in settings[/]")
            return

        if subcommand == "export" or export_json:
            # Export to JSON
            from pathlib import Path

            output = collector.export_to_json()
            if export_json:
                export_path = Path.cwd() / "metrics_export.json"
                export_path.write_text(output)
                self.console.print(f"[green]Metrics exported to:[/] {export_path}")
            else:
                self.console.print(output)
            return

        if export_csv:
            # Export to CSV
            from pathlib import Path

            output = collector.export_to_csv()
            export_path = Path.cwd() / "metrics_export.csv"
            export_path.write_text(output)
            self.console.print(f"[green]Metrics exported to:[/] {export_path}")
            return

        if subcommand == "history":
            # Show recent metrics history
            recent = collector.get_recent_metrics(count=10)
            if not recent:
                self.console.print("[yellow]No metrics history available yet[/]")
                return

            table = Table(title="Recent Streaming Metrics", show_header=True)
            table.add_column("Provider", style="cyan")
            table.add_column("Model")
            table.add_column("Tokens/s", justify="right")
            table.add_column("TTFT (ms)", justify="right")
            table.add_column("Duration (s)", justify="right")
            table.add_column("Tokens", justify="right")

            for m in recent:
                # Handle both dataclass objects and dicts
                provider = getattr(m, "provider", None) or "?"
                model = getattr(m, "model", None) or "?"
                tps = getattr(m, "tokens_per_second", 0)
                ttft = getattr(m, "time_to_first_token", 0) or 0
                duration = getattr(m, "total_duration", 0)
                tokens = getattr(m, "total_tokens", 0)

                table.add_row(
                    provider,
                    model[:20] if model else "?",
                    f"{tps:.1f}",
                    f"{ttft * 1000:.0f}",
                    f"{duration:.2f}",
                    str(tokens),
                )

            self.console.print(table)
            return

        # Default: show summary
        summary = collector.get_summary()
        # Handle MetricsSummary dataclass
        total_requests = getattr(summary, "total_requests", 0) if summary else 0
        if not summary or total_requests == 0:
            self.console.print("[yellow]No streaming metrics collected yet[/]")
            self.console.print("[dim]Metrics are collected during streaming responses[/]")
            return

        # Build summary display - handle both dataclass and dict
        def get_attr(obj: Any, key: str, default: Any = 0) -> Any:
            if hasattr(obj, key):
                return getattr(obj, key, default)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default

        content = "[bold]Streaming Performance Summary[/]\n\n"
        content += f"[bold]Total Requests:[/] {get_attr(summary, 'total_requests')}\n"
        content += f"[bold]Total Tokens:[/] {get_attr(summary, 'total_tokens'):,}\n\n"

        content += "[bold]Throughput:[/]\n"
        content += f"  Avg:  {get_attr(summary, 'avg_tokens_per_second'):.1f} tokens/s\n"
        content += f"  Max:  {get_attr(summary, 'max_tokens_per_second'):.1f} tokens/s\n"
        content += f"  Min:  {get_attr(summary, 'min_tokens_per_second'):.1f} tokens/s\n\n"

        content += "[bold]Time to First Token (TTFT):[/]\n"
        ttft_ms = (get_attr(summary, "avg_time_to_first_token") or 0) * 1000
        content += f"  Avg:  {ttft_ms:.0f}ms\n\n"

        content += "[bold]Total Duration:[/]\n"
        content += f"  Avg:  {get_attr(summary, 'avg_total_duration'):.2f}s\n"
        content += f"  Max:  {get_attr(summary, 'max_total_duration'):.2f}s\n\n"

        # Provider breakdown if available
        by_provider = get_attr(summary, "by_provider", {})
        if by_provider:
            content += "[bold]By Provider:[/]\n"
            for provider, stats in by_provider.items():
                content += f"  [cyan]{provider}[/]: {get_attr(stats, 'requests')} requests, "
                content += f"{get_attr(stats, 'avg_tokens_per_second'):.1f} tok/s avg\n"

        content += "\n[dim]Use /metrics history for detailed request history[/]"
        content += "\n[dim]Use /metrics --json or --csv to export data[/]"

        self.console.print(Panel(content, title="Streaming Metrics", border_style="green"))

    def _cmd_approvals(self, args: List[str]) -> None:
        """Configure what actions require user approval."""
        from victor.agent.safety import RiskLevel, get_safety_checker

        checker = get_safety_checker()

        if not args:
            # Show current approval level
            threshold = checker.require_confirmation_threshold
            mode_names = {
                RiskLevel.SAFE: "full-auto (no confirmations)",
                RiskLevel.LOW: "auto (confirm medium+ risk)",
                RiskLevel.MEDIUM: "cautious (confirm medium+ risk)",
                RiskLevel.HIGH: "suggest (confirm high+ risk)",
                RiskLevel.CRITICAL: "strict (confirm critical only)",
            }
            current_mode = mode_names.get(threshold, "suggest")

            self.console.print(
                Panel(
                    f"[bold]Current Approval Mode:[/] {current_mode}\n\n"
                    "[bold]Available modes:[/]\n"
                    "  [cyan]suggest[/]   - Confirm HIGH and CRITICAL risk operations (default)\n"
                    "  [cyan]auto[/]      - Confirm only CRITICAL operations\n"
                    "  [cyan]full-auto[/] - No confirmations (YOLO mode - use with caution!)\n\n"
                    "[dim]Usage: /approvals <mode>[/]",
                    title="Approval Settings",
                    border_style="blue",
                )
            )
            return

        mode = args[0].lower()
        mode_map = {
            "suggest": RiskLevel.HIGH,
            "auto": RiskLevel.CRITICAL,
            "full-auto": RiskLevel.SAFE,  # Nothing requires confirmation
            "yolo": RiskLevel.SAFE,
        }

        if mode not in mode_map:
            self.console.print(f"[red]Unknown mode:[/] {mode}")
            self.console.print("[dim]Valid modes: suggest, auto, full-auto[/]")
            return

        checker.require_confirmation_threshold = mode_map[mode]

        if mode in ("full-auto", "yolo"):
            self.console.print(
                Panel(
                    "[bold red]YOLO MODE ENABLED[/bold red]\n\n"
                    "All operations will execute without confirmation.\n"
                    "[yellow]Warning:[/] This bypasses safety checks!\n\n"
                    "Use [cyan]/approvals suggest[/] to restore confirmations.",
                    title="Full Auto Mode",
                    border_style="red",
                )
            )
        else:
            self.console.print(f"[green]Approval mode set to:[/] {mode}")

    def _cmd_resume(self, args: List[str]) -> None:
        """Resume the most recent session."""
        try:
            session_manager = get_session_manager()
            session = session_manager.get_latest_session()

            if session is None:
                self.console.print("[dim]No previous sessions found[/]")
                self.console.print("[dim]Use '/sessions' to see all saved sessions[/]")
                return

            # Load the session
            self._cmd_load([session.metadata.session_id])

        except Exception as e:
            self.console.print(f"[red]Failed to resume:[/] {e}")
            logger.exception("Error resuming session")

    async def _cmd_plan(self, args: List[str]) -> None:
        """Enter planning mode - research before coding."""
        if not self.agent:
            self.console.print("[yellow]No active agent[/]")
            return

        task = " ".join(args) if args else None

        if not task:
            self.console.print(
                Panel(
                    "[bold]Plan Mode[/]\n\n"
                    "Plan mode helps you think through tasks before coding:\n\n"
                    "1. Researches relevant files in your codebase\n"
                    "2. Identifies dependencies and patterns\n"
                    "3. Creates a step-by-step implementation plan\n"
                    "4. Asks clarifying questions if needed\n\n"
                    "[bold]Usage:[/] /plan <describe your task>\n\n"
                    "[bold]Example:[/]\n"
                    "  /plan add user authentication with JWT tokens",
                    title="Planning Mode",
                    border_style="cyan",
                )
            )
            return

        self.console.print(f"[bold cyan]Planning:[/] {task}\n")
        self.console.print("[dim]Analyzing codebase and creating plan...[/]\n")

        plan_prompt = f"""I need to plan the following task. Please:

1. First, analyze what files/modules might be relevant
2. Identify any dependencies or existing patterns to follow
3. List potential challenges or decisions needed
4. Create a step-by-step implementation plan
5. Ask any clarifying questions before we start coding

Task: {task}

Please think through this carefully and provide a detailed plan:"""

        try:
            response = await self.agent.chat(plan_prompt)
            self.console.print(Markdown(response.content))
        except Exception as e:
            self.console.print(f"[red]Error during planning:[/] {e}")

    def _cmd_search(self, args: List[str]) -> None:
        """Toggle web search capability."""
        if not args:
            # Show current state
            web_enabled = not getattr(self.settings, "airgapped_mode", False)
            status = "[green]enabled[/]" if web_enabled else "[red]disabled[/]"

            self.console.print(
                Panel(
                    f"[bold]Web Search:[/] {status}\n\n"
                    "Web search allows Victor to fetch current information\n"
                    "from the internet when answering questions.\n\n"
                    "[bold]Usage:[/]\n"
                    "  /search on   - Enable web search\n"
                    "  /search off  - Disable web search (air-gapped mode)\n\n"
                    f"[dim]Current air-gapped mode: {self.settings.airgapped_mode}[/]",
                    title="Web Search Settings",
                    border_style="blue",
                )
            )
            return

        mode = args[0].lower()
        if mode in ("on", "enable", "true", "1"):
            self.settings.airgapped_mode = False
            self.console.print("[green]Web search enabled[/]")
        elif mode in ("off", "disable", "false", "0"):
            self.settings.airgapped_mode = True
            self.console.print("[yellow]Web search disabled (air-gapped mode)[/]")
        else:
            self.console.print(f"[red]Unknown option:[/] {mode}")
            self.console.print("[dim]Use 'on' or 'off'[/]")

    def _cmd_copy(self, args: List[str]) -> None:
        """Copy last assistant response to clipboard."""
        if not self.agent:
            self.console.print("[yellow]No active session[/]")
            return

        # Find the last assistant message
        messages = self.agent.conversation.messages
        last_assistant = None
        for msg in reversed(messages):
            if msg.role == "assistant":
                last_assistant = msg.content
                break

        if not last_assistant:
            self.console.print("[dim]No assistant response to copy[/]")
            return

        try:
            import subprocess
            import sys

            if sys.platform == "darwin":
                # macOS
                process = subprocess.Popen(
                    ["pbcopy"],
                    stdin=subprocess.PIPE,
                    text=True,
                )
                process.communicate(last_assistant)
            elif sys.platform == "linux":
                # Linux with xclip
                try:
                    process = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE,
                        text=True,
                    )
                    process.communicate(last_assistant)
                except FileNotFoundError:
                    # Try xsel as fallback
                    process = subprocess.Popen(
                        ["xsel", "--clipboard", "--input"],
                        stdin=subprocess.PIPE,
                        text=True,
                    )
                    process.communicate(last_assistant)
            elif sys.platform == "win32":
                # Windows
                process = subprocess.Popen(
                    ["clip"],
                    stdin=subprocess.PIPE,
                    text=True,
                )
                process.communicate(last_assistant)
            else:
                self.console.print(f"[yellow]Clipboard not supported on {sys.platform}[/]")
                return

            # Show preview
            preview = last_assistant[:100] + "..." if len(last_assistant) > 100 else last_assistant
            self.console.print(f"[green]✓[/] Copied to clipboard ({len(last_assistant)} chars)")
            self.console.print(f"[dim]Preview: {preview}[/]")

        except FileNotFoundError:
            self.console.print("[yellow]Clipboard tool not found[/]")
            self.console.print("[dim]Install xclip or xsel on Linux[/]")
        except Exception as e:
            self.console.print(f"[red]Failed to copy:[/] {e}")

    def _cmd_directory(self, args: List[str]) -> None:
        """Show or change working directory."""
        import os

        if not args:
            # Show current directory
            cwd = os.getcwd()
            self.console.print(
                Panel(
                    f"[bold]Current Directory:[/]\n{cwd}\n\n"
                    "[bold]Usage:[/]\n"
                    "  /directory <path>  - Change to directory\n"
                    "  /directory ..      - Go up one level\n"
                    "  /directory ~       - Go to home directory",
                    title="Working Directory",
                    border_style="blue",
                )
            )
            return

        target = args[0]

        # Handle special paths
        if target == "~":
            target = os.path.expanduser("~")
        elif target == "-":
            # Could store previous directory, for now just show help
            self.console.print("[yellow]Previous directory tracking not implemented[/]")
            return

        target = os.path.expanduser(target)
        target = os.path.abspath(target)

        if not os.path.exists(target):
            self.console.print(f"[red]Directory not found:[/] {target}")
            return

        if not os.path.isdir(target):
            self.console.print(f"[red]Not a directory:[/] {target}")
            return

        try:
            os.chdir(target)
            self.console.print(f"[green]✓[/] Changed to: {target}")

            # If agent has project context, suggest reloading
            if self.agent and hasattr(self.agent, "project_context"):
                victor_md = Path(target) / ".victor.md"
                if victor_md.exists():
                    self.console.print("[dim]Found .victor.md - use /context to reload[/]")

        except PermissionError:
            self.console.print(f"[red]Permission denied:[/] {target}")
        except Exception as e:
            self.console.print(f"[red]Error:[/] {e}")

    def _cmd_snapshots(self, args: List[str]) -> None:
        """Manage workspace snapshots for safe rollback.

        Subcommands:
            list              - List recent snapshots (default)
            create [desc]     - Create a new snapshot
            restore <id>      - Restore to a snapshot
            diff <id>         - Show diff between snapshot and current
            clear             - Clear all snapshots
        """
        from victor.agent.snapshot_store import get_snapshot_store

        manager = get_snapshot_store()
        subcommand = args[0].lower() if args else "list"

        if subcommand == "list":
            snapshots = manager.list_snapshots(limit=10)

            if not snapshots:
                self.console.print("[dim]No snapshots found[/]")
                self.console.print("[dim]Snapshots are created automatically before AI edits[/]")
                self.console.print("[dim]Or manually with: /snapshots create [description][/]")
                return

            table = Table(title="Workspace Snapshots")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Time", style="dim")
            table.add_column("Files", justify="right")
            table.add_column("Description")
            table.add_column("Git Ref", style="yellow")

            for snap in snapshots:
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(snap.created_at)
                    time_str = dt.strftime("%H:%M:%S")
                except Exception:
                    time_str = snap.created_at[:8]

                table.add_row(
                    snap.snapshot_id,
                    time_str,
                    str(snap.file_count),
                    (
                        snap.description[:40] + "..."
                        if len(snap.description) > 40
                        else snap.description
                    ),
                    snap.git_ref or "-",
                )

            self.console.print(table)
            self.console.print("\n[dim]Commands: /snapshots restore <id>, /snapshots diff <id>[/]")

        elif subcommand == "create":
            description = " ".join(args[1:]) if len(args) > 1 else "Manual snapshot"

            snapshot_id = manager.create_snapshot(description=description)
            created_snap = manager.get_snapshot(snapshot_id)
            file_count = created_snap.file_count if created_snap else 0

            self.console.print(
                Panel(
                    f"[bold]Snapshot created![/]\n\n"
                    f"[bold]ID:[/] {snapshot_id}\n"
                    f"[bold]Files:[/] {file_count}\n"
                    f"[bold]Description:[/] {description}",
                    title="Snapshot Created",
                    border_style="green",
                )
            )

        elif subcommand == "restore":
            if len(args) < 2:
                self.console.print("[yellow]Usage:[/] /snapshots restore <snapshot_id>")
                return

            snapshot_id = args[1]
            restore_snap = manager.get_snapshot(snapshot_id)

            if not restore_snap:
                self.console.print(f"[red]Snapshot not found:[/] {snapshot_id}")
                return

            self.console.print(f"[yellow]Restoring snapshot:[/] {snapshot_id}")
            self.console.print(f"[dim]This will restore {restore_snap.file_count} files[/]")

            if manager.restore_snapshot(snapshot_id):
                self.console.print(f"[green]✓[/] Restored to snapshot {snapshot_id}")
            else:
                self.console.print("[red]Failed to restore snapshot[/]")

        elif subcommand == "diff":
            if len(args) < 2:
                self.console.print("[yellow]Usage:[/] /snapshots diff <snapshot_id>")
                return

            snapshot_id = args[1]
            diffs = manager.diff_snapshot(snapshot_id)

            if not diffs:
                self.console.print("[yellow]Snapshot not found or no files to compare[/]")
                return

            # Group by status
            added = [d for d in diffs if d.status == "added"]
            modified = [d for d in diffs if d.status == "modified"]
            deleted = [d for d in diffs if d.status == "deleted"]
            unchanged = [d for d in diffs if d.status == "unchanged"]

            output = []
            if added:
                output.append(f"[green]Added ({len(added)}):[/]")
                for d in added[:5]:
                    output.append(f"  + {d.path}")
            if modified:
                output.append(f"[yellow]Modified ({len(modified)}):[/]")
                for d in modified[:5]:
                    output.append(f"  M {d.path}")
            if deleted:
                output.append(f"[red]Deleted ({len(deleted)}):[/]")
                for d in deleted[:5]:
                    output.append(f"  - {d.path}")
            if unchanged:
                output.append(f"[dim]Unchanged ({len(unchanged)})[/]")

            self.console.print(
                Panel(
                    "\n".join(output) if output else "No changes",
                    title=f"Diff: {snapshot_id} vs Current",
                    border_style="cyan",
                )
            )

        elif subcommand == "clear":
            count = manager.clear_all()
            self.console.print(f"[green]✓[/] Cleared {count} snapshots")

        else:
            # Show help
            self.console.print(
                Panel(
                    "[bold]Snapshot Commands[/]\n\n"
                    "[cyan]/snapshots[/] or [cyan]/snapshots list[/]\n"
                    "  List recent snapshots\n\n"
                    "[cyan]/snapshots create [description][/]\n"
                    "  Create a manual snapshot\n\n"
                    "[cyan]/snapshots restore <id>[/]\n"
                    "  Restore workspace to snapshot state\n\n"
                    "[cyan]/snapshots diff <id>[/]\n"
                    "  Show what changed since snapshot\n\n"
                    "[cyan]/snapshots clear[/]\n"
                    "  Clear all snapshots",
                    title="Snapshot Help",
                    border_style="blue",
                )
            )

    def _cmd_commit(self, args: List[str]) -> None:
        """Commit current changes with AI-generated message.

        Usage:
            /commit              - Auto-generate message and commit
            /commit <message>    - Commit with custom message
            /commit --undo       - Undo last Victor commit
            /commit --status     - Show commit status
        """
        from victor.agent.auto_commit import get_auto_committer

        committer = get_auto_committer()

        if not committer.is_git_repo():
            self.console.print("[yellow]Not a git repository[/]")
            return

        # Handle flags
        if args and args[0] in ("--undo", "-u"):
            if committer.is_last_commit_by_victor():
                if committer.undo_last_commit(keep_changes=True):
                    self.console.print("[green]✓[/] Undid last Victor commit (changes kept)")
                else:
                    self.console.print("[red]Failed to undo commit[/]")
            else:
                self.console.print("[yellow]Last commit was not made by Victor[/]")
                self.console.print("[dim]Use 'git reset --soft HEAD~1' to undo manually[/]")
            return

        if args and args[0] in ("--status", "-s"):
            info = committer.get_last_commit_info()
            changed = committer.get_changed_files()

            output = []
            if info:
                output.append(f"[bold]Last commit:[/] {info['hash'][:8]}")
                output.append(f"[bold]Message:[/] {info['subject']}")
                output.append(f"[bold]By Victor:[/] {'Yes' if info['is_victor'] else 'No'}")
            if changed:
                output.append(f"\n[bold]Uncommitted changes:[/] {len(changed)} files")
                for f in changed[:5]:
                    output.append(f"  {f}")
                if len(changed) > 5:
                    output.append(f"  ... and {len(changed) - 5} more")
            else:
                output.append("\n[dim]No uncommitted changes[/]")

            self.console.print(Panel("\n".join(output), title="Commit Status", border_style="blue"))
            return

        # Check for changes
        changed_files = committer.get_changed_files()
        if not changed_files:
            self.console.print("[dim]No changes to commit[/]")
            return

        # Get message
        if args:
            message = " ".join(args)
        else:
            # Auto-generate message
            message = f"AI-assisted changes to {len(changed_files)} file(s)"

        # Show what will be committed
        self.console.print(f"[bold]Committing {len(changed_files)} file(s):[/]")
        for f in changed_files[:5]:
            self.console.print(f"  {f}")
        if len(changed_files) > 5:
            self.console.print(f"  ... and {len(changed_files) - 5} more")

        # Commit
        result = committer.commit_changes(
            files=changed_files,
            description=message,
        )

        if result.success:
            file_count = len(result.files_committed) if result.files_committed else 0
            self.console.print(
                Panel(
                    f"[bold]Committed![/]\n\n"
                    f"[bold]Hash:[/] {result.commit_hash}\n"
                    f"[bold]Files:[/] {file_count}\n\n"
                    f"[dim]Use '/commit --undo' to revert[/]",
                    title="Commit Successful",
                    border_style="green",
                )
            )
        else:
            self.console.print(f"[red]Commit failed:[/] {result.error}")

    async def _cmd_reindex(self, args: List[str]) -> None:
        """Reindex codebase for semantic search.

        Subcommands:
            (none)      - Reindex if stale, show status
            --force     - Force full reindex even if up-to-date
            --stats     - Show indexing statistics
        """
        if not self.agent:
            self.console.print("[yellow]No active agent[/]")
            return

        # Get codebase index from agent if available
        codebase_index = getattr(self.agent, "codebase_index", None)

        if not codebase_index:
            # Try to create one if not present
            try:
                from victor.codebase.indexer import CodebaseIndex

                import os

                cwd = os.getcwd()
                self.console.print(f"[dim]No codebase index found. Creating one for {cwd}...[/]")

                codebase_index = CodebaseIndex(
                    root_path=cwd,
                    use_embeddings=False,  # Start without embeddings for speed
                    enable_watcher=True,
                )

                # Attach to agent for future use
                self.agent.codebase_index = codebase_index

            except Exception as e:
                self.console.print(f"[red]Failed to create codebase index:[/] {e}")
                logger.exception("Error creating codebase index")
                return

        # Handle --stats flag
        if args and args[0] in ("--stats", "-s", "stats"):
            stats = codebase_index.get_stats()

            # Format staleness info
            staleness_info = ""
            if stats.get("is_stale"):
                staleness_info = f"\n[yellow]Index is stale[/] ({stats.get('changed_files_count', 0)} files changed)"
            elif stats.get("is_indexed"):
                staleness_info = "\n[green]Index is up to date[/]"
            else:
                staleness_info = "\n[yellow]Not yet indexed[/]"

            # Format watcher status
            watcher_status = ""
            if stats.get("watcher_enabled"):
                watcher_status = (
                    "[green]running[/]"
                    if stats.get("watcher_running")
                    else "[yellow]not started[/]"
                )
            else:
                watcher_status = "[dim]disabled[/]"

            # Format last indexed time
            last_indexed = stats.get("last_indexed")
            if last_indexed:
                from datetime import datetime

                dt = datetime.fromtimestamp(last_indexed)
                last_indexed_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_indexed_str = "Never"

            self.console.print(
                Panel(
                    f"[bold]Codebase Index Statistics[/]\n\n"
                    f"[bold]Files indexed:[/] {stats.get('total_files', 0)}\n"
                    f"[bold]Symbols indexed:[/] {stats.get('total_symbols', 0)}\n"
                    f"[bold]Total lines:[/] {stats.get('total_lines', 0):,}\n"
                    f"[bold]Embeddings:[/] {'enabled' if stats.get('embeddings_enabled') else 'disabled'}\n"
                    f"[bold]File watcher:[/] {watcher_status}\n"
                    f"[bold]Last indexed:[/] {last_indexed_str}\n"
                    f"{staleness_info}",
                    title="Index Statistics",
                    border_style="blue",
                )
            )
            return

        # Handle --force flag
        force = "--force" in args or "-f" in args or "force" in args

        # Check if reindex is needed
        if not force and not codebase_index.is_stale:
            stats = codebase_index.get_stats()
            self.console.print("[green]Index is up to date[/]")
            self.console.print(
                f"[dim]{stats.get('total_files', 0)} files, "
                f"{stats.get('total_symbols', 0)} symbols indexed[/]"
            )
            self.console.print("[dim]Use '/reindex --force' to force a full reindex[/]")
            return

        # Perform reindex
        self.console.print("[dim]Reindexing codebase...[/]")

        try:
            if force:
                result = await codebase_index.reindex()
            else:
                # Use incremental reindex if few files changed
                if codebase_index.changed_files_count <= 10:
                    result = await codebase_index.incremental_reindex()
                else:
                    result = await codebase_index.reindex()

            if result.get("success"):
                self.console.print(
                    Panel(
                        f"[bold green]Reindex Complete![/]\n\n"
                        f"[bold]Files indexed:[/] {result.get('files_indexed', result.get('files_reindexed', 0))}\n"
                        f"[bold]Symbols indexed:[/] {result.get('symbols_indexed', 'N/A')}\n"
                        f"[bold]Time:[/] {result.get('elapsed_seconds', 0):.2f}s\n"
                        f"[bold]Embeddings:[/] {'enabled' if result.get('embeddings_enabled') else 'disabled'}",
                        title="Reindex Complete",
                        border_style="green",
                    )
                )
            else:
                self.console.print(
                    f"[red]Reindex failed:[/] {result.get('message', 'Unknown error')}"
                )

        except Exception as e:
            self.console.print(f"[red]Reindex failed:[/] {e}")
            logger.exception("Error during reindex")
