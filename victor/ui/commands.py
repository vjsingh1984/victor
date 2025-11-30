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
                usage="/init [--force] [--simple]",
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
                aliases=["ctx"],
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

    def _cmd_init(self, args: List[str]) -> None:
        """Initialize .victor.md file with smart codebase analysis."""
        force = "--force" in args or "-f" in args
        simple = "--simple" in args or "-s" in args
        target_path = Path.cwd() / ".victor.md"

        if target_path.exists() and not force:
            self.console.print(f"[yellow].victor.md already exists at {target_path}[/]")
            self.console.print("Use [bold]/init --force[/] to overwrite")
            self.console.print("Use [bold]/init --simple[/] for basic template")

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

        self.console.print("[dim]Analyzing codebase...[/]")

        try:
            if simple:
                # Use simple generator
                from victor.context.project_context import generate_victor_md

                content = generate_victor_md()
            else:
                # Use smart analyzer
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
            from victor.providers.ollama import OllamaProvider

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
                self.console.print(f"[yellow]Profile switching requires restart[/]")
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
