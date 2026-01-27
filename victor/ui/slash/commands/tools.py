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

"""Tool-related slash commands: tools, context, lmstudio, mcp, search."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from victor.ui.common.constants import FIRST_ARG_INDEX

from victor.config.settings import VICTOR_CONTEXT_FILE, VICTOR_DIR_NAME
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class ToolsCommand(BaseSlashCommand):
    """List available tools."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="tools",
            description="List available tools",
            usage="/tools [search_pattern]",
            category="tools",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        search = ctx.args[FIRST_ARG_INDEX].lower() if ctx.args else None

        # Get tool objects from the tool registry
        tool_registry = getattr(ctx.agent, "tools", None)
        if tool_registry is None:
            ctx.console.print("[yellow]Tool registry not available[/]")
            return

        tools = tool_registry.list_tools(only_enabled=True)

        if search:
            tools = [
                t
                for t in tools
                if search in getattr(t, "name", "").lower()
                or search in getattr(t, "description", "").lower()
            ]

        if not tools:
            ctx.console.print("[yellow]No matching tools found[/]")
            return

        table = Table(title=f"Available Tools ({len(tools)})", show_header=True)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Cost", style="yellow", justify="right")

        for tool in sorted(tools, key=lambda t: getattr(t, "name", "")):
            name = getattr(tool, "name", "unknown")
            full_desc = getattr(tool, "description", "")
            desc = full_desc[:60]
            if len(full_desc) > 60:
                desc += "..."
            cost_tier = getattr(tool, "cost_tier", None)
            cost = cost_tier.value if cost_tier else "unknown"
            table.add_row(name, desc, cost)

        ctx.console.print(table)
        ctx.console.print(f"\n[dim]Tool budget: {ctx.settings.tool_call_budget}[/]")


@register_command
class ContextCommand(BaseSlashCommand):
    """Show loaded project context (.victor/init.md)."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="context",
            description="Show loaded project context (.victor/init.md)",
            usage="/context",
            aliases=["ctx", "memory"],
            category="tools",
        )

    def execute(self, ctx: CommandContext) -> None:
        from victor.config.settings import get_project_paths

        paths = get_project_paths()
        context_file = paths.project_context_file

        if not context_file.exists():
            ctx.console.print(
                Panel(
                    f"[yellow]No context file found[/]\n\n"
                    f"Create one with: [bold]/init[/]\n\n"
                    f"Expected location: {context_file}",
                    title="Project Context",
                    border_style="yellow",
                )
            )
            return

        content = context_file.read_text(encoding="utf-8")
        preview = content[:2000]
        if len(content) > 2000:
            preview += f"\n\n... ({len(content) - 2000} more characters)"

        ctx.console.print(
            Panel(
                Markdown(preview),
                title=f"Project Context ({context_file.name})",
                border_style="blue",
            )
        )


@register_command
class LMStudioCommand(BaseSlashCommand):
    """Probe LMStudio endpoints and suggest a VRAM-friendly model."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="lmstudio",
            description="Probe LMStudio endpoints and suggest a VRAM-friendly model",
            usage="/lmstudio",
            aliases=["lm"],
            category="tools",
        )

    async def execute(self, ctx: CommandContext) -> None:
        try:
            from victor.providers.lmstudio_provider import LMStudioProvider

            ctx.console.print("[dim]Probing LMStudio endpoints...[/]")

            provider_settings = ctx.settings.get_provider_settings("lmstudio")
            lmstudio = LMStudioProvider(**provider_settings)

            # Check connectivity
            try:
                # Test basic connectivity with a simple request
                response = await lmstudio.client.get("/models", timeout=5.0)
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get("data", [])
                    has_models = len(models) > 0
                else:
                    has_models = False
            except Exception:
                has_models = False
                ctx.console.print(
                    Panel(
                        "[red]LMStudio not detected[/]\n\n"
                        "Make sure LMStudio is running and the server is enabled.\n\n"
                        "Default endpoints:\n"
                        "  - http://localhost:1234\n"
                        "  - http://localhost:8000\n"
                        "  - http://127.0.0.1:1234",
                        title="LMStudio Status",
                        border_style="red",
                    )
                )
                return

            # Use the models we already fetched
            endpoint = lmstudio._raw_base_urls

            content = f"[green]LMStudio Connected[/]\n\n"
            endpoint_str = endpoint[0] if isinstance(endpoint, list) else endpoint
            content += f"[bold]Endpoint:[/] {endpoint_str}\n"
            content += f"[bold]Loaded Models:[/] {len(models)}\n\n"

            if models:
                for model in models[:5]:
                    content += f"  - {model.get('id', 'unknown')}\n"
                if len(models) > 5:
                    content += f"  ... and {len(models) - 5} more\n"
            else:
                content += "[yellow]No models loaded. Load a model in LMStudio.[/]\n"

            content += "\n[dim]Tip: Use /provider lmstudio to switch[/]"

            ctx.console.print(Panel(content, title="LMStudio Status", border_style="green"))

        except ImportError:
            ctx.console.print("[red]LMStudio provider not available[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error probing LMStudio:[/] {e}")


@register_command
class MCPCommand(BaseSlashCommand):
    """Show MCP server status and connections."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="mcp",
            description="Show MCP server status and connections",
            usage="/mcp",
            aliases=["servers"],
            category="tools",
        )

    def execute(self, ctx: CommandContext) -> None:
        try:
            from victor.integrations.mcp.registry import MCPRegistry

            registry = MCPRegistry()
            server_names = registry.list_servers()

            if not server_names:
                ctx.console.print(
                    Panel(
                        "[yellow]No MCP servers configured[/]\n\n"
                        "Configure MCP servers in ~/.victor/mcp_servers.yaml\n"
                        "or use: victor mcp --add <name> <command>",
                        title="MCP Status",
                        border_style="yellow",
                    )
                )
                return

            table = Table(title="MCP Servers", show_header=True)
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Tools", justify="right")

            for server_name in server_names:
                server_info = registry.get_server_status(server_name)
                if server_info:
                    status = "connected" if server_info.get("connected") else "disconnected"
                    tools = server_info.get("tools", [])
                    tool_count = len(tools)
                    table.add_row(
                        server_name,
                        server_info.get("type", "stdio"),
                        status,
                        str(tool_count),
                    )
                else:
                    table.add_row(server_name, "stdio", "disconnected", "0")

            ctx.console.print(table)
            ctx.console.print("\n[dim]Manage servers: victor mcp --help[/]")

        except ImportError:
            ctx.console.print("[red]MCP module not available[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error checking MCP status:[/] {e}")


@register_command
class SearchCommand(BaseSlashCommand):
    """Toggle web search capability."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="search",
            description="Toggle web search capability",
            usage="/search [on|off]",
            aliases=["web"],
            category="tools",
        )

    def execute(self, ctx: CommandContext) -> None:
        current = not ctx.settings.airgapped_mode

        if not ctx.args:
            status = "[green]enabled[/]" if current else "[red]disabled[/]"
            ctx.console.print(f"Web search is currently {status}")
            ctx.console.print("[dim]Toggle: /search on|off[/]")
            return

        action = ctx.args[FIRST_ARG_INDEX].lower()
        if action in ("on", "enable", "1", "true"):
            ctx.settings.airgapped_mode = False
            ctx.console.print("[green]Web search enabled[/]")
        elif action in ("off", "disable", "0", "false"):
            ctx.settings.airgapped_mode = True
            ctx.console.print("[yellow]Web search disabled (air-gapped mode)[/]")
        else:
            ctx.console.print(f"[red]Invalid option:[/] {action}")
            ctx.console.print("[dim]Use: /search on|off[/]")


@register_command
class ReviewCommand(BaseSlashCommand):
    """Request a code review for recent changes."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="review",
            description="Request a code review for recent changes",
            usage="/review [file or directory]",
            category="tools",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        target = ctx.args[FIRST_ARG_INDEX] if ctx.args else "."
        target_path = Path(target).resolve()

        if not target_path.exists():
            ctx.console.print(f"[red]Path not found:[/] {target}")
            return

        ctx.console.print(f"[dim]Reviewing {target_path}...[/]")

        # Build review prompt
        if target_path.is_file():
            prompt = f"Please review the code in {target_path} and provide feedback on:\n"
        else:
            prompt = f"Please review recent changes in {target_path} and provide feedback on:\n"

        prompt += """
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Suggestions for improvement
"""

        try:
            response = await ctx.agent.chat(prompt)  # type: ignore[union-attr]
            ctx.console.print(
                Panel(
                    Markdown(response.content),
                    title=f"Code Review: {target_path.name}",
                    border_style="blue",
                )
            )
        except Exception as e:
            ctx.console.print(f"[red]Review failed:[/] {e}")
