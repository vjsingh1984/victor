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

"""System slash commands: help, config, status, exit, clear, theme."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from victor.ui.common.constants import FIRST_ARG_INDEX
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import get_command_registry, register_command


@register_command
class HelpCommand(BaseSlashCommand):
    """Show available commands or help for a specific command."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="help",
            description="Show available commands",
            usage="/help [command]",
            aliases=["?", "commands"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        registry = get_command_registry()

        if ctx.args:
            # Help for specific command
            cmd_name = ctx.args[FIRST_ARG_INDEX].lstrip("/")
            command = registry.get(cmd_name)
            if command:
                meta = command.metadata
                aliases = ", ".join(f"/{a}" for a in meta.aliases) if meta.aliases else "none"
                ctx.console.print(
                    Panel(
                        f"[bold]/{meta.name}[/]\n\n"
                        f"{meta.description}\n\n"
                        f"[dim]Usage:[/] {meta.usage}\n"
                        f"[dim]Aliases:[/] {aliases}",
                        title=f"Help: /{meta.name}",
                        border_style="blue",
                    )
                )
            else:
                ctx.console.print(f"[yellow]Unknown command:[/] /{cmd_name}")
            return

        # Show all commands grouped by category
        table = Table(title="Available Commands", show_header=True)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")
        table.add_column("Aliases", style="dim")

        for name, meta in registry.list_commands():
            aliases = ", ".join(f"/{a}" for a in meta.aliases) if meta.aliases else ""
            table.add_row(f"/{name}", meta.description, aliases)

        ctx.console.print(table)
        ctx.console.print("\n[dim]Type /help <command> for more details[/]")


@register_command
class ConfigCommand(BaseSlashCommand):
    """Show current configuration."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="config",
            description="Show current configuration",
            usage="/config",
            aliases=["settings"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        ctx.console.print(
            Panel(
                f"[bold]Provider:[/] {ctx.settings.default_provider}\n"
                f"[bold]Model:[/] {ctx.settings.default_model}\n"
                f"[bold]Ollama URL:[/] {ctx.settings.ollama_base_url}\n"
                f"[bold]Air-gapped:[/] {ctx.settings.airgapped_mode}\n"
                f"[bold]Semantic Selection:[/] {ctx.settings.use_semantic_tool_selection}\n"
                f"[bold]Embedding Model:[/] {ctx.settings.unified_embedding_model}\n"
                f"[bold]Tool Budget:[/] {ctx.settings.tool_call_budget}\n"
                f"[bold]Config Dir:[/] {ctx.settings.get_config_dir()}",
                title="Configuration",
                border_style="blue",
            )
        )


@register_command
class StatusCommand(BaseSlashCommand):
    """Show current session status."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="status",
            description="Show current session status",
            usage="/status",
            aliases=["info"],
            category="system",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        agent = ctx.agent
        history = agent.conversation
        tool_calls = getattr(agent, "_tool_calls", 0)
        tool_budget = ctx.settings.tool_call_budget

        content = (
            f"[bold]Provider:[/] {agent.provider_name}\n"
            f"[bold]Model:[/] {agent.model}\n"
            f"[bold]Messages:[/] {history.message_count()}\n"
            f"[bold]Tool Calls:[/] {tool_calls} / {tool_budget}\n"
        )

        # Add conversation state if available
        state_machine = getattr(agent, "conversation_state", None)
        if state_machine:
            stage = state_machine.get_stage()
            content += f"[bold]Stage:[/] {stage.name}\n"

        # Add mode controller info
        mode_controller = getattr(agent, "_mode_controller", None)
        if mode_controller:
            current_mode = mode_controller.get_current_mode()
            content += f"[bold]Mode:[/] {current_mode.value}\n"

        # Add RL recommendation if available
        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("model_selector")
            if learner:
                rec = learner.recommend(agent.provider_name, "coding")
                if rec and rec.confidence > 0:
                    content += f"\n[dim]RL: Using optimal provider (Q={rec.confidence:.2f})[/]"
        except Exception:
            pass

        ctx.console.print(Panel(content, title="Session Status", border_style="blue"))


@register_command
class ClearCommand(BaseSlashCommand):
    """Clear conversation history."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="clear",
            description="Clear conversation history",
            usage="/clear",
            aliases=["reset"],
            category="system",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        ctx.agent.reset_conversation()
        ctx.console.print("[green]Conversation cleared[/]")


@register_command
class ExitCommand(BaseSlashCommand):
    """Exit Victor."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="exit",
            description="Exit Victor",
            usage="/exit",
            aliases=["quit", "bye"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        ctx.console.print("[dim]Goodbye![/]")
        raise SystemExit(0)


@register_command
class ThemeCommand(BaseSlashCommand):
    """Toggle between dark and light theme."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="theme",
            description="Toggle between dark and light theme",
            usage="/theme [dark|light]",
            aliases=["dark", "light"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        current_theme = getattr(ctx.settings, "theme", "dark")

        if ctx.args:
            new_theme = ctx.args[FIRST_ARG_INDEX].lower()
            if new_theme not in ("dark", "light"):
                ctx.console.print(f"[red]Invalid theme:[/] {new_theme}")
                ctx.console.print("[dim]Available: dark, light[/]")
                return
        else:
            # Toggle
            new_theme = "light" if current_theme == "dark" else "dark"

        ctx.settings.theme = new_theme
        ctx.console.print(f"[green]Theme set to:[/] {new_theme}")


@register_command
class BugCommand(BaseSlashCommand):
    """Report an issue or bug."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="bug",
            description="Report an issue or bug",
            usage="/bug",
            aliases=["issue", "feedback"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        ctx.console.print(
            Panel(
                "[bold]Report Issues[/]\n\n"
                "For bugs or feature requests, please visit:\n\n"
                "[link=https://github.com/vijayksingh/victor/issues]"
                "https://github.com/vijayksingh/victor/issues[/link]\n\n"
                "When reporting, please include:\n"
                "- Victor version ([cyan]victor --version[/])\n"
                "- Provider and model being used\n"
                "- Steps to reproduce the issue\n"
                "- Any error messages",
                title="Report an Issue",
                border_style="yellow",
            )
        )


@register_command
class ApprovalsCommand(BaseSlashCommand):
    """Configure what actions require user approval."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="approvals",
            description="Configure what actions require user approval",
            usage="/approvals [suggest|auto|full-auto]",
            aliases=["safety"],
            category="system",
        )

    def execute(self, ctx: CommandContext) -> None:
        current_mode = getattr(ctx.settings, "approval_mode", "suggest")

        if not ctx.args:
            # Show current mode
            ctx.console.print(
                Panel(
                    f"[bold]Current Mode:[/] {current_mode}\n\n"
                    "[bold]Available Modes:[/]\n"
                    "  [cyan]suggest[/]   - Ask before any write operation (safest)\n"
                    "  [cyan]auto[/]      - Auto-approve safe operations, ask for others\n"
                    "  [cyan]full-auto[/] - Auto-approve everything (use with caution)",
                    title="Approval Settings",
                    border_style="yellow",
                )
            )
            return

        mode = ctx.args[FIRST_ARG_INDEX].lower()
        if mode not in ("suggest", "auto", "full-auto"):
            ctx.console.print(f"[red]Invalid mode:[/] {mode}")
            ctx.console.print("[dim]Available: suggest, auto, full-auto[/]")
            return

        ctx.settings.approval_mode = mode
        ctx.console.print(f"[green]Approval mode set to:[/] {mode}")

        if mode == "full-auto":
            ctx.console.print(
                "[yellow]Warning: full-auto mode will auto-approve all actions. "
                "Use with caution![/]"
            )
