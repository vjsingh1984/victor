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

"""Agent mode slash commands: mode, build, explore, plan."""

from __future__ import annotations

import logging

from rich.panel import Panel

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class ModeCommand(BaseSlashCommand):
    """Switch agent mode (build/plan/explore)."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="mode",
            description="Switch agent mode (build/plan/explore)",
            usage="/mode [build|plan|explore]",
            aliases=["m"],
            category="mode",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.agent.adaptive_mode_controller import AgentMode

        mode_controller = getattr(ctx.agent, "_mode_controller", None)

        if not ctx.args:
            # Show current mode
            current_mode = (
                mode_controller.get_current_mode()
                if mode_controller
                else getattr(ctx.agent, "_current_mode", AgentMode.BUILD)
            )

            ctx.console.print(
                Panel(
                    f"[bold]Current Mode:[/] [cyan]{current_mode.value}[/]\n\n"
                    "[bold]Available Modes:[/]\n"
                    "  [cyan]build[/]   - Implementation mode (default)\n"
                    "  [cyan]plan[/]    - Planning and research mode\n"
                    "  [cyan]explore[/] - Code navigation and analysis mode\n\n"
                    "[dim]Switch with: /mode <mode_name>[/]",
                    title="Agent Mode",
                    border_style="cyan",
                )
            )
            return

        mode_name = ctx.args[0].lower()
        valid_modes = {"build", "plan", "explore"}

        if mode_name not in valid_modes:
            ctx.console.print(f"[red]Invalid mode:[/] {mode_name}")
            ctx.console.print(f"[dim]Available modes: {', '.join(valid_modes)}[/]")
            return

        try:
            new_mode = AgentMode(mode_name)

            if mode_controller:
                mode_controller.switch_mode(new_mode)
            else:
                ctx.agent._current_mode = new_mode

            ctx.console.print(f"[green]Switched to mode:[/] [cyan]{mode_name}[/]")

            # Show mode-specific hints
            hints = {
                "build": "Implementation mode: Focused on writing and modifying code",
                "plan": "Planning mode: Research and design before implementation",
                "explore": "Explore mode: Code navigation and analysis without changes",
            }
            ctx.console.print(f"[dim]{hints.get(mode_name, '')}[/]")

        except Exception as e:
            ctx.console.print(f"[red]Failed to switch mode:[/] {e}")


@register_command
class BuildCommand(BaseSlashCommand):
    """Switch to build mode for implementation."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="build",
            description="Switch to build mode for implementation",
            usage="/build",
            category="mode",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        # Delegate to mode command
        ctx.args = ["build"]
        ModeCommand().execute(ctx)


@register_command
class ExploreCommand(BaseSlashCommand):
    """Switch to explore mode for code navigation."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="explore",
            description="Switch to explore mode for code navigation",
            usage="/explore",
            category="mode",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        # Delegate to mode command
        ctx.args = ["explore"]
        ModeCommand().execute(ctx)


@register_command
class PlanCommand(BaseSlashCommand):
    """Enter planning mode - research before coding."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="plan",
            description="Enter planning mode - research before coding",
            usage="/plan [task description]",
            category="mode",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.agent.adaptive_mode_controller import AgentMode

        # Switch to plan mode
        mode_controller = getattr(ctx.agent, "_mode_controller", None)
        if mode_controller:
            mode_controller.switch_mode(AgentMode.PLAN)
        else:
            ctx.agent._current_mode = AgentMode.PLAN

        ctx.console.print("[green]Switched to planning mode[/]")

        # If task description provided, start planning
        if ctx.args:
            task = " ".join(ctx.args)
            ctx.console.print(f"[dim]Planning task: {task}[/]")

            planning_prompt = (
                f"I need to plan the following task before implementation:\n\n"
                f"{task}\n\n"
                f"Please analyze this task and provide:\n"
                f"1. Understanding of requirements\n"
                f"2. Relevant files and code sections to examine\n"
                f"3. Proposed approach and implementation steps\n"
                f"4. Potential challenges and considerations\n"
            )

            try:
                response = await ctx.agent.chat(planning_prompt)
                from rich.markdown import Markdown

                ctx.console.print(
                    Panel(
                        Markdown(response.content),
                        title="Planning Analysis",
                        border_style="blue",
                    )
                )
            except Exception as e:
                ctx.console.print(f"[red]Planning failed:[/] {e}")
        else:
            ctx.console.print(
                "[dim]Tip: Use /plan <task> to start planning a specific task[/]"
            )
