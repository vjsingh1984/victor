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

"""Checkpoint management slash commands for time-travel debugging.

Commands:
- /checkpoint save: Create a manual checkpoint with optional description
- /checkpoint list: List checkpoints for current session
- /checkpoint restore: Restore to a previous checkpoint
- /checkpoint diff: Compare two checkpoints
- /checkpoint timeline: Show visual timeline of checkpoints
"""

from __future__ import annotations

import asyncio
import logging

from rich.panel import Panel

from victor.ui.common.constants import FIRST_ARG_INDEX, SECOND_ARG_INDEX
from rich.table import Table
from rich.text import Text

from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class CheckpointCommand(BaseSlashCommand):
    """Manage conversation state checkpoints for time-travel debugging."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="checkpoint",
            description="Manage conversation checkpoints for time-travel debugging",
            usage="/checkpoint <save|list|restore|diff|timeline> [args]",
            category="checkpoint",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        if not ctx.args:
            self._show_help(ctx)
            return

        subcommand = ctx.args[FIRST_ARG_INDEX].lower()
        subargs = ctx.args[1:]

        if subcommand == "save":
            self._handle_save(ctx, subargs)
        elif subcommand == "list":
            self._handle_list(ctx, subargs)
        elif subcommand == "restore":
            self._handle_restore(ctx, subargs)
        elif subcommand == "diff":
            self._handle_diff(ctx, subargs)
        elif subcommand == "timeline":
            self._handle_timeline(ctx, subargs)
        else:
            ctx.console.print(f"[yellow]Unknown subcommand:[/] {subcommand}")
            self._show_help(ctx)

    def _show_help(self, ctx: CommandContext) -> None:
        """Show checkpoint command help."""
        help_text = """
[bold]Checkpoint Commands[/]

[cyan]/checkpoint save[/] [description]
  Create a manual checkpoint with optional description
  Example: /checkpoint save before refactoring auth module

[cyan]/checkpoint list[/] [limit]
  List checkpoints for current session (default limit: 10)
  Example: /checkpoint list 20

[cyan]/checkpoint restore[/] <checkpoint_id>
  Restore conversation state to a previous checkpoint
  Example: /checkpoint restore ckpt_abc123def456

[cyan]/checkpoint diff[/] <checkpoint_a> <checkpoint_b>
  Compare two checkpoints to see differences
  Example: /checkpoint diff ckpt_abc123 ckpt_def456

[cyan]/checkpoint timeline[/]
  Show ASCII timeline of all checkpoints
"""
        ctx.console.print(Panel(help_text, title="Checkpoint Help", border_style="blue"))

    def _handle_save(self, ctx: CommandContext, args: list[str]) -> None:
        """Handle checkpoint save subcommand."""
        if not ctx.agent or not ctx.agent.checkpoint_manager:  # type: ignore[attr-defined]
            ctx.console.print(
                "[yellow]Checkpoint system not enabled.[/] "
                "Set checkpoint_enabled=True in settings."
            )
            return

        description = " ".join(args) if args else None

        try:
            # Run async method in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task if we're already in an async context
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        ctx.agent.save_checkpoint(description=description),  # type: ignore[attr-defined]
                    )
                    checkpoint_id = future.result(timeout=10)
            else:
                checkpoint_id = loop.run_until_complete(
                    ctx.agent.save_checkpoint(description=description)  # type: ignore[attr-defined]
                )

            if checkpoint_id:
                ctx.console.print(
                    Panel(
                        f"Checkpoint saved successfully!\n\n"
                        f"[bold]ID:[/] {checkpoint_id}\n"
                        f"[bold]Description:[/] {description or '(none)'}\n\n"
                        f"[dim]Use '/checkpoint restore {checkpoint_id}' to restore[/]",
                        title="Checkpoint Saved",
                        border_style="green",
                    )
                )
            else:
                ctx.console.print("[red]Failed to save checkpoint[/]")

        except Exception as e:
            ctx.console.print(f"[red]Error saving checkpoint:[/] {e}")
            logger.exception("Checkpoint save error")

    def _handle_list(self, ctx: CommandContext, args: list[str]) -> None:
        """Handle checkpoint list subcommand."""
        if not ctx.agent or not ctx.agent.checkpoint_manager:  # type: ignore[attr-defined]
            ctx.console.print(
                "[yellow]Checkpoint system not enabled.[/] "
                "Set checkpoint_enabled=True in settings."
            )
            return

        limit = 10
        if args:
            try:
                limit = int(args[FIRST_ARG_INDEX])
            except ValueError:
                ctx.console.print(f"[yellow]Invalid limit:[/] {args[FIRST_ARG_INDEX]}")
                return

        try:
            # Get session ID from agent
            session_id = getattr(ctx.agent, "_memory_session_id", None) or "default"

            # Run async method
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        ctx.agent.checkpoint_manager.list_checkpoints(session_id, limit=limit),  # type: ignore[attr-defined]
                    )
                    checkpoints = future.result(timeout=10)
            else:
                checkpoints = loop.run_until_complete(
                    ctx.agent.checkpoint_manager.list_checkpoints(session_id, limit=limit)  # type: ignore[attr-defined]
                )

            if not checkpoints:
                ctx.console.print("[dim]No checkpoints found for this session.[/]")
                return

            table = Table(title=f"Checkpoints (showing {len(checkpoints)})")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Timestamp", style="dim")
            table.add_column("Stage", style="green")
            table.add_column("Tools", justify="right")
            table.add_column("Messages", justify="right")
            table.add_column("Description")

            for cp in checkpoints:
                table.add_row(
                    cp.checkpoint_id[:16] + "...",
                    cp.timestamp.strftime("%Y-%m-%d %H:%M"),
                    cp.stage,
                    str(cp.tool_count),
                    str(cp.message_count),
                    (cp.description or "")[:30],
                )

            ctx.console.print(table)

        except Exception as e:
            ctx.console.print(f"[red]Error listing checkpoints:[/] {e}")
            logger.exception("Checkpoint list error")

    def _handle_restore(self, ctx: CommandContext, args: list[str]) -> None:
        """Handle checkpoint restore subcommand."""
        if not args:
            ctx.console.print("[yellow]Usage:[/] /checkpoint restore <checkpoint_id>")
            ctx.console.print("[dim]Use '/checkpoint list' to see available checkpoints[/]")
            return

        if not ctx.agent or not ctx.agent.checkpoint_manager:  # type: ignore[attr-defined]
            ctx.console.print(
                "[yellow]Checkpoint system not enabled.[/] "
                "Set checkpoint_enabled=True in settings."
            )
            return

        checkpoint_id = args[FIRST_ARG_INDEX]

        try:
            # Run async method
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        ctx.agent.restore_checkpoint(checkpoint_id),  # type: ignore[attr-defined]
                    )
                    success = future.result(timeout=10)
            else:
                success = loop.run_until_complete(ctx.agent.restore_checkpoint(checkpoint_id))  # type: ignore[attr-defined]

            if success:
                ctx.console.print(
                    Panel(
                        f"Checkpoint restored successfully!\n\n"
                        f"[bold]ID:[/] {checkpoint_id}\n\n"
                        f"[dim]Conversation state has been rolled back.[/]",
                        title="Checkpoint Restored",
                        border_style="green",
                    )
                )
            else:
                ctx.console.print(f"[red]Failed to restore checkpoint:[/] {checkpoint_id}")

        except Exception as e:
            ctx.console.print(f"[red]Error restoring checkpoint:[/] {e}")
            logger.exception("Checkpoint restore error")

    def _handle_diff(self, ctx: CommandContext, args: list[str]) -> None:
        """Handle checkpoint diff subcommand."""
        if len(args) < 2:
            ctx.console.print("[yellow]Usage:[/] /checkpoint diff <checkpoint_a> <checkpoint_b>")
            return

        if not ctx.agent or not ctx.agent.checkpoint_manager:  # type: ignore[attr-defined]
            ctx.console.print(
                "[yellow]Checkpoint system not enabled.[/] "
                "Set checkpoint_enabled=True in settings."
            )
            return

        checkpoint_a, checkpoint_b = args[FIRST_ARG_INDEX], args[SECOND_ARG_INDEX]

        try:
            # Run async method
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        ctx.agent.checkpoint_manager.diff_checkpoints(checkpoint_a, checkpoint_b),  # type: ignore[attr-defined]
                    )
                    diff = future.result(timeout=10)
            else:
                diff = loop.run_until_complete(
                    ctx.agent.checkpoint_manager.diff_checkpoints(checkpoint_a, checkpoint_b)  # type: ignore[attr-defined]
                )

            # Display diff summary
            ctx.console.print(
                Panel(
                    diff.summary(),
                    title="Checkpoint Diff",
                    border_style="blue",
                )
            )

        except Exception as e:
            ctx.console.print(f"[red]Error comparing checkpoints:[/] {e}")
            logger.exception("Checkpoint diff error")

    def _handle_timeline(self, ctx: CommandContext, args: list[str]) -> None:
        """Handle checkpoint timeline subcommand."""
        if not ctx.agent or not ctx.agent.checkpoint_manager:  # type: ignore[attr-defined]
            ctx.console.print(
                "[yellow]Checkpoint system not enabled.[/] "
                "Set checkpoint_enabled=True in settings."
            )
            return

        try:
            # Get session ID from agent
            session_id = getattr(ctx.agent, "_memory_session_id", None) or "default"

            # Run async method
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        ctx.agent.checkpoint_manager.get_timeline(session_id),  # type: ignore[attr-defined]
                    )
                    timeline = future.result(timeout=10)
            else:
                timeline = loop.run_until_complete(
                    ctx.agent.checkpoint_manager.get_timeline(session_id)  # type: ignore[attr-defined]
                )

            if not timeline:
                ctx.console.print("[dim]No checkpoints found for timeline.[/]")
                return

            # Format timeline as ASCII art
            ascii_timeline = ctx.agent.checkpoint_manager.format_timeline_ascii(timeline)  # type: ignore[attr-defined]
            ctx.console.print(
                Panel(ascii_timeline, title="Checkpoint Timeline", border_style="blue")
            )

        except Exception as e:
            ctx.console.print(f"[red]Error generating timeline:[/] {e}")
            logger.exception("Checkpoint timeline error")
