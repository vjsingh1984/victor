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

"""Switch command for changing models/providers with optional session resume."""

from __future__ import annotations

import logging

from rich.panel import Panel
from rich.table import Table

from victor.ui.common.constants import FIRST_ARG_INDEX, MODEL_PART_INDEX, PROVIDER_PART_INDEX
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class SwitchCommand(BaseSlashCommand):
    """Switch model/provider and optionally resume a session.

    Usage:
        /switch <model_name>                    # Switch model only
        /switch <provider>:<model>             # Switch provider and model
        /switch --resume [session_id]           # Resume session first, then switch
        /switch <model> --resume <session_id>   # Switch and resume in one command

    Examples:
        /switch claude-sonnet-4-20250514
        /switch anthropic:claude-opus-4-20250514
        /switch --resume                        # Resume last session, then switch
        /switch claude-sonnet-4-20250514 --resume  # Resume last, then switch to this model
        /switch ollama:qwen2.5-coder:7b --resume 20250107_153045
    """

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="switch",
            description="Switch model/provider with optional session resume",
            usage="/switch <model>|<provider:model> [--resume [session_id]]",
            category="model",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        if not ctx.args:
            # Show current model/provider
            self._show_current(ctx)
            return

        # Parse arguments
        args = list(ctx.args)
        resume_session_id = None
        resume_index = None

        # Check for --resume flag
        if "--resume" in args:
            resume_idx = args.index("--resume")
            args.pop(resume_idx)  # Remove --resume

            # Check if session_id provided after --resume
            if resume_idx < len(args):
                session_id = args[resume_idx]
                if session_id.isdigit():
                    resume_index = int(session_id)
                else:
                    resume_session_id = session_id
                args.pop(resume_idx)  # Remove session_id

        # If no more args and --resume was used, show session selection
        if not args and resume_session_id is None:
            self._show_resume_selection(ctx)
            return

        # Parse model/provider argument
        target = args[FIRST_ARG_INDEX]
        provider = None
        model = None

        if ":" in target:
            # Format: provider:model
            parts = target.split(":", 1)
            provider = parts[PROVIDER_PART_INDEX]
            model = parts[MODEL_PART_INDEX]
        else:
            # Just model name - use current provider
            model = target

        # Resume session if requested
        if resume_session_id or resume_index is not None:
            if not self._resume_session(ctx, resume_session_id or "", resume_index or 1):
                # Resume failed, don't switch
                return

        # Perform the switch
        if provider and model:
            # Switch both provider and model
            self._switch_provider(ctx, provider, model)
        elif model:
            # Switch just model
            self._switch_model(ctx, model)

    def _show_current(self, ctx: CommandContext) -> None:
        """Show current model/provider information.

        Args:
            ctx: Command context
        """
        try:
            info = ctx.agent.get_current_provider_info()  # type: ignore[union-attr]

            ctx.console.print(
                Panel(
                    f"[bold]Provider:[/] {info.get('provider', 'N/A')}\n"
                    f"[bold]Model:[/] {info.get('model', 'N/A')}\n"
                    f"[bold]Native Tools:[/] {info.get('native_tool_calls', 'N/A')}\n"
                    f"[bold]Thinking Mode:[/] {info.get('thinking_mode', 'N/A')}\n\n"
                    f"[dim]Switch models:[/]\n"
                    f"[dim]/switch <model>[/]\n"
                    f"[dim]/switch <provider>:<model>[/]\n"
                    f"[dim]/switch --resume <session_id>[/]",
                    title="Current Model/Provider",
                    border_style="cyan",
                )
            )
        except Exception as e:
            ctx.console.print(f"[red]Error getting provider info:[/] {e}")

    def _show_resume_selection(self, ctx: CommandContext) -> None:
        """Show session selection for resume.

        Args:
            ctx: Command context
        """
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

        try:
            persistence = get_sqlite_session_persistence()
            sessions = persistence.list_sessions(limit=20)

            if not sessions:
                ctx.console.print("[yellow]No sessions to resume[/]")
                return

            table = Table(title="Select Session to Resume")
            table.add_column("#", style="cyan", no_wrap=True, width=4)
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Messages", justify="right")

            for idx, session in enumerate(sessions, 1):
                title = (
                    session["title"][:30] + "..."
                    if len(session["title"]) > 30
                    else session["title"]
                )
                table.add_row(
                    str(idx),
                    session["session_id"],
                    title,
                    session["model"],
                    str(session["message_count"]),
                )

            ctx.console.print(table)
            ctx.console.print("\n[dim]Usage:[/]")
            ctx.console.print("[dim]/switch <model> --resume <number>[/]")

        except Exception as e:
            ctx.console.print(f"[red]Error listing sessions:[/] {e}")
            logger.exception("Error listing sessions")

    def _resume_session(self, ctx: CommandContext, session_id: str = "", index: int = 1) -> bool:
        """Resume a session.

        Args:
            ctx: Command context
            session_id: Optional session ID
            index: Optional session index (1-based)

        Returns:
            True if resume succeeded
        """
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

        try:
            persistence = get_sqlite_session_persistence()

            # Get session ID from index if needed
            if index is not None:
                sessions = persistence.list_sessions(limit=20)
                if not sessions or index > len(sessions):
                    ctx.console.print(f"[red]Invalid session index:[/] {index}")
                    return False
                session_id = sessions[index - 1]["session_id"]

            # Load and restore session
            session_data = persistence.load_session(session_id)
            if not session_data:
                ctx.console.print(f"[red]Session not found:[/] {session_id}")
                return False

            # Restore conversation
            from victor.agent.message_history import MessageHistory
            from victor.agent.conversation_state import ConversationStateMachine

            metadata = session_data.get("metadata", {})
            conversation_dict = session_data.get("conversation", {})

            ctx.agent.conversation = MessageHistory.from_dict(conversation_dict)  # type: ignore[union-attr]

            # Set active session ID for parallel session support
            ctx.agent.active_session_id = session_id  # type: ignore[union-attr]

            conversation_state_dict = session_data.get("conversation_state")
            if conversation_state_dict and ctx.agent:
                try:
                    ctx.agent.conversation_state = ConversationStateMachine.from_dict(
                        conversation_state_dict
                    )
                except Exception as e:
                    logger.warning(f"Failed to restore conversation state: {e}")

            ctx.console.print(
                f"[green]✓[/] Resumed: {metadata.get('title', 'Untitled')} "
                f"({metadata.get('message_count', 0)} messages)\n"
            )
            return True

        except Exception as e:
            ctx.console.print(f"[red]Failed to resume session:[/] {e}")
            logger.exception("Error resuming session")
            return False

    def _switch_model(self, ctx: CommandContext, model: str) -> None:
        """Switch to a different model.

        Args:
            ctx: Command context
            model: Model name
        """
        try:
            if ctx.agent and hasattr(ctx.agent, 'switch_model') and ctx.agent.switch_model(model):
                # Check if agent has get_current_provider_info method
                if hasattr(ctx.agent, 'get_current_provider_info'):
                    info = ctx.agent.get_current_provider_info()
                else:
                    info = {'native_tool_calls': 'N/A', 'thinking_mode': 'N/A'}
                ctx.console.print(
                    f"[green]✓[/] Switched to [cyan]{model}[/]\n"
                    f"  [dim]Native tools: {info.get('native_tool_calls', 'N/A')}, "
                    f"Thinking: {info.get('thinking_mode', 'N/A')}[/]"
                )
            else:
                ctx.console.print(f"[red]Failed to switch model to {model}[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error switching model:[/] {e}")
            logger.exception("Error switching model")

    def _switch_provider(self, ctx: CommandContext, provider: str, model: str) -> None:
        """Switch to a different provider and model.

        Args:
            ctx: Command context
            provider: Provider name
            model: Model name
        """
        try:
            if not ctx.agent:
                ctx.console.print("[red]No active agent[/]")
                return

            if hasattr(ctx.agent, 'switch_provider') and ctx.agent.switch_provider(provider_name=provider, model=model):
                # Check if agent has get_current_provider_info method
                if hasattr(ctx.agent, 'get_current_provider_info'):
                    info = ctx.agent.get_current_provider_info()
                else:
                    info = {'native_tool_calls': 'N/A', 'thinking_mode': 'N/A'}
                ctx.console.print(
                    f"[green]✓[/] Switched to [cyan]{provider}:{model}[/]\n"
                    f"  [dim]Native tools: {info.get('native_tool_calls', 'N/A')}, "
                    f"Thinking: {info.get('thinking_mode', 'N/A')}[/]"
                )
            else:
                ctx.console.print(f"[red]Failed to switch to {provider}:{model}[/]")
        except Exception as e:
            ctx.console.print(f"[red]Error switching provider:[/] {e}")
            logger.exception("Error switching provider")


__all__ = ["SwitchCommand"]
