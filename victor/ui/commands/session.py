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

"""Session management commands.

Commands for saving, loading, and managing conversation sessions.
"""

from __future__ import annotations

import logging
from typing import List

from rich.table import Table

from victor.ui.commands.base import (
    CommandContext,
    CommandGroup,
    SlashCommand,
)

logger = logging.getLogger(__name__)


class SessionCommands(CommandGroup):
    """Session management commands."""

    @property
    def group_name(self) -> str:
        return "session"

    @property
    def group_description(self) -> str:
        return "Manage conversation sessions"

    def get_commands(self) -> List[SlashCommand]:
        return [
            SlashCommand(
                name="save",
                description="Save current session",
                handler=self._cmd_save,
                usage="/save [name]",
                group=self.group_name,
            ),
            SlashCommand(
                name="load",
                description="Load a saved session",
                handler=self._cmd_load,
                usage="/load <session_id>",
                group=self.group_name,
            ),
            SlashCommand(
                name="sessions",
                description="List saved sessions",
                handler=self._cmd_sessions,
                aliases=["ls"],
                group=self.group_name,
            ),
            SlashCommand(
                name="resume",
                description="Resume last session",
                handler=self._cmd_resume,
                group=self.group_name,
            ),
            SlashCommand(
                name="clear",
                description="Clear conversation history",
                handler=self._cmd_clear,
                group=self.group_name,
            ),
        ]

    def _cmd_save(self, ctx: CommandContext, args: List[str]) -> None:
        """Save current session."""
        from victor.agent.session import get_session_manager

        if not ctx.agent:
            ctx.print_error("No active session to save")
            return

        session_manager = get_session_manager()
        if session_manager is None:
            ctx.print_error("Session manager not available")
            return

        name = args[0] if args else None

        try:
            session_id = session_manager.save_session(
                messages=ctx.agent.messages,
                model=ctx.agent.model,
                provider_name=ctx.agent.provider_name,
                name=name,
            )
            ctx.print_success(f"Session saved: {session_id}")
        except Exception as e:
            ctx.print_error(f"Failed to save session: {e}")

    def _cmd_load(self, ctx: CommandContext, args: List[str]) -> None:
        """Load a saved session."""
        from victor.agent.session import get_session_manager

        if not args:
            ctx.print_error("Usage: /load <session_id>")
            return

        session_manager = get_session_manager()
        if session_manager is None:
            ctx.print_error("Session manager not available")
            return

        session_id = args[0]

        try:
            session = session_manager.load_session(session_id)
            if session is None:
                ctx.print_error(f"Session not found: {session_id}")
                return

            if ctx.agent:
                # Restore messages to agent
                ctx.agent._messages = list(session.messages)
                ctx.print_success(f"Loaded session: {session.name or session_id}")
                ctx.print(f"  Messages: {len(session.messages)}")
                ctx.print(f"  Model: {session.model}")
            else:
                ctx.print_error("No agent to load session into")
        except Exception as e:
            ctx.print_error(f"Failed to load session: {e}")

    def _cmd_sessions(self, ctx: CommandContext, args: List[str]) -> None:
        """List saved sessions."""
        from victor.agent.session import get_session_manager

        session_manager = get_session_manager()
        if session_manager is None:
            ctx.print_error("Session manager not available")
            return

        try:
            sessions = session_manager.list_sessions(limit=20)

            if not sessions:
                ctx.print("[dim]No saved sessions[/dim]")
                return

            table = Table(title="Saved Sessions")
            table.add_column("ID", style="cyan")
            table.add_column("Name")
            table.add_column("Model")
            table.add_column("Messages", justify="right")
            table.add_column("Created")

            for session in sessions:
                table.add_row(
                    session.session_id[:8],
                    session.name or "-",
                    session.model or "-",
                    str(len(session.messages)),
                    session.created_at.strftime("%Y-%m-%d %H:%M") if session.created_at else "-",
                )

            ctx.console.print(table)
        except Exception as e:
            ctx.print_error(f"Failed to list sessions: {e}")

    def _cmd_resume(self, ctx: CommandContext, args: List[str]) -> None:
        """Resume last session."""
        from victor.agent.session import get_session_manager

        session_manager = get_session_manager()
        if session_manager is None:
            ctx.print_error("Session manager not available")
            return

        try:
            sessions = session_manager.list_sessions(limit=1)
            if not sessions:
                ctx.print("[dim]No sessions to resume[/dim]")
                return

            latest = sessions[0]
            if ctx.agent:
                ctx.agent._messages = list(latest.messages)
                ctx.print_success(f"Resumed session: {latest.name or latest.session_id[:8]}")
            else:
                ctx.print_error("No agent to resume session into")
        except Exception as e:
            ctx.print_error(f"Failed to resume session: {e}")

    def _cmd_clear(self, ctx: CommandContext, args: List[str]) -> None:
        """Clear conversation history."""
        if not ctx.agent:
            ctx.print_error("No active session to clear")
            return

        count = len(ctx.agent.messages)
        ctx.agent._messages.clear()
        ctx.agent._system_added = False
        ctx.print_success(f"Cleared {count} messages from history")
