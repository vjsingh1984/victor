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

"""Session management slash commands: save, load, sessions, resume, compact."""

from __future__ import annotations

import logging

from rich.panel import Panel
from rich.table import Table

from victor.ui.common.constants import FIRST_ARG_INDEX
from victor.ui.slash.protocol import BaseSlashCommand, CommandContext, CommandMetadata
from victor.ui.slash.registry import register_command

logger = logging.getLogger(__name__)


@register_command
class SaveCommand(BaseSlashCommand):
    """Save current conversation to a session file."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="save",
            description="Save current conversation to SQLite session (updates active session or creates new)",
            usage="/save [--new] [name]",
            category="session",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.agent.session import get_session_manager
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

        # Check for --new flag
        force_new = self._has_flag(ctx, "--new", "-n")

        # Extract title (remove --new flag if present)
        args = [arg for arg in ctx.args if arg not in ("--new", "-n")]
        title = " ".join(args) if args else None

        try:
            sqlite_persistence = get_sqlite_session_persistence()

            # Determine session_id to use
            if force_new:
                # Create new session (ignore active_session_id)
                session_id = None
                action = "Created new session"
            elif ctx.agent and getattr(ctx.agent, "active_session_id", None):
                # Update existing active session
                session_id = ctx.agent.active_session_id
                action = f"Updated session {session_id}"
            else:
                # No active session, create new
                session_id = None
                action = "Created new session"

            # Save to SQLite
            session_id = sqlite_persistence.save_session(
                conversation=ctx.agent.conversation if ctx.agent else None,
                model=getattr(ctx.agent, 'model', None) if ctx.agent else None,  # type: ignore[attr-defined]
                provider=getattr(ctx.agent, 'provider_name', None) if ctx.agent else None,  # type: ignore[attr-defined]
                profile=getattr(ctx.settings, "current_profile", "default"),
                session_id=session_id,  # Use existing or None for new
                title=title,
                conversation_state=getattr(ctx.agent, "conversation_state", None) if ctx.agent else None,
            )

            if session_id and ctx.agent:
                # Set active_session_id on agent
                ctx.agent.active_session_id = session_id

                ctx.console.print(
                    Panel(
                        f"{action}!\n\n"
                        f"[bold]Session ID:[/] {session_id}\n"
                        f"[bold]Database:[/] {sqlite_persistence._db_path}\n"
                        f"[bold]Title:[/] {title or 'Auto-generated'}\n\n"
                        f"[dim]Use '/resume {session_id}' to restore[/]\n"
                        f"[dim]Use '/save --new' to create a new session[/]",
                        title="Session Saved",
                        border_style="green",
                    )
                )
            else:
                ctx.console.print("[red]Failed to save session to SQLite[/]")
                logger.error("Session ID was empty after save")

        except Exception as e:
            ctx.console.print(f"[red]Failed to save session:[/] {e}")
            logger.exception("Error saving session")


@register_command
class LoadCommand(BaseSlashCommand):
    """Load a saved session."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="load",
            description="Load a saved session",
            usage="/load <session_id>",
            category="session",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not ctx.args:
            ctx.console.print("[yellow]Usage:[/] /load <session_id>")
            ctx.console.print("[dim]Use '/sessions' to list available sessions[/]")
            return

        if not self._require_agent(ctx):
            return

        from victor.agent.conversation_state import ConversationStateMachine
        from victor.agent.message_history import MessageHistory
        from victor.agent.session import get_session_manager

        session_id = ctx.args[FIRST_ARG_INDEX]

        try:
            session_manager = get_session_manager()
            session = session_manager.load_session(session_id)

            if session is None:
                ctx.console.print(f"[red]Session not found:[/] {session_id}")
                return

            # Restore conversation
            if ctx.agent:
                ctx.agent.conversation = MessageHistory.from_dict(session.conversation)

                # Set active session ID for parallel session support
                ctx.agent.active_session_id = session_id

                # Restore conversation state machine if available
                if session.conversation_state:
                    ctx.agent.conversation_state = ConversationStateMachine.from_dict(
                        session.conversation_state
                    )
                    logger.info(
                        f"Restored conversation state: stage={ctx.agent.conversation_state.get_stage().name}"
                    )

            ctx.console.print(
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
            ctx.console.print(f"[red]Failed to load session:[/] {e}")
            logger.exception("Error loading session")


@register_command
class SessionsCommand(BaseSlashCommand):
    """List saved sessions."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="sessions",
            description="List saved sessions",
            usage="/sessions [limit]",
            aliases=["history"],
            category="session",
        )

    def execute(self, ctx: CommandContext) -> None:
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

        limit = self._parse_int_arg(ctx, 0, default=10)

        try:
            persistence = get_sqlite_session_persistence()
            sessions = persistence.list_sessions(limit=limit)

            if not sessions:
                ctx.console.print("[dim]No saved sessions found[/]")
                ctx.console.print(f"[dim]Database: {persistence._db_path}[/]")
                return

            table = Table(title=f"Saved Sessions (SQLite - last {len(sessions)})")
            table.add_column("ID", style="cyan", no_wrap=True)
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

            ctx.console.print(table)
            ctx.console.print("\n[dim]Use '/resume <session_id>' to restore a session[/]")
            ctx.console.print(
                "[dim]Or '/switch <model> --resume <session_id>' to resume and switch[/]"
            )
        except Exception as e:
            ctx.console.print(f"[red]Failed to list sessions:[/] {e}")
            logger.exception("Error listing sessions")


@register_command
class ResumeCommand(BaseSlashCommand):
    """Resume a session with interactive selection from SQLite."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="resume",
            description="Resume a session from SQLite history",
            usage="/resume [session_id]",
            category="session",
            requires_agent=True,
        )

    def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        from victor.agent.conversation_state import ConversationStateMachine
        from victor.agent.message_history import MessageHistory
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

        # If session_id provided as argument, load it directly
        if ctx.args:
            session_id = ctx.args[FIRST_ARG_INDEX]
            self._load_session(ctx, session_id)
            return

        # Otherwise, show interactive selection
        try:
            persistence = get_sqlite_session_persistence()
            sessions = persistence.list_sessions(limit=20)

            if not sessions:
                ctx.console.print("[yellow]No sessions found in SQLite database[/]")
                ctx.console.print("[dim]Start a conversation to create sessions[/]")
                return

            # Display sessions with numbers
            table = Table(title="Recent Sessions (SQLite)")
            table.add_column("#", style="cyan", no_wrap=True, width=4)
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Messages", justify="right")
            table.add_column("Date", style="dim")

            for idx, session in enumerate(sessions, 1):
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
                    str(idx),
                    session["session_id"],
                    title,
                    session["model"],
                    str(session["message_count"]),
                    date_str,
                )

            ctx.console.print(table)
            ctx.console.print("\n[dim]Enter session number to resume (1-{})[/]", len(sessions))
            ctx.console.print("[dim]Or use: /resume <session_id>[/]")

        except Exception as e:
            ctx.console.print(f"[red]Failed to list sessions:[/] {e}")
            logger.exception("Error listing sessions")

    def _load_session(self, ctx: CommandContext, session_id: str) -> None:
        """Load and restore a session.

        Args:
            ctx: Command context
            session_id: Session ID to load
        """
        from victor.agent.conversation_state import ConversationStateMachine
        from victor.agent.message_history import MessageHistory
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

        try:
            persistence = get_sqlite_session_persistence()
            session_data = persistence.load_session(session_id)

            if session_data is None:
                ctx.console.print(f"[red]Session not found:[/] {session_id}")
                return

            # Restore conversation
            metadata = session_data.get("metadata", {})
            conversation_dict = session_data.get("conversation", {})

            if ctx.agent:
                ctx.agent.conversation = MessageHistory.from_dict(conversation_dict)

                # Set active session ID for parallel session support
                ctx.agent.active_session_id = session_id

                # Restore conversation state if available
                conversation_state_dict = session_data.get("conversation_state")
                if conversation_state_dict:
                    try:
                        ctx.agent.conversation_state = ConversationStateMachine.from_dict(
                            conversation_state_dict
                        )
                        logger.info(
                            f"Restored conversation state: stage={ctx.agent.conversation_state.get_stage().name}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to restore conversation state: {e}")

            ctx.console.print(
                Panel(
                    f"Session restored from SQLite!\n\n"
                    f"[bold]ID:[/] {metadata.get('session_id', session_id)}\n"
                    f"[bold]Title:[/] {metadata.get('title', 'Untitled')}\n"
                    f"[bold]Model:[/] {metadata.get('model', 'N/A')}\n"
                    f"[bold]Provider:[/] {metadata.get('provider', 'N/A')}\n"
                    f"[bold]Messages:[/] {metadata.get('message_count', 0)}\n"
                    f"[bold]Created:[/] {metadata.get('created_at', 'N/A')}",
                    title="Session Resumed",
                    border_style="green",
                )
            )
        except Exception as e:
            ctx.console.print(f"[red]Failed to load session:[/] {e}")
            logger.exception("Error loading session")


@register_command
class CompactCommand(BaseSlashCommand):
    """Compress conversation history to reduce context size."""

    @property
    def metadata(self) -> CommandMetadata:
        return CommandMetadata(
            name="compact",
            description="Compress conversation history (use --smart for AI summarization)",
            usage="/compact [--smart]",
            aliases=["summarize"],
            category="session",
            requires_agent=True,
            is_async=True,
        )

    async def execute(self, ctx: CommandContext) -> None:
        if not self._require_agent(ctx):
            return

        if ctx.agent:
            original_count = ctx.agent.conversation.message_count()

            if original_count < 5:
                ctx.console.print("[dim]Conversation is already small enough, nothing to compact[/]")
                return

            use_smart = self._has_flag(ctx, "--smart", "-s")
            keep_recent = 6

            # Parse --keep N
            keep_val = self._get_flag_value(ctx, "--keep") or self._get_flag_value(ctx, "-k")
            if keep_val:
                try:
                    keep_recent = int(keep_val)
                except ValueError:
                    pass

            keep_recent = min(keep_recent, len(ctx.agent.conversation.messages) // 2)
            messages = ctx.agent.conversation.messages
        else:
            ctx.console.print("[red]No agent available for compaction[/]")
            return

        if use_smart:
            ctx.console.print("[dim]Generating AI summary...[/]")

            conversation_text = "\n".join(
                [
                    (
                        f"{msg.role}: {msg.content[:200]}..."
                        if len(msg.content) > 200
                        else f"{msg.role}: {msg.content}"
                    )
                    for msg in messages[:-keep_recent]
                ]
            )

            summary_prompt = (
                "Summarize this conversation concisely, focusing on key decisions, "
                f"code changes, and context needed for continuation:\n\n{conversation_text}"
            )

            try:
                summary_response = await ctx.agent.chat(summary_prompt)
                summary = summary_response.content

                # Create new conversation with summary + recent messages
                from victor.agent.message_types import Message  # type: ignore[import-not-found]

                new_messages = [
                    Message(role="system", content=f"[Previous conversation summary]\n{summary}"),
                    *messages[-keep_recent:],
                ]
                # Update messages through proper method
                ctx.agent.conversation.clear_messages()
                for msg in new_messages:
                    ctx.agent.conversation.add_message(msg)  # type: ignore[attr-defined]

                ctx.console.print(
                    Panel(
                        f"Compacted {original_count} messages to {len(new_messages)}.\n\n"
                        f"[bold]Summary:[/]\n{summary[:500]}...",
                        title="Smart Compaction Complete",
                        border_style="green",
                    )
                )
            except Exception as e:
                ctx.console.print(f"[red]Smart compaction failed:[/] {e}")
        else:
            # Simple truncation
            if ctx.agent:
                # Update messages through proper method
                ctx.agent.conversation.clear_messages()
                for msg in messages[-keep_recent:]:
                    ctx.agent.conversation.add_message(msg)  # type: ignore[attr-defined]
                new_count = ctx.agent.conversation.message_count()

                ctx.console.print(
                    f"[green]Compacted:[/] {original_count} -> {new_count} messages\n"
                    "[dim]Use /compact --smart for AI-powered summarization[/]"
                )
