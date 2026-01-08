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

"""Sessions command for managing Victor conversation sessions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

sessions_app = typer.Typer(name="sessions", help="Manage conversation sessions.")
console = Console()


@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of sessions to list"),
    all: bool = typer.Option(False, "--all", help="List all sessions (no limit)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List saved conversation sessions.

    Examples:
        victor sessions list              # List last 10 sessions
        victor sessions list --limit 20   # List last 20 sessions
        victor sessions list --all        # List all sessions
        victor sessions list --json       # Output as JSON
    """
    try:
        persistence = get_sqlite_session_persistence()
        # If --all is specified, use a very high limit
        actual_limit = 100000 if all else limit
        sessions = persistence.list_sessions(limit=actual_limit)

        if not sessions:
            console.print("[dim]No sessions found[/]")
            sys.exit(0)

        if json_output:
            # Output as JSON
            console.print_json(json.dumps(sessions, indent=2))
        else:
            # Output as table
            table = Table(title=f"Saved Sessions (last {len(sessions)})")
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
            console.print("[dim]Use 'victor sessions show <session_id>' for details[/]")

    except Exception as e:
        console.print(f"[red]Error listing sessions:[/] {e}")
        sys.exit(1)


@sessions_app.command("show")
def sessions_show(
    session_id: str = typer.Argument(..., help="Session ID to show"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show details of a specific session.

    Examples:
        victor sessions show myproj-9Kx7Z2
        victor sessions show myproj-9Kx7Z2 --json
    """
    try:
        persistence = get_sqlite_session_persistence()
        session_data = persistence.load_session(session_id)

        if not session_data:
            console.print(f"[red]Session not found:[/] {session_id}")
            sys.exit(1)

        metadata = session_data.get("metadata", {})

        if json_output:
            # Output as JSON
            console.print_json(json.dumps(session_data, indent=2))
        else:
            # Output formatted
            from rich.panel import Panel

            # Get messages
            messages = session_data.get("conversation", {}).get("messages", [])
            message_count = len(messages)

            panel_content = (
                f"[bold]Session ID:[/] {metadata.get('session_id', session_id)}\n"
                f"[bold]Title:[/] {metadata.get('title', 'Untitled')}\n"
                f"[bold]Model:[/] {metadata.get('model', 'N/A')}\n"
                f"[bold]Provider:[/] {metadata.get('provider', 'N/A')}\n"
                f"[bold]Profile:[/] {metadata.get('profile', 'N/A')}\n"
                f"[bold]Messages:[/] {message_count}\n"
                f"[bold]Created:[/] {metadata.get('created_at', 'N/A')}\n"
                f"[bold]Updated:[/] {metadata.get('updated_at', 'N/A')}\n"
            )

            console.print(Panel(panel_content, title="Session Details", border_style="cyan"))

            # Show recent messages (last 5)
            if messages:
                console.print("\n[bold]Recent Messages:[/]")
                for msg in messages[-5:]:
                    role_style = {"user": "cyan", "assistant": "green", "system": "dim"}.get(
                        msg.get("role", ""), "white"
                    )
                    content = msg.get("content", "")
                    # Truncate long messages
                    if len(content) > 200:
                        content = content[:200] + "..."
                    console.print(f"[{role_style}]{msg.get('role', '').capitalize()}:[/] {content}")

    except Exception as e:
        console.print(f"[red]Error showing session:[/] {e}")
        sys.exit(1)


@sessions_app.command("search")
def sessions_search(
    query: str = typer.Argument(..., help="Search query string"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search sessions by title or content.

    Examples:
        victor sessions search CI/CD
        victor sessions search "authentication" --limit 5
        victor sessions search test --json
    """
    try:
        persistence = get_sqlite_session_persistence()
        sessions = persistence.search_sessions(query, limit=limit)

        if not sessions:
            console.print(f"[dim]No sessions found matching '{query}'[/]")
            sys.exit(0)

        if json_output:
            console.print_json(json.dumps(sessions, indent=2))
        else:
            table = Table(title=f"Sessions matching '{query}'")
            table.add_column("Session ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="white")
            table.add_column("Model", style="yellow")
            table.add_column("Provider", style="blue")
            table.add_column("Messages", justify="right")

            for session in sessions:
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
                )

            console.print(table)
            console.print(f"\n[dim]Found {len(sessions)} session(s)[/]")

    except Exception as e:
        console.print(f"[red]Error searching sessions:[/] {e}")
        sys.exit(1)


@sessions_app.command("delete")
def sessions_delete(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a session.

    Examples:
        victor sessions delete myproj-9Kx7Z2
        victor sessions delete myproj-9Kx7Z2 --yes
    """
    try:
        if not yes:
            from rich.prompt import Confirm

            if not Confirm.ask(f"Delete session {session_id}?"):
                console.print("[dim]Cancelled[/]")
                sys.exit(0)

        persistence = get_sqlite_session_persistence()
        success = persistence.delete_session(session_id)

        if success:
            console.print(f"[green]✓[/] Deleted session: {session_id}")
        else:
            console.print(f"[red]Failed to delete session:[/] {session_id}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error deleting session:[/] {e}")
        sys.exit(1)


@sessions_app.command("export")
def sessions_export(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty print JSON"),
) -> None:
    """Export all sessions to JSON file.

    Examples:
        victor sessions export
        victor sessions export --output sessions.json
        victor sessions export --no-pretty
    """
    try:
        persistence = get_sqlite_session_persistence()
        all_sessions = persistence.list_sessions(limit=1000)  # Get all sessions

        if not all_sessions:
            console.print("[dim]No sessions to export[/]")
            sys.exit(0)

        # Export data
        export_data = []
        for session in all_sessions:
            session_id = session["session_id"]
            full_session = persistence.load_session(session_id)
            if full_session:
                export_data.append(full_session)

        # Determine output path
        if not output:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = Path(f"victor_sessions_export_{timestamp}.json")

        # Write to file
        with open(output, "w") as f:
            if pretty:
                json.dump(export_data, f, indent=2)
            else:
                json.dump(export_data, f)

        console.print(f"[green]✓[/] Exported {len(export_data)} session(s) to {output}")

    except Exception as e:
        console.print(f"[red]Error exporting sessions:[/] {e}")
        sys.exit(1)


@sessions_app.command("clear")
def sessions_clear(
    prefix: Optional[str] = typer.Argument(None, help="Clear sessions with IDs starting with this prefix (min 6 chars)"),
    all: bool = typer.Option(False, "--all", help="Clear all sessions (default behavior when no prefix specified)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Clear sessions from the database.

    This will permanently delete conversation sessions. Use with caution!

    Examples:
        victor sessions clear                    # Clear all sessions (with confirmation)
        victor sessions clear --all --yes        # Clear all sessions (skip confirmation)
        victor sessions clear myproj-9Kx         # Clear sessions starting with prefix
        victor sessions clear myproj-9Kx --yes   # Clear sessions by prefix (skip confirmation)
    """
    try:
        # Validate prefix if provided
        if prefix and len(prefix) < 6:
            console.print("[red]Error:[/] Prefix must be at least 6 characters long.")
            sys.exit(1)

        persistence = get_sqlite_session_persistence()

        # Get all sessions to show what will be deleted
        all_sessions = persistence.list_sessions(limit=100000)

        # Filter sessions by prefix if specified
        if prefix:
            sessions_to_delete = [s for s in all_sessions if s["session_id"].startswith(prefix)]
            if not sessions_to_delete:
                console.print(f"[dim]No sessions found matching prefix '{prefix}'[/]")
                sys.exit(0)
        else:
            sessions_to_delete = all_sessions

        count = len(sessions_to_delete)

        if count == 0:
            console.print("[dim]No sessions found. Database is already empty.[/]")
            sys.exit(0)

        # Show summary of what will be deleted
        if prefix:
            console.print(f"[yellow]⚠[/]  Found {count} session(s) matching prefix '{prefix}'.")
            if not yes:
                from rich.prompt import Confirm

                if not Confirm.ask(
                    f"Are you sure you want to delete {count} session(s) starting with '{prefix}'? This cannot be undone.",
                    default=False,
                ):
                    console.print("[dim]Cancelled[/]")
                    sys.exit(0)
        else:
            console.print(f"[yellow]⚠[/]  Found {count} session(s) in database.")
            if not yes:
                from rich.prompt import Confirm

                if not Confirm.ask(
                    f"Are you sure you want to delete ALL {count} session(s)? This cannot be undone.",
                    default=False,
                ):
                    console.print("[dim]Cancelled[/]")
                    sys.exit(0)

        # Delete sessions
        deleted_count = 0
        for session in sessions_to_delete:
            session_id = session["session_id"]
            if persistence.delete_session(session_id):
                deleted_count += 1

        if prefix:
            console.print(f"[green]✓[/] Cleared {deleted_count} session(s) matching prefix '{prefix}'.")
        else:
            console.print(f"[green]✓[/] Cleared {deleted_count} session(s) from database.")

    except Exception as e:
        console.print(f"[red]Error clearing sessions:[/] {e}")
        sys.exit(1)
