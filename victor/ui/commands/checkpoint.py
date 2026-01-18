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

"""Checkpoint CLI commands for Victor.

Provides commands for managing checkpoints (time-travel debugging) including
creating, listing, restoring, and forking checkpoints.

Commands:
    save       - Create a new checkpoint
    list       - List all checkpoints
    show       - Show detailed checkpoint information
    restore    - Restore to a previous checkpoint
    fork       - Create a new session from a checkpoint
    cleanup    - Remove old checkpoints
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

checkpoint_app = typer.Typer(
    name="checkpoint",
    help="Manage checkpoints for time-travel debugging.",
)

console = Console()


@checkpoint_app.command("save")
def save_checkpoint(
    description: str = typer.Argument(
        ...,
        help="Description of the checkpoint",
    ),
    session_id: str = typer.Option(
        "current",
        "--session",
        "-s",
        help="Session ID to associate with checkpoint",
    ),
    auto: bool = typer.Option(
        False,
        "--auto",
        "-a",
        help="Enable auto-checkpointing for this session",
    ),
    interval: int = typer.Option(
        5,
        "--interval",
        "-i",
        help="Auto-checkpoint interval (tool calls)",
    ),
) -> None:
    """Create a new checkpoint.

    Saves the current working tree state and optionally conversation state
    to enable rollback to this point.

    Example:
        victor checkpoint save "Before refactoring user service"
        victor checkpoint save "Major changes" --auto --interval 10
    """
    from victor.agent.enhanced_checkpoints import (
        EnhancedCheckpointManager,
        CheckpointState,
    )

    try:
        manager = EnhancedCheckpointManager(
            auto_checkpoint=auto,
            checkpoint_interval=interval,
        )

        # Create minimal checkpoint state for CLI usage
        state = CheckpointState(
            session_id=session_id,
            metadata={"created_via": "cli", "auto_checkpoint_enabled": auto},
        )

        checkpoint = manager.save_checkpoint(
            description=description,
            session_id=session_id,
            state=state,
        )

        console.print(f"[bold green]✓[/] Checkpoint created")
        console.print(f"  ID: {checkpoint.id}")
        console.print(f"  Description: {description}")
        console.print(f"  Timestamp: {checkpoint.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        if checkpoint.git_checkpoint_id:
            console.print(f"  Git checkpoint: {checkpoint.git_checkpoint_id}")

        if auto:
            console.print(f"\n[dim]Auto-checkpointing enabled (every {interval} tool calls)[/]")

    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to create checkpoint: {e}")
        raise typer.Exit(1)


@checkpoint_app.command("list")
def list_checkpoints(
    session_id: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Filter by session ID",
    ),
    include_git: bool = typer.Option(
        True,
        "--include-git/--no-include-git",
        help="Include git-only checkpoints",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of checkpoints to show",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json",
    ),
) -> None:
    """List all checkpoints.

    Shows both SQLite state checkpoints and git working tree checkpoints.

    Example:
        victor checkpoint list
        victor checkpoint list --session abc123
        victor checkpoint list --limit 50 --format json
    """
    from victor.agent.enhanced_checkpoints import EnhancedCheckpointManager

    try:
        manager = EnhancedCheckpointManager()
        checkpoints = manager.list_checkpoints(
            session_id=session_id,
            include_git=include_git,
        )

        # Apply limit
        checkpoints = checkpoints[:limit]

        if not checkpoints:
            console.print("[dim]No checkpoints found.[/]")
            return

        # Output based on format
        if format == "json":
            console.print(json.dumps(checkpoints, indent=2, default=str))
        else:
            # Table format
            table = Table(title=f"Checkpoints ({len(checkpoints)} shown)")
            table.add_column("ID", style="cyan")
            table.add_column("Timestamp", style="green")
            table.add_column("Description")
            table.add_column("Type", style="yellow")
            table.add_column("Size", justify="right")

            for cp in checkpoints:
                # Determine checkpoint type
                cp_type = "State+Git" if cp.get("git_checkpoint_id") else "State"
                if cp.get("git_only"):
                    cp_type = "Git"

                # Format size
                size = cp.get("size_bytes", 0)
                if size > 0:
                    size_str = f"{size:,} bytes"
                else:
                    size_str = "-"

                # Format timestamp
                timestamp = cp.get("timestamp", "")
                if timestamp:
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass

                table.add_row(
                    cp["id"][:20],
                    timestamp,
                    cp.get("description", "")[:50],
                    cp_type,
                    size_str,
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to list checkpoints: {e}")
        raise typer.Exit(1)


@checkpoint_app.command("show")
def show_checkpoint(
    checkpoint_id: str = typer.Argument(
        ...,
        help="Checkpoint ID to show",
    ),
) -> None:
    """Show detailed information about a checkpoint.

    Displays checkpoint metadata and state information.

    Example:
        victor checkpoint show checkpoint_abc123
    """
    from victor.agent.enhanced_checkpoints import EnhancedCheckpointManager

    try:
        manager = EnhancedCheckpointManager()

        # Try to load from SQLite first
        state = manager.load_checkpoint(checkpoint_id)

        if state:
            # Show state checkpoint
            console.print(f"\n[bold cyan]Checkpoint:[/] {checkpoint_id}\n")
            console.print("[dim]" + "─" * 50 + "[/]\n")

            # Basic info table
            table = Table(show_header=False, box=None)
            table.add_column("Property", style="cyan")
            table.add_column("Value")

            table.add_row("Session ID", state.session_id)
            table.add_row("Timestamp", state.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            table.add_row("Messages", str(len(state.messages)))
            table.add_row("Tool Calls", str(len(state.tool_calls)))

            console.print(table)

            # Show context keys
            if state.context:
                console.print("\n[bold]Context Keys:[/]")
                for key in list(state.context.keys())[:20]:
                    console.print(f"  • {key}")

            # Show execution state keys
            if state.execution_state:
                console.print("\n[bold]Execution State Keys:[/]")
                for key in list(state.execution_state.keys())[:20]:
                    console.print(f"  • {key}")

            # Show metadata
            if state.metadata:
                console.print("\n[bold]Metadata:[/]")
                console.print(
                    Panel(
                        json.dumps(state.metadata, indent=2, default=str),
                        border_style="blue",
                    )
                )
        else:
            # Try to find in git checkpoints
            from victor.agent.checkpoints import GitCheckpointManager

            git_manager = GitCheckpointManager()
            git_cp = git_manager.get_checkpoint(checkpoint_id)

            if git_cp:
                console.print(f"\n[bold cyan]Git Checkpoint:[/] {checkpoint_id}\n")
                console.print("[dim]" + "─" * 50 + "[/]\n")

                table = Table(show_header=False, box=None)
                table.add_column("Property", style="cyan")
                table.add_column("Value")

                table.add_row("ID", git_cp.id)
                table.add_row("Description", git_cp.description)
                table.add_row("Timestamp", git_cp.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                table.add_row("Stash Ref", git_cp.stash_ref or "N/A")

                console.print(table)
            else:
                console.print(f"[bold red]Error:[/] Checkpoint not found: {checkpoint_id}")
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to show checkpoint: {e}")
        raise typer.Exit(1)


@checkpoint_app.command("restore")
def restore_checkpoint(
    checkpoint_id: str = typer.Argument(
        ...,
        help="Checkpoint ID to restore",
    ),
    restore_git: bool = typer.Option(
        True,
        "--git/--no-git",
        help="Also restore git working tree",
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Restore to a previous checkpoint.

    Restores both the conversation state and optionally the git working tree
    to the state when the checkpoint was created.

    WARNING: This will discard current uncommitted changes.

    Example:
        victor checkpoint restore checkpoint_abc123
        victor checkpoint restore checkpoint_abc123 --no-git --confirm
    """
    if not confirm:
        console.print(
            "[bold yellow]Warning:[/] This will restore to the checkpoint state.\n"
            "Current uncommitted changes may be lost."
        )
        confirm_restore = typer.confirm("Are you sure you want to continue?", default=False)

        if not confirm_restore:
            console.print("[dim]Operation cancelled.[/]")
            raise typer.Exit(0)

    from victor.agent.enhanced_checkpoints import EnhancedCheckpointManager

    try:
        manager = EnhancedCheckpointManager()

        console.print(f"\n[dim]Restoring checkpoint {checkpoint_id}...[/]")

        success = manager.restore_checkpoint(
            checkpoint_id=checkpoint_id,
            restore_git=restore_git,
        )

        if success:
            console.print("[bold green]✓[/] Checkpoint restored successfully")
            console.print(f"  Checkpoint: {checkpoint_id}")
            console.print(f"  Git restored: {'Yes' if restore_git else 'No'}")
        else:
            console.print(f"[bold red]✗[/] Failed to restore checkpoint")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to restore checkpoint: {e}")
        raise typer.Exit(1)


@checkpoint_app.command("fork")
def fork_session(
    checkpoint_id: str = typer.Argument(
        ...,
        help="Checkpoint ID to fork from",
    ),
) -> None:
    """Create a new session forked from a checkpoint.

    Creates a new session with the checkpoint's state, allowing you to
    explore alternate execution paths from a specific point.

    Example:
        victor checkpoint fork checkpoint_abc123
    """
    from victor.agent.enhanced_checkpoints import EnhancedCheckpointManager

    try:
        manager = EnhancedCheckpointManager()

        new_session_id = manager.fork_session(checkpoint_id)

        console.print(f"[bold green]✓[/] Session forked successfully")
        console.print(f"  New session ID: {new_session_id}")
        console.print(f"  Forked from: {checkpoint_id}")
        console.print(
            "\n[dim]Use --session {new_session_id} with other commands to use this session[/]"
        )

    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to fork session: {e}")
        raise typer.Exit(1)


@checkpoint_app.command("cleanup")
def cleanup_checkpoints(
    keep_count: int = typer.Option(
        20,
        "--keep",
        "-k",
        help="Number of recent checkpoints to keep per session",
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Remove old checkpoints to free up space.

    Keeps the N most recent checkpoints per session and removes the rest.

    Example:
        victor checkpoint cleanup --keep 10
        victor checkpoint cleanup --confirm
    """
    if not confirm:
        console.print(
            f"[bold yellow]Warning:[/] This will remove old checkpoints (keeping {keep_count} most recent).\n"
            "This operation cannot be undone."
        )
        confirm_cleanup = typer.confirm("Continue with cleanup?", default=False)

        if not confirm_cleanup:
            console.print("[dim]Operation cancelled.[/]")
            raise typer.Exit(0)

    from victor.agent.enhanced_checkpoints import EnhancedCheckpointManager

    try:
        manager = EnhancedCheckpointManager()

        results = manager.cleanup_old(keep_count=keep_count)

        console.print(f"[bold green]✓[/] Cleanup complete")
        console.print(f"  SQLite checkpoints removed: {results['sqlite']}")
        console.print(f"  Git checkpoints removed: {results['git']}")

    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to cleanup checkpoints: {e}")
        raise typer.Exit(1)


@checkpoint_app.command("auto")
def configure_auto_checkpoint(
    enable: bool = typer.Option(
        True,
        "--enable/--disable",
    ),
    interval: int = typer.Option(
        5,
        "--interval",
        "-i",
        help="Checkpoint interval (tool calls)",
    ),
) -> None:
    """Configure automatic checkpointing.

    Enables or disables automatic checkpointing after every N tool calls.

    Example:
        victor checkpoint auto --enable --interval 10
        victor checkpoint auto --disable
    """
    console.print(f"\n[bold]Auto-checkpointing Configuration:[/]\n")
    console.print(f"  Status: {'[green]Enabled[/]' if enable else '[red]Disabled[/]'}")
    console.print(f"  Interval: Every {interval} tool calls\n")

    if enable:
        console.print(
            "[dim]Auto-checkpointing will create checkpoints automatically during tool execution.[/]"
        )
        console.print("[dim]Use 'victor checkpoint list' to view all checkpoints.[/]")
    else:
        console.print(
            "[dim]Auto-checkpointing disabled. Use 'victor checkpoint save' for manual checkpoints.[/]"
        )


__all__ = ["checkpoint_app"]
