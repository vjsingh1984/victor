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

"""Victor Dashboard CLI command.

Provides the `victor dashboard` command for launching the
Textual-based observability dashboard.

Usage:
    victor dashboard                    # Launch dashboard
    victor dashboard --log-file events.jsonl  # Load historical events
    victor dashboard --no-live          # Disable live event streaming
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

dashboard_app = typer.Typer(
    name="dashboard",
    help="Launch the observability dashboard for visualizing events and traces.",
)

console = Console()


@dashboard_app.callback(invoke_without_command=True)
def dashboard(
    ctx: typer.Context,
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        "-f",
        help="Path to a JSONL log file to load historical events.",
    ),
    live: bool = typer.Option(
        True,
        "--live/--no-live",
        help="Enable/disable live event streaming from EventBus.",
    ),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Run in demo mode with simulated events.",
    ),
) -> None:
    """Launch the Victor observability dashboard.

    The dashboard provides a TUI for visualizing:
    - Real-time events from the EventBus
    - Historical events from JSONL log files
    - Tool execution statistics and history
    - Vertical integration traces

    Examples:
        # Launch with live event streaming
        victor dashboard

        # Load historical events from a file
        victor dashboard --log-file ~/.victor/events.jsonl

        # Disable live streaming, only show historical data
        victor dashboard --log-file events.jsonl --no-live

        # Run demo mode with simulated events
        victor dashboard --demo
    """
    if ctx.invoked_subcommand is None:
        asyncio.run(
            run_dashboard(
                log_file=log_file,
                live=live,
                demo=demo,
            )
        )


async def run_dashboard(
    log_file: Optional[str] = None,
    live: bool = True,
    demo: bool = False,
) -> None:
    """Run the observability dashboard.

    Args:
        log_file: Optional path to JSONL log file
        live: Whether to enable live event streaming
        demo: Whether to run in demo mode
    """
    try:
        from victor.observability.dashboard import ObservabilityDashboard
    except ImportError as e:
        console.print(
            "[bold red]Error:[/] Failed to import dashboard components. "
            "Ensure Textual is installed."
        )
        console.print(f"[dim]{e}[/]")
        raise typer.Exit(1)

    # Validate log file if provided
    if log_file:
        log_path = Path(log_file)
        if not log_path.exists():
            console.print(f"[bold red]Error:[/] Log file not found: {log_file}")
            raise typer.Exit(1)
        if log_path.suffix.lower() not in (".jsonl", ".json", ".log"):
            console.print(
                "[bold yellow]Warning:[/] Log file does not have expected extension "
                "(.jsonl, .json, .log). Proceeding anyway."
            )

    # Run dashboard
    if demo:
        await run_demo_dashboard(log_file, live)
    else:
        app = ObservabilityDashboard(
            log_file=log_file,
            subscribe_to_bus=live,
        )
        await app.run_async()


async def run_demo_dashboard(
    log_file: Optional[str] = None,
    live: bool = True,
) -> None:
    """Run the dashboard in demo mode with simulated events.

    Args:
        log_file: Optional path to JSONL log file
        live: Whether to enable live event streaming
    """
    import random
    from datetime import datetime, timezone

    from victor.observability.dashboard import ObservabilityDashboard
    from victor.observability.event_bus import (
        EventBus,
        EventCategory,
        VictorEvent,
    )

    # Create dashboard
    app = ObservabilityDashboard(
        log_file=log_file,
        subscribe_to_bus=live,
    )

    # Demo event generator
    async def generate_demo_events() -> None:
        """Generate simulated events for demo mode."""
        bus = EventBus.get_instance()

        tool_names = [
            "read_file",
            "write_file",
            "run_bash",
            "search_codebase",
            "analyze_code",
            "git_status",
            "git_diff",
        ]

        stages = [
            "INITIAL",
            "PLANNING",
            "READING",
            "ANALYZING",
            "EXECUTING",
            "VERIFICATION",
            "COMPLETION",
        ]

        verticals = ["coding", "research", "devops", "dataanalysis"]

        current_stage = "INITIAL"

        # Wait for app to start
        await asyncio.sleep(1)

        while True:
            try:
                # Generate random event
                event_type = random.choice(["tool", "state", "model", "vertical", "error"])

                if event_type == "tool":
                    tool = random.choice(tool_names)
                    # Start event
                    bus.publish(
                        VictorEvent(
                            category=EventCategory.TOOL,
                            name=f"{tool}.start",
                            data={
                                "tool_name": tool,
                                "arguments": {"path": "/demo/file.py"},
                            },
                        )
                    )
                    await asyncio.sleep(random.uniform(0.1, 0.5))

                    # End event
                    success = random.random() > 0.1
                    bus.publish(
                        VictorEvent(
                            category=EventCategory.TOOL,
                            name=f"{tool}.end",
                            data={
                                "tool_name": tool,
                                "success": success,
                                "duration_ms": random.uniform(10, 500),
                                "result": "Demo result" if success else None,
                                "error": "Demo error" if not success else None,
                            },
                        )
                    )

                elif event_type == "state":
                    current_idx = stages.index(current_stage)
                    if current_idx < len(stages) - 1:
                        new_stage = stages[current_idx + 1]
                    else:
                        new_stage = stages[0]

                    bus.emit_state_change(
                        old_stage=current_stage,
                        new_stage=new_stage,
                        confidence=random.uniform(0.7, 1.0),
                    )
                    current_stage = new_stage

                elif event_type == "model":
                    bus.publish(
                        VictorEvent(
                            category=EventCategory.MODEL,
                            name="response",
                            data={
                                "provider": "anthropic",
                                "model": "claude-3-5-sonnet",
                                "tokens_used": random.randint(100, 2000),
                                "latency_ms": random.uniform(200, 2000),
                            },
                        )
                    )

                elif event_type == "vertical":
                    vertical = random.choice(verticals)
                    bus.publish(
                        VictorEvent(
                            category=EventCategory.VERTICAL,
                            name="vertical.applied",
                            data={
                                "vertical": vertical,
                                "action": "applied",
                                "config": {
                                    "mode": "build",
                                    "tool_budget": 25,
                                },
                            },
                        )
                    )

                elif event_type == "error":
                    bus.emit_error(
                        Exception("Demo error for testing"),
                        context={"component": "demo"},
                        recoverable=True,
                    )

                await asyncio.sleep(random.uniform(0.5, 2.0))

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)

    # Start event generator as background task
    if live:
        generator_task = asyncio.create_task(generate_demo_events())
    else:
        generator_task = None

    try:
        await app.run_async()
    finally:
        if generator_task:
            generator_task.cancel()
            try:
                await generator_task
            except asyncio.CancelledError:
                pass


@dashboard_app.command("status")
def dashboard_status() -> None:
    """Show current event bus status and statistics."""
    try:
        from victor.observability.event_bus import EventBus
    except ImportError:
        console.print("[red]Error: Could not import EventBus[/]")
        raise typer.Exit(1)

    bus = EventBus.get_instance()

    console.print("\n[bold]EventBus Status[/]\n")
    console.print(f"  Queue Depth: {bus.get_queue_depth()}")
    console.print(f"  Queue Capacity: {bus.get_queue_capacity()}")
    console.print(f"  Queue Full: {bus.is_queue_full()}")
    console.print(f"  Pending Tasks: {bus.get_pending_task_count()}")

    # Subscription counts
    console.print("\n[bold]Subscriptions[/]\n")
    total_subs = bus.get_subscription_count()
    console.print(f"  Total: {total_subs}")

    # Backpressure metrics
    metrics = bus.get_backpressure_metrics()
    console.print("\n[bold]Backpressure Metrics[/]\n")
    console.print(f"  Events Dropped: {metrics.events_dropped}")
    console.print(f"  Events Rejected: {metrics.events_rejected}")
    console.print(f"  Peak Queue Depth: {metrics.peak_queue_depth}")
    console.print(f"  Backpressure Events: {metrics.backpressure_events}")

    # Sampling metrics if configured
    sampling_metrics = bus.get_sampling_metrics()
    if sampling_metrics.events_sampled > 0 or sampling_metrics.events_dropped > 0:
        console.print("\n[bold]Sampling Metrics[/]\n")
        console.print(f"  Sampled: {sampling_metrics.events_sampled}")
        console.print(f"  Dropped: {sampling_metrics.events_dropped}")

    console.print("")
