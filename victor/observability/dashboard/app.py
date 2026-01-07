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

"""Observability Dashboard - Textual TUI Application.

Main Textual application providing a dashboard for visualizing:
- Real-time events from ObservabilityBus
- Historical events from JSONL log files
- Tool execution traces
- Vertical configuration and integration results
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from victor.config.settings import get_settings
from victor.core.events import Event, ObservabilityBus, get_observability_bus

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
    RichLog,
)
from textual.reactive import reactive
from textual import work


# TODO: Migrate


class EventStats(Static):
    """Widget displaying event statistics."""

    total_events: reactive[int] = reactive(0)
    tool_events: reactive[int] = reactive(0)
    state_events: reactive[int] = reactive(0)
    error_events: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static(id="stats-content")

    def watch_total_events(self, value: int) -> None:
        self._update_display()

    def watch_tool_events(self, value: int) -> None:
        self._update_display()

    def watch_state_events(self, value: int) -> None:
        self._update_display()

    def watch_error_events(self, value: int) -> None:
        self._update_display()

    def _update_display(self) -> None:
        content = self.query_one("#stats-content", Static)
        content.update(
            f"[bold]Events:[/] {self.total_events}  "
            f"[cyan]Tools:[/] {self.tool_events}  "
            f"[yellow]States:[/] {self.state_events}  "
            f"[red]Errors:[/] {self.error_events}"
        )

    def increment(self, category: str) -> None:
        """Increment counters based on event category."""
        self.total_events += 1
        if category.startswith("tool."):  # MIGRATED from EventCategory.TOOL
            self.tool_events += 1
        elif category.startswith("state."):  # MIGRATED from EventCategory.STATE
            self.state_events += 1
        elif category.startswith("error."):  # MIGRATED from EventCategory.ERROR
            self.error_events += 1


class EventLogView(RichLog):
    """Real-time event log viewer."""

    def add_event(self, event: Event) -> None:
        """Add an event to the log."""
        timestamp = event.datetime.strftime("%H:%M:%S.%f")[:-3]
        category_colors = {
            "tool.": "cyan",
            "state.": "yellow",
            "model.": "green",
            "error.": "red",
            "audit.": "magenta",
            "metric.": "blue",
            "lifecycle.": "white",
            "vertical.": "bright_magenta",
            "custom": "dim",
        }
        color = category_colors.get(event.category, "white")
        category_name = event.category.upper()

        self.write(f"[dim]{timestamp}[/] [{color}]{category_name:10}[/] {event.topic}")

        # Show key data fields for certain events
        # TODO: Migrate
        if event.topic.startswith("tool.") and event.data:
            tool_name = event.data.get("tool_name", "")
            if "duration_ms" in event.data:
                duration = event.data["duration_ms"]
                success = event.data.get("success", True)
                status = "[green]OK[/]" if success else "[red]FAIL[/]"
                self.write(f"         [dim]{tool_name} {status} ({duration:.0f}ms)[/]")


class EventTableView(DataTable):
    """Tabular view of events."""

    def on_mount(self) -> None:
        """Set up the table columns."""
        self.add_columns("Time", "Category", "Name", "Details")
        self.cursor_type = "row"

    def add_event(self, event: Event) -> None:
        """Add an event row to the table."""
        timestamp = event.datetime.strftime("%H:%M:%S")
        category = event.category

        # Build details string
        details = ""
        if event.data:
            if "tool_name" in event.data:
                details = event.data["tool_name"]
                if "duration_ms" in event.data:
                    details += f" ({event.data['duration_ms']:.0f}ms)"
            elif "old_stage" in event.data and "new_stage" in event.data:
                details = f"{event.data['old_stage']} -> {event.data['new_stage']}"
            elif "message" in event.data:
                details = str(event.data["message"])[:50]

        self.add_row(timestamp, category, event.topic, details)


class ToolExecutionView(DataTable):
    """View showing tool execution history."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_stats: Dict[str, Dict[str, Any]] = {}

    def on_mount(self) -> None:
        """Set up the table columns."""
        self.add_columns("Tool", "Calls", "Avg Time", "Success Rate", "Last Called")
        self.cursor_type = "row"

    def add_tool_event(self, event: Event) -> None:
        """Process a tool event and update statistics."""
        if not event.topic.startswith("tool."):
            return

        data = event.data or {}
        tool_name = data.get("tool_name", "unknown")

        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {
                "calls": 0,
                "total_time": 0.0,
                "successes": 0,
                "last_called": None,
            }

        stats = self._tool_stats[tool_name]

        # Only count end events for stats
        # TODO: Migrate
        if event.topic.endswith(".end"):
            stats["calls"] += 1
            if "duration_ms" in data:
                stats["total_time"] += data["duration_ms"]
            if data.get("success", True):
                stats["successes"] += 1
            stats["last_called"] = event.timestamp

        self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the table with current stats."""
        self.clear()
        for tool_name, stats in sorted(self._tool_stats.items()):
            calls = stats["calls"]
            avg_time = stats["total_time"] / calls if calls > 0 else 0
            success_rate = (stats["successes"] / calls * 100) if calls > 0 else 0
            last_called = stats["last_called"].strftime("%H:%M:%S") if stats["last_called"] else "-"
            self.add_row(
                tool_name,
                str(calls),
                f"{avg_time:.0f}ms",
                f"{success_rate:.0f}%",
                last_called,
            )


class VerticalTraceView(ScrollableContainer):
    """View showing vertical integration traces."""

    def compose(self) -> ComposeResult:
        yield Static("[dim]Vertical integration traces will appear here...[/]", id="trace-content")

    def add_vertical_event(self, event: Event) -> None:
        """Add a vertical integration event."""
        if not event.topic.startswith("vertical."):
            return

        content = self.query_one("#trace-content", Static)
        data = event.data or {}

        timestamp = event.datetime.strftime("%H:%M:%S")
        vertical_name = data.get("vertical", "unknown")
        action = data.get("action", event.topic)

        # Get current content
        text = content.content
        if text == "Vertical integration traces will appear here...":
            text = ""

        new_entry = f"[dim]{timestamp}[/] [magenta]{vertical_name}[/]: {action}"
        if "config" in data:
            config_preview = str(data["config"])[:100]
            new_entry += f"\n         [dim]{config_preview}[/]"

        content.update(f"{text}\n{new_entry}" if text else new_entry)


class JSONLBrowser(ScrollableContainer):
    """Browser for historical JSONL log files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._events: List[Event] = []

    def compose(self) -> ComposeResult:
        yield Static("[dim]Load a JSONL file to browse historical events...[/]", id="jsonl-content")
        yield DataTable(id="jsonl-table")

    def on_mount(self) -> None:
        """Set up the table."""
        table = self.query_one("#jsonl-table", DataTable)
        table.add_columns("Time", "Category", "Name", "Session", "Data Preview")
        table.cursor_type = "row"
        table.display = False

    def load_file(self, path: Path) -> int:
        """Load events from a JSONL file.

        Args:
            path: Path to the JSONL file

        Returns:
            Number of events loaded
        """
        self._events.clear()
        content = self.query_one("#jsonl-content", Static)
        table = self.query_one("#jsonl-table", DataTable)

        if not path.exists():
            content.update(f"[red]File not found: {path}[/]")
            return 0

        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            # VictorEvent removed - migrate to Event.from_dict()
                            # TODO: Parse Event from JSON data properly
                            # For now, just validate JSON
                            json.loads(line)
                            # Skip this event for now
                            continue
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue

            # Update display
            # TODO: Migrate
            content.update(f"[green]Loaded {len(self._events)} events from {path.name}[/]")
            table.display = True
            table.clear()

            for event in self._events[-100:]:  # Show last 100 events
                # TODO: Migrate
                timestamp = event.datetime.strftime("%Y-%m-%d %H:%M:%S")
                category = event.category
                session = (event.correlation_id or "")[:8]
                data_preview = str(event.data)[:50] if event.data else ""
                table.add_row(timestamp, category, event.topic, session, data_preview)

            return len(self._events)

        except Exception as e:
            content.update(f"[red]Error loading file: {e}[/]")
            return 0


class ExecutionTraceView(ScrollableContainer):
    """View showing execution span tree from ExecutionTracer.

    Displays hierarchical execution flow with parent-child relationships.
    """

    def compose(self) -> ComposeResult:
        yield Static("[dim]Execution trace will appear here...[/]", id="trace-content")

    def add_span_event(self, event: Event) -> None:
        """Add a span event to the trace view."""
        if not event.topic.startswith("lifecycle."):
            return

        content = self.query_one("#trace-content", Static)
        data = event.data or {}

        timestamp = event.datetime.strftime("%H:%M:%S")
        span_type = data.get("span_type", "unknown")
        agent_id = data.get("agent_id", "unknown")
        parent_id = data.get("parent_id")

        # Get current content
        text = content.content
        if text == "Execution trace will appear here...":
            text = ""

        # Build tree representation
        indent = ""
        if parent_id:
            indent = "  └─"  # Indent for child spans

        status = ""
        if "status" in data:
            status_color = "green" if data["status"] == "success" else "red"
            status = f" [{status_color}]{data['status']}[/]"

        new_entry = f"[dim]{timestamp}[/]{indent} [{span_type}]{span_type}[/] ({agent_id}){status}"
        if "duration_ms" in data:
            duration = data["duration_ms"]
            new_entry += f" [dim]{duration:.0f}ms[/]"

        content.update(f"{text}\n{new_entry}" if text else new_entry)


class ToolCallHistoryView(DataTable):
    """View showing detailed tool call history from ToolCallTracer.

    Shows all tool calls with linkage to execution spans.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calls: List[Dict[str, Any]] = []

    def on_mount(self) -> None:
        """Set up the table columns."""
        self.add_columns("Time", "Tool", "Status", "Duration", "Span ID", "Arguments")
        self.cursor_type = "row"

    def add_tool_call_event(self, event: Event) -> None:
        """Add a tool call event to the history."""
        if not event.topic.startswith("tool."):
            return

        data = event.data or {}
        event_name = event.topic

        # Process tool_call_completed events
        if event_name == "tool_call_completed":
            tool_name = data.get("tool_name", "unknown")
            duration_ms = data.get("duration_ms", 0)
            parent_span_id = data.get("parent_span_id", "unknown")[:8]
            arguments = data.get("arguments", {})

            timestamp = event.datetime.strftime("%H:%M:%S")
            args_preview = str(arguments)[:30] if arguments else ""

            self.add_row(
                timestamp,
                tool_name,
                "[green]OK[/]",
                f"{duration_ms:.0f}ms",
                parent_span_id,
                args_preview,
            )

        # Process tool_call_failed events
        elif event_name == "tool_call_failed":
            tool_name = data.get("tool_name", "unknown")
            duration_ms = data.get("duration_ms", 0)
            parent_span_id = data.get("parent_span_id", "unknown")[:8]
            error = data.get("error", "Unknown error")

            timestamp = event.datetime.strftime("%H:%M:%S")
            error_preview = error[:30] if error else ""

            self.add_row(
                timestamp,
                tool_name,
                f"[red]FAIL[/] ({error_preview})",
                f"{duration_ms:.0f}ms",
                parent_span_id,
                "",
            )


class StateTransitionView(DataTable):
    """View showing state transition history from StateTracer.

    Shows state changes across all scopes with metadata.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transitions: List[Dict[str, Any]] = []

    def on_mount(self) -> None:
        """Set up the table columns."""
        self.add_columns("Time", "Scope", "Key", "Old Value", "New Value")
        self.cursor_type = "row"

    def add_state_event(self, event: Event) -> None:
        """Add a state transition event."""
        if not event.topic.startswith("state."):
            return

        data = event.data or {}
        scope = data.get("scope", "unknown")
        key = data.get("key", "unknown")
        old_value = data.get("old_value")
        new_value = data.get("new_value")

        timestamp = event.datetime.strftime("%H:%M:%S")

        # Truncate values for display
        # TODO: Migrate
        old_preview = str(old_value)[:30] if old_value is not None else "None"
        new_preview = str(new_value)[:30] if new_value is not None else "None"

        self.add_row(timestamp, scope, key, old_preview, new_preview)


class PerformanceMetricsView(Static):
    """View showing performance summary from all tracers.

    Aggregates metrics from ExecutionTracer, ToolCallTracer, and StateTracer.
    """

    total_spans: reactive[int] = reactive(0)
    active_spans: reactive[int] = reactive(0)
    total_tool_calls: reactive[int] = reactive(0)
    failed_tool_calls: reactive[int] = reactive(0)
    avg_tool_duration: reactive[float] = reactive(0.0)
    total_state_transitions: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static(id="metrics-content")

    def _update_display(self) -> None:
        """Update the metrics display."""
        content = self.query_one("#metrics-content", Static)

        success_rate = 0.0
        if self.total_tool_calls > 0:
            success_rate = (
                (self.total_tool_calls - self.failed_tool_calls) / self.total_tool_calls
            ) * 100

        content.update(
            f"[bold]Execution Spans:[/] {self.total_spans} total, {self.active_spans} active\n\n"
            f"[bold]Tool Calls:[/] {self.total_tool_calls} total, {self.failed_tool_calls} failed "
            f"({success_rate:.0f}% success)\n"
            f"[bold]Avg Duration:[/] {self.avg_tool_duration:.0f}ms\n\n"
            f"[bold]State Transitions:[/] {self.total_state_transitions}"
        )

    def watch_total_spans(self, value: int) -> None:
        self._update_display()

    def watch_active_spans(self, value: int) -> None:
        self._update_display()

    def watch_total_tool_calls(self, value: int) -> None:
        self._update_display()

    def watch_failed_tool_calls(self, value: int) -> None:
        self._update_display()

    def watch_avg_tool_duration(self, value: float) -> None:
        self._update_display()

    def watch_total_state_transitions(self, value: int) -> None:
        self._update_display()

    def update_from_span_event(self, event: Event) -> None:
        """Update metrics from span event."""
        if event.topic.startswith("lifecycle."):
            if event.topic == "span_started":
                self.total_spans += 1
                self.active_spans += 1
            elif event.topic == "span_ended":
                self.active_spans -= 1

    def update_from_tool_event(self, event: Event) -> None:
        """Update metrics from tool event."""
        if event.topic.startswith("tool."):
            data = event.data or {}
            if event.topic == "tool_call_completed":
                self.total_tool_calls += 1
                duration = data.get("duration_ms", 0)
                # Update rolling average
                current_avg = self.avg_tool_duration
                count = self.total_tool_calls
                self.avg_tool_duration = ((current_avg * (count - 1)) + duration) / count
            elif event.topic == "tool_call_failed":
                self.total_tool_calls += 1
                self.failed_tool_calls += 1

    def update_from_state_event(self, event: Event) -> None:
        """Update metrics from state event."""
        if event.topic.startswith("state."):
            self.total_state_transitions += 1


class ObservabilityDashboard(App):
    """Textual TUI Dashboard for Victor Observability.

    Features:
    - Real-time event streaming from EventBus
    - Multiple views: Log, Table, Tools, Verticals
    - JSONL file browser for historical data
    - Event statistics and filtering

    Layout:
    +--------------------------------------------------+
    | Victor Observability Dashboard                    |
    +--------------------------------------------------+
    | Events: 42  Tools: 15  States: 8  Errors: 2      |
    +--------------------------------------------------+
    | [Events] [Tools] [Verticals] [History]           |
    | +----------------------------------------------+ |
    | | Event log / table / tool stats / etc         | |
    | |                                              | |
    | +----------------------------------------------+ |
    +--------------------------------------------------+
    | Ctrl+C Exit | Ctrl+P Pause | Ctrl+L Clear        |
    +--------------------------------------------------+
    """

    CSS = """
    Screen {
        background: $background;
    }

    #stats-bar {
        height: 3;
        background: $panel;
        border-bottom: solid $primary;
        padding: 1;
    }

    #stats-content {
        width: 100%;
        text-align: center;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1;
    }

    EventLogView {
        height: 100%;
        border: round $primary;
        scrollbar-gutter: stable;
    }

    EventTableView, ToolExecutionView {
        height: 100%;
    }

    # New debugging views
    ExecutionTraceView {
        height: 100%;
        border: round $primary;
        padding: 1;
    }

    ToolCallHistoryView, StateTransitionView {
        height: 100%;
    }

    PerformanceMetricsView {
        height: 100%;
        border: round $primary;
        padding: 2;
    }

    DataTable {
        height: 100%;
    }

    VerticalTraceView {
        height: 100%;
        border: round $primary;
        padding: 1;
    }

    JSONLBrowser {
        height: 100%;
        border: round $primary;
        padding: 1;
    }

    #jsonl-table {
        height: 1fr;
        margin-top: 1;
    }

    .paused-indicator {
        color: $warning;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Exit", show=True),
        Binding("ctrl+p", "toggle_pause", "Pause/Resume", show=True),
        Binding("ctrl+l", "clear_events", "Clear", show=True),
        Binding("ctrl+r", "refresh", "Refresh", show=True),
        Binding("f1", "show_help", "Help", show=True),
    ]

    TITLE = "Victor Observability Dashboard"

    def __init__(
        self,
        log_file: Optional[str] = None,
        subscribe_to_bus: bool = True,
        **kwargs,
    ):
        """Initialize the dashboard.

        Args:
            log_file: Optional path to a JSONL log file to load
            subscribe_to_bus: Whether to subscribe to EventBus for real-time events
        """
        super().__init__(**kwargs)
        self._log_file = Path(log_file) if log_file else None
        self._subscribe_to_bus = subscribe_to_bus
        self._paused = False
        self._unsubscribe: Optional[Callable[[], None]] = None
        self._event_buffer: List[Event] = []
        self._subscription_handles: List[Any] = []  # For canonical event system subscriptions

        # Widget references
        self._stats: Optional[EventStats] = None
        self._event_log: Optional[EventLogView] = None
        self._event_table: Optional[EventTableView] = None
        self._tool_view: Optional[ToolExecutionView] = None
        self._vertical_view: Optional[VerticalTraceView] = None
        self._jsonl_browser: Optional[JSONLBrowser] = None
        # New debugging views
        self._execution_trace_view: Optional[ExecutionTraceView] = None
        self._tool_call_history_view: Optional[ToolCallHistoryView] = None
        self._state_transition_view: Optional[StateTransitionView] = None
        self._performance_metrics_view: Optional[PerformanceMetricsView] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield EventStats(id="stats-bar")

        with TabbedContent():
            with TabPane("Events", id="tab-events"):
                yield EventLogView(id="event-log", highlight=True, markup=True)
            with TabPane("Table", id="tab-table"):
                yield EventTableView(id="event-table")
            with TabPane("Tools", id="tab-tools"):
                yield ToolExecutionView(id="tool-view")
            with TabPane("Verticals", id="tab-verticals"):
                yield VerticalTraceView(id="vertical-view")
            with TabPane("History", id="tab-history"):
                yield JSONLBrowser(id="jsonl-browser")
            # New debugging tabs
            # TODO: Migrate
            with TabPane("Execution", id="tab-execution"):
                yield ExecutionTraceView(id="execution-trace-view")
            with TabPane("Tool Calls", id="tab-tool-calls"):
                yield ToolCallHistoryView(id="tool-call-history-view")
            with TabPane("State", id="tab-state"):
                yield StateTransitionView(id="state-transition-view")
            with TabPane("Metrics", id="tab-metrics"):
                yield PerformanceMetricsView(id="performance-metrics-view")
        yield Footer()

    def on_mount(self) -> None:
        """Handle mount event."""
        import logging
        import os

        logger = logging.getLogger(__name__)

        logger.info("[Dashboard] Mounting dashboard with ObservabilityBus")

        # Get ObservabilityBus
        self._event_bus = get_observability_bus()
        logger.info(f"[Dashboard] ObservabilityBus acquired: {type(self._event_bus).__name__}")

        # Subscribe to EventBus
        # TODO: Migrate
        self._subscribe_to_events()

        # Get ALL widget references FIRST before loading events
        # This ensures _process_event() can access all views
        # TODO: Migrate
        self._stats = self.query_one("#stats-bar", EventStats)
        self._event_log = self.query_one("#event-log", EventLogView)
        self._event_table = self.query_one("#event-table", EventTableView)
        self._tool_view = self.query_one("#tool-view", ToolExecutionView)
        self._vertical_view = self.query_one("#vertical-view", VerticalTraceView)
        self._jsonl_browser = self.query_one("#jsonl-browser", JSONLBrowser)
        # New debugging views
        # TODO: Migrate
        self._execution_trace_view = self.query_one("#execution-trace-view", ExecutionTraceView)
        self._tool_call_history_view = self.query_one(
            "#tool-call-history-view", ToolCallHistoryView
        )
        self._state_transition_view = self.query_one("#state-transition-view", StateTransitionView)
        self._performance_metrics_view = self.query_one(
            "#performance-metrics-view", PerformanceMetricsView
        )

        logger.info("[Dashboard] All widget references acquired")

        # Check if using JSONL backend - if so, load historical events
        # For in-memory backend, we only see events published after dashboard starts
        # TODO: Migrate
        self._jsonl_path = Path(os.path.expanduser("~/.victor/metrics/victor.jsonl"))
        self._last_position = 0
        self._event_counter = 0

        # Load historical events after widgets are rendered
        # TODO: Migrate
        self.call_after_refresh(self._setup_event_source)

    def on_unmount(self) -> None:
        """Handle unmount event."""
        import logging

        logger = logging.getLogger(__name__)
        logger.info("[Dashboard] Unmounting")
        self._polling = False

        if self._unsubscribe:
            self._unsubscribe()

    def _subscribe_to_events(self) -> None:
        """Subscribe to ObservabilityBus for real-time events.

        Migrated to canonical event system using pattern-based subscriptions.
        """
        import logging
        import asyncio

        logger = logging.getLogger(__name__)

        event_counter = [0]  # Use list to avoid nonlocal

        async def handle_event(event: Event) -> None:
            """Handle incoming event from canonical event system.

            This is called asynchronously from the ObservabilityBus backend.
            """
            event_counter[0] += 1
            # Use INFO level to ensure it's always visible
            logger.info(f"[Dashboard.handle_event] #{event_counter[0]}: RECEIVED [{event.topic}]")

            if self._paused:
                logger.info(f"[Dashboard.handle_event] #{event_counter[0]}: PAUSED, buffering")
                self._event_buffer.append(event)
                return

            # Schedule UI update on Textual's event loop
            # Note: If we're already in the main thread (e.g., during on_mount()),
            # call_from_thread() will fail, so process directly
            logger.info(f"[Dashboard.handle_event] #{event_counter[0]}: Processing event")
            try:
                self.call_from_thread(self._process_event, event)
            except RuntimeError as e:
                # Already in main thread, process directly
                if "must run in a different thread" in str(e):
                    logger.info(
                        f"[Dashboard.handle_event] #{event_counter[0]}: Already in main thread, processing directly"
                    )
                    self._process_event(event)
                else:
                    raise
            except Exception as e:
                logger.error(f"[Dashboard.handle_event] #{event_counter[0]}: Exception: {e}")
                import traceback

                logger.error(f"[Dashboard.handle_event] Traceback: {traceback.format_exc()}")

        # Subscribe to all event patterns using canonical event system
        # The canonical system uses pattern-based subscriptions (e.g., "tool.*")
        patterns = ["tool.*", "state.*", "model.*", "error.*", "lifecycle.*", "metric.*", "trace.*"]

        async def do_subscribe():
            """Perform async subscription to ObservabilityBus."""
            handles = []
            for pattern in patterns:
                try:
                    handle = await self._event_bus.subscribe(pattern, handle_event)
                    handles.append(handle)
                    logger.debug(f"[Dashboard] Subscribed to pattern: {pattern}")
                except Exception as e:
                    logger.warning(f"[Dashboard] Failed to subscribe to {pattern}: {e}")

            # Store handles for cleanup
            self._subscription_handles = handles
            logger.info(f"[Dashboard] Successfully subscribed to {len(handles)} event patterns")

        # Run subscription in background since on_mount is sync
        try:
            # Try to get running loop
            loop = asyncio.get_running_loop()
            loop.create_task(do_subscribe())
            logger.info("[Dashboard] Event subscription initiated")
        except RuntimeError:
            # No event loop yet, defer subscription
            logger.warning("[Dashboard] No event loop yet, deferring subscription")
            self._subscription_handles = []

        # Create unsubscribe function for cleanup
        def unsubscribe():
            """Unsubscribe from all event patterns."""

            async def do_unsubscribe():
                for handle in self._subscription_handles:
                    try:
                        if hasattr(handle, "unsubscribe"):
                            await handle.unsubscribe()
                    except Exception as e:
                        logger.warning(f"[Dashboard] Failed to unsubscribe: {e}")

            # Run unsubscription
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(do_unsubscribe())
            except RuntimeError:
                logger.warning("[Dashboard] No event loop for unsubscription")

        self._unsubscribe = unsubscribe
        logger.info("[Dashboard] Subscription handler registered")

    def _setup_event_source(self) -> None:
        """Set up event source based on backend type.

        - For JSONL backend: Load historical events and poll for new ones
        - For in-memory backend: Only receive events published after dashboard starts
        """
        import logging

        logger = logging.getLogger(__name__)

        settings = get_settings()
        backend = getattr(settings, "eventbus_backend", "memory")

        logger.info(f"[Dashboard] Setting up event source for backend: {backend}")

        # Load historical events from JSONL file (for both memory and jsonl backends)
        if self._jsonl_path.exists():
            logger.info(f"[Dashboard] Loading historical events from {self._jsonl_path}")
            self._setup_jsonl_source()
        else:
            logger.info(f"[Dashboard] JSONL file not found: {self._jsonl_path}")
            logger.info("[Dashboard] Will only show new events")

    def _setup_jsonl_source(self) -> None:
        """Set up JSONL file as event source."""
        import logging

        logger = logging.getLogger(__name__)

        # Start polling for new events
        self._polling = True
        self._poll_jsonl_file()

        # Load last 100 events
        # TODO: Migrate
        if not self._jsonl_path.exists():
            logger.warning(f"[Dashboard] JSONL file does not exist: {self._jsonl_path}")
            return

        try:
            with open(self._jsonl_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            # Take last 100 lines
            # TODO: Migrate
            lines_to_process = all_lines[-100:] if len(all_lines) > 100 else all_lines

            logger.info(
                f"[Dashboard] Loading {len(lines_to_process)} initial events from {len(all_lines)} total lines"
            )

            for line in lines_to_process:
                line = line.strip()
                if not line:
                    continue

                event = self._parse_jsonl_line(line)
                if event:
                    self._event_counter += 1
                    logger.info(
                        f"[Dashboard] Initial event #{self._event_counter}: [{event.category}/{event.topic}]"
                    )
                    self._process_event(event)

            # Update position to end of file
            # TODO: Migrate
            self._last_position = self._jsonl_path.stat().st_size
            logger.info(
                f"[Dashboard] Loaded {self._event_counter} events, file position: {self._last_position}"
            )

        except Exception as e:
            logger.error(f"[Dashboard] Error loading initial events: {e}")
            import traceback

            logger.error(f"[Dashboard] Traceback: {traceback.format_exc()}")

    @work(exclusive=True)
    async def _poll_jsonl_file(self) -> None:
        """Poll JSONL file for new events."""
        import logging

        logger = logging.getLogger(__name__)

        logger.info("[Dashboard] Starting JSONL file polling")

        while self._polling:
            try:
                if not self._jsonl_path.exists():
                    await asyncio.sleep(1.0)
                    continue

                current_size = self._jsonl_path.stat().st_size

                # Check if file has new content
                if current_size > self._last_position:
                    logger.debug(
                        f"[Dashboard] File grew from {self._last_position} to {current_size}"
                    )

                    # Read new content
                    # TODO: Migrate
                    with open(self._jsonl_path, "r", encoding="utf-8") as f:
                        f.seek(self._last_position)
                        new_lines = f.readlines()
                        self._last_position = f.tell()

                    logger.debug(f"[Dashboard] Read {len(new_lines)} new lines")

                    # Process each new line
                    for line in new_lines:
                        line = line.strip()
                        if not line:
                            continue

                        event = self._parse_jsonl_line(line)
                        if event:
                            self._event_counter += 1
                            logger.info(
                                f"[Dashboard] New event #{self._event_counter}: [{event.category}/{event.topic}]"
                            )

                            # Process in main thread
                            # TODO: Migrate
                            self.call_from_thread(self._process_event, event)

                # Poll every 1 second
                # TODO: Migrate
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"[Dashboard] Error polling file: {e}")
                import traceback

                logger.error(f"[Dashboard] Traceback: {traceback.format_exc()}")
                await asyncio.sleep(1.0)

    def _parse_jsonl_line(self, line: str) -> Optional[Event]:
        """Parse a JSONL event line into an Event with backward compatibility."""
        try:
            import logging

            logger = logging.getLogger(__name__)

            # Parse JSON
            data = json.loads(line)

            # Backward compatibility: handle old format (category + name) vs new format (topic)
            if "topic" not in data:
                # Old format: construct topic from category + name
                category_str = data.get("category", "custom")
                name = data.get("name", "unknown")
                data["topic"] = f"{category_str.lower()}.{name.lower()}"

            # Handle timestamp format
            if "timestamp" in data and isinstance(data["timestamp"], str):
                from datetime import datetime

                # Convert ISO format string to unix timestamp
                try:
                    if "T" in data["timestamp"]:
                        dt = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                    else:
                        dt = datetime.fromisoformat(data["timestamp"])
                    data["timestamp"] = dt.timestamp()
                except Exception:
                    data["timestamp"] = time.time()

            # Create Event from dictionary
            event = Event.from_dict(data)
            return event

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            # Only log at debug level to avoid spamming - these are expected for old format events
            logger.debug(f"[Dashboard] Failed to parse JSONL line: {e}")
            logger.debug(f"[Dashboard] Line content: {line[:300]}")
            return None

    def _process_event(self, event: Event) -> None:
        """Process and display an event."""
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"[Dashboard._process_event] PROCESSING [{event.category}/{event.topic}]")

        # Update stats
        # TODO: Migrate
        self._stats.increment(event.category)
        logger.debug("[Dashboard._process_event] Stats updated")

        # Add to views
        # TODO: Migrate
        logger.debug("[Dashboard._process_event] Adding to EventLogView")
        self._event_log.add_event(event)

        logger.debug("[Dashboard._process_event] Adding to EventTableView")
        self._event_table.add_event(event)

        # Handle specific event types
        # TODO: Migrate
        if event.topic.startswith("tool."):
            self._tool_view.add_tool_event(event)
            # Update debugging views
            # TODO: Migrate
            self._tool_call_history_view.add_tool_call_event(event)
            self._performance_metrics_view.update_from_tool_event(event)
        elif event.topic.startswith("vertical."):
            self._vertical_view.add_vertical_event(event)
        elif event.topic.startswith("lifecycle."):
            # Update execution trace view
            # TODO: Migrate
            self._execution_trace_view.add_span_event(event)
            self._performance_metrics_view.update_from_span_event(event)
        elif event.topic.startswith("state."):
            # Update state transition view
            # TODO: Migrate
            self._state_transition_view.add_state_event(event)
            self._performance_metrics_view.update_from_state_event(event)

        # Explicitly refresh display to show new events
        # TODO: Migrate
        self.refresh()

    @work
    async def _load_historical_file(self) -> None:
        """Load historical events from JSONL file."""
        if self._log_file and self._jsonl_browser:
            count = self._jsonl_browser.load_file(self._log_file)
            if count > 0:
                self.notify(f"Loaded {count} events from {self._log_file.name}")

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_toggle_pause(self) -> None:
        """Toggle event stream pause."""
        self._paused = not self._paused
        if self._paused:
            self.notify("Event stream paused", severity="warning")
            self.sub_title = "[PAUSED]"
        else:
            self.sub_title = ""
            # Process buffered events
            for event in self._event_buffer:
                self._process_event(event)
            self._event_buffer.clear()
            self.notify("Event stream resumed")

    def action_clear_events(self) -> None:
        """Clear all event displays (internal state only, not log file)."""
        # Clear stats
        self._stats.total_events = 0
        self._stats.tool_events = 0
        self._stats.state_events = 0
        self._stats.error_events = 0

        # Clear event log
        # TODO: Migrate
        self._event_log.clear()

        # Clear table
        # TODO: Migrate
        self._event_table.clear()

        # Clear tool view
        # TODO: Migrate
        self._tool_view.clear()
        self._tool_view._tool_stats.clear()

        # Clear vertical trace view
        # TODO: Migrate
        self._vertical_view.clear()

        # Clear debugging views
        # TODO: Migrate
        self._execution_trace_view.clear()
        self._execution_trace_view._spans.clear()

        self._tool_call_history_view.clear()
        self._tool_call_history_view._tool_calls.clear()

        self._state_transition_view.clear()
        self._state_transition_view._transitions.clear()

        self._performance_metrics_view.clear()

        self.notify("Events cleared (log file preserved)")

    def action_refresh(self) -> None:
        """Refresh the display."""
        self.refresh()
        self.notify("Display refreshed")

    def action_show_help(self) -> None:
        """Show help information."""
        help_text = """
Victor Observability Dashboard

Keyboard Shortcuts:
  Ctrl+C    Exit dashboard
  Ctrl+P    Pause/Resume event stream
  Ctrl+L    Clear display (log file preserved)
  Ctrl+R    Refresh display
  F1        Show this help

Features:
  • Loads historical events from ~/.victor/logs/victor.log at startup
  • Real-time event streaming from running agents
  • Clear only affects display, not log file

Tabs:
  Events    - Real-time event log
  Table     - Tabular event view
  Tools     - Tool execution statistics
  Verticals - Vertical integration traces
  History   - Browse JSONL log files
"""
        self.notify(help_text.strip(), timeout=10)


async def run_dashboard(
    log_file: Optional[str] = None,
    subscribe_to_bus: bool = True,
) -> None:
    """Run the observability dashboard.

    Args:
        log_file: Optional path to a JSONL log file to load
        subscribe_to_bus: Whether to subscribe to EventBus for real-time events
    """
    app = ObservabilityDashboard(
        log_file=log_file,
        subscribe_to_bus=subscribe_to_bus,
    )
    await app.run_async()
