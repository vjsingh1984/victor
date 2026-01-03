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
- Real-time events from EventBus
- Historical events from JSONL log files
- Tool execution traces
- Vertical configuration and integration results
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

from victor.observability.event_bus import EventBus, EventCategory, VictorEvent


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

    def increment(self, category: EventCategory) -> None:
        """Increment counters based on event category."""
        self.total_events += 1
        if category == EventCategory.TOOL:
            self.tool_events += 1
        elif category == EventCategory.STATE:
            self.state_events += 1
        elif category == EventCategory.ERROR:
            self.error_events += 1


class EventLogView(RichLog):
    """Real-time event log viewer."""

    def add_event(self, event: VictorEvent) -> None:
        """Add an event to the log."""
        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        category_colors = {
            EventCategory.TOOL: "cyan",
            EventCategory.STATE: "yellow",
            EventCategory.MODEL: "green",
            EventCategory.ERROR: "red",
            EventCategory.AUDIT: "magenta",
            EventCategory.METRIC: "blue",
            EventCategory.LIFECYCLE: "white",
            EventCategory.VERTICAL: "bright_magenta",
            EventCategory.CUSTOM: "dim",
        }
        color = category_colors.get(event.category, "white")
        category_name = event.category.value.upper() if event.category else "UNKNOWN"

        self.write(f"[dim]{timestamp}[/] [{color}]{category_name:10}[/] {event.name}")

        # Show key data fields for certain events
        if event.category == EventCategory.TOOL and event.data:
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

    def add_event(self, event: VictorEvent) -> None:
        """Add an event row to the table."""
        timestamp = event.timestamp.strftime("%H:%M:%S")
        category = event.category.value if event.category else "unknown"

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

        self.add_row(timestamp, category, event.name, details)


class ToolExecutionView(DataTable):
    """View showing tool execution history."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_stats: Dict[str, Dict[str, Any]] = {}

    def on_mount(self) -> None:
        """Set up the table columns."""
        self.add_columns("Tool", "Calls", "Avg Time", "Success Rate", "Last Called")
        self.cursor_type = "row"

    def add_tool_event(self, event: VictorEvent) -> None:
        """Process a tool event and update statistics."""
        if event.category != EventCategory.TOOL:
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
        if event.name.endswith(".end"):
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

    def add_vertical_event(self, event: VictorEvent) -> None:
        """Add a vertical integration event."""
        if event.category != EventCategory.VERTICAL:
            return

        content = self.query_one("#trace-content", Static)
        data = event.data or {}

        timestamp = event.timestamp.strftime("%H:%M:%S")
        vertical_name = data.get("vertical", "unknown")
        action = data.get("action", event.name)

        text = content.renderable
        if text == "[dim]Vertical integration traces will appear here...[/]":
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
        self._events: List[VictorEvent] = []

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
                            data = json.loads(line)
                            event = VictorEvent.from_dict(data)
                            self._events.append(event)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue

            # Update display
            content.update(f"[green]Loaded {len(self._events)} events from {path.name}[/]")
            table.display = True
            table.clear()

            for event in self._events[-100:]:  # Show last 100 events
                timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                category = event.category.value if event.category else "unknown"
                session = (event.session_id or "")[:8]
                data_preview = str(event.data)[:50] if event.data else ""
                table.add_row(timestamp, category, event.name, session, data_preview)

            return len(self._events)

        except Exception as e:
            content.update(f"[red]Error loading file: {e}[/]")
            return 0


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
        self._event_buffer: List[VictorEvent] = []

        # Widget references
        self._stats: Optional[EventStats] = None
        self._event_log: Optional[EventLogView] = None
        self._event_table: Optional[EventTableView] = None
        self._tool_view: Optional[ToolExecutionView] = None
        self._vertical_view: Optional[VerticalTraceView] = None
        self._jsonl_browser: Optional[JSONLBrowser] = None

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
        yield Footer()

    def on_mount(self) -> None:
        """Handle mount event."""
        # Get widget references
        self._stats = self.query_one("#stats-bar", EventStats)
        self._event_log = self.query_one("#event-log", EventLogView)
        self._event_table = self.query_one("#event-table", EventTableView)
        self._tool_view = self.query_one("#tool-view", ToolExecutionView)
        self._vertical_view = self.query_one("#vertical-view", VerticalTraceView)
        self._jsonl_browser = self.query_one("#jsonl-browser", JSONLBrowser)

        # Subscribe to EventBus
        if self._subscribe_to_bus:
            self._subscribe_to_events()

        # Load log file if provided
        if self._log_file:
            self._load_historical_file()

    def on_unmount(self) -> None:
        """Handle unmount event."""
        if self._unsubscribe:
            self._unsubscribe()

    def _subscribe_to_events(self) -> None:
        """Subscribe to EventBus for real-time events."""
        bus = EventBus.get_instance()

        def handle_event(event: VictorEvent) -> None:
            """Handle incoming event."""
            if self._paused:
                self._event_buffer.append(event)
                return
            self._process_event(event)

        # Subscribe to all event categories
        self._unsubscribe = bus.subscribe_all(handle_event)

    def _process_event(self, event: VictorEvent) -> None:
        """Process and display an event."""
        # Update stats
        self._stats.increment(event.category)

        # Add to views
        self._event_log.add_event(event)
        self._event_table.add_event(event)

        # Handle specific event types
        if event.category == EventCategory.TOOL:
            self._tool_view.add_tool_event(event)
        elif event.category == EventCategory.VERTICAL:
            self._vertical_view.add_vertical_event(event)

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
        """Clear all event displays."""
        # Clear stats
        self._stats.total_events = 0
        self._stats.tool_events = 0
        self._stats.state_events = 0
        self._stats.error_events = 0

        # Clear event log
        self._event_log.clear()

        # Clear table
        self._event_table.clear()

        # Clear tool view
        self._tool_view.clear()
        self._tool_view._tool_stats.clear()

        self.notify("Events cleared")

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
  Ctrl+L    Clear all events
  Ctrl+R    Refresh display
  F1        Show this help

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
