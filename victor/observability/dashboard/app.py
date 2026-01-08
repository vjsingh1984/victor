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
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"[EventStats._update_display] Updating display: total={self.total_events}, tools={self.tool_events}, states={self.state_events}, errors={self.error_events}"
        )

        content = self.query_one("#stats-content", Static)
        display_text = (
            f"[bold]Events:[/] {self.total_events}  "
            f"[cyan]Tools:[/] {self.tool_events}  "
            f"[yellow]States:[/] {self.state_events}  "
            f"[red]Errors:[/] {self.error_events}"
        )
        content.update(display_text)
        logger.debug(f"[EventStats._update_display] Display updated with: {display_text}")

    def increment(self, category: str) -> None:
        """Increment counters based on event category."""
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"[EventStats.increment] START: category={category}, current total={self.total_events}"
        )

        self.total_events += 1
        # category is just the first part of topic (e.g., "state", "tool", "lifecycle")
        if category == "tool":
            self.tool_events += 1
            logger.debug(f"[EventStats.increment] Incremented tool_events to {self.tool_events}")
        elif category == "state":
            self.state_events += 1
            logger.debug(f"[EventStats.increment] Incremented state_events to {self.state_events}")
        elif category == "error":
            self.error_events += 1
            logger.debug(f"[EventStats.increment] Incremented error_events to {self.error_events}")
        else:
            logger.debug(
                f"[EventStats.increment] Category '{category}' not tracked in specific counters"
            )

        logger.debug(
            f"[EventStats.increment] END: total={self.total_events}, tools={self.tool_events}, states={self.state_events}, errors={self.error_events}"
        )


class EventLogView(RichLog):
    """Real-time event log viewer with newest events on top."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_lines: list[str] = []  # Store lines in reverse order (newest first)
        self._max_lines = 1000

    def add_event(self, event: Event) -> None:
        """Add an event to the log with rich details (prepends to show newest first)."""
        timestamp = event.datetime.strftime("%H:%M:%S.%f")[:-3]
        # Map category (first part of topic) to colors
        category_colors = {
            "tool": "cyan",
            "state": "yellow",
            "model": "green",
            "error": "red",
            "audit": "magenta",
            "metric": "blue",
            "lifecycle": "white",
            "vertical": "bright_magenta",
        }
        color = category_colors.get(event.category, "dim gray")
        category_name = event.category.upper()

        # Build event header
        lines = []
        log_message = f"[dim]{timestamp}[/] [{color}]{category_name:10}[/] {event.topic}"
        lines.append(log_message)

        # Display rich event details based on topic and data
        if event.data:
            detail_lines = self._format_event_details(event)
            lines.extend(detail_lines)

        # Prepend to stored lines (newest first)
        self._event_lines = lines + self._event_lines

        # Trim to max lines
        if len(self._event_lines) > self._max_lines:
            self._event_lines = self._event_lines[: self._max_lines]

        # Refresh display
        self._refresh_display()

    def _format_event_details(self, event: Event) -> list[str]:
        """Format event details into lines."""
        topic = event.topic
        data = event.data
        lines = []

        # State transition events
        if topic == "state.stage_changed":
            old_stage = data.get("old_stage", "?")
            new_stage = data.get("new_stage", "?")
            confidence = data.get("confidence", 0.0)
            transition_count = data.get("transition_count", 0)
            has_cycle = data.get("has_cycle", False)
            visit_count = data.get("visit_count", 0)

            lines.append(f"         [cyan]Transition:[/] {old_stage} → {new_stage}")
            lines.append(
                f"         [dim]Confidence: {confidence:.1%} | Visits: {visit_count} | Transitions: {transition_count}[/]"
            )
            if has_cycle:
                lines.append("         [red]⚠ Cycle detected![/]")

            # Show tool history if available
            if data.get("tool_history"):
                tools = data["tool_history"]
                if len(tools) <= 5:
                    lines.append(f"         [dim]Tools: {', '.join(tools)}[/]")
                else:
                    lines.append(
                        f"         [dim]Tools: {', '.join(tools[:5])}... ({len(tools)} total)[/]"
                    )

            # Show stage sequence if available
            if data.get("stage_sequence"):
                sequence = data["stage_sequence"]
                if len(sequence) <= 8:
                    lines.append(f"         [dim]Path: {' → '.join(sequence)}[/]")
                else:
                    lines.append(
                        f"         [dim]Path: {' → '.join(sequence[:4])}... → {' → '.join(sequence[-4:])}[/]"
                    )

        # Tool events
        elif topic.startswith("tool."):
            tool_name = data.get("tool_name", "unknown")
            lines.append(f"         [cyan]Tool:[/] {tool_name}")

            # Duration and success for tool.end events
            if "duration_ms" in data:
                duration = data["duration_ms"]
                success = data.get("success", True)
                status = "[green]✓[/]" if success else "[red]✗[/]"
                lines.append(f"         {status} [dim]{duration:.0f}ms[/]")

            # Error information
            if not data.get("success", True) and data.get("error"):
                error_msg = data["error"]
                if isinstance(error_msg, str) and len(error_msg) > 80:
                    error_msg = error_msg[:77] + "..."
                lines.append(f"         [red]Error:[/] {error_msg}")

            # Arguments for tool.start events
            if "arguments" in data:
                args = data["arguments"]
                if args:
                    # Show first few arguments
                    arg_items = list(args.items())[:3]
                    arg_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in arg_items)
                    if len(args) > 3:
                        arg_str += f" ... ({len(args)} args)"
                    lines.append(f"         [dim]Args: {arg_str}[/]")

        # Lifecycle events
        elif topic.startswith("lifecycle."):
            # Chunk tool start events
            if topic == "lifecycle.chunk.tool_start":
                tool_name = data.get("tool_name", "")
                status_msg = data.get("status_msg", "")
                lines.append(f"         [cyan]{tool_name}[/] [dim]{status_msg}[/]")

            # Session events
            elif topic == "lifecycle.session.start":
                if data.get("project"):
                    lines.append(f"         [cyan]Project:[/] {data['project']}")
                if data.get("profile"):
                    lines.append(f"         [dim]Profile: {data['profile']}[/]")

            elif topic == "lifecycle.session.end":
                tool_calls = data.get("tool_calls", 0)
                duration = data.get("duration_seconds", 0)
                success = data.get("success", True)
                status = "[green]✓[/]" if success else "[red]✗[/]"
                lines.append(f"         {status} [dim]{tool_calls} tool calls, {duration:.1f}s[/]")

        # Error events
        elif topic.startswith("error."):
            error_type = topic.split(".", 1)[1] if "." in topic else "error"
            severity = data.get("severity", "error")

            if severity == "warning":
                lines.append(f"         [yellow]⚠ Warning:[/] {error_type}")
            else:
                lines.append(f"         [red]✗ Error:[/] {error_type}")

            if data.get("error"):
                error_msg = data["error"]
                if isinstance(error_msg, str) and len(error_msg) > 100:
                    error_msg = error_msg[:97] + "..."
                lines.append(f"         [dim]{error_msg}[/]")

            # Cycle warnings
            if topic == "error.cycle_warning":
                stage = data.get("stage", "?")
                visit_count = data.get("visit_count", 0)
                sequence = data.get("sequence", [])
                lines.append(f"         [red]Stage '{stage}' visited {visit_count} times[/]")
                if sequence:
                    lines.append(f"         [dim]Recent: {' → '.join(sequence)}[/]")

        # Model events
        elif topic.startswith("model."):
            provider = data.get("provider", "unknown")
            model = data.get("model", "unknown")

            if topic == "model.request":
                message_count = data.get("message_count", 0)
                tool_count = data.get("tool_count", 0)
                lines.append(
                    f"         [cyan]{provider}/{model}[/] [dim]{message_count} msgs, {tool_count} tools[/]"
                )

            elif topic == "model.response":
                tokens = data.get("tokens_used")
                tool_calls = data.get("tool_calls", 0)
                latency = data.get("latency_ms")

                parts = [f"[cyan]{provider}/{model}[/]"]
                if tokens:
                    parts.append(f"[dim]{tokens} tokens[/]")
                if tool_calls:
                    parts.append(f"[dim]{tool_calls} tool calls[/]")
                if latency:
                    parts.append(f"[dim]{latency:.0f}ms[/]")

                lines.append(f"         {' '.join(parts)}")

        # Continuation events
        elif topic.startswith("state.continuation."):
            if topic == "state.continuation.cumulative_intervention_nudge":
                cumulative = data.get("cumulative_interventions", 0)
                read_count = data.get("read_files_count", 0)
                lines.append(
                    f"         [yellow]Interventions:[/] {cumulative} [dim]| Files read: {read_count}[/]"
                )

        # Generic fallback for any event with data
        else:
            # Show a few key fields from the data
            important_fields = ["name", "title", "status", "result", "value", "count"]
            shown_fields = []

            for field in important_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, (str, int, float, bool)):
                        if len(str(value)) <= 50:
                            shown_fields.append(f"{field}={value}")

            if shown_fields:
                lines.append(f"         [dim]{', '.join(shown_fields)}[/]")

        return lines

    def _refresh_display(self) -> None:
        """Refresh the display with current lines."""
        # Clear and rewrite all content
        self.clear()
        for line in self._event_lines:
            self.write(line, expand=True)


class EventTableView(DataTable):
    """Tabular view of events."""

    def on_mount(self) -> None:
        """Set up the table columns."""
        self.add_columns("Time", "Category", "Name", "Details")
        self.cursor_type = "row"

    def add_event(self, event: Event) -> None:
        """Add an event row to the table.

        Events are processed in descending order (newest first), so we append
        to maintain that order in the table.
        """
        import logging

        logger = logging.getLogger(__name__)

        timestamp = event.datetime.strftime("%H:%M:%S")
        timestamp_raw = event.timestamp
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

        # Debug: Log event being added
        try:
            current_row_count = len(self.rows)
        except Exception:
            current_row_count = 0

        logger.info(
            f"[EventTableView.add_event] Adding row #{current_row_count}: timestamp={timestamp} ({timestamp_raw}), topic={event.topic}"
        )

        # Append row (events are already processed in descending order)
        try:
            self.add_row(timestamp, category, event.topic, details)
            logger.debug(
                f"[EventTableView.add_event] Successfully added row for timestamp={timestamp}"
            )
        except Exception as e:
            logger.error(f"[EventTableView.add_event] Error adding row: {e}")
            return

        # Debug: Log state after adding
        try:
            if len(self.rows) > 0:
                first_row = self.rows[0]
                last_row = self.rows[-1] if len(self.rows) > 1 else first_row
                logger.info(
                    f"[EventTableView.add_event] After add: total_rows={len(self.rows)}, first_row_time={first_row[0]}, last_row_time={last_row[0]}"
                )
        except Exception as e:
            logger.debug(f"[EventTableView.add_event] Could not log row state: {e}")


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
        # Handle both tool.* and lifecycle.chunk.tool_start events
        is_tool_event = (
            event.topic.startswith("tool.") or event.topic == "lifecycle.chunk.tool_start"
        )
        if not is_tool_event:
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

        # Count lifecycle.chunk.tool_start events as tool executions
        # For tool.* events, only count end events for stats
        # TODO: Migrate
        if event.topic == "lifecycle.chunk.tool_start":
            stats["calls"] += 1
            stats["successes"] += 1  # Assume success for start events
            stats["last_called"] = event.timestamp
        elif event.topic.endswith(".end"):
            stats["calls"] += 1
            if "duration_ms" in data:
                stats["total_time"] += data["duration_ms"]
            if data.get("success", True):
                stats["successes"] += 1
            stats["last_called"] = event.timestamp

        self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the table with current stats."""
        from datetime import datetime

        self.clear()
        for tool_name, stats in sorted(self._tool_stats.items()):
            calls = stats["calls"]
            avg_time = stats["total_time"] / calls if calls > 0 else 0
            success_rate = (stats["successes"] / calls * 100) if calls > 0 else 0
            # Handle both datetime objects and float timestamps
            last_called_raw = stats["last_called"]
            if last_called_raw:
                if isinstance(last_called_raw, datetime):
                    last_called = last_called_raw.strftime("%H:%M:%S")
                else:
                    # Assume it's a float timestamp
                    last_called = datetime.fromtimestamp(last_called_raw).strftime("%H:%M:%S")
            else:
                last_called = "-"
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
        """Add a vertical integration event (prepends for newest first)."""
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

        # Prepend new entry (newest first)
        content.update(f"{new_entry}\n{text}" if text else new_entry)


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

            # Iterate in reverse to show newest events first
            for event in reversed(self._events[-100:]):  # Show last 100 events, newest first
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
        """Add a span event to the trace view (prepends for newest first)."""
        if not event.topic.startswith("lifecycle."):
            return

        content = self.query_one("#trace-content", Static)
        data = event.data or {}

        timestamp = event.datetime.strftime("%H:%M:%S")

        # Handle lifecycle.chunk.tool_start events
        if event.topic == "lifecycle.chunk.tool_start":
            span_type = data.get("tool_name", "chunk_operation")
            status_msg = data.get("status_msg", "")

            # Get current content
            text = content.content
            if text == "Execution trace will appear here...":
                text = ""

            # Build entry
            new_entry = f"[dim]{timestamp}[/] [cyan]{span_type}[/]"
            if status_msg:
                new_entry += f" [dim]{status_msg}[/]"

            # Prepend new entry (newest first)
            content.update(f"{new_entry}\n{text}" if text else new_entry)
        else:
            # Generic lifecycle event handling
            span_type = data.get("span_type", event.topic.split(".")[-1])
            operation = data.get("operation", "unknown")

            # Get current content
            text = content.content
            if text == "Execution trace will appear here...":
                text = ""

            new_entry = f"[dim]{timestamp}[/] [yellow]{span_type}[/] {operation}"

            # Add duration if available
            if "duration_ms" in data:
                duration = data["duration_ms"]
                new_entry += f" [dim]{duration:.0f}ms[/]"

            # Prepend new entry (newest first)
            content.update(f"{new_entry}\n{text}" if text else new_entry)


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

        # Process tool.end events (contain duration and results)
        if event_name == "tool.end":
            tool_name = data.get("tool_name", "unknown")
            duration_ms = data.get("duration_ms", 0)
            success = data.get("success", True)
            tool_id = data.get("tool_id", "")[:8]  # Use tool_id as span identifier
            arguments = data.get("arguments", {})

            timestamp = event.datetime.strftime("%H:%M:%S")
            args_preview = str(arguments)[:30] if arguments else ""

            status = "[green]OK[/]" if success else "[red]FAIL[/]"

            self.add_row(
                timestamp,
                tool_name,
                status,
                f"{duration_ms:.0f}ms" if duration_ms else "N/A",
                tool_id or "N/A",
                args_preview,
            )
        # Process tool.start events (no duration yet)
        elif event_name == "tool.start":
            tool_name = data.get("tool_name", "unknown")
            tool_id = data.get("tool_id", "")[:8]
            arguments = data.get("arguments", {})

            timestamp = event.datetime.strftime("%H:%M:%S")
            args_preview = str(arguments)[:30] if arguments else ""

            self.add_row(
                timestamp,
                tool_name,
                "[dim]Running...[/]",
                "N/A",
                tool_id or "N/A",
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
        timestamp = event.datetime.strftime("%H:%M:%S")

        # Handle state.stage_changed events
        if event.topic == "state.stage_changed":
            scope = "conversation"  # State machine scope
            key = "stage"  # State key
            old_value = data.get("old_stage", "unknown")
            new_value = data.get("new_stage", "unknown")
        # Handle generic state events with scope/key structure
        else:
            scope = data.get("scope", "unknown")
            key = data.get("key", "unknown")
            old_value = data.get("old_value")
            new_value = data.get("new_value")

        # Truncate values for display
        old_preview = str(old_value)[:30] if old_value is not None else "None"
        new_preview = str(new_value)[:30] if new_value is not None else "None"

        # Append row (events are already processed in descending order)
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
        min-height: 30;
        border: round $primary;
        scrollbar-gutter: stable;
    }

    EventTableView, ToolExecutionView {
        height: 100%;
        min-height: 30;
    }

    # New debugging views
    ExecutionTraceView {
        height: 100%;
        min-height: 30;
        border: round $primary;
        padding: 1;
    }

    ToolCallHistoryView, StateTransitionView {
        height: 100%;
        min-height: 30;
    }

    PerformanceMetricsView {
        height: 100%;
        min-height: 30;
        border: round $primary;
        padding: 2;
    }

    DataTable {
        height: 100%;
        min-height: 30;
    }

    VerticalTraceView {
        height: 100%;
        min-height: 30;
        border: round $primary;
        padding: 1;
    }

    JSONLBrowser {
        height: 100%;
        min-height: 30;
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

        # Load historical events after widgets are fully rendered
        # Use call_later to ensure widgets have their size calculated
        # TODO: Migrate
        self.call_later(self._setup_event_source)

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
            # We're in an async callback, can call UI methods directly
            logger.info(f"[Dashboard.handle_event] #{event_counter[0]}: Processing event")
            self._process_event(event)

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

            # Take last 100 lines and reverse for newest-first display
            # TODO: Migrate
            lines_to_process = all_lines[-100:] if len(all_lines) > 100 else all_lines
            lines_to_process = lines_to_process[::-1]  # Reverse to show newest first

            logger.info(
                f"[Dashboard] Loading {len(lines_to_process)} initial events from {len(all_lines)} total lines (reversed for newest-first)"
            )

            # Debug: Log first and last timestamps from file
            if len(lines_to_process) > 0:
                first_line = lines_to_process[0].strip()
                last_line = lines_to_process[-1].strip()
                try:
                    first_event_data = json.loads(first_line)
                    last_event_data = json.loads(last_line)
                    first_ts = first_event_data.get("timestamp", "N/A")
                    last_ts = last_event_data.get("timestamp", "N/A")
                    logger.info(
                        f"[Dashboard] Initial load timestamp range: first={first_ts}, last={last_ts}"
                    )
                except Exception as e:
                    logger.debug(f"[Dashboard] Could not parse timestamps for debug: {e}")

            for idx, line in enumerate(lines_to_process):
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

                    # Debug: Log timestamp range of new events
                    if len(new_lines) > 0:
                        first_new = new_lines[0].strip()
                        last_new = new_lines[-1].strip()
                        try:
                            first_event_data = json.loads(first_new)
                            last_event_data = json.loads(last_new)
                            first_ts = first_event_data.get("timestamp", "N/A")
                            last_ts = last_event_data.get("timestamp", "N/A")
                            logger.info(
                                f"[Dashboard] Polling batch: {len(new_lines)} events, timestamp range: {first_ts} to {last_ts}"
                            )
                        except Exception as e:
                            logger.debug(f"[Dashboard] Could not parse polling timestamps: {e}")

                    # Process each new line in reverse order (newest first)
                    # This ensures events are added to views in descending timestamp order
                    for idx, line in enumerate(reversed(new_lines)):
                        line = line.strip()
                        if not line:
                            continue

                        event = self._parse_jsonl_line(line)
                        if event:
                            self._event_counter += 1
                            logger.info(
                                f"[Dashboard] Polling event #{self._event_counter} (batch idx {idx}): timestamp={event.timestamp}, topic={event.topic}"
                            )

                            # Process event
                            # We're in an async worker task but can call UI methods directly
                            self._process_event(event)

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
        import traceback

        logger = logging.getLogger(__name__)
        logger.info(
            f"[Dashboard._process_event] PROCESSING timestamp={event.timestamp} [{event.category}/{event.topic}]"
        )

        try:
            # Update stats
            # TODO: Migrate
            self._stats.increment(event.category)
            logger.debug("[Dashboard._process_event] Stats updated")
        except Exception as e:
            logger.error(
                f"[Dashboard._process_event] Error updating stats: {e}\n{traceback.format_exc()}"
            )

        try:
            # Add to views
            # TODO: Migrate
            logger.debug("[Dashboard._process_event] Adding to EventLogView")
            self._event_log.add_event(event)
            logger.debug("[Dashboard._process_event] EventLogView updated successfully")
        except Exception as e:
            logger.error(
                f"[Dashboard._process_event] Error adding to EventLogView: {e}\n{traceback.format_exc()}"
            )

        try:
            logger.debug("[Dashboard._process_event] Adding to EventTableView")
            self._event_table.add_event(event)
            logger.debug("[Dashboard._process_event] EventTableView updated successfully")
        except Exception as e:
            logger.error(
                f"[Dashboard._process_event] Error adding to EventTableView: {e}\n{traceback.format_exc()}"
            )

        try:
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
                # Also route chunk tool events to tool view
                if event.topic == "lifecycle.chunk.tool_start":
                    self._tool_view.add_tool_event(event)
            elif event.topic.startswith("state."):
                # Update state transition view
                # TODO: Migrate
                self._state_transition_view.add_state_event(event)
                self._performance_metrics_view.update_from_state_event(event)
        except Exception as e:
            logger.error(
                f"[Dashboard._process_event] Error updating specific views: {e}\n{traceback.format_exc()}"
            )

        # Explicitly refresh display to show new events
        # TODO: Migrate
        try:
            self._event_log.refresh()
            self._event_table.refresh()
            self._tool_view.refresh()
            self._stats.refresh()
        except Exception as e:
            logger.error(
                f"[Dashboard._process_event] Error refreshing display: {e}\n{traceback.format_exc()}"
            )

        logger.info(f"[Dashboard._process_event] COMPLETE [{event.topic}]")

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
        """Refresh all views in the display."""
        # Refresh all views to ensure latest data is shown
        self._event_log.refresh()
        self._event_table.refresh()
        self._tool_view.refresh()
        self._vertical_view.refresh()
        self._execution_trace_view.refresh()
        self._tool_call_history_view.refresh()
        self._state_transition_view.refresh()
        self._performance_metrics_view.refresh()
        self._stats.refresh()
        self._jsonl_browser.refresh()
        self.refresh()  # Refresh the dashboard itself
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
