# Victor Observability Dashboard

Real-time TUI dashboard for monitoring Victor agent execution.

## Overview

The dashboard provides a terminal-based interface (Textual TUI) for viewing events generated during agent execution. It
  connects to Victor's event system and displays multiple views of execution data.

**Launch**: `victor dashboard`

## Features

- **Real-time event streaming**: Monitors JSONL event file (~/.victor/metrics/victor.jsonl)
- **Multiple view tabs**: 9 different perspectives on execution data
- **Descending order display**: Newest events appear at top
- **Automatic event limits**: Configurable max events per view (default: 200-500)
- **Event deduplication**: Optional duplicate prevention for state events

## Dashboard Tabs

### Events Tab
**Widget**: RichLog (scrollable text)

Real-time log of all events with color-coded categories and rich formatting.

- **Shows**: Timestamp, category, topic, event details
- **Use**: Real-time debugging, event flow monitoring

### Table Tab
**Widget**: DataTable with TimeOrderedTableView

Tabular view of all events in descending timestamp order.

- **Shows**: Time, category, topic, details
- **Use**: Pattern recognition, category-focused debugging

### Tools Tab
**Widget**: DataTable with ToolExecutionView

Aggregated tool statistics (alphabetically sorted by tool name).

- **Shows**: Tool name, call count, avg time, success rate, last called
- **Use**: Performance analysis, reliability monitoring
- **Note**: Not time-ordered (alphabetical by tool name)

### Verticals Tab
**Widget**: Static with manual prepend

Vertical plugin integration events.

- **Shows**: Vertical name, action, config preview
- **Use**: Plugin debugging, integration testing

### History Tab
**Widget**: JSON file browser

Historical event file browser for loading and viewing past sessions.

- **Shows**: Loaded historical events in table format
- **Use**: Post-mortem analysis, session review

### Execution Tab
**Widget**: DataTable with TimeOrderedTableView

Execution lifecycle events in descending order.

- **Shows**: Time, type, operation, duration
- **Use**: Performance monitoring, lifecycle debugging

### Tool Calls Tab
**Widget**: DataTable with TimeOrderedTableView

Detailed tool call history in descending order.

- **Shows**: Time, tool name, status, duration, span ID, arguments
- **Use**: Detailed debugging, failure analysis, performance profiling
- **Note**: One row per call invocation (unlike Tools tab which aggregates)

### State Tab
**Widget**: DataTable with TimeOrderedTableView (with deduplication)

State machine transitions in descending order.

- **Shows**: Time, scope, key, old value, new value
- **Use**: State machine debugging, flow validation
- **Feature**: Prevents duplicate state events by ID

### Metrics Tab
**Widget**: Static text display

Aggregated performance statistics.

- **Shows**: Span counts, tool call metrics, timing data
- **Use**: Performance monitoring, resource optimization

## Architecture

### Event Ordering

All time-based tabs use a consistent `TimeOrderedTableView` base class:

```python
class TimeOrderedTableView(DataTable):
    """Base class for time-ordered event display.

    - Receives events in descending order (newest first) from file watcher
    - Appends to maintain order (newest at index 0)
    - Automatically trims oldest events
    - Rebuilds table on each update
    - Optional deduplication
    """
```text

**Views using TimeOrderedTableView**:
- EventTableView
- StateTransitionView
- ToolCallHistoryView
- ExecutionTraceView

### File Watcher

The dashboard uses `EventFileWatcher` to monitor the event file:

```python
# Location: victor/observability/dashboard/file_watcher.py

class EventFileWatcher:
    """Watches ~/.victor/metrics/victor.jsonl for new events.

    - Loads last 100 events at startup (in descending order)
    - Polls for new lines every 0.1 seconds
    - Processes events in descending order (newest first)
    - Emits to EventBus for distribution to views
    """
```

### Event Processing Flow

```text
JSONL File (ascending)
    ↓
EventFileWatcher (reverses to descending)
    ↓
EventBus.publish()
    ↓
Dashboard._process_event()
    ↓
Views (append in descending order)
```

## Usage

### Basic Launch

```bash
# Launch dashboard
victor dashboard

# With debug logging
victor dashboard --log-level debug
```text

### Keyboard Navigation

- `Tab` / `Shift+Tab`: Switch between tabs
- `↑` `↓`: Navigate rows (DataTable views)
- `Enter`: View details
- `q`: Quit

### Viewing Live Events

Events appear automatically as the agent generates them. The dashboard:
1. Loads the last 100 events from the JSONL file at startup
2. Polls for new events every 0.1 seconds
3. Displays newest events at the top of each view
4. Trims oldest events when exceeding limits

## Configuration

### Event Limits

Each view has configurable max events:

```python
# In victor/observability/dashboard/app.py

EventTableView(max_rows=500)
StateTransitionView(max_rows=200, enable_dedup=True)
ToolCallHistoryView(max_rows=200)
ExecutionTraceView(max_rows=300)
```

### Log Levels

```bash
# Debug logging for event processing
victor dashboard --log-level debug

# View logs
tail -f ~/.victor/logs/victor.log
```text

## Implementation Details

### TimeOrderedTableView Pattern

All time-based DataTable views inherit from `TimeOrderedTableView`:

```python
class MyEventView(TimeOrderedTableView):
    def __init__(self, *args, max_rows: int = 200, **kwargs):
        super().__init__(*args, max_rows=max_rows, **kwargs)

    def on_mount(self) -> None:
        self.add_columns("Time", "Type", "Details")
        self.cursor_type = "row"

    def add_my_event(self, event: Event) -> None:
        if not event.topic.startswith("my_prefix."):
            return
        self.add_event(event)  # Base class handles ordering/trimming

    def _format_event_row(self, event: Event) -> tuple:
        # Return tuple matching columns
        return (
            event.datetime.strftime("%H:%M:%S"),
            event.data.get("type", "unknown"),
            str(event.data.get("details", ""))[:50],
        )
```

### Key Behaviors

1. **Descending Order**: All time-based views show newest at top
2. **Automatic Trimming**: Oldest events removed when exceeding max_rows
3. **Rebuild Pattern**: Table cleared and rebuilt on each event (simple but correct)
4. **Optional Deduplication**: StateTransitionView prevents duplicate state events
5. **Error Handling**: Format errors logged but don't crash dashboard

## Troubleshooting

### Events Not Appearing

**Check**:
1. Is the event file being written? `ls -la ~/.victor/metrics/victor.jsonl`
2. Is the dashboard running? `ps aux | grep victor`
3. Are events being generated? Check logs: `tail -f ~/.victor/logs/victor.log`

### Wrong Display Order

**Verify**: All time-based tabs should show newest at top, oldest at bottom.

Check logs for event order:
```text
grep "FIRST event at" ~/.victor/logs/victor.log
# Should show descending timestamps (newest first)
```

### High Memory Usage

**Adjust limits**: Reduce max_rows in view constructors.

```python
# In app.py
EventTableView(max_rows=200)  # Reduced from 500
```text

## Testing

```bash
# Unit tests
pytest tests/unit/agent/test_continuation_loop_fix.py -v

# Manual testing
victor dashboard
# In another terminal: run a task that generates events
```

## Code Quality

- **Ruff**: `ruff check victor/observability/dashboard/` ✅ Passes
- **Black**: `black victor/observability/dashboard/` ✅ Formatted
- **Type Hints**: Partial (gradual typing)
- **Logging**: Comprehensive debug logging for troubleshooting

## Limitations

- **TUI Only**: Terminal-based interface (not web)
- **Single Session**: Views events from one agent run
- **No Historical Search**: File browser loads full file (no search/filter)
- **Manual Refresh**: Relies on file polling (not push notifications)

## Future Improvements

Potential enhancements:
- [ ] Search/filter functionality across all tabs
- [ ] Export events to JSON/CSV
- [ ] Customizable column layouts
- [ ] Event correlation across views
- [ ] Historical event replay

---

**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
