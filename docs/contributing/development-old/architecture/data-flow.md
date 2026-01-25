# EventBus Backend Factory

> **Archived**: This document is kept for historical context and may be outdated. See `docs/contributing/index.md` for current guidance.


## Overview

Victor now has a **configurable EventBus backend system** that allows you to choose how events are stored and retrieved. This is especially useful for cross-process scenarios (e.g., agent and dashboard running in separate processes).

## Architecture

### EventBus Factory (`victor/observability/event_bus_factory.py`)

Central factory module that creates EventBus instances with different backends:

```python
from victor.observability.event_bus_factory import get_event_bus

# Get EventBus with configured backend (default: memory)
bus = get_event_bus()

# Override backend for specific use case
bus = get_event_bus("jsonl")  # Use JSONL file backend
bus = get_event_bus("persistent")  # Use persistent backend (Redis/SQLite)
```

### Backend Types

#### 1. **memory** (default)
- In-memory EventBus
- For same-process scenarios
- Events are lost when process exits
- No persistent storage

**Use cases:**
- Development
- Single-process scenarios
- Testing

#### 2. **jsonl** (file-based)
- In-memory EventBus + JSONL file exporter
- For cross-process communication
- Events persisted to `~/.victor/metrics/victor.jsonl`
- Dashboard can read historical events

**Use cases:**
- Dashboard in separate process from agent
- Historical event analysis
- Cross-process event streaming

**How it works:**
```
Agent Process: EventBus → JsonLineExporter → victor.jsonl
Dashboard Process: Read victor.jsonl → EventBus → Dashboard UI
```

#### 3. **persistent** (future)
- Redis or SQLite backend
- For production scenarios
- Currently falls back to JSONL backend

**Use cases:**
- Production deployments
- Distributed systems
- Multi-machine setups

## Configuration

### Settings (`victor/config/settings.py`)

```python
# EventBus Backend Type: memory | jsonl | persistent
# Default: "memory"
eventbus_backend: str = "memory"
```

### Environment Variable Override

```bash
# Override backend via environment variable
export VICTOR_EVENTBUS_BACKEND=jsonl
victor dashboard

# Or inline
VICTOR_EVENTBUS_BACKEND=jsonl victor dashboard
```

## Usage

### For Publishers (Emitters, Exporters)

**Before (hardcoded):**
```python
from victor.observability.event_bus import EventBus

bus = EventBus.get_instance()
bus.publish(event)
```

**After (configurable):**
```python
from victor.observability.event_bus_factory import get_event_bus

bus = get_event_bus()  # Respects settings.eventbus_backend
bus.publish(event)
```

### For Subscribers (Dashboard, etc.)

**Dashboard automatically uses configured backend:**
```python
from victor.observability.event_bus_factory import get_event_bus

class Dashboard:
    def on_mount(self):
        # Get EventBus from factory (respects settings.eventbus_backend)
        self._event_bus = get_event_bus()

        # Subscribe to events
        self._event_bus.subscribe_all(self.handle_event)
```

**When backend is `jsonl`:**
- Dashboard loads last 100 events from `~/.victor/metrics/victor.jsonl`
- Polls file every 1 second for new events
- Events are parsed and displayed in real-time

**When backend is `memory`:**
- Dashboard only sees events published after it starts
- No historical events available
- Real-time event streaming only

## Examples

### Example 1: In-Memory Backend (Default)

```python
from victor.observability.event_bus_factory import get_event_bus, reset_event_bus
from victor.observability.event_bus import EventCategory, VictorEvent

# Get in-memory EventBus (default)
bus = get_event_bus()

# Subscribe
def handler(event):
    print(f"Received: {event.category}/{event.name}")

bus.subscribe_all(handler)

# Publish event
event = VictorEvent(
    category=EventCategory.TOOL,
    name="test.event",
    data={"message": "Hello"}
)
bus.publish(event)
```

### Example 2: JSONL Backend (Cross-Process)

```python
from victor.observability.event_bus_factory import get_event_bus

# Get EventBus with JSONL backend
bus = get_event_bus("jsonl")

# Events are automatically written to ~/.victor/metrics/victor.jsonl
# Other processes can read this file independently
```

### Example 3: Dashboard with JSONL Backend

```bash
# Terminal 1: Start agent with JSONL backend
export VICTOR_EVENTBUS_BACKEND=jsonl
victor chat --no-tui "List Python files"

# Terminal 2: Start dashboard (automatically reads from JSONL)
export VICTOR_EVENTBUS_BACKEND=jsonl
victor dashboard --log-level DEBUG

# Dashboard will show:
# - Last 100 events from victor.jsonl
# - New events as they're written
# - Real-time updates every 1 second
```

## Testing

### Test EventBus Factory

```bash
python scripts/test_eventbus_factory.py
```

This tests:
- In-memory backend (default)
- JSONL backend
- Environment variable override

### Test Dashboard with Different Backends

```bash
# Test with in-memory backend (default)
victor dashboard

# Test with JSONL backend
export VICTOR_EVENTBUS_BACKEND=jsonl
victor dashboard --log-level DEBUG

# Generate test events in another terminal
victor chat --no-tui --log-events "List Python files"
```

## Implementation Details

### EventBus Factory Singleton

The factory maintains a singleton instance per backend:

```python
# Singleton instances
_event_bus_instance: Optional["EventBus"] = None
_event_bus_backend: Optional[EventBusBackend] = None

# Get or create instance
if _event_bus_instance is None or _event_bus_backend != backend:
    _event_bus_instance = _create_backend(backend)
    _event_bus_backend = backend
```

### JSONL Backend Implementation

The JSONL backend is implemented as:

1. **In-memory EventBus** (for pub/sub)
2. **JsonLineExporter** (writes events to file)
3. **File polling** (dashboard reads file for new events)

```python
def _create_jsonl_event_bus():
    # Create in-memory EventBus
    bus = EventBus.get_instance()

    # Add JSONL exporter
    exporter = JsonLineExporter(
        path="~/.victor/metrics/victor.jsonl",
        buffer_size=10,
        flush_interval_seconds=60,
    )
    bus.add_exporter(exporter)

    return bus
```

### Dashboard JSONL Support

The dashboard supports JSONL backend via:

1. **Initial load**: Reads last 100 events from file
2. **Polling**: Checks file every 1 second for new events
3. **Parsing**: Converts JSONL lines to VictorEvent objects
4. **Display**: Shows events in real-time

```python
def _setup_jsonl_source(self):
    # Load last 100 events
    with open(self._jsonl_path, "r") as f:
        lines = f.readlines()[-100:]
        for line in lines:
            event = self._parse_jsonl_line(line)
            self._process_event(event)

    # Start polling for new events
    self._poll_jsonl_file()
```

## Migration Guide

### For Existing Code

**Step 1: Update imports**

```python
# Before
from victor.observability.event_bus import EventBus

# After
from victor.observability.event_bus_factory import get_event_bus
```

**Step 2: Update EventBus access**

```python
# Before
bus = EventBus.get_instance()

# After
bus = get_event_bus()  # Now respects settings.eventbus_backend
```

### For Dashboard Users

**No changes needed!** The dashboard automatically uses the configured backend via `get_event_bus()`.

To use JSONL backend:
```bash
export VICTOR_EVENTBUS_BACKEND=jsonl
victor dashboard
```

## Benefits

1. **Flexibility**: Choose backend based on use case
2. **Cross-process**: JSONL backend enables dashboard in separate process
3. **Historical events**: JSONL backend allows dashboard to show past events
4. **No breaking changes**: In-memory backend is default (same behavior as before)
5. **Configuration**: Easy to switch via environment variable or settings
6. **Extensible**: New backends (Redis, SQLite, etc.) can be added easily

## Future Enhancements

- [ ] Redis backend implementation
- [ ] SQLite backend implementation
- [ ] Multi-process shared memory backend
- [ ] WebSocket-based real-time streaming
- [ ] Event filtering and querying for JSONL backend
- [ ] JSONL file rotation and cleanup
