# Victor Observability System

Complete observability and debugging system for Victor framework with real-time dashboard.

## Features

- **Real-time Event Tracking**: Monitor all agent activities as they happen
- **9 Dashboard Tabs**: Comprehensive views for different aspects of execution
- **Modular Emitters**: Focused event emitters for different event categories
- **Unified API**: Simple facade pattern for easy integration
- **Production Ready**: Extensive test coverage (62 tests, 100% pass rate)

## Dashboard Tabs

The Victor dashboard provides 9 different views for observability:

### 1. Events Tab (EventLogView)
Real-time log of all events flowing through the system.

**What it shows**:
- Timestamp
- Event category
- Event name
- Event data

**Use cases**:
- Real-time debugging
- Event flow monitoring
- System health checks

### 2. Table Tab (EventTableView)
Categorized table view of events.

**What it shows**:
- Events grouped by category (TOOL, MODEL, STATE, LIFECYCLE, ERROR)
- Filterable by event type
- Sorted chronologically

**Use cases**:
- Event filtering by type
- Category-focused debugging
- Pattern recognition

### 3. Tools Tab (ToolExecutionView)
Aggregated tool performance statistics.

**What it shows**:
- Tool name
- Total calls
- Average execution time
- Success rate
- Last called timestamp

**Use cases**:
- Performance analysis
- Tool usage patterns
- Reliability monitoring

**Key difference from Tool Calls**: This is **aggregated statistics** (one row per tool type).

### 4. Verticals Tab (VerticalTraceView)
Vertical plugin integration traces.

**What it shows**:
- Vertical-specific events
- Plugin loading status
- Vertical lifecycle

**Use cases**:
- Vertical plugin debugging
- Integration testing
- Plugin performance

### 5. History Tab (HistoryView)
Historical event replay and session history.

**What it shows**:
- Past events with timestamps
- Session-based grouping
- Chronological event replay

**Use cases**:
- Post-mortem analysis
- Session review
- Historical debugging

### 6. Execution Tab (ExecutionTraceView)
Execution span trees and lifecycle tracking.

**What it shows**:
- Session start/end
- Execution duration
- Lifecycle events

**Use cases**:
- Performance monitoring
- Session analysis
- Lifecycle debugging

### 7. Tool Calls Tab (ToolCallHistoryView)
Detailed tool call history with execution context.

**What it shows**:
- Timestamp
- Tool name
- Status (OK/FAIL)
- Duration
- Span ID (links to execution trace)
- Arguments preview

**Use cases**:
- Detailed call debugging
- Failure analysis
- Performance profiling

**Key difference from Tools**: This is **detailed history** (one row per call invocation).

### 8. State Tab (StateTransitionView)
State machine transition tracking.

**What it shows**:
- Old state → New state
- Transition confidence
- Transition timestamp
- Metadata

**Use cases**:
- State machine debugging
- Transition analysis
- Flow validation

### 9. Metrics Tab (PerformanceMetricsView)
Aggregated performance metrics and statistics.

**What it shows**:
- Tool performance metrics
- Model token usage
- Success rates
- Average/Min/Max durations

**Use cases**:
- Performance monitoring
- Resource optimization
- Capacity planning

## Quick Start

### 1. Launch the Dashboard

```bash
python -m victor.observability.dashboard.app
```

The dashboard will open in your terminal showing all 9 tabs.

### 2. Run the Demo

In a separate terminal:

```bash
python scripts/demo_observability.py
```

This will emit live events that appear in the dashboard in real-time.

### 3. View the Events

Switch between tabs to see different views:
- Press `Tab` to switch between tabs
- Press `q` to quit the dashboard
- Press `Ctrl+C` to stop the demo

## Architecture

### Event Flow

```
Orchestrator/Agent
    ↓
ObservabilityBridge (Facade)
    ↓
Event Emitters (Tool, Model, State, Lifecycle, Error)
    ↓
EventBus (Pub/Sub)
    ↓
Dashboard Views (9 tabs)
```

### Components

#### 1. ObservabilityBridge
**Location**: `victor/observability/bridge.py`

Unified facade for all observability operations.

```python
from victor.observability.bridge import ObservabilityBridge

bridge = ObservabilityBridge.get_instance()

# Emit events
bridge.tool_start("read_file", {"path": "file.txt"})
bridge.model_request("anthropic", "claude-3-5-sonnet-20250929", 1000)
bridge.state_transition("thinking", "tool_execution", 0.85)
```

#### 2. Event Emitters
**Location**: `victor/observability/emitters/`

Modular emitters for different event categories:

- **ToolEventEmitter**: Tool execution tracking
- **ModelEventEmitter**: LLM interaction tracking
- **StateEventEmitter**: State transition tracking
- **LifecycleEventEmitter**: Session lifecycle tracking
- **ErrorEventEmitter**: Error tracking

#### 3. EventBus
**Location**: `victor/observability/event_bus.py`

Thread-safe pub/sub event distribution system.

```python
from victor.observability.event_bus import EventBus

bus = EventBus.get_instance()

# Subscribe to events
def handle_event(event):
    print(f"Event: {event.name}")

bus.subscribe_all(handle_event)
```

## Integration Guide

### For Agent Developers

Add observability to your agents:

```python
from victor.observability.bridge import ObservabilityBridge

class MyAgent:
    def __init__(self):
        self._observability = ObservabilityBridge.get_instance()
        self._session_id = None

    async def execute(self, task: str):
        # Start session
        self._session_id = f"session-{uuid.uuid4().hex[:8]}"
        self._observability.session_start(
            self._session_id,
            agent_id=self.id,
            model=self.model,
        )

        try:
            # Execute task
            result = await self._process(task)
            return result
        finally:
            # End session
            self._observability.session_end(self._session_id)
```

### For Tool Developers

Track tool execution:

```python
from victor.observability.bridge import ObservabilityBridge

bridge = ObservabilityBridge.get_instance()

# Using context manager
with bridge.track_tool("read_file", {"path": "file.txt"}):
    result = read_file("file.txt")
# Events emitted automatically!

# Or manually
bridge.tool_start("read_file", {"path": "file.txt"})
try:
    result = read_file("file.txt")
    bridge.tool_end("read_file", 150.0, result=result)
except Exception as e:
    bridge.tool_failure("read_file", 50.0, e)
```

### For Custom Events

Emit custom events with metadata:

```python
bridge.tool_start(
    "my_tool",
    {"arg": "value"},
    agent_id="agent-1",
    session_id="session-123",
    custom_field="custom_value",
)
```

## Testing

### Run All Tests

```bash
# Unit tests
pytest tests/unit/observability/

# Integration tests
pytest tests/integration/test_dashboard_integration.py

# All tests
pytest tests/unit/observability/ tests/integration/test_dashboard_integration.py -v
```

### Test Coverage

- **38 unit tests** for emitters and bridge
- **24 integration tests** for dashboard tabs
- **Total**: 62 tests, 100% pass rate

## Session ID Format

Sessions use the format: `{repo_short}-{timestamp_base62}`

Example: `glm-bra-1a2b3c`

- **repo_short**: First 6 chars of repository/directory name
- **timestamp_base62**: First 6 chars of base62-encoded timestamp

Benefits:
- Project traceability
- Sequential ordering
- Human-readable format
- Unique across projects

## SOLID Principles

The observability system follows SOLID principles:

- **SRP**: Each emitter handles one event category
- **OCP**: Extensible via Protocol interfaces
- **LSP**: All emitters implement substitutable protocols
- **ISP**: Focused protocols per emitter type
- **DIP**: Depends on EventBus abstraction, not concrete implementations

## Design Patterns

- **Facade Pattern**: ObservabilityBridge simplifies complex subsystem
- **Protocol Pattern**: Type-safe interfaces with `typing.Protocol`
- **Singleton Pattern**: Single bridge instance
- **Context Managers**: Automatic tracking
- **Pub/Sub Pattern**: EventBus for event distribution

## Troubleshooting

### Dashboard Not Showing Events

**Problem**: Dashboard starts but no events appear.

**Solution**:
1. Check that events are being emitted:
   ```python
   from victor.observability.bridge import ObservabilityBridge
   bridge = ObservabilityBridge.get_instance()
   bridge.tool_start("test", {"arg": "value"})  # Should appear in dashboard
   ```

2. Check EventBus is running:
   ```python
   from victor.observability.event_bus import EventBus
   bus = EventBus.get_instance()
   print(f"Subscribers: {len(bus._subscribers)}")
   ```

3. Verify observability is enabled in orchestrator:
   - Check logs for "Observability enabled" message
   - Ensure no "Failed to initialize observability bridge" warnings

### High Memory Usage

**Problem**: Dashboard consumes too much memory over time.

**Solution**: The dashboard has built-in event limits. Adjust if needed:

```python
# In victor/observability/dashboard/app.py
MAX_EVENTS = 1000  # Reduce from default
```

### Missing Events

**Problem**: Some events don't appear in the dashboard.

**Solution**: Check event category filtering:
- Events are categorized (TOOL, MODEL, STATE, etc.)
- Each tab only shows specific categories
- Use the "Events" tab to see all events regardless of category

## API Reference

### ObservabilityBridge

```python
class ObservabilityBridge:
    """Unified facade for Victor observability system."""

    @classmethod
    def get_instance(cls) -> ObservabilityBridge:
        """Get singleton bridge instance."""
        pass

    # Tool events
    def tool_start(self, tool_name: str, arguments: Dict[str, Any], **metadata) -> None
    def tool_end(self, tool_name: str, duration_ms: float, result: Any, **metadata) -> None
    def tool_failure(self, tool_name: str, duration_ms: float, error: Exception, **metadata) -> None
    def track_tool(self, tool_name: str, arguments: Dict[str, Any], **metadata) -> ContextManager

    # Model events
    def model_request(self, provider: str, model: str, prompt_tokens: int, **metadata) -> None
    def model_response(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int, latency_ms: float, **metadata) -> None
    def model_streaming_delta(self, provider: str, model: str, delta: str, **metadata) -> None
    def model_error(self, provider: str, model: str, error: Exception, **metadata) -> None

    # State events
    def state_transition(self, old_stage: str, new_stage: str, confidence: float, **metadata) -> None

    # Lifecycle events
    def session_start(self, session_id: str, **metadata) -> None
    def session_end(self, session_id: Optional[str] = None, **metadata) -> None
    def track_session(self, session_id: str, **metadata) -> ContextManager

    # Error events
    def error(self, error: Exception, recoverable: bool, context: Optional[Dict] = None, **metadata) -> None

    # Control
    def enable(self) -> None
    def disable(self) -> None
    def is_enabled(self) -> bool
```

## Contributing

When adding new event types:

1. Create new emitter in `victor/observability/emitters/`
2. Add protocol to `victor/observability/emitters/base.py`
3. Add convenience methods to `ObservabilityBridge`
4. Add unit tests in `tests/unit/observability/test_emitters.py`
5. Add integration tests in `tests/integration/test_dashboard_integration.py`

## License

Apache License 2.0 - See LICENSE file for details
