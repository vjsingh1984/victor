# Victor Dashboard Testing Guide

## Quick Test

The dashboard needs to receive events to display data. Here's how to test it:

### Method 1: Run the Demo Script

**Terminal 1** - Start the dashboard:
```bash
victor dashboard
```

**Terminal 2** - Run the comprehensive demo:
```bash
python scripts/demo_observability.py
```

This will emit a variety of events that populate all dashboard tabs.

### Method 2: Simple Live Test

**Terminal 1** - Start the dashboard:
```bash
victor dashboard
```

**Terminal 2** - Run the simple test:
```bash
python scripts/test_dashboard_live.py
```

This continuously emits test events until you stop it with Ctrl+C.

### Method 3: Within Victor Agent Execution

When you run a Victor agent normally, the orchestrator will automatically emit events to the dashboard:

```bash
# In one terminal, start the dashboard
victor dashboard

# In another terminal, run any Victor command
victor chat "Write a Python function to calculate fibonacci"
```

The dashboard will show all events from the agent execution in real-time.

## Expected Behavior

When events are being emitted, you should see:

### Dashboard Stats Bar (Top)
- Events counter increasing
- Tools counter increasing
- States counter increasing (if state changes occur)
- Errors counter (if errors occur)

### Events Tab
Real-time log of all events with timestamps and categories

### Table Tab
Categorized table of events

### Tools Tab
Aggregated tool statistics:
- Tool name
- Number of calls
- Average execution time
- Success rate
- Last called time

### Tool Calls Tab
Detailed history of each tool call:
- Time
- Tool name
- Status (OK/FAIL)
- Duration
- Span ID
- Arguments preview

### State Tab
State machine transitions if any occur

### Metrics Tab
Performance metrics aggregations

## Controls

- **q** - Quit dashboard
- **Tab** - Switch between tabs
- **Ctrl+P** - Pause/resume event stream
- **Ctrl+L** - Clear all events
- **Ctrl+R** - Refresh

## Troubleshooting

### Dashboard Shows No Events

**Problem**: All tabs are empty

**Solution 1**: Make sure events are being emitted
```bash
# Check if demo script is running
python scripts/demo_observability.py
```

**Solution 2**: Check if EventBus is working
```python
from victor.observability.event_bus import EventBus
bus = EventBus.get_instance()
print(f"Subscribers: {len(bus._subscribers)}")
```

**Solution 3**: Check observability is enabled in orchestrator
- Look for "Observability enabled" message in logs
- Check that no "Failed to initialize observability bridge" warnings appear

### 'q' Key Doesn't Work

**Problem**: Pressing 'q' doesn't quit

**Solution**: Use Ctrl+C instead (this always works)

Note: The 'q' binding has been added in the latest version. Make sure you have the latest code:
```bash
pip install -e .
```

### Dashboard Not Starting

**Problem**: `victor dashboard` command doesn't work

**Solution**: Install Victor in development mode:
```bash
cd /path/to/victor
pip install -e .
```

Then try:
```bash
victor dashboard
```

Or directly:
```bash
python -m victor.observability.dashboard.app
```

## Understanding the Tabs

### Tools vs Tool Calls (Common Question)

**Tools Tab** (Aggregated Statistics):
- One row per tool type
- Shows: calls, avg time, success rate
- Use for: Performance overview

**Tool Calls Tab** (Detailed History):
- One row per tool call invocation
- Shows: time, status, duration, span ID, arguments
- Use for: Detailed debugging

These are **complementary views**, not duplicates!

## Test Scenarios

### Scenario 1: Basic Tool Execution
Expected events:
- Session start
- Tool start
- Tool end
- Session end

Visible in:
- Events tab (all events)
- Table tab (categorized)
- Tool Calls tab (detailed)
- Metrics tab (aggregated)

### Scenario 2: Error Handling
Expected events:
- Tool start
- Tool failure
- Error event

Visible in:
- Events tab
- Table tab
- Tool Calls tab (status: FAIL)
- State tab (if applicable)

### Scenario 3: Multiple Tools
Expected events:
- Multiple tool start/end pairs
- Aggregated statistics

Visible in:
- Tools tab (statistics per tool)
- Metrics tab (aggregated metrics)
