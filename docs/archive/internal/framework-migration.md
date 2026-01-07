# Framework Migration Guide

This guide explains how to migrate from direct `AgentOrchestrator` usage to the unified Framework API, and how to use the `FrameworkShim` for backward compatibility.

## Overview

Victor's Framework Evolution (Phases 1-6) introduced a layered architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     User Code / CLI                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────────┐    ┌────────────────────────────────────┐ │
│   │   Framework API     │    │     FrameworkShim (Adapter)         │ │
│   │   (Agent.create)    │    │     (Legacy CLI → Framework)        │ │
│   └──────────┬──────────┘    └───────────────┬────────────────────┘ │
│              │                                │                      │
│              └────────────┬───────────────────┘                      │
│                           ▼                                          │
│              ┌────────────────────────┐                              │
│              │   AgentOrchestrator    │                              │
│              │   (Core Engine)        │                              │
│              └────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Migration Paths

### Path 1: Framework API (Recommended)

For new code, use the Framework API directly:

```python
from victor.framework import Agent, ToolSet

# Simple creation
agent = await Agent.create(provider="anthropic")
result = await agent.run("Write a hello world function")

# With configuration
agent = await Agent.create(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    tools=ToolSet.default(),
    thinking=True,
)

# Streaming
async for event in agent.stream("Refactor this module"):
    if event.type == EventType.CONTENT:
        print(event.content, end="")
```

### Path 2: FrameworkShim (Migration Bridge)

For existing code using `AgentOrchestrator.from_settings()`, the `FrameworkShim` provides framework features without code changes:

```python
# Before: Direct orchestrator (no observability)
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = await AgentOrchestrator.from_settings(settings, "default")

# After: FrameworkShim (with observability)
from victor.framework.shim import FrameworkShim

shim = FrameworkShim(settings, profile_name="default")
orchestrator = await shim.create_orchestrator()
```

### Path 3: Legacy Mode (Backward Compatibility)

If you need to bypass framework features entirely:

```bash
# CLI with legacy flag
victor chat --legacy "Your prompt here"
```

```python
# Python with legacy orchestrator creation
orchestrator = await AgentOrchestrator.from_settings(
    settings,
    profile_name="default",
    thinking=False,
)
```

## FrameworkShim Usage

### Basic Usage

```python
from victor.framework.shim import FrameworkShim

# Minimal - inherits all framework features
shim = FrameworkShim(settings)
orchestrator = await shim.create_orchestrator()
```

### With Vertical

```python
from victor.framework.shim import FrameworkShim
from victor.verticals import CodingAssistant

shim = FrameworkShim(
    settings,
    profile_name="default",
    vertical=CodingAssistant,  # Apply vertical configuration
)
orchestrator = await shim.create_orchestrator()
```

### With Observability Control

```python
# Enable observability (default)
shim = FrameworkShim(
    settings,
    enable_observability=True,
    session_id="my-session-123",  # Custom session ID
)

# Disable observability
shim = FrameworkShim(
    settings,
    enable_observability=False,
)
```

### Session Lifecycle Events

```python
import time

shim = FrameworkShim(settings, enable_observability=True)
orchestrator = await shim.create_orchestrator()

# Emit session start
shim.emit_session_start({
    "mode": "interactive",
    "vertical": "coding",
})

start_time = time.time()
tool_calls = 0

try:
    # ... run your agent loop ...
    tool_calls = orchestrator.tool_call_count

finally:
    # Emit session end
    shim.emit_session_end(
        tool_calls=tool_calls,
        duration_seconds=time.time() - start_time,
        success=True,
    )
```

## CLI Options

The CLI now uses `FrameworkShim` by default. New options:

```bash
# Use a specific vertical
victor chat --vertical research "Research AI trends"
victor chat -V devops "Setup CI/CD pipeline"

# Disable observability
victor chat --no-observability "Quick question"

# Legacy mode (bypass FrameworkShim)
victor chat --legacy "Use old code path"

# List available verticals
victor chat --help  # Shows vertical options in help text
```

## API Comparison

| Feature | Direct Orchestrator | FrameworkShim | Framework API |
|---------|---------------------|---------------|---------------|
| Observability | Manual wiring | Automatic | Automatic |
| Verticals | Manual config | `vertical=` param | `vertical=` param |
| Session events | Not available | `emit_session_*` | Built-in |
| CQRS bridge | Manual | `enable_cqrs_bridge=` | Automatic |
| Backward compat | N/A | Full | Full |

## Observability Integration

FrameworkShim automatically wires `ObservabilityIntegration`:

```python
# Access observability after orchestrator creation
obs = shim.observability

# Subscribe to events
from victor.observability import EventCategory

obs.event_bus.subscribe(
    EventCategory.TOOL,
    lambda e: print(f"Tool: {e.name}")
)
```

## Vertical Configuration

When a vertical is specified, FrameworkShim applies:

1. **System Prompt** - Domain-specific instructions
2. **Tool Filter** - Tools stored in `orchestrator._framework_tools`
3. **Stage Config** - Stages stored in `orchestrator._vertical_stages`

```python
from victor.verticals import DevOpsAssistant

shim = FrameworkShim(settings, vertical=DevOpsAssistant)
orchestrator = await shim.create_orchestrator()

# Access applied configuration
print(orchestrator._framework_tools)  # ["docker", "shell", ...]
print(orchestrator._vertical_stages)  # {"ASSESSMENT": ..., ...}
```

## Migration Checklist

- [ ] Replace `AgentOrchestrator.from_settings()` with `FrameworkShim`
- [ ] Add session lifecycle events (`emit_session_start`, `emit_session_end`)
- [ ] Configure observability as needed
- [ ] Consider using verticals for domain-specific behavior
- [ ] Update CLI scripts to use new options (`--vertical`, `--observability`)
- [ ] Test with `--legacy` flag for backward compatibility

## Best Practices

1. **Use Framework API for new code** - Cleaner, more features
2. **Use FrameworkShim for migration** - Minimal changes, full features
3. **Use Legacy mode for debugging** - Isolate framework vs orchestrator issues
4. **Always emit session events** - Enables telemetry and debugging
5. **Choose appropriate verticals** - Optimizes prompts and tools for domain

## Troubleshooting

### "property 'observability' has no setter" Error

Ensure you're using the latest code. The `AgentOrchestrator` now has an `observability` setter.

### Vertical Not Found

```python
from victor.verticals import list_verticals, get_vertical

# List available verticals
print(list_verticals())  # ["coding", "research", "devops"]

# Case-insensitive lookup
vertical = get_vertical("CODING")  # Works
```

### Session ID Not Propagating

```python
# Pass session_id explicitly
shim = FrameworkShim(
    settings,
    session_id="my-custom-session-id",
    enable_observability=True,
)
```

## Related Documentation

- [VERTICALS.md](./VERTICALS.md) - Domain-specific assistants
- [STATE_MACHINE.md](./STATE_MACHINE.md) - State machine architecture
- [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) - Full API reference
