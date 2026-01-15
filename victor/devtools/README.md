# Victor Developer Tools

Developer debugging and inspection tools for the Victor codebase.

## Overview

This package provides CLI tools to help developers understand, debug, and work with Victor's new architecture including protocols, events, dependency injection, and dependency graphs.

## Tools

### 1. Protocol Inspector (`protocol_inspector.py`)

Inspect all protocols in the Victor system.

**Features:**
- List all protocols with their methods
- Show detailed protocol information
- Find classes implementing protocols
- Find where protocols are used
- Export protocol information to JSON

**Usage:**

```bash
# List all protocols
python -m victor.devtools.protocol_inspector --list

# Show detailed protocol information
python -m victor.devtools.protocol_inspector --protocol IToolSelector --verbose

# Find implementations of a protocol
python -m victor.devtools.protocol_inspector --implementations IToolSelector

# Find where a protocol is used
python -m victor.devtools.protocol_inspector --usage IToolSelector

# Export to JSON
python -m victor.devtools.protocol_inspector --export protocols.json
```

**Examples:**

```bash
# List all protocols
$ python -m victor.devtools.protocol_inspector --list
Found 93 protocols:

  ArgumentNormalizerProtocol [victor.protocols.agent_tools]
  AutoCommitterProtocol [victor.protocols.agent_budget]
  ConversationControllerProtocol [victor.protocols.agent_conversation]
  ...

# Show details for a specific protocol
$ python -m victor.devtools.protocol_inspector --protocol IToolSelector --verbose

============================================================
Protocol: IToolSelector
============================================================
Module: victor.protocols.tool_selector
File: victor/protocols/tool_selector.py:24

Description:
Protocol for tool selection strategy...

Methods (7):
  - select_tools(context, available_tools, max_tools)
  - configure(config)
  - get_available_tools()
  ...

Implementations (3):
  - SemanticToolSelector [victor.tools.semantic_selector]
    victor/tools/semantic_selector.py:45
  - HybridToolSelector [victor.tools.hybrid_selector]
    victor/tools/hybrid_selector.py:67

Used in (12 locations):
  - victor/agent/tool_pipeline.py:123: def select_tools(...)
  - victor/agent/orchestrator.py:456: selector: IToolSelector
  ...
```

### 2. Event Bus Monitor (`event_monitor.py`)

Monitor events flowing through the Victor system in real-time.

**Features:**
- Subscribe to all events and print them
- Filter by event category or type
- Show event timing and frequency
- Export event streams to files
- Color-coded output by category

**Usage:**

```bash
# Monitor all events
python -m victor.devtools.event_monitor

# Filter by category
python -m victor.devtools.event_monitor --filter tool

# Filter by event type
python -m victor.devtools.event_monitor --event-type ToolExecutionEvent

# Monitor for specific duration
python -m victor.devtools.event_monitor --duration 30

# Monitor until N events received
python -m victor.devtools.event_monitor --event-count 100

# Export events to JSON
python -m victor.devtools.event_monitor --export events.json

# Verbose mode with full event data
python -m victor.devtools.event_monitor --verbose
```

**Event Categories:**
- `tool`: Tool execution events (green)
- `state`: State transition events (blue)
- `model`: Model inference events (yellow)
- `error`: Error events (red)
- `audit`: Audit events (magenta)
- `metric`: Metric events (cyan)
- `lifecycle`: Lifecycle events (white)

**Examples:**

```bash
# Monitor all tool events
$ python -m victor.devtools.event_monitor --filter tool
Starting Event Monitor...
Filter: category=tool, event_type=None
Duration: infinite

[2025-01-14T12:34:56.789] tool - ToolExecutionEvent
  Source: ToolPipeline
  Data:
    tool_name: code_search
    arguments: {"query": "protocol"}
    duration_ms: 234

[2025-01-14T12:34:57.123] tool - ToolCompletionEvent
  Source: ToolPipeline
  Data:
    tool_name: code_search
    success: true
    result_size: 1523

============================================================
Event Statistics
============================================================
Total Events: 127
Duration: 34.56 seconds
Event Rate: 3.67 events/sec

Top Event Types:
  ToolExecutionEvent: 45
  ToolCompletionEvent: 45
  StateTransitionEvent: 23
  ...
```

### 3. Dependency Visualizer (`dependency_viz.py`)

Visualize dependency relationships between modules.

**Features:**
- Show which components depend on which
- Identify circular dependencies
- Output as text-based graph or DOT format
- Analyze import dependencies
- Find most depended-upon modules

**Usage:**

```bash
# Show overview of all dependencies
python -m victor.devtools.dependency_viz

# Show dependencies for a specific module
python -m victor.devtools.dependency_viz --module victor.agent.orchestrator

# Show dependents (what depends on this module)
python -m victor.devtools.dependency_viz --module victor.protocols --show-dependents

# Find circular dependencies
python -m victor.devtools.dependency_viz --find-cycles

# Export to DOT format (for Graphviz)
python -m victor.devtools.dependency_viz --format dot > graph.dot

# Analyze specific directory
python -m victor.devtools.dependency_viz --directory victor/protocols
```

**Examples:**

```bash
# Show overview
$ python -m victor.devtools.dependency_viz

Dependency Graph Overview
============================================================
Total modules: 156
Total dependencies: 423
Avg dependencies per module: 2.71

Top 10 modules with most dependencies:
  victor.agent.orchestrator: 23
  victor.agent.tool_pipeline: 18
  victor.framework.workflow_engine: 15
  ...

Top 10 most depended-upon modules:
  victor.protocols.agent: 45
  victor.protocols.tool_selector: 32
  victor.core.container: 28
  ...

# Show circular dependencies
$ python -m victor.devtools.dependency_viz --find-cycles

Found 3 circular dependencies:

1. victor.agent.orchestrator -> victor.agent.conversation_controller -> victor.protocols.agent_conversation -> victor.agent.orchestrator
2. victor.framework.workflow_engine -> victor.agent.tool_pipeline -> victor.framework.workflow_engine
...
```

### 4. DI Container Inspector (`di_inspector.py`)

Inspect the dependency injection container state.

**Features:**
- Show all registered services
- Show service dependencies and lifecycle
- Identify singleton vs scoped services
- Help debug dependency resolution issues
- Detect circular dependencies

**Usage:**

```bash
# List all services
python -m victor.devtools.di_inspector --list

# Show singletons only
python -m victor.devtools.di_inspector --list --lifetime singleton

# Show details for a specific service
python -m victor.devtools.di_inspector --service ToolRegistry

# Show dependency graph
python -m victor.devtools.di_inspector --show-dependencies

# Check resolution order (detect cycles)
python -m victor.devtools.di_inspector --check-resolution

# Export to JSON
python -m victor.devtools.di_inspector --export container.json
```

**Service Lifetimes:**
- 🔷 **Singleton**: One instance for the entire application
- 🔸 **Scoped**: One instance per scope (e.g., per request)
- 🔹 **Transient**: New instance every time

**Examples:**

```bash
# List all services
$ python -m victor.devtools.di_inspector --list

Found 56 services:

  🔷 ArgumentNormalizerProtocol [singleton] ✓
  🔷 AutoCommitterProtocol [singleton] ✓
  🔷 ContextCompactorProtocol [singleton] ✗
  🔸 ConversationStateMachineProtocol [scoped] ✓
  🔷 IToolSelector [singleton] ✓
  ...

# Show details for a specific service
$ python -m victor.devtools.di_inspector --service IToolSelector

============================================================
Service: IToolSelector
============================================================
Type: IToolSelector
Implementation: ToolSelector
Lifetime: singleton
Status: Created

Dependencies (3):
  ✓ ToolRegistry
  ✓ Settings
  ✓ EmbeddingService

Depended on by (8):
  - ToolPipeline
  - OrchestratorFactory
  - ToolCoordinator
  ...

# Check resolution order
$ python -m victor.devtools.di_inspector --check-resolution

Checking service resolution order...

Valid resolution order (56 services):
  1. Settings
  2. ToolRegistry
  3. EmbeddingService
  4. IToolSelector
  5. ToolPipeline
  ...
```

## Common Use Cases

### Debugging Protocol Implementations

When you need to understand which classes implement a protocol:

```bash
# Find all implementations
python -m victor.devtools.protocol_inspector --implementations IToolSelector

# See where it's used
python -m victor.devtools.protocol_inspector --usage IToolSelector
```

### Understanding Module Dependencies

When refactoring or analyzing code structure:

```bash
# See what a module depends on
python -m victor.devtools.dependency_viz --module victor.agent.orchestrator

# Check for circular dependencies
python -m victor.devtools.dependency_viz --find-cycles

# Export graph for visualization
python -m victor.devtools.dependency_viz --format dot > graph.dot
dot -Tpng graph.dot -o graph.png
```

### Debugging DI Issues

When services aren't resolving correctly:

```bash
# Check if service is registered
python -m victor.devtools.di_inspector --list | grep MyService

# Show service details and dependencies
python -m victor.devtools.di_inspector --service MyService

# Check for circular dependencies
python -m victor.devtools.di_inspector --check-resolution
```

### Monitoring Event Flow

When debugging event-driven behavior:

```bash
# Monitor all events
python -m victor.devtools.event_monitor --verbose

# Filter to specific category
python -m victor.devtools.event_monitor --filter tool --duration 60

# Export for analysis
python -m victor.devtools.event_monitor --export events.json
```

## Development

### Adding New Tools

To add a new developer tool:

1. Create a new Python file in `victor/devtools/`
2. Implement a `main()` function that returns an exit code
3. Add appropriate argument parsing with `argparse`
4. Include `--help` documentation
5. Update this README

### Tool Guidelines

- **Runnable as modules**: All tools must support `python -m victor.devtools.tool_name`
- **Help flag**: Must include `--help` with usage examples
- **Logging**: Use `logging` module for debug output
- **Error handling**: Handle errors gracefully with clear messages
- **Docstrings**: Include Google-style docstrings
- **Type hints**: Use type hints on all public APIs

### Testing

Test tools manually before committing:

```bash
# Test each tool's help
python -m victor.devtools.protocol_inspector --help
python -m victor.devtools.event_monitor --help
python -m victor.devtools.dependency_viz --help
python -m victor.devtools.di_inspector --help

# Test basic functionality
python -m victor.devtools.protocol_inspector --list
python -m victor.devtools.dependency_viz
python -m victor.devtools.di_inspector --list
```

## Contributing

When adding features to existing tools:

1. Maintain backward compatibility
2. Update help text and examples
3. Add logging for debugging
4. Handle edge cases
5. Update this README

## License

Apache License 2.0 - See LICENSE file for details
