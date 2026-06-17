# Debugging and Profiling in Victor

This guide explains the different debugging and profiling tools available in Victor and when to use each one.

## Overview

Victor provides four distinct tools for debugging and profiling:

| Tool | Purpose | Use Case | Location |
|------|---------|----------|----------|
| **DebugLogger** | Runtime logging | Monitor agent execution in real-time | `victor.agent.debug_logger` |
| **AgentDebugger** | Post-execution analysis | Analyze completed agent runs | `victor.observability.debugger` |
| **PerformanceProfiler** | Business logic timing | Identify slow application operations | `victor.agent.performance_profiler` |
| **ProfilerManager** | System profiling | Deep-dive CPU/memory optimization | `victor.observability.profiler` |

---

## DebugLogger: Runtime Logging

**Purpose**: Real-time logging of agent execution for monitoring and debugging during development.

**When to Use**:
- Monitoring agent execution as it happens
- Debugging iteration-by-iteration behavior
- Tracking tool calls and results in real-time
- Watching context size growth
- Understanding conversation flow

**Key Features**:
- One-line log entries for scannability
- Iteration tracking with summaries
- Tool call logging with compact previews
- Context size warnings (at 75K and 150K characters)
- Conversation statistics (message counts, tool calls, elapsed time)

**Example**:
```python
from victor.agent.debug_logger import DebugLogger

logger = DebugLogger()
logger.log_iteration_start(iteration=1)
logger.log_tool_call("read_file", {"path": "main.py"}, iteration=1)
logger.log_tool_result("read_file", success=True, output="...", elapsed_ms=45.2)
logger.log_iteration_end(iteration=1, has_tool_calls=True)

# Output:
# ── ITER 1 ─────────────────────────────────────────────────────
#    ⚙️ read_file(path=main.py)
#    ✅ read_file: 1,234 chars (45ms)
#    → msgs=5 (12,345 chars) | tools=1 | iter=1 | 2.3s → tools
```

**API Reference**:
- `log_iteration_start(iteration, **context)` - Log iteration start
- `log_iteration_end(iteration, has_tool_calls, **context)` - Log iteration end
- `log_tool_call(tool_name, args, iteration)` - Log tool call
- `log_tool_result(tool_name, success, output, elapsed_ms)` - Log tool result
- `log_context_size(char_count, estimated_tokens)` - Log context size with warnings
- `log_limits(tool_budget, tool_calls_used, max_iterations, current_iteration, is_analysis_task)` - Log iteration/tool limits
- `log_conversation_summary(messages)` - Log conversation summary

---

## AgentDebugger: Post-Execution Analysis

**Purpose**: Analyze completed agent runs to understand what happened, why it failed, or what went slowly.

**When to Use**:
- Investigating why an agent failed
- Understanding the execution flow after completion
- Analyzing tool call patterns
- Identifying slow operations
- Reviewing state transitions

**Key Features**:
- Execution span trees (nested operations)
- Tool call history with timing
- State transition tracking
- Performance summaries
- Slow operation detection

**Example**:
```python
from victor.core.events import ObservabilityBus as EventBus
from victor.observability.debugger import AgentDebugger

event_bus = EventBus()
debugger = AgentDebugger(event_bus)

# After agent execution completes:
trace = debugger.get_execution_trace()
print(trace)

# Get tool calls
calls = debugger.get_tool_calls(tool_name="read_file")
for call in calls:
    print(f"{call['timestamp']}: {call['duration_ms']}ms")

# Get performance summary
summary = debugger.get_performance_summary()
print(f"Total time: {summary['total_duration_ms']}ms")
print(f"Slowest operation: {summary['slowest_operation']}")
```

**API Reference**:
- `get_execution_trace(agent_id=None, span_id=None)` - Get execution span tree
- `get_execution_spans(agent_id=None, span_type=None)` - Get spans as list
- `get_tool_calls(tool_name=None, agent_id=None)` - Get tool call history
- `get_state_transitions(scope=None, state_name=None)` - Get state transitions
- `get_performance_summary(agent_id=None)` - Get performance analysis

---

## PerformanceProfiler: Business Logic Timing

**Purpose**: Fine-grained timing instrumentation for application-level operations (tool execution, provider calls, internal processing).

**When to Use**:
- Timing specific operations (e.g., tool execution, provider calls)
- Building hierarchical span trees for nested operations
- Identifying slow business logic
- Generating timing reports for analysis
- Profiling custom code with decorators/context managers

**Key Features**:
- Hierarchical span tracking (parent/child relationships)
- Context manager and decorator support
- Category-based grouping (tool, provider, internal)
- Markdown report generation
- Per-tool breakdowns

**Example**:
```python
from victor.agent.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()

# Context manager
with profiler.span("tool_execution", category="tool", tool="read_file") as span:
    result = tool.execute(**args)
    span.add_metadata("file_size", len(result))

# Decorator
@profiler.profile("provider_call", category="provider")
async def call_provider(messages):
    return await provider.chat(messages)

# Get report
report = profiler.get_report()
print(report.to_markdown())

# Output:
# ## Performance Report
#
# ### tool_execution (2,450ms)
# - read_file: 200ms
# - grep_search: 400ms
# - write_file: 1,850ms
#
# ### provider_call (650ms)
# - time_to_first_token: 200ms
# - streaming_completion: 450ms
```

**API Reference**:
- `span(name, category="default", **metadata)` - Context manager for timing
- `profile(name, category="default")` - Decorator for automatic timing
- `get_report()` - Get profiling report
- `get_spans()` - Get all spans
- `reset()` - Reset profiler state

---

## ProfilerManager: System Profiling

**Purpose**: Deep-dive system-level performance analysis using Python's profiling tools (cProfile, memory_profiler, line_profiler).

**When to Use**:
- Optimizing CPU usage at the function level
- Analyzing memory allocations and leaks
- Line-by-line performance analysis
- Generating flame graphs
- Benchmarking function performance
- Comparing alternative implementations

**Key Features**:
- CPU profiling with cProfile
- Memory profiling with memory_profiler
- Line-by-line profiling with line_profiler
- Flame graph generation
- Call graph analysis
- Hot spot detection
- Benchmarking and comparison

**Example**:
```python
from victor.observability.profiler import get_profiler_manager, ProfilerType

manager = get_profiler_manager()

# Profile a function
result, profile = manager.profile_function(my_function, profiler_type=ProfilerType.CPU)
print(manager.format_report(profile))

# Profile a code block
with manager.profile(ProfilerType.CPU) as get_result:
    data = [i**2 for i in range(10000)]
result = get_result()

# Benchmark a function
benchmark = manager.benchmark(my_function, iterations=1000)
print(f"Mean time: {benchmark.mean_time*1000:.2f}ms")
print(f"Min: {benchmark.min_time*1000:.2f}ms")
print(f"Max: {benchmark.max_time*1000:.2f}ms")

# Compare functions
suite = manager.compare_functions([
    ("builtin_sum", lambda: sum(range(1000))),
    ("manual_sum", lambda: reduce(lambda a,b: a+b, range(1000))),
])
print(suite.summary())
```

**API Reference**:
- `profile_function(func, profiler_type=ProfilerType.CPU)` - Profile a function
- `profile(profiler_type)` - Context manager for profiling code blocks
- `benchmark(func, iterations=100)` - Benchmark a function
- `compare_functions(funcs)` - Compare multiple functions
- `format_report(profile)` - Format profile result for display
- `get_timing_stats(operation_name)` - Get timing statistics

---

## Decision Guide: Which Tool Should I Use?

### Question 1: What do you want to achieve?

**Monitor agent execution in real-time**
→ Use **DebugLogger**
```python
logger.log_iteration_start(iteration=1)
logger.log_tool_call("read_file", {...}, iteration=1)
```

**Analyze why an agent failed after it completed**
→ Use **AgentDebugger**
```python
trace = debugger.get_execution_trace()
calls = debugger.get_tool_calls()
```

**Time specific operations (tool execution, provider calls)**
→ Use **PerformanceProfiler**
```python
with profiler.span("tool_execution", category="tool"):
    result = tool.execute(**args)
```

**Optimize CPU/memory usage at the system level**
→ Use **ProfilerManager**
```python
result, profile = manager.profile_function(my_func, ProfilerType.CPU)
```

### Question 2: When do you need the information?

**During execution (real-time)**
→ DebugLogger or PerformanceProfiler

**After execution (post-mortem)**
→ AgentDebugger or ProfilerManager

### Question 3: What level of detail do you need?

**High-level (iterations, tool calls, conversation flow)**
→ DebugLogger

**Application-level (operation timing, spans)**
→ PerformanceProfiler

**System-level (CPU, memory, line-by-line)**
→ ProfilerManager

**Full execution trace with state transitions**
→ AgentDebugger

---

## Combining Multiple Tools

You can use multiple tools together for comprehensive analysis:

```python
from victor.agent.debug_logger import DebugLogger
from victor.agent.performance_profiler import PerformanceProfiler
from victor.observability.debugger import AgentDebugger
from victor.core.events import ObservabilityBus

# Setup
logger = DebugLogger()
profiler = PerformanceProfiler()
event_bus = ObservabilityBus()
debugger = AgentDebugger(event_bus)

# During execution
logger.log_iteration_start(iteration=1)

with profiler.span("tool_execution", category="tool"):
    result = tool.execute(**args)

logger.log_iteration_end(iteration=1, has_tool_calls=True)

# After execution
trace = debugger.get_execution_trace()
report = profiler.get_report()

# Use all three perspectives:
print("=== Runtime Logs ===")
print(logger.stats.summary())

print("\n=== Performance Report ===")
print(report.to_markdown())

print("\n=== Execution Trace ===")
print(trace)
```

---

## Best Practices

### DebugLogger
- Enable during development, disable in production (via `enabled=False`)
- Use INFO level for key events visible by default
- Use DEBUG level for detailed internal state
- Keep log messages scannable (one-line format)

### AgentDebugger
- Initialize once per agent session
- Call analysis methods after execution completes
- Use span_id for specific execution traces
- Filter by agent_id for multi-agent scenarios

### PerformanceProfiler
- Profile specific operations, not entire runs
- Use categories to group related spans
- Add metadata for context (file sizes, batch sizes, etc.)
- Reset profiler between runs to avoid memory buildup

### ProfilerManager
- Use for optimization, not production monitoring
- Profile CPU first, then memory (optimization order)
- Use line profiler for hot spots identified by CPU profiler
- Compare alternatives before refactoring

---

## Performance Considerations

| Tool | Runtime Overhead | Memory Overhead | Recommendation |
|------|-----------------|-----------------|----------------|
| DebugLogger | Low (string formatting) | Low (stats dict) | Always enable in dev |
| AgentDebugger | Low (event bus) | Medium (trace storage) | Enable for debugging |
| PerformanceProfiler | Low (timing only) | Low (span objects) | Use selectively |
| ProfilerManager | High (cProfile/memory_profiler) | High (profiling data) | Development only |

**Guideline**: DebugLogger and PerformanceProfiler have low overhead and can be used in development. AgentDebugger adds moderate overhead. ProfilerManager adds significant overhead and should only be used for targeted optimization.

---

## See Also

- **Event System**: `victor.core.events.ObservabilityBus` - Event bus for AgentDebugger
- **Tracing**: `victor.observability.tracing` - Low-level tracing primitives
- **Metrics**: `victor.agent.metrics_collector` - Metrics collection (separate from profiling)
- **Stream Metrics**: `victor.agent.stream_handler.StreamMetrics` - Token usage and streaming metrics
