# Execution Tracing Guide

This guide covers comprehensive execution tracing capabilities for Victor workflows.

## Overview

Execution tracing provides detailed visibility into workflow execution:

- **Node-level tracing**: Every node execution is logged
- **Input/output capture**: All data flowing through nodes
- **Performance metrics**: Execution time for each node
- **Tool call tracking**: All tool invocations and results
- **Error logging**: Detailed error information
- **Export formats**: JSON, CSV, HTML reports

## Quick Start

### Basic Tracing

Execute workflow with tracing:

```bash
victor workflow trace my_workflow.yaml --output trace.json
```

### View Trace Summary

```bash
victor workflow trace my_workflow.yaml
```

This displays a summary without saving the trace.

### Export in Different Formats

```bash
# JSON (default)
victor workflow trace my_workflow.yaml -o trace.json

# CSV
victor workflow trace my_workflow.yaml -o trace.csv --format csv

# HTML report
victor workflow trace my_workflow.yaml -o report.html --format html
```

## Trace Data

### What Gets Traced

Every trace includes:

**Workflow-level:**
- Workflow name and ID
- Start and end timestamps
- Total duration
- Input and output state
- Success/failure status

**Node-level:**
- Node ID and type
- Execution timestamp
- Input state
- Output state
- Duration
- Success/failure status
- Error messages (if failed)
- Tool calls made

**Tool-level:**
- Tool name
- Tool inputs
- Tool outputs
- Execution duration
- Success/failure status
- Error messages (if failed)

### Trace Structure

```json
{
  "trace_id": "abc-123-def",
  "workflow_name": "my_workflow",
  "start_time": 1640995200.0,
  "end_time": 1640995210.5,
  "duration_seconds": 10.5,
  "inputs": {"task": "fix bug"},
  "outputs": {"result": "success"},
  "metadata": {},
  "nodes": [
    {
      "node_id": "node_1",
      "node_type": "agent",
      "timestamp": 1640995200.0,
      "duration_seconds": 2.5,
      "success": true,
      "error": null,
      "inputs": {"task": "fix bug"},
      "outputs": {"analysis": "complete"},
      "tool_calls": [
        {
          "tool_name": "read_file",
          "duration_seconds": 0.5,
          "success": true,
          "error": null
        }
      ]
    }
  ]
}
```

## Programmatic Tracing

### Basic Usage

```python
from victor.workflows.trace import WorkflowTracer

tracer = WorkflowTracer()

# Start trace
tracer.start_trace(
    workflow_name="my_workflow",
    inputs={"task": "fix bug"},
    metadata={"env": "production"},
)

# Execute workflow...
# tracer.trace_node(...)
# tracer.trace_tool_call(...)

# End trace
trace = tracer.end_trace(outputs={"result": "success"})
```

### Tracing Nodes

```python
# Trace node execution
tracer.trace_node(
    node_id="agent_1",
    node_type="agent",
    inputs={"task": "fix bug"},
    outputs={"analysis": "complete"},
    duration_seconds=2.5,
    tool_calls=[...],
    success=True,
)
```

### Tracing Tool Calls

```python
# Trace tool call
tracer.trace_tool_call(
    tool_name="read_file",
    inputs={"path": "file.py"},
    outputs={"content": "def main():..."},
    duration_seconds=0.5,
    success=True,
)
```

### Auto-Tracing

Enable automatic tool call tracing:

```python
tracer = WorkflowTracer(auto_tool_calls=True)

# Tool calls will be automatically traced
# when tracing nodes
```

## Trace Analysis

### Performance Analysis

Analyze trace for performance insights:

```python
# Get trace
trace = tracer.get_trace()

# Analyze
analysis = tracer.analyze_trace(trace)

# View results
print(f"Total duration: {analysis['total_duration_seconds']:.2f}s")
print(f"Total nodes: {analysis['total_nodes']}")
print(f"Successful: {analysis['successful_nodes']}")
print(f"Failed: {analysis['failed_nodes']}")

print("\nSlowest nodes:")
for node in analysis['slowest_nodes']:
    print(f"  {node['node_id']}: {node['duration']:.2f}s")

print("\nMost used tools:")
for tool in analysis['most_used_tools']:
    print(f"  {tool['tool']}: {tool['calls']} calls")
```

### Identifying Bottlenecks

Find slow nodes:

```python
# Get all node durations
node_durations = [
    (node.node_id, node.duration_seconds)
    for node in trace.nodes
]

# Sort by duration
node_durations.sort(key=lambda x: x[1], reverse=True)

# Show top 5
print("Top 5 slowest nodes:")
for node_id, duration in node_durations[:5]:
    print(f"  {node_id}: {duration:.2f}s")
```

### Error Analysis

Find failed executions:

```python
# Get failed nodes
failed_nodes = [
    node for node in trace.nodes
    if not node.success
]

print(f"Failed nodes: {len(failed_nodes)}")
for node in failed_nodes:
    print(f"  {node.node_id}: {node.error}")
```

### Tool Usage Analysis

Analyze tool usage:

```python
from collections import Counter

# Count tool calls
tool_counts = Counter()
for node in trace.nodes:
    for tool_call in node.tool_calls:
        tool_counts[tool_call.tool_name] += 1

# Show results
print("Tool usage:")
for tool, count in tool_counts.most_common():
    print(f"  {tool}: {count} calls")
```

## Export Formats

### JSON Format

Default export format with full trace data:

```bash
victor workflow trace workflow.yaml -o trace.json
```

**Use cases:**
- Programmatic analysis
- Data processing
- Archiving
- Integration with other tools

### CSV Format

Tabular format for spreadsheet analysis:

```bash
victor workflow trace workflow.yaml -o trace.csv --format csv
```

**Columns:**
- node_id
- node_type
- timestamp
- duration_seconds
- success
- error
- tool_calls_count

**Use cases:**
- Spreadsheet analysis
- Data visualization
- Custom reporting
- Statistical analysis

### HTML Format

Human-readable report with visualizations:

```bash
victor workflow trace workflow.yaml -o report.html --format html
```

**Features:**
- Summary statistics
- Node execution table
- Color-coded status
- Performance metrics
- Browser-viewable

**Use cases:**
- Executive reports
- Code reviews
- Documentation
- Presentations

## Trace Queries

### Querying by Node

Get events for specific node:

```python
# Get node events
node_events = trace.get_events_by_node("agent_1")

for event in node_events:
    print(f"{event.event_type}: {event.timestamp}")
```

### Querying by Type

Get events of specific type:

```python
from victor.workflows.execution_engine import ExecutionEventType

# Get all failures
failed_events = trace.get_events_by_type(
    ExecutionEventType.NODE_FAILED
)

for event in failed_events:
    print(f"Failed node: {event.node_id}")
    print(f"Error: {event.error}")
```

### Filtering Traces

Filter trace data:

```python
# Get failed nodes
failed_nodes = [
    node for node in trace.nodes
    if not node.success
]

# Get slow nodes (> 1 second)
slow_nodes = [
    node for node in trace.nodes
    if node.duration_seconds > 1.0
]

# Get nodes with tool calls
nodes_with_tools = [
    node for node in trace.nodes
    if node.tool_calls
]
```

## Trace Comparison

### Comparing Traces

Compare two executions:

```python
# Get two traces
trace_1 = tracer.get_trace_by_id("trace_1")
trace_2 = tracer.get_trace_by_id("trace_2")

# Compare durations
duration_1 = trace_1.end_time - trace_1.start_time
duration_2 = trace_2.end_time - trace_2.start_time

print(f"Duration 1: {duration_1:.2f}s")
print(f"Duration 2: {duration_2:.2f}s")
print(f"Difference: {duration_2 - duration_1:.2f}s")
```

### Regression Detection

Detect performance regressions:

```python
# Get baseline trace
baseline = tracer.get_trace_by_id("baseline")

# Get current trace
current = tracer.get_trace_by_id("current")

# Compare node durations
for node in current.nodes:
    baseline_node = next(
        (n for n in baseline.nodes if n.node_id == node.node_id),
        None
    )

    if baseline_node:
        slowdown = node.duration_seconds - baseline_node.duration_seconds
        if slowdown > 0.5:  # 500ms slowdown
            print(f"{node.node_id} slowed by {slowdown:.2f}s")
```

## Trace Storage

### Automatic Storage

Traces are automatically stored in:

```
~/.victor/workflow_traces/
├── workflow_1_abc123.json
├── workflow_1_def456.json
└── workflow_2_ghi789.json
```

### Manual Storage

Save traces to custom location:

```python
import json

# Get trace
trace = tracer.get_trace()

# Save to custom path
output_path = Path("/tmp/my_trace.json")
output_path.write_text(json.dumps(trace.to_dict(), indent=2))
```

### Trace Management

Manage trace files:

```python
# List all traces
traces = tracer.get_trace_history()
print(f"Total traces: {len(traces)}")

# Get recent traces
recent = tracer.get_trace_history()[-10:]

# Clean old traces
# (manual implementation)
import os
trace_dir = Path.home() / ".victor" / "workflow_traces"
for trace_file in trace_dir.glob("*.json"):
    # Delete old traces
    if trace_file.stat().st_mtime < time.time() - 86400 * 30:  # 30 days
        trace_file.unlink()
```

## Advanced Tracing

### Custom Trace Handlers

Implement custom trace handling:

```python
from victor.workflows.trace import WorkflowTracer

class CustomTracer(WorkflowTracer):
    def on_node_complete(self, node_record):
        # Custom handling
        if node_record.duration_seconds > 5.0:
            print(f"WARNING: Slow node {node_record.node_id}")
            # Send alert, log to system, etc.

    def on_tool_call(self, tool_record):
        # Track tool usage
        self.tool_usage[tool_record.tool_name] += 1

tracer = CustomTracer()
```

### Real-time Monitoring

Monitor execution in real-time:

```python
async def monitor_workflow(workflow, inputs):
    tracer = WorkflowTracer()
    tracer.start_trace("workflow", inputs)

    async for event in executor.execute_stream(workflow, inputs):
        # Process event in real-time
        if event.event_type == "node_completed":
            print(f"{event.node_id} completed in {event.duration_seconds:.2f}s")

        elif event.event_type == "node_failed":
            print(f"{event.node_id} failed: {event.error}")

    tracer.end_trace()
    return tracer.get_trace()
```

### Distributed Tracing

Trace distributed workflows:

```python
tracer = WorkflowTracer()

# Add trace context to state
trace_id = str(uuid4())
state = {
    "trace_id": trace_id,
    "trace_parent": "parent_workflow",
}

# Trace propagates to sub-workflows
tracer.start_trace("child_workflow", state)
```

## Best Practices

### 1. Use Descriptive Metadata

Add context to traces:

```python
tracer.start_trace(
    "my_workflow",
    inputs={...},
    metadata={
        "environment": "production",
        "version": "1.2.3",
        "run_id": "deploy_456",
    }
)
```

### 2. Export Regularly

Save traces for long-term analysis:

```python
# After execution
trace = tracer.end_trace()

# Export with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = Path(f"traces/trace_{timestamp}.json")
tracer.export_trace(output_path)
```

### 3. Analyze Patterns

Look for patterns across traces:

```python
traces = tracer.get_trace_history()

# Find consistently slow nodes
node_times = {}
for trace in traces:
    for node in trace.nodes:
        if node.node_id not in node_times:
            node_times[node.node_id] = []
        node_times[node.node_id].append(node.duration_seconds)

# Find averages
for node_id, times in node_times.items():
    avg = sum(times) / len(times)
    if avg > 2.0:
        print(f"{node_id}: avg {avg:.2f}s")
```

### 4. Monitor Errors

Track error patterns:

```python
# Get all errors
errors = []
for trace in traces:
    for node in trace.nodes:
        if not node.success and node.error:
            errors.append({
                "node": node.node_id,
                "error": node.error,
                "timestamp": node.timestamp,
            })

# Analyze error patterns
from collections import Counter
error_messages = Counter(e["error"] for e in errors)

print("Common errors:")
for error, count in error_messages.most_common(5):
    print(f"  {error}: {count}x")
```

### 5. Use HTML Reports for Reviews

Generate reports for code reviews:

```bash
# Generate HTML report
victor workflow trace workflow.yaml \
  --format html \
  --output review.html \
  --context '{"review_id": "PR-123"}'

# Share with team
```

## Troubleshooting

### Trace File Not Created

**Possible causes:**
1. Output directory doesn't exist
2. Write permissions
3. Invalid file path

**Solutions:**
```bash
# Check directory exists
ls -la ~/.victor/workflow_traces/

# Create if needed
mkdir -p ~/.victor/workflow_traces/

# Check permissions
chmod 755 ~/.victor/workflow_traces/
```

### Trace Missing Data

**Possible causes:**
1. Tracing not enabled
2. Nodes not executed
3. Trace not finalized

**Solutions:**
```python
# Ensure tracing enabled
tracer = WorkflowTracer(auto_tool_calls=True)

# Start trace
tracer.start_trace(...)

# Execute workflow...

# End trace to finalize
trace = tracer.end_trace()
```

### Large Trace Files

**Possible causes:**
1. Many iterations
2. Large state data
3. Long execution

**Solutions:**
```python
# Filter trace data before export
summary_trace = {
    "trace_id": trace.trace_id,
    "workflow_name": trace.workflow_name,
    "duration_seconds": trace.end_time - trace.start_time,
    "nodes": [
        {
            "node_id": node.node_id,
            "duration_seconds": node.duration_seconds,
            "success": node.success,
        }
        for node in trace.nodes
    ],
}

# Export summary
with open("summary.json", "w") as f:
    json.dump(summary_trace, f, indent=2)
```

## Examples

### Example 1: Performance Profiling

```python
tracer = WorkflowTracer()
tracer.start_trace("workflow", inputs)

# Execute workflow...
result = await executor.execute(workflow, inputs)

trace = tracer.end_trace()

# Find bottlenecks
analysis = tracer.analyze_trace(trace)
print("Bottlenecks:")
for node in analysis['slowest_nodes'][:3]:
    print(f"  {node['node_id']}: {node['duration']:.2f}s")
```

### Example 2: Error Tracking

```python
tracer = WorkflowTracer()
tracer.start_trace("workflow", inputs)

# Execute workflow...
try:
    result = await executor.execute(workflow, inputs)
except Exception as e:
    trace = tracer.end_trace()

    # Analyze errors
    failed_nodes = [n for n in trace.nodes if not n.success]
    print(f"Failed nodes: {len(failed_nodes)}")
    for node in failed_nodes:
        print(f"  {node.node_id}: {node.error}")
```

### Example 3: Regression Testing

```python
# Run baseline
baseline_tracer = WorkflowTracer()
baseline_tracer.start_trace("workflow", inputs)
result = await executor.execute(workflow, inputs)
baseline_trace = baseline_tracer.end_trace()

# Make code changes...

# Run comparison
compare_tracer = WorkflowTracer()
compare_tracer.start_trace("workflow", inputs)
result = await executor.execute(workflow, inputs)
compare_trace = compare_tracer.end_trace()

# Compare
for node in compare_trace.nodes:
    baseline_node = next(
        (n for n in baseline_trace.nodes if n.node_id == node.node_id),
        None
    )

    if baseline_node:
        delta = node.duration_seconds - baseline_node.duration_seconds
        if abs(delta) > 0.1:  # 100ms difference
            print(f"{node.node_id}: {delta:+.2f}s")
```

## Additional Resources

- [Workflow Execution Guide](WORKFLOW_EXECUTION.md)
- [Workflow Debugging Guide](WORKFLOW_DEBUGGING.md)
- [Trace API Reference](../victor/workflows/trace.py)
