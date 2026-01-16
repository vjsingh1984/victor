# Workflow Execution Guide

This guide covers the comprehensive workflow execution engine with debugging, tracing, and monitoring capabilities in Victor.

## Overview

The Victor workflow execution engine provides production-ready execution of YAML-defined workflows with advanced debugging and monitoring features:

- **Step-by-step debugging** with breakpoints
- **Execution tracing** and detailed logging
- **State inspection** and visualization
- **Execution history** and replay
- **Error recovery** and retry mechanisms
- **Analytics** and performance metrics

## Quick Start

### Basic Execution

Execute a workflow with inputs:

```bash
victor workflow run my_workflow.yaml --context '{"task": "fix bug"}'
```

### Debug Mode

Debug a workflow with breakpoints:

```bash
victor workflow debug my_workflow.yaml --breakpoints node_1,node_3
```

### Trace Execution

Execute with detailed tracing:

```bash
victor workflow trace my_workflow.yaml --output trace.json
```

### View History

List past executions:

```bash
victor workflow history --limit 20
```

## Execution Modes

### 1. Standard Execution

Normal workflow execution:

```bash
victor workflow run workflow.yaml -c '{"input": "data"}'
```

**Features:**
- Executes workflow from start to finish
- Reports success/failure status
- Shows final state
- Displays execution duration

### 2. Debug Mode

Interactive debugging with breakpoints:

```bash
victor workflow debug workflow.yaml \
  --breakpoints node_1,node_2 \
  --context '{"key": "value"}'
```

**Features:**
- Set breakpoints on specific nodes
- Step through execution
- Inspect state at breakpoints
- View variable values
- Continue to next breakpoint

### 3. Trace Mode

Detailed execution tracing:

```bash
victor workflow trace workflow.yaml \
  --output trace.json \
  --format json
```

**Features:**
- Captures all node executions
- Records inputs and outputs
- Tracks execution time
- Logs tool calls
- Export to JSON/CSV/HTML

## Breakpoints

### Setting Breakpoints

Set breakpoints on specific nodes:

```bash
# Single breakpoint
victor workflow debug workflow.yaml --breakpoints node_1

# Multiple breakpoints
victor workflow debug workflow.yaml --breakpoints node_1,node_2,node_3

# With context
victor workflow debug workflow.yaml \
  --breakpoints node_1 \
  --context '{"user": "john"}'
```

### Conditional Breakpoints

Conditional breakpoints can be set programmatically:

```python
from victor.workflows.debugger import WorkflowDebugger

debugger = WorkflowDebugger(workflow)

# Break when value > 100
debugger.set_breakpoint(
    "node_1",
    condition=lambda state: state.get("value", 0) > 100
)

# Break when error occurs
debugger.set_breakpoint(
    "node_2",
    condition=lambda state: state.get("_error") is not None
)
```

### Breakpoint Commands

- `--breakpoints`: Set breakpoints on nodes
- `--stop-on-entry`: Stop before first node
- `--log-level DEBUG`: Enable detailed logging

## State Inspection

### Viewing State

Inspect workflow state during execution:

```bash
# View current state
victor workflow inspect exec_abc123

# Query specific variable
victor workflow inspect exec_abc123 --query user.name

# View specific snapshot
victor workflow inspect exec_abc123 --snapshot snap_xyz
```

### State Queries

Query state using JSONPath-like syntax:

```python
from victor.workflows.state_manager import WorkflowStateManager

manager = WorkflowStateManager()
snapshot_id = manager.capture_state(state, node_id="node_1")

# Simple query
value = manager.query_state(snapshot_id, "user.name")

# Array access
item = manager.query_state(snapshot_id, "items[0].id")

# Nested path
result = manager.query_state(snapshot_id, "data.results.summary")
```

### State Visualization

Visualize state changes:

```python
# Visualize current state
print(manager.visualize_state(snapshot_id))

# Visualize diff
print(manager.visualize_diff(snapshot_1, snapshot_2))
```

## Execution Tracing

### Trace Formats

Export traces in multiple formats:

```bash
# JSON format (default)
victor workflow trace workflow.yaml --output trace.json

# CSV format
victor workflow trace workflow.yaml --format csv --output trace.csv

# HTML report
victor workflow trace workflow.yaml --format html --output report.html
```

### Trace Analysis

Analyze trace for insights:

```python
from victor.workflows.trace import WorkflowTracer

tracer = WorkflowTracer()
tracer.start_trace("workflow_1", inputs={"task": "fix bug"})

# Execute workflow...
tracer.end_trace(outputs={"result": "success"})

# Analyze
analysis = tracer.analyze_trace()
print(f"Total duration: {analysis['total_duration_seconds']:.3f}s")
print(f"Slowest nodes: {analysis['slowest_nodes']}")
print(f"Most used tools: {analysis['most_used_tools']}")
```

### Trace Data

Trace includes:
- **Node executions** with timestamps
- **Inputs and outputs** for each node
- **Execution duration** per node
- **Tool calls** with results
- **Success/failure** status
- **Error messages** if failed

## Execution History

### Viewing History

List past executions:

```bash
# All executions
victor workflow history

# Filter by workflow
victor workflow history --workflow my_workflow

# Limit results
victor workflow history --limit 50

# Export history
victor workflow history --export history.json
victor workflow history --format csv --export report.csv
```

### History Data

Each history record includes:
- Execution ID and timestamp
- Workflow name
- Success/failure status
- Execution duration
- Nodes executed
- Inputs and outputs

### Execution Replay

Replay a previous execution:

```bash
victor workflow replay exec_abc123
```

This re-runs the workflow with the same inputs, allowing you to:
- Compare results with original execution
- Debug intermittent issues
- Verify fixes

## Metrics and Analytics

### Workflow Metrics

View metrics for a workflow:

```bash
# All executions of a workflow
victor workflow metrics --workflow my_workflow

# Specific execution
victor workflow metrics --execution exec_abc123

# Export metrics
victor workflow metrics --workflow my_workflow --export metrics.json
```

### Metrics Include

- **Total executions**: Number of times workflow ran
- **Success rate**: Percentage of successful executions
- **Average duration**: Mean execution time
- **Min/Max duration**: Fastest and slowest executions
- **Node execution counts**: How many times each node ran
- **Tool usage**: Tool call statistics

### Performance Analysis

Analyze workflow performance:

```python
from victor.workflows.history import WorkflowExecutionHistory

history = WorkflowExecutionHistory()

# Get workflow stats
stats = history.get_workflow_stats("my_workflow")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average duration: {stats['average_duration']:.2f}s")

# Get trends
trends = history.get_execution_trends("my_workflow", window_size=10)
print(f"Duration trend: {trends['duration_trend']}")
print(f"Change: {trends['duration_change_percent']:.1f}%")
```

## Error Recovery

### Recovery Strategies

The execution engine supports multiple error recovery strategies:

1. **Fail Fast** (default): Stop immediately on error
2. **Retry**: Retry failed nodes with exponential backoff
3. **Continue**: Skip failed nodes and continue
4. **Skip**: Skip failed nodes entirely
5. **Pause**: Pause on error for debugging

### Configuring Retry

Configure retry behavior:

```python
from victor.workflows.execution_engine import ErrorRecovery, ErrorRecoveryStrategy

recovery = ErrorRecovery(
    strategy=ErrorRecoveryStrategy.RETRY,
    max_retries=3,
    retry_delay_seconds=1.0,
    backoff_multiplier=2.0,
)

executor = WorkflowExecutor(error_recovery=recovery)
```

### Error Handling in Workflows

Configure error handling in YAML:

```yaml
nodes:
  - id: risky_node
    type: agent
    role: executor
    goal: "Perform risky operation"
    continue_on_error: true  # Continue even if this node fails
    retry_policy:
      max_attempts: 3
      backoff_multiplier: 2.0
```

## Programmatic API

### Execution API

Execute workflows programmatically:

```python
from victor.workflows.execution_engine import WorkflowExecutor
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

# Compile workflow
compiler = UnifiedWorkflowCompiler()
compiled = compiler.compile_yaml("workflow.yaml", "my_workflow")

# Execute
executor = WorkflowExecutor(debug_mode=True, trace_mode=True)
result = await executor.execute(
    compiled,
    inputs={"task": "fix bug"},
    breakpoints=["node_1", "node_3"],
)

# Get trace
trace = executor.get_trace()
summary = trace.get_summary()

# Export trace
executor.export_trace("trace.json")
```

### Streaming API

Stream execution events:

```python
async for event in executor.execute_stream(compiled, inputs):
    print(f"{event.node_id}: {event.event_type.value}")
    print(f"  Status: {event.status.value}")
    print(f"  Duration: {event.duration_seconds:.3f}s")
```

### Debugger API

Interactive debugging:

```python
from victor.workflows.debugger import WorkflowDebugger

debugger = WorkflowDebugger(workflow)

# Set breakpoints
debugger.set_breakpoint("node_1")
debugger.set_breakpoint("node_3", condition=lambda s: s.get("value") > 10)

# Start debugging
await debugger.start(inputs={"task": "fix bug"})

# Step execution
await debugger.step_over()
await debugger.step_into()
await debugger.step_out()

# Inspect state
state = debugger.get_state()
variables = debugger.get_variables()

# Get stack trace
stack = debugger.get_stack_trace()

# Continue
await debugger.continue_execution()
```

### State Manager API

State management:

```python
from victor.workflows.state_manager import WorkflowStateManager

manager = WorkflowStateManager()

# Capture snapshots
snapshot_id = manager.capture_state(state, node_id="node_1")

# Visualize state
print(manager.visualize_state(snapshot_id))

# Query state
value = manager.query_state(snapshot_id, "user.name")

# Compute diff
diff = manager.compute_diff(snapshot_1, snapshot_2)

# Rollback
restored_state = manager.rollback_to(snapshot_id)
```

## CLI Command Reference

### workflow run

Execute a workflow:

```bash
victor workflow run WORKFLOW_PATH [OPTIONS]
```

**Options:**
- `-c, --context TEXT`: Initial context as JSON
- `-f, --context-file PATH`: Context JSON file
- `-n, --name TEXT`: Workflow name (if multiple)
- `-p, --profile TEXT`: Profile for agent nodes
- `--dry-run`: Validate without executing
- `--log-level TEXT`: Set logging level

### workflow debug

Debug a workflow:

```bash
victor workflow debug WORKFLOW_PATH [OPTIONS]
```

**Options:**
- `-c, --context TEXT`: Initial context as JSON
- `-b, --breakpoints TEXT`: Comma-separated node IDs
- `-n, --name TEXT`: Workflow name
- `--stop-on-entry`: Stop before first node
- `--log-level TEXT`: Logging level (default: DEBUG)

### workflow trace

Trace workflow execution:

```bash
victor workflow trace WORKFLOW_PATH [OPTIONS]
```

**Options:**
- `-c, --context TEXT`: Initial context as JSON
- `-o, --output PATH`: Output trace file
- `-n, --name TEXT`: Workflow name
- `-f, --format TEXT`: Output format (json/csv/html)

### workflow history

Show execution history:

```bash
victor workflow history [OPTIONS]
```

**Options:**
- `-w, --workflow TEXT`: Filter by workflow name
- `-l, --limit INTEGER`: Max executions to show
- `-e, --export PATH`: Export to file
- `-f, --format TEXT`: Export format (json/csv)

### workflow replay

Replay execution:

```bash
victor workflow replay EXECUTION_ID [OPTIONS]
```

**Options:**
- `-e, --export-trace`: Export trace from replay

### workflow metrics

Show metrics:

```bash
victor workflow metrics [OPTIONS]
```

**Options:**
- `-w, --workflow TEXT`: Workflow name
- `-e, --execution TEXT`: Execution ID
- `-o, --export PATH`: Export metrics

## Best Practices

### 1. Use Tracing for Debugging

Enable tracing when investigating issues:

```bash
victor workflow trace workflow.yaml --output trace.json
```

Review the trace to identify:
- Slow nodes
- Failed tool calls
- Error patterns

### 2. Set Strategic Breakpoints

Set breakpoints at:
- Before complex operations
- After data transformations
- At decision points (condition nodes)

### 3. Monitor Performance

Regularly check metrics:

```bash
victor workflow metrics --workflow my_workflow
```

Look for:
- Increasing execution times
- Decreasing success rates
- Resource-intensive nodes

### 4. Use History for Analysis

Export and analyze history:

```bash
victor workflow history --export history.json
```

Identify trends and patterns over time.

### 5. Implement Error Recovery

Configure retry policies for flaky operations:

```yaml
nodes:
  - id: api_call
    type: compute
    handler: call_external_api
    retry_policy:
      max_attempts: 3
      initial_delay: 1.0
      backoff_multiplier: 2.0
```

## Troubleshooting

### Execution Fails

1. **Check trace for errors:**
   ```bash
   victor workflow trace workflow.yaml --output trace.json
   ```

2. **Review error messages in trace**
3. **Check node inputs and outputs**
4. **Verify tool configurations**

### Slow Execution

1. **Identify slow nodes:**
   ```bash
   victor workflow metrics --workflow workflow_name
   ```

2. **Review trace for bottlenecks**
3. **Consider parallelizing independent nodes**
4. **Optimize tool calls**

### Breakpoint Not Hit

1. **Verify node ID is correct**
2. **Check conditional breakpoint logic**
3. **Ensure node is executed in workflow path**

### State Not Found

1. **Check execution ID is correct**
2. **Verify history hasn't been cleared**
3. **Use `victor workflow history` to list executions**

## Examples

### Example 1: Debug Failing Workflow

```bash
# 1. Run with trace
victor workflow trace failing_workflow.yaml --output trace.json

# 2. Review trace
cat trace.json | jq '.nodes[] | select(.success == false)'

# 3. Debug with breakpoints at failing node
victor workflow debug failing_workflow.yaml \
  --breakpoints failing_node \
  --log-level DEBUG
```

### Example 2: Performance Analysis

```bash
# 1. Check metrics
victor workflow metrics --workflow slow_workflow

# 2. Get detailed trace
victor workflow trace slow_workflow.yaml --output trace.json

# 3. Analyze slowest nodes
cat trace.json | jq '.nodes | sort_by(.duration_seconds) | reverse | .[0:5]'
```

### Example 3: Compare Executions

```python
from victor.workflows.history import WorkflowExecutionHistory

history = WorkflowExecutionHistory()

# Compare two executions
comparison = history.compare_executions("exec_1", "exec_2")

print("Output differences:", comparison.output_diff)
print("Performance difference:", comparison.performance_diff)
```

## Advanced Topics

### Custom Trace Handlers

Implement custom trace handlers:

```python
from victor.workflows.trace import WorkflowTracer

class CustomTracer(WorkflowTracer):
    def on_node_complete(self, node_record):
        # Custom handling
        if node_record.duration_seconds > 10.0:
            print(f"Slow node: {node_record.node_id}")
```

### State Snapshots

Automatically capture snapshots:

```python
from victor.workflows.state_manager import WorkflowStateManager

manager = WorkflowStateManager(auto_capture=True)

# Automatically captures state at each node
```

### Export Formats

Custom export formats:

```python
from victor.workflows.trace import WorkflowTracer

tracer = WorkflowTracer()

# Export to custom format
tracer.export_trace(
    output_path="custom.txt",
    format="custom",
)
```

## Additional Resources

- [Workflow Debugging Guide](WORKFLOW_DEBUGGING.md)
- [Execution Tracing Guide](EXECUTION_TRACING.md)
- [State Management Reference](state_manager.py)
- [Error Handling Guide](docs/error_patterns.md)
