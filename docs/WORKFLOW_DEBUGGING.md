# Workflow Debugging Guide

This guide covers interactive debugging capabilities for Victor workflows.

## Overview

The Victor workflow debugger provides comprehensive debugging features similar to traditional code debuggers:

- **Breakpoints**: Set conditional/unconditional breakpoints on nodes
- **Stepping**: Step over, into, and out of nodes
- **State Inspection**: View variables and state at any point
- **Stack Traces**: See execution path and call stack
- **Watch Variables**: Monitor variable changes
- **Interactive Console**: Execute code in debug context

## Getting Started

### Basic Debugging Session

Start a debug session:

```bash
victor workflow debug my_workflow.yaml --breakpoints node_1,node_3
```

Or programmatically:

```python
from victor.workflows.debugger import WorkflowDebugger

debugger = WorkflowDebugger(workflow)
debugger.set_breakpoint("node_1")
await debugger.start(inputs={"task": "fix bug"})
```

## Breakpoints

### Setting Breakpoints

#### CLI

Set breakpoints via command line:

```bash
# Single breakpoint
victor workflow debug workflow.yaml --breakpoints node_1

# Multiple breakpoints
victor workflow debug workflow.yaml --breakpoints node_1,node_2,node_3

# With context
victor workflow debug workflow.yaml \
  --breakpoints node_1 \
  --context '{"key": "value"}'
```

#### Programmatic

Set breakpoints programmatically:

```python
# Simple breakpoint
debugger.set_breakpoint("node_1")

# Conditional breakpoint
debugger.set_breakpoint(
    "node_2",
    condition=lambda state: state.get("error_count", 0) > 5
)

# Temporary breakpoint (removed after first hit)
debugger.set_breakpoint("node_3", temporary=True)
```

### Breakpoint Conditions

Conditional breakpoints allow you to break only when a condition is true:

```python
# Break when value exceeds threshold
debugger.set_breakpoint(
    "process_node",
    condition=lambda s: s.get("value", 0) > 100
)

# Break when error occurs
debugger.set_breakpoint(
    "any_node",
    condition=lambda s: s.get("_error") is not None
)

# Break on specific user input
debugger.set_breakpoint(
    "user_input",
    condition=lambda s: s.get("user_action") == "cancel"
)

# Complex condition
debugger.set_breakpoint(
    "validate",
    condition=lambda s: (
        s.get("status") == "failed" and
        s.get("retry_count", 0) >= 3
    )
)
```

### Managing Breakpoints

List, enable, disable, and clear breakpoints:

```python
# List all breakpoints
breakpoints = debugger.list_breakpoints()
for bp in breakpoints:
    print(f"{bp.node_id}: enabled={bp.enabled}, hits={bp.hit_count}")

# Disable a breakpoint
debugger.disable_breakpoint("node_1")

# Enable a breakpoint
debugger.enable_breakpoint("node_1")

# Clear a specific breakpoint
debugger.clear_breakpoint("node_1")

# Clear all breakpoints
debugger.clear_all_breakpoints()
```

## Stepping Through Execution

### Step Over

Execute the current node and move to the next one:

```python
await debugger.step_over()
```

Use this when you want to execute the current node without diving into details.

### Step Into

Step into the current node to see internal execution:

```python
await debugger.step_into()
```

Use this for agent nodes or parallel nodes to see what's happening inside.

### Step Out

Step out of the current node to return to the caller:

```python
await debugger.step_out()
```

Use this when you're done examining the current node and want to return.

### Continue

Continue execution until the next breakpoint:

```python
await debugger.continue_execution()
```

## State Inspection

### Viewing State

Get the current workflow state:

```python
# Get full state
state = debugger.get_state()
print(json.dumps(state, indent=2))

# Get only user variables (non-internal)
variables = debugger.get_variables()
for key, value in variables.items():
    print(f"{key}: {value}")
```

### Querying Variables

Get specific variable values:

```python
# Get single variable
user_name = debugger.get_variable("user.name")

# Check if variable exists
if debugger.get_variable("_error"):
    print("An error occurred!")

# Get nested variable
result = debugger.get_variable("data.results.summary")
```

### Modifying Variables

Set variable values during debugging:

```python
# Change a variable
debugger.set_variable("retry_count", 0)

# Fix a value
debugger.set_variable("api_key", "correct_key")

# Add a new variable
debugger.set_variable("debug_mode", True)
```

## Stack Traces

### Viewing Call Stack

See the execution path:

```python
# Get full stack trace
stack = debugger.get_stack_trace()
for frame in stack:
    print(f"{frame.node_id} ({frame.node_type})")

# Get as simple list
call_stack = debugger.get_call_stack()
print(" -> ".join(call_stack))

# Get current frame
current_frame = debugger.get_current_frame()
if current_frame:
    print(f"Currently at: {current_frame.node_id}")
```

### Stack Frame Information

Each stack frame contains:

```python
for frame in stack:
    print(f"Node: {frame.node_id}")
    print(f"Type: {frame.node_type}")
    print(f"Timestamp: {frame.timestamp}")
    print(f"State keys: {list(frame.state.keys())}")
```

## Debugging Workflow

### Typical Debugging Session

1. **Start with breakpoints:**
   ```python
   debugger = WorkflowDebugger(workflow)
   debugger.set_breakpoint("node_1")
   await debugger.start(inputs={"task": "fix bug"})
   ```

2. **Hit first breakpoint:**
   ```python
   # Inspect state
   state = debugger.get_state()
   print(f"Current value: {state.get('value')}")
   ```

3. **Step through code:**
   ```python
   # Execute next node
   await debugger.step_over()

   # Check if state changed
   new_state = debugger.get_state()
   ```

4. **Continue to next breakpoint:**
   ```python
   await debugger.continue_execution()
   ```

5. **Inspect final state:**
   ```python
   final_state = debugger.get_state()
   print(f"Result: {final_state.get('result')}")
   ```

### Debugging Common Issues

#### Issue: Node Failing

```python
# Set breakpoint before failing node
debugger.set_breakpoint("before_failing_node")

# Check inputs
state = debugger.get_state()
print(f"Inputs: {state.get('inputs')}")

# Step into failing node
await debugger.step_into()

# Check error
if "_error" in state:
    print(f"Error: {state['_error']}")
```

#### Issue: Wrong Output

```python
# Set breakpoint at output generation
debugger.set_breakpoint("output_node")

# Inspect intermediate results
state = debugger.get_state()
print(f"Intermediate: {state.get('intermediate_result')}")

# Step through transformation
await debugger.step_over()
```

#### Issue: Infinite Loop

```python
# Set breakpoint on loop node
debugger.set_breakpoint("loop_node")

# Check loop counter
state = debugger.get_state()
iteration = state.get("iteration", 0)

# Set max iterations
if iteration > 100:
    debugger.set_variable("stop_loop", True)
```

## Advanced Debugging

### Watch Expressions

Monitor variable changes:

```python
# Store initial value
last_value = debugger.get_variable("status")

# Continue execution
await debugger.step_over()

# Check if changed
current_value = debugger.get_variable("status")
if current_value != last_value:
    print(f"Status changed: {last_value} -> {current_value}")
```

### Conditional Debugging

Debug only when certain conditions are met:

```python
# Only debug when error occurs
debugger.set_breakpoint(
    "any_node",
    condition=lambda s: s.get("_error") is not None
)

# Only debug for specific user
debugger.set_breakpoint(
    "user_action",
    condition=lambda s: s.get("user_id") == "problem_user"
)
```

### Debugging Parallel Nodes

When debugging parallel execution:

```python
# Set breakpoint on parallel node
debugger.set_breakpoint("parallel_node")

# Step into to see parallel branches
await debugger.step_into()

# Check results from each branch
state = debugger.get_state()
parallel_results = state.get("_parallel_results", {})
for branch_id, result in parallel_results.items():
    print(f"Branch {branch_id}: {result}")
```

## Debug Session Management

### Session Information

Get information about the current debug session:

```python
info = debugger.get_session_info()
print(f"Session ID: {info['session_id']}")
print(f"State: {info['state']}")
print(f"Duration: {info['duration_seconds']:.2f}s")
print(f"Breakpoints: {len(info['breakpoints'])}")
```

### Session Events

View debug session events:

```python
events = debugger.get_events(limit=10)
for event in events:
    print(f"{event['type']}: {event['data']}")
```

### Saving Session State

Save debug session for later analysis:

```python
import json

# Get session state
state = debugger.get_state()

# Save to file
with open("debug_state.json", "w") as f:
    json.dump(state, f, indent=2, default=str)
```

## Debugging Tips

### 1. Start Simple

Begin with obvious breakpoints:
- At the start of workflow
- Before complex operations
- After data transformations

### 2. Use Conditional Breakpoints

Avoid manual inspection by using conditions:
```python
debugger.set_breakpoint(
    "node",
    condition=lambda s: s.get("value") > threshold
)
```

### 3. Inspect State Regularly

Check state at each breakpoint:
```python
state = debugger.get_state()
print(f"Keys: {list(state.keys())}")
```

### 4. Watch for Errors

Always check for errors:
```python
if "_error" in state:
    print(f"Error at {current_frame.node_id}: {state['_error']}")
```

### 5. Use Stack Traces

Understand execution flow:
```python
stack = debugger.get_call_stack()
print(" -> ".join(stack))
```

## CLI Debugging Commands

### Interactive Commands

When debugging interactively:

```
(victor-debug) help          # Show help
(victor-debug) step          # Step over
(victor-debug) step_into     # Step into
(victor-debug) step_out      # Step out
(victor-debug) continue      # Continue to breakpoint
(victor-debug) break <node>  # Set breakpoint
(victor-debug) clear <node>  # Clear breakpoint
(victor-debug) list          # List breakpoints
(victor-debug) state         # Show state
(victor-debug) vars          # Show variables
(victor-debug) stack         # Show stack trace
(victor-debug) quit          # Quit debugger
```

## Examples

### Example 1: Debug Failing API Call

```python
debugger = WorkflowDebugger(workflow)

# Break before API call
debugger.set_breakpoint("before_api")

# Break after API call
debugger.set_breakpoint("after_api")

await debugger.start(inputs={"api_endpoint": "/api/data"})

# First breakpoint
state = debugger.get_state()
print(f"Request: {state.get('request_data')}")

# Continue to API call
await debugger.continue_execution()

# Second breakpoint
state = debugger.get_state()
if state.get("api_success"):
    print(f"Response: {state.get('response_data')}")
else:
    print(f"Error: {state.get('_error')}")
```

### Example 2: Debug Data Transformation

```python
debugger = WorkflowDebugger(workflow)

# Break at each transformation step
debugger.set_breakpoint("transform_1")
debugger.set_breakpoint("transform_2")
debugger.set_breakpoint("transform_3")

await debugger.start(inputs={"data": [1, 2, 3]})

# Track data through transformations
data = None
while True:
    state = debugger.get_state()
    new_data = state.get("data")

    if new_data != data:
        print(f"Data changed: {data} -> {new_data}")
        data = new_data

    # Continue until end
    if debugger.session.state == DebugState.STOPPED:
        break

    await debugger.continue_execution()
```

### Example 3: Debug Parallel Execution

```python
debugger = WorkflowDebugger(workflow)

debugger.set_breakpoint("parallel_start")
debugger.set_breakpoint("parallel_end")

await debugger.start(inputs={"items": [1, 2, 3, 4, 5]})

# At parallel start
state = debugger.get_state()
items = state.get("items", [])
print(f"Processing {len(items)} items in parallel")

# Step into parallel execution
await debugger.step_into()

# At parallel end
state = debugger.get_state()
results = state.get("_parallel_results", {})
for item_id, result in results.items():
    print(f"Item {item_id}: {result}")
```

## Troubleshooting

### Breakpoint Not Hit

**Possible causes:**
1. Node ID is incorrect
2. Conditional breakpoint condition never true
3. Node not in execution path

**Solutions:**
```python
# Verify node exists in workflow
print(workflow.nodes.keys())

# Check condition
condition = lambda s: s.get("value") > 100
test_state = {"value": 50}
print(f"Condition result: {condition(test_state)}")

# Verify execution path
print("Start node:", workflow.start_node)
```

### Can't Access State

**Possible causes:**
1. Execution hasn't started
2. Session not paused
3. State not initialized

**Solutions:**
```python
# Check session state
print(debugger.session.state)

# Ensure execution started
if debugger.session.state == DebugState.IDLE:
    await debugger.start(inputs={...})

# Wait for breakpoint
while debugger.session.state != DebugState.PAUSED:
    await asyncio.sleep(0.1)
```

### Step Commands Not Working

**Possible causes:**
1. Not in debug mode
2. Session not paused
3. Execution completed

**Solutions:**
```python
# Check debug mode
if not debugger.session.debug_mode:
    print("Not in debug mode")

# Check if paused
if debugger.session.state != DebugState.PAUSED:
    print("Session not paused")

# Check if execution complete
if debugger.session.state == DebugState.STOPPED:
    print("Execution complete")
```

## Best Practices

1. **Use meaningful breakpoint conditions:**
   ```python
   # Good
   condition=lambda s: s.get("error_count", 0) > threshold

   # Bad
   condition=lambda s: True  # Always breaks
   ```

2. **Inspect state at each breakpoint:**
   ```python
   state = debugger.get_state()
   for key in sorted(state.keys()):
       if not key.startswith("_"):
           print(f"{key}: {state[key]}")
   ```

3. **Use stack traces to understand flow:**
   ```python
   stack = debugger.get_call_stack()
   print("Execution path: " + " -> ".join(stack))
   ```

4. **Save debug session for analysis:**
   ```python
   with open(f"debug_{debugger.session.session_id}.json", "w") as f:
       json.dump(debugger.get_state(), f, indent=2)
   ```

5. **Clear breakpoints after debugging:**
   ```python
   debugger.clear_all_breakpoints()
   ```

## Additional Resources

- [Workflow Execution Guide](WORKFLOW_EXECUTION.md)
- [State Management Reference](../victor/workflows/state_manager.py)
- [Execution Tracing Guide](EXECUTION_TRACING.md)
