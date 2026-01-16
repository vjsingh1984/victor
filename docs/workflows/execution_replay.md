# Workflow Execution Replay System

The workflow execution replay system provides comprehensive recording and debugging capabilities for workflow executions, enabling you to record, replay, and analyze workflow executions with minimal overhead.

## Overview

The execution replay system captures all events during workflow execution, including:
- Node execution with inputs/outputs
- Team member communications
- Recursion depth changes
- State snapshots at key points
- Timing information for performance analysis

This enables powerful debugging and analysis workflows:
- **Debug failed executions** - Replay exactly what happened
- **Performance analysis** - Identify bottlenecks with timing data
- **Behavior comparison** - Compare executions side-by-side
- **Visualization** - Generate execution graphs

## Key Features

- **Low overhead** - <5% performance impact during recording
- **Efficient storage** - Optional gzip compression
- **State inspection** - View workflow state at any point
- **Step-through debugging** - Navigate execution event by event
- **Comparison** - Diff multiple executions
- **Visualization** - Export execution graphs

## Quick Start

### Recording a Workflow Execution

```python
from victor.workflows.recording_integration import record_workflow
from victor.workflows.unified_executor import StateGraphExecutor

# Automatically record workflow execution
with record_workflow("my_workflow") as recorder:
    executor = StateGraphExecutor(orchestrator)
    result = await executor.execute(workflow, initial_context)

# Recording is automatically finalized
print(f"Recorded {recorder.metadata.event_count} events")
```

### Saving Recordings

```python
from victor.workflows.recording_storage import FileRecordingStorage

# Save to file
await recorder.save("/path/to/recording.json")

# Or use storage backend
storage = FileRecordingStorage(base_path="./recordings")
recording_id = await storage.save(recorder)
print(f"Saved recording: {recording_id}")
```

### Replaying a Recording

```python
from victor.workflows.execution_recorder import ExecutionReplayer

# Load recording
replayer = ExecutionReplayer.load("/path/to/recording.json.gz")

# Step through execution
for event in replayer.step_forward():
    print(f"[{event.timestamp}] {event.event_type.value} @ {event.node_id}")

    # Inspect state at this point
    state = replayer.get_state_at_event(event.event_id)
    if state:
        print(f"  State: {list(state.keys())}")
```

## Recording Strategies

### 1. Manual Recording

Record specific workflow executions when needed:

```python
with record_workflow("important_workflow", tags=["production", "critical"]) as recorder:
    result = await executor.execute(workflow, context)

await recorder.save("important_execution.json")
```

### 2. On-Failure Recording

Automatically record failed executions for debugging:

```python
from victor.workflows.recording_integration import enable_workflow_recording

recorder = enable_workflow_recording(
    workflow_name="my_workflow",
    tags=["auto-failure"]
)

try:
    result = await executor.execute(workflow, context)
except Exception as e:
    # Recording will have captured the failure
    await recorder.save(f"failure_{int(time.time())}.json")
    raise
```

### 3. Sampling Recording

Record a sample of executions for analysis:

```python
import random

if random.random() < 0.1:  # 10% sampling rate
    with record_workflow("sampled_workflow") as recorder:
        result = await executor.execute(workflow, context)
    await recorder.save(f"sample_{recorder.metadata.recording_id}.json")
```

### 4. Conditional Recording

Record based on workflow characteristics:

```python
# Record only long-running workflows
start = time.time()
result = await executor.execute(workflow, context)
duration = time.time() - start

if duration > 60:  # Longer than 60 seconds
    recorder = get_current_recorder()
    if recorder:
        await recorder.save(f"long_running_{int(duration)}s.json")
```

## CLI Tool

The `scripts/workflows/replay.py` CLI provides convenient commands for managing recordings:

### List Recordings

```bash
python scripts/workflows/replay.py list

# Filter by workflow name
python scripts/workflows/replay.py list --workflow my_workflow

# Filter by date range
python scripts/workflows/replay.py list --start 2025-01-01 --end 2025-01-31

# Filter by status
python scripts/workflows/replay.py list --success false

# Filter by tags
python scripts/workflows/replay.py list --tags production,critical

# Sort and paginate
python scripts/workflows/replay.py list --sort-by duration --sort-order desc --limit 10
```

### Inspect Recording Metadata

```bash
python scripts/workflows/replay.py inspect <recording_id>
```

Output:
```
Recording: abc123def456
Workflow: my_workflow
--------------------------------------------------

Status:
  Success: True
  Error: None

Timing:
  Started: 2025-01-15 10:30:00
  Completed: 2025-01-15 10:30:15
  Duration: 15.2s

Execution:
  Nodes executed: 5
  Teams spawned: 2
  Max recursion depth: 2
  Total events: 24

Storage:
  File size: 12.3KB
  Checksum: a1b2c3d4...
```

### Replay with Step-Through

```bash
# Interactive step-through mode
python scripts/workflows/replay.py replay <recording_id> --step

# Show event data
python scripts/workflows/replay.py replay <recording_id> --step --show-data

# Show state at each event
python scripts/workflows/replay.py replay <recording_id> --step --show-state
```

### Compare Recordings

```bash
python scripts/workflows/replay.py compare <id1> <id2>
```

Output:
```
Comparing recordings:
  1: my_workflow (abc123de)
  2: my_workflow (def456gh)

Metadata differences:
  Duration: 15.2s vs 18.5s
  Nodes: 5 vs 5
  Teams: 2 vs 2
  Events: 24 vs 26

Node differences:
  Only in 1: set()
  Only in 2: set()
  Common: 5 nodes

First path difference:
  Position: 3
  Recording 1: process_data
  Recording 2: validate_input
```

### Export Visualizations

```bash
# Export as Graphviz DOT
python scripts/workflows/replay.py export <recording_id> --format dot --output workflow.dot

# Render to PNG
dot -Tpng workflow.dot -o workflow.png

# Export as JSON
python scripts/workflows/replay.py export <recording_id> --format json --output workflow.json

# Export as human-readable summary
python scripts/workflows/replay.py export <recording_id> --format summary --output summary.txt
```

### Storage Statistics

```bash
python scripts/workflows/replay.py stats
```

Output:
```
Storage Statistics: ./recordings
--------------------------------------------------

Recordings:
  Total: 47
  Successful: 42
  Failed: 5

Storage:
  Total size: 2.3MB
  Total duration: 12m 34s

Time range:
  Oldest: 2025-01-01 00:00:00
  Newest: 2025-01-15 10:30:00

Workflows:
  data_processing: 20
  code_review: 15
  research_task: 12
```

### Cleanup Old Recordings

```bash
# Dry-run (show what would be deleted)
python scripts/workflows/replay.py cleanup --max-age-days 30 --dry-run

# Actually delete recordings older than 30 days
python scripts/workflows/replay.py cleanup --max-age-days 30

# Keep only the most recent 100 recordings
python scripts/workflows/replay.py cleanup --max-count 100

# Delete failed recordings as well
python scripts/workflows/replay.py cleanup --max-age-days 7 --delete-failed
```

## Storage Backends

### File-Based Storage (Default)

```python
from victor.workflows.recording_storage import FileRecordingStorage

storage = FileRecordingStorage(
    base_path="./recordings",
    compress=True,  # Use gzip compression
)

# Save
recording_id = await storage.save(recorder)

# Load
replayer = await storage.load(recording_id)

# List
recordings = await storage.list()

# Search with query
from victor.workflows.recording_storage import RecordingQuery

query = RecordingQuery(
    workflow_name="my_workflow",
    success=True,
    min_duration=10.0,
)
results = await storage.list(query)
```

### In-Memory Storage (Testing)

```python
from victor.workflows.recording_storage import InMemoryRecordingStorage

storage = InMemoryRecordingStorage()

# Use same API as FileRecordingStorage
recording_id = await storage.save(recorder)
replayer = await storage.load(recording_id)

# Clear all recordings
storage.clear()
```

## Retention Policies

Manage storage with automated retention policies:

```python
from victor.workflows.recording_storage import RetentionPolicy

# Keep recordings for 30 days
policy = RetentionPolicy(max_age_days=30)

# Keep only the most recent 100 recordings
policy = RetentionPolicy(max_count=100)

# Keep recordings under 1GB total
policy = RetentionPolicy(max_size_gb=1.0)

# Keep failed recordings for debugging
policy = RetentionPolicy(
    max_age_days=7,
    keep_failed=True,  # Keep failures longer
)

# Always keep recordings with specific tags
policy = RetentionPolicy(
    max_age_days=30,
    tags_to_keep=["critical", "production"],
)

# Apply policy
result = await storage.apply_retention_policy(policy, dry_run=False)
print(f"Deleted {result['to_delete']} recordings")
print(f"Freed {result['total_size_bytes']} bytes")
```

## Advanced Usage

### State Inspection

Inspect workflow state at any point during execution:

```python
# Load recording
replayer = ExecutionReplayer.load("recording.json.gz")

# Get state at a specific event
event_id = replayer.events[5].event_id
state = replayer.get_state_at_event(event_id)

print(f"State keys: {list(state.keys())}")
print(f"Values: {state}")
```

### Navigation

Navigate through recordings with precision:

```python
replayer = ExecutionReplayer.load("recording.json.gz")

# Jump to specific event
replayer.jump_to_event(event_id="abc123")

# Jump to position
replayer.jump_to_position(10)

# Step forward
for event in replayer.step_forward(steps=5):
    print(event.event_type)

# Step backward
for event in replayer.step_backward(steps=3):
    print(event.event_type)

# Reset to beginning
replayer.reset()
```

### Comparison and Diffing

Compare two executions to identify differences:

```python
replayer1 = ExecutionReplayer.load("recording1.json.gz")
replayer2 = ExecutionReplayer.load("recording2.json.gz")

diff = replayer1.compare(replayer2)

# Check metadata differences
print(f"Duration diff: {diff['metadata_diff']['duration_diff']}s")

# Check node differences
print(f"Only in 1: {diff['node_diff']['only_in_self']}")
print(f"Only in 2: {diff['node_diff']['only_in_other']}")

# Check execution path differences
if diff['path_diff']['first_difference']:
    fd = diff['path_diff']['first_difference']
    print(f"Paths diverge at position {fd['position']}")
    print(f"  1: {fd['self_node']}")
    print(f"  2: {fd['other_node']}")
```

### Custom Event Recording

Record custom events during execution:

```python
from victor.workflows.execution_recorder import RecordingEventType

recorder.record_recursion_enter("workflow", "nested_workflow")
recorder.record_recursion_exit("workflow", "nested_workflow")

# Record team execution
recorder.record_team_start(
    team_id="review_team",
    formation="parallel",
    member_count=3,
    context={"task": "review code"},
)

# Record team communication
recorder.record_team_member_communication(
    team_id="review_team",
    from_member="security_reviewer",
    to_member="quality_reviewer",
    message="Found 3 security issues",
)
```

## Best Practices

### 1. Tag Your Recordings

Use tags for easy filtering and organization:

```python
tags = ["production", "critical", "security-scan"]
with record_workflow("security_scan", tags=tags) as recorder:
    result = await executor.execute(workflow, context)
```

### 2. Compress Large Recordings

Enable compression to save space:

```python
recorder = ExecutionRecorder(
    workflow_name="large_workflow",
    compress=True,  # Reduces size by 5-10x
)
```

### 3. Selective Recording

Record only what you need:

```python
recorder = ExecutionRecorder(
    workflow_name="my_workflow",
    record_inputs=True,   # Record node inputs
    record_outputs=True,  # Record node outputs
    record_state_snapshots=False,  # Skip state snapshots (saves space)
)
```

### 4. Automatic Cleanup

Set up retention policies to prevent storage bloat:

```python
# Daily cleanup task
async def cleanup_recordings():
    storage = FileRecordingStorage(base_path="./recordings")

    # Delete recordings older than 30 days
    policy = RetentionPolicy(max_age_days=30)
    result = await storage.apply_retention_policy(policy)

    logger.info(f"Cleaned up {result['to_delete']} recordings")
```

### 5. Debug Failed Executions

Always record failures for post-mortem analysis:

```python
recorder = enable_workflow_recording(
    workflow_name="critical_workflow",
    tags=["auto-record"],
)

try:
    result = await executor.execute(workflow, context)
except Exception as e:
    # Save recording with error context
    await recorder.save(f"failure_{time.time()}.json")
    logger.error(f"Workflow failed, recording saved: {e}")
    raise
```

### 6. Performance Analysis

Use timing data to identify bottlenecks:

```python
replayer = ExecutionReplayer.load("recording.json.gz")

# Calculate node durations
node_durations = {}
for event in replayer.events:
    if event.event_type == RecordingEventType.NODE_COMPLETE:
        node_id = event.node_id
        duration = event.data.get("duration_seconds", 0)
        node_durations[node_id] = node_durations.get(node_id, 0) + duration

# Sort by duration
for node_id, duration in sorted(node_durations.items(), key=lambda x: -x[1]):
    print(f"{node_id}: {duration:.2f}s")
```

## Performance Considerations

### Recording Overhead

The recording system is designed for minimal overhead:
- **Event recording**: <1ms per event
- **State snapshots**: 1-5ms (depending on state size)
- **Compression**: 10-50ms at the end
- **Overall**: <5% performance impact

### Storage Optimization

To minimize storage usage:
1. Enable compression (`compress=True`)
2. Disable state snapshots unless needed
3. Use selective recording (only record what you need)
4. Implement retention policies
5. Sample executions instead of recording all

Example storage sizes:
- Small workflow (10 events): ~1KB (uncompressed), ~300B (compressed)
- Medium workflow (100 events): ~10KB (uncompressed), ~2KB (compressed)
- Large workflow (1000 events): ~100KB (uncompressed), ~15KB (compressed)

## Troubleshooting

### Recording Not Created

**Problem**: Recording file not created after execution.

**Solution**: Check that recording is properly enabled:
```python
from victor.workflows.recording_integration import get_current_recorder

recorder = get_current_recorder()
if recorder is None:
    print("Recording not enabled!")
```

### Large File Sizes

**Problem**: Recording files are too large.

**Solution**: Enable compression and disable unnecessary features:
```python
recorder = ExecutionRecorder(
    workflow_name="my_workflow",
    compress=True,
    record_state_snapshots=False,  # Disable snapshots
)
```

### Out of Memory During Replay

**Problem**: Loading large recordings causes memory issues.

**Solution**: Use streaming instead of loading all events:
```python
# Instead of loading all events, use step_forward
for event in replayer.step_forward(steps=100):
    process_event(event)
```

## API Reference

See the API documentation for complete details:
- `victor.workflows.execution_recorder` - Recording and replay classes
- `victor.workflows.recording_storage` - Storage backends
- `victor.workflows.recording_integration` - Integration utilities

## Examples

See `examples/workflows/execution_replay_demo.py` for complete examples.
