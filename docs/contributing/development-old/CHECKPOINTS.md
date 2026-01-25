# Checkpoint System - Time-Travel Debugging

> **Archived**: This document is kept for historical context and may be outdated. See `docs/contributing/index.md` for current guidance.


**Victor Framework**

Victor's checkpoint system provides time-travel debugging capabilities similar to LangGraph, enabling safe rollback and exploration of alternate execution paths.

## Overview

The checkpoint system uses a **dual-backend architecture**:

1. **Git Backend**: Working tree snapshots via git stash
2. **SQLite Backend**: Fine-grained conversation/execution state persistence

### Key Features

- **State Persistence**: Full conversation context and execution state
- **Auto-Checkpointing**: Automatic checkpoints every N tool calls
- **Session Forking**: Create alternate execution paths from any checkpoint
- **Time-Travel Debugging**: Restore to any previous state
- **Dual Backend**: Git for files, SQLite for state

## Quick Start

### CLI Usage

```bash
# Create a checkpoint
victor checkpoint save "Before refactoring"

# List all checkpoints
victor checkpoint list

# Show checkpoint details
victor checkpoint show checkpoint_abc123

# Restore to a checkpoint
victor checkpoint restore checkpoint_abc123

# Fork a new session from checkpoint
victor checkpoint fork checkpoint_abc123

# Clean up old checkpoints
victor checkpoint cleanup --keep 10

# Configure auto-checkpointing
victor checkpoint auto --enable --interval 5
```

### Python API

#### Basic Checkpoint Operations

```python
from victor.agent.enhanced_checkpoints import (
    EnhancedCheckpointManager,
    CheckpointState,
)

# Create checkpoint manager
manager = EnhancedCheckpointManager(
    auto_checkpoint=True,
    checkpoint_interval=5,
)

# Create a checkpoint
state = CheckpointState(
    session_id="my_session",
    messages=[...],  # Conversation messages
    context={"user": "John"},  # Conversation context
    execution_state={"step": "planning"},  # Execution state
    tool_calls=[...],  # Tool call history
)

checkpoint = manager.save_checkpoint(
    description="Before major refactoring",
    session_id="my_session",
    state=state,
)

print(f"Checkpoint ID: {checkpoint.id}")
```

#### Restoring Checkpoints

```python
# Load checkpoint state
state = manager.load_checkpoint(checkpoint.id)

if state:
    print(f"Session: {state.session_id}")
    print(f"Messages: {len(state.messages)}")
    print(f"Context: {state.context}")
    print(f"Execution State: {state.execution_state}")

# Restore to checkpoint (includes git state)
success = manager.restore_checkpoint(
    checkpoint_id=checkpoint.id,
    restore_git=True,  # Also restore git working tree
)
```

#### Listing Checkpoints

```python
# List all checkpoints
checkpoints = manager.list_checkpoints()

for cp in checkpoints:
    print(f"{cp['id']}: {cp['description']}")
    print(f"  Timestamp: {cp['timestamp']}")
    print(f"  Session: {cp['session_id']}")

# Filter by session
session_checkpoints = manager.list_checkpoints(session_id="my_session")
```

#### Session Forking

```python
# Fork a new session from checkpoint
new_session_id = manager.fork_session(checkpoint_id)

print(f"New session: {new_session_id}")

# The new session has the same state as the checkpoint
new_state = manager.load_checkpoint_by_session(new_session_id)
```

## Auto-Checkpointing

Enable automatic checkpointing to create checkpoints at regular intervals:

```python
# Enable auto-checkpointing (every 5 tool calls)
manager = EnhancedCheckpointManager(
    auto_checkpoint=True,
    checkpoint_interval=5,
)

# Record tool calls (checkpoint created automatically every N calls)
def execute_tool(tool_name: str, **kwargs):
    result = tool.execute(**kwargs)

    # Record the tool call
    checkpoint_id = manager.record_tool_call()

    if checkpoint_id:
        print(f"Auto-checkpoint created: {checkpoint_id}")

    return result
```

### Configuration

```python
from victor.agent.enhanced_checkpoints import CheckpointState

# Configure auto-checkpointing behavior
manager.auto_checkpoint = True
manager.checkpoint_interval = 10  # Checkpoint every 10 tool calls

# Or create manager with config
manager = EnhancedCheckpointManager(
    auto_checkpoint=True,
    checkpoint_interval=10,
    db_path=Path("/path/to/checkpoints.db"),  # Custom database path
)
```

## Checkpoint State Structure

### CheckpointState

```python
@dataclass
class CheckpointState:
    """Serialized state for checkpointing."""

    session_id: str                              # Session identifier
    messages: List[Dict[str, Any]]               # Conversation messages
    context: Dict[str, Any]                       # Conversation context
    execution_state: Dict[str, Any]              # Execution state
    tool_calls: List[Dict[str, Any]]             # Tool call history
    timestamp: datetime                           # Creation time
    metadata: Dict[str, Any]                      # Additional metadata
```

### State Serialization

```python
from victor.agent.enhanced_checkpoints import StateSerializer

# Serialize messages
serialized = StateSerializer.serialize_messages(messages)

# Deserialize messages
messages = StateSerializer.deserialize_messages(serialized)

# Serialize context
context_dict = StateSerializer.serialize_context(context)

# Serialize execution state
state_dict = StateSerializer.serialize_execution_state(state)
```

## Architecture

### Dual-Backend Design

```
┌─────────────────────────────────────────────────────────────┐
│              Enhanced Checkpoint Manager                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐          ┌──────────────────┐         │
│  │ Git Backend      │          │ SQLite Backend   │         │
│  ├──────────────────┤          ├──────────────────┤         │
│  │ Working tree     │          │ Conversation     │         │
│  │ snapshots        │          │ state            │         │
│  │ (git stash)      │          │ Execution state  │         │
│  └──────────────────┘          │ Message history  │         │
│             │                   └──────────────────┘         │
│             │                                                 │
│             ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Unified Checkpoint API                    │   │
│  │  - save_checkpoint()                                 │   │
│  │  - load_checkpoint()                                 │   │
│  │  - restore_checkpoint()                              │   │
│  │  - fork_session()                                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Checkpoint Creation Flow

```
User Action (e.g., "Before refactoring")
    │
    ▼
EnhancedCheckpointManager.save_checkpoint()
    │
    ├──► GitBackend.create() ──► git stash (working tree)
    │
    └──► SQLiteBackend.save() ──► checkpoints table (state)
```

### Checkpoint Restore Flow

```
User Action (Restore checkpoint)
    │
    ▼
EnhancedCheckpointManager.restore_checkpoint()
    │
    ├──► Load state from SQLite
    │       └──► CheckpointState (messages, context, execution)
    │
    └──► GitBackend.rollback() ──► git stash apply (working tree)
```

## Advanced Usage

### Custom State Serialization

```python
from victor.agent.enhanced_checkpoints import CheckpointState, StateSerializer

# Create custom state with additional fields
state = CheckpointState(
    session_id="custom_session",
    messages=StateSerializer.serialize_messages(conversation.messages),
    context=StateSerializer.serialize_context({
        "user_id": "123",
        "project": "victor",
        "workspace": "/path/to/project",
    }),
    execution_state=StateSerializer.serialize_execution_state({
        "current_step": "implementation",
        "completed_steps": ["planning", "analysis"],
        "pending_steps": ["testing", "documentation"],
    }),
    tool_calls=[{
        "name": "read",
        "args": {"file": "main.py"},
        "result": "success",
        "timestamp": "2025-01-11T10:00:00",
    }],
    metadata={
        "custom_field": "custom_value",
        "created_by": "user",
    },
)

checkpoint = manager.save_checkpoint(
    description="Custom checkpoint with extended state",
    session_id="custom_session",
    state=state,
)
```

### Integration with Orchestrator

```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.enhanced_checkpoints import EnhancedCheckpointManager, CheckpointState

class CheckpointEnabledOrchestrator(AgentOrchestrator):
    """Orchestrator with automatic checkpointing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_manager = EnhancedCheckpointManager(
            auto_checkpoint=True,
            checkpoint_interval=5,
        )

    async def execute_tool(self, tool_name: str, **kwargs):
        """Execute tool with checkpoint tracking."""
        result = await super().execute_tool(tool_name, **kwargs)

        # Record tool call and auto-checkpoint
        checkpoint_id = self.checkpoint_manager.record_tool_call()

        return result

    async def create_checkpoint(self, description: str):
        """Create manual checkpoint."""
        state = CheckpointState(
            session_id=self.session_id,
            messages=StateSerializer.serialize_messages(self.conversation.messages),
            context=self.conversation.context.copy(),
            execution_state=self.get_execution_state(),
        )

        return self.checkpoint_manager.save_checkpoint(
            description=description,
            session_id=self.session_id,
            state=state,
        )
```

### Querying Checkpoints

```python
from victor.agent.enhanced_checkpoints import SQLiteCheckpointBackend

backend = SQLiteCheckpointBackend()

# List all checkpoints for a session
checkpoints = backend.list_checkpoints(session_id="my_session")

# Get checkpoint count
print(f"Total checkpoints: {len(checkpoints)}")

# Find recent checkpoints
recent = [cp for cp in checkpoints if cp["timestamp"] > "2025-01-10"]

# Delete old checkpoints
removed = backend.delete_old_checkpoints(keep_count=20)
print(f"Removed {removed} old checkpoints")
```

## Best Practices

### When to Create Checkpoints

**Create checkpoints before:**
- Major refactoring operations
- Multi-file edits
- Risky code changes
- Long-running operations
- Branching logic points

**Example:**
```python
# Before risky operation
checkpoint = await orchestrator.create_checkpoint("Before database migration")

# Perform operation
result = await migrate_database()

# If something goes wrong
if result.failed:
    orchestrator.checkpoint_manager.restore_checkpoint(checkpoint.id)
```

### Auto-Checkpointing Intervals

Choose intervals based on operation cost:

| Operation Type | Recommended Interval | Rationale |
|----------------|---------------------|------------|
| High-risk changes | 3-5 tool calls | Frequent recovery points |
| Normal development | 10-15 tool calls | Balance overhead vs. safety |
| Exploration | 20+ tool calls | Minimal overhead |

### Checkpoint Cleanup

```python
# Clean up old checkpoints regularly
manager.cleanup_old(keep_count=20)

# Schedule automatic cleanup (e.g., in a background task)
import asyncio

async def periodic_cleanup():
    while True:
        await asyncio.sleep(3600)  # Every hour
        manager.cleanup_old(keep_count=20)
```

### Session Forking for Exploration

```python
# Explore alternate implementation
checkpoint_id = await orchestrator.create_checkpoint("Before optimization")

# Continue with optimization A
await optimize_approach_a()

# Fork to explore approach B
new_session = manager.fork_session(checkpoint_id)
await explore_approach_b_in_new_session(new_session)

# Compare results and choose best
```

## CLI Commands Reference

### victor checkpoint save

Create a new checkpoint.

```bash
victor checkpoint save "Description" [OPTIONS]

Options:
  --session, -s TEXT    Session ID (default: current)
  --auto, -a          Enable auto-checkpointing
  --interval, -i INT   Auto-checkpoint interval (default: 5)
```

### victor checkpoint list

List all checkpoints.

```bash
victor checkpoint list [OPTIONS]

Options:
  --session, -s TEXT    Filter by session ID
  --include-git        Include git-only checkpoints (default: True)
  --limit, -l INT      Max checkpoints to show (default: 20)
  --format, -f TEXT    Output format: table, json (default: table)
```

### victor checkpoint show

Show detailed checkpoint information.

```bash
victor checkpoint show CHECKPOINT_ID
```

### victor checkpoint restore

Restore to a previous checkpoint.

```bash
victor checkpoint restore CHECKPOINT_ID [OPTIONS]

Options:
  --git/--no-git      Also restore git working tree (default: True)
  --confirm, -y       Skip confirmation prompt
```

### victor checkpoint fork

Create a new session from a checkpoint.

```bash
victor checkpoint fork CHECKPOINT_ID
```

### victor checkpoint cleanup

Remove old checkpoints.

```bash
victor checkpoint cleanup [OPTIONS]

Options:
  --keep, -k INT      Number of recent checkpoints to keep (default: 20)
  --confirm, -y       Skip confirmation prompt
```

### victor checkpoint auto

Configure automatic checkpointing.

```bash
victor checkpoint auto [OPTIONS]

Options:
  --enable/--disable   Enable or disable auto-checkpointing
  --interval, -i INT   Checkpoint interval (default: 5)
```

## Performance Considerations

- **Checkpoint Size**: ~1-5KB per checkpoint (state only)
- **Git Stash**: Variable (depends on working tree size)
- **SQLite Operations**: ~10-20ms per checkpoint
- **Auto-Checkpointing Overhead**: ~50-100ms every N tool calls

## Troubleshooting

### Checkpoint Creation Fails

```bash
# Check if git repository
git status

# Check git stash permissions
git stash list

# Verify database permissions
ls -la ~/.victor/checkpoints.db
```

### Restore Fails

```python
# Verify checkpoint exists
state = manager.load_checkpoint(checkpoint_id)
if not state:
    print("Checkpoint not found in SQLite")

# Try restoring state only (no git)
manager.restore_checkpoint(checkpoint_id, restore_git=False)
```

### Auto-Checkpointing Not Working

```python
# Verify auto-checkpointing is enabled
manager.auto_checkpoint  # Should be True
manager.checkpoint_interval  # Should be set

# Manually trigger checkpoint
checkpoint_id = manager.record_tool_call()
print(f"Created: {checkpoint_id}")
```

## Architecture Notes

### Comparison with LangGraph

| Feature | LangGraph | Victor |
|---------|-----------|--------|
| Checkpoint Backend | SQLite | Dual (Git + SQLite) |
| State Persistence | Thread-local | Session-based |
| Working Tree Snapshots | No | Yes (git stash) |
| Session Forking | Yes | Yes |
| Auto-Checkpointing | Manual | Built-in |

### Advantages of Dual Backend

1. **Git Backend**: Preserves actual file changes
2. **SQLite Backend**: Fast state queries and serialization
3. **Separation of Concerns**: Files vs. execution state
4. **Flexibility**: Can use either or both backends

## See Also

- [Entity Memory System](ENTITY_MEMORY.md)
- [Agent and Workflow Presets](AGENT_AND_WORKFLOW_PRESETS.md)
- [CLI Reference](../cli/README.md)
- [Architecture Overview](../architecture/ARCHITECTURE.md)
