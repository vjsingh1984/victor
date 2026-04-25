# State-Passed Architecture for Victor Orchestrator

**Status**: Foundation Implemented (P3 - Lower Priority)
**Date**: 2026-04-14
**Priority**: P3 (Architecture Improvement)

---

## Overview

The state-passed architecture is a design pattern that decouples coordinators from the orchestrator by using immutable snapshots and explicit state transitions. This addresses the 3,915 LOC orchestrator complexity issue identified in the Gemini feedback.

### Problem Statement

The current `AgentOrchestrator` has several issues:
1. **High Coupling**: 6+ coordinators hold direct references to the orchestrator
2. **Stateful Glue**: Orchestrator acts as stateful glue - if lost, all coordinators lose context
3. **Testing Difficulties**: Coordinators require full orchestrator mocks for testing
4. **Cognitive Load**: 3,915 LOC makes the orchestrator difficult to understand

### Solution: State-Passed Architecture

Coordinators become pure functions:
- **Input**: Immutable `ContextSnapshot` of current state
- **Output**: `CoordinatorResult` with explicit `StateTransition` objects
- **No Side Effects**: No direct state mutation during execution

## Current Entry Points

Use the concrete migration surfaces that are already wired into the runtime.

| Need | Preferred surface | Notes |
|------|-------------------|-------|
| Chat execution | `ChatService` or `OrchestrationFacade.chat_service` | Service-owned runtime |
| Tool execution | `ToolService` or `OrchestrationFacade.tool_service` | Service-owned runtime |
| Session lifecycle | `SessionService` or `OrchestrationFacade.session_service` | Service-owned runtime |
| Read-only exploration | `victor.agent.coordinators.ExplorationCoordinator` | First-class package-root export |
| State-passed exploration | `victor.agent.coordinators.ExplorationStatePassedCoordinator` | Snapshot/transition wrapper |
| State-passed system prompt logic | `victor.agent.coordinators.SystemPromptStatePassedCoordinator` | Prefer over direct shim imports |
| State-passed safety checks | `victor.agent.coordinators.SafetyStatePassedCoordinator` | Prefer over coordinator shim usage |

Deprecated chat shims are compatibility-only. In particular,
`ChatCoordinator.stream_chat()` now forwards in this order:
1. bound `ChatService.stream_chat()`
2. orchestrator `_get_service_streaming_runtime()`
3. legacy `_stream_chat_runtime` hook

That legacy hook remains only as a fallback for older integrations and should
not be used as the primary runtime surface in new code.

```python
from victor.agent.coordinators import (
    ExplorationCoordinator,
    ExplorationStatePassedCoordinator,
    SafetyStatePassedCoordinator,
    SystemPromptStatePassedCoordinator,
)
from victor.agent.facades import OrchestrationFacade

facade = OrchestrationFacade(
    chat_service=chat_service,
    tool_service=tool_service,
    session_service=session_service,
    exploration_state_passed=ExplorationStatePassedCoordinator(),
    system_prompt_state_passed=SystemPromptStatePassedCoordinator(
        task_analyzer=task_analyzer
    ),
    safety_state_passed=SafetyStatePassedCoordinator(),
)

findings = await ExplorationCoordinator().explore_parallel(
    "Trace the failing import boundary",
    project_root=project_root,
)
```

---

## Core Components

### 1. ContextSnapshot

Immutable snapshot of orchestrator state at a point in time.

```python
@dataclass(frozen=True)
class ContextSnapshot:
    """Immutable snapshot of orchestrator state."""

    # Core conversation state
    messages: tuple[Message, ...]
    session_id: str
    conversation_stage: str

    # Configuration
    settings: Settings
    model: str
    provider: str
    max_tokens: int
    temperature: float

    # State (key-value store)
    conversation_state: Dict[str, Any]
    session_state: Dict[str, Any]

    # Metadata
    observed_files: tuple[str, ...]
    capabilities: Dict[str, Any]
```

**Key Features**:
- **Frozen**: Cannot be modified (prevents accidental mutation)
- **Tuples**: Collections are tuples (not lists) for immutability
- **Dict copies**: State dictionaries are copied (prevents shared mutation)

**Usage**:
```python
# Read from context
message_count = context.message_count
stage = context.conversation_stage
value = context.get_state("key", default=None)

# Check capabilities
if context.has_capability("planning_mode"):
    # ...
```

### 2. StateTransition

Encapsulates a single state change or side effect.

```python
@dataclass
class StateTransition:
    """Encapsulates a state change or side effect."""

    transition_type: TransitionType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Transition Types**:
- `ADD_MESSAGE`: Add a message to conversation
- `UPDATE_STATE`: Update a key-value pair in state
- `DELETE_STATE`: Remove a key from state
- `EXECUTE_TOOL`: Request tool execution
- `UPDATE_CAPABILITY`: Update a capability flag
- `UPDATE_STAGE`: Update conversation stage
- `CREATE_SESSION` / `CLOSE_SESSION`: Session management

**Usage**:
```python
# Create transitions
transition1 = StateTransition(
    transition_type=TransitionType.UPDATE_STATE,
    data={"key": "plan_steps", "value": steps, "scope": "conversation"}
)

transition2 = StateTransition(
    transition_type=TransitionType.EXECUTE_TOOL,
    data={"tool_name": "list_files", "arguments": {"path": "."}}
)
```

### 3. TransitionBatch

Batches multiple transitions for atomic application.

```python
@dataclass
class TransitionBatch:
    """A batch of transitions to be applied atomically."""

    transitions: List[StateTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Helper Methods**:
```python
batch = TransitionBatch()

# Method chaining for fluent API
batch.update_state("key", "value") \
     .add_message(message) \
     .execute_tool("tool_name", {"arg": "value"})

# Extend with other batches
batch.extend(other_batch)
```

### 4. CoordinatorResult

Combines transitions with metadata about coordinator decisions.

```python
@dataclass
class CoordinatorResult:
    """Result of a coordinator operation."""

    transitions: TransitionBatch
    reasoning: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    should_continue: bool = True
    handoff_to: Optional[str] = None
```

**Factory Methods**:
```python
# No-op result
result = CoordinatorResult.no_op(reasoning="Nothing to do")

# Result with transitions
result = CoordinatorResult.transitions_only(
    transition1, transition2,
    reasoning="Applied updates"
)
```

### 5. TransitionApplier

Applies transitions to the actual orchestrator.

```python
class TransitionApplier:
    """Applies state transitions to an orchestrator."""

    async def apply(self, transition: StateTransition) -> None:
        """Apply a single transition."""

    async def apply_batch(self, batch: TransitionBatch) -> None:
        """Apply a batch of transitions atomically."""
```

### 6. create_snapshot()

Utility function to create snapshots from orchestrator.

```python
def create_snapshot(orchestrator: Any) -> ContextSnapshot:
    """Create a ContextSnapshot from an orchestrator."""
```

---

## Migration Pattern

### Before (Current Pattern)

```python
class ExampleCoordinator:
    def __init__(self, orchestrator):
        self._orchestrator = orchestrator  # Holds reference

    async def process(self, user_message):
        # Direct state access and mutation
        messages = self._orchestrator.messages
        self._orchestrator.conversation_state["key"] = "value"
        await self._orchestrator.execute_tool("tool", {})
```

**Issues**:
- Tight coupling to orchestrator
- Direct state mutation
- Hard to test (needs full orchestrator mock)

### After (State-Passed Pattern)

```python
class ExampleCoordinator:
    # No __init__ needed (or only config)

    async def process(self, context: ContextSnapshot, user_message: str) -> CoordinatorResult:
        # Pure function: read from context, return transitions
        messages = context.messages  # Read from immutable snapshot

        batch = TransitionBatch()
        batch.update_state("key", "value", scope="conversation")
        batch.execute_tool("tool", {})

        return CoordinatorResult(
            transitions=batch,
            reasoning="Updated state and executed tool",
        )
```

**Benefits**:
- No orchestrator reference
- No direct state mutation
- Easy to test (just create ContextSnapshot)

---

## Integration Example

### Orchestrator Integration

```python
class AgentOrchestrator:
    async def process_with_coordinator(self, user_message: str):
        from victor.agent.coordinators.state_context import (
            create_snapshot,
            TransitionApplier,
        )

        # 1. Create coordinator (no orchestrator reference)
        coordinator = ExampleStatePassedCoordinator()

        # 2. Create snapshot
        snapshot = create_snapshot(self)

        # 3. Call coordinator (pure function)
        result = await coordinator.process(snapshot, user_message)

        # 4. Apply transitions
        applier = TransitionApplier(self)
        await applier.apply_batch(result.transitions)

        # 5. Use result metadata
        if result.reasoning:
            logger.info(f"Coordinator: {result.reasoning}")

        if not result.should_continue:
            return  # Stop processing

        if result.handoff_to:
            # Hand off to next coordinator
            await self._handoff(result.handoff_to, snapshot, user_message)
```

---

## Testing Benefits

### Before (Needs Full Mock)

```python
def test_coordinator():
    # Need complex mock setup
    orchestrator = MagicMock()
    orchestrator.messages = [msg1, msg2]
    orchestrator.conversation_state = {}
    orchestrator.execute_tool = AsyncMock()

    coordinator = ExampleCoordinator(orchestrator)
    await coordinator.process("test")

    # Verify orchestrator was called
    orchestrator.execute_tool.assert_called_once()
```

### After (Simple Snapshot)

```python
def test_coordinator():
    # Just create a snapshot
    snapshot = ContextSnapshot(
        messages=(msg1, msg2),
        session_id="test",
        conversation_stage="initial",
        settings=Settings(),
        model="test",
        provider="test",
        max_tokens=4096,
        temperature=0.7,
        conversation_state={},
        session_state={},
        observed_files=(),
        capabilities={},
    )

    coordinator = ExampleStatePassedCoordinator()
    result = await coordinator.process(snapshot, "test")

    # Verify transitions (no orchestrator mock needed!)
    assert not result.transitions.is_empty()
    assert result.confidence > 0.5
```

---

## File Structure

```
victor/agent/coordinators/
├── state_context.py                      # Core abstractions
├── example_state_passed_coordinator.py   # Example implementation
├── chat_coordinator.py                   # [TODO: Refactor]
├── planning_coordinator.py               # [TODO: Refactor]
├── execution_coordinator.py              # [TODO: Refactor]
└── sync_chat_coordinator.py              # [TODO: Refactor]

tests/unit/agent/coordinators/
└── test_state_context.py                 # 34 comprehensive tests
```

---

## Migration Checklist

### Phase 1: Foundation (✅ COMPLETE)
- [x] Create `state_context.py` with core abstractions
- [x] Create `example_state_passed_coordinator.py` template
- [x] Create comprehensive unit tests (34 tests)
- [x] Document the pattern

### Phase 2: Gradual Migration (FUTURE)
- [ ] Refactor `chat_coordinator.py`
  - Replace orchestrator reference with ContextSnapshot
  - Return CoordinatorResult instead of direct mutation
  - Update tests
- [ ] Refactor `planning_coordinator.py`
- [ ] Refactor `execution_coordinator.py`
- [ ] Refactor `sync_chat_coordinator.py`
- [ ] Update orchestrator integration points

### Phase 3: Validation (FUTURE)
- [ ] Run full test suite
- [ ] Integration testing
- [ ] Performance benchmarking
- [ ] Documentation updates

---

## Design Principles

### 1. Immutability
- Snapshots are frozen (cannot be modified)
- Collections are tuples (not lists)
- State dictionaries are copied

### 2. Explicit Transitions
- All state changes are explicit (no hidden mutations)
- Transitions are validated on creation
- Atomic batch application

### 3. Testability
- Coordinators are pure functions
- No orchestrator mocks needed
- Simple snapshot construction

### 4. Gradual Migration
- Old and new patterns can coexist
- No breaking changes required
- Incremental adoption possible

---

## FAQ

### Q: Why not just use protocols?
**A**: Protocols still allow direct state mutation. State-passed ensures coordinators are pure functions with no side effects.

### Q: Won't this be slower?
**A**: Negligible impact. Snapshot creation is fast (copies dicts, tuples references). The benefit in testability and maintainability far outweighs minimal overhead.

### Q: Do I need to refactor all coordinators at once?
**A**: No! The old and new patterns can coexist. Migrate incrementally, one coordinator at a time.

### Q: How do I handle async operations (tool execution)?
**A**: Return an `EXECUTE_TOOL` transition. The orchestrator will apply it and handle the async execution.

### Q: What about nested coordinator calls?
**A**: Pass the same snapshot. Each coordinator returns its own transitions. The orchestrator merges and applies them.

---

## References

- **Gemini Feedback**: Verified claim about 3,915 LOC orchestrator complexity
- **Implementation**: `victor/agent/coordinators/state_context.py`
- **Example**: `victor/agent/coordinators/example_state_passed_coordinator.py`
- **Tests**: `tests/unit/agent/coordinators/test_state_context.py`

---

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Core Abstractions | ✅ Complete | All components implemented |
| Example Coordinator | ✅ Complete | Demonstrates the pattern |
| Unit Tests | ✅ Complete | 34 tests, all passing |
| Documentation | ✅ Complete | This document |
| Chat Coordinator | ⏳ Pending | Migration not started |
| Planning Coordinator | ⏳ Pending | Migration not started |
| Execution Coordinator | ⏳ Pending | Migration not started |

**Overall Progress**: Foundation complete. Ready for incremental coordinator migration when needed.

---

**Next Steps**: When the team is ready, begin gradual migration starting with the simplest coordinator (likely `sync_chat_coordinator.py` or a subset of `chat_coordinator.py`). Use `example_state_passed_coordinator.py` as a reference.
