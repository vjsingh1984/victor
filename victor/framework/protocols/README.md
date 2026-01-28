# Victor Framework Protocols Package

This package contains stable protocols for framework-orchestrator integration, organized into logical modules for better maintainability and extensibility.

## Package Structure

```
victor/framework/protocols/
├── __init__.py           # Main exports (backward compatible)
├── exceptions.py         # Protocol-related exceptions
├── streaming.py          # Streaming types and chunks
├── component.py          # Component protocols (state, provider, tools, etc.)
├── orchestrator.py       # Main orchestrator protocol
├── capability.py         # Capability discovery and versioning
├── utils.py              # Protocol verification utilities
└── chat.py               # Phase 1 workflow chat protocols and implementations
```

## Modules

### `__init__.py`
Main entry point that re-exports all protocols for backward compatibility.
All existing code using `from victor.framework.protocols import ...` continues to work without changes.

**Exports**:
- `IncompatibleVersionError`
- `ChunkType`, `OrchestratorStreamChunk`
- `ConversationStateProtocol`, `ProviderProtocol`, `ToolsProtocol`, `SystemPromptProtocol`, `MessagesProtocol`, `StreamingProtocol`
- `OrchestratorProtocol`
- `CapabilityType`, `OrchestratorCapability`, `CapabilityRegistryProtocol`
- `verify_protocol_conformance`
- `ChatStateProtocol`, `ChatResultProtocol`, `WorkflowChatProtocol`, `ChatResult`, `MutableChatState`

### `exceptions.py`
Exception classes for protocol-related errors.

**Classes**:
- `IncompatibleVersionError`: Raised when a capability version is incompatible with requirements

### `streaming.py`
Streaming types for orchestrator protocols.

**Classes**:
- `ChunkType`: Enum of streaming chunk types (CONTENT, THINKING, TOOL_CALL, TOOL_RESULT, TOOL_ERROR, STAGE_CHANGE, ERROR, STREAM_START, STREAM_END)
- `OrchestratorStreamChunk`: Standardized streaming chunk format from orchestrator

### `component.py`
Component protocols for orchestrator integration.

**Protocols**:
- `ConversationStateProtocol`: Access to conversation state (stage, tool usage, file tracking)
- `ProviderProtocol`: LLM provider management (current provider/model, switching)
- `ToolsProtocol`: Tool management (available/enabled tools)
- `SystemPromptProtocol`: System prompt management (get/set/append)
- `MessagesProtocol`: Conversation messages access
- `StreamingProtocol`: Streaming status access

### `orchestrator.py`
Main orchestrator protocol combining all component protocols.

**Protocols**:
- `OrchestratorProtocol`: Complete orchestrator protocol combining all 6 sub-protocols
  - ConversationStateProtocol methods
  - ProviderProtocol methods
  - ToolsProtocol methods
  - SystemPromptProtocol methods
  - MessagesProtocol methods
  - StreamingProtocol methods
  - Core chat operations (stream_chat, chat)
  - Lifecycle methods (cancel, reset)

### `capability.py`
Capability discovery and versioning protocols.

**Classes**:
- `CapabilityType`: Enum of capability types (TOOL, PROMPT, MODE, SAFETY, RL, TEAM, WORKFLOW, VERTICAL)
- `OrchestratorCapability`: Explicit capability declaration with semantic versioning
- `CapabilityRegistryProtocol`: Protocol for capability discovery and invocation

### `utils.py`
Protocol verification utilities.

**Functions**:
- `verify_protocol_conformance`: Verify that an object conforms to a protocol

### `chat.py`
Phase 1 workflow chat protocols and implementations.

**Protocols**:
- `ChatStateProtocol`: Interface for chat workflow state
- `ChatResultProtocol`: Interface for chat workflow results
- `WorkflowChatProtocol`: Interface for workflow-based chat execution

**Implementations**:
- `ChatResult`: Immutable result of chat workflow execution
- `MutableChatState`: Default implementation of chat state protocol (note: not thread-safe)

## Usage

### Basic Import (Backward Compatible)
```python
from victor.framework.protocols import (
    OrchestratorProtocol,
    ChatStateProtocol,
    MutableChatState,
    ChunkType,
)

# All existing code continues to work without changes
```

### Direct Import from Modules
```python
# Import from specific module for clarity
from victor.framework.protocols.orchestrator import OrchestratorProtocol
from victor.framework.protocols.chat import MutableChatState
from victor.framework.protocols.streaming import ChunkType
```

### MutableChatState Usage
```python
from victor.framework.protocols import MutableChatState

# Create state
state = MutableChatState()
state.add_message("user", "Hello!")
state.increment_iteration()
state.set_metadata("task_type", "coding")

# Access state
print(state.messages)  # [{'role': 'user', 'content': 'Hello!'}]
print(state.iteration_count)  # 1
print(state.get_metadata("task_type"))  # 'coding'

# Serialize
state_dict = state.to_dict()
restored = MutableChatState.from_dict(state_dict)
```

### Capability Declaration
```python
from victor.framework.protocols import OrchestratorCapability, CapabilityType

# Declare capability
capability = OrchestratorCapability(
    name="enabled_tools",
    capability_type=CapabilityType.TOOL,
    version="1.0",
    setter="set_enabled_tools",
    getter="get_enabled_tools",
    description="Set which tools are enabled",
)

# Check version compatibility
print(capability.is_compatible_with("0.9"))  # True
print(capability.is_compatible_with("1.1"))  # False
```

## Design Patterns

### Protocol-First Architecture
- Eliminates duck-typing (hasattr/getattr calls)
- Provides type safety via Protocol structural subtyping
- Enables clean mocking for tests
- Documents the exact interface contract

### Composite Protocol
`OrchestratorProtocol` combines 6 sub-protocols:
- `ConversationStateProtocol`: Stage, tool usage, file tracking
- `ProviderProtocol`: Provider/model management
- `ToolsProtocol`: Tool access and management
- `SystemPromptProtocol`: Prompt customization
- `MessagesProtocol`: Message history
- `StreamingProtocol`: Streaming status

### Capability Versioning
- Semantic versioning (MAJOR.MINOR)
- Version compatibility checking
- Backward compatibility support

## Migration from Single File

This package was created by splitting the original `victor/framework/protocols.py` (1305 lines) into 8 focused modules:

| Original Lines | Module | Purpose |
|----------------|--------|---------|
| 42-72 | exceptions.py | Exception classes |
| 74-152 | streaming.py | Streaming types |
| 154-388 | component.py | Component protocols |
| 390-589 | orchestrator.py | Main orchestrator protocol |
| 591-853 | capability.py | Capability discovery |
| 856-893 | utils.py | Verification utilities |
| 897-1305 | chat.py | Phase 1 chat protocols |

**No breaking changes**: All existing imports continue to work via `__init__.py` re-exports.

## Bug Fixes

### MutableChatState Locking Bug
The original implementation had incorrect locking code:
```python
# WRONG: threading.Lock() takes no arguments
with threading.Lock(self._lock):
    ...
```

Fixed by simplifying the implementation (removed locking since it was broken):
```python
# CORRECT: Direct access (not thread-safe)
self._messages.append(message)
```

**Note**: `MutableChatState` is now explicitly documented as not thread-safe. For concurrent access, use external locking or create a thread-safe subclass.

## Testing

All protocols can be tested independently:

```python
# Test protocol conformance
from victor.framework.protocols import verify_protocol_conformance, OrchestratorProtocol

conforms, missing = verify_protocol_conformance(orchestrator, OrchestratorProtocol)
if not conforms:
    raise TypeError(f"Missing protocol methods: {missing}")
```

## Future Enhancements

This package structure enables:
1. Easy addition of new protocols without bloating single files
2. Clear separation of concerns
3. Better navigation and code organization
4. Independent testing of protocol groups
5. Potential for protocol-specific utilities

## Related Documentation

- Framework architecture: `victor/framework/README.md`
- Protocol usage: `docs/architecture/BEST_PRACTICES.md`
- Phase 1 implementation: Plan section "Phase 1: Foundation"
