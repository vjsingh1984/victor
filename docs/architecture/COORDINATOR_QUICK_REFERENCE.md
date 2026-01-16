# Coordinator Architecture Quick Reference

## Two-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│                  victor/agent/coordinators/                     │
│  Victor-specific orchestration: Chat, Tools, Context, Analytics │
└────────────────────────────┬────────────────────────────────────┘
                             │ uses
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRAMEWORK LAYER                             │
│                 victor/framework/coordinators/                  │
│  Domain-agnostic infrastructure: YAML, Graph, HITL, Cache       │
└─────────────────────────────────────────────────────────────────┘
```

## Application Layer Coordinators (17)

| Coordinator | Purpose | Key Methods |
|-------------|---------|-------------|
| **ChatCoordinator** | LLM chat & streaming | `chat()`, `stream_chat()` |
| **ToolCoordinator** | Tool validation & execution | `select_tools()`, `execute_tool_calls()` |
| **ContextCoordinator** | Context management | `compact_context()` |
| **AnalyticsCoordinator** | Session metrics | `track_tool_call()`, `export()` |
| **PromptCoordinator** | System prompt building | `build_prompt()` |
| **ToolSelectionCoordinator** | Semantic tool selection | `select_tools()` |
| **SessionCoordinator** | Session lifecycle | `create_session()`, `update_session()` |
| **ProviderCoordinator** | Provider switching | `switch_provider()` |
| **ModeCoordinator** | Agent modes | `set_mode()`, `get_mode_config()` |
| **CheckpointCoordinator** | Workflow checkpoints | `save_checkpoint()`, `restore_checkpoint()` |
| **EvaluationCoordinator** | LLM evaluation | `run_evaluation()` |
| **MetricsCoordinator** | System metrics | `track_metric()`, `get_metrics()` |
| **ConfigCoordinator** | Configuration loading | `load_config()`, `validate()` |
| **WorkflowCoordinator** | Workflow orchestration | `execute_workflow()` |
| **ResponseCoordinator** | Response processing | `sanitize_response()`, `parse_tool_calls()` |
| **StateCoordinator** | Unified state management | `get_state()`, `transition_to()` |
| **ToolAccessConfigCoordinator** | Tool access configuration | `get_enabled_tools()`, `is_tool_enabled()` |

## Framework Layer Coordinators (4)

| Coordinator | Purpose | Key Methods |
|-------------|---------|-------------|
| **YAMLWorkflowCoordinator** | YAML workflow execution | `execute()`, `stream()` |
| **GraphExecutionCoordinator** | StateGraph execution | `execute()`, `stream()` |
| **HITLCoordinator** | Human-in-the-loop | `execute_with_hitl()` |
| **CacheCoordinator** | Workflow caching | `enable_caching()`, `clear_cache()` |

## SOLID Principles

### Single Responsibility Principle
- Each coordinator has ONE reason to change
- ChatCoordinator: Changes when chat logic changes
- ToolCoordinator: Changes when tool execution changes

### Interface Segregation Principle
- Focused protocols
- `IToolCoordinator`: Tool methods only
- `IChatCoordinator`: Chat methods only

### Dependency Inversion Principle
- Depend on protocols, not implementations
- `ChatCoordinator(orchestrator: IAgentOrchestrator)`

### Open/Closed Principle
- Open for extension, closed for modification
- Add new strategies without changing existing code
- `BaseCompactionStrategy` → Custom strategy

### Liskov Substitution Principle
- Substitutable implementations
- Any `IChatCoordinator` implementation works

## Quick Examples

### Using Application Layer

```python
from victor.agent.coordinators import ChatCoordinator

# Create coordinator
coordinator = ChatCoordinator(orchestrator)

# Chat with streaming
async for chunk in coordinator.stream_chat(messages=[...]):
    print(chunk.content)
```

### Using Framework Layer

```python
from victor.framework.coordinators import YAMLWorkflowCoordinator

# Create coordinator
coordinator = YAMLWorkflowCoordinator()

# Execute workflow
result = await coordinator.execute("workflow.yaml", state={...})
```

### Combining Layers

```python
from victor.agent.coordinators import ChatCoordinator
from victor.framework.coordinators import GraphExecutionCoordinator

# Application layer uses framework layer
chat_coord = ChatCoordinator(orchestrator)
graph_coord = GraphExecutionCoordinator()

# Chat coordinator uses graph coordinator
result = await graph_coord.execute(chat_workflow, state={...})
```

## Extending the System

### Add New Compaction Strategy

```python
from victor.agent.coordinators import BaseCompactionStrategy

class CustomCompactionStrategy(BaseCompactionStrategy):
    async def compact(self, messages, max_tokens):
        # Your logic here
        return compacted_messages

# Use it
context_coord = ContextCoordinator(strategy=CustomCompactionStrategy())
```

### Add New Analytics Exporter

```python
from victor.agent.coordinators import BaseAnalyticsExporter

class CustomExporter(BaseAnalyticsExporter):
    async def export(self, analytics):
        # Your export logic here
        pass

# Use it
analytics_coord = AnalyticsCoordinator()
await analytics_coord.export(CustomExporter())
```

### Using ResponseCoordinator

```python
from victor.agent.coordinators import ResponseCoordinator

coordinator = ResponseCoordinator(
    sanitizer=ResponseSanitizer(),
    tool_adapter=tool_calling_adapter,
)

# Sanitize response from LLM
cleaned = coordinator.sanitize_response(raw_content)

# Parse and validate tool calls
validation = coordinator.parse_and_validate_tool_calls(
    tool_calls=native_calls,
    content=response_content,
    enabled_tools={"read_file", "write_file"},
)

# Process streaming chunks with garbage detection
result = coordinator.process_stream_chunk(
    chunk=stream_chunk,
    consecutive_garbage_count=0,
    max_garbage_chunks=3,
)
if result.should_stop:
    # Stop streaming due to garbage
    pass
```

### Using StateCoordinator

```python
from victor.agent.coordinators import StateCoordinator, StateScope

coordinator = StateCoordinator(
    session_state_manager=session_state,
    conversation_state_machine=conversation_state,
)

# Get comprehensive state
state = coordinator.get_state(scope=StateScope.ALL)

# Subscribe to state changes
@coordinator.on_state_change
def handle_change(change: StateChange):
    print(f"State changed: {change.scope}")

# Transition to new stage
coordinator.transition_to(ConversationStage.EXECUTION)

# Check budget
if coordinator.is_budget_exhausted():
    # Handle budget exhaustion
    pass

# Record tool call
coordinator.record_tool_call("read_file", {"path": "/path/to/file"})
```

### Using ToolAccessConfigCoordinator

```python
from victor.agent.coordinators import ToolAccessConfigCoordinator

coordinator = ToolAccessConfigCoordinator(
    tool_access_controller=controller,
    mode_coordinator=mode_coordinator,
    tool_registry=registry,
)

# Check if tool is enabled
if coordinator.is_tool_enabled("bash"):
    # Tool is enabled for current mode/session

# Get all enabled tools
enabled = coordinator.get_enabled_tools()

# Validate mode transition
result = coordinator.validate_mode_transition(
    from_mode="build",
    to_mode="plan",
    current_tools=enabled,
)
if result.warnings:
    # Warn user about tools that will be disabled
    pass
```

## Key Benefits

1. **Single Responsibility**: Each coordinator has one clear purpose
2. **Layered Design**: Application builds on framework foundation
3. **Reusability**: Framework coordinators work across all verticals
4. **Testability**: Coordinators can be tested independently
5. **Maintainability**: Clear boundaries reduce coupling

## Dependency Flow

```
Application Layer (victor/agent/coordinators/)
    └── depends on ──→ Framework Layer (victor/framework/coordinators/)
                           └── depends on ──→ Core Protocols (victor/protocols/)
```

## Common Patterns

### Pattern 1: Application Layer Uses Framework Layer

```python
# Application layer coordinator
class ChatCoordinator:
    def __init__(self):
        # Use framework layer coordinator
        self.graph_coord = GraphExecutionCoordinator()

    async def chat(self, messages):
        return await self.graph_coord.execute(self.workflow, state={...})
```

### Pattern 2: Multiple Coordinators Work Together

```python
# Chat coordinator orchestrates multiple coordinators
class ChatCoordinator:
    async def chat(self, messages):
        context = await self.context_coord.build_context(messages)
        tools = await self.tool_coord.select_tools(context)
        prompt = await self.prompt_coord.build_prompt(context)
        # ... use tools and prompt
```

### Pattern 3: Strategy Pattern for Extensibility

```python
# Context coordinator with pluggable strategy
class ContextCoordinator:
    def __init__(self, strategy: BaseCompactionStrategy):
        self.strategy = strategy

    async def compact_context(self, messages):
        return await self.strategy.compact(messages, self.max_tokens)
```

## Testing

### Test Application Coordinator

```python
async def test_chat_coordinator():
    mock_orchestrator = MockOrchestrator()
    coordinator = ChatCoordinator(mock_orchestrator)

    result = await coordinator.chat(messages=[...])
    assert result.content == "expected"
```

### Test Framework Coordinator

```python
async def test_yaml_workflow_coordinator():
    coordinator = YAMLWorkflowCoordinator()

    result = await coordinator.execute("workflow.yaml", state={...})
    assert result.status == "completed"
```

## File Locations

```
victor/
├── agent/coordinators/
│   ├── __init__.py              # Application layer docstring
│   ├── chat_coordinator.py
│   ├── tool_coordinator.py
│   ├── context_coordinator.py
│   ├── analytics_coordinator.py
│   ├── prompt_coordinator.py
│   ├── tool_selection_coordinator.py
│   ├── session_coordinator.py
│   ├── provider_coordinator.py
│   ├── mode_coordinator.py
│   ├── checkpoint_coordinator.py
│   ├── evaluation_coordinator.py
│   ├── metrics_coordinator.py
│   ├── config_coordinator.py     # ConfigCoordinator + ToolAccessConfigCoordinator
│   ├── workflow_coordinator.py
│   ├── response_coordinator.py  # NEW: Response processing
│   └── state_coordinator.py     # NEW: Unified state management
└── framework/coordinators/
    ├── __init__.py              # Framework layer docstring
    ├── yaml_coordinator.py
    ├── graph_coordinator.py
    ├── hitl_coordinator.py
    └── cache_coordinator.py
```

## Documentation Links

- [Full Architecture Documentation](coordinator_separation.md)
- [Visual Diagrams](diagrams/coordinators.mmd)
- [Documentation Summary](COORDINATOR_DOCUMENTATION_SUMMARY.md)

## Key Takeaways

1. **Two Layers**: Application (Victor-specific) + Framework (domain-agnostic)
2. **21 Coordinators**: 17 application + 4 framework
3. **SOLID Principles**: All 5 principles implemented
4. **Reusable**: Framework coordinators used across all verticals
5. **Testable**: Each coordinator can be tested independently
6. **Extensible**: Add new strategies without modifying existing code
