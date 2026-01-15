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

## Application Layer Coordinators (14)

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
│   └── ... (11 more)
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
2. **18 Coordinators**: 14 application + 4 framework
3. **SOLID Principles**: All 5 principles implemented
4. **Reusable**: Framework coordinators used across all verticals
5. **Testable**: Each coordinator can be tested independently
6. **Extensible**: Add new strategies without modifying existing code
